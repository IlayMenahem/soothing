//! Factor-branching solver corresponding to Algorithm `Factor branching`
//! (`alg:Branching` in `paper/sections/method.tex`).
//!
//! The input generators are optionally interreduced and factored. Every
//! minimal hitting set is then interreduced and factored again until no new
//! subbranches appear. The terminal hitting sets are checked in
//! fewest-factors-first order by a selectable Gröbner backend.
//!
//! Usage: `branching [path] [--no-interreduce] [--timeout MS] [--gb BACKEND] [--summary]`
//! — default path:
//! `systems/system.txt`; default Gröbner backend: Singular.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use symbolica::LicenseManager;
use symbolica::domains::rational::Q;

use polynomial_cdcl::cover::{build_factor_index, minimal_covers, ordered_cover_masks};
use polynomial_cdcl::gb_search::{
    ComputedBasis, GbBackend, Outcome, evaluate_cover, solution_dimension,
};
use polynomial_cdcl::refine::refine_covers;
use polynomial_cdcl::{
    LoadedMathematicaSystem, PolyExtLex, default_systems_dir_from_manifest, interreduce_polys,
    load_mathematica_system_from_path_with_var_names, polynomials_from_loaded,
    try_factor_groups_from_polys,
};

#[derive(Debug)]
struct CliOptions {
    path: PathBuf,
    interreduce: bool,
    timeout_ms: u64,
    gb_backend: GbBackend,
    summary: bool,
}

fn parse_args(args: impl IntoIterator<Item = String>) -> Result<CliOptions, String> {
    let mut path: Option<PathBuf> = None;
    let mut interreduce = true;
    let mut timeout_ms = 5_000;
    let mut gb_backend = GbBackend::SingularGb;
    let mut summary = false;
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--no-interreduce" => interreduce = false,
            "--timeout" => {
                timeout_ms = args
                    .next()
                    .ok_or_else(|| "--timeout requires a value".to_string())?
                    .parse()
                    .map_err(|error| format!("invalid --timeout: {error}"))?;
            }
            "--gb" => {
                let value = args.next().ok_or_else(|| {
                    "--gb requires one of: m4gb, f5, gvw, symbolica, singular-gb, singular-facstd"
                        .to_string()
                })?;
                gb_backend = GbBackend::parse(&value).ok_or_else(|| {
                    format!(
                        "invalid --gb {value:?}: expected m4gb, f5, gvw, symbolica, singular-gb, or singular-facstd"
                    )
                })?;
            }
            "--summary" => summary = true,
            _ if arg.starts_with("--") => return Err(format!("unknown option {arg:?}")),
            _ if path.is_some() => return Err(format!("unexpected positional argument {arg:?}")),
            _ => path = Some(PathBuf::from(arg)),
        }
    }
    Ok(CliOptions {
        path: path.unwrap_or_else(|| default_systems_dir_from_manifest().join("system.txt")),
        interreduce,
        timeout_ms,
        gb_backend,
        summary,
    })
}

fn die(message: impl std::fmt::Display, code: i32) -> ! {
    eprintln!("{message}");
    std::process::exit(code)
}

fn load_system(path: &Path) -> LoadedMathematicaSystem {
    let loaded = load_mathematica_system_from_path_with_var_names(path)
        .unwrap_or_else(|error| die(format!("load error: {error}"), 1));
    println!(
        "Parsed: {} equations, {} variables (loader order)",
        loaded.system.equations.len(),
        loaded.var_names.len()
    );
    loaded
}

fn prepare_polynomials(loaded: &LoadedMathematicaSystem, interreduce: bool) -> Vec<PolyExtLex> {
    let started = Instant::now();
    let polynomials = polynomials_from_loaded(loaded, &Q).unwrap_or_else(|error| die(error, 1));
    let polynomials = if interreduce {
        interreduce_polys(polynomials)
    } else {
        polynomials
    };
    println!(
        "  {} polynomial(s), interreduction {}, elapsed {:?}",
        polynomials
            .iter()
            .filter(|polynomial| !polynomial.is_zero())
            .count(),
        if interreduce { "enabled" } else { "disabled" },
        started.elapsed()
    );
    polynomials
}

fn factor_covers(polynomials: Vec<PolyExtLex>) -> (Vec<PolyExtLex>, Vec<u64>) {
    let groups = try_factor_groups_from_polys(polynomials)
        .unwrap_or_else(|error| die(format!("factorization error: {error}"), 1));
    let (factors, index_groups) = build_factor_index(&groups);
    let covers = minimal_covers(&index_groups, factors.len()).unwrap_or_else(|| {
        die(
            "too many distinct factors (>= 64) for the bitmask enumerator",
            1,
        )
    });
    let covers = ordered_cover_masks(&covers);
    println!(
        "  {} factor group(s), {} distinct factor(s), {} minimal hitting set(s)",
        groups.len(),
        factors.len(),
        covers.len()
    );
    (factors, covers)
}

struct Component {
    cover_index: usize,
    candidate: Vec<PolyExtLex>,
    dimension: Option<usize>,
    basis: ComputedBasis,
}

struct SearchResult {
    components: Vec<Component>,
    first_solution: Option<Duration>,
    total_covers: usize,
    solved: usize,
    inconsistent: usize,
    timeouts: usize,
    elapsed: Duration,
}

impl SearchResult {
    fn complete(&self) -> bool {
        self.timeouts == 0
    }
}

fn run_search(options: &CliOptions, covers: &[Vec<PolyExtLex>], nvars: usize) -> SearchResult {
    let timeout = Duration::from_millis(options.timeout_ms);
    let started = Instant::now();
    let mut components = Vec::new();
    let mut first_solution = None;
    let mut solved = 0;
    let mut inconsistent = 0;
    let mut timeouts = 0;

    println!(
        "\nBranching over {} minimal hitting set(s), fewest-factors-first, {}ms/cover timeout, {} Groebner backend",
        covers.len(),
        options.timeout_ms,
        options.gb_backend.label()
    );
    for (cover_index, candidate) in covers.iter().enumerate() {
        let candidate = candidate.clone();
        let cover_started = Instant::now();
        match evaluate_cover(candidate.clone(), timeout, options.gb_backend) {
            Outcome::Consistent(basis) => {
                first_solution.get_or_insert_with(|| started.elapsed());
                solved += 1;
                let dimension = solution_dimension(&basis.elements, nvars);
                let dimension_label =
                    dimension.map_or_else(|| "?".to_string(), |value| value.to_string());
                println!(
                    "  cover #{cover_index}: satisfiable ({} factor(s), {} basis element(s), dimension {dimension_label}, elapsed {:?})",
                    candidate.len(),
                    basis.elements.len(),
                    cover_started.elapsed()
                );
                components.push(Component {
                    cover_index,
                    candidate,
                    dimension,
                    basis,
                });
            }
            Outcome::ConsistentComponents(bases) => {
                first_solution.get_or_insert_with(|| started.elapsed());
                solved += 1;
                println!(
                    "  cover #{cover_index}: satisfiable ({} factor(s), {} facstd component(s), elapsed {:?})",
                    candidate.len(),
                    bases.len(),
                    cover_started.elapsed()
                );
                components.extend(bases.into_iter().map(|basis| Component {
                    cover_index,
                    candidate: candidate.clone(),
                    dimension: solution_dimension(&basis.elements, nvars),
                    basis,
                }));
            }
            Outcome::Inconsistent => {
                println!(
                    "  cover #{cover_index}: unsatisfiable (solve {:?})",
                    cover_started.elapsed()
                );
                inconsistent += 1;
            }
            Outcome::Timeout => {
                println!(
                    "  cover #{cover_index}: timed out ({} factor(s), elapsed {:?})",
                    candidate.len(),
                    cover_started.elapsed()
                );
                timeouts += 1;
            }
            Outcome::BackendError(error) => die(
                format!(
                    "{} Groebner backend error: {error}",
                    options.gb_backend.label()
                ),
                1,
            ),
        }
    }

    SearchResult {
        components,
        first_solution,
        total_covers: covers.len(),
        solved,
        inconsistent,
        timeouts,
        elapsed: started.elapsed(),
    }
}

fn report_results(result: &SearchResult) {
    println!(
        "\n  {} cover(s): {} satisfiable, {} unsatisfiable, {} timeout(s), complete={}, elapsed {:?}",
        result.total_covers,
        result.solved,
        result.inconsistent,
        result.timeouts,
        result.complete(),
        result.elapsed
    );
    for component in &result.components {
        let dimension = component
            .dimension
            .map_or_else(|| "?".to_string(), |value| value.to_string());
        println!(
            "  component from cover #{}: {} factor(s), {} basis element(s), dimension {dimension}",
            component.cover_index,
            component.candidate.len(),
            component.basis.elements.len()
        );
        for polynomial in &component.candidate {
            println!("\tfactor: {polynomial}");
        }
        if component.basis.leading_monomials_only {
            println!("\tleading monomials of the external basis:");
        }
        for (index, polynomial) in component.basis.elements.iter().enumerate() {
            println!("\tg{index} = {polynomial}");
        }
    }
}

fn main() {
    LicenseManager::set_license_key(polynomial_cdcl::SYMBOLICA_LICENSE_KEY).unwrap();
    polynomial_cdcl::exit_if_solver_worker();

    let options = parse_args(std::env::args().skip(1)).unwrap_or_else(|error| die(error, 2));
    println!("Loading {}", options.path.display());
    let loaded = load_system(&options.path);
    let polynomials = prepare_polynomials(&loaded, options.interreduce);
    let nvars = polynomials
        .first()
        .map_or(loaded.var_names.len(), PolyExtLex::nvars);
    let (factors, covers) = factor_covers(polynomials);
    let (covers, refinement) = refine_covers(&factors, &covers, options.interreduce)
        .unwrap_or_else(|error| die(format!("hitting-set refinement error: {error}"), 1));
    println!(
        "  recursive refinement produced {} terminal hitting set(s) in {} pass(es), {} subdivision(s), maximum depth {}",
        covers.len(),
        refinement.passes,
        refinement.subdivisions,
        refinement.max_depth
    );
    let result = run_search(&options, &covers, nvars);

    if options.summary {
        let first_solution = result.first_solution.map_or_else(
            || "unknown".to_string(),
            |value| format!("{:.6}", value.as_secs_f64()),
        );
        let largest_dimension = result
            .components
            .iter()
            .filter_map(|component| component.dimension)
            .max()
            .map_or_else(|| "unknown".to_string(), |value| value.to_string());
        println!(
            "BRANCHING_SUMMARY total={} satisfiable={} unsatisfiable={} timeouts={} complete={} largest_dimension={} first_solution_secs={} elapsed_secs={:.6}",
            result.total_covers,
            result.solved,
            result.inconsistent,
            result.timeouts,
            result.complete(),
            largest_dimension,
            first_solution,
            result.elapsed.as_secs_f64()
        );
    } else {
        report_results(&result);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Once;

    use polynomial_cdcl::load_mathematica_system_from_str_with_var_names;

    static LICENSE: Once = Once::new();

    fn activate_license() {
        LICENSE.call_once(|| {
            let _ = LicenseManager::set_license_key(polynomial_cdcl::SYMBOLICA_LICENSE_KEY);
        });
    }

    fn test_options() -> CliOptions {
        CliOptions {
            path: PathBuf::from("unused"),
            interreduce: false,
            timeout_ms: 5_000,
            gb_backend: GbBackend::Symbolica,
            summary: true,
        }
    }

    #[test]
    fn parse_args_reads_every_option() {
        let options = parse_args(
            [
                "input.txt",
                "--no-interreduce",
                "--timeout",
                "250",
                "--gb",
                "gvw",
                "--summary",
            ]
            .into_iter()
            .map(String::from),
        )
        .expect("arguments");

        assert_eq!(options.path, PathBuf::from("input.txt"));
        assert!(!options.interreduce);
        assert_eq!(options.timeout_ms, 250);
        assert_eq!(options.gb_backend, GbBackend::Gvw);
        assert!(options.summary);
    }

    #[test]
    fn parse_args_uses_fast_defaults() {
        let options = parse_args(std::iter::empty()).expect("arguments");

        assert_eq!(options.gb_backend, GbBackend::SingularGb);
    }

    #[test]
    fn branches_over_all_minimal_hitting_sets() {
        activate_license();
        let loaded = load_mathematica_system_from_str_with_var_names("{x*(x - 1) == 0, x*y == 0}")
            .expect("load");
        let polynomials = polynomials_from_loaded(&loaded, &Q).expect("polynomials");
        let (factors, covers) = factor_covers(polynomials);
        let (covers, _) = refine_covers(&factors, &covers, false).expect("refinement");

        let result = run_search(&test_options(), &covers, 2);

        assert_eq!(covers.len(), 2);
        assert_eq!(result.solved, 2);
        assert!(result.first_solution.is_some());
        assert!(result.complete());
    }

    #[test]
    fn checks_every_inconsistent_cover() {
        activate_license();
        let loaded =
            load_mathematica_system_from_str_with_var_names("{x == 0, x - 1 == 0, y*z == 0}")
                .expect("load");
        let polynomials = polynomials_from_loaded(&loaded, &Q).expect("polynomials");
        let (factors, covers) = factor_covers(polynomials);
        let (covers, _) = refine_covers(&factors, &covers, false).expect("refinement");

        let result = run_search(&test_options(), &covers, 3);

        assert_eq!(result.inconsistent, 2);
        assert!(result.complete());
    }

    #[test]
    fn interreduction_subdivides_a_hitting_set_until_stable() {
        activate_license();
        let loaded =
            load_mathematica_system_from_str_with_var_names("{x + y == 0, x + y + y*z == 0}")
                .expect("load");
        let polynomials = polynomials_from_loaded(&loaded, &Q).expect("polynomials");
        let (factors, covers) = factor_covers(polynomials);
        assert_eq!(covers.len(), 1);

        let (terminal, stats) = refine_covers(&factors, &covers, true).expect("refinement");

        assert_eq!(terminal.len(), 2);
        assert_eq!(stats.subdivisions, 1);
        assert!(stats.max_depth >= 2);
        assert!(terminal.iter().all(|cover| cover.len() == 2));
    }
}
