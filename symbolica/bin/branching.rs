//! Factor-branching solver corresponding to Algorithm `Factor branching`
//! (`alg:Branching` in `paper/sections/method.tex`).
//!
//! The input generators are factored without interreduction. Every minimal
//! hitting set is then interreduced and factored again until no
//! new subbranches appear. Terminal hitting sets whose zero set lies in that
//! of another are dropped, and the rest are checked in
//! fewest-factors-first order by a selectable Gröbner backend.
//!
//! With `--nonzero`, the named variables must not vanish: the input becomes
//! `Saturate(P, N)` (the saturation by their product if Singular computes it
//! within `--saturation-timeout`, the input otherwise), refinement treats them
//! as nonzero side conditions, and each hitting set reaches the backend with
//! one Rabinowitsch equation per variable.
//!
//! The summary's `elapsed_secs` and `first_solution_secs` time the backend
//! search; `saturation_secs` is the time of `Saturate(P, N)` before it.
//!
//! Usage: `branching [path] [--timeout MS] [--gb BACKEND]
//! [--nonzero NAME[,NAME...]] [--saturation-timeout MS] [--summary]`
//! — default path: `systems/system.txt`; default Gröbner backend: Singular;
//! default saturation limit: the `--timeout` per hitting set.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use symbolica::domains::rational::Q;
use symbolica::license::LicenseManager;
use symbolica::poly::PolyVariable;

use polynomial_cdcl::cover::{BitSet, build_factor_index, minimal_covers, ordered_cover_masks};
use polynomial_cdcl::gb_search::{
    ComputedBasis, GbBackend, Outcome, Saturation, evaluate_cover, solution_dimension,
};
use polynomial_cdcl::nonzero::{
    condition_polynomials, nonzero_variables, saturate_by_variables, with_nonzero_equations,
};
use polynomial_cdcl::refine::{InProcess, refine_covers_under_conditions, without_conditions};
use polynomial_cdcl::{
    LoadedMathematicaSystem, PolyExtLex, default_systems_dir_from_manifest,
    load_mathematica_system_from_path_with_var_names, polynomials_from_loaded,
    try_factor_groups_from_polys,
};

#[derive(Debug)]
struct CliOptions {
    path: PathBuf,
    timeout_ms: u64,
    gb_backend: GbBackend,
    nonzero: Vec<String>,
    saturation_timeout_ms: Option<u64>,
    summary: bool,
}

fn parse_args(args: impl IntoIterator<Item = String>) -> Result<CliOptions, String> {
    let mut path: Option<PathBuf> = None;
    let mut timeout_ms = 5_000;
    let mut gb_backend = GbBackend::SingularGb;
    let mut nonzero = Vec::new();
    let mut saturation_timeout_ms = None;
    let mut summary = false;
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--timeout" => {
                timeout_ms = args
                    .next()
                    .ok_or_else(|| "--timeout requires a value".to_string())?
                    .parse()
                    .map_err(|error| format!("invalid --timeout: {error}"))?;
            }
            "--gb" => {
                let value = args.next().ok_or_else(|| {
                    "--gb requires one of: m4gb, f5, gvw, symbolica, singular-gb, singular-facstd, singular-primdecGTZ, m2-decompose"
                        .to_string()
                })?;
                gb_backend = GbBackend::parse(&value).ok_or_else(|| {
                    format!(
                        "invalid --gb {value:?}: expected m4gb, f5, gvw, symbolica, singular-gb, singular-facstd, singular-primdecGTZ, or m2-decompose"
                    )
                })?;
            }
            "--nonzero" => {
                let names = args
                    .next()
                    .ok_or_else(|| "--nonzero requires variable names".to_string())?;
                nonzero.extend(names.split(',').map(str::to_string));
            }
            "--saturation-timeout" => {
                saturation_timeout_ms = Some(
                    args.next()
                        .ok_or_else(|| "--saturation-timeout requires a value".to_string())?
                        .parse()
                        .map_err(|error| format!("invalid --saturation-timeout: {error}"))?,
                );
            }
            "--summary" => summary = true,
            _ if arg.starts_with("--") => return Err(format!("unknown option {arg:?}")),
            _ if path.is_some() => return Err(format!("unexpected positional argument {arg:?}")),
            _ => path = Some(PathBuf::from(arg)),
        }
    }
    Ok(CliOptions {
        path: path.unwrap_or_else(|| default_systems_dir_from_manifest().join("system.txt")),
        timeout_ms,
        gb_backend,
        nonzero,
        saturation_timeout_ms,
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

fn prepare_polynomials(loaded: &LoadedMathematicaSystem) -> Vec<PolyExtLex> {
    let started = Instant::now();
    let polynomials = polynomials_from_loaded(loaded, &Q).unwrap_or_else(|error| die(error, 1));
    println!(
        "  {} polynomial(s), elapsed {:?}",
        polynomials
            .iter()
            .filter(|polynomial| !polynomial.is_zero())
            .count(),
        started.elapsed()
    );
    polynomials
}

/// `Saturate(P, N)`: the saturation of `polynomials` by the product of the
/// `nonzero` variables, or `polynomials` when it does not finish in `timeout`,
/// and the time spent on it.
fn saturate_input(
    polynomials: Vec<PolyExtLex>,
    nonzero: &[PolyVariable],
    timeout: Duration,
) -> (Vec<PolyExtLex>, Duration) {
    if nonzero.is_empty() {
        return (polynomials, Duration::ZERO);
    }
    let started = Instant::now();
    let (saturated, description) = match saturate_by_variables(&polynomials, nonzero, timeout) {
        Saturation::Basis(basis) => {
            let description = format!("{} generator(s)", basis.len());
            (basis, description)
        }
        Saturation::Unit => (vec![polynomials[0].one()], "the unit ideal".to_string()),
        Saturation::Timeout => (polynomials, "timed out; keeping the input".to_string()),
        Saturation::Error(error) => (polynomials, format!("failed ({error}); keeping the input")),
    };
    let elapsed = started.elapsed();
    println!(
        "  saturation by {} nonzero variable(s): {description}, elapsed {elapsed:?}",
        nonzero.len()
    );
    (saturated, elapsed)
}

/// The factors and ordered minimal hitting sets of `polynomials`, with the
/// factors among the nonzero `conditions` removed.
fn factor_covers(
    polynomials: Vec<PolyExtLex>,
    conditions: &[PolyExtLex],
) -> (Vec<PolyExtLex>, Vec<BitSet>) {
    let groups = try_factor_groups_from_polys(polynomials)
        .unwrap_or_else(|error| die(format!("factorization error: {error}"), 1));
    let groups = without_conditions(groups, conditions);
    let (factors, index_groups) = build_factor_index(&groups);
    let covers = ordered_cover_masks(&minimal_covers(&index_groups));
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
    /// The hitting sets the backend did not decide in time, with their indices.
    timed_out: Vec<(usize, Vec<PolyExtLex>)>,
    elapsed: Duration,
}

impl SearchResult {
    fn complete(&self) -> bool {
        self.timed_out.is_empty()
    }
}

/// Check every hitting set with the backend; the `nonzero` variables reach it
/// as Rabinowitsch equations, and a dimension counts their auxiliary
/// variables, which a solution determines.
fn run_search(
    options: &CliOptions,
    covers: &[Vec<PolyExtLex>],
    nvars: usize,
    nonzero: &[PolyVariable],
) -> SearchResult {
    let timeout = Duration::from_millis(options.timeout_ms);
    let started = Instant::now();
    let mut components = Vec::new();
    let mut first_solution = None;
    let mut solved = 0;
    let mut inconsistent = 0;
    let mut timed_out = Vec::new();

    println!(
        "\nBranching over {} minimal hitting set(s), fewest-factors-first, {}ms/cover timeout, {} Groebner backend",
        covers.len(),
        options.timeout_ms,
        options.gb_backend.label()
    );
    for (cover_index, candidate) in covers.iter().enumerate() {
        let candidate = candidate.clone();
        let cover_started = Instant::now();
        let solver_input = with_nonzero_equations(candidate.clone(), nonzero)
            .unwrap_or_else(|error| die(format!("nonzero conditions: {error}"), 1));
        let nvars = solver_input.first().map_or(nvars, PolyExtLex::nvars);
        match evaluate_cover(solver_input, timeout, options.gb_backend) {
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
                timed_out.push((cover_index, candidate));
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
        timed_out,
        elapsed: started.elapsed(),
    }
}

fn report_results(result: &SearchResult) {
    println!(
        "\n  {} cover(s): {} satisfiable, {} unsatisfiable, {} timeout(s), complete={}, elapsed {:?}",
        result.total_covers,
        result.solved,
        result.inconsistent,
        result.timed_out.len(),
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
    for (cover_index, candidate) in &result.timed_out {
        println!(
            "  timed-out cover #{cover_index}: {} factor(s)",
            candidate.len()
        );
        for polynomial in candidate {
            println!("\tfactor: {polynomial}");
        }
    }
}

fn main() {
    LicenseManager::set_license_key(polynomial_cdcl::SYMBOLICA_LICENSE_KEY).unwrap();
    polynomial_cdcl::exit_if_solver_worker();

    let options = parse_args(std::env::args().skip(1)).unwrap_or_else(|error| die(error, 2));
    println!("Loading {}", options.path.display());
    let loaded = load_system(&options.path);
    let nonzero = nonzero_variables(&loaded, &options.nonzero)
        .unwrap_or_else(|error| die(format!("--nonzero: {error}"), 2));
    let saturation_timeout =
        Duration::from_millis(options.saturation_timeout_ms.unwrap_or(options.timeout_ms));
    let (polynomials, saturation_elapsed) =
        saturate_input(prepare_polynomials(&loaded), &nonzero, saturation_timeout);
    let nvars = polynomials
        .first()
        .map_or(loaded.var_names.len(), PolyExtLex::nvars);
    let conditions = polynomials
        .first()
        .map_or(Ok(Vec::new()), |template| {
            condition_polynomials(template, &nonzero)
        })
        .unwrap_or_else(|error| die(format!("--nonzero: {error}"), 2));
    let (factors, covers) = factor_covers(polynomials, &conditions);
    let (covers, refinement) =
        refine_covers_under_conditions(&factors, &covers, &conditions, &mut InProcess)
            .unwrap_or_else(|error| die(format!("hitting-set refinement error: {error}"), 1));
    println!(
        "  recursive refinement produced {} terminal hitting set(s) in {} pass(es), {} subdivision(s), maximum depth {}, {} pruned as contained in another, {} dropped by a side condition",
        covers.len(),
        refinement.passes,
        refinement.subdivisions,
        refinement.max_depth,
        refinement.pruned,
        refinement.condition_kills
    );
    let result = run_search(&options, &covers, nvars, &nonzero);

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
            "BRANCHING_SUMMARY total={} satisfiable={} unsatisfiable={} timeouts={} complete={} largest_dimension={} first_solution_secs={} elapsed_secs={:.6} saturation_secs={:.6}",
            result.total_covers,
            result.solved,
            result.inconsistent,
            result.timed_out.len(),
            result.complete(),
            largest_dimension,
            first_solution,
            result.elapsed.as_secs_f64(),
            saturation_elapsed.as_secs_f64()
        );
    } else {
        report_results(&result);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Once;

    use polynomial_cdcl::cover::decode_cover;
    use polynomial_cdcl::load_mathematica_system_from_str_with_var_names;
    use polynomial_cdcl::refine::refine_covers;

    static LICENSE: Once = Once::new();

    fn activate_license() {
        LICENSE.call_once(|| {
            let _ = LicenseManager::set_license_key(polynomial_cdcl::SYMBOLICA_LICENSE_KEY);
        });
    }

    fn test_options() -> CliOptions {
        CliOptions {
            path: PathBuf::from("unused"),
            timeout_ms: 5_000,
            gb_backend: GbBackend::Symbolica,
            nonzero: Vec::new(),
            saturation_timeout_ms: None,
            summary: true,
        }
    }

    #[test]
    fn parse_args_reads_every_option() {
        let options = parse_args(
            ["input.txt", "--timeout", "250", "--gb", "gvw", "--summary"]
                .into_iter()
                .map(String::from),
        )
        .expect("arguments");

        assert_eq!(options.path, PathBuf::from("input.txt"));
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
        let (factors, covers) = factor_covers(polynomials, &[]);
        let (covers, _) = refine_covers(&factors, &covers).expect("refinement");

        let result = run_search(&test_options(), &covers, 2, &[]);

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
        let (factors, covers) = factor_covers(polynomials, &[]);
        let covers = covers
            .iter()
            .map(|cover| decode_cover(&factors, cover))
            .collect::<Vec<_>>();

        let result = run_search(&test_options(), &covers, 3, &[]);

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
        let (factors, covers) = factor_covers(polynomials, &[]);
        assert_eq!(covers.len(), 1);

        let (terminal, stats) = refine_covers(&factors, &covers).expect("refinement");

        assert_eq!(terminal.len(), 2);
        assert_eq!(stats.subdivisions, 1);
        assert!(stats.max_depth >= 2);
        assert!(terminal.iter().all(|cover| cover.len() == 2));
    }

    #[test]
    fn nonzero_variable_leaves_its_factor_out_and_reaches_the_solver() {
        activate_license();
        let loaded =
            load_mathematica_system_from_str_with_var_names("{x*y == 0, x*z == 0}").expect("load");
        let polynomials = polynomials_from_loaded(&loaded, &Q).expect("polynomials");
        let nonzero = nonzero_variables(&loaded, &["x".to_string()]).expect("variables");
        let conditions = condition_polynomials(&polynomials[0], &nonzero).expect("conditions");
        let (factors, covers) = factor_covers(polynomials, &conditions);
        let (covers, _) =
            refine_covers_under_conditions(&factors, &covers, &conditions, &mut InProcess)
                .expect("refinement");

        let result = run_search(&test_options(), &covers, 3, &nonzero);

        assert_eq!(covers.len(), 1);
        assert_eq!(covers[0].len(), 2);
        assert!(!covers[0].contains(&conditions[0]));
        assert_eq!(result.solved, 1);
        assert_eq!(result.components[0].dimension, Some(1));
    }
}
