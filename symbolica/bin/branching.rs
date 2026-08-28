//! Factor-branching solver corresponding to Algorithm `Branching` in
//! `paper/sections/method.tex`.
//!
//! The input generators are optionally interreduced and factored. Every
//! minimal hitting set is then interreduced and factored again until no new
//! subbranches appear. The terminal hitting sets are checked in
//! fewest-factors-first order by a selectable Gröbner backend; inconsistent
//! sets yield iteratively refined GB-Cores that prune later supersets.
//!
//! Usage: `branching [path] [--unsat-core-search]` — default path:
//! `systems/system.txt`; default Gröbner backend: Singular.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use symbolica::LicenseManager;
use symbolica::domains::rational::Q;

use polynomial_cdcl::cover::{
    build_factor_index, decode_cover, minimal_covers, ordered_cover_masks,
};
use polynomial_cdcl::gb_core::iteratively_refined_unsat_core;
use polynomial_cdcl::gb_search::{
    ComputedBasis, Core, GbBackend, Outcome, evaluate_cover, polynomial_set_contains,
    solution_dimension,
};
use polynomial_cdcl::{
    LoadedMathematicaSystem, PolyExtLex, default_systems_dir_from_manifest, interreduce_polys,
    load_mathematica_system_from_path_with_var_names, polynomials_from_loaded,
    try_factor_groups_from_polys,
};

#[derive(Debug)]
struct CliOptions {
    path: PathBuf,
    interreduce: bool,
    max_covers: usize,
    timeout_ms: u64,
    gb_backend: GbBackend,
    unsat_core_search: bool,
    summary: bool,
}

fn parse_args(args: impl IntoIterator<Item = String>) -> Result<CliOptions, String> {
    let mut path: Option<PathBuf> = None;
    let mut interreduce = true;
    let mut max_covers = usize::MAX;
    let mut timeout_ms = 5_000;
    let mut gb_backend = GbBackend::SingularGb;
    let mut unsat_core_search = false;
    let mut summary = false;
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--no-interreduce" => interreduce = false,
            "--max-covers" => {
                max_covers = args
                    .next()
                    .ok_or_else(|| "--max-covers requires a value".to_string())?
                    .parse()
                    .map_err(|error| format!("invalid --max-covers: {error}"))?;
            }
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
            "--unsat-core-search" => unsat_core_search = true,
            "--summary" => summary = true,
            _ if arg.starts_with("--") => return Err(format!("unknown option {arg:?}")),
            _ if path.is_some() => return Err(format!("unexpected positional argument {arg:?}")),
            _ => path = Some(PathBuf::from(arg)),
        }
    }
    Ok(CliOptions {
        path: path.unwrap_or_else(|| default_systems_dir_from_manifest().join("system.txt")),
        interreduce,
        max_covers,
        timeout_ms,
        gb_backend,
        unsat_core_search,
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

#[derive(Default)]
struct RefinementStats {
    passes: usize,
    subdivisions: usize,
    max_depth: usize,
}

fn normalize_candidate(polynomials: Vec<PolyExtLex>) -> Vec<PolyExtLex> {
    polynomials
        .into_iter()
        .filter(|polynomial| !polynomial.is_zero())
        .map(PolyExtLex::make_monic)
        .fold(Vec::new(), |mut normalized, polynomial| {
            if !normalized.contains(&polynomial) {
                normalized.push(polynomial);
            }
            normalized
        })
}

fn same_polynomial_set(left: &[PolyExtLex], right: &[PolyExtLex]) -> bool {
    left.len() == right.len()
        && polynomial_set_contains(left, right)
        && polynomial_set_contains(right, left)
}

fn factor_cover_candidates(polynomials: Vec<PolyExtLex>) -> Result<Vec<Vec<PolyExtLex>>, String> {
    let groups = try_factor_groups_from_polys(polynomials)?;
    let (factors, index_groups) = build_factor_index(&groups);
    let covers = minimal_covers(&index_groups, factors.len()).ok_or_else(|| {
        format!(
            "too many distinct factors ({}); the cover enumerator supports at most 63",
            factors.len()
        )
    })?;
    Ok(ordered_cover_masks(&covers)
        .into_iter()
        .map(|cover| decode_cover(&factors, cover))
        .collect())
}

fn refine_cover(
    candidate: Vec<PolyExtLex>,
    interreduce: bool,
    depth: usize,
    ancestors: &mut Vec<Vec<PolyExtLex>>,
    terminal: &mut Vec<Vec<PolyExtLex>>,
    stats: &mut RefinementStats,
) -> Result<(), String> {
    stats.passes += 1;
    stats.max_depth = stats.max_depth.max(depth);

    let candidate = normalize_candidate(candidate);
    let reduced = if interreduce {
        interreduce_polys(candidate.clone())
    } else {
        candidate.clone()
    };
    let children = factor_cover_candidates(reduced.clone())?;

    if children.is_empty() {
        return Ok(());
    }

    let reduced = normalize_candidate(reduced);
    if children.len() == 1 && same_polynomial_set(&children[0], &reduced) {
        if !terminal
            .iter()
            .any(|existing| same_polynomial_set(existing, &children[0]))
        {
            terminal.push(children.into_iter().next().expect("one child"));
        }
        return Ok(());
    }

    if children.len() > 1 {
        stats.subdivisions += children.len() - 1;
    }
    ancestors.push(candidate);
    for child in children {
        if ancestors
            .iter()
            .any(|ancestor| same_polynomial_set(ancestor, &child))
        {
            if !terminal
                .iter()
                .any(|existing| same_polynomial_set(existing, &child))
            {
                terminal.push(child);
            }
        } else {
            refine_cover(child, interreduce, depth + 1, ancestors, terminal, stats)?;
        }
    }
    ancestors.pop();
    Ok(())
}

fn refine_covers(
    factors: &[PolyExtLex],
    covers: &[u64],
    interreduce: bool,
) -> Result<(Vec<Vec<PolyExtLex>>, RefinementStats), String> {
    let mut terminal = Vec::new();
    let mut stats = RefinementStats::default();
    for &cover in covers {
        refine_cover(
            decode_cover(factors, cover),
            interreduce,
            1,
            &mut Vec::new(),
            &mut terminal,
            &mut stats,
        )?;
    }
    terminal.sort_by_key(Vec::len);
    Ok((terminal, stats))
}

struct SearchResult {
    components: Vec<Component>,
    cores: Vec<Core>,
    first_solution: Option<Duration>,
    total_covers: usize,
    considered: usize,
    solved: usize,
    inconsistent: usize,
    core_pruned: usize,
    timeouts: usize,
    elapsed: Duration,
}

impl SearchResult {
    fn complete(&self) -> bool {
        self.considered == self.total_covers && self.timeouts == 0
    }
}

fn run_search(options: &CliOptions, covers: &[Vec<PolyExtLex>], nvars: usize) -> SearchResult {
    let timeout = Duration::from_millis(options.timeout_ms);
    let started = Instant::now();
    let mut components = Vec::new();
    let mut cores: Vec<Core> = Vec::new();
    let mut first_solution = None;
    let mut considered = 0;
    let mut solved = 0;
    let mut inconsistent = 0;
    let mut core_pruned = 0;
    let mut timeouts = 0;

    println!(
        "\nBranching over {} minimal hitting set(s), fewest-factors-first, {}ms/cover timeout, {} Groebner backend",
        covers.len(),
        options.timeout_ms,
        options.gb_backend.label()
    );
    for (cover_index, candidate) in covers.iter().enumerate() {
        if considered >= options.max_covers {
            break;
        }
        considered += 1;
        let candidate = candidate.clone();

        if cores
            .iter()
            .any(|core| polynomial_set_contains(&candidate, core))
        {
            core_pruned += 1;
            println!(
                "  cover #{cover_index}: core-pruned ({} factor(s))",
                candidate.len()
            );
            continue;
        }

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
                if options.unsat_core_search {
                    let core_started = Instant::now();
                    let core = iteratively_refined_unsat_core(&candidate, timeout);
                    println!(
                        "  cover #{cover_index}: unsatisfiable; learned a {}-factor core (solve {:?}, refinement {:?})",
                        core.len(),
                        cover_started.elapsed(),
                        core_started.elapsed()
                    );
                    cores.push(core);
                } else {
                    println!(
                        "  cover #{cover_index}: unsatisfiable; core search disabled (solve {:?})",
                        cover_started.elapsed()
                    );
                }
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
        cores,
        first_solution,
        total_covers: covers.len(),
        considered,
        solved,
        inconsistent,
        core_pruned,
        timeouts,
        elapsed: started.elapsed(),
    }
}

fn report_results(result: &SearchResult) {
    println!(
        "\n  {} of {} cover(s) considered: {} satisfiable, {} unsatisfiable, {} core-pruned, {} timeout(s), complete={}, elapsed {:?}",
        result.considered,
        result.total_covers,
        result.solved,
        result.inconsistent,
        result.core_pruned,
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
    if !result.cores.is_empty() {
        println!(
            "  learned {} iteratively refined unsatisfiable core(s)",
            result.cores.len()
        );
    }
}

fn main() {
    let key = "5568d092#6bb8b4fb#ff936916-65ba-552f-b3c1-bdc64597cd21";
    LicenseManager::set_license_key(key).unwrap();
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
            "BRANCHING_SUMMARY total={} considered={} satisfiable={} unsatisfiable={} core_pruned={} timeouts={} complete={} largest_dimension={} first_solution_secs={} elapsed_secs={:.6}",
            result.total_covers,
            result.considered,
            result.solved,
            result.inconsistent,
            result.core_pruned,
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
            let _ = LicenseManager::set_license_key(
                "5568d092#6bb8b4fb#ff936916-65ba-552f-b3c1-bdc64597cd21",
            );
        });
    }

    fn test_options() -> CliOptions {
        CliOptions {
            path: PathBuf::from("unused"),
            interreduce: false,
            max_covers: usize::MAX,
            timeout_ms: 5_000,
            gb_backend: GbBackend::Symbolica,
            unsat_core_search: true,
            summary: true,
        }
    }

    #[test]
    fn parse_args_matches_idos_style() {
        let options = parse_args(
            [
                "input.txt",
                "--no-interreduce",
                "--max-covers",
                "17",
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
        assert_eq!(options.max_covers, 17);
        assert_eq!(options.timeout_ms, 250);
        assert_eq!(options.gb_backend, GbBackend::Gvw);
        assert!(!options.unsat_core_search);
        assert!(options.summary);
    }

    #[test]
    fn parse_args_uses_fast_defaults() {
        let options = parse_args(std::iter::empty()).expect("arguments");

        assert_eq!(options.gb_backend, GbBackend::SingularGb);
        assert!(!options.unsat_core_search);
    }

    #[test]
    fn parse_args_can_enable_unsat_core_search() {
        let options =
            parse_args(["--unsat-core-search"].into_iter().map(String::from)).expect("arguments");

        assert!(options.unsat_core_search);
    }

    #[test]
    fn parse_args_rejects_removed_no_unsat_core_search_flag() {
        let error = parse_args(["--no-unsat-core-search"].into_iter().map(String::from))
            .expect_err("removed flag should be rejected");

        assert_eq!(error, "unknown option \"--no-unsat-core-search\"");
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
        assert_eq!(result.considered, 2);
        assert_eq!(result.solved, 2);
        assert!(result.first_solution.is_some());
        assert!(result.complete());
    }

    #[test]
    fn learned_core_prunes_a_later_superset() {
        activate_license();
        let loaded =
            load_mathematica_system_from_str_with_var_names("{x == 0, x - 1 == 0, y*z == 0}")
                .expect("load");
        let polynomials = polynomials_from_loaded(&loaded, &Q).expect("polynomials");
        let (factors, covers) = factor_covers(polynomials);
        let (covers, _) = refine_covers(&factors, &covers, false).expect("refinement");

        let result = run_search(&test_options(), &covers, 3);

        assert_eq!(result.inconsistent, 1);
        assert_eq!(result.core_pruned, 1);
        assert!(result.cores.iter().all(|core| !core.is_empty()));
        assert!(result.complete());
    }

    #[test]
    fn disabled_core_search_checks_every_inconsistent_cover() {
        activate_license();
        let loaded =
            load_mathematica_system_from_str_with_var_names("{x == 0, x - 1 == 0, y*z == 0}")
                .expect("load");
        let polynomials = polynomials_from_loaded(&loaded, &Q).expect("polynomials");
        let (factors, covers) = factor_covers(polynomials);
        let (covers, _) = refine_covers(&factors, &covers, false).expect("refinement");
        let mut options = test_options();
        options.unsat_core_search = false;

        let result = run_search(&options, &covers, 3);

        assert_eq!(result.inconsistent, 2);
        assert_eq!(result.core_pruned, 0);
        assert!(result.cores.is_empty());
        assert!(result.complete());
    }

    #[test]
    fn interreduction_subdivides_a_hitting_set_until_stable() {
        activate_license();
        let loaded =
            load_mathematica_system_from_str_with_var_names("{x + y == 0, x + y + y*z == 0}")
                .expect("load");
        let candidate = polynomials_from_loaded(&loaded, &Q).expect("polynomials");
        let mut terminal = Vec::new();
        let mut stats = RefinementStats::default();

        refine_cover(
            candidate,
            true,
            1,
            &mut Vec::new(),
            &mut terminal,
            &mut stats,
        )
        .expect("refinement");

        assert_eq!(terminal.len(), 2);
        assert_eq!(stats.subdivisions, 1);
        assert!(stats.max_depth >= 2);
        assert!(terminal.iter().all(|cover| cover.len() == 2));
    }
}
