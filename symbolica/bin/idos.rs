//! A CDCL-style search over the minimal factor covers of a polynomial system.
//!
//! Implements Algorithm 1 of the method statement: the minimal factor covers
//! are tried easiest-first — more zeroed variables first, then fewer factors;
//! a cover that contains a learned
//! unsatisfiable core is skipped; a cover contained in a learned satisfiable
//! cover is also skipped by the dual monotonicity property; an inconsistent
//! cover is shrunk to a minimal unsatisfiable core that is learned to prune later
//! covers.
//!
//! Usage: `idos [path]` — default path: `systems/system.txt`.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use symbolica::LicenseManager;
use symbolica::domains::algebraic_number::AlgebraicExtension;
use symbolica::domains::rational::Q;

use polynomial_cdcl::cover::{
    Core, WorkItem, build_factor_index, core_subsumes, minimal_covers, ordered_work_items,
    satisfiable_cover_contains,
};
use polynomial_cdcl::gb_search::{
    GbBackend, Outcome, PolyExtGrevLex, evaluate_cover, minimal_unsat_core, solution_dimension,
};
use polynomial_cdcl::sparse::{decode_work_item, resolve_candidates, usable_sparse_variables};
use polynomial_cdcl::{
    LoadedMathematicaSystem, PolyExtLex, default_systems_dir_from_manifest,
    load_mathematica_system_from_path_with_var_names, preprocess,
};

#[derive(Debug)]
struct CliOptions {
    path: PathBuf,
    interreduce: bool,
    max_covers: usize,
    timeout_ms: u64,
    gb_backend: GbBackend,
    sparse: bool,
    sparse_names: Vec<String>,
}

fn parse_args(args: impl IntoIterator<Item = String>) -> Result<CliOptions, String> {
    let mut path: Option<PathBuf> = None;
    let mut interreduce = true;
    let mut max_covers: usize = 2_000;
    let mut timeout_ms: u64 = 5_000;
    let mut gb_backend = GbBackend::M4gb;
    let mut sparse = false;
    let mut sparse_names = Vec::new();
    let mut args = args.into_iter().peekable();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--no-interreduce" => interreduce = false,
            "--max-covers" => {
                max_covers = args
                    .next()
                    .ok_or_else(|| "--max-covers requires a value".to_string())?
                    .parse()
                    .map_err(|e| format!("invalid --max-covers: {e}"))?;
            }
            "--timeout" => {
                timeout_ms = args
                    .next()
                    .ok_or_else(|| "--timeout requires a value".to_string())?
                    .parse()
                    .map_err(|e| format!("invalid --timeout: {e}"))?;
            }
            "--gb" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--gb requires one of: m4gb, f5, gvw, symbolica".to_string())?;
                gb_backend = GbBackend::parse(&value).ok_or_else(|| {
                    format!("invalid --gb {value:?}: expected m4gb, f5, gvw, or symbolica")
                })?;
            }
            "--sparse" => {
                sparse = true;
                if args.peek().is_some_and(|next| !next.starts_with("--")) {
                    sparse_names.push(args.next().expect("peeked sparse argument"));
                }
            }
            _ if arg.starts_with("--") => return Err(format!("unknown option {arg:?}")),
            _ => path = Some(PathBuf::from(arg)),
        }
    }
    Ok(CliOptions {
        path: path.unwrap_or_else(|| default_systems_dir_from_manifest().join("system.txt")),
        interreduce,
        max_covers,
        timeout_ms,
        gb_backend,
        sparse,
        sparse_names,
    })
}

fn die(msg: impl std::fmt::Display, code: i32) -> ! {
    eprintln!("{msg}");
    std::process::exit(code)
}

fn load_system(path: &Path) -> LoadedMathematicaSystem {
    let loaded = match load_mathematica_system_from_path_with_var_names(path) {
        Ok(s) => s,
        Err(e) => die(format!("load error: {e}"), 1),
    };
    println!(
        "Parsed: {} equations, {} variables (loader order)",
        loaded.system.equations.len(),
        loaded.var_names.len()
    );
    loaded
}

fn preprocess_system(
    loaded: &LoadedMathematicaSystem,
    field: &AlgebraicExtension<Q>,
    interreduce: bool,
) -> Vec<Vec<PolyExtLex>> {
    let t_pre = Instant::now();
    let pre = match preprocess(loaded, field, interreduce) {
        Ok(pre) => pre,
        Err(e) => die(e, 1),
    };
    println!(
        "  {} non-trivial polynomial(s), elapsed {:?}",
        pre.len(),
        t_pre.elapsed()
    );
    pre
}

fn build_cover_universe(pre: &[Vec<PolyExtLex>]) -> (Vec<PolyExtLex>, Vec<u64>) {
    let (factors, index_groups) = build_factor_index(pre);
    let base_covers = match minimal_covers(&index_groups, factors.len()) {
        Some(covers) => covers,
        None => die(
            "too many distinct factors (>= 64) for the bitmask enumerator",
            1,
        ),
    };
    println!(
        "  {} minimal factor cover(s) (hitting sets, no variable zeroing)",
        base_covers.len()
    );
    (factors, base_covers)
}

fn resolve_sparse_variables(
    options: &CliOptions,
    loaded: &LoadedMathematicaSystem,
    pre: &[Vec<PolyExtLex>],
) -> Vec<(String, usize)> {
    if !options.sparse {
        return Vec::new();
    }
    let candidates = match resolve_candidates(loaded, &options.sparse_names) {
        Ok(candidates) => candidates,
        Err(e) => die(e, 2),
    };
    let sparse_variables = usable_sparse_variables(&pre[0][0], &candidates);
    if sparse_variables.len() >= 64 {
        die(
            "too many usable sparse variables (>= 64) for the bitmask enumerator",
            1,
        );
    }
    if options.sparse_names.is_empty() {
        println!(
            "  sparsity enabled for all variables: {} usable substitution(s)",
            sparse_variables.len()
        );
    } else {
        println!(
            "  sparsity enabled for {} named variable(s): {} usable substitution(s)",
            options.sparse_names.len(),
            sparse_variables.len()
        );
    }
    sparse_variables
}

struct SearchResult {
    solutions: Vec<(usize, usize, Option<usize>)>,
    largest_dimension: Option<(usize, WorkItem, usize, Vec<PolyExtGrevLex>)>,
    cores: Vec<Core>,
    inconsistent: usize,
    timeouts: usize,
    core_pruned: usize,
    satisfiable_inferred: usize,
    elapsed: Duration,
}

fn run_search(
    options: &CliOptions,
    factors: &[PolyExtLex],
    sparse_variables: &[(String, usize)],
    base_covers: &[u64],
) -> SearchResult {
    let total_items = (base_covers.len() as u128) << sparse_variables.len();
    if options.sparse {
        println!("  {total_items} cover variant(s) after optional sparsity subsets");
    }

    let tried = total_items.min(options.max_covers as u128);
    println!(
        "\nCDCL over {tried} of {total_items} minimal cover(s), sparsest-first then fewest-factors order, {}ms/cover timeout, {} Groebner backend",
        options.timeout_ms,
        options.gb_backend.label(),
    );

    let timeout = Duration::from_millis(options.timeout_ms);
    let t_gb = Instant::now();
    let mut cores: Vec<Core> = Vec::new();
    let mut satisfiable_covers: Vec<(usize, WorkItem)> = Vec::new();
    let mut solutions: Vec<(usize, usize, Option<usize>)> = Vec::new();
    let mut largest_dimension: Option<(usize, WorkItem, usize, Vec<PolyExtGrevLex>)> = None;
    let mut inconsistent = 0usize;
    let mut timeouts = 0usize;
    let mut core_pruned = 0usize;
    let mut satisfiable_inferred = 0usize;
    for (idx, item) in ordered_work_items(base_covers, sparse_variables.len())
        .take(options.max_covers)
        .enumerate()
    {
        // Skip the cover if any learned core is a subset of it (not an
        // equality membership test, despite `manual_contains`).
        #[allow(clippy::manual_contains)]
        let subsumed = cores.iter().any(|&core| core_subsumes(core, item));
        if subsumed {
            core_pruned += 1;
            continue;
        }

        // If F is contained in a previously satisfiable F', then
        // V(F) ⊇ V(F') != ∅, so F is also satisfiable. This is the dual of the
        // unsatisfiable-core superset rule above.
        #[allow(clippy::manual_contains)]
        let contained_in_satisfiable = satisfiable_covers
            .iter()
            .find(|&&(_, solution)| satisfiable_cover_contains(solution, item));
        let inferred_from = contained_in_satisfiable.map(|&(solution_idx, _)| solution_idx);
        if inferred_from.is_some() {
            satisfiable_inferred += 1;
        }
        let t_cover = Instant::now();
        match evaluate_cover(
            decode_work_item(factors, sparse_variables, item),
            timeout,
            options.gb_backend,
        ) {
            Outcome::Consistent(basis) => {
                let len = basis.len();
                let dimension = solution_dimension(&basis, factors[0].nvars());
                let dimension_label = dimension.map_or_else(|| "?".to_string(), |d| d.to_string());
                if let Some(solution_idx) = inferred_from {
                    println!(
                        "  cover #{idx}: satisfiable-inferred from cover #{solution_idx}, basis computed ({len} element(s), dimension {dimension_label}, elapsed {:?})",
                        t_cover.elapsed()
                    );
                } else {
                    println!(
                        "  cover #{idx}: satisfiable ({len} basis element(s), dimension {dimension_label}, elapsed {:?})",
                        t_cover.elapsed()
                    );
                    satisfiable_covers.push((idx, item));
                }
                solutions.push((idx, len, dimension));
                if let Some(dimension) = dimension {
                    let improves =
                        largest_dimension
                            .as_ref()
                            .is_none_or(|&(_, _, best_dim, ref best)| {
                                dimension > best_dim || (dimension == best_dim && len < best.len())
                            });
                    if improves {
                        largest_dimension = Some((idx, item, dimension, basis));
                    }
                }
            }
            Outcome::Inconsistent => {
                if let Some(solution_idx) = inferred_from {
                    println!(
                        "  cover #{idx}: unsatisfiable despite containment in satisfiable cover #{solution_idx} (elapsed {:?})",
                        t_cover.elapsed()
                    );
                } else {
                    println!(
                        "  cover #{idx}: unsatisfiable (elapsed {:?})",
                        t_cover.elapsed()
                    );
                }
                cores.push(minimal_unsat_core(
                    factors,
                    sparse_variables,
                    item,
                    timeout,
                    options.gb_backend,
                ));
                inconsistent += 1;
            }
            Outcome::Timeout => {
                if let Some(solution_idx) = inferred_from {
                    println!(
                        "  cover #{idx}: satisfiable-inferred from cover #{solution_idx}, basis timed out (elapsed {:?})",
                        t_cover.elapsed()
                    );
                } else {
                    println!(
                        "  cover #{idx}: timed out (elapsed {:?})",
                        t_cover.elapsed()
                    );
                }
                timeouts += 1;
            }
        }
    }

    SearchResult {
        solutions,
        largest_dimension,
        cores,
        inconsistent,
        timeouts,
        core_pruned,
        satisfiable_inferred,
        elapsed: t_gb.elapsed(),
    }
}

fn report_results(
    result: &SearchResult,
    sparse_variables: &[(String, usize)],
    loaded: &LoadedMathematicaSystem,
) {
    println!(
        "  {} satisfiable, {} unsatisfiable, {} core-pruned, {} satisfiable-inferred, {} timed out, elapsed {:?}",
        result.solutions.len(),
        result.inconsistent,
        result.core_pruned,
        result.satisfiable_inferred,
        result.timeouts,
        result.elapsed
    );
    for (idx, len, dimension) in result.solutions.iter().take(20) {
        let dimension_label = dimension.map_or_else(|| "?".to_string(), |d| d.to_string());
        println!("\tcover #{idx}: {len} basis element(s), dimension {dimension_label}");
    }
    if let Some((idx, item, dimension, basis)) = &result.largest_dimension {
        println!(
            "  largest solution dimension found: cover #{idx}, dimension {dimension} ({} basis element(s))",
            basis.len()
        );
        let zeroed_variables: Vec<_> = sparse_variables
            .iter()
            .enumerate()
            .filter_map(|(i, (name, _))| {
                (item.zero_mask & (1u64 << i) != 0).then_some(name.as_str())
            })
            .collect();
        let variables_not_zeroed: Vec<_> = loaded
            .var_names
            .iter()
            .filter_map(|name| {
                zeroed_variables
                    .iter()
                    .all(|&zeroed| zeroed != name.as_str())
                    .then_some(name.as_str())
            })
            .collect();
        if variables_not_zeroed.is_empty() {
            println!("\tvariables not zeroed: <none>");
        } else {
            println!(
                "\tvariables not zeroed: {}",
                variables_not_zeroed.join(", ")
            );
        }
        if basis.is_empty() {
            println!("\t<empty basis>");
        } else {
            for (i, polynomial) in basis.iter().enumerate() {
                println!("\tg{i} = {polynomial}");
            }
        }
    }
    if !result.cores.is_empty() {
        println!(
            "  learned {} minimal unsatisfiable core(s)",
            result.cores.len()
        );
    }
}

fn main() {
    let key = "5568d092#6bb8b4fb#ff936916-65ba-552f-b3c1-bdc64597cd21";
    LicenseManager::set_license_key(key).unwrap();

    let options = match parse_args(std::env::args().skip(1)) {
        Ok(options) => options,
        Err(e) => die(e, 1),
    };
    let loaded = load_system(&options.path);

    let field = AlgebraicExtension::new_complex(Q);
    let pre = preprocess_system(&loaded, &field, options.interreduce);
    if pre.is_empty() {
        println!("Nothing to cover — all polynomials reduced to zero.");
        return;
    }

    let (factors, base_covers) = build_cover_universe(&pre);
    if base_covers.is_empty() {
        println!("No factor cover exists — some equation has no factors.");
        return;
    }

    let sparse_variables = resolve_sparse_variables(&options, &loaded, &pre);

    let result = run_search(&options, &factors, &sparse_variables, &base_covers);
    report_results(&result, &sparse_variables, &loaded);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_args_sparse_all_variables() {
        let options = parse_args(["--sparse"].into_iter().map(String::from)).unwrap();
        assert!(options.sparse);
        assert!(options.sparse_names.is_empty());
    }

    #[test]
    fn parse_args_sparse_named_variables() {
        let options = parse_args(
            ["system.txt", "--sparse", "x", "--sparse", "y"]
                .into_iter()
                .map(String::from),
        )
        .unwrap();
        assert!(options.sparse);
        assert_eq!(options.path, PathBuf::from("system.txt"));
        assert_eq!(options.sparse_names, vec!["x", "y"]);
    }

    #[test]
    fn parse_args_existing_options_still_work() {
        let options = parse_args(
            ["--no-interreduce", "--max-covers", "17", "--timeout", "250"]
                .into_iter()
                .map(String::from),
        )
        .unwrap();
        assert!(!options.interreduce);
        assert_eq!(options.max_covers, 17);
        assert_eq!(options.timeout_ms, 250);
        assert_eq!(options.gb_backend, GbBackend::M4gb);
        assert!(!options.sparse);
    }

    #[test]
    fn parse_args_groebner_backend() {
        let options = parse_args(["--gb", "gvw"].into_iter().map(String::from)).unwrap();
        assert_eq!(options.gb_backend, GbBackend::Gvw);
    }

    #[test]
    fn parse_args_invalid_groebner_backend_fails() {
        assert!(parse_args(["--gb", "buchberger"].into_iter().map(String::from)).is_err());
    }

    #[test]
    fn parse_args_unknown_flag_fails() {
        assert!(parse_args(["--missing"].into_iter().map(String::from)).is_err());
    }
}
