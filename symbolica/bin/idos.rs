//! Outer branching over structure hints followed by a CDCL-style search over
//! each augmented polynomial system's minimal factor covers.
//!
//! Implements Algorithm 1 of the method statement: optionally interreduce the
//! input, then try structure-hint subsets easiest-first. A hint subset is minimized and
//! learned only after every one of its minimal factor covers has been proved
//! inconsistent, allowing every later superset to be pruned. Learned
//! factor-cover cores are normalized polynomial sets, so they remain valid
//! across branches whose factor-index universes differ.
//!
//! Usage: `idos [path]` — default path: `systems/system.txt`; default Gröbner
//! backend: Singular.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use symbolica::domains::rational::Q;
use symbolica::LicenseManager;

use polynomial_cdcl::cover::{
    build_factor_index, decode_cover, minimal_covers, ordered_cover_masks, ordered_hint_masks,
};
use polynomial_cdcl::gb_core::{iteratively_refined_unsat_core, paper_gb_core};
use polynomial_cdcl::gb_search::{
    evaluate_cover, polynomial_set_contains, solution_dimension, ComputedBasis, Core, GbBackend,
    Outcome,
};
use polynomial_cdcl::sparse::{decode_hint_mask, resolve_candidates, usable_sparse_variables};
use polynomial_cdcl::{
    default_systems_dir_from_manifest, interreduce_polys,
    load_mathematica_system_from_path_with_var_names, polynomials_from_loaded,
    try_factor_groups_from_polys, LoadedMathematicaSystem, PolyExtLex,
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
    summary: bool,
}

fn parse_args(args: impl IntoIterator<Item = String>) -> Result<CliOptions, String> {
    let mut path: Option<PathBuf> = None;
    let mut interreduce = true;
    let mut max_covers: usize = 2_000;
    let mut timeout_ms: u64 = 5_000;
    let mut gb_backend = GbBackend::SingularGb;
    let mut sparse = false;
    let mut sparse_names = Vec::new();
    let mut summary = false;
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
                let value = args.next().ok_or_else(|| {
                    "--gb requires one of: m4gb, f5, gvw, symbolica, singular-gb".to_string()
                })?;
                gb_backend = GbBackend::parse(&value).ok_or_else(|| {
                    format!(
                        "invalid --gb {value:?}: expected m4gb, f5, gvw, symbolica, or singular-gb"
                    )
                })?;
            }
            "--sparse" => {
                sparse = true;
                if args.peek().is_some_and(|next| !next.starts_with("--")) {
                    sparse_names.push(args.next().expect("peeked sparse argument"));
                }
            }
            "--summary" => summary = true,
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
        summary,
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

fn prepare_base_polynomials(
    loaded: &LoadedMathematicaSystem,
    field: &Q,
    interreduce: bool,
) -> Vec<PolyExtLex> {
    let t_pre = Instant::now();
    let polynomials = match polynomials_from_loaded(loaded, field) {
        Ok(polynomials) if interreduce => interreduce_polys(polynomials),
        Ok(polynomials) => polynomials,
        Err(e) => die(e, 1),
    };
    println!(
        "  {} polynomial(s), interreduction {}, elapsed {:?}",
        polynomials.iter().filter(|poly| !poly.is_zero()).count(),
        if interreduce { "enabled" } else { "disabled" },
        t_pre.elapsed()
    );
    polynomials
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
    (factors, base_covers)
}

fn resolve_sparse_variables(
    options: &CliOptions,
    loaded: &LoadedMathematicaSystem,
    template: Option<&PolyExtLex>,
) -> Vec<(String, usize)> {
    if !options.sparse {
        return Vec::new();
    }
    let candidates = match resolve_candidates(loaded, &options.sparse_names) {
        Ok(candidates) => candidates,
        Err(e) => die(e, 2),
    };
    let sparse_variables = match template {
        Some(template) => usable_sparse_variables(template, &candidates),
        None if candidates.is_empty() => Vec::new(),
        None => die(
            "cannot construct sparse hints without an input polynomial",
            1,
        ),
    };
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

fn normalized_candidate(
    factors: &[PolyExtLex],
    cover_mask: u64,
    hints: &[PolyExtLex],
) -> Vec<PolyExtLex> {
    let mut candidate = decode_cover(factors, cover_mask);
    for hint in hints {
        if !candidate.contains(hint) {
            candidate.push(hint.clone());
        }
    }
    candidate
}

/// Shrink an unsatisfiable set of structure hints while keeping the base
/// polynomial system fixed. A hint is removed exactly when the resulting
/// augmented system is still inconsistent. Timeouts conservatively retain it.
fn minimal_unsat_hint_core(
    base_polynomials: &[PolyExtLex],
    hints: &[PolyExtLex],
    timeout: Duration,
    backend: GbBackend,
) -> Core {
    let mut augmented = base_polynomials.to_vec();
    augmented.extend(hints.iter().cloned());
    let mut core = paper_gb_core(&augmented, timeout)
        .map(|certificate| {
            hints
                .iter()
                .filter(|hint| certificate.contains(hint))
                .cloned()
                .collect()
        })
        .unwrap_or_else(|| hints.to_vec());
    let mut index = 0;
    while index < core.len() {
        let mut reduced = core.clone();
        reduced.remove(index);
        let mut augmented = base_polynomials.to_vec();
        augmented.extend(reduced.iter().cloned());
        let augmented = interreduce_polys(augmented);
        if matches!(
            evaluate_cover(augmented, timeout, backend),
            Outcome::Inconsistent
        ) {
            core = reduced;
        } else {
            index += 1;
        }
    }
    core
}

fn branch_proves_inconsistent(
    branch_complete: bool,
    branch_satisfiable: bool,
    branch_timed_out: bool,
) -> bool {
    branch_complete && !branch_satisfiable && !branch_timed_out
}

struct SearchResult {
    solutions: Vec<(usize, usize, Option<usize>)>,
    largest_dimension: Option<(usize, u64, usize, ComputedBasis)>,
    cores: Vec<Core>,
    hint_cores: Vec<Core>,
    branches: usize,
    considered: usize,
    inconsistent: usize,
    timeouts: usize,
    core_pruned: usize,
    hint_core_pruned: usize,
    hint_inconsistent: usize,
    satisfiable_inferred: usize,
    elapsed: Duration,
}

fn run_search(
    options: &CliOptions,
    base_polynomials: &[PolyExtLex],
    template: Option<&PolyExtLex>,
    sparse_variables: &[(String, usize)],
    nvars: usize,
) -> SearchResult {
    let total_branches = 1u128 << sparse_variables.len();
    println!(
        "\nOuter branching over {total_branches} hint subset(s), at most {} candidate cover(s), most-hints-first then fewest-factors order, {}ms/cover timeout, {} Groebner backend",
        options.max_covers,
        options.timeout_ms,
        options.gb_backend.label(),
    );

    let timeout = Duration::from_millis(options.timeout_ms);
    let t_gb = Instant::now();
    let mut cores: Vec<Core> = Vec::new();
    let mut hint_cores: Vec<Core> = Vec::new();
    let mut satisfiable_covers: Vec<(usize, Vec<PolyExtLex>)> = Vec::new();
    let mut solutions: Vec<(usize, usize, Option<usize>)> = Vec::new();
    let mut largest_dimension: Option<(usize, u64, usize, ComputedBasis)> = None;
    let mut branches = 0usize;
    let mut considered = 0usize;
    let mut inconsistent = 0usize;
    let mut timeouts = 0usize;
    let mut core_pruned = 0usize;
    let mut hint_core_pruned = 0usize;
    let mut hint_inconsistent = 0usize;
    let mut satisfiable_inferred = 0usize;
    for (branch_idx, hint_mask) in ordered_hint_masks(sparse_variables.len()).enumerate() {
        if considered >= options.max_covers {
            break;
        }
        branches += 1;
        let hints = template.map_or_else(Vec::new, |template| {
            decode_hint_mask(template, sparse_variables, hint_mask)
        });

        if hint_cores
            .iter()
            .any(|core| polynomial_set_contains(&hints, core))
        {
            hint_core_pruned += 1;
            continue;
        }

        let mut augmented = base_polynomials.to_vec();
        augmented.extend(hints.iter().cloned());
        let t_branch = Instant::now();
        let augmented = if options.interreduce {
            interreduce_polys(augmented)
        } else {
            augmented
        };
        let groups = try_factor_groups_from_polys(augmented)
            .unwrap_or_else(|error| die(format!("factorization error: {error}"), 1));
        let (factors, covers) = build_cover_universe(&groups);
        let ordered_covers = ordered_cover_masks(&covers);
        println!(
            "  branch #{branch_idx}: {} hint(s), {} factor group(s), {} minimal cover(s), preprocessing {:?}",
            hint_mask.count_ones(),
            groups.len(),
            ordered_covers.len(),
            t_branch.elapsed()
        );

        let mut branch_complete = true;
        let mut branch_satisfiable = false;
        let mut branch_timed_out = false;
        for cover_mask in ordered_covers {
            if considered >= options.max_covers {
                branch_complete = false;
                break;
            }
            let idx = considered;
            considered += 1;
            let candidate = normalized_candidate(&factors, cover_mask, &hints);

            if cores
                .iter()
                .any(|core| polynomial_set_contains(&candidate, core))
            {
                core_pruned += 1;
                continue;
            }

            // If the current generators are contained in a previously
            // satisfiable generator set, their variety contains a known point.
            // We still compute a basis for reporting and dimension comparison.
            let inferred_from = satisfiable_covers
                .iter()
                .find(|(_, solution)| polynomial_set_contains(solution, &candidate))
                .map(|(solution_idx, _)| *solution_idx);
            if inferred_from.is_some() {
                satisfiable_inferred += 1;
                branch_satisfiable = true;
            }
            let t_cover = Instant::now();
            match evaluate_cover(candidate.clone(), timeout, options.gb_backend) {
                Outcome::Consistent(basis) => {
                    branch_satisfiable = true;
                    let len = basis.elements.len();
                    let dimension = solution_dimension(&basis.elements, nvars);
                    let dimension_label =
                        dimension.map_or_else(|| "?".to_string(), |d| d.to_string());
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
                        satisfiable_covers.push((idx, candidate));
                    }
                    solutions.push((idx, len, dimension));
                    if let Some(dimension) = dimension {
                        let improves =
                            largest_dimension
                                .as_ref()
                                .is_none_or(|&(_, _, best_dim, ref best)| {
                                    dimension > best_dim
                                        || (dimension == best_dim && len < best.elements.len())
                                });
                        if improves {
                            largest_dimension = Some((idx, hint_mask, dimension, basis));
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
                    cores.push(iteratively_refined_unsat_core(&candidate, timeout));
                    inconsistent += 1;
                }
                Outcome::Timeout => {
                    branch_timed_out = true;
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
                Outcome::BackendError(error) => {
                    die(
                        format!(
                            "{} Groebner backend error: {error}",
                            options.gb_backend.label()
                        ),
                        1,
                    );
                }
            }
        }

        // A timeout or a partially explored branch is not evidence that the
        // augmented system is inconsistent. Learn a hint core only when every
        // minimal hitting set has been either checked or soundly core-pruned,
        // and none was satisfiable.
        if branch_proves_inconsistent(branch_complete, branch_satisfiable, branch_timed_out) {
            let t_core = Instant::now();
            let core =
                minimal_unsat_hint_core(base_polynomials, &hints, timeout, options.gb_backend);
            println!(
                "  branch #{branch_idx}: no satisfiable minimal cover; learned a {}-hint core ({:?})",
                core.len(),
                t_core.elapsed()
            );
            hint_cores.push(core);
            hint_inconsistent += 1;
        }
    }

    SearchResult {
        solutions,
        largest_dimension,
        cores,
        hint_cores,
        branches,
        considered,
        inconsistent,
        timeouts,
        core_pruned,
        hint_core_pruned,
        hint_inconsistent,
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
        "  {} branch(es), {} candidate(s): {} satisfiable, {} unsatisfiable, {} cover-core-pruned, {} hint-unsatisfiable, {} hint-core-pruned, {} satisfiable-inferred, {} cover timeout(s), elapsed {:?}",
        result.branches,
        result.considered,
        result.solutions.len(),
        result.inconsistent,
        result.core_pruned,
        result.hint_inconsistent,
        result.hint_core_pruned,
        result.satisfiable_inferred,
        result.timeouts,
        result.elapsed
    );
    for (idx, len, dimension) in result.solutions.iter().take(20) {
        let dimension_label = dimension.map_or_else(|| "?".to_string(), |d| d.to_string());
        println!("\tcover #{idx}: {len} basis element(s), dimension {dimension_label}");
    }
    if let Some((idx, hint_mask, dimension, basis)) = &result.largest_dimension {
        println!(
            "  largest solution dimension found: cover #{idx}, dimension {dimension} ({} basis element(s))",
            basis.elements.len()
        );
        let zeroed_variables: Vec<_> = sparse_variables
            .iter()
            .enumerate()
            .filter_map(|(i, (name, _))| (hint_mask & (1u64 << i) != 0).then_some(name.as_str()))
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
        if basis.elements.is_empty() {
            println!("\t<empty basis>");
        } else {
            if basis.leading_monomials_only {
                println!("\tleading monomials of the external basis:");
            }
            for (i, polynomial) in basis.elements.iter().enumerate() {
                println!("\tg{i} = {polynomial}");
            }
        }
    }
    if !result.cores.is_empty() {
        println!(
            "  learned {} iteratively refined unsatisfiable core(s)",
            result.cores.len()
        );
    }
    if !result.hint_cores.is_empty() {
        println!(
            "  learned {} minimal unsatisfiable structure-hint core(s)",
            result.hint_cores.len()
        );
    }
}

fn main() {
    let key = "5568d092#6bb8b4fb#ff936916-65ba-552f-b3c1-bdc64597cd21";
    LicenseManager::set_license_key(key).unwrap();
    polynomial_cdcl::exit_if_solver_worker();

    let options = match parse_args(std::env::args().skip(1)) {
        Ok(options) => options,
        Err(e) => die(e, 1),
    };
    let loaded = load_system(&options.path);

    let field = Q;
    let base_polynomials = prepare_base_polynomials(&loaded, &field, options.interreduce);
    let template = base_polynomials.first();
    let nvars = template.map_or(loaded.var_names.len(), PolyExtLex::nvars);
    let sparse_variables = resolve_sparse_variables(&options, &loaded, template);

    let result = run_search(
        &options,
        &base_polynomials,
        template,
        &sparse_variables,
        nvars,
    );
    if options.summary {
        let (largest_dimension, largest_basis_len) = result.largest_dimension.as_ref().map_or_else(
            || ("unknown".to_string(), 0),
            |(_, _, dimension, basis)| (dimension.to_string(), basis.elements.len()),
        );
        println!(
            "IDOS_SUMMARY branches={} candidates={} satisfiable={} unsatisfiable={} timeouts={} largest_dimension={} largest_basis_len={} elapsed_secs={:.6}",
            result.branches,
            result.considered,
            result.solutions.len(),
            result.inconsistent,
            result.timeouts,
            largest_dimension,
            largest_basis_len,
            result.elapsed.as_secs_f64(),
        );
    } else {
        report_results(&result, &sparse_variables, &loaded);
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
        assert_eq!(options.gb_backend, GbBackend::SingularGb);
        assert!(!options.sparse);
    }

    #[test]
    fn parse_args_groebner_backend() {
        let options = parse_args(["--gb", "gvw"].into_iter().map(String::from)).unwrap();
        assert_eq!(options.gb_backend, GbBackend::Gvw);

        let options = parse_args(["--gb", "singular-gb"].into_iter().map(String::from)).unwrap();
        assert_eq!(options.gb_backend, GbBackend::SingularGb);
    }

    #[test]
    fn parse_args_invalid_groebner_backend_fails() {
        assert!(parse_args(["--gb", "buchberger"].into_iter().map(String::from)).is_err());
    }

    #[test]
    fn parse_args_unknown_flag_fails() {
        assert!(parse_args(["--missing"].into_iter().map(String::from)).is_err());
    }

    #[test]
    fn parse_args_interreduces_by_default() {
        let options = parse_args(std::iter::empty()).unwrap();
        assert!(options.interreduce);
    }

    #[test]
    fn hint_core_requires_every_cover_to_be_proved_inconsistent() {
        assert!(branch_proves_inconsistent(true, false, false));
        assert!(!branch_proves_inconsistent(false, false, false));
        assert!(!branch_proves_inconsistent(true, true, false));
        assert!(!branch_proves_inconsistent(true, false, true));
    }

    #[test]
    fn structure_hint_is_interreduced_before_factorization() {
        activate_license();
        let loaded =
            load_mathematica_system_from_str_with_var_names("{(x + 1)*y == 0}").expect("load");
        let field = Q;
        let base =
            interreduce_polys(polynomials_from_loaded(&loaded, &field).expect("polynomials"));
        let candidates = resolve_candidates(&loaded, &["x".to_string()]).expect("candidate");
        let sparse_variables = usable_sparse_variables(&base[0], &candidates);
        let hints = decode_hint_mask(&base[0], &sparse_variables, 0b1);
        let mut augmented = base;
        augmented.extend(hints);

        let groups = try_factor_groups_from_polys(interreduce_polys(augmented))
            .expect("factor structure hint");
        let (factors, covers) = build_cover_universe(&groups);

        assert_eq!(groups.len(), 2);
        assert_eq!(factors.len(), 2);
        assert_eq!(covers.len(), 1);
        assert_eq!(covers[0].count_ones(), 2);
    }

    #[test]
    fn unsatisfiable_hint_core_minimizes_only_hints() {
        activate_license();
        let loaded =
            load_mathematica_system_from_str_with_var_names("{x - 1 == 0, y == 0}").expect("load");
        let field = Q;
        let base =
            interreduce_polys(polynomials_from_loaded(&loaded, &field).expect("polynomials"));
        let candidates =
            resolve_candidates(&loaded, &["x".to_string(), "y".to_string()]).expect("candidates");
        let sparse_variables = usable_sparse_variables(&base[0], &candidates);
        let all_hints_mask = (1u64 << sparse_variables.len()) - 1;
        let hints = decode_hint_mask(&base[0], &sparse_variables, all_hints_mask);
        let x_index = sparse_variables
            .iter()
            .position(|(name, _)| name == "x")
            .expect("x candidate");
        let x_hint = decode_hint_mask(&base[0], &sparse_variables, 1u64 << x_index);

        let core =
            minimal_unsat_hint_core(&base, &hints, Duration::from_secs(5), GbBackend::Symbolica);

        assert_eq!(core, x_hint);
    }

    #[test]
    fn paper_gb_core_extracts_certificate_support() {
        activate_license();
        let loaded =
            load_mathematica_system_from_str_with_var_names("{x == 0, x - 1 == 0, y == 0}")
                .expect("load");
        let field = Q;
        let candidate = polynomials_from_loaded(&loaded, &field).expect("polynomials");

        let core = paper_gb_core(&candidate, Duration::from_secs(5)).expect("GB-Core certificate");

        assert_eq!(core.len(), 2);
        assert!(core.contains(&candidate[0]));
        assert!(core.contains(&candidate[1]));
        assert!(!core.contains(&candidate[2]));
        assert!(matches!(
            evaluate_cover(core, Duration::from_secs(5), GbBackend::Symbolica),
            Outcome::Inconsistent
        ));
    }

    #[test]
    fn iterative_refinement_returns_a_sound_core() {
        activate_license();
        let loaded = load_mathematica_system_from_str_with_var_names(
            "{x == 0, x - 1 == 0, y == 0, y - 1 == 0, z == 0}",
        )
        .expect("load");
        let field = Q;
        let candidate = polynomials_from_loaded(&loaded, &field).expect("polynomials");

        let core = iteratively_refined_unsat_core(&candidate, Duration::from_secs(5));

        assert!(core.len() < candidate.len());
        assert!(core.iter().all(|polynomial| candidate.contains(polynomial)));
        assert!(matches!(
            evaluate_cover(core.clone(), Duration::from_secs(5), GbBackend::Symbolica),
            Outcome::Inconsistent
        ));
    }
}
