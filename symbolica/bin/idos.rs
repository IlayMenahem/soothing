//! A CDCL-style search over the minimal factor covers of a polynomial system.
//!
//! Implements Algorithm 1 of the method statement: the minimal factor covers
//! are tried cheapest-first by Bézout bound; a cover that contains a learned
//! unsatisfiable core is skipped; an inconsistent cover is shrunk to a minimal
//! unsatisfiable core that is learned to prune later covers.
//!
//! Usage: `idos [path]` — default path: `systems/system.txt`.

use std::cmp::Reverse;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use symbolica::LicenseManager;
use symbolica::atom::Symbol;
use symbolica::domains::algebraic_number::AlgebraicExtension;
use symbolica::domains::rational::Q;
use symbolica::poly::groebner::GroebnerBasis;
use symbolica::poly::polynomial::MultivariatePolynomial;
use symbolica::poly::{GrevLexOrder, MonomialOrder, PolyVariable};

use polynomial_cdcl::factor_prep::ExtQ;
use polynomial_cdcl::{
    LoadedMathematicaSystem, PolyExtLex, default_systems_dir_from_manifest,
    load_mathematica_system_from_path_with_var_names, preprocess,
};

/// Deduplicated factor universe (made monic) together with each equation's
/// factor group expressed as indices into that universe. Mirrors the dedup of
/// `build_problem`, so a factor shared by several equations gets one index.
fn build_factor_index(groups: &[Vec<PolyExtLex>]) -> (Vec<PolyExtLex>, Vec<Vec<usize>>) {
    let mut factor_index: HashMap<PolyExtLex, usize> = HashMap::new();
    let mut factors: Vec<PolyExtLex> = Vec::new();
    let index_groups = groups
        .iter()
        .map(|group| {
            group
                .iter()
                .map(|f| {
                    let monic = f.clone().make_monic();
                    *factor_index.entry(monic.clone()).or_insert_with(|| {
                        factors.push(monic);
                        factors.len() - 1
                    })
                })
                .collect()
        })
        .collect();
    (factors, index_groups)
}

/// Each factor group as a bitmask over the factor universe, deduplicated.
fn group_masks(index_groups: &[Vec<usize>]) -> Vec<u64> {
    let mut masks: Vec<u64> = index_groups
        .iter()
        .map(|ids| ids.iter().fold(0u64, |mask, &i| mask | (1u64 << i)))
        .collect();
    masks.sort_unstable();
    masks.dedup();
    masks
}

/// Keep only the inclusion-minimal masks (an antichain), discarding any mask
/// that is a superset of another; duplicates collapse since a mask is a
/// (non-strict) superset of its equal.
fn keep_minimal(mut masks: Vec<u64>) -> Vec<u64> {
    masks.sort_by_key(|m| m.count_ones());
    masks.into_iter().fold(Vec::new(), |mut minimal, cand| {
        if minimal.iter().all(|&m| m & cand != m) {
            minimal.push(cand);
        }
        minimal
    })
}

/// Minimal factor covers as bitmasks over the factor universe: the sets of
/// factors that hit every polynomial and from which no proper subset still hits
/// every polynomial (the minimal transversals of the group hypergraph). Computed
/// by Berge multiplication --- fold the groups in, extending each partial
/// transversal that misses a group by every factor of that group, then prune to
/// the minimal elements. An empty group yields no cover.
///
/// Returns `None` when the distinct-factor count exceeds the 63-bit mask limit.
fn minimal_covers(index_groups: &[Vec<usize>], n_factors: usize) -> Option<Vec<u64>> {
    if n_factors >= 64 {
        return None;
    }
    let covers = group_masks(index_groups)
        .into_iter()
        .fold(vec![0u64], |transversals, group| {
            let expanded = transversals
                .into_iter()
                .flat_map(|t| {
                    if t & group != 0 {
                        vec![t]
                    } else {
                        (0..n_factors)
                            .filter(|&v| group & (1u64 << v) != 0)
                            .map(|v| t | (1u64 << v))
                            .collect()
                    }
                })
                .collect();
            keep_minimal(expanded)
        });
    Some(covers)
}

/// Total degree of a polynomial: the largest monomial exponent sum.
fn total_degree(p: &PolyExtLex) -> u64 {
    p.exponents_iter()
        .map(|term| term.iter().map(|&e| e as u64).sum::<u64>())
        .max()
        .unwrap_or(0)
}

/// Single-bit masks for every set bit of `mask`, least significant first.
fn set_bits(mask: u64) -> impl Iterator<Item = u64> {
    (0..64).map(|i| 1u64 << i).filter(move |b| mask & b != 0)
}

/// Bézout bound of a cover: the product of the degrees of its factors. Used as
/// the cost heuristic that orders covers from easiest to hardest to check. The
/// product saturates rather than overflowing, which only matters for covers far
/// beyond what is solvable anyway.
fn bezout_bound(degrees: &[u64], mask: u64) -> u128 {
    set_bits(mask).fold(1u128, |acc, b| {
        acc.saturating_mul(degrees[b.trailing_zeros() as usize] as u128)
    })
}

/// Decode a cover bitmask back into its factors.
fn decode_cover(factors: &[PolyExtLex], mask: u64) -> Vec<PolyExtLex> {
    (0..factors.len())
        .filter(|&i| mask & (1u64 << i) != 0)
        .map(|i| factors[i].clone())
        .collect()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct WorkItem {
    cover_mask: u64,
    zero_mask: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Core {
    cover_mask: u64,
    zero_mask: u64,
}

fn expand_with_sparse_subsets(covers: &[u64], sparse_count: usize) -> Vec<WorkItem> {
    let sparse_mask = if sparse_count == 0 {
        0
    } else {
        (1u64 << sparse_count) - 1
    };
    covers
        .iter()
        .flat_map(|&cover_mask| {
            submasks(sparse_mask).map(move |zero_mask| WorkItem {
                cover_mask,
                zero_mask,
            })
        })
        .collect()
}

/// Every submask of `mask`, including the empty mask and `mask` itself. Used to
/// extend each cover by every subset of the sparsity factors. For `mask == 0`
/// this yields the single empty mask, so a run without `--sparse` leaves the
/// cover list unchanged.
fn submasks(mask: u64) -> impl Iterator<Item = u64> {
    let mut next = Some(mask);
    std::iter::from_fn(move || {
        let cur = next?;
        next = (cur != 0).then(|| (cur - 1) & mask);
        Some(cur)
    })
}

/// Resolve the requested `--sparse` candidate variable names to their symbols.
/// An empty list means "all variables" (bare `--sparse`); a non-empty list is
/// resolved against the loader's names, with an unknown name an error.
fn resolve_candidates(
    loaded: &LoadedMathematicaSystem,
    names: &[String],
) -> Result<Vec<(String, Symbol)>, String> {
    if names.is_empty() {
        return Ok(loaded
            .var_names
            .iter()
            .cloned()
            .zip(loaded.id_to_sym.iter().copied())
            .collect());
    }
    names
        .iter()
        .map(|name| {
            loaded
                .var_names
                .iter()
                .position(|n| n == name)
                .map(|i| (name.clone(), loaded.id_to_sym[i]))
                .ok_or_else(|| format!("unknown variable {name:?} for --sparse"))
        })
        .collect()
}

/// Resolve sparse candidates to variable positions in `template`'s variable map.
/// A variable absent from the map imposes no substitution and is skipped.
fn usable_sparse_variables(
    template: &PolyExtLex,
    candidates: &[(String, Symbol)],
) -> Vec<(String, usize)> {
    candidates
        .iter()
        .filter_map(|(name, sym)| {
            template
                .variables
                .iter()
                .position(|var| *var == PolyVariable::from(*sym))
                .map(|position| (name.clone(), position))
        })
        .collect()
}

fn zero_positions<'a>(
    sparse_variables: &'a [(String, usize)],
    zero_mask: u64,
) -> impl Iterator<Item = usize> + 'a {
    sparse_variables
        .iter()
        .enumerate()
        .filter(move |(i, _)| zero_mask & (1u64 << i) != 0)
        .map(|(_, (_, position))| *position)
}

fn substitute_zeroes(poly: &PolyExtLex, zero_positions: &[usize]) -> PolyExtLex {
    if zero_positions.is_empty() || poly.is_zero() {
        return poly.clone();
    }
    let mut out = poly.zero_with_capacity(poly.nterms());
    for i in 0..poly.nterms() {
        let exponents = poly.exponents(i);
        if zero_positions
            .iter()
            .all(|&position| exponents[position] == 0)
        {
            out.append_monomial_back(poly.coefficients[i].clone(), exponents);
        }
    }
    out
}

fn decode_work_item(
    factors: &[PolyExtLex],
    sparse_variables: &[(String, usize)],
    item: WorkItem,
) -> Vec<PolyExtLex> {
    let zero_positions: Vec<_> = zero_positions(sparse_variables, item.zero_mask).collect();
    decode_cover(factors, item.cover_mask)
        .into_iter()
        .map(|factor| substitute_zeroes(&factor, &zero_positions))
        .filter(|factor| !factor.is_zero())
        .collect()
}

/// A reduced degrevlex Gröbner basis equal to `{1}` (a non-zero constant)
/// certifies that the cover's ideal is the whole ring, i.e. the cover is
/// inconsistent.
fn is_unit_ideal<O: MonomialOrder>(system: &[MultivariatePolynomial<ExtQ, u16, O>]) -> bool {
    system.iter().any(|p| !p.is_zero() && p.is_constant())
}

/// Outcome of computing the Gröbner basis of a single cover.
enum Outcome {
    /// Non-unit reduced basis with this many elements: the cover is consistent.
    Consistent(usize),
    /// Reduced basis is `{1}`: the cover's ideal is the whole ring.
    Inconsistent,
    /// Computation exceeded the per-cover time budget.
    Timeout,
}

/// Compute the cover's Gröbner basis on a worker thread, abandoning it if it
/// outlives `timeout`. The cover is reordered to degrevlex before the
/// computation, which is typically far cheaper than lex for deciding
/// satisfiability (Section 3.1). Cancellation is cooperative-free, so a
/// timed-out thread keeps running until it finishes on its own.
fn evaluate_cover(cover: Vec<PolyExtLex>, timeout: Duration) -> Outcome {
    if cover.is_empty() {
        return Outcome::Consistent(0);
    }
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let grevlex: Vec<_> = cover.iter().map(|p| p.reorder::<GrevLexOrder>()).collect();
        let _ = tx.send(GroebnerBasis::new(&grevlex, false).system);
    });
    match rx.recv_timeout(timeout) {
        Ok(system) if is_unit_ideal(&system) => Outcome::Inconsistent,
        Ok(system) => Outcome::Consistent(system.len()),
        Err(_) => Outcome::Timeout,
    }
}

/// Shrink an inconsistent work item to a minimal unsatisfiable core by deletion:
/// drop each factor or zero substitution whose removal leaves the ideal equal to
/// the whole ring. A removal that turns the item satisfiable keeps the bit; a
/// removal that times out also keeps it, conservatively, since the larger item is
/// still known inconsistent. The result is deletion-minimal over both masks.
fn minimal_unsat_core(
    factors: &[PolyExtLex],
    sparse_variables: &[(String, usize)],
    item: WorkItem,
    timeout: Duration,
) -> Core {
    let mut core = item;
    loop {
        let mut changed = false;

        for bit in set_bits(core.cover_mask).collect::<Vec<_>>() {
            let reduced = WorkItem {
                cover_mask: core.cover_mask & !bit,
                zero_mask: core.zero_mask,
            };
            if matches!(
                evaluate_cover(
                    decode_work_item(factors, sparse_variables, reduced),
                    timeout
                ),
                Outcome::Inconsistent
            ) {
                core = reduced;
                changed = true;
            }
        }

        for bit in set_bits(core.zero_mask).collect::<Vec<_>>() {
            let reduced = WorkItem {
                cover_mask: core.cover_mask,
                zero_mask: core.zero_mask & !bit,
            };
            if matches!(
                evaluate_cover(
                    decode_work_item(factors, sparse_variables, reduced),
                    timeout
                ),
                Outcome::Inconsistent
            ) {
                core = reduced;
                changed = true;
            }
        }

        if !changed {
            break;
        }
    }

    Core {
        cover_mask: core.cover_mask,
        zero_mask: core.zero_mask,
    }
}

fn core_subsumes(core: Core, item: WorkItem) -> bool {
    core.cover_mask & item.cover_mask == core.cover_mask
        && core.zero_mask & item.zero_mask == core.zero_mask
}

fn sort_work_items(items: &mut [WorkItem], degrees: &[u64]) {
    items.sort_by_key(|item| {
        (
            bezout_bound(degrees, item.cover_mask),
            item.cover_mask,
            Reverse(item.zero_mask.count_ones()),
            item.zero_mask,
        )
    })
}

#[derive(Debug)]
struct CliOptions {
    path: PathBuf,
    interreduce: bool,
    max_covers: usize,
    timeout_ms: u64,
    sparse: bool,
    sparse_names: Vec<String>,
}

fn parse_args(args: impl IntoIterator<Item = String>) -> Result<CliOptions, String> {
    let mut path: Option<PathBuf> = None;
    let mut interreduce = true;
    let mut max_covers: usize = 2_000;
    let mut timeout_ms: u64 = 5_000;
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
            "--timeout-ms" => {
                timeout_ms = args
                    .next()
                    .ok_or_else(|| "--timeout-ms requires a value".to_string())?
                    .parse()
                    .map_err(|e| format!("invalid --timeout-ms: {e}"))?;
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
        sparse,
        sparse_names,
    })
}

fn main() {
    let key = "5568d092#6bb8b4fb#ff936916-65ba-552f-b3c1-bdc64597cd21";
    LicenseManager::set_license_key(key).unwrap();

    let options = match parse_args(std::env::args().skip(1)) {
        Ok(options) => options,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    };

    println!("polynomial-cdcl — CDCL over minimal factor covers\n");
    println!("Loading {}", options.path.display());

    let loaded = match load_mathematica_system_from_path_with_var_names(&options.path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("load error: {e}");
            std::process::exit(1);
        }
    };
    println!(
        "Parsed: {} equations, {} variables (loader order)",
        loaded.system.equations.len(),
        loaded.var_names.len()
    );

    let field = AlgebraicExtension::new_complex(Q);

    println!(
        "\nPreprocessing ({}factor)",
        if options.interreduce {
            "interreduce + "
        } else {
            ""
        }
    );
    let t_pre = Instant::now();
    let pre = match preprocess(&loaded, &field, options.interreduce) {
        Ok(pre) => pre,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };
    println!(
        "  {} non-trivial polynomial(s), elapsed {:?}",
        pre.len(),
        t_pre.elapsed()
    );
    if pre.is_empty() {
        println!("Nothing to cover — all polynomials reduced to zero.");
        return;
    }

    let (factors, index_groups) = build_factor_index(&pre);
    let base_covers = match minimal_covers(&index_groups, factors.len()) {
        Some(covers) => covers,
        None => {
            eprintln!("too many distinct factors (>= 64) for the bitmask enumerator");
            std::process::exit(1);
        }
    };
    println!(
        "  {} minimal factor cover(s) (hitting sets, no variable zeroing)",
        base_covers.len()
    );
    if base_covers.is_empty() {
        println!("No factor cover exists — some equation has no factors.");
        return;
    }

    let sparse_variables = if options.sparse {
        let candidates = match resolve_candidates(&loaded, &options.sparse_names) {
            Ok(candidates) => candidates,
            Err(e) => {
                eprintln!("{e}");
                std::process::exit(2);
            }
        };
        let sparse_variables = usable_sparse_variables(&pre[0][0], &candidates);
        if sparse_variables.len() >= 64 {
            eprintln!("too many usable sparse variables (>= 64) for the bitmask enumerator");
            std::process::exit(1);
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
    } else {
        println!("  sparsity disabled");
        Vec::new()
    };

    let mut covers = expand_with_sparse_subsets(&base_covers, sparse_variables.len());
    let degrees: Vec<u64> = factors.iter().map(total_degree).collect();
    if options.sparse {
        println!(
            "  {} cover variant(s) after optional sparsity subsets",
            covers.len()
        );
    }

    sort_work_items(&mut covers, &degrees);

    let tried = covers.len().min(options.max_covers);
    println!(
        "\nCDCL over {tried} of {} minimal cover(s) in Bézout order, {}ms/cover timeout",
        covers.len(),
        options.timeout_ms,
    );

    let timeout = Duration::from_millis(options.timeout_ms);
    let t_gb = Instant::now();
    let mut cores: Vec<Core> = Vec::new();
    let mut solutions: Vec<(usize, usize)> = Vec::new();
    let mut inconsistent = 0usize;
    let mut timeouts = 0usize;
    let mut pruned = 0usize;
    for (idx, &item) in covers.iter().take(tried).enumerate() {
        // Skip the cover if any learned core is a subset of it (not an
        // equality membership test, despite `manual_contains`).
        #[allow(clippy::manual_contains)]
        let subsumed = cores.iter().any(|&core| core_subsumes(core, item));
        if subsumed {
            pruned += 1;
            continue;
        }
        let t_cover = Instant::now();
        match evaluate_cover(decode_work_item(&factors, &sparse_variables, item), timeout) {
            Outcome::Consistent(len) => {
                println!(
                    "  cover #{idx}: satisfiable ({len} basis element(s), elapsed {:?})",
                    t_cover.elapsed()
                );
                solutions.push((idx, len));
            }
            Outcome::Inconsistent => {
                println!(
                    "  cover #{idx}: unsatisfiable (elapsed {:?})",
                    t_cover.elapsed()
                );
                cores.push(minimal_unsat_core(
                    &factors,
                    &sparse_variables,
                    item,
                    timeout,
                ));
                inconsistent += 1;
            }
            Outcome::Timeout => {
                println!(
                    "  cover #{idx}: timed out (elapsed {:?})",
                    t_cover.elapsed()
                );
                timeouts += 1;
            }
        }
    }

    println!(
        "  {} satisfiable, {} unsatisfiable, {} core-pruned, {} timed out, elapsed {:?}",
        solutions.len(),
        inconsistent,
        pruned,
        timeouts,
        t_gb.elapsed()
    );
    for (idx, len) in solutions.iter().take(20) {
        println!("\tcover #{idx}: {len} basis element(s)");
    }
    if !cores.is_empty() {
        println!("  learned {} minimal unsatisfiable core(s)", cores.len());
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::Once;
    use std::time::Duration;

    use polynomial_cdcl::{
        PolyExtLex, ideal_ext_lex, load_mathematica_system_from_str_with_var_names,
        parse_expression, variable_order,
    };
    use symbolica::LicenseManager;
    use symbolica::atom::AtomCore;
    use symbolica::domains::algebraic_number::AlgebraicExtension;
    use symbolica::domains::rational::Q;

    use super::{
        Core, WorkItem, core_subsumes, expand_with_sparse_subsets, keep_minimal, minimal_covers,
        minimal_unsat_core, parse_args, sort_work_items, substitute_zeroes,
    };

    static LICENSE: Once = Once::new();

    fn activate_license() {
        LICENSE.call_once(|| {
            let _ = LicenseManager::set_license_key(
                "5568d092#6bb8b4fb#ff936916-65ba-552f-b3c1-bdc64597cd21",
            );
        });
    }

    fn test_poly(source: &str) -> PolyExtLex {
        activate_license();
        let loaded = load_mathematica_system_from_str_with_var_names(&format!("{{{source} == 0}}"))
            .expect("load");
        let field = AlgebraicExtension::new_complex(Q);
        let order = variable_order(&loaded.id_to_sym);
        ideal_ext_lex(&loaded.system, &field, order)
            .expect("polynomial")
            .remove(0)
    }

    fn poly_like(template: &PolyExtLex, source: &str) -> PolyExtLex {
        parse_expression(source.to_string())
            .expect("parse")
            .try_to_polynomial::<_, u16>(&template.ring, Some(template.variables.clone()))
            .expect("polynomial")
    }

    /// Collect a cover list into a sorted vec of sorted factor-index vectors.
    fn as_sets(masks: Vec<u64>) -> Vec<Vec<usize>> {
        let mut sets: Vec<Vec<usize>> = masks
            .into_iter()
            .map(|m| (0..64).filter(|&i| m & (1u64 << i) != 0).collect())
            .collect();
        sets.sort();
        sets
    }

    #[test]
    fn minimal_covers_disjoint_groups() {
        // Groups {0,1} and {2}: every cover must take 2 and one of {0,1}.
        let covers = minimal_covers(&[vec![0, 1], vec![2]], 3).unwrap();
        assert_eq!(as_sets(covers), vec![vec![0, 2], vec![1, 2]]);
    }

    #[test]
    fn minimal_covers_shared_factor() {
        // Groups {0,1} and {1,2} share factor 1; minimal transversals are
        // {1} (hits both) and {0,2}; {0,1} is dropped as a superset of {1}.
        let covers = minimal_covers(&[vec![0, 1], vec![1, 2]], 3).unwrap();
        assert_eq!(as_sets(covers), vec![vec![0, 2], vec![1]]);
    }

    #[test]
    fn minimal_covers_empty_group_has_none() {
        // A group with no factors can never be hit ⇒ no cover exists.
        assert_eq!(
            minimal_covers(&[vec![0], vec![]], 1).unwrap(),
            Vec::<u64>::new()
        );
    }

    #[test]
    fn keep_minimal_drops_supersets_and_dups() {
        // {0} subsumes {0,1}; the duplicate {2} collapses.
        let kept = keep_minimal(vec![0b001, 0b011, 0b100, 0b100]);
        assert_eq!(kept, vec![0b001, 0b100]);
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
            [
                "--no-interreduce",
                "--max-covers",
                "17",
                "--timeout-ms",
                "250",
            ]
            .into_iter()
            .map(String::from),
        )
        .unwrap();
        assert!(!options.interreduce);
        assert_eq!(options.max_covers, 17);
        assert_eq!(options.timeout_ms, 250);
        assert!(!options.sparse);
    }

    #[test]
    fn parse_args_unknown_flag_fails() {
        assert!(parse_args(["--missing"].into_iter().map(String::from)).is_err());
    }

    #[test]
    fn sparse_expansion_no_mask_keeps_base_covers() {
        assert_eq!(
            expand_with_sparse_subsets(&[0b001, 0b010], 0),
            vec![
                WorkItem {
                    cover_mask: 0b001,
                    zero_mask: 0
                },
                WorkItem {
                    cover_mask: 0b010,
                    zero_mask: 0
                }
            ]
        );
    }

    #[test]
    fn sparse_expansion_preserves_zero_subsets() {
        let expanded = expand_with_sparse_subsets(&[0b001], 2);
        assert_eq!(
            expanded,
            vec![
                WorkItem {
                    cover_mask: 0b001,
                    zero_mask: 0b011
                },
                WorkItem {
                    cover_mask: 0b001,
                    zero_mask: 0b010
                },
                WorkItem {
                    cover_mask: 0b001,
                    zero_mask: 0b001
                },
                WorkItem {
                    cover_mask: 0b001,
                    zero_mask: 0
                }
            ]
        );
    }

    #[test]
    fn work_item_order_checks_more_zeroes_first_for_same_cover() {
        let degrees = vec![2, 3];
        let mut items = expand_with_sparse_subsets(&[0b001], 2);

        sort_work_items(&mut items, &degrees);

        assert_eq!(
            items,
            vec![
                WorkItem {
                    cover_mask: 0b001,
                    zero_mask: 0b011
                },
                WorkItem {
                    cover_mask: 0b001,
                    zero_mask: 0b001
                },
                WorkItem {
                    cover_mask: 0b001,
                    zero_mask: 0b010
                },
                WorkItem {
                    cover_mask: 0b001,
                    zero_mask: 0
                }
            ]
        );
    }

    #[test]
    fn core_subsumes_zero_supersets() {
        let core = Core {
            cover_mask: 0b001,
            zero_mask: 0b001,
        };

        assert!(core_subsumes(
            core,
            WorkItem {
                cover_mask: 0b101,
                zero_mask: 0b011
            }
        ));
    }

    #[test]
    fn core_subsumes_requires_cover_and_zero_bits() {
        let core = Core {
            cover_mask: 0b101,
            zero_mask: 0b011,
        };

        assert!(!core_subsumes(
            core,
            WorkItem {
                cover_mask: 0b001,
                zero_mask: 0b011
            }
        ));
        assert!(!core_subsumes(
            core,
            WorkItem {
                cover_mask: 0b101,
                zero_mask: 0b001
            }
        ));
    }

    #[test]
    fn minimal_unsat_core_retains_required_zero_substitution() {
        let factor = test_poly("x + 1");
        let factors = vec![factor];
        let sparse_variables = vec![("x".to_string(), 0)];

        let core = minimal_unsat_core(
            &factors,
            &sparse_variables,
            WorkItem {
                cover_mask: 0b1,
                zero_mask: 0b1,
            },
            Duration::from_secs(5),
        );

        assert_eq!(
            core,
            Core {
                cover_mask: 0b1,
                zero_mask: 0b1
            }
        );
    }

    #[test]
    fn minimal_unsat_core_drops_unneeded_zero_substitution() {
        let template = test_poly("x + y + 1");
        let factors = vec![poly_like(&template, "x + 1")];
        let sparse_variables = vec![("x".to_string(), 0), ("y".to_string(), 1)];

        let core = minimal_unsat_core(
            &factors,
            &sparse_variables,
            WorkItem {
                cover_mask: 0b1,
                zero_mask: 0b11,
            },
            Duration::from_secs(5),
        );

        assert_eq!(
            core,
            Core {
                cover_mask: 0b1,
                zero_mask: 0b01
            }
        );
    }

    #[test]
    fn substitute_zeroes_drops_terms_containing_zeroed_variables() {
        let poly = test_poly("x*y + z");

        let substituted = substitute_zeroes(&poly, &[0]);

        assert_eq!(substituted, poly_like(&poly, "z"));
        assert!(substitute_zeroes(&test_poly("x*y"), &[0]).is_zero());
        let unaffected = poly_like(&poly, "z + 1");
        assert_eq!(substitute_zeroes(&unaffected, &[0]), unaffected);
    }
}
