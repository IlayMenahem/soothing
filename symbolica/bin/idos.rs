//! A Gröbner basis of a minimal factor cover of a polynomial system.
//!
//! Usage: `idos [path]` — default path: `systems/system.txt`.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use rayon::prelude::*;
use symbolica::LicenseManager;
use symbolica::domains::algebraic_number::AlgebraicExtension;
use symbolica::domains::rational::Q;
use symbolica::poly::groebner::GroebnerBasis;

use polynomial_milp::{
    PolyExtLex, default_systems_dir_from_manifest,
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

/// Decode a cover bitmask back into its factors.
fn decode_cover(factors: &[PolyExtLex], mask: u64) -> Vec<PolyExtLex> {
    (0..factors.len())
        .filter(|&i| mask & (1u64 << i) != 0)
        .map(|i| factors[i].clone())
        .collect()
}

/// A reduced lex Gröbner basis equal to `{1}` (a non-zero constant) certifies
/// that the cover's ideal is the whole ring, i.e. the cover is inconsistent.
fn is_unit_ideal(system: &[PolyExtLex]) -> bool {
    system.iter().any(|p| !p.is_zero() && p.is_constant())
}

/// Cover indices to evaluate: the full range when it fits the budget, otherwise
/// `budget` pseudo-random samples (splitmix64) drawn from `0..total`.
fn cover_indices(total: u128, budget: u128, seed: u64) -> Box<dyn Iterator<Item = u128>> {
    if total <= budget {
        return Box::new(0..total);
    }
    let mut state = seed;
    Box::new((0..budget).map(move |_| {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z as u128) % total
    }))
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
/// outlives `timeout`. Cancellation is cooperative-free, so a timed-out thread
/// keeps running until it finishes on its own.
fn evaluate_cover(cover: Vec<PolyExtLex>, timeout: Duration) -> Outcome {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let _ = tx.send(GroebnerBasis::new(&cover, false).system);
    });
    match rx.recv_timeout(timeout) {
        Ok(system) if is_unit_ideal(&system) => Outcome::Inconsistent,
        Ok(system) => Outcome::Consistent(system.len()),
        Err(_) => Outcome::Timeout,
    }
}

#[derive(Debug)]
struct CliOptions {
    path: PathBuf,
    interreduce: bool,
    max_covers: u128,
    seed: u64,
    timeout_ms: u64,
}

fn parse_args(args: impl IntoIterator<Item = String>) -> Result<CliOptions, String> {
    let mut path: Option<PathBuf> = None;
    let mut interreduce = true;
    let mut max_covers: u128 = 2_000;
    let mut seed: u64 = 0x1234_5678;
    let mut timeout_ms: u64 = 5_000;
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        let mut next_value = |label: &str| {
            args.next()
                .ok_or_else(|| format!("{label} requires a value"))
        };
        match arg.as_str() {
            "--no-interreduce" => interreduce = false,
            "--max-covers" => {
                max_covers = next_value("--max-covers")?
                    .parse()
                    .map_err(|e| format!("invalid --max-covers: {e}"))?;
            }
            "--seed" => {
                seed = next_value("--seed")?
                    .parse()
                    .map_err(|e| format!("invalid --seed: {e}"))?;
            }
            "--timeout-ms" => {
                timeout_ms = next_value("--timeout-ms")?
                    .parse()
                    .map_err(|e| format!("invalid --timeout-ms: {e}"))?;
            }
            _ if arg.starts_with("--") => return Err(format!("unknown option {arg:?}")),
            _ => path = Some(PathBuf::from(arg)),
        }
    }
    Ok(CliOptions {
        path: path.unwrap_or_else(|| default_systems_dir_from_manifest().join("system.txt")),
        interreduce,
        max_covers,
        seed,
        timeout_ms,
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

    println!("polynomial-milp — minimal factor covers\n");
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
    let covers = match minimal_covers(&index_groups, factors.len()) {
        Some(covers) => covers,
        None => {
            eprintln!("too many distinct factors (>= 64) for the bitmask enumerator");
            std::process::exit(1);
        }
    };
    println!(
        "  {} minimal factor cover(s) (hitting sets, no variable zeroing)",
        covers.len()
    );
    if covers.is_empty() {
        println!("No factor cover exists — some equation has no factors.");
        return;
    }

    let total = covers.len() as u128;
    let tried = total.min(options.max_covers);
    println!(
        "\ntrying {tried} of {total} minimal cover(s){} on {} threads, {}ms/cover timeout",
        if total > options.max_covers {
            " (random sample)"
        } else {
            ""
        },
        rayon::current_num_threads(),
        options.timeout_ms,
    );

    let timeout = Duration::from_millis(options.timeout_ms);
    let t_gb = Instant::now();
    let outcomes: Vec<(usize, Outcome)> = cover_indices(total, options.max_covers, options.seed)
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|idx| {
            let cover = decode_cover(&factors, covers[idx as usize]);
            (idx as usize, evaluate_cover(cover, timeout))
        })
        .collect();

    let consistent: Vec<(usize, usize)> = outcomes
        .iter()
        .filter_map(|(idx, o)| match o {
            Outcome::Consistent(len) => Some((*idx, *len)),
            _ => None,
        })
        .collect();
    let timeouts = outcomes
        .iter()
        .filter(|(_, o)| matches!(o, Outcome::Timeout))
        .count();

    println!(
        "  {} consistent, {} inconsistent, {} timed out, elapsed {:?}",
        consistent.len(),
        outcomes.len() - consistent.len() - timeouts,
        timeouts,
        t_gb.elapsed()
    );
    for (idx, len) in consistent.iter().take(20) {
        println!("\tcover #{idx}: {len} basis element(s)");
    }
}

#[cfg(test)]
mod tests {
    use super::{keep_minimal, minimal_covers};

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
}
