//! A groebner basis of a radical cover of a polynomial system.
//!
//! Usage: `idos [path]` — default path: `systems/system.txt`.

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

/// Number of distinct radical covers: one factor chosen per equation group.
fn cover_count(groups: &[Vec<PolyExtLex>]) -> u128 {
    groups.iter().map(|g| g.len() as u128).product()
}

/// The `idx`-th radical cover, decoding `idx` as a mixed-radix index over the
/// factor groups (radix per digit = number of factors in that group).
fn cover_at(groups: &[Vec<PolyExtLex>], idx: u128) -> Vec<PolyExtLex> {
    groups
        .iter()
        .scan(idx, |rem, factors| {
            let k = (*rem % factors.len() as u128) as usize;
            *rem /= factors.len() as u128;
            Some(factors[k].clone())
        })
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

/// Outcome of computing the Gröbner basis of a single radical cover.
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
fn evaluate_cover(groups: &[Vec<PolyExtLex>], idx: u128, timeout: Duration) -> Outcome {
    let cover = cover_at(groups, idx);
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

    println!("polynomial-milp — MILP radical cover (good_lp / HiGHS)\n");
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

    let total = cover_count(&pre);
    let tried = total.min(options.max_covers);
    println!(
        "\n{total} radical cover(s) total; trying {tried}{} on {} threads, {}ms/cover timeout",
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
    let outcomes: Vec<(u128, Outcome)> = cover_indices(total, options.max_covers, options.seed)
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|idx| (idx, evaluate_cover(&pre, idx, timeout)))
        .collect();

    let consistent: Vec<(u128, usize)> = outcomes
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
