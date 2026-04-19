//! Load `systems/simplified_mathematica_system.txt` and run DFS linear elimination.
//!
//! Uses [`soothing_linear::dfs_eliminate`] with [`soothing_linear::is_solution_nonzero_kappah`]
//! (κ≠0 in expanded substitution), aligned with `python/scripts/solving_symbolic.py`.
//!
//! From repo root: `cargo run -p soothing-linear --bin solve_simplified_mathematica`

use std::fs::File;
use std::path::{Path, PathBuf};
use std::time::Instant;

use soothing_linear::{
    SampleConfig, VarId, default_systems_dir_from_manifest, dfs_eliminate,
    is_solution_nonzero_kappah, load_mathematica_system_from_path_with_var_names,
    print_substitution_bindings,
};

/// Trace output path relative to the current working directory.
const TRACE_LOG_FILE: &str = "elimination_trace_dag.log";

fn main() {
    let path: PathBuf = std::env::args().nth(1).map_or_else(
        || default_systems_dir_from_manifest().join("system.txt"),
        PathBuf::from,
    );

    let trace_path = Path::new(TRACE_LOG_FILE);

    println!("soothing-linear — DFS elimination on Mathematica system\n");
    println!("Loading {}", path.display());
    println!("Tracing elimination to {}", trace_path.display());
    let loaded = match load_mathematica_system_from_path_with_var_names(&path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("load error: {e}");
            std::process::exit(1);
        }
    };

    let n_eq = loaded.system.equations.len();
    let all_vars: std::collections::BTreeSet<_> = loaded
        .system
        .equations
        .iter()
        .flat_map(|&r| soothing_linear::Dag::vars_in_root(&loaded.system.dag, r))
        .collect();
    let var_names = loaded.var_names.clone();
    println!(
        "Parsed: {} equations, {} distinct variables across system",
        n_eq,
        all_vars.len()
    );

    let cfg = SampleConfig::default();

    let kappah = match loaded.var_names.iter().position(|n| n == "kappah") {
        Some(i) => VarId(u32::try_from(i).unwrap()),
        None => {
            eprintln!("missing variable \"kappah\" in loaded system");
            std::process::exit(1);
        }
    };

    let mut trace_file = match File::create(trace_path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("trace log {}: {e}", trace_path.display());
            std::process::exit(1);
        }
    };

    let t0 = Instant::now();
    let res = match dfs_eliminate(&loaded.system, &cfg, Some(&mut trace_file), |r| {
        is_solution_nonzero_kappah(r, kappah, &cfg)
    }) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("dfs_eliminate error: {e}");
            std::process::exit(1);
        }
    };
    let elapsed = t0.elapsed();

    println!(
        "DFS linear elimination: {} substitution(s), {} remaining equation(s), elapsed {:.2?}",
        res.substitution.len(),
        res.reduced_system.equations.len(),
        elapsed
    );
    if !res.reduced_system.equations.is_empty() {
        println!("Note: residual is non-empty (expected for highly nonlinear systems).");
    }

    println!("Substitutions:");
    let vn = |v: VarId| {
        var_names
            .get(v.0 as usize)
            .cloned()
            .unwrap_or_else(|| format!("v{}", v.0))
    };
    print_substitution_bindings(&res.substitution, &vn);

    println!("\nDone.");
}
