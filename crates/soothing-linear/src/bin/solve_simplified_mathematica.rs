//! Load `systems/simplified_mathematica_system.txt` and run DFS linear elimination.
//!
//! From repo root: `cargo run -p soothing-linear --bin solve_simplified_mathematica`
//! Optional: pass a path to another Mathematica rule list.

use std::path::PathBuf;
use std::time::Instant;

use soothing_linear::{
    default_systems_dir_from_manifest, dfs_linear_elimination,
    load_mathematica_system_from_path_with_var_names, print_substitution_bindings, LinearEliminationOptions,
    SampleConfig, VarId,
};

fn main() {
    let path: PathBuf = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            default_systems_dir_from_manifest().join("simplified_mathematica_system.txt")
        });

    println!("soothing-linear — DFS elimination on Mathematica system\n");
    println!("Loading {}", path.display());
    let loaded = match load_mathematica_system_from_path_with_var_names(&path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("load error: {e}");
            std::process::exit(1);
        }
    };
    let sys = loaded.system;
    let var_names = &loaded.var_names;

    let n_eq = sys.equations.len();
    let all_vars: std::collections::BTreeSet<_> = sys
        .equations
        .iter()
        .flat_map(|&r| soothing_linear::Dag::vars_in_root(&sys.dag, r))
        .collect();
    println!(
        "Parsed: {} equations, {} distinct variables across system",
        n_eq,
        all_vars.len()
    );

    let cfg = SampleConfig::default();
    let t0 = Instant::now();
    let res = match dfs_linear_elimination(
        sys,
        &cfg,
        LinearEliminationOptions {
            simplify_each_step: true,
        },
    ) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("dfs_linear_elimination error: {e}");
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
