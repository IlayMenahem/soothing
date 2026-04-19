//! Load a Mathematica-exported system and run DFS or beam linear elimination.
//!
//! Uses [`soothing_linear::dfs_eliminate`] or [`soothing_linear::beam_eliminate`] with
//! [`soothing_linear::is_solution_nonzero_kappah`] (κ≠0 in expanded substitution), aligned with
//! `python/scripts/solving_symbolic.py`.
//!
//! - **DFS (default):** `cargo run -p soothing-linear --bin solve`
//! - **Beam:** `cargo run -p soothing-linear --bin solve -- --beam 8`
//! - Optional path: first positional argument (defaults to `systems/system.txt` via manifest).

use std::fs::File;
use std::path::{Path, PathBuf};
use std::time::Instant;

use soothing_linear::{
    EliminationResult, SampleConfig, VarId, beam_eliminate, default_systems_dir_from_manifest,
    dfs_eliminate, is_solution_nonzero_kappah, load_mathematica_system_from_path_with_var_names,
    print_substitution_bindings,
};

/// Trace output path relative to the current working directory.
const TRACE_LOG_FILE: &str = "elimination_trace_dag.log";

fn parse_cli() -> (PathBuf, Option<usize>) {
    let mut path: Option<PathBuf> = None;
    let mut beam: Option<usize> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        if a == "--beam" {
            let w = args.next().unwrap_or_else(|| {
                eprintln!("error: --beam requires a width (e.g. --beam 8)");
                std::process::exit(1);
            });
            match w.parse::<usize>() {
                Ok(n) => beam = Some(n),
                Err(_) => {
                    eprintln!("error: --beam width must be a non-negative integer, got {w:?}");
                    std::process::exit(1);
                }
            }
        } else if a.starts_with('-') {
            eprintln!("error: unknown flag {a}");
            std::process::exit(1);
        } else if path.is_some() {
            eprintln!("error: unexpected extra argument {a}");
            std::process::exit(1);
        } else {
            path = Some(PathBuf::from(a));
        }
    }
    let path = path.unwrap_or_else(|| default_systems_dir_from_manifest().join("simplified_mathematica_system.txt"));
    (path, beam)
}

fn main() {
    let (path, beam_opt) = parse_cli();
    let use_beam = beam_opt.is_some_and(|w| w >= 1);
    let beam_width = beam_opt.filter(|&w| w >= 1).unwrap_or(1);

    let trace_path = Path::new(TRACE_LOG_FILE);

    let title = if use_beam {
        format!("soothing-linear — beam elimination (width {beam_width}) on Mathematica system\n")
    } else {
        "soothing-linear — DFS elimination on Mathematica system\n".to_string()
    };
    print!("{title}");
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
    let is_sol = |r: &EliminationResult| is_solution_nonzero_kappah(r, kappah, &cfg);

    let res = if use_beam {
        match beam_eliminate(
            &loaded.system,
            &cfg,
            beam_width,
            Some(&mut trace_file),
            is_sol,
        ) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("beam_eliminate error: {e}");
                std::process::exit(1);
            }
        }
    } else {
        match dfs_eliminate(&loaded.system, &cfg, Some(&mut trace_file), is_sol) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("dfs_eliminate error: {e}");
                std::process::exit(1);
            }
        }
    };
    let elapsed = t0.elapsed();

    if use_beam {
        println!(
            "Beam linear elimination (width {}): {} substitution(s), {} remaining equation(s), elapsed {:.2?}",
            beam_width,
            res.substitution.len(),
            res.reduced_system.equations.len(),
            elapsed
        );
    } else {
        println!(
            "DFS linear elimination: {} substitution(s), {} remaining equation(s), elapsed {:.2?}",
            res.substitution.len(),
            res.reduced_system.equations.len(),
            elapsed
        );
    }
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
