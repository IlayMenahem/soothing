//! Load `systems/simplified_mathematica_system.txt` and run DFS linear elimination (Symbolica).

use std::fs::File;
use std::path::{Path, PathBuf};
use std::time::Instant;

use soothing_symbolica_elim::{
    default_systems_dir_from_manifest, dfs_linear_elimination_solving_symbolic,
    load_mathematica_system_from_path_with_var_names, SampleConfig, SubstitutionMap, VarId,
};

const TRACE_LOG_FILE: &str = "elimination_trace.log";

fn print_substitution_bindings(sub: &SubstitutionMap, var_name: &dyn Fn(VarId) -> String) {
    sub.insertion_order().iter().copied().for_each(|v| {
        if let Some(atom) = sub.get(v) {
            let s = format!("{}", symbolica::hide_namespace!(atom));
            println!("  {} = {}", var_name(v), s);
        }
    });
}

fn main() {
    let path: PathBuf = std::env::args().nth(1).map_or_else(
        || default_systems_dir_from_manifest().join("simplified_mathematica_system.txt"),
        PathBuf::from,
    );

    let trace_path = Path::new(TRACE_LOG_FILE);

    println!("soothing-symbolica-elim — DFS elimination on Mathematica system (Symbolica)\n");
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
        .flat_map(|eq| soothing_symbolica_elim::atom_util::vars_in_atom(eq, &loaded.sym_to_vid))
        .collect();
    let var_names = loaded.var_names.clone();
    println!(
        "Parsed: {} equations, {} distinct variables across system",
        n_eq,
        all_vars.len()
    );

    let cfg = SampleConfig::default();

    let mut trace_file = match File::create(trace_path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("trace log {}: {e}", trace_path.display());
            std::process::exit(1);
        }
    };

    let t0 = Instant::now();
    let res = match dfs_linear_elimination_solving_symbolic(&loaded, &cfg, Some(&mut trace_file)) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("dfs_linear_elimination_solving_symbolic error: {e}");
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
