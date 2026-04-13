//! Load `systems/simplified_mathematica_system.txt` and run DFS linear elimination (exact `Q` predicates).

use std::fs::File;
use std::path::{Path, PathBuf};
use std::time::Instant;

use symbolica::LicenseManager;

use symbolica_elim::{
    SubstitutionMap, VarId, default_systems_dir_from_manifest, dfs_eliminate_exact,
    is_solution_nonzero_kappah_exact, load_mathematica_system_from_path_with_var_names,
};

const TRACE_LOG_FILE: &str = "elimination_trace_naive_exact.log";

fn print_substitution_bindings(sub: &SubstitutionMap, var_name: &dyn Fn(VarId) -> String) {
    sub.insertion_order().iter().copied().for_each(|v| {
        if let Some(atom) = sub.get(v) {
            let s = format!("{}", symbolica::hide_namespace!(atom));
            println!("  {} = {}", var_name(v), s);
        }
    });
}

fn main() {
    let key = "5568d092#6bb8b4fb#ff936916-65ba-552f-b3c1-bdc64597cd21";
    LicenseManager::set_license_key(key).unwrap();

    let path: PathBuf = std::env::args().nth(1).map_or_else(
        || default_systems_dir_from_manifest().join("simplified_mathematica_system.txt"),
        PathBuf::from,
    );

    let trace_path = Path::new(TRACE_LOG_FILE);

    println!("symbolica-elim — DFS elimination (exact Q predicates)\n");
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
        .flat_map(|eq| symbolica_elim::atom_util::vars_in_atom(eq, &loaded.sym_to_vid))
        .collect();
    let var_names = loaded.var_names.clone();
    println!(
        "Parsed: {} equations, {} distinct variables across system",
        n_eq,
        all_vars.len()
    );

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
    let res = match dfs_eliminate_exact(
        &loaded.system,
        Some(&mut trace_file),
        |r| is_solution_nonzero_kappah_exact(r, kappah, &loaded.sym_to_vid, &loaded.id_to_sym),
        &loaded.sym_to_vid,
        &loaded.id_to_sym,
    ) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("dfs_eliminate_exact error: {e}");
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
