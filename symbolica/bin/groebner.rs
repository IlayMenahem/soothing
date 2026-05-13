//! Load a Mathematica-style system and compute a lex Gröbner basis over ℚ (Symbolica `GroebnerBasis`).
//!
//! Usage: `groebner [path]` — default path: `systems/simplified_mathematica_system.txt` next to the crate.

use std::path::PathBuf;
use std::time::Instant;

use symbolica::LicenseManager;

use polynomial_milp::{
    default_systems_dir_from_manifest, groebner_basis_from_system_lex_q,
    load_mathematica_system_from_path_with_var_names, variable_order,
};

fn main() {
    let key = "5568d092#6bb8b4fb#ff936916-65ba-552f-b3c1-bdc64597cd21";
    LicenseManager::set_license_key(key).unwrap();

    let path: PathBuf = std::env::args().nth(1).map_or_else(
        || default_systems_dir_from_manifest().join("s_system.txt"),
        PathBuf::from,
    );

    println!("polynomial-milp — Gröbner basis (lex over ℚ)\n");
    println!("Loading {}", path.display());

    let loaded = match load_mathematica_system_from_path_with_var_names(&path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("load error: {e}");
            std::process::exit(1);
        }
    };

    let n_eq = loaded.system.equations.len();
    println!(
        "Parsed: {} equations, {} variables (loader order)",
        n_eq,
        loaded.var_names.len()
    );

    let var_order = variable_order(&loaded.id_to_sym);

    println!("\nGroebner basis (lex over Q, Symbolica `GroebnerBasis`)");
    let t0 = Instant::now();
    let gb = match groebner_basis_from_system_lex_q(&loaded.system, var_order, false) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("equation -> polynomial over Q: {e}");
            std::process::exit(1);
        }
    };
    println!(
        "  {} basis polynomial(s), elapsed {:?}",
        gb.system.len(),
        t0.elapsed()
    );
    for g in &gb.system {
        println!("\t{g}");
    }

    println!("\nDone.");
}
