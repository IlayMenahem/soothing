//! Load a Mathematica-style system and compute a lex Gröbner basis over ℚ (Symbolica `GroebnerBasis`).
//!
//! Usage: `groebner [path]` — default path: `systems/simplified_mathematica_system.txt` next to the crate.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use symbolica::LicenseManager;
use symbolica::atom::{AtomCore, Symbol};
use symbolica::domains::rational::Q;
use symbolica::poly::{
    LexOrder, PolyVariable, groebner::GroebnerBasis, polynomial::MultivariatePolynomial,
};

use symbolica_elim::{
    PolynomialSystem, default_systems_dir_from_manifest,
    load_mathematica_system_from_path_with_var_names,
};

type QPolyLex = MultivariatePolynomial<Q, u16, LexOrder>;

fn variable_order_lex(symbols: &[Symbol]) -> Arc<Vec<PolyVariable>> {
    Arc::new(symbols.iter().copied().map(PolyVariable::from).collect())
}

fn ideal_lex_q(
    system: &PolynomialSystem,
    var_order: Arc<Vec<PolyVariable>>,
) -> Result<Vec<QPolyLex>, String> {
    system
        .equations
        .iter()
        .map(|eq| {
            eq.expand()
                .try_to_polynomial::<_, u16>(&Q, Some(var_order.clone()))
        })
        .collect()
}

fn groebner_basis_lex_q(ideal: &[QPolyLex], print_stats: bool) -> GroebnerBasis<Q, u16, LexOrder> {
    GroebnerBasis::new(ideal, print_stats)
}

fn groebner_basis_from_system_lex_q(
    system: &PolynomialSystem,
    var_order: Arc<Vec<PolyVariable>>,
    print_stats: bool,
) -> Result<GroebnerBasis<Q, u16, LexOrder>, String> {
    let ideal = ideal_lex_q(system, var_order)?;
    Ok(groebner_basis_lex_q(&ideal, print_stats))
}

fn main() {
    let key = "5568d092#6bb8b4fb#ff936916-65ba-552f-b3c1-bdc64597cd21";
    LicenseManager::set_license_key(key).unwrap();

    let path: PathBuf = std::env::args().nth(1).map_or_else(
        || default_systems_dir_from_manifest().join("simplified_mathematica_system.txt"),
        PathBuf::from,
    );

    println!("symbolica-elim — Gröbner basis (lex over ℚ)\n");
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

    let var_order = variable_order_lex(&loaded.id_to_sym);

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
