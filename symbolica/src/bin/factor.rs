//! Load a Mathematica-style system, interreduce the converted ideal (each polynomial
//! reduced modulo the others via [`MultivariatePolynomial::reduce`](https://docs.rs/symbolica/latest/symbolica/poly/polynomial/struct.MultivariatePolynomial.html#method.reduce)).
//! Interreduction and factorization both use lexicographic monomial order ([`LexOrder`](https://docs.rs/symbolica/latest/symbolica/poly/struct.LexOrder.html)), consistent with Symbolica’s default lex Groebner workflow unless polynomials are reordered to e.g. [`GrevLexOrder`](https://docs.rs/symbolica/latest/symbolica/poly/struct.GrevLexOrder.html).
//!
//! Field: Gaussian rationals ℚ(i) via [`AlgebraicExtension::new_complex`].
//!
//! Usage: `factor [path]` — default path: `systems/system.txt` next to the crate.

use std::path::PathBuf;
use std::time::Instant;

use symbolica::LicenseManager;
use symbolica::domains::algebraic_number::AlgebraicExtension;
use symbolica::domains::rational::Q;
use symbolica::poly::PolyVariable;
use symbolica::poly::factor::Factorize;

use symbolica_elim::{
    LoadedMathematicaSystem, PolyExtLex, default_systems_dir_from_manifest, ideal_ext_lex,
    interreduce_polys, load_mathematica_system_from_path_with_var_names, variable_order,
};

fn linear_var_names_for_factor(f: &PolyExtLex, loaded: &LoadedMathematicaSystem) -> Vec<String> {
    let mut out = Vec::new();
    for (xi, pv) in f.get_vars_ref().iter().enumerate() {
        if !f.contains(xi) {
            continue;
        }
        let PolyVariable::Symbol(sym) = pv else {
            continue;
        };
        let Some(&vid) = loaded.sym_to_vid.get(sym) else {
            continue;
        };
        if f.degree(xi) <= 1 {
            out.push(loaded.var_names[vid.0 as usize].clone());
        }
    }
    out
}

fn main() {
    let key = "5568d092#6bb8b4fb#ff936916-65ba-552f-b3c1-bdc64597cd21";
    LicenseManager::set_license_key(key).unwrap();

    let path: PathBuf = std::env::args().nth(1).map_or_else(
        || default_systems_dir_from_manifest().join("system.txt"),
        PathBuf::from,
    );

    println!("symbolica-elim — factor over AlgebraicExtension<RationalField> (ℚ(i))\n");
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

    let number_field = AlgebraicExtension::new_complex(Q);
    let var_order = variable_order(&loaded.id_to_sym);

    println!("\nInterreduction (`MultivariatePolynomial::reduce` modulo other equations)");
    let t_inter = Instant::now();
    let polys = match ideal_ext_lex(&loaded.system, &number_field, var_order) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("equation -> polynomial over AlgebraicExtension<Q>: {e}");
            std::process::exit(1);
        }
    };
    let polys = interreduce_polys(polys);
    println!(
        "  {} equation(s), elapsed {:?}",
        polys.len(),
        t_inter.elapsed()
    );

    let t0 = Instant::now();

    for (i, poly) in polys.iter().enumerate() {
        println!("\n[{i}]  {poly}");

        if poly.is_zero() {
            println!("  Factors: (zero)");
        } else {
            for f in poly.clone().square_free_factorization() {
                let linear_vars = linear_var_names_for_factor(&f.0, &loaded);
                println!("    Linear in: {linear_vars:?}");
            }
        }
    }

    println!("\nElapsed {:?}\nDone.", t0.elapsed());
}
