//! Library entry point for Symbolica's lex Gröbner basis over ℚ.
//!
//! Mirrors the `groebner` binary so other binaries (notably the
//! `bench` harness) can call it directly.

use std::sync::Arc;

use symbolica::atom::AtomCore;
use symbolica::domains::rational::Q;
use symbolica::poly::{
    LexOrder, PolyVariable, groebner::GroebnerBasis, polynomial::MultivariatePolynomial,
};

use crate::system::PolynomialSystem;

pub type QPolyLex = MultivariatePolynomial<Q, u16, LexOrder>;

pub fn ideal_lex_q(
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

pub fn groebner_basis_lex_q(
    ideal: &[QPolyLex],
    print_stats: bool,
) -> GroebnerBasis<Q, u16, LexOrder> {
    GroebnerBasis::new(ideal, print_stats)
}

pub fn groebner_basis_from_system_lex_q(
    system: &PolynomialSystem,
    var_order: Arc<Vec<PolyVariable>>,
    print_stats: bool,
) -> Result<GroebnerBasis<Q, u16, LexOrder>, String> {
    let ideal = ideal_lex_q(system, var_order)?;
    Ok(groebner_basis_lex_q(&ideal, print_stats))
}
