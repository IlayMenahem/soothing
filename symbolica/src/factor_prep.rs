//! Shared preprocessing for polynomial-system binaries: conversion into
//! Symbolica polynomials, interreduction, and square-free factorization.
//!
//! Implements steps 1–4 of `problem statement.md`:
//! 1. interreduce the polynomials under `GrevLexOrder`,
//! 2. square-free factorize each equation,
//! 3. drop constant factors,
//! 4. discard zero polynomials (empty factor groups).

use std::sync::Arc;

use symbolica::atom::{AtomCore, Symbol};
use symbolica::domains::algebraic_number::AlgebraicExtension;
use symbolica::domains::rational::Q;
use symbolica::poly::factor::Factorize;
use symbolica::poly::{GrevLexOrder, LexOrder, PolyVariable, polynomial::MultivariatePolynomial};

use crate::mathematica::LoadedMathematicaSystem;
use crate::system::PolynomialSystem;

pub type RationalField = Q;
pub type ExtQ = AlgebraicExtension<RationalField>;
pub type PolyExtLex = MultivariatePolynomial<ExtQ, u16, LexOrder>;
pub type PolyExtGrevLex = MultivariatePolynomial<ExtQ, u16, GrevLexOrder>;

#[must_use]
pub fn variable_order(symbols: &[Symbol]) -> Arc<Vec<PolyVariable>> {
    Arc::new(symbols.iter().copied().map(PolyVariable::from).collect())
}

pub fn ideal_ext_grevlex(
    system: &PolynomialSystem,
    field: &ExtQ,
    var_order: Arc<Vec<PolyVariable>>,
) -> Result<Vec<PolyExtGrevLex>, String> {
    system
        .equations
        .iter()
        .map(|eq| {
            eq.expand()
                .try_to_polynomial::<_, u16>(field, Some(var_order.clone()))
                .map(|p: PolyExtLex| p.reorder::<GrevLexOrder>())
        })
        .collect()
}

/// Fixpoint interreduction: each polynomial is replaced by its remainder
/// modulo all the others until nothing changes.
pub fn interreduce_polys(mut polys: Vec<PolyExtGrevLex>) -> Vec<PolyExtGrevLex> {
    if polys.is_empty() {
        return polys;
    }
    MultivariatePolynomial::unify_variables_list(&mut polys);

    loop {
        let n = polys.len();
        let next: Vec<PolyExtGrevLex> = (0..n)
            .map(|i| {
                let others: Vec<PolyExtGrevLex> = polys
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| *j != i)
                    .map(|(_, p)| p.clone())
                    .filter(|p| !p.is_zero())
                    .collect();
                polys[i].reduce(&others)
            })
            .collect();

        if next == polys {
            return next;
        }

        polys = next;
    }
}

/// Preprocessed view of the input system: for each (non-trivial) original
/// polynomial, the list of its non-constant square-free factors in
/// [`LexOrder`]. Zero polynomials are dropped entirely.
pub struct Preprocessed {
    pub factor_groups: Vec<Vec<PolyExtLex>>,
}

#[must_use]
pub fn preprocess(loaded: &LoadedMathematicaSystem, field: &ExtQ) -> Preprocessed {
    let var_order = variable_order(&loaded.id_to_sym);
    let polys = match ideal_ext_grevlex(&loaded.system, field, var_order) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("equation -> polynomial over AlgebraicExtension<Q>: {e}");
            std::process::exit(1);
        }
    };
    let polys = interreduce_polys(polys);

    let factor_groups: Vec<Vec<PolyExtLex>> = polys
        .into_iter()
        .filter_map(|p| {
            let p_lex = p.reorder::<LexOrder>();
            if p_lex.is_zero() {
                return None;
            }
            let factors: Vec<PolyExtLex> = p_lex
                .square_free_factorization()
                .into_iter()
                .map(|(f, _pow)| f)
                .filter(|f| !f.is_constant())
                .collect();
            if factors.is_empty() {
                None
            } else {
                Some(factors)
            }
        })
        .collect();

    Preprocessed { factor_groups }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mathematica::load_mathematica_system_from_str_with_var_names;
    use std::sync::Once;
    use symbolica::LicenseManager;

    static LICENSE: Once = Once::new();
    fn activate_license() {
        LICENSE.call_once(|| {
            let _ = LicenseManager::set_license_key(
                "5568d092#6bb8b4fb#ff936916-65ba-552f-b3c1-bdc64597cd21",
            );
        });
    }

    fn load(source: &str) -> crate::mathematica::LoadedMathematicaSystem {
        activate_license();
        load_mathematica_system_from_str_with_var_names(source).expect("load")
    }

    #[test]
    fn interreduce_empty_is_empty() {
        assert!(interreduce_polys(Vec::new()).is_empty());
    }

    #[test]
    fn preprocess_drops_pure_constant_equation() {
        // equation is q == q + 1 → q - (q + 1) = -1, a nonzero constant.
        let loaded = load("{q == q + 1}");
        let pre = preprocess(&loaded, &AlgebraicExtension::new_complex(Q));
        assert!(
            pre.factor_groups.is_empty(),
            "constant-only system should produce no factor groups, got {:?}",
            pre.factor_groups.len()
        );
    }

    #[test]
    fn preprocess_factors_are_non_constant_and_non_zero() {
        let loaded = load("{(x + 1)^2 * (y + 2) == 0, (x^2 + 1) == 0}");
        let pre = preprocess(&loaded, &AlgebraicExtension::new_complex(Q));
        assert!(!pre.factor_groups.is_empty());
        for group in &pre.factor_groups {
            assert!(!group.is_empty());
            for f in group {
                assert!(!f.is_constant());
                assert!(!f.is_zero());
            }
        }
    }

    #[test]
    fn preprocess_returns_one_group_per_nonzero_equation() {
        let loaded = load("{x + 1 == 0, y + 2 == 0, z + 3 == 0}");
        let pre = preprocess(&loaded, &AlgebraicExtension::new_complex(Q));
        assert_eq!(pre.factor_groups.len(), 3);
        for group in &pre.factor_groups {
            assert_eq!(group.len(), 1);
        }
    }
}
