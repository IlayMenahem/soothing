//! Evaluates a candidate factor cover against a Gröbner-basis backend and
//! extracts a minimal unsat core when the cover turns out inconsistent.

use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use symbolica::poly::polynomial::MultivariatePolynomial;
use symbolica::poly::{GrevLexOrder, MonomialOrder};

use crate::cover::minimal_covers;
use crate::factor_prep::{ExtQ, PolyExtLex};
use crate::{groebner_f5, groebner_gvw, groebner_m4gb};

pub type PolyExtGrevLex = MultivariatePolynomial<ExtQ, u16, GrevLexOrder>;

pub fn solution_dimension(basis: &[PolyExtGrevLex], nvars: usize) -> Option<usize> {
    let supports: Vec<Vec<usize>> = basis
        .iter()
        .filter(|p| !p.is_zero())
        .map(|p| {
            p.max_exp()
                .iter()
                .enumerate()
                .filter(|&(_, &e)| e > 0)
                .map(|(i, _)| i)
                .collect()
        })
        .collect();
    minimal_covers(&supports, nvars)?
        .into_iter()
        .map(|transversal| nvars - transversal.count_ones() as usize)
        .max()
}

/// A reduced degrevlex Gröbner basis equal to `{1}` (a non-zero constant)
/// certifies that the cover's ideal is the whole ring, i.e. the cover is
/// inconsistent.
fn is_unit_ideal<O: MonomialOrder>(system: &[MultivariatePolynomial<ExtQ, u16, O>]) -> bool {
    system.iter().any(|p| !p.is_zero() && p.is_constant())
}

/// Outcome of computing the Gröbner basis of a single cover.
pub enum Outcome {
    /// Non-unit reduced basis: the cover is consistent.
    Consistent(Vec<PolyExtGrevLex>),
    /// Reduced basis is `{1}`: the cover's ideal is the whole ring.
    Inconsistent,
    /// Computation exceeded the per-cover time budget.
    Timeout,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GbBackend {
    M4gb,
    F5,
    Gvw,
    Symbolica,
}

impl GbBackend {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "m4gb" => Some(Self::M4gb),
            "f5" => Some(Self::F5),
            "gvw" => Some(Self::Gvw),
            "symbolica" => Some(Self::Symbolica),
            _ => None,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::M4gb => "m4gb",
            Self::F5 => "f5",
            Self::Gvw => "gvw",
            Self::Symbolica => "symbolica",
        }
    }
}

/// Compute the cover's Gröbner basis on a worker thread, abandoning it if it
/// outlives `timeout`. The cover is reordered to degrevlex before the
/// computation, which is typically far cheaper than lex for deciding
/// satisfiability (Section 3.1). Cancellation is cooperative-free, so a
/// timed-out thread keeps running until it finishes on its own.
pub fn evaluate_cover(cover: Vec<PolyExtLex>, timeout: Duration, backend: GbBackend) -> Outcome {
    // Interreduction retains zero polynomials to preserve the input slots, but
    // they impose no constraint. In particular, Symbolica 1.5.1 attempts to
    // normalize a zero generator and divides by its zero leading coefficient.
    let cover: Vec<_> = cover.into_iter().filter(|poly| !poly.is_zero()).collect();
    if cover.is_empty() {
        return Outcome::Consistent(Vec::new());
    }
    // Do not send an explicit unit ideal to a Gröbner backend. Besides being
    // unnecessary, Symbolica 1.5.1 attempts to normalize the constant again
    // and can divide by zero in its algebraic-number arithmetic.
    if is_unit_ideal(&cover) {
        return Outcome::Inconsistent;
    }
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let grevlex: Vec<_> = cover.iter().map(|p| p.reorder::<GrevLexOrder>()).collect();
        let system = match backend {
            GbBackend::M4gb => groebner_m4gb(&grevlex).system,
            GbBackend::F5 => groebner_f5(&grevlex).system,
            GbBackend::Gvw => groebner_gvw(&grevlex).system,
            GbBackend::Symbolica => {
                symbolica::poly::groebner::GroebnerBasis::new(&grevlex, false).system
            }
        };
        let _ = tx.send(system);
    });
    match rx.recv_timeout(timeout) {
        Ok(system) if is_unit_ideal(&system) => Outcome::Inconsistent,
        Ok(system) => Outcome::Consistent(system),
        Err(_) => Outcome::Timeout,
    }
}

pub type Core = Vec<PolyExtLex>;

/// Whether `superset` contains every normalized polynomial in `subset`.
/// Candidates from different hint branches have different factor-index
/// universes, so pruning must compare the actual generators rather than masks.
pub fn polynomial_set_contains(superset: &[PolyExtLex], subset: &[PolyExtLex]) -> bool {
    subset.iter().all(|poly| superset.contains(poly))
}

/// Shrink an inconsistent candidate to a minimal unsatisfiable core by
/// deletion. A removal that turns the candidate satisfiable keeps the
/// polynomial; a removal that times out also keeps it conservatively. Since
/// removing constraints cannot make a satisfiable system inconsistent, one
/// pass produces a deletion-minimal core.
pub fn minimal_unsat_core(candidate: &[PolyExtLex], timeout: Duration, backend: GbBackend) -> Core {
    let mut core = candidate.to_vec();
    let mut index = 0;
    while index < core.len() {
        let mut reduced = core.clone();
        reduced.remove(index);
        if matches!(
            evaluate_cover(reduced.clone(), timeout, backend),
            Outcome::Inconsistent
        ) {
            core = reduced;
        } else {
            index += 1;
        }
    }
    core
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Once;

    use symbolica::LicenseManager;
    use symbolica::atom::AtomCore;
    use symbolica::domains::algebraic_number::AlgebraicExtension;
    use symbolica::domains::rational::Q;

    use crate::atom_util::parse_expression;
    use crate::factor_prep::{ideal_ext_lex, variable_order};
    use crate::mathematica::load_mathematica_system_from_str_with_var_names;

    static LICENSE: Once = Once::new();

    fn activate_license() {
        LICENSE.call_once(|| {
            let _ = LicenseManager::set_license_key(
                "5568d092#6bb8b4fb#ff936916-65ba-552f-b3c1-bdc64597cd21",
            );
        });
    }

    fn test_poly(source: &str) -> PolyExtLex {
        activate_license();
        let loaded = load_mathematica_system_from_str_with_var_names(&format!("{{{source} == 0}}"))
            .expect("load");
        let field = AlgebraicExtension::new_complex(Q);
        let order = variable_order(&loaded.id_to_sym);
        ideal_ext_lex(&loaded.system, &field, order)
            .expect("polynomial")
            .remove(0)
    }

    fn poly_like(template: &PolyExtLex, source: &str) -> PolyExtLex {
        parse_expression(source.to_string())
            .expect("parse")
            .try_to_polynomial::<_, u16>(&template.ring, Some(template.variables.clone()))
            .expect("polynomial")
    }

    fn grevlex_basis(template: &PolyExtLex, sources: &[&str]) -> Vec<PolyExtGrevLex> {
        sources
            .iter()
            .map(|source| poly_like(template, source).reorder::<GrevLexOrder>())
            .collect()
    }

    #[test]
    fn solution_dimension_counts_independent_variables() {
        let template = test_poly("x + y + z");
        assert_eq!(
            solution_dimension(&grevlex_basis(&template, &["x"]), 3),
            Some(2)
        );
        assert_eq!(
            solution_dimension(&grevlex_basis(&template, &["x*y"]), 3),
            Some(2)
        );
        assert_eq!(
            solution_dimension(&grevlex_basis(&template, &["x*y", "z"]), 3),
            Some(1)
        );
        assert_eq!(
            solution_dimension(&grevlex_basis(&template, &["x", "y", "z"]), 3),
            Some(0)
        );
    }

    #[test]
    fn solution_dimension_of_empty_basis_is_ambient() {
        assert_eq!(solution_dimension(&[], 3), Some(3));
    }

    #[test]
    fn explicit_unit_ideal_skips_symbolica_backend() {
        let template = test_poly("x");
        let one = poly_like(&template, "1");

        assert!(matches!(
            evaluate_cover(vec![one], Duration::from_millis(10), GbBackend::Symbolica),
            Outcome::Inconsistent
        ));
    }

    #[test]
    fn zero_generators_are_removed_before_symbolica_backend() {
        let template = test_poly("x");
        let zero = poly_like(&template, "0");

        assert!(matches!(
            evaluate_cover(
                vec![zero, template],
                Duration::from_secs(5),
                GbBackend::Symbolica
            ),
            Outcome::Consistent(_)
        ));
    }

    #[test]
    fn minimal_unsat_core_retains_required_zero_substitution() {
        let factor = test_poly("x + 1");
        let hint = poly_like(&factor, "x");

        let core = minimal_unsat_core(
            &[factor.clone(), hint.clone()],
            Duration::from_secs(5),
            GbBackend::M4gb,
        );

        assert_eq!(core, vec![factor, hint]);
    }

    #[test]
    fn minimal_unsat_core_drops_unneeded_zero_substitution() {
        let template = test_poly("x + y + 1");
        let factor = poly_like(&template, "x + 1");
        let x_hint = poly_like(&template, "x");
        let y_hint = poly_like(&template, "y");

        let core = minimal_unsat_core(
            &[factor.clone(), x_hint.clone(), y_hint],
            Duration::from_secs(5),
            GbBackend::M4gb,
        );

        assert_eq!(core, vec![factor, x_hint]);
    }

    #[test]
    fn polynomial_set_containment_uses_actual_generators() {
        let template = test_poly("x + y");
        let x = poly_like(&template, "x");
        let y = poly_like(&template, "y");

        assert!(polynomial_set_contains(
            &[x.clone(), y.clone()],
            std::slice::from_ref(&x)
        ));
        assert!(!polynomial_set_contains(&[x], &[y]));
    }
}
