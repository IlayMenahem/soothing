//! Evaluates a candidate factor cover against a Gröbner-basis backend and
//! extracts a minimal unsat core (the zero-substitutions actually needed)
//! when the cover turns out inconsistent.

use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use symbolica::poly::polynomial::MultivariatePolynomial;
use symbolica::poly::{GrevLexOrder, MonomialOrder};

use crate::cover::{Core, WorkItem, minimal_covers, set_bits};
use crate::factor_prep::{ExtQ, PolyExtLex};
use crate::sparse::decode_work_item;
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
    if cover.is_empty() {
        return Outcome::Consistent(Vec::new());
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

/// Shrink an inconsistent work item to a minimal unsatisfiable core by deletion:
/// drop each factor or zero substitution whose removal leaves the ideal equal to
/// the whole ring. A removal that turns the item satisfiable keeps the bit; a
/// removal that times out also keeps it, conservatively, since the larger item is
/// still known inconsistent. The result is deletion-minimal over both masks.
pub fn minimal_unsat_core(
    factors: &[PolyExtLex],
    sparse_variables: &[(String, usize)],
    item: WorkItem,
    timeout: Duration,
    backend: GbBackend,
) -> Core {
    let mut core = item;
    loop {
        let mut changed = false;

        for bit in set_bits(core.cover_mask).collect::<Vec<_>>() {
            let reduced = WorkItem {
                cover_mask: core.cover_mask & !bit,
                zero_mask: core.zero_mask,
            };
            if matches!(
                evaluate_cover(
                    decode_work_item(factors, sparse_variables, reduced),
                    timeout,
                    backend,
                ),
                Outcome::Inconsistent
            ) {
                core = reduced;
                changed = true;
            }
        }

        for bit in set_bits(core.zero_mask).collect::<Vec<_>>() {
            let reduced = WorkItem {
                cover_mask: core.cover_mask,
                zero_mask: core.zero_mask & !bit,
            };
            if matches!(
                evaluate_cover(
                    decode_work_item(factors, sparse_variables, reduced),
                    timeout,
                    backend,
                ),
                Outcome::Inconsistent
            ) {
                core = reduced;
                changed = true;
            }
        }

        if !changed {
            break;
        }
    }

    Core {
        cover_mask: core.cover_mask,
        zero_mask: core.zero_mask,
    }
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
    fn minimal_unsat_core_retains_required_zero_substitution() {
        let factor = test_poly("x + 1");
        let factors = vec![factor];
        let sparse_variables = vec![("x".to_string(), 0)];

        let core = minimal_unsat_core(
            &factors,
            &sparse_variables,
            WorkItem {
                cover_mask: 0b1,
                zero_mask: 0b1,
            },
            Duration::from_secs(5),
            GbBackend::M4gb,
        );

        assert_eq!(
            core,
            Core {
                cover_mask: 0b1,
                zero_mask: 0b1
            }
        );
    }

    #[test]
    fn minimal_unsat_core_drops_unneeded_zero_substitution() {
        let template = test_poly("x + y + 1");
        let factors = vec![poly_like(&template, "x + 1")];
        let sparse_variables = vec![("x".to_string(), 0), ("y".to_string(), 1)];

        let core = minimal_unsat_core(
            &factors,
            &sparse_variables,
            WorkItem {
                cover_mask: 0b1,
                zero_mask: 0b11,
            },
            Duration::from_secs(5),
            GbBackend::M4gb,
        );

        assert_eq!(
            core,
            Core {
                cover_mask: 0b1,
                zero_mask: 0b01
            }
        );
    }
}
