//! Shared preprocessing for polynomial-system binaries: conversion into
//! Symbolica polynomials, interreduction, and complete factorization.
//!
//! Implements steps 1–4 of `problem statement.md`:
//! 1. interreduce the polynomials under [`LexOrder`],
//! 2. clear zero polynomials,
//! 3. completely factorize each surviving polynomial,
//! 4. drop constant factors.

use std::sync::Arc;

use symbolica::atom::{AtomCore, Symbol};
use symbolica::domains::rational::Q;
use symbolica::poly::factor::Factorize;
use symbolica::poly::{LexOrder, PolyVariable, polynomial::MultivariatePolynomial};

use crate::mathematica::LoadedMathematicaSystem;
use crate::system::PolynomialSystem;

pub type PolyExtLex = MultivariatePolynomial<Q, u16, LexOrder>;

#[must_use]
pub fn variable_order(symbols: &[Symbol]) -> Arc<Vec<PolyVariable>> {
    Arc::new(symbols.iter().copied().map(PolyVariable::from).collect())
}

fn equation_supports(loaded: &LoadedMathematicaSystem) -> Vec<Vec<usize>> {
    loaded
        .system
        .equations
        .iter()
        .map(|equation| {
            let mut support: Vec<usize> = equation
                .get_all_symbols(false)
                .into_iter()
                .filter_map(|symbol| loaded.sym_to_vid.get(&symbol).map(|vid| vid.0 as usize))
                .collect();
            support.sort_unstable();
            support.dedup();
            support
        })
        .collect()
}

// For variable `v`, returns the minimum number of new variable slots introduced
// across all non-definer equations when `v` is eliminated. This is minimized
// over all choices of definer equation, since lex interreduction picks one
// polynomial as v's definer and spreads only its variables into the others.
// The pairwise co-occurrence graph overestimates here because it assumes all
// pairs of v's neighbors must be cliqued, but only the definer's variables
// actually propagate to each other equation.
fn hyperedge_fill(variable: usize, supports: &[Vec<usize>]) -> usize {
    let containing: Vec<usize> = supports
        .iter()
        .enumerate()
        .filter(|(_, s)| !s.is_empty() && s.binary_search(&variable).is_ok())
        .map(|(i, _)| i)
        .collect();

    if containing.len() <= 1 {
        return 0;
    }

    containing
        .iter()
        .map(|&di| {
            containing
                .iter()
                .filter(|&&ei| ei != di)
                .map(|&ei| {
                    supports[di]
                        .iter()
                        .filter(|&&v| v != variable && supports[ei].binary_search(&v).is_err())
                        .count()
                })
                .sum::<usize>()
        })
        .min()
        .unwrap_or(0)
}

fn min_fill_order(variable_count: usize, supports: &[Vec<usize>]) -> Vec<usize> {
    let mut active_supports: Vec<Vec<usize>> = supports
        .iter()
        .map(|s| {
            let mut v = s.clone();
            v.sort_unstable();
            v.dedup();
            v
        })
        .collect();
    let mut active = vec![true; variable_count];
    let mut order = Vec::with_capacity(variable_count);

    while order.len() < variable_count {
        let selected = (0..variable_count)
            .filter(|&v| active[v])
            .min_by_key(|&v| {
                let eq_count = active_supports
                    .iter()
                    .filter(|s| !s.is_empty() && s.binary_search(&v).is_ok())
                    .count();
                (hyperedge_fill(v, &active_supports), eq_count, v)
            })
            .expect("at least one active variable remains");

        let containing: Vec<usize> = active_supports
            .iter()
            .enumerate()
            .filter(|(_, s)| !s.is_empty() && s.binary_search(&selected).is_ok())
            .map(|(i, _)| i)
            .collect();

        if containing.len() > 1 {
            let definer_idx = containing
                .iter()
                .copied()
                .min_by_key(|&di| {
                    containing
                        .iter()
                        .filter(|&&ei| ei != di)
                        .map(|&ei| {
                            active_supports[di]
                                .iter()
                                .filter(|&&v| {
                                    v != selected && active_supports[ei].binary_search(&v).is_err()
                                })
                                .count()
                        })
                        .sum::<usize>()
                })
                .unwrap();

            let definer_vars: Vec<usize> = active_supports[definer_idx]
                .iter()
                .copied()
                .filter(|&v| v != selected)
                .collect();

            for &ei in &containing {
                if ei == definer_idx {
                    continue;
                }
                let eq = &mut active_supports[ei];
                eq.retain(|&v| v != selected);
                for &dv in &definer_vars {
                    if eq.binary_search(&dv).is_err() {
                        eq.push(dv);
                    }
                }
                eq.sort_unstable();
            }
            active_supports[definer_idx].clear();
        } else if containing.len() == 1 {
            active_supports[containing[0]].clear();
        }

        active[selected] = false;
        order.push(selected);
    }

    order
}

fn heuristic_variable_order(loaded: &LoadedMathematicaSystem) -> Arc<Vec<PolyVariable>> {
    let supports = equation_supports(loaded);
    let symbols: Vec<Symbol> = min_fill_order(loaded.id_to_sym.len(), &supports)
        .into_iter()
        .map(|vid| loaded.id_to_sym[vid])
        .collect();
    variable_order(&symbols)
}

/// Re-express `polys` in the min-fill variable order of their own supports,
/// the order [`polynomials_from_loaded`] derives for a loaded system. Ties are
/// broken by variable name, matching the loader's sorted-identifier ids.
///
/// Lex interreduction and factorization depend on the order, so a subsystem
/// is best refined under an order derived from its own structure rather than
/// one inherited from the system it was split from.
pub fn reorder_by_min_fill(mut polys: Vec<PolyExtLex>) -> Result<Vec<PolyExtLex>, String> {
    if polys.is_empty() {
        return Ok(polys);
    }
    MultivariatePolynomial::unify_variables_list(&mut polys);
    let variables = polys[0].variables.clone();
    let mut by_name: Vec<usize> = (0..variables.len()).collect();
    by_name.sort_by_cached_key(|&position| variables[position].to_string());
    let canonical_id: Vec<usize> =
        by_name
            .iter()
            .enumerate()
            .fold(vec![0; variables.len()], |mut ids, (id, &position)| {
                ids[position] = id;
                ids
            });
    let supports: Vec<Vec<usize>> = polys
        .iter()
        .map(|polynomial| {
            (0..variables.len())
                .filter(|&position| polynomial.contains(position))
                .map(|position| canonical_id[position])
                .collect()
        })
        .collect();
    let order: Vec<PolyVariable> = min_fill_order(variables.len(), &supports)
        .into_iter()
        .map(|id| variables[by_name[id]].clone())
        .collect();
    polys
        .iter()
        .map(|polynomial| polynomial.rearrange_with_growth(&order))
        .collect()
}

pub fn ideal_ext_lex(
    system: &PolynomialSystem,
    field: &Q,
    var_order: Arc<Vec<PolyVariable>>,
) -> Result<Vec<PolyExtLex>, String> {
    system
        .equations
        .iter()
        .map(|eq| {
            eq.expand()
                .try_to_polynomial::<_, u16>(field, Some(var_order.clone()))
        })
        .collect()
}

/// Fixpoint interreduction using Gauss-Seidel sequential updates: each
/// polynomial is reduced modulo all others and written back immediately, so
/// later reductions in the same pass see already-updated values.
///
/// This guarantees the polynomial ideal is preserved at every step: replacing
/// `p_i` with `p_i - q·p_j` keeps `p_i = (p_i - q·p_j) + q·p_j` in the
/// span of the new set, so `⟨old⟩ = ⟨new⟩` after each single substitution.
/// The simultaneous (Jacobi) approach lacks this guarantee and can silently
/// shrink the ideal.
pub fn interreduce_polys(mut polys: Vec<PolyExtLex>) -> Vec<PolyExtLex> {
    if polys.is_empty() {
        return polys;
    }
    MultivariatePolynomial::unify_variables_list(&mut polys);

    loop {
        let mut changed = false;
        for i in 0..polys.len() {
            let others: Vec<PolyExtLex> = polys
                .iter()
                .enumerate()
                .filter(|(j, p)| *j != i && !p.is_zero())
                .map(|(_, p)| p.clone())
                .collect();
            let reduced = polys[i].reduce(&others);
            if reduced != polys[i] {
                polys[i] = reduced;
                changed = true;
            }
        }
        if !changed {
            return polys;
        }
    }
}

fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else {
        "unknown factorization panic".to_string()
    }
}

fn non_constant_factors(poly: PolyExtLex) -> Result<Vec<PolyExtLex>, String> {
    // Symbolica's multivariate factorizer chooses randomized evaluation points.
    // A bad choice can occasionally make its leading-coefficient reconstruction
    // panic even though another choice factors the same polynomial correctly.
    // Retry the operation, but never reinterpret an unfactored polynomial as an
    // irreducible factor: exhausting the retries is an explicit error.
    const ATTEMPTS: usize = 4;
    let mut last_error = String::new();
    for _ in 0..ATTEMPTS {
        let candidate = poly.clone();
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| candidate.factor())) {
            Ok(factors) => {
                return Ok(factors
                    .into_iter()
                    .map(|(factor, _pow)| factor)
                    .filter(|factor| !factor.is_constant())
                    .collect());
            }
            Err(payload) => last_error = panic_message(payload.as_ref()),
        }
    }
    Err(format!(
        "factorization failed after {ATTEMPTS} randomized attempts: {last_error}"
    ))
}

/// Factor each nonzero polynomial into its nonconstant irreducible factors.
///
/// A nonzero constant is retained as an empty factor group: the equation
/// `c = 0` is inconsistent and therefore admits no hitting set. Zero
/// polynomials, on the other hand, impose no constraint and are discarded.
pub fn try_factor_groups_from_polys(
    polys: Vec<PolyExtLex>,
) -> Result<Vec<Vec<PolyExtLex>>, String> {
    polys
        .into_iter()
        .filter(|poly| !poly.is_zero())
        .map(non_constant_factors)
        .collect()
}

/// Compatibility wrapper for callers that cannot propagate preprocessing
/// failures. New command-line code should use `try_factor_groups_from_polys`.
pub fn factor_groups_from_polys(polys: Vec<PolyExtLex>) -> Vec<Vec<PolyExtLex>> {
    try_factor_groups_from_polys(polys).unwrap_or_else(|error| panic!("{error}"))
}

/// Convert the loaded equations to a shared lexicographic polynomial ring,
/// using the deterministic fill heuristic for the variable order.
pub fn polynomials_from_loaded(
    loaded: &LoadedMathematicaSystem,
    field: &Q,
) -> Result<Vec<PolyExtLex>, String> {
    let var_order = heuristic_variable_order(loaded);
    ideal_ext_lex(&loaded.system, field, var_order)
        .map_err(|e| format!("equation -> polynomial over Q: {e}"))
}

pub fn preprocess(
    loaded: &LoadedMathematicaSystem,
    field: &Q,
) -> Result<Vec<Vec<PolyExtLex>>, String> {
    try_factor_groups_from_polys(interreduce_polys(polynomials_from_loaded(loaded, field)?))
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
    fn equation_supports_are_sorted_global_variable_ids() {
        let loaded = load("{x*y + z == 0, y + 1 == 0}");

        let supports = equation_supports(&loaded);

        let x = loaded.sym_to_vid[&loaded.id_to_sym[0]].0 as usize;
        let y = loaded.sym_to_vid[&loaded.id_to_sym[1]].0 as usize;
        let z = loaded.sym_to_vid[&loaded.id_to_sym[2]].0 as usize;
        assert_eq!(supports, vec![vec![x, y, z], vec![y]]);
    }

    #[test]
    fn min_fill_order_is_deterministic_on_ties() {
        let supports = vec![vec![0, 1], vec![1, 2], vec![2, 3], vec![3, 0]];

        let order = min_fill_order(4, &supports);

        assert_eq!(order, vec![0, 1, 2, 3]);
    }

    #[test]
    fn heuristic_variable_order_uses_min_fill_order() {
        let loaded = load("{a*b == 0, b*c == 0, c*d == 0, d*a == 0}");

        let order = heuristic_variable_order(&loaded);
        let names: Vec<String> = order.iter().map(|variable| variable.to_string()).collect();

        assert_eq!(names, vec!["a", "b", "c", "d"]);
    }

    #[test]
    fn reorder_by_min_fill_matches_loader_order_and_preserves_polynomials() {
        let loaded = load("{a*b + c == 0, b*c == 0, c*d - a == 0, d*a == 0}");
        let expected = polynomials_from_loaded(&loaded, &Q).expect("heuristic order");
        let alphabetical = ideal_ext_lex(&loaded.system, &Q, variable_order(&loaded.id_to_sym))
            .expect("alphabetical order");

        let reordered = reorder_by_min_fill(alphabetical).expect("reorder");

        assert_eq!(reordered, expected);
        assert_eq!(reordered[0].variables, expected[0].variables);
    }

    #[test]
    fn interreduce_empty_is_empty() {
        assert!(interreduce_polys(Vec::new()).is_empty());
    }

    #[test]
    fn interreduce_preserves_ideal_for_symmetric_pair() {
        // With Jacobi (simultaneous) updates, both x+y and x-y are reduced
        // against each other at the same time, giving {2y, -2y}, then {0, 0},
        // which loses x entirely — the ideal shrinks from ⟨x,y⟩ to ⟨y⟩ to 0.
        // Gauss-Seidel updates x+y → 2y first, then reduces x-y by 2y and
        // finds x-y unchanged, so the result is {2y, x-y} — both generators
        // of ⟨x,y⟩ survive.
        activate_license();
        let loaded = load("{x + y == 0, x - y == 0}");
        let field = Q;
        let var_order = variable_order(&loaded.id_to_sym);
        let polys = ideal_ext_lex(&loaded.system, &field, var_order).expect("ideal_ext_lex");
        let reduced = interreduce_polys(polys);
        let non_zero_count = reduced.iter().filter(|p| !p.is_zero()).count();
        assert_eq!(
            non_zero_count, 2,
            "both generators of <x,y> must survive; got {non_zero_count} non-zero polynomial(s)"
        );
    }

    #[test]
    fn preprocess_retains_pure_constant_as_empty_factor_group() {
        // equation is q == q + 1 → q - (q + 1) = -1, a nonzero constant.
        let loaded = load("{q == q + 1}");
        let pre = preprocess(&loaded, &Q).expect("preprocess");
        assert_eq!(pre, vec![Vec::<PolyExtLex>::new()]);
    }

    #[test]
    fn preprocess_factors_are_non_constant_and_non_zero() {
        let loaded = load("{(x + 1)^2 * (y + 2) == 0, (x^2 + 1) == 0}");
        let pre = preprocess(&loaded, &Q).expect("preprocess");
        assert!(!pre.is_empty());
        for group in &pre {
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
        let pre = preprocess(&loaded, &Q).expect("preprocess");
        assert_eq!(pre.len(), 3);
        for group in &pre {
            assert_eq!(group.len(), 1);
        }
    }
}
