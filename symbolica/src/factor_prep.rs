//! Shared preprocessing for polynomial-system binaries: conversion into
//! Symbolica polynomials, interreduction, and square-free factorization.
//!
//! Implements steps 1–4 of `problem statement.md`:
//! 1. interreduce the polynomials under [`LexOrder`],
//! 2. square-free factorize each equation,
//! 3. drop constant factors,
//! 4. discard zero polynomials (empty factor groups).

use std::collections::HashSet;
use std::sync::Arc;

use symbolica::atom::{AtomCore, Symbol};
use symbolica::domains::algebraic_number::AlgebraicExtension;
use symbolica::domains::rational::Q;
use symbolica::poly::factor::Factorize;
use symbolica::poly::{LexOrder, PolyVariable, polynomial::MultivariatePolynomial};

use crate::mathematica::LoadedMathematicaSystem;
use crate::system::PolynomialSystem;

pub type RationalField = Q;
pub type ExtQ = AlgebraicExtension<RationalField>;
pub type PolyExtLex = MultivariatePolynomial<ExtQ, u16, LexOrder>;

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

fn cooccurrence_graph(variable_count: usize, supports: &[Vec<usize>]) -> Vec<HashSet<usize>> {
    let mut graph = vec![HashSet::new(); variable_count];

    for support in supports {
        for (i, &left) in support.iter().enumerate() {
            for &right in support.iter().skip(i + 1) {
                graph[left].insert(right);
                graph[right].insert(left);
            }
        }
    }

    graph
}

fn active_neighbors(graph: &[HashSet<usize>], active: &[bool], variable: usize) -> Vec<usize> {
    let mut neighbors: Vec<usize> = graph[variable]
        .iter()
        .copied()
        .filter(|&neighbor| active[neighbor])
        .collect();
    neighbors.sort_unstable();
    neighbors
}

fn fill_edge_count(graph: &[HashSet<usize>], active: &[bool], variable: usize) -> usize {
    let neighbors = active_neighbors(graph, active, variable);
    let mut missing_edges = 0;

    for (i, &left) in neighbors.iter().enumerate() {
        for &right in neighbors.iter().skip(i + 1) {
            if !graph[left].contains(&right) {
                missing_edges += 1;
            }
        }
    }

    missing_edges
}

fn min_fill_order(variable_count: usize, supports: &[Vec<usize>]) -> Vec<usize> {
    let mut graph = cooccurrence_graph(variable_count, supports);
    let mut active = vec![true; variable_count];
    let mut order = Vec::with_capacity(variable_count);

    while order.len() < variable_count {
        let selected = (0..variable_count)
            .filter(|&variable| active[variable])
            .min_by_key(|&variable| {
                let neighbors = active_neighbors(&graph, &active, variable);
                (
                    fill_edge_count(&graph, &active, variable),
                    neighbors.len(),
                    variable,
                )
            })
            .expect("at least one active variable remains");

        let neighbors = active_neighbors(&graph, &active, selected);
        for (i, &left) in neighbors.iter().enumerate() {
            for &right in neighbors.iter().skip(i + 1) {
                graph[left].insert(right);
                graph[right].insert(left);
            }
        }
        for &neighbor in &neighbors {
            graph[neighbor].remove(&selected);
        }
        graph[selected].clear();
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

pub fn ideal_ext_lex(
    system: &PolynomialSystem,
    field: &ExtQ,
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

/// Fixpoint interreduction: each polynomial is replaced by its remainder
/// modulo all the others until nothing changes.
pub fn interreduce_polys(mut polys: Vec<PolyExtLex>) -> Vec<PolyExtLex> {
    if polys.is_empty() {
        return polys;
    }
    MultivariatePolynomial::unify_variables_list(&mut polys);

    loop {
        let n = polys.len();
        let next: Vec<PolyExtLex> = (0..n)
            .map(|i| {
                let others: Vec<PolyExtLex> = polys
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

fn non_constant_square_free_factors(poly: PolyExtLex) -> Vec<PolyExtLex> {
    poly.square_free_factorization()
        .into_iter()
        .map(|(factor, _pow)| factor)
        .filter(|factor| !factor.is_constant())
        .collect()
}

fn factor_groups_from_polys(polys: Vec<PolyExtLex>) -> Vec<Vec<PolyExtLex>> {
    polys
        .into_iter()
        .filter(|poly| !poly.is_zero())
        .filter_map(|poly| {
            let factors = non_constant_square_free_factors(poly);
            (!factors.is_empty()).then_some(factors)
        })
        .collect()
}

pub fn preprocess(loaded: &LoadedMathematicaSystem, field: &ExtQ) -> Result<Preprocessed, String> {
    let var_order = heuristic_variable_order(loaded);
    let polys = ideal_ext_lex(&loaded.system, field, var_order)
        .map_err(|e| format!("equation -> polynomial over AlgebraicExtension<Q>: {e}"))?;
    let polys = interreduce_polys(polys);

    Ok(Preprocessed {
        factor_groups: factor_groups_from_polys(polys),
    })
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
    fn interreduce_empty_is_empty() {
        assert!(interreduce_polys(Vec::new()).is_empty());
    }

    #[test]
    fn preprocess_drops_pure_constant_equation() {
        // equation is q == q + 1 → q - (q + 1) = -1, a nonzero constant.
        let loaded = load("{q == q + 1}");
        let pre = preprocess(&loaded, &AlgebraicExtension::new_complex(Q)).expect("preprocess");
        assert!(
            pre.factor_groups.is_empty(),
            "constant-only system should produce no factor groups, got {:?}",
            pre.factor_groups.len()
        );
    }

    #[test]
    fn preprocess_factors_are_non_constant_and_non_zero() {
        let loaded = load("{(x + 1)^2 * (y + 2) == 0, (x^2 + 1) == 0}");
        let pre = preprocess(&loaded, &AlgebraicExtension::new_complex(Q)).expect("preprocess");
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
        let pre = preprocess(&loaded, &AlgebraicExtension::new_complex(Q)).expect("preprocess");
        assert_eq!(pre.factor_groups.len(), 3);
        for group in &pre.factor_groups {
            assert_eq!(group.len(), 1);
        }
    }
}
