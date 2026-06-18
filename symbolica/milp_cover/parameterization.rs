//! Symbolic substitution from a cover, and exact correctness checks
//! against the original / preprocessed systems.

use std::collections::{HashMap, HashSet};
use std::ops::{Div, Sub};

use symbolica::atom::{Atom, AtomCore};
use thiserror::Error;

use crate::atom_util::{parse_expression, substitute_symbol};
use crate::exact::{
    ExactPredicateError, is_numerator_zero_exact, is_zero_exact, rhs_denominator_nonzero_exact,
};
use crate::factor_prep::PolyExtLex;
use crate::mathematica::LoadedMathematicaSystem;
use crate::milp_cover::milp::Cover;
use crate::milp_cover::problem::{ProblemData, var_name};
use crate::substitution::SubstitutionMap;
use crate::var_id::VarId;

#[derive(Debug, Eq, PartialEq)]
pub struct ParameterizationReport {
    pub checked_equations: usize,
    pub binding_count: usize,
}

#[derive(Debug, Error)]
pub enum ParameterizationError {
    #[error("could not parse selected factor `{factor}` as a Symbolica expression: {parse_error}")]
    FactorParse { factor: String, parse_error: String },
    #[error("cover assigns variable `{variable}` more than once")]
    DuplicateBinding { variable: String },
    #[error("selected factor is not linear in `{variable}` after conversion to an expression")]
    NonlinearSelectedFactor { variable: String },
    #[error("selected solve for `{variable}` has a zero or non-polynomial denominator")]
    DegenerateSubstitution { variable: String },
    #[error("cyclic dependency between cover substitutions")]
    SubstitutionCycle,
    #[error("exact validation failed for original equation {equation_index}: {source}")]
    ExactCheck {
        equation_index: usize,
        source: ExactPredicateError,
    },
    #[error(
        "original equation {equation_index} does not vanish after cover substitution: {residual}"
    )]
    EquationDoesNotVanish {
        equation_index: usize,
        residual: Atom,
    },
    #[error(
        "algebraic substitution for degree-{degree} factor defining `{variable}` is not yet implemented"
    )]
    AlgebraicSubstitutionNotImplemented { variable: String, degree: u16 },
}

fn var_id_from_global(global_vid: usize) -> VarId {
    VarId(u32::try_from(global_vid).expect("global variable id must fit in VarId"))
}

pub(crate) fn factor_to_atom(factor: &PolyExtLex) -> Result<Atom, ParameterizationError> {
    let factor_string = factor
        .to_string()
        .split_once(" % ")
        .map_or_else(|| factor.to_string(), |(expr, _modulus)| expr.to_string());
    parse_expression(factor_string.clone()).map_err(|parse_error| {
        ParameterizationError::FactorParse {
            factor: factor_string,
            parse_error,
        }
    })
}

fn linear_factor_rhs(
    loaded: &LoadedMathematicaSystem,
    factor: &PolyExtLex,
    selected_vid: usize,
) -> Result<Atom, ParameterizationError> {
    let variable = var_name(loaded, selected_vid).to_string();
    let selected_symbol = loaded.id_to_sym[selected_vid];
    let factor_atom = factor_to_atom(factor)?;
    let b_coeff = substitute_symbol(&factor_atom, selected_symbol, Atom::num(0i64));
    let p_at_one = substitute_symbol(&factor_atom, selected_symbol, Atom::num(1i64));
    let a_coeff = p_at_one.sub(&b_coeff).expand();

    if is_zero_exact(&a_coeff, &loaded.sym_to_vid, &loaded.id_to_sym).unwrap_or(true) {
        return Err(ParameterizationError::NonlinearSelectedFactor { variable });
    }

    let rhs = b_coeff
        .div(&Atom::num(0i64).sub(&a_coeff).expand())
        .expand();
    let denominator_nonzero =
        rhs_denominator_nonzero_exact(&rhs, &loaded.sym_to_vid, &loaded.id_to_sym).unwrap_or(false);
    if !denominator_nonzero {
        return Err(ParameterizationError::DegenerateSubstitution { variable });
    }

    Ok(rhs)
}

fn insert_cover_binding(
    loaded: &LoadedMathematicaSystem,
    substitution: &mut SubstitutionMap,
    assigned: &mut HashSet<VarId>,
    var: VarId,
    rhs: Atom,
) -> Result<(), ParameterizationError> {
    if !assigned.insert(var) {
        return Err(ParameterizationError::DuplicateBinding {
            variable: var_name(loaded, var.0 as usize).to_string(),
        });
    }
    substitution.insert(var, rhs);
    Ok(())
}

pub fn cover_substitution(
    loaded: &LoadedMathematicaSystem,
    data: &ProblemData,
    cover: &Cover,
) -> Result<SubstitutionMap, ParameterizationError> {
    let mut substitution = SubstitutionMap::new();
    let mut assigned = HashSet::new();

    for &local_var in &cover.zeroed_locals {
        let var = var_id_from_global(data.v_ids[local_var]);
        insert_cover_binding(
            loaded,
            &mut substitution,
            &mut assigned,
            var,
            Atom::num(0i64),
        )?;
    }

    for &(factor_index, selected_vid, _) in &cover.selected {
        let var = var_id_from_global(selected_vid);
        let fi = &data.factors[factor_index];
        let max_deg = fi
            .vars
            .iter()
            .position(|&v| v == selected_vid)
            .map_or(0u16, |slot| {
                fi.monomial_degrees
                    .iter()
                    .map(|degs| degs[slot])
                    .max()
                    .unwrap_or(0)
            });
        if max_deg > 1 {
            return Err(ParameterizationError::AlgebraicSubstitutionNotImplemented {
                variable: var_name(loaded, selected_vid).to_string(),
                degree: max_deg,
            });
        }
        let rhs = linear_factor_rhs(loaded, &fi.poly, selected_vid)?;
        insert_cover_binding(loaded, &mut substitution, &mut assigned, var, rhs)?;
    }

    Ok(substitution)
}

pub fn expand_cover_substitution(
    substitution: &SubstitutionMap,
    loaded: &LoadedMathematicaSystem,
) -> Result<(Vec<VarId>, HashMap<VarId, Atom>), ParameterizationError> {
    let order = substitution
        .replay_order(&loaded.sym_to_vid)
        .map_err(|_| ParameterizationError::SubstitutionCycle)?;
    let mut expanded: HashMap<VarId, Atom> = HashMap::new();

    for &var in &order {
        let mut rhs = substitution
            .get(var)
            .expect("replay_order returned a variable missing from substitution")
            .clone();
        for &dependency in &order {
            if dependency == var {
                break;
            }
            if let Some(dependency_rhs) = expanded.get(&dependency) {
                let dependency_symbol = loaded.id_to_sym[dependency.0 as usize];
                rhs = substitute_symbol(&rhs, dependency_symbol, dependency_rhs.clone());
            }
        }
        expanded.insert(var, rhs.expand());
    }

    Ok((order, expanded))
}

pub fn check_cover_parameterizes_original(
    loaded: &LoadedMathematicaSystem,
    data: &ProblemData,
    cover: &Cover,
) -> Result<ParameterizationReport, ParameterizationError> {
    let substitution = cover_substitution(loaded, data, cover)?;
    let binding_count = substitution.len();
    let (order, expanded) = expand_cover_substitution(&substitution, loaded)?;

    for (equation_index, equation) in loaded.system.equations.iter().enumerate() {
        let mut residual = equation.clone();
        for &var in &order {
            let Some(rhs) = expanded.get(&var) else {
                continue;
            };
            let symbol = loaded.id_to_sym[var.0 as usize];
            residual = substitute_symbol(&residual, symbol, rhs.clone());
        }
        residual = residual.together().expand();

        let vanishes = is_numerator_zero_exact(&residual, &loaded.sym_to_vid, &loaded.id_to_sym)
            .map_err(|source| ParameterizationError::ExactCheck {
                equation_index,
                source,
            })?;
        if !vanishes {
            return Err(ParameterizationError::EquationDoesNotVanish {
                equation_index,
                residual,
            });
        }
    }

    Ok(ParameterizationReport {
        checked_equations: loaded.system.equations.len(),
        binding_count,
    })
}

#[derive(Debug, Eq, PartialEq)]
pub struct PreprocessedCheckReport {
    pub checked_factor_groups: usize,
    pub uncovered_count: usize,
}

/// For each preprocessed factor group, verify that at least one factor vanishes
/// after applying the cover substitutions. This checks correctness of the MILP
/// solution on the preprocessed system rather than the original equations.
pub fn check_cover_preprocessed(
    loaded: &LoadedMathematicaSystem,
    data: &ProblemData,
    cover: &Cover,
) -> Result<PreprocessedCheckReport, ParameterizationError> {
    let substitution = cover_substitution(loaded, data, cover)?;
    let (order, expanded) = expand_cover_substitution(&substitution, loaded)?;

    let mut uncovered_count = 0;

    for (poly_idx, factor_indices) in data.f_by_poly.iter().enumerate() {
        let mut covered = false;
        for &fi in factor_indices {
            let factor_atom = factor_to_atom(&data.factors[fi].poly)?;
            let mut residual = factor_atom;
            for &var in &order {
                let Some(rhs) = expanded.get(&var) else {
                    continue;
                };
                let symbol = loaded.id_to_sym[var.0 as usize];
                residual = substitute_symbol(&residual, symbol, rhs.clone());
            }
            residual = residual.together().expand();

            let vanishes =
                is_numerator_zero_exact(&residual, &loaded.sym_to_vid, &loaded.id_to_sym).map_err(
                    |source| ParameterizationError::ExactCheck {
                        equation_index: poly_idx,
                        source,
                    },
                )?;
            if vanishes {
                covered = true;
                break;
            }
        }
        if !covered {
            uncovered_count += 1;
            eprintln!(
                "preprocessed cover check failed: factor group {poly_idx} is not covered after substitution"
            );
        }
    }

    Ok(PreprocessedCheckReport {
        checked_factor_groups: data.f_by_poly.len(),
        uncovered_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::factor_prep::preprocess;
    use crate::mathematica::load_mathematica_system_from_str_with_var_names;
    use crate::milp_cover::milp::solve_milp;
    use crate::milp_cover::problem::build_problem;
    use std::sync::Once;
    use symbolica::LicenseManager;
    use symbolica::domains::algebraic_number::AlgebraicExtension;
    use symbolica::domains::rational::Q;

    static LICENSE: Once = Once::new();
    fn activate_license() {
        LICENSE.call_once(|| {
            let _ = LicenseManager::set_license_key(
                "5568d092#6bb8b4fb#ff936916-65ba-552f-b3c1-bdc64597cd21",
            );
        });
    }

    fn problem_from(source: &str) -> (LoadedMathematicaSystem, ProblemData) {
        activate_license();
        let loaded = load_mathematica_system_from_str_with_var_names(source).expect("load");
        let field = AlgebraicExtension::new_complex(Q);
        let pre = preprocess(&loaded, &field, true).expect("preprocess");
        let data = build_problem(pre, &loaded);
        (loaded, data)
    }

    #[test]
    fn parameterization_check_accepts_independent_linear_system() {
        let (loaded, data) = problem_from("{p -> x + 1, q -> y + 2}");
        let cover = solve_milp(&data).expect("feasible");

        let report = check_cover_parameterizes_original(&loaded, &data, &cover)
            .expect("cover should parameterize original equations");
        assert_eq!(report.checked_equations, 2);
        assert_eq!(report.binding_count, 2);
    }

    #[test]
    fn parameterization_check_expands_dependency_chain() {
        let (loaded, data) = problem_from("{p -> x + y, q -> y + 1}");
        let cover = solve_milp(&data).expect("feasible");

        let report = check_cover_parameterizes_original(&loaded, &data, &cover)
            .expect("dependent substitutions should be expanded before validation");
        assert_eq!(report.checked_equations, 2);
    }

    #[test]
    fn parameterization_check_rejects_cover_that_leaves_original_equation_nonzero() {
        let (loaded, data) = problem_from("{p -> x + 1}");
        let cover = Cover {
            selected: Vec::new(),
            zeroed_locals: Vec::new(),
            zeroed_polynomials: Vec::new(),
        };

        let err = check_cover_parameterizes_original(&loaded, &data, &cover)
            .expect_err("empty cover must not parameterize x + 1 = 0");
        assert!(matches!(
            err,
            ParameterizationError::EquationDoesNotVanish {
                equation_index: 0,
                ..
            }
        ));
    }

    #[test]
    fn parameterization_check_accepts_zeroed_cover() {
        let (loaded, data) = problem_from("{x == 0}");
        let x_vid = loaded
            .var_names
            .iter()
            .position(|name| name == "x")
            .expect("x should be loaded");
        let x_local = data
            .v_idx
            .get(&x_vid)
            .copied()
            .expect("x should appear in the problem");
        let cover = Cover {
            selected: Vec::new(),
            zeroed_locals: vec![x_local],
            zeroed_polynomials: vec![0],
        };

        let report = check_cover_parameterizes_original(&loaded, &data, &cover)
            .expect("x -> 0 should satisfy x = 0");
        assert_eq!(report.checked_equations, 1);
        assert_eq!(report.binding_count, 1);
    }
}
