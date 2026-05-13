//! Optional symbolic certification (see `problem statement.md`).
//!
//! Given an MILP cover, substitute the zeroed variables into the selected
//! factors `{f_1, ..., f_k}` and compute the lex Gröbner basis with variable
//! order `x_k > x_{k-1} > ... > x_1 > (free parameters)`.
//!
//! Acyclicity of the dependency graph guarantees that `{f_1, ..., f_k}` is
//! already a triangular set under this order; the Gröbner basis therefore
//! exposes (generically) the elimination shape
//! `{g_1(x_1, params), g_2(x_1, x_2, params), ...}`, from which non-emptiness
//! of the parametrized component can be read off.

use std::collections::HashSet;
use std::sync::Arc;

use symbolica::atom::{Atom, AtomCore};
use symbolica::poly::PolyVariable;
use thiserror::Error;

use crate::atom_util::substitute_symbol;
use crate::groebner::{QPolyLex, groebner_basis_from_system_lex_q};
use crate::mathematica::LoadedMathematicaSystem;
use crate::milp_cover::milp::Cover;
use crate::milp_cover::parameterization::{ParameterizationError, factor_to_atom};
use crate::milp_cover::problem::{ProblemData, var_name};
use crate::system::PolynomialSystem;

#[derive(Debug, Error)]
pub enum CertificationError {
    #[error("failed to convert selected factor to atom: {0}")]
    FactorAtom(#[from] ParameterizationError),
    #[error("Gröbner basis over ℚ failed (likely non-rational coefficients): {0}")]
    GroebnerOverQ(String),
}

pub struct CertificationReport {
    /// Names of variables in the lex order used (descending, i.e. the first is
    /// the largest variable in the lex order).
    pub var_order_names: Vec<String>,
    /// Names of the selected variables `x_1, ..., x_k` (in topological order).
    pub selected_names: Vec<String>,
    /// Names of the variables zeroed by the cover (substituted out).
    pub zeroed_names: Vec<String>,
    /// Names of the free parameters: variables appearing in the selected
    /// factors after substitution that are neither selected nor zeroed.
    pub free_parameter_names: Vec<String>,
    /// The selected factors, after substituting zeroed variables to 0.
    pub substituted_factors: Vec<Atom>,
    /// The lex Gröbner basis of `substituted_factors` under `var_order_names`.
    pub basis: Vec<QPolyLex>,
}

/// Run symbolic certification on the cover's selected factors.
pub fn certify_cover_groebner(
    loaded: &LoadedMathematicaSystem,
    data: &ProblemData,
    cover: &Cover,
) -> Result<CertificationReport, CertificationError> {
    let zeroed_vids: HashSet<usize> = cover
        .zeroed_locals
        .iter()
        .map(|&local| data.v_ids[local])
        .collect();
    let zeroed_names: Vec<String> = zeroed_vids
        .iter()
        .map(|&vid| var_name(loaded, vid).to_string())
        .collect();

    let substituted_factors: Vec<Atom> = cover
        .selected
        .iter()
        .map(|&(factor_idx, _, _)| substitute_zeroed(loaded, data, factor_idx, &zeroed_vids))
        .collect::<Result<_, _>>()?;

    let selected_vids: Vec<usize> = cover.selected.iter().map(|&(_, vid, _)| vid).collect();
    let selected_set: HashSet<usize> = selected_vids.iter().copied().collect();
    let selected_names: Vec<String> = selected_vids
        .iter()
        .map(|&vid| var_name(loaded, vid).to_string())
        .collect();

    let free_vids = free_parameter_vids(data, cover, &selected_set, &zeroed_vids);
    let free_parameter_names: Vec<String> = free_vids
        .iter()
        .map(|&vid| var_name(loaded, vid).to_string())
        .collect();

    let ordered_vids: Vec<usize> = selected_vids
        .iter()
        .rev()
        .copied()
        .chain(free_vids.iter().copied())
        .collect();
    let var_order: Arc<Vec<PolyVariable>> = Arc::new(
        ordered_vids
            .iter()
            .map(|&vid| PolyVariable::from(loaded.id_to_sym[vid]))
            .collect(),
    );
    let var_order_names: Vec<String> = ordered_vids
        .iter()
        .map(|&vid| var_name(loaded, vid).to_string())
        .collect();

    let basis = if substituted_factors.is_empty() {
        Vec::new()
    } else {
        let system = PolynomialSystem::new(substituted_factors.clone());
        groebner_basis_from_system_lex_q(&system, var_order, false)
            .map_err(CertificationError::GroebnerOverQ)?
            .system
    };

    Ok(CertificationReport {
        var_order_names,
        selected_names,
        zeroed_names,
        free_parameter_names,
        substituted_factors,
        basis,
    })
}

fn substitute_zeroed(
    loaded: &LoadedMathematicaSystem,
    data: &ProblemData,
    factor_idx: usize,
    zeroed_vids: &HashSet<usize>,
) -> Result<Atom, CertificationError> {
    let zero = Atom::num(0i64);
    let mut atom = factor_to_atom(&data.factors[factor_idx].poly)?;
    for &zvid in zeroed_vids {
        let zsym = loaded.id_to_sym[zvid];
        atom = substitute_symbol(&atom, zsym, zero.clone());
    }
    Ok(atom.expand())
}

fn free_parameter_vids(
    data: &ProblemData,
    cover: &Cover,
    selected_set: &HashSet<usize>,
    zeroed_vids: &HashSet<usize>,
) -> Vec<usize> {
    let mut free: Vec<usize> = cover
        .selected
        .iter()
        .flat_map(|&(fi, _, _)| data.factors[fi].vars.iter().copied())
        .filter(|vid| !selected_set.contains(vid) && !zeroed_vids.contains(vid))
        .collect();
    free.sort_unstable();
    free.dedup();
    free
}

pub fn print_certification(report: &CertificationReport) {
    println!("\nSymbolic certification (lex Gröbner basis on selected factors)");
    println!(
        "  Variable order (lex, descending): {}",
        report.var_order_names.join(" > ")
    );
    println!("  Selected variables       X = {:?}", report.selected_names);
    println!("  Zeroed (substituted)     W = {:?}", report.zeroed_names);
    println!(
        "  Free parameters in factors Y = {:?}",
        report.free_parameter_names
    );

    println!(
        "\n  Substituted selected factors ({}):",
        report.substituted_factors.len()
    );
    for (i, factor) in report.substituted_factors.iter().enumerate() {
        println!("    f_{} = {}", i + 1, factor);
    }

    println!(
        "\n  Lex Gröbner basis ({} polynomial(s)):",
        report.basis.len()
    );
    if report.basis.is_empty() {
        println!("    (empty — no factors were selected by the cover)");
        return;
    }
    for (i, g) in report.basis.iter().enumerate() {
        println!("    g_{} = {}", i + 1, g);
    }

    let is_unit_ideal = report.basis.iter().any(|g| g.is_constant() && !g.is_zero());
    if is_unit_ideal {
        println!(
            "\n  Non-emptiness check: basis contains a non-zero constant ⇒ the ideal is ⟨1⟩ \
             (the system has NO common solution; the parametrization is spurious)."
        );
    } else {
        println!(
            "\n  Non-emptiness check: basis contains no non-zero constant ⇒ the ideal is proper \
             and the parametrized component is non-empty over the algebraic closure."
        );
    }
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

    fn problem_from(source: &str) -> (LoadedMathematicaSystem, ProblemData, Cover) {
        activate_license();
        let loaded = load_mathematica_system_from_str_with_var_names(source).expect("load");
        let field = AlgebraicExtension::new_complex(Q);
        let pre = preprocess(&loaded, &field).expect("preprocess");
        let data = build_problem(pre, &loaded);
        let cover = solve_milp(&data).expect("feasible");
        (loaded, data, cover)
    }

    #[test]
    fn certification_runs_on_independent_linear_system() {
        let (loaded, data, cover) = problem_from("{x + 1 == 0, y + 2 == 0}");
        let report = certify_cover_groebner(&loaded, &data, &cover).expect("certification");
        assert_eq!(report.basis.len(), 2);
        assert!(!report.basis.iter().any(|g| g.is_constant() && !g.is_zero()));
    }

    #[test]
    fn certification_reports_triangular_basis_for_chain() {
        let (loaded, data, cover) = problem_from("{x + y == 0, y + 1 == 0}");
        let report = certify_cover_groebner(&loaded, &data, &cover).expect("certification");
        assert!(!report.basis.is_empty());
        // Lex order is y > x (y was the last in topo: defined from x->y via x+y=0?)
        // Either way, the basis must define both variables; ideal is proper.
        assert!(!report.basis.iter().any(|g| g.is_constant() && !g.is_zero()));
    }

    #[test]
    fn certification_with_only_zeroing_produces_empty_basis() {
        let (loaded, data, cover) = problem_from("{x == 0}");
        let report = certify_cover_groebner(&loaded, &data, &cover).expect("certification");
        assert!(report.basis.is_empty());
    }
}
