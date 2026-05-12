//! Problem-data assembly from preprocessed factor groups. No `good_lp`, no IO.

use std::collections::HashMap;

use symbolica::atom::Symbol;
use symbolica::poly::PolyVariable;

use crate::factor_prep::PolyExtLex;
use crate::mathematica::LoadedMathematicaSystem;
use crate::var_id::VarId;

/// A single square-free factor together with the global-variable ids it
/// depends on and per-monomial variable lists.
pub struct FactorInfo {
    pub poly: PolyExtLex,
    /// Global variable ids that occur in `poly`.
    pub vars: Vec<usize>,
    /// Per-monomial: the list of global variable ids in the monomial.
    pub monomial_vars: Vec<Vec<usize>>,
    /// Per-monomial: the per-variable degree, parallel to `monomial_vars` and
    /// keyed by the position in `vars`.
    pub monomial_degrees: Vec<Vec<u16>>,
}

impl FactorInfo {
    /// Indices of monomials whose degree in the global variable `vid` is exactly `target_degree`.
    pub fn monomials_deg(&self, vid: usize, target_degree: u16) -> Vec<usize> {
        let Some(slot) = self.vars.iter().position(|v| *v == vid) else {
            return Vec::new();
        };
        self.monomial_degrees
            .iter()
            .enumerate()
            .filter_map(|(idx, degs)| (degs[slot] == target_degree).then_some(idx))
            .collect()
    }

    pub fn monomial_degree(&self, monomial: usize, vid: usize) -> u16 {
        let Some(slot) = self.vars.iter().position(|v| *v == vid) else {
            return 0;
        };
        self.monomial_degrees[monomial][slot]
    }
}

/// Problem data assembled from the preprocessed factor groups.
pub struct ProblemData {
    pub factors: Vec<FactorInfo>,
    /// For each original (non-zero, non-trivial) polynomial, the indices of
    /// its distinct square-free factors in `factors`.
    pub f_by_poly: Vec<Vec<usize>>,
    /// Global variable ids that appear in at least one factor.
    pub v_ids: Vec<usize>,
    /// Inverse of `v_ids`.
    pub v_idx: HashMap<usize, usize>,
    /// Admissible pairs `(factor_idx, local_var_idx)`: those for which the
    /// factor has at least one monomial of degree between 1 and 4 in the variable.
    pub admissible: Vec<(usize, usize)>,
    /// For each admissible pair, the indices into the factor's `monomial_vars`
    /// of monomials of degree 1 to 4 in the selected variable (i.e. `M^{[1,4]}_{f,x}`).
    pub deg1to4_monomials: Vec<Vec<usize>>,
    /// For each admissible pair, the indices into the factor's `monomial_vars`
    /// of monomials of degree exactly 0 in the selected variable (the "constant term"
    /// w.r.t. that variable). Used by C2d.
    pub deg0_monomials: Vec<Vec<usize>>,
}

fn factor_info(f: PolyExtLex, sym_to_vid: &HashMap<Symbol, VarId>) -> FactorInfo {
    let local_to_global: Vec<Option<usize>> = f
        .get_vars_ref()
        .iter()
        .map(|pv| match pv {
            PolyVariable::Symbol(sym) => sym_to_vid.get(sym).map(|vid| vid.0 as usize),
            _ => None,
        })
        .collect();

    let mut vars: Vec<usize> = (0..f.get_vars_ref().len())
        .filter(|&xi| f.contains(xi))
        .filter_map(|xi| local_to_global[xi])
        .collect();
    vars.sort_unstable();
    vars.dedup();

    let mut monomial_vars = Vec::new();
    let mut monomial_degrees = Vec::new();
    for exponents in f.exponents_iter() {
        let mut mv = Vec::new();
        let mut mdeg = vec![0u16; vars.len()];
        for (local, exp) in exponents.iter().enumerate() {
            if *exp == 0 {
                continue;
            }
            let Some(global) = local_to_global[local] else {
                continue;
            };
            mv.push(global);
            let slot = vars.iter().position(|v| *v == global).unwrap();
            mdeg[slot] = *exp;
        }
        monomial_vars.push(mv);
        monomial_degrees.push(mdeg);
    }

    FactorInfo {
        poly: f,
        vars,
        monomial_vars,
        monomial_degrees,
    }
}

pub fn build_problem(
    factor_groups: Vec<Vec<PolyExtLex>>,
    loaded: &LoadedMathematicaSystem,
) -> ProblemData {
    let mut factor_index: HashMap<PolyExtLex, usize> = HashMap::new();
    let mut factors: Vec<FactorInfo> = Vec::new();
    let mut f_by_poly: Vec<Vec<usize>> = Vec::with_capacity(factor_groups.len());

    for group in factor_groups {
        let mut ids = Vec::with_capacity(group.len());
        for f in group {
            let canon = f.make_monic();
            let idx = *factor_index.entry(canon.clone()).or_insert_with(|| {
                factors.push(factor_info(canon, &loaded.sym_to_vid));
                factors.len() - 1
            });
            if !ids.contains(&idx) {
                ids.push(idx);
            }
        }
        f_by_poly.push(ids);
    }

    let mut v_set: Vec<usize> = factors
        .iter()
        .flat_map(|fi| fi.vars.iter().copied())
        .collect();
    v_set.sort_unstable();
    v_set.dedup();
    let v_idx: HashMap<usize, usize> = v_set.iter().enumerate().map(|(i, v)| (*v, i)).collect();

    let mut admissible = Vec::new();
    let mut deg1to4_monomials = Vec::new();
    let mut deg0_monomials = Vec::new();
    for (fi, info) in factors.iter().enumerate() {
        if info.monomial_vars.len() == 1 {
            // C5: a single-monomial factor cannot define any variable.
            continue;
        }
        for &vid in &info.vars {
            let m1to4: Vec<usize> = (1u16..=4)
                .flat_map(|d| info.monomials_deg(vid, d))
                .collect();
            if !m1to4.is_empty() {
                admissible.push((fi, v_idx[&vid]));
                deg1to4_monomials.push(m1to4);
                deg0_monomials.push(info.monomials_deg(vid, 0));
            }
        }
    }

    ProblemData {
        factors,
        f_by_poly,
        v_ids: v_set,
        v_idx,
        admissible,
        deg1to4_monomials,
        deg0_monomials,
    }
}

pub fn var_name(loaded: &LoadedMathematicaSystem, global_vid: usize) -> &str {
    &loaded.var_names[global_vid]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::factor_prep::{ideal_ext_lex, preprocess};
    use crate::mathematica::load_mathematica_system_from_str_with_var_names;
    use crate::milp_cover::milp::solve_milp;
    use good_lp::ResolutionError;
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
        let pre = preprocess(&loaded, &field).expect("preprocess");
        let data = build_problem(pre.factor_groups, &loaded);
        (loaded, data)
    }

    fn problem_from_raw_factors(source: &str) -> (LoadedMathematicaSystem, ProblemData) {
        activate_license();
        let loaded = load_mathematica_system_from_str_with_var_names(source).expect("load");
        let field = AlgebraicExtension::new_complex(Q);
        let var_order = crate::factor_prep::variable_order(&loaded.id_to_sym);
        let factor_groups = ideal_ext_lex(&loaded.system, &field, var_order)
            .expect("polynomial conversion")
            .into_iter()
            .map(|poly| vec![poly])
            .collect();
        let data = build_problem(factor_groups, &loaded);
        (loaded, data)
    }

    #[test]
    fn build_problem_dedupes_shared_factors() {
        activate_license();
        let loaded = load_mathematica_system_from_str_with_var_names("{x + 1 == 0, y + 2 == 0}")
            .expect("load");
        let field = AlgebraicExtension::new_complex(Q);
        let mut pre = preprocess(&loaded, &field).expect("preprocess");
        assert_eq!(pre.factor_groups.len(), 2);
        let shared = pre.factor_groups[0][0].clone();
        pre.factor_groups[1].push(shared);

        let data = build_problem(pre.factor_groups, &loaded);
        assert_eq!(
            data.factors.len(),
            2,
            "shared factor must be deduplicated into a single entry in F"
        );
        let shared_idx = data.f_by_poly[0][0];
        assert!(
            data.f_by_poly[1].contains(&shared_idx),
            "dedup must reuse the same factor index across polynomials, got {:?}",
            data.f_by_poly
        );
    }

    #[test]
    fn admissible_requires_bounded_degree_monomial() {
        let (_loaded, data) = problem_from_raw_factors("{x^5 + y^5 == 0}");
        assert_eq!(data.factors.len(), 1);
        assert!(
            data.admissible.is_empty(),
            "no monomial has degree 1-4 in any single variable; got {:?}",
            data.admissible
        );
    }

    #[test]
    fn admissible_excludes_pure_high_degree_polynomial() {
        let (_loaded, data) = problem_from("{x^5 + x + 3 == 0}");
        assert!(!data.admissible.is_empty());
        let cover = solve_milp(&data);
        assert!(matches!(cover, Err(ResolutionError::Infeasible)));
    }

    #[test]
    fn admissible_includes_degree_2_monomials() {
        let (_loaded, data) = problem_from_raw_factors("{x^2 + y^2 + 1 == 0}");
        assert!(
            !data.admissible.is_empty(),
            "degree-2 monomials in x and y should make the pairs admissible"
        );
    }

    #[test]
    fn monomial_factor_not_admissible() {
        let (_loaded, data) = problem_from_raw_factors("{x*y == 0}");
        assert!(
            data.admissible.is_empty(),
            "single-monomial factor must not appear in admissible set; got {:?}",
            data.admissible
        );
    }
}
