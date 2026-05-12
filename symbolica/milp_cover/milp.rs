//! MILP model construction and solving via `good_lp` / HiGHS.

use good_lp::{
    Expression, ProblemVariables, ResolutionError, Solution, SolverModel, Variable, constraint,
    default_solver, variable, variables,
};

use crate::milp_cover::problem::ProblemData;

/// Result of solving the MILP.
#[cfg_attr(test, derive(Debug))]
pub struct Cover {
    /// Selected pairs as `(factor_idx, global_var_id, topo_position)` in
    /// topological order.
    pub selected: Vec<(usize, usize, f64)>,
    /// Local variable indices (into `data.v_ids`) zeroed by the solution.
    pub zeroed_locals: Vec<usize>,
    /// Original-polynomial indices made identically zero by the zeroing alone
    /// (i.e. `y_p > 0.5` in the MILP solution).
    pub zeroed_polynomials: Vec<usize>,
}

/// All MILP decision/auxiliary variables, indexed parallel to the data
/// structures in [`ProblemData`].
struct MilpVariables {
    s: Vec<Variable>,
    z: Vec<Variable>,
    zeta: Vec<Variable>,
    y: Vec<Variable>,
    a: Vec<Vec<Variable>>,
    /// C2d: per-admissible-pair survival indicators for degree-0 monomials.
    b: Vec<Vec<Variable>>,
    t: Vec<Variable>,
}

fn add_milp_variables(vars: &mut ProblemVariables, data: &ProblemData) -> MilpVariables {
    let n = data.v_ids.len() as f64;
    let unit = || variable().min(0.0).max(1.0);

    let s: Vec<_> = data
        .admissible
        .iter()
        .map(|_| vars.add(variable().binary()))
        .collect();
    let z: Vec<_> = data
        .v_ids
        .iter()
        .map(|_| vars.add(variable().binary()))
        .collect();
    let zeta: Vec<_> = data.factors.iter().map(|_| vars.add(unit())).collect();
    let y: Vec<_> = data.f_by_poly.iter().map(|_| vars.add(unit())).collect();
    let a: Vec<Vec<_>> = data
        .deg1to4_monomials
        .iter()
        .map(|ms| ms.iter().map(|_| vars.add(unit())).collect())
        .collect();
    let b: Vec<Vec<_>> = data
        .deg0_monomials
        .iter()
        .map(|ms| ms.iter().map(|_| vars.add(unit())).collect())
        .collect();
    let t: Vec<_> = data
        .v_ids
        .iter()
        .map(|_| vars.add(variable().min(0.0).max(n)))
        .collect();

    MilpVariables {
        s,
        z,
        zeta,
        y,
        a,
        b,
        t,
    }
}

fn sum_vars(vars: &[Variable]) -> Expression {
    vars.iter().map(|v| Expression::from(*v)).sum()
}

fn objective(vars: &MilpVariables) -> Expression {
    let n = (vars.z.len() + vars.s.len()) as f64;
    // eps*(|z|+|s|) < 1: secondary tiebreaker never overrides a 1-unit primary difference.
    let eps = 1.0 / (n + 1.0);
    let primary = sum_vars(&vars.y);
    let secondary = sum_vars(&vars.z) + sum_vars(&vars.s);
    primary - eps * secondary
}

fn add_factor_zero_constraints<M: SolverModel>(
    mut model: M,
    data: &ProblemData,
    vars: &MilpVariables,
) -> M {
    for (fi, info) in data.factors.iter().enumerate() {
        for monomial in &info.monomial_vars {
            let zeroed: Expression = monomial
                .iter()
                .filter_map(|vid| {
                    data.v_idx
                        .get(vid)
                        .map(|&vloc| Expression::from(vars.z[vloc]))
                })
                .sum();
            model = model.with(constraint!(vars.zeta[fi] - zeroed <= 0));
        }
    }
    model
}

fn add_polynomial_zero_link<M: SolverModel>(
    mut model: M,
    data: &ProblemData,
    vars: &MilpVariables,
) -> M {
    for (p, fs) in data.f_by_poly.iter().enumerate() {
        let zeta_sum: Expression = fs.iter().map(|&fi| Expression::from(vars.zeta[fi])).sum();
        model = model.with(constraint!(vars.y[p] - zeta_sum <= 0));
    }
    model
}

fn add_coverage_constraints<M: SolverModel>(
    mut model: M,
    data: &ProblemData,
    vars: &MilpVariables,
) -> Result<M, ResolutionError> {
    for (p, fs) in data.f_by_poly.iter().enumerate() {
        if fs.is_empty() {
            eprintln!("error: polynomial {p} has no factors. The instance is infeasible.");
            return Err(ResolutionError::Infeasible);
        }
        let zeta_sum: Expression = fs.iter().map(|&fi| Expression::from(vars.zeta[fi])).sum();
        let s_sum: Expression = data
            .admissible
            .iter()
            .enumerate()
            .filter(|(_, (fi, _))| fs.contains(fi))
            .map(|(k, _)| Expression::from(vars.s[k]))
            .sum();
        model = model.with(constraint!(zeta_sum + s_sum >= 1));
    }
    Ok(model)
}

fn add_linearity_constraints<M: SolverModel>(
    mut model: M,
    data: &ProblemData,
    vars: &MilpVariables,
) -> M {
    for (k, &(fi, vloc)) in data.admissible.iter().enumerate() {
        let info = &data.factors[fi];
        let selected_vid = data.v_ids[vloc];

        // C2a: every monomial of degree >= 5 in the selected variable must
        // have one of its non-selected variables zeroed.
        for (m, monomial) in info.monomial_vars.iter().enumerate() {
            if info.monomial_degree(m, selected_vid) < 5 {
                continue;
            }
            let killers: Expression = monomial
                .iter()
                .filter(|&&vid| vid != selected_vid)
                .filter_map(|vid| data.v_idx.get(vid).map(|&vl| Expression::from(vars.z[vl])))
                .sum();
            model = model.with(constraint!(vars.s[k] - killers <= 0));
        }

        // C2b: a monomial of degree 1-4 in x survives only if none of its
        // non-selected variables are zeroed.
        for (slot, &m) in data.deg1to4_monomials[k].iter().enumerate() {
            for &vid in &info.monomial_vars[m] {
                if vid == selected_vid {
                    continue;
                }
                let Some(&vl) = data.v_idx.get(&vid) else {
                    continue;
                };
                model = model.with(constraint!(vars.a[k][slot] + vars.z[vl] <= 1));
            }
        }

        // C2c: at least one monomial of degree 1-4 must survive.
        let surviving = sum_vars(&vars.a[k]);
        model = model.with(constraint!(vars.s[k] - surviving <= 0));
    }
    model
}

/// C2d: for each selected pair `(f, x)`, at least one degree-0-in-x monomial of `f`
/// (the "constant term" w.r.t. x, i.e. f(x=0)) must not be entirely zeroed. This
/// guarantees x=0 is not a root of f after zeroing, preventing the defined variable
/// from taking the trivial value zero through the substitution chain.
fn add_nonzero_rhs_constraints<M: SolverModel>(
    mut model: M,
    data: &ProblemData,
    vars: &MilpVariables,
) -> M {
    for (k, &(fi, _)) in data.admissible.iter().enumerate() {
        let info = &data.factors[fi];
        for (slot, &m) in data.deg0_monomials[k].iter().enumerate() {
            for &vid in &info.monomial_vars[m] {
                let Some(&vl) = data.v_idx.get(&vid) else {
                    continue;
                };
                model = model.with(constraint!(vars.b[k][slot] + vars.z[vl] <= 1));
            }
        }
        let surviving = sum_vars(&vars.b[k]);
        model = model.with(constraint!(vars.s[k] - surviving <= 0));
    }
    model
}

fn add_acyclicity_constraints<M: SolverModel>(
    mut model: M,
    data: &ProblemData,
    vars: &MilpVariables,
) -> M {
    let big_m = (data.v_ids.len() as f64) + 1.0;
    for (k, &(fi, vloc)) in data.admissible.iter().enumerate() {
        let info = &data.factors[fi];
        let selected_vid = data.v_ids[vloc];
        for &other_vid in &info.vars {
            if other_vid == selected_vid {
                continue;
            }
            let Some(&other_vloc) = data.v_idx.get(&other_vid) else {
                continue;
            };
            model = model.with(constraint!(
                vars.t[other_vloc] + 1.0
                    <= vars.t[vloc] + big_m * (1.0 - Expression::from(vars.s[k]))
            ));
        }
    }
    model
}

fn add_distinctness_constraints<M: SolverModel>(
    mut model: M,
    data: &ProblemData,
    vars: &MilpVariables,
) -> M {
    for (vloc, _) in data.v_ids.iter().enumerate() {
        let s_sum: Expression = data
            .admissible
            .iter()
            .enumerate()
            .filter(|(_, (_, vl))| *vl == vloc)
            .map(|(k, _)| Expression::from(vars.s[k]))
            .sum();
        model = model.with(constraint!(s_sum + vars.z[vloc] <= 1));
    }
    model
}

fn cover_from_solution<S: Solution>(
    solution: &S,
    data: &ProblemData,
    vars: &MilpVariables,
) -> Cover {
    let mut selected: Vec<(usize, usize, f64)> = data
        .admissible
        .iter()
        .enumerate()
        .filter(|(k, _)| solution.value(vars.s[*k]) > 0.5)
        .map(|(_, &(fi, vloc))| {
            let vid = data.v_ids[vloc];
            (fi, vid, solution.value(vars.t[vloc]))
        })
        .collect();
    selected.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

    let zeroed_locals: Vec<usize> = (0..data.v_ids.len())
        .filter(|&i| solution.value(vars.z[i]) > 0.5)
        .collect();
    let zeroed_polynomials: Vec<usize> = (0..data.f_by_poly.len())
        .filter(|&p| solution.value(vars.y[p]) > 0.5)
        .collect();

    Cover {
        selected,
        zeroed_locals,
        zeroed_polynomials,
    }
}

pub fn solve_milp(data: &ProblemData) -> Result<Cover, ResolutionError> {
    let mut problem_vars = variables!();
    let m = add_milp_variables(&mut problem_vars, data);

    let model = problem_vars.maximise(objective(&m)).using(default_solver);
    let model = add_factor_zero_constraints(model, data, &m);
    let model = add_polynomial_zero_link(model, data, &m);
    let model = add_coverage_constraints(model, data, &m)?;
    let model = add_linearity_constraints(model, data, &m);
    let model = add_nonzero_rhs_constraints(model, data, &m);
    let model = add_acyclicity_constraints(model, data, &m);
    let model = add_distinctness_constraints(model, data, &m);
    let solution = model.solve()?;

    Ok(cover_from_solution(&solution, data, &m))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::factor_prep::{ideal_ext_lex, preprocess};
    use crate::mathematica::LoadedMathematicaSystem;
    use crate::mathematica::load_mathematica_system_from_str_with_var_names;
    use crate::milp_cover::problem::build_problem;
    use std::collections::HashMap;
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

    fn topo_ok(data: &ProblemData, cover: &Cover) -> bool {
        let topo: HashMap<usize, f64> =
            cover.selected.iter().map(|&(_, vid, t)| (vid, t)).collect();
        for &(fi, vid, ti) in &cover.selected {
            for &other_vid in &data.factors[fi].vars {
                if other_vid == vid {
                    continue;
                }
                let Some(&tj) = topo.get(&other_vid) else {
                    continue;
                };
                if tj + 1.0 > ti + 1e-6 {
                    return false;
                }
            }
        }
        true
    }

    #[test]
    fn solve_covers_every_polynomial_linearly() {
        let (_loaded, data) = problem_from("{p -> x + 1, q -> y + 2, r -> z + 3}");
        let cover = solve_milp(&data).expect("feasible");
        assert_eq!(cover.selected.len(), 3);
        for j in 0..data.f_by_poly.len() {
            let covered = cover
                .selected
                .iter()
                .any(|(fi, _, _)| data.f_by_poly[j].contains(fi));
            assert!(covered, "polynomial {j} is not covered");
        }
        assert!(topo_ok(&data, &cover));
    }

    #[test]
    fn solve_respects_acyclicity() {
        let (_loaded, data) = problem_from("{p -> x + 1, q -> x*y + x^2}");
        let cover = solve_milp(&data).expect("feasible");
        assert!(topo_ok(&data, &cover));
        for j in 0..data.f_by_poly.len() {
            let by_selection = cover
                .selected
                .iter()
                .any(|(fi, _, _)| data.f_by_poly[j].contains(fi));
            let by_zeroing = cover.zeroed_polynomials.contains(&j);
            assert!(by_selection || by_zeroing, "polynomial {j} is not covered");
        }
    }

    #[test]
    fn solve_zero_kills_polynomial_with_unkillable_constant() {
        let (_loaded, data) = problem_from("{x^5 == 0}");
        let cover = solve_milp(&data).expect("feasible by zeroing x");
        assert_eq!(cover.zeroed_polynomials.len(), 1);
        assert_eq!(cover.zeroed_locals.len(), 1);
    }

    #[test]
    fn solve_prefers_zeroing_when_it_satisfies_a_polynomial() {
        let (_loaded, data) = problem_from_raw_factors("{y == 0, x + z == 0}");
        let cover = solve_milp(&data).expect("feasible");
        assert!(
            cover.zeroed_polynomials.contains(&0),
            "polynomial 0 (y == 0) should be marked zeroed; cover = {cover:?}"
        );
        assert!(topo_ok(&data, &cover));
    }

    #[test]
    fn solve_does_not_select_pair_with_unkillable_degree5_monomial() {
        let (loaded, data) = problem_from("{(x + 1) * (x^5 + y + 1) == 0}");
        let x_vid = loaded.var_names.iter().position(|n| n == "x").unwrap();
        let x_local = data.v_idx[&x_vid];
        for &(fi, vloc) in &data.admissible {
            if vloc != x_local {
                continue;
            }
            assert_eq!(
                data.factors[fi].vars.len(),
                1,
                "only the linear factor (x+1) should be admissible for x"
            );
        }
        let _cover = solve_milp(&data).expect("feasible");
    }

    #[test]
    fn solve_selects_degree_two_factor_when_needed() {
        let (loaded, data) = problem_from("{x^2 + z == 0, z + 1 == 0}");
        let cover = solve_milp(&data).expect("feasible");
        assert!(topo_ok(&data, &cover));
        for j in 0..data.f_by_poly.len() {
            let by_selection = cover
                .selected
                .iter()
                .any(|(fi, _, _)| data.f_by_poly[j].contains(fi));
            let by_zeroing = cover.zeroed_polynomials.contains(&j);
            assert!(by_selection || by_zeroing, "polynomial {j} is not covered");
        }
        let x_vid = loaded.var_names.iter().position(|n| n == "x").unwrap();
        assert!(
            cover.selected.iter().any(|&(_, vid, _)| vid == x_vid),
            "x should be a selected (defined) variable; cover = {cover:?}"
        );
    }

    #[test]
    fn solve_degree_five_monomial_not_admissible_for_that_variable() {
        let (loaded, data) = problem_from_raw_factors("{x^5 + y == 0}");
        let x_vid = loaded.var_names.iter().position(|n| n == "x").unwrap();
        let x_local = *data.v_idx.get(&x_vid).expect("x in problem");
        for &(_, vloc) in &data.admissible {
            assert_ne!(
                vloc, x_local,
                "x must not be admissible (degree-5 only in x)"
            );
        }
        let cover = solve_milp(&data).expect("feasible");
        assert!(
            !cover.selected.iter().any(|&(_, vid, _)| vid == x_vid),
            "x must not be selected as a defined variable"
        );
    }
}
