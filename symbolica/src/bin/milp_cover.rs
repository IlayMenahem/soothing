//! MILP-based radical cover as described in `problem statement.md`.
//!
//! Preprocess the input polynomial system (interreduce, square-free factor,
//! drop constants/zero polynomials), then solve an integer program with
//! [`good_lp`]/HiGHS to pick tuples `(f_i, x_i)` that cover every polynomial
//! by a radical-solvable, acyclic assignment.
//!
//! Usage: `milp_cover [path]` — default path: `systems/system.txt`.

use std::collections::HashMap;
use std::ops::{Add, Div, Mul, Sub};
use std::path::PathBuf;
use std::time::Instant;

use good_lp::{
    Expression, ProblemVariables, ResolutionError, Solution, SolverModel, Variable, constraint,
    default_solver, variable, variables,
};

use symbolica::LicenseManager;
use symbolica::atom::{Atom, AtomCore, Symbol};
use symbolica::domains::algebraic_number::AlgebraicExtension;
use symbolica::domains::rational::Q;
use symbolica::poly::PolyVariable;

use symbolica_elim::{
    LoadedMathematicaSystem, PolyExtLex,
    atom_util::{parse_expression, substitute_symbol, vars_in_atom},
    default_systems_dir_from_manifest, load_mathematica_system_from_path_with_var_names,
    preprocess, rhs_denominator_nonzero_exact,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CoverObjective {
    Count,
    Degree,
    CountDegree,
}

impl CoverObjective {
    fn label(self) -> &'static str {
        match self {
            CoverObjective::Count => "minimize cover size",
            CoverObjective::Degree => "minimize total degree",
            CoverObjective::CountDegree => "minimize cover size, then total degree",
        }
    }
}

#[derive(Debug)]
struct CliOptions {
    path: PathBuf,
    objective: CoverObjective,
}

fn parse_args(args: impl IntoIterator<Item = String>) -> Result<CliOptions, String> {
    let mut path: Option<PathBuf> = None;
    let mut objective = CoverObjective::Count;
    let mut args = args.into_iter();

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--objective" => match args.next().as_deref() {
                Some("count") => objective = CoverObjective::Count,
                Some("degree") => objective = CoverObjective::Degree,
                Some("count-degree") => objective = CoverObjective::CountDegree,
                other => {
                    return Err(format!(
                        "--objective expects 'count', 'degree', or 'count-degree' (got {other:?})"
                    ));
                }
            },
            _ => path = Some(PathBuf::from(arg)),
        }
    }

    Ok(CliOptions {
        path: path.unwrap_or_else(|| default_systems_dir_from_manifest().join("system.txt")),
        objective,
    })
}

/// A single square-free factor together with the global-variable ids it
/// depends on and their (per-variable) degrees.
struct FactorInfo {
    poly: PolyExtLex,
    /// `(global_var_id, degree)` for every variable occurring in `poly`.
    vars: Vec<(usize, u16)>,
}

#[cfg_attr(test, derive(Debug))]
struct ParametrizationInfo {
    factor_idx: usize,
    solved_vloc: usize,
    solved_vid: usize,
    degree: u16,
    rhs: Atom,
    dependency_vlocs: Vec<usize>,
}

/// Problem data assembled from the preprocessed factor groups.
struct ProblemData {
    factors: Vec<FactorInfo>,
    /// For each original (non-zero, non-trivial) polynomial, the indices of
    /// its distinct square-free factors in `factors`.
    f_by_poly: Vec<Vec<usize>>,
    /// Global variable ids that appear in at least one factor.
    v_ids: Vec<usize>,
    /// Admissible solved parametrizations with `1 <= deg <= 2`.
    admissible: Vec<ParametrizationInfo>,
}

fn factor_vars(
    f: &PolyExtLex,
    sym_to_vid: &HashMap<Symbol, symbolica_elim::VarId>,
) -> Vec<(usize, u16)> {
    let mut out = Vec::new();
    for (xi, pv) in f.get_vars_ref().iter().enumerate() {
        if !f.contains(xi) {
            continue;
        }
        let PolyVariable::Symbol(sym) = pv else {
            continue;
        };
        let Some(&vid) = sym_to_vid.get(sym) else {
            continue;
        };
        out.push((vid.0 as usize, f.degree(xi)));
    }
    out
}

fn poly_to_atom(poly: &PolyExtLex) -> Option<Atom> {
    let rendered = poly.to_string();
    let expression = rendered
        .split_once(" % ")
        .map_or(rendered.as_str(), |(expr, _)| expr);
    parse_expression(expression.to_string()).ok()
}

fn substitute_vid(
    expr: &Atom,
    vid: usize,
    replacement: Atom,
    loaded: &LoadedMathematicaSystem,
) -> Atom {
    substitute_symbol(expr, loaded.id_to_sym[vid], replacement)
}

fn atom_vars_as_vlocs(
    expr: &Atom,
    solved_vid: usize,
    v_idx: &HashMap<usize, usize>,
    loaded: &LoadedMathematicaSystem,
) -> Vec<usize> {
    let mut deps: Vec<usize> = vars_in_atom(expr, &loaded.sym_to_vid)
        .into_iter()
        .map(|vid| vid.0 as usize)
        .filter(|&vid| vid != solved_vid)
        .filter_map(|vid| v_idx.get(&vid).copied())
        .collect();
    deps.sort_unstable();
    deps.dedup();
    deps
}

fn is_valid_rhs(rhs: &Atom, loaded: &LoadedMathematicaSystem) -> bool {
    rhs_denominator_nonzero_exact(rhs, &loaded.sym_to_vid, &loaded.id_to_sym).unwrap_or(true)
}

fn linear_rhs(factor: &Atom, solved_vid: usize, loaded: &LoadedMathematicaSystem) -> Option<Atom> {
    let zero = Atom::num(0i64);
    let one = Atom::num(1i64);
    let b_coeff = substitute_vid(factor, solved_vid, zero, loaded);
    let p_at_one = substitute_vid(factor, solved_vid, one, loaded);
    let a_coeff = p_at_one.sub(&b_coeff).expand();
    let neg_a = Atom::num(0i64).sub(&a_coeff).expand();
    let rhs = b_coeff.div(&neg_a).expand();

    is_valid_rhs(&rhs, loaded).then_some(rhs)
}

fn quadratic_roots(
    factor: &Atom,
    solved_vid: usize,
    loaded: &LoadedMathematicaSystem,
) -> Option<[Atom; 2]> {
    let zero = Atom::num(0i64);
    let one = Atom::num(1i64);
    let minus_one = Atom::num(-1i64);
    let two = Atom::num(2i64);
    let four = Atom::num(4i64);

    let c_coeff = substitute_vid(factor, solved_vid, zero, loaded);
    let p_at_one = substitute_vid(factor, solved_vid, one, loaded);
    let p_at_minus_one = substitute_vid(factor, solved_vid, minus_one, loaded);
    let two_c = two.clone().mul(&c_coeff);
    let numerator_a = p_at_one.clone().add(&p_at_minus_one).sub(&two_c).expand();
    let a_coeff = numerator_a.div(&two).expand();
    let b_coeff = p_at_one.sub(&c_coeff).sub(&a_coeff).expand();
    let discriminant = b_coeff
        .clone()
        .mul(&b_coeff)
        .sub(&four.mul(&a_coeff).mul(&c_coeff))
        .expand();
    let sqrt_disc = parse_expression(format!("sqrt({discriminant})")).ok()?;
    let neg_b = Atom::num(0i64).sub(&b_coeff).expand();
    let denominator = Atom::num(2i64).mul(&a_coeff).expand();
    let plus_root = neg_b.clone().add(&sqrt_disc).div(&denominator).expand();
    let minus_root = neg_b.sub(&sqrt_disc).div(&denominator).expand();

    if is_valid_rhs(&plus_root, loaded) && is_valid_rhs(&minus_root, loaded) {
        Some([plus_root, minus_root])
    } else {
        None
    }
}

fn parametrization_rhs(
    factor: &Atom,
    solved_vid: usize,
    degree: u16,
    loaded: &LoadedMathematicaSystem,
) -> Vec<Atom> {
    match degree {
        1 => linear_rhs(factor, solved_vid, loaded).into_iter().collect(),
        2 => quadratic_roots(factor, solved_vid, loaded)
            .into_iter()
            .flatten()
            .collect(),
        _ => Vec::new(),
    }
}

fn build_problem(
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
                let info = FactorInfo {
                    vars: factor_vars(&canon, &loaded.sym_to_vid),
                    poly: canon,
                };
                factors.push(info);
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
        .flat_map(|fi| fi.vars.iter().map(|(v, _)| *v))
        .collect();
    v_set.sort_unstable();
    v_set.dedup();
    let v_idx: HashMap<usize, usize> = v_set.iter().enumerate().map(|(i, v)| (*v, i)).collect();

    let mut admissible = Vec::new();
    for (fi, info) in factors.iter().enumerate() {
        let Some(factor_atom) = poly_to_atom(&info.poly) else {
            continue;
        };
        for (vid, deg) in &info.vars {
            if !(1..=2).contains(deg) {
                continue;
            }

            let solved_vloc = v_idx[vid];
            for rhs in parametrization_rhs(&factor_atom, *vid, *deg, loaded) {
                let dependency_vlocs = atom_vars_as_vlocs(&rhs, *vid, &v_idx, loaded);
                admissible.push(ParametrizationInfo {
                    factor_idx: fi,
                    solved_vloc,
                    solved_vid: *vid,
                    degree: *deg,
                    rhs,
                    dependency_vlocs,
                });
            }
        }
    }

    ProblemData {
        factors,
        f_by_poly,
        v_ids: v_set,
        admissible,
    }
}

fn var_name(loaded: &LoadedMathematicaSystem, global_vid: usize) -> &str {
    &loaded.var_names[global_vid]
}

/// Result of solving the MILP: list of selected `(factor_idx, global_var_id)`
/// pairs in topological order along with the chosen degrees.
#[cfg_attr(test, derive(Debug))]
struct Cover {
    selected: Vec<(usize, f64)>,
    dependent_locals: Vec<usize>,
}

struct MilpVariables {
    selected: Vec<Variable>,
    dependent: Vec<Variable>,
    topo: Vec<Variable>,
}

fn add_milp_variables(vars: &mut ProblemVariables, data: &ProblemData) -> MilpVariables {
    let n = data.v_ids.len();
    let selected: Vec<_> = data
        .admissible
        .iter()
        .map(|_| vars.add(variable().binary()))
        .collect();
    let dependent: Vec<_> = (0..n).map(|_| vars.add(variable().binary())).collect();
    let topo: Vec<_> = (0..n)
        .map(|_| vars.add(variable().min(0.0).max(n as f64)))
        .collect();

    MilpVariables {
        selected,
        dependent,
        topo,
    }
}

fn pairs_by_variable(data: &ProblemData) -> Vec<Vec<usize>> {
    let mut pairs_by_var: Vec<Vec<usize>> = vec![Vec::new(); data.v_ids.len()];
    for (k, param) in data.admissible.iter().enumerate() {
        pairs_by_var[param.solved_vloc].push(k);
    }
    pairs_by_var
}

fn selected_pair_count(vars: &MilpVariables) -> Expression {
    vars.selected.iter().map(|v| Expression::from(*v)).sum()
}

fn selected_degree_sum(data: &ProblemData, vars: &MilpVariables) -> Expression {
    data.admissible
        .iter()
        .enumerate()
        .map(|(k, param)| f64::from(param.degree) * vars.selected[k])
        .sum()
}

fn objective(data: &ProblemData, vars: &MilpVariables, objective: CoverObjective) -> Expression {
    match objective {
        CoverObjective::Count => selected_pair_count(vars),
        CoverObjective::Degree => selected_degree_sum(data, vars),
        CoverObjective::CountDegree => {
            let max_degree_sum = 2.0 * data.admissible.len() as f64;
            (max_degree_sum + 1.0) * selected_pair_count(vars) + selected_degree_sum(data, vars)
        }
    }
}

fn add_variable_linking_constraints<M: SolverModel>(
    mut model: M,
    pairs_by_var: &[Vec<usize>],
    vars: &MilpVariables,
) -> M {
    for (vloc, ks) in pairs_by_var.iter().enumerate() {
        let sum: Expression = ks.iter().map(|&k| Expression::from(vars.selected[k])).sum();
        model = model.with(constraint!(vars.dependent[vloc] - sum.clone() == 0));
        model = model.with(constraint!(sum <= 1));
    }
    model
}

fn add_coverage_constraints<M: SolverModel>(
    mut model: M,
    data: &ProblemData,
    vars: &MilpVariables,
) -> Result<M, ResolutionError> {
    for (j, fs) in data.f_by_poly.iter().enumerate() {
        let selected_sum: Expression = data
            .admissible
            .iter()
            .enumerate()
            .filter(|(_, param)| fs.contains(&param.factor_idx))
            .map(|(k, _)| Expression::from(vars.selected[k]))
            .sum();
        if data
            .admissible
            .iter()
            .all(|param| !fs.contains(&param.factor_idx))
        {
            eprintln!(
                "error: polynomial {j} has no factor with an admissible parametrization. The instance is infeasible."
            );
            return Err(ResolutionError::Infeasible);
        }
        model = model.with(constraint!(selected_sum >= 1));
    }
    Ok(model)
}

fn add_selected_pair_constraints<M: SolverModel>(
    mut model: M,
    data: &ProblemData,
    vars: &MilpVariables,
) -> M {
    let big_m = (data.v_ids.len() as f64) + 1.0;

    for (k, param) in data.admissible.iter().enumerate() {
        for &vloc_other in &param.dependency_vlocs {
            model = model.with(constraint!(
                vars.topo[vloc_other] + 1.0
                    <= vars.topo[param.solved_vloc]
                        + big_m
                            * (2.0
                                - Expression::from(vars.selected[k])
                                - Expression::from(vars.dependent[vloc_other]))
            ));
        }
    }
    model
}

fn cover_from_solution<S: Solution>(
    solution: &S,
    data: &ProblemData,
    vars: &MilpVariables,
) -> Cover {
    let n = data.v_ids.len();
    let mut selected = Vec::new();
    for (k, param) in data.admissible.iter().enumerate() {
        if solution.value(vars.selected[k]) > 0.5 {
            selected.push((k, solution.value(vars.topo[param.solved_vloc])));
        }
    }
    selected.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let dependent_locals: Vec<usize> = (0..n)
        .filter(|&i| solution.value(vars.dependent[i]) > 0.5)
        .collect();

    Cover {
        selected,
        dependent_locals,
    }
}

fn solve_milp(
    data: &ProblemData,
    cover_objective: CoverObjective,
) -> Result<Cover, ResolutionError> {
    let mut problem_vars = variables!();
    let milp_vars = add_milp_variables(&mut problem_vars, data);
    let pairs_by_var = pairs_by_variable(data);
    let objective = objective(data, &milp_vars, cover_objective);

    let model = problem_vars.minimise(objective).using(default_solver);
    let model = add_variable_linking_constraints(model, &pairs_by_var, &milp_vars);
    let model = add_coverage_constraints(model, data, &milp_vars)?;
    let model = add_selected_pair_constraints(model, data, &milp_vars);
    let solution = model.solve()?;

    Ok(cover_from_solution(&solution, data, &milp_vars))
}

fn print_cover(loaded: &LoadedMathematicaSystem, data: &ProblemData, cover: &Cover) {
    println!("\nCover (in topological order):");
    for (rank, (param_idx, t)) in cover.selected.iter().enumerate() {
        let param = &data.admissible[*param_idx];
        let info = &data.factors[param.factor_idx];
        let deps: Vec<&str> = param
            .dependency_vlocs
            .iter()
            .map(|&vloc| var_name(loaded, data.v_ids[vloc]))
            .collect();
        println!(
            "  [{rank}] t={t:.2}  x = {:<16} deg_x(f) = {}\n        rhs = {}\n        deps = {:?}\n        f = {}",
            var_name(loaded, param.solved_vid),
            param.degree,
            param.rhs,
            deps,
            info.poly,
        );
    }
}

fn print_variable_sets(loaded: &LoadedMathematicaSystem, data: &ProblemData, cover: &Cover) {
    let dep: Vec<&str> = cover
        .dependent_locals
        .iter()
        .map(|&i| var_name(loaded, data.v_ids[i]))
        .collect();
    let all_used: Vec<&str> = data
        .v_ids
        .iter()
        .map(|&vid| var_name(loaded, vid))
        .collect();
    let free: Vec<&str> = all_used
        .iter()
        .copied()
        .filter(|name| !dep.contains(name))
        .collect();

    println!("\nDependent variables X = {dep:?}");
    println!("Free parameters    Y = {free:?}");
    println!("(Variables not appearing in any factor are also free parameters.)");
}

fn main() {
    let key = "5568d092#6bb8b4fb#ff936916-65ba-552f-b3c1-bdc64597cd21";
    LicenseManager::set_license_key(key).unwrap();

    let options = match parse_args(std::env::args().skip(1)) {
        Ok(options) => options,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    };

    println!("symbolica-elim — MILP radical cover (good_lp / HiGHS)\n");
    println!("Loading {}", options.path.display());

    let loaded = match load_mathematica_system_from_path_with_var_names(&options.path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("load error: {e}");
            std::process::exit(1);
        }
    };
    println!(
        "Parsed: {} equations, {} variables (loader order)",
        loaded.system.equations.len(),
        loaded.var_names.len()
    );

    let field = AlgebraicExtension::new_complex(Q);

    println!("\nPreprocessing (interreduce + square-free factor)");
    let t_pre = Instant::now();
    let pre = match preprocess(&loaded, &field) {
        Ok(pre) => pre,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };
    println!(
        "  {} non-trivial polynomial(s), elapsed {:?}",
        pre.factor_groups.len(),
        t_pre.elapsed()
    );
    if pre.factor_groups.is_empty() {
        println!("Nothing to cover — all polynomials reduced to zero.");
        return;
    }

    let data = build_problem(pre.factor_groups, &loaded);
    println!(
        "  |F|={}, |V|={}, |A|={}",
        data.factors.len(),
        data.v_ids.len(),
        data.admissible.len()
    );

    println!(
        "\nSolving MILP ({}) with HiGHS ...",
        options.objective.label()
    );
    let t_solve = Instant::now();
    let cover = match solve_milp(&data, options.objective) {
        Ok(c) => c,
        Err(ResolutionError::Infeasible) => {
            eprintln!("infeasible: no valid cover exists for this system.");
            std::process::exit(3);
        }
        Err(e) => {
            eprintln!("solver error: {e}");
            std::process::exit(4);
        }
    };
    println!("  solved in {:?}", t_solve.elapsed());

    print_cover(&loaded, &data, &cover);
    print_variable_sets(&loaded, &data, &cover);

    println!("\nDone.");
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Once;
    use symbolica_elim::{load_mathematica_system_from_str_with_var_names, preprocess};

    static LICENSE: Once = Once::new();
    fn activate_license() {
        LICENSE.call_once(|| {
            let _ = LicenseManager::set_license_key(
                "5568d092#6bb8b4fb#ff936916-65ba-552f-b3c1-bdc64597cd21",
            );
        });
    }

    #[test]
    fn parse_args_accepts_degree_objective_and_path() {
        let options = parse_args([
            "systems/s_system.txt".to_string(),
            "--objective".to_string(),
            "degree".to_string(),
        ])
        .expect("parse args");

        assert_eq!(options.path, PathBuf::from("systems/s_system.txt"));
        assert_eq!(options.objective, CoverObjective::Degree);
    }

    #[test]
    fn parse_args_accepts_count_degree_objective() {
        let options = parse_args(["--objective".to_string(), "count-degree".to_string()])
            .expect("parse args");

        assert_eq!(options.objective, CoverObjective::CountDegree);
    }

    #[test]
    fn parse_args_uses_system_txt_by_default() {
        let options = parse_args([]).expect("parse args");

        assert!(options.path.ends_with("systems/system.txt"));
        assert_eq!(options.objective, CoverObjective::Count);
    }

    #[test]
    fn parse_args_rejects_unknown_objective() {
        let err = parse_args(["--objective".to_string(), "size".to_string()]).expect_err("error");

        assert!(err.contains("--objective expects"));
    }

    fn problem_from(source: &str) -> (LoadedMathematicaSystem, ProblemData) {
        activate_license();
        let loaded = load_mathematica_system_from_str_with_var_names(source).expect("load");
        let field = AlgebraicExtension::new_complex(Q);
        let pre = preprocess(&loaded, &field).expect("preprocess");
        let data = build_problem(pre.factor_groups, &loaded);
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
    fn coefficient_can_be_variable() {
        let (_loaded, data) = problem_from("{x^2 * y^6 + x * y^3 + z^2 == 0}");
        assert_eq!(data.factors.len(), 1);

        // the polynomial should be of degree 2 in x and 6 in y and 2 in z
        assert_eq!(data.factors[0].vars.len(), 3);
        assert_eq!(data.factors[0].vars[0].1, 2);
        assert_eq!(data.factors[0].vars[1].1, 6);
        assert_eq!(data.factors[0].vars[2].1, 2);

        assert_eq!(data.admissible.len(), 4);
        assert!(data.admissible.iter().all(|param| param.degree <= 2));
    }

    #[test]
    fn admissible_respects_degree_bounds() {
        let (_loaded, data) = problem_from("{x^3 + y == 0}");
        assert_eq!(data.factors.len(), 1);
        assert!(
            data.factors[0].vars.iter().any(|(_, d)| *d == 3),
            "test precondition: factor should have a variable of degree 3"
        );
        assert!(
            data.admissible.iter().all(|param| param.degree <= 2),
            "admissible parametrizations must be degree 1 or 2, got {:?}",
            data.admissible
        );
    }

    #[test]
    fn dependencies_come_from_simplified_rhs() {
        let (loaded, data) = problem_from("{x*y - y*z == 0}");
        let y_vid = loaded.sym_to_vid[&loaded.id_to_sym[1]].0 as usize;
        let z_vid = loaded.sym_to_vid[&loaded.id_to_sym[2]].0 as usize;
        let x_param = data
            .admissible
            .iter()
            .find(|param| var_name(&loaded, param.solved_vid) == "x")
            .expect("x parametrization");
        let deps: Vec<&str> = x_param
            .dependency_vlocs
            .iter()
            .map(|&vloc| var_name(&loaded, data.v_ids[vloc]))
            .collect();

        assert!(
            !x_param
                .dependency_vlocs
                .iter()
                .any(|&vloc| data.v_ids[vloc] == y_vid),
            "y should simplify away from x = y*z/y; dependencies were {deps:?}"
        );
        assert!(
            x_param
                .dependency_vlocs
                .iter()
                .any(|&vloc| data.v_ids[vloc] == z_vid),
            "the simplified RHS should still depend on z; dependencies were {deps:?}"
        );
    }

    #[test]
    fn quadratic_parametrizations_store_rhs_dependencies() {
        let (loaded, data) = problem_from("{x^2 + y == 0}");
        let x_vid = loaded.sym_to_vid[&loaded.id_to_sym[0]].0 as usize;
        let x_params: Vec<_> = data
            .admissible
            .iter()
            .filter(|param| data.v_ids[param.solved_vloc] == x_vid)
            .collect();

        assert_eq!(x_params.len(), 2, "quadratics should yield two roots");
        for param in x_params {
            let deps: Vec<&str> = param
                .dependency_vlocs
                .iter()
                .map(|&vloc| var_name(&loaded, data.v_ids[vloc]))
                .collect();
            assert_eq!(param.degree, 2);
            assert_eq!(deps, vec!["y"]);
        }
    }

    fn topo_ok(data: &ProblemData, cover: &Cover) -> bool {
        let dep_vids: std::collections::HashSet<usize> = cover
            .dependent_locals
            .iter()
            .map(|&i| data.v_ids[i])
            .collect();
        let t: std::collections::HashMap<usize, f64> = cover
            .selected
            .iter()
            .map(|&(param_idx, t)| (data.admissible[param_idx].solved_vid, t))
            .collect();
        for &(param_idx, ti) in &cover.selected {
            let param = &data.admissible[param_idx];
            for &vloc_other in &param.dependency_vlocs {
                let vother = data.v_ids[vloc_other];
                if !dep_vids.contains(&vother) {
                    continue;
                }
                let tj = t.get(&vother).copied().unwrap_or(f64::INFINITY);
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
        let cover = solve_milp(&data, CoverObjective::Count).expect("feasible");
        assert_eq!(cover.selected.len(), 3);
        for j in 0..data.f_by_poly.len() {
            let covered = cover.selected.iter().any(|(param_idx, _)| {
                data.f_by_poly[j].contains(&data.admissible[*param_idx].factor_idx)
            });
            assert!(covered, "polynomial {j} is not covered");
        }
        assert!(topo_ok(&data, &cover));
    }

    #[test]
    fn solve_respects_acyclicity() {
        // q factors as x*(y + x); picking (y + x, y) forces t_x < t_y.
        let (_loaded, data) = problem_from("{p -> x + 1, q -> x*y + x^2}");
        let cover = solve_milp(&data, CoverObjective::Count).expect("feasible");
        assert!(topo_ok(&data, &cover));
        for j in 0..data.f_by_poly.len() {
            assert!(cover.selected.iter().any(|(param_idx, _)| {
                data.f_by_poly[j].contains(&data.admissible[*param_idx].factor_idx)
            }));
        }
    }

    #[test]
    fn solve_rejects_high_degree_without_admissible_parametrization() {
        let (_loaded, data) = problem_from("{x^5 + x + 3 == 0}");
        let res = solve_milp(&data, CoverObjective::Count);
        assert!(matches!(res, Err(ResolutionError::Infeasible)));
    }

    #[test]
    fn solve_high_degree_monomial_via_square_free_linear_factor() {
        let (_loaded, data) = problem_from("{x^5 == 0}");
        let cover = solve_milp(&data, CoverObjective::Count).expect("feasible by selecting x = 0");
        assert_eq!(cover.selected.len(), 1);
        assert_eq!(data.admissible[cover.selected[0].0].degree, 1);
    }

    #[test]
    fn minimize_degree_prefers_linear() {
        // p has two square-free factors (x+1) (degree 1 in x) and (x^2 + y + 1) (degree 2 in x,
        // degree 1 in y). Count-minimization picks any 1 tuple; degree-minimization should still
        // pick 1 tuple but prefer the degree-1 one.
        let (_loaded, data) = problem_from("{p -> (x + 1) * (x^2 + y + 1)}");
        let cover_deg = solve_milp(&data, CoverObjective::Degree).expect("feasible");
        let total_deg: u32 = cover_deg
            .selected
            .iter()
            .map(|(param_idx, _)| u32::from(data.admissible[*param_idx].degree))
            .sum();
        assert_eq!(
            total_deg, 1,
            "degree-minimization should pick the linear factor"
        );
    }

    #[test]
    fn minimize_count_degree_breaks_count_ties_by_degree() {
        let (_loaded, data) = problem_from("{p -> (x + 1) * (x^2 + y + 1)}");
        let cover = solve_milp(&data, CoverObjective::CountDegree).expect("feasible");
        let total_deg: u32 = cover
            .selected
            .iter()
            .map(|(param_idx, _)| u32::from(data.admissible[*param_idx].degree))
            .sum();

        assert_eq!(cover.selected.len(), 1);
        assert_eq!(total_deg, 1);
    }
}
