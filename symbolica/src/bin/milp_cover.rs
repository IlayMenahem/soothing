//! MILP-based radical cover as described in `problem statement.md`.
//!
//! Preprocess the input polynomial system (interreduce, square-free factor,
//! drop constants/zero polynomials), then solve an integer program with
//! [`good_lp`]/HiGHS to pick tuples `(f_i, x_i)` that cover every polynomial
//! by a radical-solvable, acyclic assignment.
//!
//! Usage: `milp_cover [path]` — default path: `systems/system.txt`.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use good_lp::{
    Expression, ProblemVariables, ResolutionError, Solution, SolverModel, Variable, constraint,
    default_solver, variable, variables,
};

use symbolica::LicenseManager;
use symbolica::atom::Symbol;
use symbolica::domains::algebraic_number::AlgebraicExtension;
use symbolica::domains::rational::Q;
use symbolica::poly::PolyVariable;

use symbolica_elim::{
    LoadedMathematicaSystem, PolyExtLex, default_systems_dir_from_manifest,
    load_mathematica_system_from_path_with_var_names, preprocess,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CoverObjective {
    Count,
    Degree,
    Zeroed,
}

impl CoverObjective {
    fn label(self) -> &'static str {
        match self {
            CoverObjective::Count => "minimize cover size",
            CoverObjective::Degree => "minimize total degree",
            CoverObjective::Zeroed => "minimize zeroed variables, then cover size",
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
                Some("zeroed") => objective = CoverObjective::Zeroed,
                other => {
                    return Err(format!(
                        "--objective expects 'count', 'degree', or 'zeroed' (got {other:?})"
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
    /// Local Symbolica variable index for each global variable id in `vars`.
    local_var_by_global: HashMap<usize, usize>,
    /// Global variable ids in each monomial.
    monomial_vars: Vec<Vec<usize>>,
}

/// Problem data assembled from the preprocessed factor groups.
struct ProblemData {
    factors: Vec<FactorInfo>,
    /// For each original (non-zero, non-trivial) polynomial, the indices of
    /// its distinct square-free factors in `factors`.
    f_by_poly: Vec<Vec<usize>>,
    /// Global variable ids that appear in at least one factor.
    v_ids: Vec<usize>,
    /// Inverse of `v_ids`.
    v_idx: HashMap<usize, usize>,
    /// Admissible pairs as `(factor_idx, local_var_idx)` with `1 <= deg <= 4`.
    admissible: Vec<(usize, usize)>,
    /// `deg_x(f)` indexed the same way as `admissible`.
    deg_of_pair: Vec<u16>,
}

fn factor_vars(
    f: &PolyExtLex,
    sym_to_vid: &HashMap<Symbol, symbolica_elim::VarId>,
) -> (Vec<(usize, u16)>, HashMap<usize, usize>) {
    let mut out = Vec::new();
    let mut local_var_by_global = HashMap::new();
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
        let global_vid = vid.0 as usize;
        out.push((global_vid, f.degree(xi)));
        local_var_by_global.insert(global_vid, xi);
    }
    (out, local_var_by_global)
}

fn factor_monomial_vars(
    f: &PolyExtLex,
    sym_to_vid: &HashMap<Symbol, symbolica_elim::VarId>,
) -> Vec<Vec<usize>> {
    f.exponents_iter()
        .map(|exponents| {
            exponents
                .iter()
                .zip(f.get_vars_ref().iter())
                .filter_map(|(exponent, pv)| {
                    if *exponent == 0 {
                        return None;
                    }
                    let PolyVariable::Symbol(sym) = pv else {
                        return None;
                    };
                    sym_to_vid.get(sym).map(|vid| vid.0 as usize)
                })
                .collect()
        })
        .collect()
}

fn zeroing_invalidates_pair(info: &FactorInfo, selected_vid: usize, zeroed_vid: usize) -> bool {
    let mut saw_selected_variable = false;
    let mut zeroed_kills_every_selected_monomial = true;
    let mut zeroed_kills_every_monomial = true;

    for monomial in &info.monomial_vars {
        let has_selected = monomial.contains(&selected_vid);
        let has_zeroed = monomial.contains(&zeroed_vid);

        if has_selected {
            saw_selected_variable = true;
            zeroed_kills_every_selected_monomial &= has_zeroed;
        }
        zeroed_kills_every_monomial &= has_zeroed;
    }

    saw_selected_variable && zeroed_kills_every_selected_monomial && !zeroed_kills_every_monomial
}

fn linear_coefficient(info: &FactorInfo, selected_vid: usize) -> Option<PolyExtLex> {
    info.local_var_by_global
        .get(&selected_vid)
        .map(|&local_var| info.poly.derivative(local_var))
}

fn factor_invalidates_linear_pair(
    invalidator: &FactorInfo,
    target: &FactorInfo,
    target_vid: usize,
) -> bool {
    let Some(coefficient) = linear_coefficient(target, target_vid) else {
        return false;
    };

    coefficient.try_div(&invalidator.poly).is_some()
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
                let (vars, local_var_by_global) = factor_vars(&canon, &loaded.sym_to_vid);
                let info = FactorInfo {
                    vars,
                    local_var_by_global,
                    monomial_vars: factor_monomial_vars(&canon, &loaded.sym_to_vid),
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
    let mut deg_of_pair = Vec::new();
    for (fi, info) in factors.iter().enumerate() {
        for (vid, deg) in &info.vars {
            if *deg == 1 {
                admissible.push((fi, v_idx[vid]));
                deg_of_pair.push(*deg);
            }
        }
    }

    ProblemData {
        factors,
        f_by_poly,
        v_ids: v_set,
        v_idx,
        admissible,
        deg_of_pair,
    }
}

fn var_name(loaded: &LoadedMathematicaSystem, global_vid: usize) -> &str {
    &loaded.var_names[global_vid]
}

/// Result of solving the MILP: list of selected `(factor_idx, global_var_id)`
/// pairs in topological order along with the chosen degrees.
#[cfg_attr(test, derive(Debug))]
struct Cover {
    selected: Vec<(usize, usize, u16, f64)>,
    dependent_locals: Vec<usize>,
    zeroed_locals: Vec<usize>,
}

struct MilpVariables {
    selected: Vec<Variable>,
    dependent: Vec<Variable>,
    zeroed: Vec<Variable>,
    zero_killed: Vec<Variable>,
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
    let zeroed: Vec<_> = (0..n).map(|_| vars.add(variable().binary())).collect();
    let zero_killed: Vec<_> = data
        .factors
        .iter()
        .map(|_| vars.add(variable().binary()))
        .collect();
    let topo: Vec<_> = (0..n)
        .map(|_| vars.add(variable().min(0.0).max(n as f64)))
        .collect();

    MilpVariables {
        selected,
        dependent,
        zeroed,
        zero_killed,
        topo,
    }
}

fn pairs_by_variable(data: &ProblemData) -> Vec<Vec<usize>> {
    let mut pairs_by_var: Vec<Vec<usize>> = vec![Vec::new(); data.v_ids.len()];
    for (k, &(_, vloc)) in data.admissible.iter().enumerate() {
        pairs_by_var[vloc].push(k);
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
        .map(|(k, _)| f64::from(data.deg_of_pair[k]) * vars.selected[k])
        .sum()
}

fn zeroed_variable_count(vars: &MilpVariables) -> Expression {
    vars.zeroed.iter().map(|v| Expression::from(*v)).sum()
}

fn objective(data: &ProblemData, vars: &MilpVariables, objective: CoverObjective) -> Expression {
    let zeroed_count = zeroed_variable_count(vars);
    match objective {
        CoverObjective::Count => selected_pair_count(vars) + zeroed_count,
        CoverObjective::Degree => selected_degree_sum(data, vars) + zeroed_count,
        CoverObjective::Zeroed => {
            let tie_breaker_weight = data.admissible.len() as f64 + 1.0;
            tie_breaker_weight * zeroed_count + selected_pair_count(vars)
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
        model = model.with(constraint!(vars.dependent[vloc] + vars.zeroed[vloc] <= 1));
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
            .filter(|(_, (fi, _))| fs.contains(fi))
            .map(|(k, _)| Expression::from(vars.selected[k]))
            .sum();
        let zeroed_sum: Expression = fs
            .iter()
            .map(|&fi| Expression::from(vars.zero_killed[fi]))
            .sum();
        if data.admissible.iter().all(|(fi, _)| !fs.contains(fi)) && fs.is_empty() {
            eprintln!(
                "error: polynomial {j} has no factor that can be selected or zero-killed. The instance is infeasible."
            );
            return Err(ResolutionError::Infeasible);
        }
        model = model.with(constraint!(selected_sum + zeroed_sum >= 1));
    }
    Ok(model)
}

fn add_zero_kill_constraints<M: SolverModel>(
    mut model: M,
    data: &ProblemData,
    vars: &MilpVariables,
) -> M {
    for (fi, info) in data.factors.iter().enumerate() {
        for monomial in &info.monomial_vars {
            let zeroed_vars_in_monomial: Expression = monomial
                .iter()
                .filter_map(|vid| {
                    data.v_idx
                        .get(vid)
                        .map(|&vloc| Expression::from(vars.zeroed[vloc]))
                })
                .sum();
            model = model.with(constraint!(
                vars.zero_killed[fi] - zeroed_vars_in_monomial <= 0
            ));
        }
    }
    model
}

fn add_selected_pair_constraints<M: SolverModel>(
    mut model: M,
    data: &ProblemData,
    vars: &MilpVariables,
) -> M {
    let big_m = (data.v_ids.len() as f64) + 1.0;

    for (k, &(fi, vloc)) in data.admissible.iter().enumerate() {
        let info = &data.factors[fi];
        let selected_vid = data.v_ids[vloc];
        for (zeroed_vloc, &zeroed_vid) in data.v_ids.iter().enumerate() {
            if zeroing_invalidates_pair(info, selected_vid, zeroed_vid) {
                model = model.with(constraint!(
                    vars.selected[k] + vars.zeroed[zeroed_vloc] <= 1
                ));
            }
        }
        for (vid_other, _deg) in &info.vars {
            let Some(&vloc_other) = data.v_idx.get(vid_other) else {
                continue;
            };
            if vloc_other == vloc {
                continue;
            }
            model = model.with(constraint!(
                vars.topo[vloc_other] + 1.0
                    <= vars.topo[vloc]
                        + big_m
                            * (2.0
                                - Expression::from(vars.selected[k])
                                - Expression::from(vars.dependent[vloc_other])
                                + Expression::from(vars.zeroed[vloc_other]))
            ));
        }
    }
    model
}

fn add_pair_incompatibility_constraints<M: SolverModel>(
    mut model: M,
    data: &ProblemData,
    vars: &MilpVariables,
) -> M {
    for (target_k, &(target_fi, target_vloc)) in data.admissible.iter().enumerate() {
        let target = &data.factors[target_fi];
        let target_vid = data.v_ids[target_vloc];

        for (invalidator_k, &(invalidator_fi, _)) in data.admissible.iter().enumerate() {
            if invalidator_k == target_k {
                continue;
            }
            let invalidator = &data.factors[invalidator_fi];
            if factor_invalidates_linear_pair(invalidator, target, target_vid) {
                model = model.with(constraint!(
                    vars.selected[invalidator_k] + vars.selected[target_k] <= 1
                ));
            }
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
    for (k, &(fi, vloc)) in data.admissible.iter().enumerate() {
        if solution.value(vars.selected[k]) > 0.5 {
            let vid = data.v_ids[vloc];
            selected.push((
                fi,
                vid,
                data.deg_of_pair[k],
                solution.value(vars.topo[vloc]),
            ));
        }
    }
    selected.sort_by(|a, b| a.3.partial_cmp(&b.3).unwrap_or(std::cmp::Ordering::Equal));

    let dependent_locals: Vec<usize> = (0..n)
        .filter(|&i| solution.value(vars.dependent[i]) > 0.5)
        .collect();
    let zeroed_locals: Vec<usize> = (0..n)
        .filter(|&i| solution.value(vars.zeroed[i]) > 0.5)
        .collect();

    Cover {
        selected,
        dependent_locals,
        zeroed_locals,
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
    let model = add_zero_kill_constraints(model, data, &milp_vars);
    let model = add_selected_pair_constraints(model, data, &milp_vars);
    let model = add_pair_incompatibility_constraints(model, data, &milp_vars);
    let solution = model.solve()?;

    Ok(cover_from_solution(&solution, data, &milp_vars))
}

fn print_cover(loaded: &LoadedMathematicaSystem, data: &ProblemData, cover: &Cover) {
    println!("\nCover (in topological order):");
    for (rank, (fi, vid, deg, t)) in cover.selected.iter().enumerate() {
        let info = &data.factors[*fi];
        println!(
            "  [{rank}] t={t:.2}  x = {:<16} deg_x(f) = {deg}\n        f = {}",
            var_name(loaded, *vid),
            info.poly
        );
    }
}

fn print_variable_sets(loaded: &LoadedMathematicaSystem, data: &ProblemData, cover: &Cover) {
    let dep: Vec<&str> = cover
        .dependent_locals
        .iter()
        .map(|&i| var_name(loaded, data.v_ids[i]))
        .collect();
    let zeroed: Vec<&str> = cover
        .zeroed_locals
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
        .filter(|name| !dep.contains(name) && !zeroed.contains(name))
        .collect();

    println!("\nDependent variables X = {dep:?}");
    println!("Zeroed variables    W = {zeroed:?}");
    println!("Free parameters    Y = {free:?}");
    println!("(Variables not appearing in any factor are also free parameters unless zeroed.)");
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
    use symbolica_elim::{
        ideal_ext_lex, load_mathematica_system_from_str_with_var_names, preprocess,
    };

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
    fn parse_args_accepts_zeroed_objective() {
        let options =
            parse_args(["--objective".to_string(), "zeroed".to_string()]).expect("parse args");

        assert_eq!(options.objective, CoverObjective::Zeroed);
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

    fn problem_from_raw_factors(source: &str) -> (LoadedMathematicaSystem, ProblemData) {
        activate_license();
        let loaded = load_mathematica_system_from_str_with_var_names(source).expect("load");
        let field = AlgebraicExtension::new_complex(Q);
        let var_order = symbolica_elim::variable_order(&loaded.id_to_sym);
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
    fn coefficient_can_be_variable() {
        let (_loaded, data) = problem_from("{x^2 * y^6 + x * y^3 + z^2 == 0}");
        assert_eq!(data.factors.len(), 1);

        // the polynomial should be of degree 2 in x and 6 in y and 2 in z
        assert_eq!(data.factors[0].vars.len(), 3);
        assert_eq!(data.factors[0].vars[0].1, 2);
        assert_eq!(data.factors[0].vars[1].1, 6);
        assert_eq!(data.factors[0].vars[2].1, 2);

        assert!(
            data.admissible.is_empty(),
            "only linear pairs are admissible; got {:?}",
            data.admissible
        );
    }

    #[test]
    fn admissible_respects_degree_bounds() {
        // A single polynomial of degree 5 in x whose (irreducible) square-free part
        // is itself the degree-5 polynomial. No admissible pair can exist.
        let (_loaded, data) = problem_from("{x^5 + x + 3 == 0}");
        assert_eq!(data.factors.len(), 1);
        assert!(
            data.factors[0].vars.iter().any(|(_, d)| *d > 1),
            "test precondition: factor should have a nonlinear variable"
        );
        assert!(
            data.admissible.is_empty(),
            "no admissible pair can exist when every variable is nonlinear, got {:?}",
            data.admissible
        );
        for &d in &data.deg_of_pair {
            assert_eq!(d, 1);
        }
    }

    #[test]
    fn admissible_excludes_quadratic_radical_solves() {
        let (_loaded, data) = problem_from("{x^2 + y^2 + 1 == 0}");

        assert!(
            data.admissible.is_empty(),
            "degree 2-4 radical solves are no longer admissible"
        );
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
            .map(|&(_, vid, _, t)| (vid, t))
            .collect();
        for &(fi, vid, _, ti) in &cover.selected {
            for (vother, _) in &data.factors[fi].vars {
                if *vother == vid || !dep_vids.contains(vother) {
                    continue;
                }
                let tj = t.get(vother).copied().unwrap_or(f64::INFINITY);
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
            let covered = cover
                .selected
                .iter()
                .any(|(fi, _, _, _)| data.f_by_poly[j].contains(fi));
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
            assert!(
                cover
                    .selected
                    .iter()
                    .any(|(fi, _, _, _)| data.f_by_poly[j].contains(fi))
            );
        }
    }

    #[test]
    fn solve_rejects_pair_that_zeroes_another_linear_coefficient() {
        let (_loaded, data) = problem_from_raw_factors("{y == 0, y*x + 1 == 0}");

        let res = solve_milp(&data, CoverObjective::Count);

        assert!(
            matches!(res, Err(ResolutionError::Infeasible)),
            "selecting y = 0 must not be compatible with solving y*x + 1 for x"
        );
    }

    #[test]
    fn solve_allows_independent_linear_coefficients() {
        let (_loaded, data) = problem_from_raw_factors("{y == 0, x + z == 0}");

        let cover = solve_milp(&data, CoverObjective::Zeroed).expect("feasible");

        assert_eq!(cover.selected.len(), 2);
        assert!(topo_ok(&data, &cover));
    }

    #[test]
    fn solve_zeroes_single_high_degree_monomial() {
        let (_loaded, data) = problem_from("{x^5 + x + 3 == 0}");
        let res = solve_milp(&data, CoverObjective::Count);
        assert!(matches!(res, Err(ResolutionError::Infeasible)));
    }

    #[test]
    fn solve_zero_kills_high_degree_monomial() {
        let (_loaded, data) = problem_from("{x^5 == 0}");
        let cover = solve_milp(&data, CoverObjective::Count).expect("feasible by zeroing x");
        assert_eq!(cover.selected.len() + cover.zeroed_locals.len(), 1);
    }

    #[test]
    fn solve_does_not_hide_kappah_behind_zero_e_component() {
        let (loaded, data) = problem_from(include_str!("../../../systems/s_system.txt"));
        let cover = solve_milp(&data, CoverObjective::Count).expect("feasible");
        let zeroed_names: Vec<&str> = cover
            .zeroed_locals
            .iter()
            .map(|&i| var_name(&loaded, data.v_ids[i]))
            .collect();

        assert!(
            zeroed_names.contains(&"kappah"),
            "kappah must be determined, not free, because e = 0 leaves constraints that force kappah = 0; zeroed variables were {zeroed_names:?}"
        );
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
            .map(|(_, _, d, _)| u32::from(*d))
            .sum();
        assert_eq!(
            total_deg, 1,
            "degree-minimization should pick the linear factor"
        );
    }

    #[test]
    fn minimize_zeroed_prefers_no_zeroed_variables() {
        let (_loaded, data) = problem_from("{p -> x^5 * (y + 1)}");
        let cover = solve_milp(&data, CoverObjective::Zeroed).expect("feasible");

        assert!(
            cover.zeroed_locals.is_empty(),
            "zeroed objective should prefer selecting a factor over zeroing variables"
        );
        assert_eq!(cover.selected.len(), 1);
    }

    #[test]
    fn minimize_zeroed_breaks_ties_by_cover_size() {
        let (_loaded, data) = problem_from("{p -> (x + 1) * (y + 1)}");
        let cover = solve_milp(&data, CoverObjective::Zeroed).expect("feasible");

        assert!(cover.zeroed_locals.is_empty());
        assert_eq!(
            cover.selected.len(),
            1,
            "zeroed objective should minimize cover size among equally zero-minimal covers"
        );
    }
}
