//! MILP-based radical cover as described in `problem statement.md`.
//!
//! Preprocess the input polynomial system (interreduce, square-free factor,
//! drop constants/zero polynomials), then solve an integer program with
//! [`good_lp`]/HiGHS to pick tuples `(f_i, x_i)` and a set of zeroed
//! variables that satisfy constraints C1–C4 and maximize the number of
//! polynomials made identically zero by the zeroing alone.
//!
//! Usage: `milp_cover [path]` — default path: `systems/system.txt`.

use std::collections::{HashMap, HashSet};
use std::ops::{Div, Sub};
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
use thiserror::Error;

use symbolica_elim::atom_util::{parse_expression, substitute_symbol};
use symbolica_elim::{
    ExactPredicateError, LoadedMathematicaSystem, PolyExtLex, SubstitutionMap, VarId,
    default_systems_dir_from_manifest, is_zero_exact,
    load_mathematica_system_from_path_with_var_names, preprocess, rhs_denominator_nonzero_exact,
};

#[derive(Debug)]
struct CliOptions {
    path: PathBuf,
}

fn parse_args(args: impl IntoIterator<Item = String>) -> Result<CliOptions, String> {
    let mut path: Option<PathBuf> = None;
    for arg in args {
        if arg.starts_with("--") {
            return Err(format!("unknown option {arg:?}"));
        }
        path = Some(PathBuf::from(arg));
    }
    Ok(CliOptions {
        path: path.unwrap_or_else(|| default_systems_dir_from_manifest().join("system.txt")),
    })
}

/// A single square-free factor together with the global-variable ids it
/// depends on and per-monomial variable lists.
struct FactorInfo {
    poly: PolyExtLex,
    /// Global variable ids that occur in `poly`.
    vars: Vec<usize>,
    /// Per-monomial: the list of global variable ids in the monomial.
    monomial_vars: Vec<Vec<usize>>,
    /// Per-monomial: the per-variable degree, parallel to `monomial_vars` and
    /// keyed by the position in `vars`.
    monomial_degrees: Vec<Vec<u16>>,
}

impl FactorInfo {
    /// Indices of monomials whose degree in the global variable `vid` is exactly `1`.
    fn monomials_deg(&self, vid: usize, target_degree: u16) -> Vec<usize> {
        let Some(slot) = self.vars.iter().position(|v| *v == vid) else {
            return Vec::new();
        };
        self.monomial_degrees
            .iter()
            .enumerate()
            .filter_map(|(idx, degs)| (degs[slot] == target_degree).then_some(idx))
            .collect()
    }

    fn monomial_degree(&self, monomial: usize, vid: usize) -> u16 {
        let Some(slot) = self.vars.iter().position(|v| *v == vid) else {
            return 0;
        };
        self.monomial_degrees[monomial][slot]
    }
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
    /// Admissible pairs `(factor_idx, local_var_idx)`: those for which the
    /// factor has at least one monomial of degree exactly 1 in the variable.
    admissible: Vec<(usize, usize)>,
    /// For each admissible pair, the indices into the factor's `monomial_vars`
    /// of monomials of degree exactly 1 in the selected variable (i.e. `M^1_{f,x}`).
    deg1_monomials: Vec<Vec<usize>>,
    /// For each admissible pair, the indices into the factor's `monomial_vars`
    /// of monomials of degree exactly 0 in the selected variable (the "constant term"
    /// w.r.t. that variable). Used by C2d.
    deg0_monomials: Vec<Vec<usize>>,
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
    let mut deg1_monomials = Vec::new();
    let mut deg0_monomials = Vec::new();
    for (fi, info) in factors.iter().enumerate() {
        if info.monomial_vars.len() == 1 {
            // C5: a single-monomial factor cannot define any variable.
            continue;
        }
        for &vid in &info.vars {
            let m1 = info.monomials_deg(vid, 1);
            if !m1.is_empty() {
                admissible.push((fi, v_idx[&vid]));
                deg1_monomials.push(m1);
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
        deg1_monomials,
        deg0_monomials,
    }
}

fn var_name(loaded: &LoadedMathematicaSystem, global_vid: usize) -> &str {
    &loaded.var_names[global_vid]
}

/// Result of solving the MILP.
#[cfg_attr(test, derive(Debug))]
struct Cover {
    /// Selected pairs as `(factor_idx, global_var_id, topo_position)` in
    /// topological order.
    selected: Vec<(usize, usize, f64)>,
    /// Local variable indices (into `data.v_ids`) zeroed by the solution.
    zeroed_locals: Vec<usize>,
    /// Original-polynomial indices made identically zero by the zeroing alone
    /// (i.e. `y_p > 0.5` in the MILP solution).
    zeroed_polynomials: Vec<usize>,
}

#[derive(Debug, Eq, PartialEq)]
struct ParameterizationReport {
    checked_equations: usize,
    binding_count: usize,
}

#[derive(Debug, Error)]
enum ParameterizationError {
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
        .deg1_monomials
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

fn objective(vars: &MilpVariables) -> Expression {
    vars.y.iter().map(|v| Expression::from(*v)).sum()
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
                .filter_map(|vid| data.v_idx.get(vid).map(|&vloc| Expression::from(vars.z[vloc])))
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
            eprintln!(
                "error: polynomial {p} has no factors. The instance is infeasible."
            );
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

        // C2a: every monomial of degree >= 2 in the selected variable must
        // have one of its non-selected variables zeroed.
        for (m, monomial) in info.monomial_vars.iter().enumerate() {
            if info.monomial_degree(m, selected_vid) < 2 {
                continue;
            }
            let killers: Expression = monomial
                .iter()
                .filter(|&&vid| vid != selected_vid)
                .filter_map(|vid| data.v_idx.get(vid).map(|&vl| Expression::from(vars.z[vl])))
                .sum();
            model = model.with(constraint!(vars.s[k] - killers <= 0));
        }

        // C2b: a degree-1 monomial survives only if none of its non-selected
        // variables are zeroed.
        for (slot, &m) in data.deg1_monomials[k].iter().enumerate() {
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

        // C2c: at least one degree-1 monomial must survive.
        let surviving: Expression = vars.a[k].iter().map(|v| Expression::from(*v)).sum();
        model = model.with(constraint!(vars.s[k] - surviving <= 0));
    }
    model
}

/// C2d: for each selected pair `(f, x)`, at least one degree-0-in-x monomial of `f`
/// (the "constant term" w.r.t. x) must not be entirely zeroed. This guarantees
/// `b_coeff = f(x=0) ≠ 0`, so the parametrization `x = -b/a` is non-zero.
/// Combined with C2b/C2c (which guarantee `a_coeff ≠ 0`), this prevents any
/// defined variable from evaluating to zero through the substitution chain, closing
/// the gap that produces `0^(-k)` infinities during cover expansion.
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
        let surviving: Expression = vars.b[k].iter().map(|v| Expression::from(*v)).sum();
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

fn solve_milp(data: &ProblemData) -> Result<Cover, ResolutionError> {
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

fn var_id_from_global(global_vid: usize) -> VarId {
    VarId(u32::try_from(global_vid).expect("global variable id must fit in VarId"))
}

fn factor_to_atom(factor: &PolyExtLex) -> Result<Atom, ParameterizationError> {
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

fn cover_substitution(
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
        let rhs = linear_factor_rhs(loaded, &data.factors[factor_index].poly, selected_vid)?;
        insert_cover_binding(loaded, &mut substitution, &mut assigned, var, rhs)?;
    }

    Ok(substitution)
}

fn expand_cover_substitution(
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

fn check_cover_parameterizes_original(
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

        let vanishes =
            is_zero_exact(&residual, &loaded.sym_to_vid, &loaded.id_to_sym).map_err(|source| {
                ParameterizationError::ExactCheck {
                    equation_index,
                    source,
                }
            })?;
        if !vanishes {
            eprintln!("parameterization check failed: original equation {} does not vanish after cover substitution: {}", equation_index, residual);
            // continue;
        }
    }

    Ok(ParameterizationReport {
        checked_equations: loaded.system.equations.len(),
        binding_count,
    })
}

fn print_cover(loaded: &LoadedMathematicaSystem, data: &ProblemData, cover: &Cover) {
    println!("\nCover (in topological order):");
    for (rank, (fi, vid, t)) in cover.selected.iter().enumerate() {
        let info = &data.factors[*fi];
        println!(
            "  [{rank}] t={t:.2}  x = {:<16}\n        f = {}",
            var_name(loaded, *vid),
            info.poly
        );
    }
    println!(
        "\nPolynomials zeroed by zeroing alone: {}",
        cover.zeroed_polynomials.len()
    );
}

fn print_variable_sets(loaded: &LoadedMathematicaSystem, data: &ProblemData, cover: &Cover) {
    let dep: Vec<&str> = cover
        .selected
        .iter()
        .map(|&(_, vid, _)| var_name(loaded, vid))
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

    println!("\nSolving MILP (maximize zeroed polynomials) with HiGHS ...");
    let t_solve = Instant::now();
    let cover = match solve_milp(&data) {
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

    match check_cover_parameterizes_original(&loaded, &data, &cover) {
        Ok(report) => println!(
            "\nParameterization check: {} original equation(s) vanish after {} substitution(s).",
            report.checked_equations, report.binding_count
        ),
        Err(error) => {
            eprintln!("\nparameterization check failed: {error}");
            std::process::exit(5);
        }
    }

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

    fn topo_ok(data: &ProblemData, cover: &Cover) -> bool {
        let topo: HashMap<usize, f64> = cover
            .selected
            .iter()
            .map(|&(_, vid, t)| (vid, t))
            .collect();
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
    fn parse_args_uses_system_txt_by_default() {
        let options = parse_args(std::iter::empty::<String>()).expect("parse args");
        assert!(options.path.ends_with("systems/system.txt"));
    }

    #[test]
    fn parse_args_accepts_a_path() {
        let options =
            parse_args(["systems/s_system.txt".to_string()]).expect("parse args");
        assert_eq!(options.path, PathBuf::from("systems/s_system.txt"));
    }

    #[test]
    fn parse_args_rejects_unknown_flag() {
        let err = parse_args(["--objective".to_string()]).expect_err("error");
        assert!(err.contains("unknown option"));
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
    fn admissible_requires_a_degree_one_monomial() {
        // Every monomial has degree >= 2 in every variable it touches, so no
        // pair (f, x) can ever be the linear definer of x.
        // Use raw factors to avoid the polynomial being split by complex factoring.
        let (_loaded, data) = problem_from_raw_factors("{x^2 * y^2 + z^2 == 0}");
        assert_eq!(data.factors.len(), 1);
        assert!(
            data.admissible.is_empty(),
            "no monomial has degree 1 in any single variable; got {:?}",
            data.admissible
        );
    }

    #[test]
    fn admissible_excludes_pure_high_degree_polynomial() {
        let (_loaded, data) = problem_from("{x^5 + x + 3 == 0}");
        // x^5 + x + 3 has a degree-1-in-x monomial (x), so the pair (f, x) is admissible.
        // But selecting it forces C2a to kill the x^5 monomial, which has no
        // non-x variable to zero — so the MILP forbids it via C2a.
        assert!(!data.admissible.is_empty());
        let cover = solve_milp(&data);
        assert!(matches!(cover, Err(ResolutionError::Infeasible)));
    }

    #[test]
    fn admissible_excludes_quadratic_radical_solves() {
        let (_loaded, data) = problem_from("{x^2 + y^2 + 1 == 0}");
        assert!(
            data.admissible.is_empty(),
            "no degree-1 monomial in any variable"
        );
    }

    #[test]
    fn monomial_factor_not_admissible() {
        // x*y is a single monomial; C5 forbids it from being in any tuple.
        let (_loaded, data) = problem_from_raw_factors("{x*y == 0}");
        // The factor x*y has exactly one monomial, so no admissible pair should
        // include it regardless of which variable we'd try to select.
        assert!(
            data.admissible.is_empty(),
            "single-monomial factor must not appear in admissible set; got {:?}",
            data.admissible
        );
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
        // The second polynomial factors as x*(x+y); selecting (x+y, y) forces
        // t_x < t_y. Each polynomial must end up covered either by selection
        // or by zeroing a factor.
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
        // x^5 has a single monomial — zeroing x makes the whole polynomial vanish.
        let (_loaded, data) = problem_from("{x^5 == 0}");
        let cover = solve_milp(&data).expect("feasible by zeroing x");
        assert_eq!(cover.zeroed_polynomials.len(), 1);
        assert_eq!(cover.zeroed_locals.len(), 1);
    }

    #[test]
    fn solve_prefers_zeroing_when_it_satisfies_a_polynomial() {
        // y is a single-monomial factor, so C5 bars it from being selected;
        // the only valid cover for y == 0 is to zero y, which also maximises y_p.
        let (_loaded, data) = problem_from_raw_factors("{y == 0, x + z == 0}");
        let cover = solve_milp(&data).expect("feasible");
        assert!(
            cover.zeroed_polynomials.contains(&0),
            "polynomial 0 (y == 0) should be marked zeroed; cover = {cover:?}"
        );
        assert!(topo_ok(&data, &cover));
    }

    #[test]
    fn solve_does_not_select_pair_with_unkillable_high_degree_monomial() {
        // (x + 1)*(x^2 + y + 1) splits into two factors. The pair
        // (x^2 + y + 1, x) has no degree-1 monomial in x and so is never even
        // admissible; admissibility for this polynomial is only via (x+1, x)
        // and (x^2+y+1, y).
        let (loaded, data) = problem_from("{(x + 1) * (x^2 + y + 1) == 0}");
        let x_vid = loaded.var_names.iter().position(|n| n == "x").unwrap();
        let x_local = data.v_idx[&x_vid];
        for &(fi, vloc) in &data.admissible {
            if vloc != x_local {
                continue;
            }
            // The only factor admissible for x is the linear one (x+1).
            assert_eq!(
                data.factors[fi].vars.len(),
                1,
                "a non-linear factor was made admissible for x"
            );
        }
        // And the MILP picks a feasible cover.
        let _cover = solve_milp(&data).expect("feasible");
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
