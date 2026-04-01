//! Iterative linear elimination by substitution; [`dfs_linear_elimination`] searches elimination orders.

use crate::dag::{Dag, DagBuilder, Node, NodeId, VarId};
use crate::field::{q_one, q_zero};
use crate::mathematica_system::LoadedMathematicaSystem;
use crate::predicates::{
    expr_finite_at_some_random_sample, is_linear_in, is_zero, PredicateError, SampleConfig,
};
use crate::substitution::{SubstitutionError, SubstitutionMap};
use crate::system::PolynomialSystem;
use rayon::prelude::*;
use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::io::Write;
use std::time::Instant;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum EliminationError {
    #[error("missing variable \"kappah\" in loaded system (required for solving_symbolic-style elimination)")]
    MissingKappah,
    #[error("illegitimate elimination: a=0 and b≠0 for variable {var:?}")]
    IllegitimateElimination { var: VarId },
    #[error("degenerate substitution: RHS never evaluates successfully at random exact samples (variable {var:?})")]
    DegenerateSubstitution { var: VarId },
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Predicate(#[from] crate::predicates::PredicateError),
}

#[derive(Default)]
struct EliminationTraceState {
    last_tick: Option<Instant>,
    dfs_visits: u64,
}

/// Residual polynomial system and substitution map produced by elimination.
pub struct EliminationResult {
    /// Equations remaining after elimination.
    pub reduced_system: PolynomialSystem,
    /// Map from eliminated variables to defining expressions.
    pub substitution: SubstitutionMap,
}

/// `true` when the residual system has no equations — default DFS acceptance predicate.
pub fn default_is_solution(r: &EliminationResult) -> bool {
    r.reduced_system.equations.is_empty()
}

/// Specialized symbolic acceptance predicate used to mirror the Python / SymPy solving stack.
///
/// This is an implementation-specific extension, not part of the paper's default elimination
/// contract. It behaves like [`default_is_solution`], but if `kappah` appears in the substitution
/// map, reject when its **fully expanded** RHS (after applying other bindings in
/// [`SubstitutionMap::replay_order`]) is identically zero — matching SymPy’s
/// `substitution_map[kappah] == 0` in `soothing/linear_substitution_order.py` (`_is_kappah_zero`).
pub fn is_solution_nonzero_kappah(
    r: &EliminationResult,
    kappah: VarId,
    cfg: &SampleConfig,
) -> bool {
    if !default_is_solution(r) {
        return false;
    }
    if !r.substitution.contains(kappah) {
        return true;
    }
    let expanded_zero = match expand_substitution_bindings(r) {
        Ok((_order, b, expanded_rhs)) => {
            let Some(&expanded_root) = expanded_rhs.get(&kappah) else {
                return false;
            };
            let snap = b.snapshot();
            let vs: Vec<VarId> = collect_vars(&snap, std::slice::from_ref(&expanded_root))
                .into_iter()
                .collect();
            match is_zero(&snap, expanded_root, &vs, cfg) {
                Ok(z) => z,
                Err(_) => false,
            }
        }
        Err(SubstitutionError::Cycle) => return false,
    };

    !expanded_zero
}

/// Expand each substituted RHS by applying earlier bindings in [`SubstitutionMap::replay_order`]
/// (same construction as elimination validation).
fn expand_substitution_bindings(
    result: &EliminationResult,
) -> Result<(Vec<VarId>, DagBuilder, HashMap<VarId, NodeId>), SubstitutionError> {
    let order = result.substitution.replay_order()?;
    let subst_dag = result.substitution.dag();
    let mut b = DagBuilder::new();
    let mut expanded_rhs: HashMap<VarId, NodeId> = HashMap::new();
    for &v in &order {
        let rhs = result
            .substitution
            .get(v)
            .expect("expand_substitution_bindings: missing binding for variable in replay order");
        let mut node = b.merge_root(&subst_dag, rhs);
        for &u in &order {
            if u == v {
                break;
            }
            if let Some(&u_root) = expanded_rhs.get(&u) {
                let snap_inner = b.snapshot();
                node = b.substitute_in_root(&snap_inner, node, u, u_root);
            }
        }
        expanded_rhs.insert(v, node);
    }
    Ok((order, b, expanded_rhs))
}

/// When the residual is empty and the substitution map is non-empty, require each original
/// equation to vanish after applying [`SubstitutionMap::replay_order`] (aligned with
/// `validate_substitution_map` in `soothing/system_substitution.py`). If equations remain, the
/// substitution is only meaningful modulo the residual, so this static check is skipped.
fn validate_elimination_result(
    original: &PolynomialSystem,
    result: &EliminationResult,
    cfg: &SampleConfig,
) {
    if result.substitution.is_empty() || !result.reduced_system.equations.is_empty() {
        return;
    }
    let (order, mut b, expanded_rhs) = expand_substitution_bindings(result)
        .expect("elimination validation: cyclic substitution map");

    for (i, &eq_root) in original.equations.iter().enumerate() {
        let mut r = b.merge_root(&original.dag, eq_root);
        for &v in &order {
            let repl = *expanded_rhs
                .get(&v)
                .expect("elimination validation: expanded binding missing");
            let snap = b.snapshot();
            r = b.substitute_in_root(&snap, r, v, repl);
        }
        let snap = b.snapshot();
        let free_vars = snap.vars_in_root(r);
        let ok = match is_zero(&snap, r, &free_vars, cfg) {
            Ok(z) => z,
            Err(e) => panic!(
                "elimination validation: predicate error on original equation index {i}: {e}"
            ),
        };
        if !ok {
            panic!(
                "elimination validation failed: original equation index {i} does not vanish under substitution"
            );
        }
    }
}

/// Run linear elimination until no equation is linear in any remaining variable.
///
/// At each step the **first** linear pair `(equation, variable)` in a fixed deterministic order is
/// chosen (greedy). For search over orders, see [`dfs_linear_elimination`].
pub fn linear_elimination_greedy(
    system: PolynomialSystem,
    cfg: &SampleConfig,
) -> Result<EliminationResult, EliminationError> {
    let original = system.clone();
    let mut b = DagBuilder::new();
    let mut eq_roots: Vec<NodeId> = system
        .equations
        .iter()
        .map(|&r| b.merge_root(&system.dag, r))
        .collect();
    let mut subst = SubstitutionMap::new();
    let mut remaining: BTreeSet<VarId> =
        collect_vars(&b.snapshot(), &eq_roots).into_iter().collect();

    loop {
        let snap = b.snapshot();
        let mut choice: Option<(VarId, usize)> = None;
        'search: for &x in &remaining {
            let others = other_vars(&remaining, x);
            let results: Vec<Result<bool, PredicateError>> = (0..eq_roots.len())
                .into_par_iter()
                .map(|i| is_linear_in(&snap, eq_roots[i], x, &others, cfg))
                .collect();
            for (i, res) in results.into_iter().enumerate() {
                match res {
                    Ok(true) => {
                        choice = Some((x, i));
                        break 'search;
                    }
                    Ok(false) => {}
                    Err(e) => return Err(EliminationError::Predicate(e)),
                }
            }
        }
        let Some((x, i)) = choice else {
            break;
        };

        apply_one_elimination(&mut b, &mut eq_roots, &mut subst, x, i, &remaining, cfg)?;
        remaining = collect_vars(&b.snapshot(), &eq_roots).into_iter().collect();
    }

    let final_dag = b.snapshot();
    let result = EliminationResult {
        reduced_system: PolynomialSystem::new(final_dag, eq_roots),
        substitution: subst,
    };
    validate_elimination_result(&original, &result, cfg);
    Ok(result)
}

/// Depth-first search over linear elimination choices: at each step, try every `(equation, variable)`
/// pair that is linear; prefer fewer remaining equations.
///
/// **Tie-break:** same residual equation count → prefer **more free variables** in the residual
/// system, matching the appendix.
///
/// **Early exit:** if any branch satisfies the solution predicate (default: no residual equations),
/// return that result immediately without trying further siblings at the same level.
///
/// Pairs that would be [`EliminationError::IllegitimateElimination`] or
/// [`EliminationError::DegenerateSubstitution`] are skipped (other branches are still tried).
///
/// When `trace_writer` is `Some`, each DFS node appends timing lines to the writer (flushed after
/// every line) for performance diagnosis.
///
/// `is_solution` decides when a leaf counts as a successful terminal state for early exit; use
/// [`default_is_solution`] unless you need a stricter condition (e.g. [`is_solution_nonzero_kappah`]).
///
/// For the same stack as `python/scripts/solving_symbolic.py` and
/// `tests/python_reference_simplified_mathematica.rs`, use [`dfs_linear_elimination_solving_symbolic`].
pub fn dfs_linear_elimination<F>(
    system: PolynomialSystem,
    cfg: &SampleConfig,
    trace_writer: Option<&mut dyn Write>,
    is_solution: F,
) -> Result<EliminationResult, EliminationError>
where
    F: Fn(&EliminationResult) -> bool,
{
    let original = system.clone();
    let mut b = DagBuilder::new();
    let eq_roots: Vec<NodeId> = system
        .equations
        .iter()
        .map(|&r| b.merge_root(&system.dag, r))
        .collect();
    let subst = SubstitutionMap::new();
    let result = match trace_writer {
        None => dfs_linear_elimination_inner(b, eq_roots, subst, cfg, None, None, &is_solution)?,
        Some(w) => {
            let trace_w = RefCell::new(w);
            let trace_st = RefCell::new(EliminationTraceState::default());
            dfs_linear_elimination_inner(
                b,
                eq_roots,
                subst,
                cfg,
                Some(&trace_w),
                Some(&trace_st),
                &is_solution,
            )?
        }
    };
    validate_elimination_result(&original, &result, cfg);
    Ok(result)
}

/// [`dfs_linear_elimination`] configured like `python/scripts/solving_symbolic.py` and
/// `tests/python_reference_simplified_mathematica.rs`.
///
/// This entry point is kept for symbolic parity with the Python stack; it is not the paper's
/// default elimination API.
pub fn dfs_linear_elimination_solving_symbolic(
    loaded: LoadedMathematicaSystem,
    cfg: &SampleConfig,
    trace_writer: Option<&mut dyn Write>,
) -> Result<EliminationResult, EliminationError> {
    let kappah = loaded
        .var_names
        .iter()
        .position(|n| n == "kappah")
        .map(|i| VarId(i as u32))
        .ok_or(EliminationError::MissingKappah)?;
    dfs_linear_elimination(loaded.system, cfg, trace_writer, |r| {
        is_solution_nonzero_kappah(r, kappah, cfg)
    })
}

fn dfs_linear_elimination_inner<F>(
    b: DagBuilder,
    eq_roots: Vec<NodeId>,
    subst: SubstitutionMap,
    cfg: &SampleConfig,
    trace_w: Option<&RefCell<&mut dyn Write>>,
    trace_st: Option<&RefCell<EliminationTraceState>>,
    is_solution: &F,
) -> Result<EliminationResult, EliminationError>
where
    F: Fn(&EliminationResult) -> bool,
{
    let snap = b.snapshot();

    if let (Some(cw), Some(cs)) = (trace_w, trace_st) {
        let mut st = cs.borrow_mut();
        let mut out = cw.borrow_mut();
        st.dfs_visits += 1;
        let elapsed = st.last_tick.map(|t| t.elapsed()).unwrap_or_default();
        st.last_tick = Some(Instant::now());
        let remaining_eq = eq_roots.len();
        writeln!(
            &mut *out,
            "{} {} elapsed: {:.6}s",
            st.dfs_visits,
            remaining_eq,
            elapsed.as_secs_f64()
        )?;
        out.flush()?;
    }

    if eq_roots.is_empty() {
        return Ok(EliminationResult {
            reduced_system: PolynomialSystem::new(snap, eq_roots),
            substitution: subst,
        });
    }

    let remaining: BTreeSet<VarId> = collect_vars(&snap, &eq_roots).into_iter().collect();
    let pairs = collect_linear_pairs(&snap, &eq_roots, &remaining, cfg);
    if pairs.is_empty() {
        return Ok(EliminationResult {
            reduced_system: PolynomialSystem::new(snap, eq_roots),
            substitution: subst,
        });
    }

    let mut best = EliminationResult {
        reduced_system: PolynomialSystem::new(snap, eq_roots.clone()),
        substitution: subst.clone(),
    };

    let mut branch_skips: usize = 0;
    for (x, i) in pairs {
        let mut b_branch = b.clone();
        let mut eq_branch = eq_roots.clone();
        let mut subst_branch = subst.clone();
        let remaining_branch: BTreeSet<VarId> = collect_vars(&b_branch.snapshot(), &eq_branch)
            .into_iter()
            .collect();
        match apply_one_elimination(
            &mut b_branch,
            &mut eq_branch,
            &mut subst_branch,
            x,
            i,
            &remaining_branch,
            cfg,
        ) {
            Ok(()) => {}
            Err(EliminationError::IllegitimateElimination { .. })
            | Err(EliminationError::DegenerateSubstitution { .. }) => {
                branch_skips += 1;
                continue;
            }
            Err(e) => return Err(e),
        }

        if let Some(cw) = trace_w {
            let mut out = cw.borrow_mut();
            writeln!(
                &mut *out,
                "branch var={:?} eq_index={} skips_before_success={}",
                x, i, branch_skips
            )?;
            out.flush()?;
        }

        let r = dfs_linear_elimination_inner(
            b_branch,
            eq_branch,
            subst_branch,
            cfg,
            trace_w,
            trace_st,
            is_solution,
        )?;
        if is_solution(&r) {
            return Ok(r);
        }
        if is_better_elimination(&r, &best) {
            best = r;
        }
    }

    Ok(best)
}

/// Candidates where `is_linear_in` is true. Enumeration matches Python
/// `linear_substitution_order._linear_substitution_candidates`: for each equation index in order,
/// then each variable appearing in that equation, sorted by [`VarId`] (aligned with sorted
/// `LoadedMathematicaSystem::var_names` and SymPy `default_sort_key` on simple identifiers).
///
/// Pairs for which the sampling predicate hard-errors are skipped.
fn collect_linear_pairs(
    dag: &Dag,
    eq_roots: &[NodeId],
    remaining: &BTreeSet<VarId>,
    cfg: &SampleConfig,
) -> Vec<(VarId, usize)> {
    let mut candidates: Vec<(usize, VarId)> = Vec::new();
    for i in 0..eq_roots.len() {
        let mut vars_in_eq: Vec<VarId> = collect_vars(dag, std::slice::from_ref(&eq_roots[i]))
            .into_iter()
            .filter(|v| remaining.contains(v))
            .collect();
        vars_in_eq.sort_by_key(|v| v.0);
        for x in vars_in_eq {
            candidates.push((i, x));
        }
    }
    let linear_ok: Vec<bool> = candidates
        .par_iter()
        .map(|&(i, x)| {
            let others = other_vars(remaining, x);
            matches!(is_linear_in(dag, eq_roots[i], x, &others, cfg), Ok(true))
        })
        .collect();
    let mut out = Vec::new();
    for (idx, &(i, x)) in candidates.iter().enumerate() {
        if linear_ok[idx] {
            out.push((x, i));
        }
    }
    out
}

fn is_better_elimination(a: &EliminationResult, b: &EliminationResult) -> bool {
    let na = a.reduced_system.equations.len();
    let nb = b.reduced_system.equations.len();
    if na != nb {
        return na < nb;
    }
    let free_a = collect_vars(&a.reduced_system.dag, &a.reduced_system.equations).len();
    let free_b = collect_vars(&b.reduced_system.dag, &b.reduced_system.equations).len();
    free_a > free_b
}

fn other_vars(remaining: &BTreeSet<VarId>, x: VarId) -> Vec<VarId> {
    remaining.iter().filter(|&&v| v != x).copied().collect()
}

/// Drop equations that the sampling predicate classifies as identically zero. On predicate error,
/// keep the equation.
fn retain_nonzero_equations(dag: &Dag, eq_roots: &mut Vec<NodeId>, cfg: &SampleConfig) {
    let keep: Vec<bool> = eq_roots
        .par_iter()
        .map(|&r| {
            let vs: Vec<VarId> = collect_vars(dag, std::slice::from_ref(&r))
                .into_iter()
                .collect();
            match is_zero(dag, r, &vs, cfg) {
                Ok(true) => false,
                Ok(false) => true,
                Err(_) => true,
            }
        })
        .collect();
    let mut new_eqs = Vec::with_capacity(eq_roots.len());
    for (&r, &k) in eq_roots.iter().zip(&keep) {
        if k {
            new_eqs.push(r);
        }
    }
    *eq_roots = new_eqs;
}

fn apply_one_elimination(
    builder: &mut DagBuilder,
    eq_roots: &mut Vec<NodeId>,
    subst: &mut SubstitutionMap,
    x: VarId,
    i: usize,
    remaining: &BTreeSet<VarId>,
    cfg: &SampleConfig,
) -> Result<(), EliminationError> {
    let snap = builder.snapshot();
    let eq_root = eq_roots[i];
    // Affine in x: p = a*x + b with a,b constant in x. Paper: b = p|_{x=0}, a = p|_{x=1} - p|_{x=0}.
    let b_coeff = builder.substitute_var_with_const(&snap, eq_root, x, q_zero());
    let p_at_one = builder.substitute_var_with_const(&snap, eq_root, x, q_one());
    let a_coeff = builder.sub(p_at_one, b_coeff);
    let snap2 = builder.snapshot();
    let free = other_vars(remaining, x);
    let a_is_zero = is_zero(&snap2, a_coeff, &free, cfg)?;
    let b_is_zero = is_zero(&snap2, b_coeff, &free, cfg)?;
    // a = 0, b ≠ 0: cannot solve for x.
    if a_is_zero && !b_is_zero {
        return Err(EliminationError::IllegitimateElimination { var: x });
    }
    // a = 0, b = 0: tautology; drop this equation.
    if a_is_zero && b_is_zero {
        let mut new_eqs = Vec::with_capacity(eq_roots.len().saturating_sub(1));
        for (j, &er) in eq_roots.iter().enumerate() {
            if j != i {
                new_eqs.push(er);
            }
        }
        *eq_roots = new_eqs;
        let snap_taut = builder.snapshot();
        retain_nonzero_equations(&snap_taut, eq_roots, cfg);
        return Ok(());
    }

    // a ≠ 0: x ↦ -b/a = b_coeff / (-a_coeff).
    let neg_a = builder.neg(a_coeff);
    let x_sol = builder.div(b_coeff, neg_a);

    let snap_check = builder.snapshot();
    let x_sol_vars: Vec<VarId> = snap_check.vars_in_root(x_sol);
    if !expr_finite_at_some_random_sample(&snap_check, x_sol, &x_sol_vars, cfg) {
        return Err(EliminationError::DegenerateSubstitution { var: x });
    }

    let snap3 = builder.snapshot();
    subst.insert(x, &snap3, x_sol);

    let mut new_eqs = Vec::with_capacity(eq_roots.len().saturating_sub(1));
    for (j, &er) in eq_roots.iter().enumerate() {
        if j == i {
            continue;
        }
        new_eqs.push(builder.substitute_in_root(&snap3, er, x, x_sol));
    }
    *eq_roots = new_eqs;

    let snap4 = builder.snapshot();
    retain_nonzero_equations(&snap4, eq_roots, cfg);
    Ok(())
}

fn collect_vars(dag: &Dag, roots: &[NodeId]) -> HashSet<VarId> {
    let mut out = HashSet::new();
    let mut stack: Vec<NodeId> = roots.to_vec();
    let mut seen = HashSet::new();
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        match dag.node(id) {
            Node::Var(v) => {
                out.insert(*v);
            }
            Node::Const(_) | Node::ConstPi => {}
            Node::Add(cs) | Node::Mul(cs) => {
                for c in cs {
                    stack.push(*c);
                }
            }
            Node::Neg(x) => stack.push(*x),
            Node::Div(n, d) => {
                stack.push(*n);
                stack.push(*d);
            }
            Node::Pow(b, e) => {
                stack.push(*b);
                stack.push(*e);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::is_better_elimination;
    use crate::dag::DagBuilder;
    use crate::substitution::SubstitutionMap;
    use crate::system::PolynomialSystem;
    use crate::{EliminationResult, VarId};

    #[test]
    fn dfs_tie_break_prefers_more_free_variables() {
        let mut left_builder = DagBuilder::new();
        let x = VarId(0);
        let y = VarId(1);
        let left_x = left_builder.variable(x);
        let left_y = left_builder.variable(y);
        let left_eq = left_builder.add(vec![left_x, left_y]);
        let left_system = PolynomialSystem::new(left_builder.build(), vec![left_eq]);

        let mut right_builder = DagBuilder::new();
        let right_x0 = right_builder.variable(x);
        let right_x1 = right_builder.variable(x);
        let right_eq = right_builder.mul2(right_x0, right_x1);
        let right_system = PolynomialSystem::new(right_builder.build(), vec![right_eq]);

        let mut right_subst = SubstitutionMap::new();
        let mut subst_builder = DagBuilder::new();
        let zero = subst_builder.constant(crate::Q::from_integer(0.into()));
        let subst_dag = subst_builder.build();
        right_subst.insert(x, &subst_dag, zero);
        right_subst.insert(y, &subst_dag, zero);

        let left = EliminationResult {
            reduced_system: left_system,
            substitution: SubstitutionMap::new(),
        };
        let right = EliminationResult {
            reduced_system: right_system,
            substitution: right_subst,
        };

        assert!(
            is_better_elimination(&left, &right),
            "with the same residual equation count, DFS should prefer the state with more free variables"
        );
    }
}
