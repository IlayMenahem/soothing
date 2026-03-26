//! Iterative linear elimination by substitution; [`dfs_linear_elimination`] searches elimination orders.

use crate::dag::{Dag, DagBuilder, Node, NodeId, VarId};
use crate::field::{q_one, q_zero};
use crate::predicates::{expr_finite_at_some_random_sample, is_linear_in, is_zero, SampleConfig};
use crate::substitution::{SubstitutionError, SubstitutionMap};
use crate::system::PolynomialSystem;
use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::io::Write;
use std::time::Instant;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum EliminationError {
    #[error("illegitimate elimination: a=0 and b≠0 for variable {var:?}")]
    IllegitimateElimination { var: VarId },
    #[error("degenerate substitution: RHS never evaluates to a finite float at random samples (variable {var:?})")]
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

/// Full elimination with κ ≠ 0: same as [`default_is_solution`], but if `kappah` appears in the
/// substitution map, reject when its **fully expanded** RHS (after applying other bindings in
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
    // `ConstPi` uses `f64` in predicates; replay validation is only wired for pure-`Q` systems.
    if any_equation_contains_const_pi(original) {
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
            for (i, &eq) in eq_roots.iter().enumerate() {
                match is_linear_in(&snap, eq, x, &others, cfg) {
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
/// **Tie-break:** same equation count → prefer more **distinct variables** appearing in the union
/// of residual equation DAGs (not Jacobian rank or affine dimension).
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

/// Candidates where `is_linear_in` is true. Pairs for which the sampling predicate hard-errors
/// (e.g. missing variable) are omitted so enumeration stays robust when probing every
/// `(equation, variable)` — unlike greedy elimination, which stops at the first success per step.
fn collect_linear_pairs(
    dag: &Dag,
    eq_roots: &[NodeId],
    remaining: &BTreeSet<VarId>,
    cfg: &SampleConfig,
) -> Vec<(VarId, usize)> {
    let mut out = Vec::new();
    for &x in remaining {
        let others = other_vars(remaining, x);
        for (i, &eq) in eq_roots.iter().enumerate() {
            match is_linear_in(dag, eq, x, &others, cfg) {
                Ok(true) => out.push((x, i)),
                Ok(false) => {}
                Err(_) => {}
            }
        }
    }
    out
}

/// Count of distinct variables that appear in **any** residual equation (union of subgraph vars).
/// Used only as a DFS tie-break; not Jacobian rank or Hilbert dimension.
fn distinct_var_count_across_equations(system: &PolynomialSystem) -> usize {
    let mut seen = HashSet::new();
    for &r in &system.equations {
        seen.extend(collect_vars(&system.dag, std::slice::from_ref(&r)));
    }
    seen.len()
}

fn is_better_elimination(a: &EliminationResult, b: &EliminationResult) -> bool {
    let na = a.reduced_system.equations.len();
    let nb = b.reduced_system.equations.len();
    if na != nb {
        return na < nb;
    }
    distinct_var_count_across_equations(&a.reduced_system)
        > distinct_var_count_across_equations(&b.reduced_system)
}

fn other_vars(remaining: &BTreeSet<VarId>, x: VarId) -> Vec<VarId> {
    remaining.iter().filter(|&&v| v != x).copied().collect()
}

/// Drop equations that the sampling predicate classifies as identically zero. On predicate error,
/// keep the equation.
fn retain_nonzero_equations(dag: &Dag, eq_roots: &mut Vec<NodeId>, cfg: &SampleConfig) {
    eq_roots.retain(|&r| {
        let vs: Vec<VarId> = collect_vars(dag, std::slice::from_ref(&r))
            .into_iter()
            .collect();
        match is_zero(dag, r, &vs, cfg) {
            Ok(true) => false,
            Ok(false) => true,
            Err(_) => true,
        }
    });
}

fn apply_one_elimination(
    b: &mut DagBuilder,
    eq_roots: &mut Vec<NodeId>,
    subst: &mut SubstitutionMap,
    x: VarId,
    i: usize,
    remaining: &BTreeSet<VarId>,
    cfg: &SampleConfig,
) -> Result<(), EliminationError> {
    let snap = b.snapshot();
    let eq_root = eq_roots[i];
    let p0 = b.substitute_var_with_const(&snap, eq_root, x, q_zero());
    let p1 = b.substitute_var_with_const(&snap, eq_root, x, q_one());
    let neg_p1 = b.neg(p1);
    let den = b.add(vec![p0, neg_p1]);
    let snap2 = b.snapshot();
    let free = other_vars(remaining, x);
    if is_zero(&snap2, den, &free, cfg)? && !is_zero(&snap2, p0, &free, cfg)? {
        return Err(EliminationError::IllegitimateElimination { var: x });
    }
    // Both linear coefficients vanish: drop the equation (no solvable x).
    if is_zero(&snap2, p0, &free, cfg)? && is_zero(&snap2, den, &free, cfg)? {
        let mut new_eqs = Vec::with_capacity(eq_roots.len().saturating_sub(1));
        for (j, &er) in eq_roots.iter().enumerate() {
            if j != i {
                new_eqs.push(er);
            }
        }
        *eq_roots = new_eqs;
        let snap_taut = b.snapshot();
        retain_nonzero_equations(&snap_taut, eq_roots, cfg);
        return Ok(());
    }

    let x_sol = b.div(p0, den);

    let snap_check = b.snapshot();
    let x_sol_vars: Vec<VarId> = snap_check.vars_in_root(x_sol);
    if !expr_finite_at_some_random_sample(&snap_check, x_sol, &x_sol_vars, cfg) {
        return Err(EliminationError::DegenerateSubstitution { var: x });
    }

    let snap3 = b.snapshot();
    subst.insert(x, &snap3, x_sol);

    let mut new_eqs = Vec::with_capacity(eq_roots.len().saturating_sub(1));
    for (j, &er) in eq_roots.iter().enumerate() {
        if j == i {
            continue;
        }
        new_eqs.push(b.substitute_in_root(&snap3, er, x, x_sol));
    }
    *eq_roots = new_eqs;

    let snap4 = b.snapshot();
    retain_nonzero_equations(&snap4, eq_roots, cfg);
    Ok(())
}

fn any_equation_contains_const_pi(system: &PolynomialSystem) -> bool {
    let dag = &system.dag;
    system
        .equations
        .iter()
        .any(|&r| subtree_contains_const_pi(dag, r))
}

fn subtree_contains_const_pi(dag: &Dag, root: NodeId) -> bool {
    let mut stack = vec![root];
    let mut seen = HashSet::new();
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        match dag.node(id) {
            Node::ConstPi => return true,
            Node::Const(_) | Node::Var(_) => {}
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
    false
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
