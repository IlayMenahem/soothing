//! Iterative linear elimination by substitution; [`dfs_eliminate`] and [`beam_eliminate`] search elimination orders.

use crate::dag::{Dag, DagBuilder, Node, NodeId, VarId};
use crate::field::{q_one, q_zero};
use crate::predicates::{
    PredicateError, SampleConfig, expr_finite_at_some_random_sample, is_linear_in, is_zero,
};
use crate::simplify::{SimplifyError, simplify};
use crate::substitution::{SubstitutionError, SubstitutionMap};
use crate::system::PolynomialSystem;
use rayon::prelude::*;
use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::io::Write;
use std::time::Instant;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum EliminationError {
    #[error("beam_width must be at least 1 (got {got})")]
    InvalidBeamWidth { got: usize },
    #[error("illegitimate elimination: a=0 and b≠0 for variable {var:?}")]
    IllegitimateElimination { var: VarId },
    #[error(
        "degenerate substitution: RHS never evaluates successfully at random exact samples (variable {var:?})"
    )]
    DegenerateSubstitution { var: VarId },
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Predicate(#[from] crate::predicates::PredicateError),
    #[error(transparent)]
    Simplify(#[from] SimplifyError),
}

#[derive(Default)]
struct EliminationTraceState {
    last_tick: Option<Instant>,
    dfs_visits: u64,
}

#[derive(Default)]
struct BeamTraceState {
    last_tick: Option<Instant>,
    layer: u64,
}

/// Residual polynomial system and substitution map produced by elimination.
pub struct EliminationResult {
    /// Equations remaining after elimination.
    pub reduced_system: PolynomialSystem,
    /// Map from eliminated variables to defining expressions.
    pub substitution: SubstitutionMap,
}

/// `true` when the residual system has no equations — default DFS acceptance predicate.
#[must_use]
pub fn default_is_solution(r: &EliminationResult) -> bool {
    r.reduced_system.equations.is_empty()
}

/// Like [`default_is_solution`], but if `kappah` appears in the substitution map, reject when its
/// fully expanded RHS (after [`SubstitutionMap::replay_order`] substitutions) is identically zero.
#[must_use]
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
            is_zero(&snap, expanded_root, &vs, cfg).unwrap_or_default()
        }

        Err(SubstitutionError::Cycle) => return false,
    };

    !expanded_zero
}

type ExpandedSubstitution = (Vec<VarId>, DagBuilder, HashMap<VarId, NodeId>);

/// Expand each substituted RHS by applying earlier bindings in [`SubstitutionMap::replay_order`]
/// (same construction as elimination validation).
fn expand_substitution_bindings(
    result: &EliminationResult,
) -> Result<ExpandedSubstitution, SubstitutionError> {
    let order = result.substitution.replay_order()?;
    let subst_dag = result.substitution.dag();
    let mut b = DagBuilder::new();
    let mut expanded_rhs: HashMap<VarId, NodeId> = HashMap::new();
    order.iter().copied().for_each(|v| {
        let rhs = result
            .substitution
            .get(v)
            .expect("expand_substitution_bindings: missing binding for variable in replay order");

        let mut node = b.merge_root(&subst_dag, rhs);
        order.iter().copied().take_while(|&u| u != v).for_each(|u| {
            if let Some(&u_root) = expanded_rhs.get(&u) {
                let snap_inner = b.snapshot();
                node = b.substitute_in_root(&snap_inner, node, u, u_root);
            }
        });
        expanded_rhs.insert(v, node);
    });
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

    original.equations.iter().enumerate().for_each(|(i, &eq_root)| {
        let mut r = b.merge_root(&original.dag, eq_root);
        order.iter().copied().for_each(|v| {
            let repl = *expanded_rhs
                .get(&v)
                .expect("elimination validation: expanded binding missing");
            let snap = b.snapshot();
            r = b.substitute_in_root(&snap, r, v, repl);
        });
        let snap = b.snapshot();
        let free_vars = snap.vars_in_root(r);
        let ok = match is_zero(&snap, r, &free_vars, cfg) {
            Ok(z) => z,
            Err(e) => panic!(
                "elimination validation: predicate error on original equation index {i}: {e}"
            ),
        };
        assert!(
            ok,
            "elimination validation failed: original equation index {i} does not vanish under substitution"
        );
    });
}

/// Run linear elimination until no equation is linear in any remaining variable.
///
/// At each step the **first** linear pair `(equation, variable)` in a fixed deterministic order is
/// chosen (greedy). For search over orders, see [`dfs_eliminate`].
///
/// # Errors
///
/// Returns [`EliminationError`] when a predicate fails ([`EliminationError::Predicate`]), an
/// elimination step is illegitimate or degenerate, or I/O fails when tracing is enabled.
pub fn linear_elimination_greedy(
    system: &PolynomialSystem,
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
    simplify_equation_roots(&mut b, &mut eq_roots)?;
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

        match apply_one_elimination(&mut b, &mut eq_roots, &mut subst, x, i, &remaining, cfg) {
            Ok(()) => {}
            Err(EliminationError::Simplify(SimplifyError::DivisionByZero)) => continue,
            Err(e) => return Err(e),
        }
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

/// Depth-first search over linear elimination choices: at each step, consider every `(equation, variable)`
/// pair that is linear, rank successful one-step successors by [`cmp_residual_preference`] (same
/// ordering as [`beam_eliminate`] before width truncation), then recurse in that order.
///
/// **Ranking:** fewer residual equations wins; tie-break: **more** distinct free variables in the
/// residual, matching the appendix and beam search.
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
/// [`default_is_solution`] unless you supply a custom predicate.
///
/// # Errors
///
/// Same as [`linear_elimination_greedy`]: [`EliminationError`] from predicates, bad elimination
/// steps, or I/O.
pub fn dfs_eliminate<F>(
    system: &PolynomialSystem,
    cfg: &SampleConfig,
    trace_writer: Option<&mut dyn Write>,
    is_solution: F,
) -> Result<EliminationResult, EliminationError>
where
    F: Fn(&EliminationResult) -> bool,
{
    let original = system.clone();
    let mut b = DagBuilder::new();
    let mut eq_roots: Vec<NodeId> = system
        .equations
        .iter()
        .map(|&r| b.merge_root(&system.dag, r))
        .collect();
    simplify_equation_roots(&mut b, &mut eq_roots)?;
    let subst = SubstitutionMap::new();
    let result = match trace_writer {
        None => dfs_eliminate_inner(&b, eq_roots, subst, cfg, None, None, &is_solution)?,
        Some(w) => {
            let trace_w = RefCell::new(w);
            let trace_st = RefCell::new(EliminationTraceState::default());
            dfs_eliminate_inner(
                &b,
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

fn dfs_eliminate_inner<F>(
    b: &DagBuilder,
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

    let mut children: Vec<(DagBuilder, Vec<NodeId>, SubstitutionMap, VarId, usize)> = Vec::new();
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
            Ok(()) => children.push((b_branch, eq_branch, subst_branch, x, i)),
            Err(
                EliminationError::IllegitimateElimination { .. }
                | EliminationError::DegenerateSubstitution { .. }
                | EliminationError::Simplify(SimplifyError::DivisionByZero),
            ) => continue,
            Err(e) => return Err(e),
        }
    }

    children.sort_by(|(b1, e1, _, _, _), (b2, e2, _, _, _)| cmp_elimination_state(b1, e1, b2, e2));

    for (b_branch, eq_branch, subst_branch, x, i) in children {
        if let Some(cw) = trace_w {
            let mut out = cw.borrow_mut();
            writeln!(&mut *out, "branch var={x:?} eq_index={i}")?;
            out.flush()?;
        }

        let r = dfs_eliminate_inner(
            &b_branch,
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

/// Compare two residual systems for search preference: fewer equations wins; tie-break: more
/// distinct variables appearing in the residual equations wins.
fn cmp_residual_preference(
    dag_a: &Dag,
    roots_a: &[NodeId],
    dag_b: &Dag,
    roots_b: &[NodeId],
) -> Ordering {
    let na = roots_a.len();
    let nb = roots_b.len();
    match na.cmp(&nb) {
        Ordering::Equal => {
            let free_a = collect_vars(dag_a, roots_a).len();
            let free_b = collect_vars(dag_b, roots_b).len();
            free_b.cmp(&free_a)
        }
        o => o,
    }
}

fn cmp_elimination_state(
    b1: &DagBuilder,
    e1: &[NodeId],
    b2: &DagBuilder,
    e2: &[NodeId],
) -> Ordering {
    let d1 = b1.snapshot();
    let d2 = b2.snapshot();
    cmp_residual_preference(&d1, e1, &d2, e2)
}

fn sort_states_by_residual_preference(
    states: &mut [(DagBuilder, Vec<NodeId>, SubstitutionMap)],
) {
    states.sort_by(|(b1, e1, _), (b2, e2, _)| cmp_elimination_state(b1, e1, b2, e2));
}

fn elimination_result_from_state(
    builder: &DagBuilder,
    eq_roots: &[NodeId],
    subst: &SubstitutionMap,
) -> EliminationResult {
    let snap = builder.snapshot();
    EliminationResult {
        reduced_system: PolynomialSystem::new(snap, eq_roots.to_vec()),
        substitution: subst.clone(),
    }
}

fn consider_best_elimination(best: &mut Option<EliminationResult>, candidate: EliminationResult) {
    match best {
        None => *best = Some(candidate),
        Some(b) if is_better_elimination(&candidate, b) => *best = Some(candidate),
        Some(_) => {}
    }
}

/// Beam search over linear elimination: each layer applies one elimination step from every beam
/// state, then keeps the best `beam_width` successors ranked by [`cmp_residual_preference`]
/// (fewer residual equations at the same depth; same tie-break as DFS).
///
/// **Early exit:** if any expanded state satisfies `is_solution`, return it immediately.
///
/// **Termination:** when the beam cannot expand further, returns the best terminal or partial state
/// seen (same aggregate preference as DFS).
///
/// # Errors
///
/// [`EliminationError::InvalidBeamWidth`] when `beam_width == 0`. Otherwise same as
/// [`linear_elimination_greedy`].
pub fn beam_eliminate<F>(
    system: &PolynomialSystem,
    cfg: &SampleConfig,
    beam_width: usize,
    trace_writer: Option<&mut dyn Write>,
    is_solution: F,
) -> Result<EliminationResult, EliminationError>
where
    F: Fn(&EliminationResult) -> bool,
{
    if beam_width < 1 {
        return Err(EliminationError::InvalidBeamWidth { got: beam_width });
    }

    let original = system.clone();
    let mut b = DagBuilder::new();
    let mut eq_roots: Vec<NodeId> = system
        .equations
        .iter()
        .map(|&r| b.merge_root(&system.dag, r))
        .collect();
    simplify_equation_roots(&mut b, &mut eq_roots)?;
    let subst = SubstitutionMap::new();
    let result = match trace_writer {
        None => beam_eliminate_inner(
            beam_width,
            cfg,
            None,
            None,
            &is_solution,
            vec![(b, eq_roots, subst)],
        )?,
        Some(w) => {
            let trace_w = RefCell::new(w);
            let trace_st = RefCell::new(BeamTraceState::default());
            beam_eliminate_inner(
                beam_width,
                cfg,
                Some(&trace_w),
                Some(&trace_st),
                &is_solution,
                vec![(b, eq_roots, subst)],
            )?
        }
    };
    validate_elimination_result(&original, &result, cfg);
    Ok(result)
}

fn beam_eliminate_inner<F>(
    beam_width: usize,
    cfg: &SampleConfig,
    trace_w: Option<&RefCell<&mut dyn Write>>,
    trace_st: Option<&RefCell<BeamTraceState>>,
    is_solution: &F,
    mut beam: Vec<(DagBuilder, Vec<NodeId>, SubstitutionMap)>,
) -> Result<EliminationResult, EliminationError>
where
    F: Fn(&EliminationResult) -> bool,
{
    let mut best: Option<EliminationResult> = None;

    loop {
        if beam.is_empty() {
            return Ok(best.expect("beam_eliminate_inner: empty beam with no best"));
        }

        let beam_outcomes: Vec<EliminationResult> = beam
            .iter()
            .map(|(b, eq, s)| elimination_result_from_state(b, eq, s))
            .collect();

        if let (Some(cw), Some(cs)) = (trace_w, trace_st) {
            let mut st = cs.borrow_mut();
            let mut out = cw.borrow_mut();
            st.layer += 1;
            let elapsed = st.last_tick.map(|t| t.elapsed()).unwrap_or_default();
            st.last_tick = Some(Instant::now());
            let eq_in: Vec<usize> = beam.iter().map(|(_, eq, _)| eq.len()).collect();
            writeln!(
                &mut *out,
                "beam layer={} beam_in={} eq_counts={eq_in:?} elapsed: {:.6}s",
                st.layer,
                beam.len(),
                elapsed.as_secs_f64()
            )?;
            out.flush()?;
        }

        let mut successors: Vec<(DagBuilder, Vec<NodeId>, SubstitutionMap)> = Vec::new();

        for (b, eq_roots, subst) in beam.drain(..) {
            let snap = b.snapshot();

            if eq_roots.is_empty() {
                let r = EliminationResult {
                    reduced_system: PolynomialSystem::new(snap, eq_roots),
                    substitution: subst,
                };
                if is_solution(&r) {
                    return Ok(r);
                }
                consider_best_elimination(&mut best, r);
                continue;
            }

            let remaining: BTreeSet<VarId> = collect_vars(&snap, &eq_roots).into_iter().collect();
            let pairs = collect_linear_pairs(&snap, &eq_roots, &remaining, cfg);

            if pairs.is_empty() {
                let r = EliminationResult {
                    reduced_system: PolynomialSystem::new(snap, eq_roots),
                    substitution: subst,
                };
                if is_solution(&r) {
                    return Ok(r);
                }
                consider_best_elimination(&mut best, r);
                continue;
            }

            for (x, i) in pairs {
                let mut b_branch = b.clone();
                let mut eq_branch = eq_roots.clone();
                let mut subst_branch = subst.clone();
                let remaining_branch: BTreeSet<VarId> =
                    collect_vars(&b_branch.snapshot(), &eq_branch)
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
                    Err(
                        EliminationError::IllegitimateElimination { .. }
                        | EliminationError::DegenerateSubstitution { .. }
                        | EliminationError::Simplify(SimplifyError::DivisionByZero),
                    ) => continue,
                    Err(e) => return Err(e),
                }

                let r = elimination_result_from_state(&b_branch, &eq_branch, &subst_branch);
                if is_solution(&r) {
                    return Ok(r);
                }

                let snap_b = b_branch.snapshot();
                let remaining2: BTreeSet<VarId> =
                    collect_vars(&snap_b, &eq_branch).into_iter().collect();
                let pairs2 = collect_linear_pairs(&snap_b, &eq_branch, &remaining2, cfg);

                if eq_branch.is_empty() || pairs2.is_empty() {
                    consider_best_elimination(&mut best, r);
                } else {
                    successors.push((b_branch, eq_branch, subst_branch));
                }
            }
        }

        if successors.is_empty() {
            for r in beam_outcomes {
                consider_best_elimination(&mut best, r);
            }
            return Ok(best.expect("beam_eliminate_inner: no best after empty successors"));
        }

        let generated = successors.len();
        sort_states_by_residual_preference(&mut successors);
        successors.truncate(beam_width);
        let eq_kept: Vec<usize> = successors.iter().map(|(_, eq, _)| eq.len()).collect();

        if let (Some(cw), Some(_cs)) = (trace_w, trace_st) {
            let mut out = cw.borrow_mut();
            writeln!(
                &mut *out,
                "beam layer_kept={} eq_counts={eq_kept:?} (of {} generated successors)",
                successors.len(),
                generated
            )?;
            out.flush()?;
        }

        beam = successors;
    }
}

/// Candidates where `is_linear_in` is true. Enumeration matches Python
/// `linear_substitution_order._linear_substitution_candidates`: for each equation index in order,
/// then each variable appearing in that equation, sorted by [`VarId`] (aligned with sorted
/// `LoadedMathematicaSystem::var_names` and `SymPy` `default_sort_key` on simple identifiers).
///
/// Pairs for which the sampling predicate hard-errors are skipped.
fn collect_linear_pairs(
    dag: &Dag,
    eq_roots: &[NodeId],
    remaining: &BTreeSet<VarId>,
    cfg: &SampleConfig,
) -> Vec<(VarId, usize)> {
    let candidates: Vec<(usize, VarId)> = (0..eq_roots.len())
        .flat_map(|i| {
            let mut vars_in_eq: Vec<VarId> = collect_vars(dag, std::slice::from_ref(&eq_roots[i]))
                .into_iter()
                .filter(|v| remaining.contains(v))
                .collect();
            vars_in_eq.sort_by_key(|v| v.0);
            vars_in_eq.into_iter().map(move |x| (i, x))
        })
        .collect();
    let linear_ok: Vec<bool> = candidates
        .par_iter()
        .map(|&(i, x)| {
            let others = other_vars(remaining, x);
            matches!(is_linear_in(dag, eq_roots[i], x, &others, cfg), Ok(true))
        })
        .collect();
    candidates
        .iter()
        .zip(&linear_ok)
        .filter_map(|(&(i, x), &ok)| ok.then_some((x, i)))
        .collect()
}

fn is_better_elimination(a: &EliminationResult, b: &EliminationResult) -> bool {
    matches!(
        cmp_residual_preference(
            &a.reduced_system.dag,
            &a.reduced_system.equations,
            &b.reduced_system.dag,
            &b.reduced_system.equations,
        ),
        Ordering::Less
    )
}

fn other_vars(remaining: &BTreeSet<VarId>, x: VarId) -> Vec<VarId> {
    remaining.iter().filter(|&&v| v != x).copied().collect()
}

/// Drop equations that the sampling predicate classifies as identically zero. On predicate error,
/// keep the equation.
fn simplify_equation_roots(
    builder: &mut DagBuilder,
    eq_roots: &mut Vec<NodeId>,
) -> Result<(), SimplifyError> {
    let snap = builder.snapshot();
    let mut new_roots = Vec::with_capacity(eq_roots.len());
    for &r in eq_roots.iter() {
        let id = match simplify(builder, &snap, r) {
            Ok(id) => id,
            Err(SimplifyError::DivisionByZero) => builder.merge_root(&snap, r),
            Err(e) => return Err(e),
        };
        new_roots.push(id);
    }
    *eq_roots = new_roots;
    Ok(())
}

fn retain_nonzero_equations(dag: &Dag, eq_roots: &mut Vec<NodeId>, cfg: &SampleConfig) {
    let keep: Vec<bool> = eq_roots
        .par_iter()
        .map(|&r| {
            let vs: Vec<VarId> = collect_vars(dag, std::slice::from_ref(&r))
                .into_iter()
                .collect();
            match is_zero(dag, r, &vs, cfg) {
                Ok(true) => false,
                Ok(false) | Err(_) => true,
            }
        })
        .collect();
    *eq_roots = eq_roots
        .iter()
        .zip(&keep)
        .filter_map(|(&r, &k)| k.then_some(r))
        .collect();
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
        *eq_roots = eq_roots
            .iter()
            .enumerate()
            .filter_map(|(j, &er)| (j != i).then_some(er))
            .collect();
        let snap_taut = builder.snapshot();
        retain_nonzero_equations(&snap_taut, eq_roots, cfg);
        simplify_equation_roots(builder, eq_roots)?;
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

    *eq_roots = eq_roots
        .iter()
        .enumerate()
        .filter(|&(j, _)| j != i)
        .map(|(_, &er)| builder.substitute_in_root(&snap3, er, x, x_sol))
        .collect();

    let snap4 = builder.snapshot();
    retain_nonzero_equations(&snap4, eq_roots, cfg);
    simplify_equation_roots(builder, eq_roots)?;
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
    use super::{beam_eliminate, cmp_residual_preference, is_better_elimination};
    use crate::dag::DagBuilder;
    use crate::substitution::SubstitutionMap;
    use crate::system::PolynomialSystem;
    use crate::{EliminationError, EliminationResult, SampleConfig, VarId};
    use std::cmp::Ordering;

    use super::default_is_solution;

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
        let zero = subst_builder.constant(crate::Q::from_integer(0));
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

        assert_eq!(
            cmp_residual_preference(
                &left.reduced_system.dag,
                &left.reduced_system.equations,
                &right.reduced_system.dag,
                &right.reduced_system.equations,
            ),
            Ordering::Less
        );
    }

    #[test]
    fn beam_width_zero_errors() {
        let mut b = DagBuilder::new();
        let x = b.variable(VarId(0));
        let sys = PolynomialSystem::new(b.build(), vec![x]);
        let cfg = SampleConfig::default();
        assert!(matches!(
            beam_eliminate(&sys, &cfg, 0, None, default_is_solution),
            Err(EliminationError::InvalidBeamWidth { got: 0 })
        ));
    }

    #[test]
    fn beam_smoke_one_linear_equation() {
        let mut b = DagBuilder::new();
        let x = VarId(0);
        let y = VarId(1);
        let vx = b.variable(x);
        let vy = b.variable(y);
        let eq = b.add(vec![vx, vy]);
        let sys = PolynomialSystem::new(b.build(), vec![eq]);
        let cfg = SampleConfig::default();
        let r = beam_eliminate(&sys, &cfg, 2, None, default_is_solution).expect("beam ok");
        assert!(
            r.reduced_system.equations.is_empty() || !r.substitution.is_empty(),
            "expected full elimination or at least one substitution"
        );
    }
}
