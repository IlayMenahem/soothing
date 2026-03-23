//! Iterative linear elimination (paper Algorithm “Linear elimination by substitution”).
//!
//! [`dfs_linear_elimination`] implements Appendix Algorithm “DFS search for a linear elimination
//! order”.

use crate::dag::{Dag, DagBuilder, Node, NodeId, VarId};
use crate::field::{q_one, q_zero};
use crate::predicates::{is_linear_in, is_zero, SampleConfig};
use crate::substitution::SubstitutionMap;
use crate::system::PolynomialSystem;
use std::collections::{BTreeSet, HashSet};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum EliminationError {
    #[error("illegitimate elimination: a=0 and b≠0 for variable {var:?}")]
    IllegitimateElimination { var: VarId },
    #[error(transparent)]
    Predicate(#[from] crate::predicates::PredicateError),
}

pub struct EliminationResult {
    pub system: PolynomialSystem,
    pub substitution: SubstitutionMap,
}

/// Run linear elimination until no equation is linear in any remaining variable.
///
/// This is a greedy depth-first walk: at each step the first linear pair `(equation, variable)` in
/// deterministic order is chosen. For exploration of all branches, see [`dfs_linear_elimination`].
pub fn linear_elimination(
    system: PolynomialSystem,
    cfg: &SampleConfig,
) -> Result<EliminationResult, EliminationError> {
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
            let others: Vec<VarId> = remaining.iter().filter(|&&v| v != x).copied().collect();
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
    Ok(EliminationResult {
        system: PolynomialSystem::new(final_dag, eq_roots),
        substitution: subst,
    })
}

/// DFS over linear elimination choices (paper Appendix: “DFS search for a linear elimination
/// order”). Explores every `(equation, variable)` pair that is linear at each step; prefers a
/// result with fewer remaining equations, breaking ties by more distinct variables in the residual.
///
/// If some branch eliminates all equations, that result is returned immediately. Pairs that would
/// be [`EliminationError::IllegitimateElimination`] are skipped (other branches are still tried).
pub fn dfs_linear_elimination(
    system: PolynomialSystem,
    cfg: &SampleConfig,
) -> Result<EliminationResult, EliminationError> {
    let mut b = DagBuilder::new();
    let eq_roots: Vec<NodeId> = system
        .equations
        .iter()
        .map(|&r| b.merge_root(&system.dag, r))
        .collect();
    let subst = SubstitutionMap::new();
    dfs_linear_elimination_inner(b, eq_roots, subst, cfg)
}

fn dfs_linear_elimination_inner(
    b: DagBuilder,
    eq_roots: Vec<NodeId>,
    subst: SubstitutionMap,
    cfg: &SampleConfig,
) -> Result<EliminationResult, EliminationError> {
    let snap = b.snapshot();
    if eq_roots.is_empty() {
        return Ok(EliminationResult {
            system: PolynomialSystem::new(snap, eq_roots),
            substitution: subst,
        });
    }

    let remaining: BTreeSet<VarId> = collect_vars(&snap, &eq_roots).into_iter().collect();
    let pairs = collect_linear_pairs(&snap, &eq_roots, &remaining, cfg);
    if pairs.is_empty() {
        return Ok(EliminationResult {
            system: PolynomialSystem::new(snap, eq_roots),
            substitution: subst,
        });
    }

    let mut best = EliminationResult {
        system: PolynomialSystem::new(snap, eq_roots.clone()),
        substitution: subst.clone(),
    };

    for (x, i) in pairs {
        let mut b_branch = b.clone();
        let mut eq_branch = eq_roots.clone();
        let mut subst_branch = subst.clone();
        let remaining_branch: BTreeSet<VarId> =
            collect_vars(&b_branch.snapshot(), &eq_branch).into_iter().collect();
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
            Err(EliminationError::IllegitimateElimination { .. }) => continue,
            Err(e) => return Err(e),
        }

        let r = dfs_linear_elimination_inner(b_branch, eq_branch, subst_branch, cfg)?;
        if r.system.equations.is_empty() {
            return Ok(r);
        }
        if is_better_elimination(&r, &best) {
            best = r;
        }
    }

    Ok(best)
}

/// Candidates where `is_linear_in` is true. Pairs for which the sampling predicate errors (e.g.
/// division by zero on a sample) are omitted so enumeration stays robust when probing every
/// `(equation, variable)` — unlike greedy elimination, which stops at the first success per step.
fn collect_linear_pairs(
    dag: &Dag,
    eq_roots: &[NodeId],
    remaining: &BTreeSet<VarId>,
    cfg: &SampleConfig,
) -> Vec<(VarId, usize)> {
    let mut out = Vec::new();
    for &x in remaining {
        let others: Vec<VarId> = remaining.iter().filter(|&&v| v != x).copied().collect();
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

fn free_var_count(system: &PolynomialSystem) -> usize {
    let mut seen = HashSet::new();
    for &r in &system.equations {
        seen.extend(collect_vars(&system.dag, std::slice::from_ref(&r)));
    }
    seen.len()
}

fn is_better_elimination(a: &EliminationResult, b: &EliminationResult) -> bool {
    let na = a.system.equations.len();
    let nb = b.system.equations.len();
    if na != nb {
        return na < nb;
    }
    free_var_count(&a.system) > free_var_count(&b.system)
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
    let free: Vec<VarId> = remaining.iter().filter(|&&v| v != x).copied().collect();
    if is_zero(&snap2, den, &free, cfg)? && !is_zero(&snap2, p0, &free, cfg)? {
        return Err(EliminationError::IllegitimateElimination { var: x });
    }
    let x_sol = b.div(p0, den);

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
    eq_roots.retain(|&r| {
        let vs: Vec<VarId> = collect_vars(&snap4, std::slice::from_ref(&r))
            .into_iter()
            .collect();
        match is_zero(&snap4, r, &vs, cfg) {
            Ok(true) => false,
            Ok(false) => true,
            Err(_) => true,
        }
    });
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
            Node::Const(_) => {}
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
