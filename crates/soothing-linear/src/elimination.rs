//! Iterative linear elimination (paper Algorithm “Linear elimination by substitution”).

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
    let mut remaining: BTreeSet<VarId> = collect_vars(&b.snapshot(), &eq_roots)
        .into_iter()
        .collect();

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
        eq_roots = new_eqs;
        remaining.remove(&x);

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
        remaining = collect_vars(&b.snapshot(), &eq_roots)
            .into_iter()
            .collect();
    }

    let final_dag = b.snapshot();
    Ok(EliminationResult {
        system: PolynomialSystem::new(final_dag, eq_roots),
        substitution: subst,
    })
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
        }
    }
    out
}
