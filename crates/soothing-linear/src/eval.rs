//! Topological evaluation of a DAG (paper: evaluate nodes in topological order).

use crate::dag::{Dag, Node, NodeId, VarId};
use crate::field::Q;
use num_traits::{One, Zero};
use std::collections::HashMap;
use std::collections::HashSet;

/// Evaluation error.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EvalError {
    MissingVariable(VarId),
    DivisionByZero,
}

/// Evaluate the expression at `root` using `assignment` for variables.
pub fn evaluate(
    dag: &Dag,
    root: NodeId,
    assignment: &HashMap<VarId, Q>,
) -> Result<Q, EvalError> {
    let order = topological_order(dag, root);
    let mut values: Vec<Option<Q>> = vec![None; dag.len()];
    for &nid in &order {
        let v = eval_node(dag, nid, assignment, &values)?;
        values[nid.0] = Some(v);
    }
    Ok(values[root.0].take().expect("root must appear in topological order"))
}

fn eval_node(
    dag: &Dag,
    id: NodeId,
    assignment: &HashMap<VarId, Q>,
    values: &[Option<Q>],
) -> Result<Q, EvalError> {
    match dag.node(id) {
        Node::Const(c) => Ok(*c),
        Node::Var(x) => assignment
            .get(x)
            .copied()
            .ok_or(EvalError::MissingVariable(*x)),
        Node::Add(children) => {
            let mut s = Q::zero();
            for c in children {
                s += values[c.0].as_ref().expect("topo order");
            }
            Ok(s)
        }
        Node::Mul(children) => {
            let mut p = Q::one();
            for c in children {
                p *= values[c.0].as_ref().expect("topo order");
            }
            Ok(p)
        }
        Node::Neg(x) => Ok(-values[x.0].as_ref().expect("topo order")),
        Node::Div(n, d) => {
            let num = values[n.0].as_ref().expect("topo order");
            let den = values[d.0].as_ref().expect("topo order");
            if den.is_zero() {
                return Err(EvalError::DivisionByZero);
            }
            Ok(num / den)
        }
    }
}

/// Nodes reachable from `root` in reverse topological order (children before parents).
fn topological_order(dag: &Dag, root: NodeId) -> Vec<NodeId> {
    let mut visited = HashSet::new();
    let mut stack: Vec<NodeId> = Vec::new();
    dfs_post(dag, root, &mut visited, &mut stack);
    stack
}

fn dfs_post(
    dag: &Dag,
    id: NodeId,
    visited: &mut HashSet<NodeId>,
    out: &mut Vec<NodeId>,
) {
    if !visited.insert(id) {
        return;
    }
    match dag.node(id) {
        Node::Const(_) | Node::Var(_) => {}
        Node::Add(cs) | Node::Mul(cs) => {
            for c in cs {
                dfs_post(dag, *c, visited, out);
            }
        }
        Node::Neg(x) => dfs_post(dag, *x, visited, out),
        Node::Div(n, d) => {
            dfs_post(dag, *n, visited, out);
            dfs_post(dag, *d, visited, out);
        }
    }
    out.push(id);
}
