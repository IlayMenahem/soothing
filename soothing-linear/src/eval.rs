//! Topological evaluation of a DAG (children before parents).
//!
//! Node kinds: constant, variable, add, mul, neg, and also **`Div`** and **`Pow`** for rational
//! expressions and integer powers.

use crate::dag::{Dag, Node, NodeId, VarId};
use crate::field::{Q, pi_approx_q};
use num_traits::{Pow, Zero};
use std::collections::HashMap;
use std::collections::HashSet;
use std::hash::BuildHasher;
use std::panic::{AssertUnwindSafe, catch_unwind};

/// Evaluation error.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EvalError {
    MissingVariable(VarId),
    DivisionByZero,
    NonIntegerExponent,
}

/// Evaluate the expression at `root` using `assignment` for variables.
///
/// # Errors
///
/// Returns [`EvalError::MissingVariable`] if a variable in the subgraph is absent from `assignment`,
/// [`EvalError::DivisionByZero`] for a zero divisor, and [`EvalError::NonIntegerExponent`] when a
/// power requires a non-integer exponent in `Q`.
///
/// [`EvalError::DivisionByZero`] is also returned if exact `i128` rational arithmetic unwinds (for
/// example overflow during reduction), so predicates can skip a sample instead of aborting search.
pub fn evaluate<S: BuildHasher>(
    dag: &Dag,
    root: NodeId,
    assignment: &HashMap<VarId, Q, S>,
) -> Result<Q, EvalError> {
    match catch_unwind(AssertUnwindSafe(|| evaluate_unwind(dag, root, assignment))) {
        Ok(r) => r,
        Err(_) => Err(EvalError::DivisionByZero),
    }
}

fn evaluate_unwind<S: BuildHasher>(
    dag: &Dag,
    root: NodeId,
    assignment: &HashMap<VarId, Q, S>,
) -> Result<Q, EvalError> {
    let order = topological_order(dag, root);
    let mut values: Vec<Option<Q>> = vec![None; dag.len()];

    order.iter().copied().try_for_each(|nid| {
        let v = eval_node(dag, nid, assignment, &values)?;
        values[nid.0] = Some(v);
        Ok::<_, EvalError>(())
    })?;

    Ok(values[root.0]
        .take()
        .expect("root must appear in topological order"))
}

fn eval_node<S: BuildHasher>(
    dag: &Dag,
    id: NodeId,
    assignment: &HashMap<VarId, Q, S>,
    values: &[Option<Q>],
) -> Result<Q, EvalError> {
    match dag.node(id) {
        Node::Const(c) => Ok(*c),
        Node::ConstPi => Ok(pi_approx_q()),
        Node::Var(x) => assignment
            .get(x)
            .cloned()
            .ok_or(EvalError::MissingVariable(*x)),
        Node::Add(children) => Ok(children
            .iter()
            .map(|c| values[c.0].as_ref().expect("topo order"))
            .sum()),
        Node::Mul(children) => Ok(children
            .iter()
            .map(|c| values[c.0].as_ref().expect("topo order"))
            .product()),
        Node::Neg(x) => Ok(-values[x.0].as_ref().expect("topo order")),
        Node::Div(n, d) => {
            let num = values[n.0].as_ref().expect("topo order");
            let den = values[d.0].as_ref().expect("topo order");
            if den.is_zero() {
                return Err(EvalError::DivisionByZero);
            }
            Ok(num / den)
        }
        Node::Pow(b, e) => {
            let base = values[b.0].as_ref().expect("topo order");
            let exp = values[e.0].as_ref().expect("topo order");
            if !exp.is_integer() {
                return Err(EvalError::NonIntegerExponent);
            }
            let k = *exp.numer();
            if k < 0 && base.is_zero() {
                return Err(EvalError::DivisionByZero);
            }
            let exp_i32 = i32::try_from(k).map_err(|_| EvalError::DivisionByZero)?;
            Ok(Pow::pow(*base, exp_i32))
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

fn dfs_post(dag: &Dag, id: NodeId, visited: &mut HashSet<NodeId>, out: &mut Vec<NodeId>) {
    if !visited.insert(id) {
        return;
    }
    match dag.node(id) {
        Node::Const(_) | Node::ConstPi | Node::Var(_) => {}
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
        Node::Pow(b, e) => {
            dfs_post(dag, *b, visited, out);
            dfs_post(dag, *e, visited, out);
        }
    }
    out.push(id);
}
