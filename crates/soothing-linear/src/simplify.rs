//! Local simplifications from the paper appendix (subset implemented in v1).

use crate::dag::{Dag, DagBuilder, Node, NodeId};
use crate::field::{q_one, q_zero};
use num_traits::Zero;
use std::collections::HashMap;

/// Apply one pass of local rewrites: `-(-x)`, strip `+0`, `1·x`, `0·x`.
pub fn simplify_once(builder: &mut DagBuilder, source: &Dag, root: NodeId) -> NodeId {
    let mut memo = HashMap::new();
    simplify_dfs(builder, source, root, &mut memo)
}

fn simplify_dfs(
    b: &mut DagBuilder,
    source: &Dag,
    id: NodeId,
    memo: &mut HashMap<NodeId, NodeId>,
) -> NodeId {
    if let Some(&n) = memo.get(&id) {
        return n;
    }
    let mapped = match source.node(id) {
        Node::Const(c) => b.constant(*c),
        Node::Var(v) => b.variable(*v),
        Node::Neg(x) => {
            let y = simplify_dfs(b, source, *x, memo);
            if let Node::Neg(inner) = b.node(y) {
                *inner
            } else {
                b.neg(y)
            }
        }
        Node::Add(cs) => {
            let parts: Vec<NodeId> = cs
                .iter()
                .map(|c| simplify_dfs(b, source, *c, memo))
                .collect();
            let parts: Vec<NodeId> = parts
                .into_iter()
                .filter(|&p| {
                    if let Node::Const(c) = b.node(p) {
                        !c.is_zero()
                    } else {
                        true
                    }
                })
                .collect();
            match parts.len() {
                0 => b.constant(q_zero()),
                1 => parts[0],
                _ => b.add(parts),
            }
        }
        Node::Mul(cs) => {
            let parts: Vec<NodeId> = cs
                .iter()
                .map(|c| simplify_dfs(b, source, *c, memo))
                .collect();
            if parts.iter().any(|&p| {
                if let Node::Const(c) = b.node(p) {
                    c.is_zero()
                } else {
                    false
                }
            }) {
                b.constant(q_zero())
            } else {
                let parts: Vec<NodeId> = parts
                    .into_iter()
                    .filter(|&p| {
                        if let Node::Const(c) = b.node(p) {
                            c != &q_one()
                        } else {
                            true
                        }
                    })
                    .collect();
                match parts.len() {
                    0 => b.constant(q_one()),
                    1 => parts[0],
                    _ => b.mul(parts),
                }
            }
        }
        Node::Div(n, d) => {
            let nn = simplify_dfs(b, source, *n, memo);
            let dd = simplify_dfs(b, source, *d, memo);
            b.div(nn, dd)
        }
        Node::Pow(base, exp) => {
            let bb = simplify_dfs(b, source, *base, memo);
            let ee = simplify_dfs(b, source, *exp, memo);
            b.pow(bb, ee)
        }
    };
    memo.insert(id, mapped);
    mapped
}
