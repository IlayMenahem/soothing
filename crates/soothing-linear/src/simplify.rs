//! Local algebraic simplifications on DAG nodes (single pass).

use crate::dag::{Dag, DagBuilder, Node, NodeId, VarId};
use crate::field::{q_one, q_zero, Q};
use num_traits::{One, Zero};
use std::collections::BTreeMap;
use std::collections::HashMap;

/// Apply one pass of local rewrites: `-(-x)`, strip `+0`, `1·x`, `0·x`, affine `Add` folding,
/// and `Div` of an affine form by a nonzero rational constant.
pub fn simplify_once(builder: &mut DagBuilder, source: &Dag, root: NodeId) -> NodeId {
    let mut memo = HashMap::new();
    simplify_dfs(builder, source, root, &mut memo)
}

fn merge_coeff_maps(into: &mut BTreeMap<VarId, Q>, add: &BTreeMap<VarId, Q>) {
    for (&v, &c) in add {
        let e = into.entry(v).or_insert_with(q_zero);
        *e += c;
        if e.is_zero() {
            into.remove(&v);
        }
    }
}

fn try_affine_linear(b: &DagBuilder, id: NodeId) -> Option<(BTreeMap<VarId, Q>, Q)> {
    match b.node(id) {
        Node::Const(c) => Some((BTreeMap::new(), *c)),
        Node::Var(v) => {
            let mut m = BTreeMap::new();
            m.insert(*v, q_one());
            Some((m, q_zero()))
        }
        Node::Neg(x) => {
            let (mut m, c) = try_affine_linear(b, *x)?;
            for coef in m.values_mut() {
                *coef = -*coef;
            }
            m.retain(|_, k| !k.is_zero());
            Some((m, -c))
        }
        Node::Add(parts) => {
            let mut acc = BTreeMap::new();
            let mut const_sum = q_zero();
            for &p in parts {
                let (m, c) = try_affine_linear(b, p)?;
                merge_coeff_maps(&mut acc, &m);
                const_sum += c;
            }
            Some((acc, const_sum))
        }
        Node::Mul(cs) if cs.len() == 2 => match (b.node(cs[0]), b.node(cs[1])) {
            (Node::Const(k), Node::Var(v)) | (Node::Var(v), Node::Const(k)) => {
                let mut m = BTreeMap::new();
                m.insert(*v, *k);
                Some((m, q_zero()))
            }
            (Node::Const(a), Node::Const(b)) => Some((BTreeMap::new(), a * b)),
            _ => None,
        },
        _ => None,
    }
}

fn try_affine_sum_parts(b: &DagBuilder, parts: &[NodeId]) -> Option<(BTreeMap<VarId, Q>, Q)> {
    let mut acc = BTreeMap::new();
    let mut const_sum = q_zero();
    for &p in parts {
        let (m, c) = try_affine_linear(b, p)?;
        merge_coeff_maps(&mut acc, &m);
        const_sum += c;
    }
    Some((acc, const_sum))
}

fn linear_term(b: &mut DagBuilder, v: VarId, coef: Q) -> NodeId {
    if coef == q_one() {
        b.variable(v)
    } else if coef == -Q::one() {
        let vx = b.variable(v);
        b.neg(vx)
    } else {
        let k = b.constant(coef);
        let vx = b.variable(v);
        b.mul(vec![k, vx])
    }
}

fn affine_to_node(b: &mut DagBuilder, coeffs: BTreeMap<VarId, Q>, constant: Q) -> NodeId {
    let mut parts = Vec::new();
    if !constant.is_zero() {
        parts.push(b.constant(constant));
    }
    for (v, coef) in coeffs {
        if coef.is_zero() {
            continue;
        }
        parts.push(linear_term(b, v, coef));
    }
    match parts.len() {
        0 => b.constant(q_zero()),
        1 => parts[0],
        _ => b.add(parts),
    }
}

fn affine_div_by_const(
    b: &mut DagBuilder,
    coeffs: BTreeMap<VarId, Q>,
    const_term: Q,
    divisor: Q,
) -> NodeId {
    let inv = Q::one() / divisor;
    let mut scaled = BTreeMap::new();
    for (v, c) in coeffs {
        let nv = c * inv;
        if !nv.is_zero() {
            scaled.insert(v, nv);
        }
    }
    let c = const_term * inv;
    affine_to_node(b, scaled, c)
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
            } else if let Some((m, c)) = try_affine_linear(b, y) {
                let mut nm = BTreeMap::new();
                for (k, v) in m {
                    let nv = -v;
                    if !nv.is_zero() {
                        nm.insert(k, nv);
                    }
                }
                affine_to_node(b, nm, -c)
            } else {
                b.neg(y)
            }
        }
        Node::Add(cs) => {
            let parts: Vec<NodeId> = cs
                .iter()
                .map(|c| simplify_dfs(b, source, *c, memo))
                .collect();
            if let Some((coeffs, const_term)) = try_affine_sum_parts(b, &parts) {
                affine_to_node(b, coeffs, const_term)
            } else {
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
            if let (Some((an, cn)), Some((ad, cd))) =
                (try_affine_linear(b, nn), try_affine_linear(b, dd))
            {
                if ad.is_empty() && !cd.is_zero() {
                    affine_div_by_const(b, an, cn, cd)
                } else {
                    b.div(nn, dd)
                }
            } else {
                b.div(nn, dd)
            }
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
