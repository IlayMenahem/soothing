//! Local algebraic simplifications on DAG nodes (bottom-up, memoized).
//!
//! Matches the appendix: [`Simplify`] with cache plus [`construct_*`] smart constructors
//! in preference order (constant fold, flatten, identities, sign/fractions, affine/constants).

use crate::dag::{Dag, DagBuilder, Node, NodeId, VarId};
use crate::field::{q_one, q_pow_integer, q_zero, Q};
use num_bigint::BigInt;
use num_traits::{One, Zero};
use std::collections::BTreeMap;
use std::collections::HashMap;
use thiserror::Error;

#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum SimplifyError {
    #[error("division by zero in constant fold")]
    DivisionByZero,
    #[error("non-integer exponent in constant power")]
    NonIntegerExponent,
    #[error("undefined power (e.g. 0 to a negative integer exponent)")]
    UndefinedPower,
}

/// One bottom-up pass: copy `source` into `builder`, simplifying each shared subexpression once.
///
/// # Errors
///
/// Returns [`SimplifyError`] when constant folding hits division by zero, non-integer powers where
/// required, or otherwise undefined exponentiation.
pub fn simplify_once(
    builder: &mut DagBuilder,
    source: &Dag,
    root: NodeId,
) -> Result<NodeId, SimplifyError> {
    let mut memo = HashMap::new();
    simplify_dfs(builder, source, root, &mut memo)
}

fn try_const(b: &DagBuilder, id: NodeId) -> Option<Q> {
    match b.node(id) {
        Node::Const(c) => Some(c.clone()),
        _ => None,
    }
}

fn flatten_add(b: &DagBuilder, ids: &[NodeId], out: &mut Vec<NodeId>) {
    ids.iter().copied().for_each(|id| match b.node(id) {
        Node::Add(cs) => flatten_add(b, cs, out),
        _ => out.push(id),
    });
}

fn flatten_mul(b: &DagBuilder, ids: &[NodeId], out: &mut Vec<NodeId>) {
    ids.iter().copied().for_each(|id| match b.node(id) {
        Node::Mul(cs) => flatten_mul(b, cs, out),
        _ => out.push(id),
    });
}

fn merge_coeff_maps(into: &mut BTreeMap<VarId, Q>, add: &BTreeMap<VarId, Q>) {
    for (&v, c) in add {
        let e = into.entry(v).or_insert_with(q_zero);
        *e += c.clone();
        if e.is_zero() {
            into.remove(&v);
        }
    }
}

fn try_affine_linear(b: &DagBuilder, id: NodeId) -> Option<(BTreeMap<VarId, Q>, Q)> {
    match b.node(id) {
        Node::Const(c) => Some((BTreeMap::new(), c.clone())),
        Node::Var(v) => {
            let mut m = BTreeMap::new();
            m.insert(*v, q_one());
            Some((m, q_zero()))
        }
        Node::Neg(x) => {
            let (mut m, c) = try_affine_linear(b, *x)?;
            for coef in m.values_mut() {
                *coef = -coef.clone();
            }
            m.retain(|_, k| !k.is_zero());
            Some((m, -c))
        }
        Node::Add(parts) => parts.iter().copied().try_fold(
            (BTreeMap::new(), q_zero()),
            |(mut acc, mut const_sum), p| {
                let (m, c) = try_affine_linear(b, p)?;
                merge_coeff_maps(&mut acc, &m);
                const_sum += c;
                Some((acc, const_sum))
            },
        ),
        Node::Mul(cs) if cs.len() == 2 => match (b.node(cs[0]), b.node(cs[1])) {
            (Node::Const(k), Node::Var(v)) | (Node::Var(v), Node::Const(k)) => {
                let mut m = BTreeMap::new();
                m.insert(*v, k.clone());
                Some((m, q_zero()))
            }
            (Node::Const(a), Node::Const(b)) => Some((BTreeMap::new(), a * b)),
            _ => None,
        },
        _ => None,
    }
}

fn try_affine_sum_parts(b: &DagBuilder, parts: &[NodeId]) -> Option<(BTreeMap<VarId, Q>, Q)> {
    parts.iter().copied().try_fold(
        (BTreeMap::new(), q_zero()),
        |(mut acc, mut const_sum), p| {
            let (m, c) = try_affine_linear(b, p)?;
            merge_coeff_maps(&mut acc, &m);
            const_sum += c;
            Some((acc, const_sum))
        },
    )
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
        b.mul2(k, vx)
    }
}

fn affine_to_node(b: &mut DagBuilder, coeffs: BTreeMap<VarId, Q>, constant: Q) -> NodeId {
    let mut parts = Vec::new();
    if !constant.is_zero() {
        parts.push(b.constant(constant));
    }
    parts.extend(
        coeffs
            .into_iter()
            .filter(|(_, coef)| !coef.is_zero())
            .map(|(v, coef)| linear_term(b, v, coef)),
    );
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
    let scaled: BTreeMap<_, _> = coeffs
        .into_iter()
        .filter_map(|(v, c)| {
            let nv = c * inv.clone();
            (!nv.is_zero()).then_some((v, nv))
        })
        .collect();
    let c = const_term * inv;
    affine_to_node(b, scaled, c)
}

fn net_terms_after_neg_cancel(b: &DagBuilder, terms: &[NodeId]) -> HashMap<NodeId, i64> {
    terms
        .iter()
        .copied()
        .fold(HashMap::<NodeId, i64>::new(), |mut m, t| {
            if let Node::Neg(u) = b.node(t) {
                *m.entry(*u).or_insert(0) -= 1;
            } else {
                *m.entry(t).or_insert(0) += 1;
            }
            m
        })
}

fn rebuild_add_terms(b: &mut DagBuilder, net: HashMap<NodeId, i64>) -> Vec<NodeId> {
    net.into_iter()
        .filter_map(|(u, c)| {
            if c == 0 {
                return None;
            }
            Some(if c == 1 {
                u
            } else if c == -1 {
                b.neg(u)
            } else {
                let k = b.constant(Q::from_integer(c.into()));
                b.mul2(k, u)
            })
        })
        .collect()
}

fn merge_const_summands(b: &mut DagBuilder, terms: Vec<NodeId>) -> Vec<NodeId> {
    let (csum, mut out) =
        terms
            .into_iter()
            .fold((q_zero(), Vec::new()), |(mut csum, mut out), t| {
                if let Node::Const(c) = b.node(t) {
                    csum += c.clone();
                } else {
                    out.push(t);
                }
                (csum, out)
            });
    if !csum.is_zero() {
        out.push(b.constant(csum));
    }
    out
}

/// Peel unary negation and negative rational constants; returns (core, parity flip).
fn peel_sign_factor(b: &mut DagBuilder, mut id: NodeId) -> (NodeId, bool) {
    let mut flip = false;
    loop {
        match b.node(id) {
            Node::Neg(x) => {
                flip ^= true;
                id = *x;
            }
            Node::Const(c) => {
                if c.is_zero() || *c == q_one() {
                    return (id, flip);
                }
                if *c == -q_one() {
                    flip ^= true;
                    return (b.constant(q_one()), flip);
                }
                if *c < q_zero() {
                    flip ^= true;
                    return (b.constant(-c.clone()), flip);
                }
                return (id, flip);
            }
            _ => return (id, flip),
        }
    }
}

fn fold_const_pow(base: Q, exp: &Q) -> Result<Q, SimplifyError> {
    if !exp.is_integer() {
        return Err(SimplifyError::NonIntegerExponent);
    }
    q_pow_integer(base, exp.numer()).ok_or(SimplifyError::UndefinedPower)
}

fn construct_add(b: &mut DagBuilder, parts: &[NodeId]) -> NodeId {
    let mut flat = Vec::new();
    flatten_add(b, parts, &mut flat);

    if flat.iter().all(|&id| matches!(b.node(id), Node::Const(_))) {
        let s: Q = flat
            .iter()
            .map(|&id| match b.node(id) {
                Node::Const(c) => c.clone(),
                _ => unreachable!(),
            })
            .sum();
        return b.constant(s);
    }

    flat.retain(|&id| !matches!(b.node(id), Node::Const(c) if c.is_zero()));

    let net = net_terms_after_neg_cancel(b, &flat);
    let mut rebuilt = rebuild_add_terms(b, net);
    rebuilt = merge_const_summands(b, rebuilt);

    let out = match rebuilt.len() {
        0 => return b.constant(q_zero()),
        1 => return rebuilt[0],
        _ => rebuilt,
    };

    if let Some((coeffs, ct)) = try_affine_sum_parts(b, &out) {
        return affine_to_node(b, coeffs, ct);
    }

    b.add(out)
}

fn construct_mul(b: &mut DagBuilder, parts: &[NodeId]) -> Result<NodeId, SimplifyError> {
    let mut flat = Vec::new();
    flatten_mul(b, parts, &mut flat);

    if flat.iter().all(|&id| matches!(b.node(id), Node::Const(_))) {
        let p = flat
            .iter()
            .map(|&id| match b.node(id) {
                Node::Const(c) => c.clone(),
                _ => unreachable!(),
            })
            .product();
        return Ok(b.constant(p));
    }

    if flat
        .iter()
        .any(|&id| matches!(b.node(id), Node::Const(c) if c.is_zero()))
    {
        return Ok(b.constant(q_zero()));
    }

    flat.retain(|&id| !matches!(b.node(id), Node::Const(c) if *c == q_one()));

    let (mut nums, dens, plain): (Vec<_>, Vec<_>, Vec<_>) = flat.iter().copied().fold(
        (Vec::new(), Vec::new(), Vec::new()),
        |(mut nums, mut dens, mut plain), f| {
            match b.node(f) {
                Node::Div(n, d) => {
                    nums.push(*n);
                    dens.push(*d);
                }
                _ => plain.push(f),
            }
            (nums, dens, plain)
        },
    );

    if !dens.is_empty() {
        nums.extend(plain);
        let n = construct_mul(b, &nums)?;
        let d = construct_mul(b, &dens)?;
        return construct_div(b, n, d);
    }

    let (flip, mut cores) =
        flat.into_iter()
            .fold((false, Vec::new()), |(mut flip, mut cores), f| {
                let (core, flp) = peel_sign_factor(b, f);
                flip ^= flp;
                if !matches!(b.node(core), Node::Const(c) if *c == q_one()) {
                    cores.push(core);
                }
                (flip, cores)
            });

    cores.retain(|&id| !matches!(b.node(id), Node::Const(c) if *c == q_one()));

    let mut r = match cores.len() {
        0 => b.constant(q_one()),
        1 => cores[0],
        _ => b.mul(cores),
    };
    if flip {
        r = b.neg(r);
    }
    Ok(r)
}

fn construct_neg(b: &mut DagBuilder, x: NodeId) -> NodeId {
    match b.node(x) {
        Node::Const(c) => b.constant(-c.clone()),
        Node::Neg(y) => *y,
        _ => {
            if let Some((m, c)) = try_affine_linear(b, x) {
                let nm: BTreeMap<_, _> = m
                    .into_iter()
                    .filter_map(|(k, v)| {
                        let nv = -v;
                        (!nv.is_zero()).then_some((k, nv))
                    })
                    .collect();
                return affine_to_node(b, nm, -c);
            }
            b.neg(x)
        }
    }
}

fn apply_div_sign(b: &mut DagBuilder, id: NodeId, flip: bool) -> NodeId {
    if flip {
        b.neg(id)
    } else {
        id
    }
}

fn construct_div(b: &mut DagBuilder, nn: NodeId, dd: NodeId) -> Result<NodeId, SimplifyError> {
    if let (Some(nc), Some(dc)) = (try_const(b, nn), try_const(b, dd)) {
        if dc.is_zero() {
            return Err(SimplifyError::DivisionByZero);
        }
        return Ok(b.constant(nc / dc));
    }

    if let Some(nc) = try_const(b, nn) {
        if nc.is_zero() {
            // 0 / f with non-constant f: rational simplification to 0 (denominator not identically 0 as const).
            return Ok(b.constant(q_zero()));
        }
    }

    if let Some(dc) = try_const(b, dd) {
        if dc == q_one() {
            return Ok(nn);
        }
    }

    let (nn2, sn) = peel_sign_factor(b, nn);
    let (dd2, sd) = peel_sign_factor(b, dd);
    let sign_flip = sn ^ sd;
    let nn = nn2;
    let dd = dd2;

    let nested_fracs = match (b.node(nn), b.node(dd)) {
        (Node::Div(a, bn), Node::Div(c, dn)) => Some((*a, *bn, *c, *dn)),
        _ => None,
    };
    if let Some((a, bn, c, dn)) = nested_fracs {
        let num = construct_mul(b, &[a, dn])?;
        let den = construct_mul(b, &[bn, c])?;
        let r = construct_div(b, num, den)?;
        return Ok(apply_div_sign(b, r, sign_flip));
    }

    if let (Some((an, cn)), Some((ad, cd))) = (try_affine_linear(b, nn), try_affine_linear(b, dd)) {
        if ad.is_empty() && !cd.is_zero() {
            let r = affine_div_by_const(b, an, cn, cd);
            return Ok(apply_div_sign(b, r, sign_flip));
        }
    }

    let r = b.div(nn, dd);
    Ok(apply_div_sign(b, r, sign_flip))
}

fn construct_pow(b: &mut DagBuilder, base: NodeId, exp: NodeId) -> Result<NodeId, SimplifyError> {
    if let (Some(bc), Some(ec)) = (try_const(b, base), try_const(b, exp)) {
        return Ok(b.constant(fold_const_pow(bc, &ec)?));
    }

    if let Some(bc) = try_const(b, base) {
        if bc.is_zero() {
            if let Some(ec) = try_const(b, exp) {
                if ec.is_integer() {
                    let n = ec.numer();
                    if n > &BigInt::from(0) {
                        return Ok(b.constant(q_zero()));
                    }
                }
            }
        }
        if bc == q_one() {
            return Ok(b.constant(q_one()));
        }
    }

    if let Some(ec) = try_const(b, exp) {
        if ec == q_one() {
            return Ok(base);
        }
        if ec.is_zero() {
            if let Some(bc) = try_const(b, base) {
                if bc.is_zero() {
                    return Ok(b.constant(fold_const_pow(bc, &ec)?));
                }
                return Ok(b.constant(q_one()));
            }
        }
    }

    Ok(b.pow(base, exp))
}

fn simplify_dfs(
    b: &mut DagBuilder,
    source: &Dag,
    id: NodeId,
    memo: &mut HashMap<NodeId, NodeId>,
) -> Result<NodeId, SimplifyError> {
    if let Some(&n) = memo.get(&id) {
        return Ok(n);
    }
    let mapped = match source.node(id) {
        Node::Const(c) => b.constant(c.clone()),
        Node::ConstPi => b.const_pi(),
        Node::Var(v) => b.variable(*v),
        Node::Neg(x) => {
            let y = simplify_dfs(b, source, *x, memo)?;
            construct_neg(b, y)
        }
        Node::Add(cs) => {
            let parts: Vec<NodeId> = cs
                .iter()
                .map(|c| simplify_dfs(b, source, *c, memo))
                .collect::<Result<_, _>>()?;
            construct_add(b, &parts)
        }
        Node::Mul(cs) => {
            let parts: Vec<NodeId> = cs
                .iter()
                .map(|c| simplify_dfs(b, source, *c, memo))
                .collect::<Result<_, _>>()?;
            construct_mul(b, &parts)?
        }
        Node::Div(n, d) => {
            let nn = simplify_dfs(b, source, *n, memo)?;
            let dd = simplify_dfs(b, source, *d, memo)?;
            construct_div(b, nn, dd)?
        }
        Node::Pow(base, exp) => {
            let bb = simplify_dfs(b, source, *base, memo)?;
            let ee = simplify_dfs(b, source, *exp, memo)?;
            construct_pow(b, bb, ee)?
        }
    };
    memo.insert(id, mapped);
    Ok(mapped)
}
