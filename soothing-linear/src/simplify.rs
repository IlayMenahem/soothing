//! Algebraic simplification: extract from the source e-graph, simplify as a [`egg::RecExpr`], then
//! insert into the target e-graph. Optional [`egg::Runner`] pass applies basic rewrites before the
//! structural rebuild.

use crate::expr_lang::{flatten_add_terms, flatten_mul_terms, RationalLang};
use crate::field::{q_pow_integer, Q};
use egg::{rewrite, AstSize, EGraph, Extractor, Id, RecExpr, Rewrite, Runner};
use num_traits::{One, Zero};
use std::cmp::Ordering;
use std::collections::HashMap;
use thiserror::Error;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Error)]
pub enum SimplifyError {
    #[error("division by zero")]
    DivisionByZero,
    #[error("non-integer exponent")]
    NonIntegerExponent,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AssociativeOp {
    Add,
    Mul,
}

// --- egg pre-pass ---

fn simplify_rules() -> Vec<Rewrite<RationalLang, ()>> {
    vec![
        rewrite!("neg-involutive"; "(neg (neg ?x))" => "?x"),
        rewrite!("add-zero-left"; "(+ 0 ?x)" => "?x"),
        rewrite!("add-zero-right"; "(+ ?x 0)" => "?x"),
        rewrite!("mul-one-left"; "(* 1 ?x)" => "?x"),
        rewrite!("mul-one-right"; "(* ?x 1)" => "?x"),
        rewrite!("mul-zero-left"; "(* 0 ?x)" => "0"),
        rewrite!("mul-zero-right"; "(* ?x 0)" => "0"),
    ]
}

fn flatten_rebuilt_children(
    out: &RecExpr<RationalLang>,
    op: AssociativeOp,
    children: Vec<Id>,
) -> Vec<Id> {
    let mut flattened = Vec::with_capacity(children.len());
    for child in children {
        match (op, &out[child]) {
            (AssociativeOp::Add, RationalLang::Add(_)) => {
                let mut buf = Vec::new();
                flatten_add_terms(out, child, &mut buf);
                flattened.extend(buf);
            }
            (AssociativeOp::Mul, RationalLang::Mul(_)) => {
                let mut buf = Vec::new();
                flatten_mul_terms(out, child, &mut buf);
                flattened.extend(buf);
            }
            _ => flattened.push(child),
        }
    }
    flattened
}

// --- structural equality & ordering ---

fn recexpr_eq(out: &RecExpr<RationalLang>, a: Id, b: Id) -> bool {
    match (&out[a], &out[b]) {
        (RationalLang::Const(ca), RationalLang::Const(cb)) => ca == cb,
        (RationalLang::Pi, RationalLang::Pi) => true,
        (RationalLang::Var(va), RationalLang::Var(vb)) => va == vb,
        (RationalLang::Neg(xa), RationalLang::Neg(xb)) => recexpr_eq(out, *xa, *xb),
        (RationalLang::Add([a1, a2]), RationalLang::Add([b1, b2])) => {
            recexpr_eq(out, *a1, *b1) && recexpr_eq(out, *a2, *b2)
        }
        (RationalLang::Mul([a1, a2]), RationalLang::Mul([b1, b2])) => {
            recexpr_eq(out, *a1, *b1) && recexpr_eq(out, *a2, *b2)
        }
        (RationalLang::Div([a1, a2]), RationalLang::Div([b1, b2])) => {
            recexpr_eq(out, *a1, *b1) && recexpr_eq(out, *a2, *b2)
        }
        (RationalLang::Pow([a1, a2]), RationalLang::Pow([b1, b2])) => {
            recexpr_eq(out, *a1, *b1) && recexpr_eq(out, *a2, *b2)
        }
        _ => false,
    }
}

fn node_kind(n: &RationalLang) -> u8 {
    match n {
        RationalLang::Const(_) => 0,
        RationalLang::Pi => 1,
        RationalLang::Var(_) => 2,
        RationalLang::Neg(_) => 3,
        RationalLang::Add(_) => 4,
        RationalLang::Mul(_) => 5,
        RationalLang::Div(_) => 6,
        RationalLang::Pow(_) => 7,
    }
}

fn cmp_recexpr_node(out: &RecExpr<RationalLang>, a: Id, b: Id) -> Ordering {
    let ka = node_kind(&out[a]);
    let kb = node_kind(&out[b]);
    if ka != kb {
        return ka.cmp(&kb);
    }
    match (&out[a], &out[b]) {
        (RationalLang::Const(ca), RationalLang::Const(cb)) => ca.cmp(cb),
        (RationalLang::Pi, RationalLang::Pi) => Ordering::Equal,
        (RationalLang::Var(va), RationalLang::Var(vb)) => va.cmp(vb),
        (RationalLang::Neg(xa), RationalLang::Neg(xb)) => cmp_recexpr_node(out, *xa, *xb),
        (RationalLang::Add([a1, a2]), RationalLang::Add([b1, b2])) => {
            cmp_recexpr_node(out, *a1, *b1).then_with(|| cmp_recexpr_node(out, *a2, *b2))
        }
        (RationalLang::Mul([a1, a2]), RationalLang::Mul([b1, b2])) => {
            cmp_recexpr_node(out, *a1, *b1).then_with(|| cmp_recexpr_node(out, *a2, *b2))
        }
        (RationalLang::Div([a1, a2]), RationalLang::Div([b1, b2])) => {
            cmp_recexpr_node(out, *a1, *b1).then_with(|| cmp_recexpr_node(out, *a2, *b2))
        }
        (RationalLang::Pow([a1, a2]), RationalLang::Pow([b1, b2])) => {
            cmp_recexpr_node(out, *a1, *b1).then_with(|| cmp_recexpr_node(out, *a2, *b2))
        }
        _ => Ordering::Equal,
    }
}

// --- flattened mul: pull out rational coeff, rebuild rest ---

fn unit_one(out: &mut RecExpr<RationalLang>) -> Id {
    out.add(RationalLang::Const(Q::one()))
}

fn fold_mul_chain(out: &mut RecExpr<RationalLang>, parts: Vec<Id>) -> Id {
    match parts.len() {
        0 => out.add(RationalLang::Const(Q::one())),
        1 => parts[0],
        _ => {
            let mut it = parts.into_iter();
            let mut acc = it.next().expect("len >= 2");
            for c in it {
                acc = out.add(RationalLang::Mul([acc, c]));
            }
            acc
        }
    }
}

fn fold_mul_factors_list(out: &mut RecExpr<RationalLang>, factors: Vec<Id>) -> Id {
    match factors.len() {
        0 => unit_one(out),
        1 => factors[0],
        _ => fold_mul_chain(out, factors),
    }
}

fn const_product_and_syms(out: &RecExpr<RationalLang>, factors: &[Id]) -> (Q, Vec<Id>) {
    let mut prod = Q::one();
    let mut syms = Vec::new();
    for &x in factors {
        match &out[x] {
            RationalLang::Const(c) => prod *= c.clone(),
            _ => syms.push(x),
        }
    }
    (prod, syms)
}

fn mul_rest_from_syms(out: &mut RecExpr<RationalLang>, syms: Vec<Id>) -> Id {
    match syms.len() {
        0 => unit_one(out),
        1 => syms[0],
        _ => {
            let mut s = syms;
            s.sort_by(|&x, &y| cmp_recexpr_node(out, x, y));
            fold_mul_chain(out, s)
        }
    }
}

fn canonical_sorted_mul(out: &mut RecExpr<RationalLang>, mut factors: Vec<Id>) -> Id {
    if factors.len() <= 1 {
        return fold_mul_factors_list(out, factors);
    }
    factors.sort_by(|&x, &y| cmp_recexpr_node(out, x, y));
    fold_mul_chain(out, factors)
}

fn flatten_mul_const_syms(out: &RecExpr<RationalLang>, id: Id) -> (Q, Vec<Id>) {
    let mut buf = Vec::new();
    flatten_mul_terms(out, id, &mut buf);
    const_product_and_syms(out, &buf)
}

fn coeff_times_mul_rest(out: &mut RecExpr<RationalLang>, coeff: Q, id: Id) -> (Q, Id) {
    match &out[id] {
        RationalLang::Mul(_) => {
            let (p, syms) = flatten_mul_const_syms(out, id);
            (coeff * p, mul_rest_from_syms(out, syms))
        }
        _ => (coeff, id),
    }
}

fn canonicalize_flat_mul(out: &mut RecExpr<RationalLang>, id: Id) -> Id {
    let mut buf = Vec::new();
    flatten_mul_terms(out, id, &mut buf);
    if buf.is_empty() {
        return unit_one(out);
    }
    if buf.len() == 1 {
        return buf[0];
    }
    canonical_sorted_mul(out, buf)
}

fn split_mul_part(out: &mut RecExpr<RationalLang>, n: Id) -> (Q, Id) {
    coeff_times_mul_rest(out, Q::one(), n)
}

fn peel_neg_coeff(out: &RecExpr<RationalLang>, mut id: Id) -> (Q, Id) {
    let mut c = Q::one();
    while let RationalLang::Neg(inner) = &out[id] {
        c = -c;
        id = *inner;
    }
    (c, id)
}

fn is_unit_rest(out: &RecExpr<RationalLang>, id: Id) -> bool {
    matches!(&out[id], RationalLang::Const(c) if *c == Q::one())
}

fn split_coeff_rest(out: &mut RecExpr<RationalLang>, id: Id) -> (Q, Id) {
    let (neg_c, id) = peel_neg_coeff(out, id);
    match out[id].clone() {
        RationalLang::Const(c) => (neg_c * c, unit_one(out)),
        RationalLang::Pi | RationalLang::Var(_) => (neg_c, id),
        RationalLang::Mul(_) => coeff_times_mul_rest(out, neg_c, id),
        RationalLang::Div([n, d]) => {
            let (cn, n_rest) = split_mul_part(out, n);
            let d_can = canonicalize_flat_mul(out, d);
            let rest = out.add(RationalLang::Div([n_rest, d_can]));
            (neg_c * cn, rest)
        }
        RationalLang::Pow([_, _]) | RationalLang::Add([_, _]) | RationalLang::Neg(_) => (neg_c, id),
    }
}

fn term_from_coeff_rest(out: &mut RecExpr<RationalLang>, c: Q, r: Id) -> Id {
    if c.is_zero() {
        return out.add(RationalLang::Const(Q::zero()));
    }
    if c == Q::one() {
        return r;
    }
    if c == -Q::one() {
        return out.add(RationalLang::Neg(r));
    }
    let c_id = out.add(RationalLang::Const(c));
    fold_mul_chain(out, vec![c_id, r])
}

fn fold_grouped_add(out: &mut RecExpr<RationalLang>, flattened: Vec<Id>) -> Id {
    let mut const_sum = Q::zero();
    let mut buckets: Vec<(Q, Id)> = Vec::new();
    for id in flattened {
        match &out[id] {
            RationalLang::Const(c) => const_sum += c.clone(),
            _ => {
                let (c, r) = split_coeff_rest(out, id);
                if c.is_zero() {
                    continue;
                }
                if is_unit_rest(out, r) {
                    const_sum += c;
                    continue;
                }
                if let Some(i) = buckets.iter().position(|(_, br)| recexpr_eq(out, r, *br)) {
                    buckets[i].0 += c;
                } else {
                    buckets.push((c, r));
                }
            }
        }
    }
    buckets.retain(|(c, _)| !c.is_zero());
    let mut parts = Vec::new();
    if !const_sum.is_zero() {
        parts.push(out.add(RationalLang::Const(const_sum)));
    }
    for (c, r) in buckets {
        parts.push(term_from_coeff_rest(out, c, r));
    }
    fold_add_chain(out, parts)
}

fn fold_add_chain(out: &mut RecExpr<RationalLang>, parts: Vec<Id>) -> Id {
    match parts.len() {
        0 => out.add(RationalLang::Const(Q::zero())),
        1 => parts[0],
        _ => {
            let mut it = parts.into_iter();
            let mut acc = it.next().expect("len >= 2");
            for c in it {
                acc = out.add(RationalLang::Add([acc, c]));
            }
            acc
        }
    }
}

fn fold_grouped_mul(out: &mut RecExpr<RationalLang>, flattened: Vec<Id>) -> Id {
    let (prod, syms) = const_product_and_syms(out, &flattened);
    if prod.is_zero() {
        return out.add(RationalLang::Const(Q::zero()));
    }
    if syms.is_empty() {
        return out.add(RationalLang::Const(prod));
    }
    let mut parts = Vec::new();
    if prod != Q::one() {
        parts.push(out.add(RationalLang::Const(prod)));
    }
    parts.extend(syms);
    fold_mul_chain(out, parts)
}

fn peel_neg_chain(recexpr: &RecExpr<RationalLang>, mut id: Id) -> (Id, u32) {
    let mut depth = 0u32;
    while let RationalLang::Neg(inner) = &recexpr[id] {
        id = *inner;
        depth += 1;
    }
    (id, depth)
}

fn negate_rebuilt_expr(out: &mut RecExpr<RationalLang>, id: Id) -> Id {
    match out[id].clone() {
        RationalLang::Const(c) => out.add(RationalLang::Const(-c)),
        RationalLang::Mul(_) => {
            let mut buf = Vec::new();
            flatten_mul_terms(out, id, &mut buf);
            if let Some(&first) = buf.first() {
                if let RationalLang::Const(c) = &out[first] {
                    let neg_c = out.add(RationalLang::Const(-c.clone()));
                    let mut rest = buf;
                    rest[0] = neg_c;
                    return fold_mul_chain(out, rest);
                }
            }
            out.add(RationalLang::Neg(id))
        }
        RationalLang::Div([n, d]) => {
            let nn = negate_rebuilt_expr(out, n);
            out.add(RationalLang::Div([nn, d]))
        }
        _ => out.add(RationalLang::Neg(id)),
    }
}

fn mul_flat(out: &mut RecExpr<RationalLang>, a: Id, b: Id) -> Id {
    let flattened = flatten_rebuilt_children(out, AssociativeOp::Mul, vec![a, b]);
    fold_grouped_mul(out, flattened)
}

fn normalize_nested_div(out: &mut RecExpr<RationalLang>, mut num: Id, mut den: Id) -> (Id, Id) {
    loop {
        let mut changed = false;
        if let RationalLang::Div([a, b]) = out[num].clone() {
            num = a;
            den = mul_flat(out, b, den);
            changed = true;
        }
        if let RationalLang::Div([y, z]) = out[den].clone() {
            num = mul_flat(out, num, z);
            den = y;
            changed = true;
        }
        if !changed {
            break;
        }
    }
    (num, den)
}

/// `a1 * ... * Div(n,d) * ...` with exactly one [`Div`] factor becomes `Div(a1 * ... * n, d)`.
fn try_mul_absorb_div(out: &mut RecExpr<RationalLang>, root: Id) -> Id {
    let mut buf = Vec::new();
    flatten_mul_terms(out, root, &mut buf);
    let div_positions: Vec<usize> = buf
        .iter()
        .enumerate()
        .filter(|(_, id)| matches!(out[**id], RationalLang::Div(_)))
        .map(|(i, _)| i)
        .collect();
    if div_positions.len() != 1 {
        return root;
    }
    let div_idx = div_positions[0];
    let RationalLang::Div([n, d]) = out[buf[div_idx]].clone() else {
        return root;
    };
    let mut num_factors = Vec::new();
    for (i, &factor) in buf.iter().enumerate() {
        if i == div_idx {
            continue;
        }
        flatten_mul_terms(out, factor, &mut num_factors);
    }
    flatten_mul_terms(out, n, &mut num_factors);
    let nn = fold_grouped_mul(out, num_factors);
    out.add(RationalLang::Div([nn, d]))
}

fn cancel_common_mul_factors(out: &mut RecExpr<RationalLang>, num: Id, den: Id) -> (Id, Id) {
    let mut ns = Vec::new();
    let mut ds = Vec::new();
    flatten_mul_terms(out, num, &mut ns);
    flatten_mul_terms(out, den, &mut ds);
    let mut i = 0;
    while i < ns.len() {
        let mut removed = false;
        let mut j = 0;
        while j < ds.len() {
            if recexpr_eq(out, ns[i], ds[j]) {
                ns.remove(i);
                ds.remove(j);
                removed = true;
                break;
            }
            j += 1;
        }
        if !removed {
            i += 1;
        }
    }
    (
        fold_mul_factors_list(out, ns),
        fold_mul_factors_list(out, ds),
    )
}

fn normalize_div_for_cancel(out: &mut RecExpr<RationalLang>, num: Id, den: Id) -> (Id, Id) {
    let (mut num, mut den) = normalize_nested_div(out, num, den);
    num = try_mul_absorb_div(out, num);
    den = try_mul_absorb_div(out, den);
    normalize_nested_div(out, num, den)
}

fn rebuild_children(
    source: &RecExpr<RationalLang>,
    out: &mut RecExpr<RationalLang>,
    children: &[Id],
    rebuilt: &mut HashMap<Id, Id>,
) -> Result<Vec<Id>, SimplifyError> {
    children
        .iter()
        .map(|c| rebuild_simplified_subexpr(source, out, *c, rebuilt))
        .collect()
}

fn rebuild_add_node(
    source: &RecExpr<RationalLang>,
    out: &mut RecExpr<RationalLang>,
    add_root: Id,
    rebuilt: &mut HashMap<Id, Id>,
) -> Result<Id, SimplifyError> {
    let mut buf = Vec::new();
    flatten_add_terms(source, add_root, &mut buf);
    let rebuilt_children = rebuild_children(source, out, &buf, rebuilt)?;
    let flattened = flatten_rebuilt_children(out, AssociativeOp::Add, rebuilt_children);
    Ok(fold_grouped_add(out, flattened))
}

fn rebuild_mul_node(
    source: &RecExpr<RationalLang>,
    out: &mut RecExpr<RationalLang>,
    mul_root: Id,
    rebuilt: &mut HashMap<Id, Id>,
) -> Result<Id, SimplifyError> {
    let mut buf = Vec::new();
    flatten_mul_terms(source, mul_root, &mut buf);
    let rebuilt_children = rebuild_children(source, out, &buf, rebuilt)?;
    let flattened = flatten_rebuilt_children(out, AssociativeOp::Mul, rebuilt_children);
    let folded = fold_grouped_mul(out, flattened);
    Ok(try_mul_absorb_div(out, folded))
}

fn rebuild_neg_node(
    source: &RecExpr<RationalLang>,
    out: &mut RecExpr<RationalLang>,
    child: Id,
    rebuilt: &mut HashMap<Id, Id>,
) -> Result<Id, SimplifyError> {
    let (base, depth) = peel_neg_chain(source, child);
    let apply_unary_minus = (depth + 1) % 2 == 1;
    let rebuilt_base = rebuild_simplified_subexpr(source, out, base, rebuilt)?;
    if !apply_unary_minus {
        return Ok(rebuilt_base);
    }
    if let RationalLang::Const(c) = &out[rebuilt_base] {
        return Ok(out.add(RationalLang::Const(-c.clone())));
    }
    if let RationalLang::Mul(_) = &out[rebuilt_base] {
        let mut buf = Vec::new();
        flatten_mul_terms(out, rebuilt_base, &mut buf);
        if let Some(&first) = buf.first() {
            if let RationalLang::Const(c) = &out[first] {
                let mut rest = buf;
                rest[0] = out.add(RationalLang::Const(-c.clone()));
                return Ok(fold_mul_chain(out, rest));
            }
        }
    }
    if let RationalLang::Div([n, d]) = out[rebuilt_base].clone() {
        let neg_n = negate_rebuilt_expr(out, n);
        return Ok(out.add(RationalLang::Div([neg_n, d])));
    }
    Ok(out.add(RationalLang::Neg(rebuilt_base)))
}

fn rebuild_div_node(
    source: &RecExpr<RationalLang>,
    out: &mut RecExpr<RationalLang>,
    n: Id,
    d: Id,
    rebuilt: &mut HashMap<Id, Id>,
) -> Result<Id, SimplifyError> {
    let rebuilt_num = rebuild_simplified_subexpr(source, out, n, rebuilt)?;
    let rebuilt_den = rebuild_simplified_subexpr(source, out, d, rebuilt)?;
    let (num, den) = normalize_div_for_cancel(out, rebuilt_num, rebuilt_den);
    let (num, den) = cancel_common_mul_factors(out, num, den);
    if let RationalLang::Const(b) = &out[den] {
        if b.is_one() {
            return Ok(num);
        }
    }
    match (&out[num], &out[den]) {
        (RationalLang::Const(a), RationalLang::Const(b)) => {
            if b.is_zero() {
                Err(SimplifyError::DivisionByZero)
            } else {
                Ok(out.add(RationalLang::Const(a.clone() / b.clone())))
            }
        }
        _ => Ok(out.add(RationalLang::Div([num, den]))),
    }
}

fn rebuild_pow_node(
    source: &RecExpr<RationalLang>,
    out: &mut RecExpr<RationalLang>,
    b: Id,
    e: Id,
    rebuilt: &mut HashMap<Id, Id>,
) -> Result<Id, SimplifyError> {
    let rebuilt_base = rebuild_simplified_subexpr(source, out, b, rebuilt)?;
    let rebuilt_exp = rebuild_simplified_subexpr(source, out, e, rebuilt)?;
    if let RationalLang::Const(exp) = &out[rebuilt_exp] {
        if exp.is_one() {
            return Ok(rebuilt_base);
        }
    }
    match (&out[rebuilt_base], &out[rebuilt_exp]) {
        (RationalLang::Const(base), RationalLang::Const(exp)) => {
            if !exp.is_integer() {
                return Err(SimplifyError::NonIntegerExponent);
            }
            q_pow_integer(base.clone(), exp.numer())
                .map(|v| out.add(RationalLang::Const(v)))
                .ok_or(SimplifyError::DivisionByZero)
        }
        _ => Ok(out.add(RationalLang::Pow([rebuilt_base, rebuilt_exp]))),
    }
}

fn rebuild_simplified_subexpr(
    source: &RecExpr<RationalLang>,
    out: &mut RecExpr<RationalLang>,
    node: Id,
    rebuilt: &mut HashMap<Id, Id>,
) -> Result<Id, SimplifyError> {
    if let Some(&r) = rebuilt.get(&node) {
        return Ok(r);
    }

    let new_id = match source[node].clone() {
        RationalLang::Const(c) => out.add(RationalLang::Const(c)),
        RationalLang::Pi => out.add(RationalLang::Pi),
        RationalLang::Var(v) => out.add(RationalLang::Var(v)),
        RationalLang::Add([_, _]) => rebuild_add_node(source, out, node, rebuilt)?,
        RationalLang::Mul([_, _]) => rebuild_mul_node(source, out, node, rebuilt)?,
        RationalLang::Neg(child) => rebuild_neg_node(source, out, child, rebuilt)?,
        RationalLang::Div([n, d]) => rebuild_div_node(source, out, n, d, rebuilt)?,
        RationalLang::Pow([b, e]) => rebuild_pow_node(source, out, b, e, rebuilt)?,
    };
    rebuilt.insert(node, new_id);
    Ok(new_id)
}

fn rebuild_simplified_recexpr(
    source: &RecExpr<RationalLang>,
    root: Id,
) -> Result<RecExpr<RationalLang>, SimplifyError> {
    let mut out = RecExpr::default();
    let mut rebuilt = HashMap::new();
    rebuild_simplified_subexpr(source, &mut out, root, &mut rebuilt)?;
    Ok(out)
}

/// Simplify using a short equality-saturation pass, then the same structural rebuild as the legacy
/// DAG simplifier (on the extracted [`RecExpr`]).
pub fn simplify(
    target: &mut EGraph<RationalLang, ()>,
    source: &EGraph<RationalLang, ()>,
    root: Id,
) -> Result<Id, SimplifyError> {
    let extractor = Extractor::new(source, AstSize);
    let (_, recexpr) = extractor.find_best(root);

    let runner = Runner::default()
        .with_iter_limit(32)
        .with_node_limit(100_000)
        .with_expr(&recexpr)
        .run(&simplify_rules());

    let ext2 = Extractor::new(&runner.egraph, AstSize);
    let (_, recexpr2) = ext2.find_best(runner.roots[0]);

    let simplified = rebuild_simplified_recexpr(&recexpr2, recexpr2.root())?;
    Ok(target.add_expr(&simplified))
}
