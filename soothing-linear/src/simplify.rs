//! Local algebraic simplifications on DAG nodes (bottom-up, memoized).
//!
//! Implements flattening, double-negation removal (including arbitrary-depth unary
//! minus chains via parity), pushing unary minus into a product with a leading
//! rational, constant folding, and partial constant grouping. Bottom-up rebuild
//! propagates constants upward.

use std::collections::HashMap;

use crate::dag::{Dag, DagBuilder, Node, NodeId};
use crate::field::Q;
use num_traits::{One, Pow, Zero};
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

fn flatten_rebuilt_children(
    builder: &DagBuilder,
    op: AssociativeOp,
    children: Vec<NodeId>,
) -> Vec<NodeId> {
    let mut flattened = Vec::with_capacity(children.len());

    for child in children {
        match (op, builder.node(child)) {
            (AssociativeOp::Add, Node::Add(nested)) | (AssociativeOp::Mul, Node::Mul(nested)) => {
                flattened.extend(nested.iter().copied());
            }
            _ => flattened.push(child),
        }
    }

    flattened
}

fn rebuild_children(
    builder: &mut DagBuilder,
    source: &Dag,
    children: &[NodeId],
    rebuilt_nodes: &mut HashMap<NodeId, NodeId>,
) -> Result<Vec<NodeId>, SimplifyError> {
    children
        .iter()
        .map(|child| rebuild_simplified_subdag(builder, source, *child, rebuilt_nodes))
        .collect()
}

fn fold_grouped_add(builder: &mut DagBuilder, flattened: Vec<NodeId>) -> NodeId {
    let mut const_sum = Q::zero();
    let mut non_const = Vec::new();
    for id in flattened {
        match builder.node(id) {
            Node::Const(c) => const_sum += *c,
            _ => non_const.push(id),
        }
    }
    if non_const.is_empty() {
        return builder.constant(const_sum);
    }
    let mut out = Vec::new();
    if !const_sum.is_zero() {
        out.push(builder.constant(const_sum));
    }
    out.extend(non_const);
    builder.add(out)
}

fn fold_grouped_mul(builder: &mut DagBuilder, flattened: Vec<NodeId>) -> NodeId {
    let mut prod = Q::one();
    let mut non_const = Vec::new();
    for id in flattened {
        match builder.node(id) {
            Node::Const(c) => prod *= *c,
            _ => non_const.push(id),
        }
    }
    if prod.is_zero() {
        return builder.constant(Q::zero());
    }
    if non_const.is_empty() {
        return builder.constant(prod);
    }
    let mut out = Vec::new();
    if prod != Q::one() {
        out.push(builder.constant(prod));
    }
    out.extend(non_const);
    builder.mul(out)
}

fn rebuild_add_node(
    builder: &mut DagBuilder,
    source: &Dag,
    children: &[NodeId],
    rebuilt_nodes: &mut HashMap<NodeId, NodeId>,
) -> Result<NodeId, SimplifyError> {
    let rebuilt_children = rebuild_children(builder, source, children, rebuilt_nodes)?;
    let flattened = flatten_rebuilt_children(builder, AssociativeOp::Add, rebuilt_children);
    Ok(fold_grouped_add(builder, flattened))
}

fn rebuild_mul_node(
    builder: &mut DagBuilder,
    source: &Dag,
    children: &[NodeId],
    rebuilt_nodes: &mut HashMap<NodeId, NodeId>,
) -> Result<NodeId, SimplifyError> {
    let rebuilt_children = rebuild_children(builder, source, children, rebuilt_nodes)?;
    let flattened = flatten_rebuilt_children(builder, AssociativeOp::Mul, rebuilt_children);
    Ok(fold_grouped_mul(builder, flattened))
}

/// Peel consecutive [`Node::Neg`] wrappers starting at `id` in `source`.
/// Returns `(base_id, neg_depth)` where `neg_depth` is how many `Neg` nodes were stripped.
fn peel_neg_chain(source: &Dag, mut id: NodeId) -> (NodeId, u32) {
    let mut depth = 0u32;
    while let Node::Neg(inner) = source.node(id) {
        id = *inner;
        depth += 1;
    }
    (id, depth)
}

/// Unary minus on an expression already rebuilt into `builder` (no [`Dag`] arena id — uses
/// [`DagBuilder::node`]).
fn negate_rebuilt_expr(builder: &mut DagBuilder, id: NodeId) -> NodeId {
    match builder.node(id).clone() {
        Node::Const(c) => builder.constant(-c),
        Node::Mul(cs) => {
            if let Some(&first) = cs.first()
                && let Node::Const(c) = builder.node(first)
            {
                let mut ncs = cs;
                ncs[0] = builder.constant(-*c);
                return builder.mul(ncs);
            }
            builder.neg(id)
        }
        Node::Div(n, d) => {
            let nn = negate_rebuilt_expr(builder, n);
            builder.div(nn, d)
        }
        _ => builder.neg(id),
    }
}

fn rebuild_neg_node(
    builder: &mut DagBuilder,
    source: &Dag,
    child: NodeId,
    rebuilt_nodes: &mut HashMap<NodeId, NodeId>,
) -> Result<NodeId, SimplifyError> {
    // `child` is the operand of the unary minus we're rebuilding: `- child`.
    // Peel nested unary minuses in the source so one pass handles `Neg(Neg(Neg(...)))`.
    // Parity: value is `- ((-1)^depth * base) = (-1)^(depth+1) * base`.
    let (base, depth) = peel_neg_chain(source, child);
    let apply_unary_minus = (depth + 1) % 2 == 1;
    let rebuilt_base = rebuild_simplified_subdag(builder, source, base, rebuilt_nodes)?;
    if !apply_unary_minus {
        return Ok(rebuilt_base);
    }
    if let Node::Const(c) = builder.node(rebuilt_base) {
        return Ok(builder.constant(-*c));
    }
    if let Node::Mul(cs) = builder.node(rebuilt_base)
        && let Some(&first) = cs.first()
        && let Node::Const(c) = builder.node(first)
    {
        let mut new_cs = cs.clone();
        new_cs[0] = builder.constant(-*c);
        return Ok(builder.mul(new_cs));
    }
    if let Node::Div(n, d) = builder.node(rebuilt_base).clone() {
        let neg_n = negate_rebuilt_expr(builder, n);
        return Ok(builder.div(neg_n, d));
    }
    Ok(builder.neg(rebuilt_base))
}

/// Flattened product of two rebuilt nodes (constants fold into a single rational when possible).
fn mul_flat(builder: &mut DagBuilder, a: NodeId, b: NodeId) -> NodeId {
    let flattened = flatten_rebuilt_children(builder, AssociativeOp::Mul, vec![a, b]);
    fold_grouped_mul(builder, flattened)
}

/// Appendix: \((x/y)/z \mapsto x/(y\cdot z)\) and \(x/(y/z) \mapsto (x\cdot z)/y\) on rebuilt nodes.
fn normalize_nested_div(
    builder: &mut DagBuilder,
    mut num: NodeId,
    mut den: NodeId,
) -> (NodeId, NodeId) {
    loop {
        let mut changed = false;
        if let Node::Div(a, b) = builder.node(num).clone() {
            num = a;
            den = mul_flat(builder, b, den);
            changed = true;
        }
        if let Node::Div(y, z) = builder.node(den).clone() {
            num = mul_flat(builder, num, z);
            den = y;
            changed = true;
        }
        if !changed {
            break;
        }
    }
    (num, den)
}

fn rebuild_div_node(
    builder: &mut DagBuilder,
    source: &Dag,
    n: NodeId,
    d: NodeId,
    rebuilt_nodes: &mut HashMap<NodeId, NodeId>,
) -> Result<NodeId, SimplifyError> {
    let rebuilt_num = rebuild_simplified_subdag(builder, source, n, rebuilt_nodes)?;
    let rebuilt_den = rebuild_simplified_subdag(builder, source, d, rebuilt_nodes)?;
    let (num, den) = normalize_nested_div(builder, rebuilt_num, rebuilt_den);
    match (builder.node(num), builder.node(den)) {
        (Node::Const(a), Node::Const(b)) => {
            if b.is_zero() {
                Err(SimplifyError::DivisionByZero)
            } else {
                Ok(builder.constant(*a / *b))
            }
        }
        _ => Ok(builder.div(num, den)),
    }
}

fn rebuild_pow_node(
    builder: &mut DagBuilder,
    source: &Dag,
    b: NodeId,
    e: NodeId,
    rebuilt_nodes: &mut HashMap<NodeId, NodeId>,
) -> Result<NodeId, SimplifyError> {
    let rebuilt_base = rebuild_simplified_subdag(builder, source, b, rebuilt_nodes)?;
    let rebuilt_exp = rebuild_simplified_subdag(builder, source, e, rebuilt_nodes)?;
    match (builder.node(rebuilt_base), builder.node(rebuilt_exp)) {
        (Node::Const(base), Node::Const(exp)) => {
            if !exp.is_integer() {
                return Err(SimplifyError::NonIntegerExponent);
            }
            let k = *exp.numer();
            if k < 0 && base.is_zero() {
                return Err(SimplifyError::DivisionByZero);
            }
            let exp_i32 = i32::try_from(k).map_err(|_| SimplifyError::DivisionByZero)?;
            Ok(builder.constant(Pow::pow(*base, exp_i32)))
        }
        _ => Ok(builder.pow(rebuilt_base, rebuilt_exp)),
    }
}

fn rebuild_simplified_subdag(
    builder: &mut DagBuilder,
    source: &Dag,
    node: NodeId,
    rebuilt_nodes: &mut HashMap<NodeId, NodeId>,
) -> Result<NodeId, SimplifyError> {
    if let Some(&rebuilt) = rebuilt_nodes.get(&node) {
        return Ok(rebuilt);
    }

    let rebuilt = match source.node(node) {
        Node::Const(c) => builder.constant(*c),
        Node::ConstPi => builder.const_pi(),
        Node::Var(v) => builder.variable(*v),
        Node::Add(children) => rebuild_add_node(builder, source, children, rebuilt_nodes)?,
        Node::Mul(children) => rebuild_mul_node(builder, source, children, rebuilt_nodes)?,
        Node::Neg(child) => rebuild_neg_node(builder, source, *child, rebuilt_nodes)?,
        Node::Div(n, d) => rebuild_div_node(builder, source, *n, *d, rebuilt_nodes)?,
        Node::Pow(b, e) => rebuild_pow_node(builder, source, *b, *e, rebuilt_nodes)?,
    };

    rebuilt_nodes.insert(node, rebuilt);
    Ok(rebuilt)
}

/// Simplify the DAG expression by applying all the possible simplifications.
///
/// Always runs the bottom-up rebuild so negation and constant grouping apply even when a static
/// pre-scan would not list any pattern.
pub fn simplify(
    builder: &mut DagBuilder,
    source: &Dag,
    root: NodeId,
) -> Result<NodeId, SimplifyError> {
    let mut rebuilt_nodes = HashMap::new();
    rebuild_simplified_subdag(builder, source, root, &mut rebuilt_nodes)
}
