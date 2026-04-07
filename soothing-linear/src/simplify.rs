//! Local algebraic simplifications on DAG nodes (bottom-up, memoized).
//!
//! This file currently implements the appendix's `Flatten` and `NegOfNeg` rules.

use std::collections::{HashMap, HashSet};

use crate::dag::{Dag, DagBuilder, Node, NodeId};
use thiserror::Error;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Error)]
pub enum SimplifyError {
    #[error("division by zero")]
    DivisionByZero,
    #[error("non-integer exponent")]
    NonIntegerExponent,
}

/// A simplification that can be applied to a node.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
enum Simplification {
    FlattenAssociativeChildren(NodeId),
    RemoveDoubleNegation(NodeId),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AssociativeOp {
    Add,
    Mul,
}

fn child_nodes(source: &Dag, node: NodeId) -> Vec<NodeId> {
    match source.node(node) {
        Node::Add(cs) | Node::Mul(cs) => cs.clone(),
        Node::Neg(x) => vec![*x],
        Node::Div(n, d) => vec![*n, *d],
        Node::Pow(b, e) => vec![*b, *e],
        Node::Const(_) | Node::ConstPi | Node::Var(_) => vec![],
    }
}

fn flattenable_op(source: &Dag, node: NodeId) -> Option<AssociativeOp> {
    match source.node(node) {
        Node::Add(cs)
            if cs
                .iter()
                .any(|child| matches!(source.node(*child), Node::Add(_))) =>
        {
            Some(AssociativeOp::Add)
        }
        Node::Mul(cs)
            if cs
                .iter()
                .any(|child| matches!(source.node(*child), Node::Mul(_))) =>
        {
            Some(AssociativeOp::Mul)
        }
        _ => None,
    }
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

fn is_double_negation(source: &Dag, node: NodeId) -> bool {
    match source.node(node) {
        Node::Neg(child) => matches!(source.node(*child), Node::Neg(_)),
        _ => false,
    }
}

fn rebuild_children(
    builder: &mut DagBuilder,
    source: &Dag,
    children: &[NodeId],
    rebuilt_nodes: &mut HashMap<NodeId, NodeId>,
) -> Vec<NodeId> {
    children
        .iter()
        .map(|child| rebuild_simplified_subdag(builder, source, *child, rebuilt_nodes))
        .collect()
}

fn rebuild_add_node(
    builder: &mut DagBuilder,
    source: &Dag,
    children: &[NodeId],
    rebuilt_nodes: &mut HashMap<NodeId, NodeId>,
) -> NodeId {
    let rebuilt_children = rebuild_children(builder, source, children, rebuilt_nodes);
    builder.add(flatten_rebuilt_children(
        builder,
        AssociativeOp::Add,
        rebuilt_children,
    ))
}

fn rebuild_mul_node(
    builder: &mut DagBuilder,
    source: &Dag,
    children: &[NodeId],
    rebuilt_nodes: &mut HashMap<NodeId, NodeId>,
) -> NodeId {
    let rebuilt_children = rebuild_children(builder, source, children, rebuilt_nodes);
    builder.mul(flatten_rebuilt_children(
        builder,
        AssociativeOp::Mul,
        rebuilt_children,
    ))
}

fn rebuild_neg_node(
    builder: &mut DagBuilder,
    source: &Dag,
    child: NodeId,
    rebuilt_nodes: &mut HashMap<NodeId, NodeId>,
) -> NodeId {
    if let Node::Neg(inner) = source.node(child) {
        rebuild_simplified_subdag(builder, source, *inner, rebuilt_nodes)
    } else {
        let rebuilt_child = rebuild_simplified_subdag(builder, source, child, rebuilt_nodes);
        builder.neg(rebuilt_child)
    }
}

fn rebuild_simplified_subdag(
    builder: &mut DagBuilder,
    source: &Dag,
    node: NodeId,
    rebuilt_nodes: &mut HashMap<NodeId, NodeId>,
) -> NodeId {
    if let Some(&rebuilt) = rebuilt_nodes.get(&node) {
        return rebuilt;
    }

    let rebuilt = match source.node(node) {
        Node::Const(c) => builder.constant(c.clone()),
        Node::ConstPi => builder.const_pi(),
        Node::Var(v) => builder.variable(*v),
        Node::Add(children) => rebuild_add_node(builder, source, children, rebuilt_nodes),
        Node::Mul(children) => rebuild_mul_node(builder, source, children, rebuilt_nodes),
        Node::Neg(child) => rebuild_neg_node(builder, source, *child, rebuilt_nodes),
        Node::Div(n, d) => {
            let rebuilt_num = rebuild_simplified_subdag(builder, source, *n, rebuilt_nodes);
            let rebuilt_den = rebuild_simplified_subdag(builder, source, *d, rebuilt_nodes);
            builder.div(rebuilt_num, rebuilt_den)
        }
        Node::Pow(b, e) => {
            let rebuilt_base = rebuild_simplified_subdag(builder, source, *b, rebuilt_nodes);
            let rebuilt_exp = rebuild_simplified_subdag(builder, source, *e, rebuilt_nodes);
            builder.pow(rebuilt_base, rebuilt_exp)
        }
    };

    rebuilt_nodes.insert(node, rebuilt);
    rebuilt
}

/// Simplify the DAG expression by applying all the possible simplifications.
pub fn simplify(
    builder: &mut DagBuilder,
    source: &Dag,
    root: NodeId,
) -> Result<NodeId, SimplifyError> {
    if collect_reachable_simplifications(source, root).is_empty() {
        return Ok(builder.merge_root(source, root));
    }

    let mut rebuilt_nodes = HashMap::new();
    Ok(rebuild_simplified_subdag(
        builder,
        source,
        root,
        &mut rebuilt_nodes,
    ))
}

/// Get all the possible simplifications that can be applied to a node.
fn detect_local_simplifications(source: &Dag, node: NodeId) -> HashSet<Simplification> {
    let mut simplifications = HashSet::new();

    if flattenable_op(source, node).is_some() {
        simplifications.insert(Simplification::FlattenAssociativeChildren(node));
    }

    if is_double_negation(source, node) {
        simplifications.insert(Simplification::RemoveDoubleNegation(node));
    }

    simplifications
}

fn collect_reachable_simplifications(source: &Dag, root: NodeId) -> HashSet<Simplification> {
    let mut simplifications: HashSet<Simplification> = HashSet::new();
    let mut stack = vec![root];

    while let Some(node) = stack.pop() {
        simplifications.extend(detect_local_simplifications(source, node));
        stack.extend(child_nodes(source, node));
    }

    simplifications
}
