//! Substitution map and dependency ordering.

use crate::dag::{Dag, DagBuilder, Node, NodeId, VarId};
use std::collections::{HashMap, HashSet};
use thiserror::Error;

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum SubstitutionError {
    #[error("cyclic dependency between substituted variables")]
    Cycle,
}

/// Maps eliminated variables to defining expressions (roots in the shared arena).
#[derive(Clone, Debug)]
pub struct SubstitutionMap {
    builder: DagBuilder,
    bindings: HashMap<VarId, NodeId>,
    insertion: Vec<VarId>,
}

impl Default for SubstitutionMap {
    fn default() -> Self {
        Self::new()
    }
}

impl SubstitutionMap {
    pub fn new() -> Self {
        Self {
            builder: DagBuilder::new(),
            bindings: HashMap::new(),
            insertion: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.bindings.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bindings.is_empty()
    }

    pub fn get(&self, v: VarId) -> Option<NodeId> {
        self.bindings.get(&v).copied()
    }

    pub fn contains(&self, v: VarId) -> bool {
        self.bindings.contains_key(&v)
    }

    /// Current DAG snapshot (all merged expressions).
    pub fn dag(&self) -> Dag {
        self.builder.snapshot()
    }

    /// Add `v \mapsto` expression copied from `expr_dag` at `expr_root`.
    pub fn insert(&mut self, v: VarId, expr_dag: &Dag, expr_root: NodeId) {
        let root = self.builder.merge_root(expr_dag, expr_root);
        if self.bindings.insert(v, root).is_some() {
            self.insertion.retain(|x| *x != v);
        }
        self.insertion.push(v);
    }

    /// Variables in insertion order.
    pub fn insertion_order(&self) -> &[VarId] {
        &self.insertion
    }

    /// Topological order: if `u` appears inside the expression for `v`, then `u` comes before `v`.
    pub fn replay_order(&self) -> Result<Vec<VarId>, SubstitutionError> {
        let dag = self.builder.snapshot();
        let mut deps: HashMap<VarId, Vec<VarId>> = HashMap::new();
        for (&v, &root) in &self.bindings {
            let ds: Vec<VarId> = vars_in_subgraph(&dag, root, &self.bindings)
                .into_iter()
                .filter(|u| *u != v)
                .collect();
            deps.insert(v, ds);
        }
        let mut order = Vec::new();
        let mut temp = HashSet::new();
        let mut perm = HashSet::new();
        for &v in &self.insertion {
            dfs_deps_first(v, &deps, &mut temp, &mut perm, &mut order)?;
        }
        Ok(order)
    }
}

/// `v` depends on each `u` in `deps[v]`; visit dependencies of `v` before appending `v`.
fn dfs_deps_first(
    v: VarId,
    deps: &HashMap<VarId, Vec<VarId>>,
    temp: &mut HashSet<VarId>,
    perm: &mut HashSet<VarId>,
    order: &mut Vec<VarId>,
) -> Result<(), SubstitutionError> {
    if perm.contains(&v) {
        return Ok(());
    }
    if temp.contains(&v) {
        return Err(SubstitutionError::Cycle);
    }
    temp.insert(v);
    if let Some(us) = deps.get(&v) {
        for &u in us {
            dfs_deps_first(u, deps, temp, perm, order)?;
        }
    }
    temp.remove(&v);
    perm.insert(v);
    order.push(v);
    Ok(())
}

fn vars_in_subgraph(dag: &Dag, root: NodeId, bound: &HashMap<VarId, NodeId>) -> HashSet<VarId> {
    let mut out = HashSet::new();
    let mut stack = vec![root];
    let mut seen = HashSet::new();
    
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }

        match dag.node(id) {
            Node::Const(_) | Node::ConstPi => {}
            Node::Var(x) => {
                if bound.contains_key(x) {
                    out.insert(*x);
                }
            }
            Node::Add(cs) | Node::Mul(cs) => {
                stack.extend(cs);
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
