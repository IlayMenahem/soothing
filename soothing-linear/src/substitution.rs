//! Substitution map and dependency ordering.

use crate::expr_build::ExprBuilder;
use crate::expr_lang::{RationalLang, VarId};
use egg::{AstSize, EGraph, Extractor, Id};
use std::collections::{HashMap, HashSet};
use thiserror::Error;

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum SubstitutionError {
    #[error("cyclic dependency between substituted variables")]
    Cycle,
}

/// Maps eliminated variables to defining expressions (e-class ids in the shared e-graph).
#[derive(Clone, Debug)]
pub struct SubstitutionMap {
    builder: ExprBuilder,
    bindings: HashMap<VarId, Id>,
    insertion: Vec<VarId>,
}

impl Default for SubstitutionMap {
    fn default() -> Self {
        Self::new()
    }
}

impl SubstitutionMap {
    #[must_use]
    pub fn new() -> Self {
        Self {
            builder: ExprBuilder::new(),
            bindings: HashMap::new(),
            insertion: Vec::new(),
        }
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.bindings.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bindings.is_empty()
    }

    #[must_use]
    pub fn get(&self, v: VarId) -> Option<Id> {
        self.bindings.get(&v).copied()
    }

    #[must_use]
    pub fn contains(&self, v: VarId) -> bool {
        self.bindings.contains_key(&v)
    }

    /// Current e-graph snapshot (all merged expressions).
    #[must_use]
    pub fn egraph(&self) -> EGraph<RationalLang, ()> {
        self.builder.snapshot()
    }

    /// Add `v \mapsto` expression copied from `expr_graph` at `expr_root`.
    pub fn insert(&mut self, v: VarId, expr_graph: &EGraph<RationalLang, ()>, expr_root: Id) {
        let root = self.builder.merge_root(expr_graph, expr_root);
        if self.bindings.insert(v, root).is_some() {
            self.insertion.retain(|x| *x != v);
        }
        self.insertion.push(v);
    }

    /// Variables in insertion order.
    #[must_use]
    pub fn insertion_order(&self) -> &[VarId] {
        &self.insertion
    }

    /// Topological order: if `u` appears inside the expression for `v`, then `u` comes before `v`.
    ///
    /// # Errors
    ///
    /// Returns [`SubstitutionError::Cycle`] when bindings contain a cyclic dependency.
    pub fn replay_order(&self) -> Result<Vec<VarId>, SubstitutionError> {
        let g = self.builder.snapshot();
        let mut deps: HashMap<VarId, Vec<VarId>> = HashMap::new();
        self.bindings.iter().for_each(|(&v, &root)| {
            let ds: Vec<VarId> = vars_in_subgraph(&g, root, &self.bindings)
                .into_iter()
                .filter(|u| *u != v)
                .collect();
            deps.insert(v, ds);
        });
        let mut order = Vec::new();
        let mut temp = HashSet::new();
        let mut perm = HashSet::new();
        self.insertion
            .iter()
            .copied()
            .try_for_each(|v| dfs_deps_first(v, &deps, &mut temp, &mut perm, &mut order))?;
        Ok(order)
    }
}

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
        us.iter()
            .copied()
            .try_for_each(|u| dfs_deps_first(u, deps, temp, perm, order))?;
    }
    temp.remove(&v);
    perm.insert(v);
    order.push(v);
    Ok(())
}

fn vars_in_subgraph(
    g: &EGraph<RationalLang, ()>,
    root: Id,
    bound: &HashMap<VarId, Id>,
) -> HashSet<VarId> {
    let extractor = Extractor::new(g, AstSize);
    let (_, recexpr) = extractor.find_best(root);
    let mut out = HashSet::new();
    let mut stack = vec![recexpr.root()];
    let mut seen = HashSet::new();
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        match recexpr[id].clone() {
            crate::expr_lang::RationalLang::Const(_) | crate::expr_lang::RationalLang::Pi => {}
            crate::expr_lang::RationalLang::Var(x) => {
                if bound.contains_key(&x) {
                    out.insert(x);
                }
            }
            crate::expr_lang::RationalLang::Add([a, b])
            | crate::expr_lang::RationalLang::Mul([a, b])
            | crate::expr_lang::RationalLang::Div([a, b])
            | crate::expr_lang::RationalLang::Pow([a, b]) => {
                stack.push(a);
                stack.push(b);
            }
            crate::expr_lang::RationalLang::Neg(x) => stack.push(x),
        }
    }
    out
}
