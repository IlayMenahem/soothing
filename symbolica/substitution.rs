//! Substitution map with dependency-ordered replay.

use std::collections::{HashMap, HashSet};

use symbolica::atom::{Atom, Symbol};
use thiserror::Error;

use crate::atom_util::vars_in_atom;
use crate::var_id::VarId;

#[derive(Debug, Error, Eq, PartialEq)]
pub enum SubstitutionError {
    #[error("cyclic dependency between substituted variables")]
    Cycle,
}

#[derive(Clone, Debug)]
pub struct SubstitutionMap {
    bindings: HashMap<VarId, Atom>,
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
            bindings: HashMap::default(),
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
    pub fn get(&self, v: VarId) -> Option<&Atom> {
        self.bindings.get(&v)
    }

    pub fn insert(&mut self, v: VarId, rhs: Atom) {
        if self.bindings.insert(v, rhs).is_some() {
            self.insertion.retain(|x| *x != v);
        }
        self.insertion.push(v);
    }

    pub fn replay_order(
        &self,
        sym_to_vid: &HashMap<Symbol, VarId>,
    ) -> Result<Vec<VarId>, SubstitutionError> {
        let dag_deps: HashMap<VarId, Vec<VarId>> = self
            .bindings
            .iter()
            .map(|(&v, rhs)| {
                let vs: Vec<VarId> = vars_in_atom(rhs, sym_to_vid)
                    .into_iter()
                    .filter(|u| *u != v && self.bindings.contains_key(u))
                    .collect();
                (v, vs)
            })
            .collect();

        let mut order = Vec::new();
        let mut temp = HashSet::new();
        let mut perm = HashSet::new();
        for &v in &self.insertion {
            dfs_deps_first(v, &dag_deps, &mut temp, &mut perm, &mut order)?;
        }
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
        for u in us {
            dfs_deps_first(*u, deps, temp, perm, order)?;
        }
    }
    temp.remove(&v);
    perm.insert(v);
    order.push(v);
    Ok(())
}
