//! A finite set of polynomial equations represented as DAG roots.

use crate::dag::{Dag, NodeId};

/// Polynomial system: each equation is a DAG root interpreted as `p = 0`.
#[derive(Clone, Debug)]
pub struct PolynomialSystem {
    pub equations: Vec<NodeId>,
    pub dag: Dag,
}

impl PolynomialSystem {
    pub fn new(dag: Dag, equations: Vec<NodeId>) -> Self {
        Self { dag, equations }
    }
}
