//! A finite set of polynomial equations stored in one e-graph (`root_i = 0`).

use crate::expr_lang::RationalLang;
use egg::{EGraph, Id};

/// Polynomial system: each equation is an e-class id interpreted as `p = 0`.
#[derive(Clone, Debug)]
pub struct PolynomialSystem {
    pub equations: Vec<Id>,
    pub egraph: EGraph<RationalLang, ()>,
}

impl PolynomialSystem {
    #[must_use]
    pub fn new(egraph: EGraph<RationalLang, ()>, equations: Vec<Id>) -> Self {
        Self { equations, egraph }
    }
}
