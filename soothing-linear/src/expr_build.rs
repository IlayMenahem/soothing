//! Construction, merge, and substitution on [`egg::EGraph`] with [`crate::expr_lang::RationalLang`].

use crate::expr_lang::{RationalLang, VarId};
use crate::field::Q;
use egg::{AstSize, EGraph, Extractor, Id, RecExpr};
use num_traits::{One, Zero};
use std::collections::HashMap;

/// Builds expressions in an e-graph (hash-consed enodes).
#[derive(Debug, Clone)]
pub struct ExprBuilder {
    pub egraph: EGraph<RationalLang, ()>,
}

impl Default for ExprBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ExprBuilder {
    #[must_use]
    pub fn new() -> Self {
        Self {
            egraph: EGraph::default(),
        }
    }

    #[must_use]
    pub fn into_egraph(self) -> EGraph<RationalLang, ()> {
        self.egraph
    }

    /// Clone of the current e-graph (for snapshots).
    #[must_use]
    pub fn snapshot(&self) -> EGraph<RationalLang, ()> {
        self.egraph.clone()
    }

    pub fn constant(&mut self, c: Q) -> Id {
        self.egraph.add(RationalLang::Const(c))
    }

    pub fn const_pi(&mut self) -> Id {
        self.egraph.add(RationalLang::Pi)
    }

    pub fn variable(&mut self, v: VarId) -> Id {
        self.egraph.add(RationalLang::Var(v))
    }

    pub fn neg(&mut self, x: Id) -> Id {
        self.egraph.add(RationalLang::Neg(x))
    }

    pub fn add(&mut self, children: Vec<Id>) -> Id {
        match children.len() {
            0 => self.constant(Q::zero()),
            1 => children[0],
            _ => {
                let mut it = children.into_iter();
                let mut acc = it.next().expect("len checked");
                for c in it {
                    acc = self.egraph.add(RationalLang::Add([acc, c]));
                }
                acc
            }
        }
    }

    pub fn mul(&mut self, children: Vec<Id>) -> Id {
        match children.len() {
            0 => self.constant(Q::one()),
            1 => children[0],
            _ => {
                let mut it = children.into_iter();
                let mut acc = it.next().expect("len checked");
                for c in it {
                    acc = self.egraph.add(RationalLang::Mul([acc, c]));
                }
                acc
            }
        }
    }

    pub fn mul2(&mut self, lhs: Id, rhs: Id) -> Id {
        self.mul(vec![lhs, rhs])
    }

    pub fn sub(&mut self, lhs: Id, rhs: Id) -> Id {
        let neg_rhs = self.neg(rhs);
        self.add(vec![lhs, neg_rhs])
    }

    pub fn div(&mut self, num: Id, den: Id) -> Id {
        self.egraph.add(RationalLang::Div([num, den]))
    }

    pub fn pow(&mut self, base: Id, exp: Id) -> Id {
        self.egraph.add(RationalLang::Pow([base, exp]))
    }

    /// Copy the expression at `root` in `source` into this builder’s e-graph.
    pub fn merge_root(&mut self, source: &EGraph<RationalLang, ()>, root: Id) -> Id {
        let extractor = Extractor::new(source, AstSize);
        let (_, recexpr) = extractor.find_best(root);
        self.egraph.add_expr(&recexpr)
    }

    /// Replace every occurrence of `var` with `repl` (an id in **this** e-graph).
    pub fn substitute_in_root(
        &mut self,
        source: &EGraph<RationalLang, ()>,
        root: Id,
        var: VarId,
        repl: Id,
    ) -> Id {
        let extractor = Extractor::new(source, AstSize);
        let (_, recexpr) = extractor.find_best(root);
        let mut memo = HashMap::new();
        self.substitute_recexpr(&recexpr, recexpr.root(), var, repl, &mut memo)
    }

    pub fn substitute_var_with_const(
        &mut self,
        source: &EGraph<RationalLang, ()>,
        root: Id,
        var: VarId,
        value: Q,
    ) -> Id {
        let c = self.constant(value);
        self.substitute_in_root(source, root, var, c)
    }

    fn substitute_recexpr(
        &mut self,
        recexpr: &RecExpr<RationalLang>,
        id: Id,
        var: VarId,
        repl: Id,
        memo: &mut HashMap<Id, Id>,
    ) -> Id {
        if let Some(&n) = memo.get(&id) {
            return n;
        }
        let mapped = match recexpr[id].clone() {
            RationalLang::Var(v) if v == var => repl,
            RationalLang::Const(c) => self.constant(c),
            RationalLang::Pi => self.const_pi(),
            RationalLang::Var(v) => self.variable(v),
            RationalLang::Add([a, b]) => {
                let aa = self.substitute_recexpr(recexpr, a, var, repl, memo);
                let bb = self.substitute_recexpr(recexpr, b, var, repl, memo);
                self.egraph.add(RationalLang::Add([aa, bb]))
            }
            RationalLang::Mul([a, b]) => {
                let aa = self.substitute_recexpr(recexpr, a, var, repl, memo);
                let bb = self.substitute_recexpr(recexpr, b, var, repl, memo);
                self.egraph.add(RationalLang::Mul([aa, bb]))
            }
            RationalLang::Neg(x) => {
                let y = self.substitute_recexpr(recexpr, x, var, repl, memo);
                self.neg(y)
            }
            RationalLang::Div([n, d]) => {
                let nn = self.substitute_recexpr(recexpr, n, var, repl, memo);
                let dd = self.substitute_recexpr(recexpr, d, var, repl, memo);
                self.div(nn, dd)
            }
            RationalLang::Pow([b, e]) => {
                let bb = self.substitute_recexpr(recexpr, b, var, repl, memo);
                let ee = self.substitute_recexpr(recexpr, e, var, repl, memo);
                self.pow(bb, ee)
            }
        };
        memo.insert(id, mapped);
        mapped
    }
}
