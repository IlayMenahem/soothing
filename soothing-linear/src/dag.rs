//! DAG representation of polynomial expressions (shared subexpressions, unexpanded form).

use crate::field::Q;
use std::collections::{BTreeSet, HashMap, HashSet};
use thiserror::Error;

/// Opaque identifier for a polynomial variable (`x_1`, …).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Ord, PartialOrd)]
pub struct VarId(pub u32);

/// Index into the DAG arena.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct NodeId(pub usize);

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) enum Node {
    Const(Q),
    /// Mathematica `\[Pi]` / identifier `pi`; not a [`VarId`]. Evaluates as pi ≈ 355/113.
    ConstPi,
    Var(VarId),
    Add(Vec<NodeId>),
    Mul(Vec<NodeId>),
    Neg(NodeId),
    Div(NodeId, NodeId),
    Pow(NodeId, NodeId),
}

/// Immutable DAG: nodes stored in an arena; roots reference subgraphs.
#[derive(Clone, Debug)]
pub struct Dag {
    pub(crate) nodes: Vec<Node>,
}

impl Dag {
    pub(crate) fn node(&self, id: NodeId) -> &Node {
        &self.nodes[id.0]
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Variables that occur in the subgraph at `root`, sorted by [`VarId`].
    #[must_use]
    pub fn vars_in_root(&self, root: NodeId) -> Vec<VarId> {
        let mut seen_node = HashSet::new();
        let mut vars = BTreeSet::new();
        self.collect_vars_dfs(root, &mut seen_node, &mut vars);
        vars.into_iter().collect()
    }

    fn collect_vars_dfs(
        &self,
        id: NodeId,
        seen_node: &mut HashSet<NodeId>,
        vars: &mut BTreeSet<VarId>,
    ) {
        if !seen_node.insert(id) {
            return;
        }
        match self.node(id) {
            Node::Var(v) => {
                vars.insert(*v);
            }
            Node::Const(_) | Node::ConstPi => {}
            Node::Add(cs) | Node::Mul(cs) => {
                for c in cs {
                    self.collect_vars_dfs(*c, seen_node, vars);
                }
            }
            Node::Neg(x) => self.collect_vars_dfs(*x, seen_node, vars),
            Node::Div(n, d) => {
                self.collect_vars_dfs(*n, seen_node, vars);
                self.collect_vars_dfs(*d, seen_node, vars);
            }
            Node::Pow(b, e) => {
                self.collect_vars_dfs(*b, seen_node, vars);
                self.collect_vars_dfs(*e, seen_node, vars);
            }
        }
    }

    pub fn format_root(&self, root: NodeId, var_name: impl Fn(VarId) -> String) -> String {
        self.fmt_node(root, &var_name)
    }

    fn fmt_node(&self, id: NodeId, var_name: &impl Fn(VarId) -> String) -> String {
        fn fmt_neg(dag: &Dag, child: NodeId, var_name: &impl Fn(VarId) -> String) -> String {
            match dag.node(child) {
                Node::Const(c) => format!("{}", -*c),
                _ => format!("-({})", dag.fmt_node(child, var_name)),
            }
        }

        fn fmt_add(dag: &Dag, ids: &[NodeId], var_name: &impl Fn(VarId) -> String) -> String {
            if ids.is_empty() {
                "0".to_string()
            } else {
                ids.iter()
                    .map(|c| dag.fmt_node(*c, var_name))
                    .collect::<Vec<_>>()
                    .join(" + ")
            }
        }

        fn fmt_mul(dag: &Dag, ids: &[NodeId], var_name: &impl Fn(VarId) -> String) -> String {
            if ids.is_empty() {
                "1".to_string()
            } else {
                ids.iter()
                    .map(|c| format!("({})", dag.fmt_node(*c, var_name)))
                    .collect::<Vec<_>>()
                    .join(" * ")
            }
        }

        fn fmt_div(
            dag: &Dag,
            num: NodeId,
            den: NodeId,
            var_name: &impl Fn(VarId) -> String,
        ) -> String {
            let ns = dag.fmt_node(num, var_name);
            let ds = dag.fmt_node(den, var_name);
            format!("({ns} / {ds})")
        }

        fn fmt_pow(
            dag: &Dag,
            base: NodeId,
            exp: NodeId,
            var_name: &impl Fn(VarId) -> String,
        ) -> String {
            let bs = dag.fmt_node(base, var_name);
            let es = dag.fmt_node(exp, var_name);
            format!("{bs} ^ {es}")
        }

        match self.node(id) {
            Node::Const(c) => format!("{c}"),
            Node::ConstPi => "pi".to_string(),
            Node::Var(v) => var_name(*v),
            Node::Neg(x) => fmt_neg(self, *x, var_name),
            Node::Add(cs) => fmt_add(self, cs, var_name),
            Node::Mul(cs) => fmt_mul(self, cs, var_name),
            Node::Div(n, d) => fmt_div(self, *n, *d, var_name),
            Node::Pow(b, e) => fmt_pow(self, *b, *e, var_name),
        }
    }

    /// Parse a rational expression into a DAG and its root.
    ///
    /// Syntax is parsed by the [`arithmetic-parser`](https://docs.rs/arithmetic-parser) crate with a
    /// custom rational literal grammar (decimal integers or `p/q`), no statements or semicolons, and
    /// features limited to basic arithmetic (no tuples, blocks, `//` is a line comment).
    ///
    /// - **Rational literals:** integers or `p/q` with `i128` numerator and denominator (parse errors if out of range).
    /// - **Variables:** `vN` for `N` decimal digits maps to [`VarId`]`(N)`. Any other identifier
    ///   gets a fresh [`VarId`] on first occurrence (stable across the string).
    /// - **Operators:** `+`, `-` (binary and unary), `*`, `/`, `^` (right-associative), parentheses.
    ///
    /// # Errors
    ///
    /// Returns [`DagParseError`] when the input is not a valid rational expression (syntax, invalid
    /// literals, trailing text, etc.).
    pub fn from_rational_string(s: &str) -> Result<(Dag, NodeId), DagParseError> {
        crate::dag_parse::parse_rational_expression(s)
    }

    /// Like [`from_rational_string`](Self::from_rational_string), but every identifier must appear in
    /// `var_map` (including `v0`-style names if used).
    ///
    /// # Errors
    ///
    /// Returns [`DagParseError`] on the same conditions as [`from_rational_string`](Self::from_rational_string),
    /// and additionally [`DagParseError::UnknownVariable`] when a name is not present in `var_map`.
    pub fn from_rational_string_with_var_map(
        s: &str,
        var_map: &std::collections::HashMap<String, VarId>,
    ) -> Result<(Dag, NodeId), DagParseError> {
        crate::dag_parse::parse_rational_expression_with_var_map(s, var_map)
    }
}

/// Error from [`Dag::from_rational_string`].
#[derive(Debug, Error, Eq, PartialEq)]
pub enum DagParseError {
    #[error("unexpected end of input")]
    UnexpectedEof,
    #[error("unexpected character {ch:?} at byte {pos}")]
    UnexpectedChar { ch: char, pos: usize },
    #[error("expected {expected} at byte {pos}")]
    Expected { expected: &'static str, pos: usize },
    #[error("invalid number at byte {pos}: {message}")]
    InvalidNumber { pos: usize, message: String },
    #[error("trailing input at byte {pos}")]
    TrailingInput { pos: usize },
    #[error("unsupported expression at byte {pos}: {reason}")]
    Unsupported { reason: &'static str, pos: usize },
    #[error("unknown variable {name:?} at byte {pos}")]
    UnknownVariable { name: String, pos: usize },
}

/// Builds [`Dag`] expressions and interns identical subexpressions when possible.
#[derive(Debug, Clone)]
pub struct DagBuilder {
    nodes: Vec<Node>,
    intern: HashMap<Node, NodeId>,
}

impl Default for DagBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl DagBuilder {
    #[must_use]
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            intern: HashMap::new(),
        }
    }

    pub(crate) fn node(&self, id: NodeId) -> &Node {
        &self.nodes[id.0]
    }

    fn push_node(&mut self, node: Node) -> NodeId {
        let id = NodeId(self.nodes.len());
        self.nodes.push(node);
        id
    }

    fn intern_node(&mut self, node: Node) -> NodeId {
        if let Some(&id) = self.intern.get(&node) {
            return id;
        }
        let id = self.push_node(node.clone());
        self.intern.insert(node, id);
        id
    }

    pub fn constant(&mut self, c: Q) -> NodeId {
        self.intern_node(Node::Const(c))
    }

    pub fn const_pi(&mut self) -> NodeId {
        self.intern_node(Node::ConstPi)
    }

    pub fn variable(&mut self, v: VarId) -> NodeId {
        self.intern_node(Node::Var(v))
    }

    pub fn neg(&mut self, x: NodeId) -> NodeId {
        self.intern_node(Node::Neg(x))
    }

    pub fn add(&mut self, children: Vec<NodeId>) -> NodeId {
        self.intern_node(Node::Add(children))
    }

    pub fn mul(&mut self, children: Vec<NodeId>) -> NodeId {
        self.intern_node(Node::Mul(children))
    }

    /// Binary multiplication (`lhs * rhs`).
    pub fn mul2(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        self.mul(vec![lhs, rhs])
    }

    /// Binary subtraction (`lhs - rhs`), represented as `lhs + (-rhs)`.
    pub fn sub(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        let neg_rhs = self.neg(rhs);
        self.add(vec![lhs, neg_rhs])
    }

    pub fn div(&mut self, num: NodeId, den: NodeId) -> NodeId {
        self.intern_node(Node::Div(num, den))
    }

    pub fn pow(&mut self, base: NodeId, exp: NodeId) -> NodeId {
        self.intern_node(Node::Pow(base, exp))
    }

    #[must_use]
    pub fn build(self) -> Dag {
        Dag { nodes: self.nodes }
    }

    /// Immutable snapshot of nodes built so far (for sharing with [`SubstitutionMap`]).
    #[must_use]
    pub fn snapshot(&self) -> Dag {
        Dag {
            nodes: self.nodes.clone(),
        }
    }

    /// Copy the subgraph reachable from `root` in `source` into this builder.
    pub fn merge_root(&mut self, source: &Dag, root: NodeId) -> NodeId {
        let mut memo = HashMap::new();
        self.merge_dfs(source, root, &mut memo)
    }

    fn merge_children(
        &mut self,
        source: &Dag,
        cs: &[NodeId],
        memo: &mut HashMap<NodeId, NodeId>,
    ) -> Vec<NodeId> {
        cs.iter()
            .map(|c| self.merge_dfs(source, *c, memo))
            .collect()
    }

    fn merge_dfs(
        &mut self,
        source: &Dag,
        id: NodeId,
        memo: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId {
        if let Some(&n) = memo.get(&id) {
            return n;
        }

        let mapped = match source.node(id) {
            Node::Const(c) => self.constant(*c),
            Node::ConstPi => self.const_pi(),
            Node::Var(v) => self.variable(*v),
            Node::Add(cs) => {
                let v = self.merge_children(source, cs, memo);
                self.add(v)
            }
            Node::Mul(cs) => {
                let v = self.merge_children(source, cs, memo);
                self.mul(v)
            }
            Node::Neg(x) => {
                let y = self.merge_dfs(source, *x, memo);
                self.neg(y)
            }
            Node::Div(n, d) => {
                let nn = self.merge_dfs(source, *n, memo);
                let dd = self.merge_dfs(source, *d, memo);
                self.div(nn, dd)
            }
            Node::Pow(b, e) => {
                let bb = self.merge_dfs(source, *b, memo);
                let ee = self.merge_dfs(source, *e, memo);
                self.pow(bb, ee)
            }
        };
        memo.insert(id, mapped);
        mapped
    }

    /// Replace every occurrence of `var` with `repl` (a node already in this builder).
    pub fn substitute_in_root(
        &mut self,
        source: &Dag,
        root: NodeId,
        var: VarId,
        repl: NodeId,
    ) -> NodeId {
        let mut memo = HashMap::new();
        self.substitute_dfs(source, root, var, repl, &mut memo)
    }

    /// Replace `var` with constant `value` throughout `source` at `root`.
    pub fn substitute_var_with_const(
        &mut self,
        source: &Dag,
        root: NodeId,
        var: VarId,
        value: Q,
    ) -> NodeId {
        let c = self.constant(value);
        self.substitute_in_root(source, root, var, c)
    }

    fn substitute_children(
        &mut self,
        source: &Dag,
        cs: &[NodeId],
        var: VarId,
        repl: NodeId,
        memo: &mut HashMap<NodeId, NodeId>,
    ) -> Vec<NodeId> {
        cs.iter()
            .map(|c| self.substitute_dfs(source, *c, var, repl, memo))
            .collect()
    }

    fn substitute_dfs(
        &mut self,
        source: &Dag,
        id: NodeId,
        var: VarId,
        repl: NodeId,
        memo: &mut HashMap<NodeId, NodeId>,
    ) -> NodeId {
        if let Some(&n) = memo.get(&id) {
            return n;
        }
        let mapped = match source.node(id) {
            Node::Var(v) if *v == var => repl,
            Node::Const(c) => self.constant(*c),
            Node::ConstPi => self.const_pi(),
            Node::Var(v) => self.variable(*v),
            Node::Add(cs) => {
                let v = self.substitute_children(source, cs, var, repl, memo);
                self.add(v)
            }
            Node::Mul(cs) => {
                let v = self.substitute_children(source, cs, var, repl, memo);
                self.mul(v)
            }
            Node::Neg(x) => {
                let y = self.substitute_dfs(source, *x, var, repl, memo);
                self.neg(y)
            }
            Node::Div(n, d) => {
                let nn = self.substitute_dfs(source, *n, var, repl, memo);
                let dd = self.substitute_dfs(source, *d, var, repl, memo);
                self.div(nn, dd)
            }
            Node::Pow(b, e) => {
                let bb = self.substitute_dfs(source, *b, var, repl, memo);
                let ee = self.substitute_dfs(source, *e, var, repl, memo);
                self.pow(bb, ee)
            }
        };
        memo.insert(id, mapped);
        mapped
    }
}
