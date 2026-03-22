//! DAG representation of polynomial expressions (paper: unexpanded graph form).

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

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Variables that occur in the subgraph at `root`, sorted by [`VarId`].
    pub fn vars_in_root(&self, root: NodeId) -> Vec<VarId> {
        let mut seen_node = HashSet::new();
        let mut vars = BTreeSet::new();
        self.collect_vars_dfs(root, &mut seen_node, &mut vars);
        vars.into_iter().collect()
    }

    fn collect_vars_dfs(&self, id: NodeId, seen_node: &mut HashSet<NodeId>, vars: &mut BTreeSet<VarId>) {
        if !seen_node.insert(id) {
            return;
        }
        match self.node(id) {
            Node::Var(v) => {
                vars.insert(*v);
            }
            Node::Const(_) => {}
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
        match self.node(id) {
            Node::Const(c) => format!("{}", c),
            Node::Var(v) => var_name(*v),
            Node::Neg(x) => {
                let inner = self.fmt_node(*x, var_name);
                if self.is_atomic(*x) {
                    format!("-{inner}")
                } else {
                    format!("-({inner})")
                }
            }
            Node::Add(cs) => {
                if cs.is_empty() {
                    "0".to_string()
                } else {
                    cs.iter()
                        .map(|c| self.fmt_node(*c, var_name))
                        .collect::<Vec<_>>()
                        .join(" + ")
                }
            }
            Node::Mul(cs) => {
                if cs.is_empty() {
                    "1".to_string()
                } else {
                    cs.iter()
                        .map(|c| {
                            let s = self.fmt_node(*c, var_name);
                            if self.is_atomic(*c) {
                                s
                            } else {
                                format!("({s})")
                            }
                        })
                        .collect::<Vec<_>>()
                        .join(" * ")
                }
            }
            Node::Div(n, d) => {
                let ns = self.fmt_node(*n, var_name);
                let ds = self.fmt_node(*d, var_name);
                let num = if self.is_atomic(*n) {
                    ns
                } else {
                    format!("({ns})")
                };
                let den = if self.is_atomic(*d) {
                    ds
                } else {
                    format!("({ds})")
                };
                format!("{num} / {den}")
            }
            Node::Pow(b, e) => {
                let bs = self.fmt_node(*b, var_name);
                let es = self.fmt_node(*e, var_name);
                let left = if self.is_atomic(*b) {
                    bs
                } else {
                    format!("({bs})")
                };
                let right = if self.is_atomic(*e) {
                    es
                } else {
                    format!("({es})")
                };
                format!("{left} ^ {right}")
            }
        }
    }

    fn is_atomic(&self, id: NodeId) -> bool {
        matches!(self.node(id), Node::Const(_) | Node::Var(_))
    }

    /// Parse a rational expression into a DAG and its root.
    ///
    /// Syntax (whitespace is ignored):
    /// - **Rational literals:** integers or `p/q` with `i64` numerator and denominator (parsed as
    ///   left-associative `/` with the same precedence as `*`).
    /// - **Variables:** `vN` for `N` decimal digits maps to [`VarId`]`(N)`. Any other identifier
    ///   gets a fresh [`VarId`] on first occurrence (stable across the string).
    /// - **Operators:** `+`, `-` (binary and unary), `*`, `/`, `^` (right-associative), parentheses.
    ///
    /// This is meant for strings in the same style as [`Dag::format_root`] (e.g. `x * y + 1 / 2`).
    pub fn from_rational_string(s: &str) -> Result<(Dag, NodeId), DagParseError> {
        parse_rational_expression(s)
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
}

struct RationalParser<'a> {
    input: &'a str,
    pos: usize,
    builder: DagBuilder,
    /// Names like `x` → id; does not include `v12`-style indices.
    name_to_var: HashMap<String, VarId>,
    next_free: u32,
}

impl<'a> RationalParser<'a> {
    fn new(input: &'a str) -> Self {
        Self {
            input,
            pos: 0,
            builder: DagBuilder::new(),
            name_to_var: HashMap::new(),
            next_free: 0,
        }
    }

    fn skip_ws(&mut self) {
        while self
            .peek_byte()
            .is_some_and(|b| b.is_ascii_whitespace())
        {
            self.pos += 1;
        }
    }

    fn peek_byte(&self) -> Option<u8> {
        self.input.as_bytes().get(self.pos).copied()
    }

    fn expect_byte(&mut self, b: u8, expected: &'static str) -> Result<(), DagParseError> {
        self.skip_ws();
        if self.peek_byte() == Some(b) {
            self.pos += 1;
            Ok(())
        } else {
            Err(DagParseError::Expected {
                expected,
                pos: self.pos,
            })
        }
    }

    fn var_for_ident(&mut self, name: &str) -> Result<VarId, DagParseError> {
        if let Some(id) = var_id_from_v_digits(name) {
            return Ok(id);
        }
        if let Some(&id) = self.name_to_var.get(name) {
            return Ok(id);
        }
        let id = VarId(self.next_free);
        self.next_free = self.next_free.saturating_add(1);
        self.name_to_var.insert(name.to_string(), id);
        Ok(id)
    }

    fn parse_i64(&mut self, start: usize) -> Result<(i64, usize), DagParseError> {
        let bytes = self.input.as_bytes();
        let mut i = start;
        if i >= bytes.len() {
            return Err(DagParseError::UnexpectedEof);
        }
        if bytes[i] == b'-' {
            i += 1;
        }
        let digit_start = i;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
        }
        if i == digit_start {
            return Err(DagParseError::InvalidNumber {
                pos: start,
                message: "expected digits".into(),
            });
        }
        let s = std::str::from_utf8(&bytes[digit_start..i]).unwrap();
        let n: i64 = s.parse::<i64>().map_err(|e| DagParseError::InvalidNumber {
            pos: start,
            message: e.to_string(),
        })?;
        let signed = if bytes.get(start) == Some(&b'-') {
            -n
        } else {
            n
        };
        Ok((signed, i))
    }

    /// Reads a rational literal starting at `start` (must be `-` or digit).
    fn parse_rational_literal(&mut self, start: usize) -> Result<(Q, usize), DagParseError> {
        let (num, mut i) = self.parse_i64(start)?;
        if self.input.as_bytes().get(i) == Some(&b'/') {
            i += 1;
            let (den, j) = self.parse_i64(i)?;
            if den == 0 {
                return Err(DagParseError::InvalidNumber {
                    pos: i,
                    message: "zero denominator".into(),
                });
            }
            Ok((Q::new(num, den), j))
        } else {
            Ok((Q::new(num, 1), i))
        }
    }

    fn parse_ident(&mut self, start: usize) -> Result<(String, usize), DagParseError> {
        let bytes = self.input.as_bytes();
        let mut i = start;
        if i >= bytes.len() {
            return Err(DagParseError::UnexpectedEof);
        }
        let b0 = bytes[i];
        if !(b0.is_ascii_alphabetic() || b0 == b'_') {
            return Err(DagParseError::UnexpectedChar {
                ch: b0 as char,
                pos: start,
            });
        }
        i += 1;
        while i < bytes.len() {
            let b = bytes[i];
            if b.is_ascii_alphanumeric() || b == b'_' {
                i += 1;
            } else {
                break;
            }
        }
        let name = self.input[start..i].to_string();
        Ok((name, i))
    }

    fn parse_primary(&mut self) -> Result<NodeId, DagParseError> {
        self.skip_ws();
        let start = self.pos;
        let Some(b0) = self.peek_byte() else {
            return Err(DagParseError::UnexpectedEof);
        };
        if b0 == b'(' {
            self.pos += 1;
            let n = self.parse_expr()?;
            self.expect_byte(b')', "`)`")?;
            return Ok(n);
        }
        if b0 == b'-' || b0.is_ascii_digit() {
            let (q, end) = self.parse_rational_literal(start)?;
            self.pos = end;
            return Ok(self.builder.constant(q));
        }
        if b0.is_ascii_alphabetic() || b0 == b'_' {
            let (name, end) = self.parse_ident(start)?;
            self.pos = end;
            let v = self.var_for_ident(&name)?;
            return Ok(self.builder.variable(v));
        }
        Err(DagParseError::UnexpectedChar {
            ch: b0 as char,
            pos: start,
        })
    }

    fn parse_unary(&mut self) -> Result<NodeId, DagParseError> {
        self.skip_ws();
        if self.peek_byte() == Some(b'-') {
            self.pos += 1;
            let x = self.parse_unary()?;
            return Ok(self.builder.neg(x));
        }
        if self.peek_byte() == Some(b'+') {
            self.pos += 1;
            return self.parse_unary();
        }
        self.parse_primary()
    }

    fn parse_pow(&mut self) -> Result<NodeId, DagParseError> {
        let left = self.parse_unary()?;
        self.skip_ws();
        if self.peek_byte() == Some(b'^') {
            self.pos += 1;
            let right = self.parse_pow()?;
            Ok(self.builder.pow(left, right))
        } else {
            Ok(left)
        }
    }

    fn parse_mul(&mut self) -> Result<NodeId, DagParseError> {
        let mut acc = self.parse_pow()?;
        loop {
            self.skip_ws();
            if self.peek_byte() == Some(b'*') {
                self.pos += 1;
                let rhs = self.parse_pow()?;
                acc = self.builder.mul(vec![acc, rhs]);
            } else if self.peek_byte() == Some(b'/') {
                self.pos += 1;
                let rhs = self.parse_pow()?;
                acc = self.builder.div(acc, rhs);
            } else {
                break;
            }
        }
        Ok(acc)
    }

    fn parse_expr(&mut self) -> Result<NodeId, DagParseError> {
        let mut terms = vec![self.parse_mul()?];
        loop {
            self.skip_ws();
            let op = self.peek_byte();
            if op == Some(b'+') || op == Some(b'-') {
                self.pos += 1;
                let rhs = self.parse_mul()?;
                if op == Some(b'+') {
                    terms.push(rhs);
                } else {
                    terms.push(self.builder.neg(rhs));
                }
            } else {
                break;
            }
        }
        if terms.len() == 1 {
            Ok(terms[0])
        } else {
            Ok(self.builder.add(terms))
        }
    }

    fn finish(mut self) -> Result<(Dag, NodeId), DagParseError> {
        let root = self.parse_expr()?;
        self.skip_ws();
        if self.pos < self.input.len() {
            return Err(DagParseError::TrailingInput { pos: self.pos });
        }
        Ok((self.builder.build(), root))
    }
}

fn var_id_from_v_digits(name: &str) -> Option<VarId> {
    let mut chars = name.chars();
    let first = chars.next()?;
    if first != 'v' && first != 'V' {
        return None;
    }
    let rest: String = chars.collect();
    if rest.is_empty() || !rest.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    rest.parse::<u32>().ok().map(VarId)
}

fn parse_rational_expression(s: &str) -> Result<(Dag, NodeId), DagParseError> {
    RationalParser::new(s).finish()
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

    pub fn div(&mut self, num: NodeId, den: NodeId) -> NodeId {
        self.intern_node(Node::Div(num, den))
    }

    pub fn pow(&mut self, base: NodeId, exp: NodeId) -> NodeId {
        self.intern_node(Node::Pow(base, exp))
    }

    pub fn build(self) -> Dag {
        Dag { nodes: self.nodes }
    }

    /// Immutable snapshot of nodes built so far (for sharing with [`SubstitutionMap`]).
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

    fn merge_dfs(&mut self, source: &Dag, id: NodeId, memo: &mut HashMap<NodeId, NodeId>) -> NodeId {
        if let Some(&n) = memo.get(&id) {
            return n;
        }
        let mapped = match source.node(id) {
            Node::Const(c) => self.constant(*c),
            Node::Var(v) => self.variable(*v),
            Node::Add(cs) => {
                let v: Vec<NodeId> = cs
                    .iter()
                    .map(|c| self.merge_dfs(source, *c, memo))
                    .collect();
                self.add(v)
            }
            Node::Mul(cs) => {
                let v: Vec<NodeId> = cs
                    .iter()
                    .map(|c| self.merge_dfs(source, *c, memo))
                    .collect();
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
            Node::Var(v) => self.variable(*v),
            Node::Add(cs) => {
                let v: Vec<NodeId> = cs
                    .iter()
                    .map(|c| self.substitute_dfs(source, *c, var, repl, memo))
                    .collect();
                self.add(v)
            }
            Node::Mul(cs) => {
                let v: Vec<NodeId> = cs
                    .iter()
                    .map(|c| self.substitute_dfs(source, *c, var, repl, memo))
                    .collect();
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
