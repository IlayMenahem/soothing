//! `egg` language for rational polynomial expressions (binary `Add` / `Mul`).

use crate::field::Q;
use egg::{define_language, EGraph, Id, RecExpr};
use std::collections::{BTreeSet, HashSet};
use thiserror::Error;

/// Opaque identifier for a polynomial variable (`x_1`, …).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Ord, PartialOrd)]
pub struct VarId(pub u32);

impl std::fmt::Display for VarId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::str::FromStr for VarId {
    type Err = std::num::ParseIntError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        s.parse::<u32>().map(VarId)
    }
}

define_language! {
    /// Rational expression AST node (children are binary except unary minus).
    pub enum RationalLang {
        Const(Q),
        "pi" = Pi,
        Var(VarId),
        "+" = Add([Id; 2]),
        "*" = Mul([Id; 2]),
        "neg" = Neg(Id),
        "/" = Div([Id; 2]),
        "^" = Pow([Id; 2]),
    }
}

/// Parse / load errors (same variants as the former DAG parser).
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

/// Parse a rational expression string into an e-graph and root e-class id.
pub fn from_rational_string(s: &str) -> Result<(EGraph<RationalLang, ()>, Id), DagParseError> {
    crate::dag_parse::parse_rational_expression(s)
}

/// Like [`from_rational_string`], with a fixed variable name map.
pub fn from_rational_string_with_var_map(
    s: &str,
    var_map: &std::collections::HashMap<String, VarId>,
) -> Result<(EGraph<RationalLang, ()>, Id), DagParseError> {
    crate::dag_parse::parse_rational_expression_with_var_map(s, var_map)
}

fn is_atomic_recexpr(recexpr: &RecExpr<RationalLang>, id: Id) -> bool {
    matches!(
        recexpr[id],
        RationalLang::Const(_) | RationalLang::Pi | RationalLang::Var(_)
    )
}

fn bracketed(
    recexpr: &RecExpr<RationalLang>,
    id: Id,
    s: String,
    _var_name: &impl Fn(VarId) -> String,
) -> String {
    if is_atomic_recexpr(recexpr, id) {
        s
    } else {
        format!("({s})")
    }
}

fn fmt_neg(
    recexpr: &RecExpr<RationalLang>,
    child: Id,
    var_name: &impl Fn(VarId) -> String,
) -> String {
    match &recexpr[child] {
        RationalLang::Const(c) => format!("{}", -c.clone()),
        RationalLang::Neg(_) => format!("-({})", fmt_recexpr_node(recexpr, child, var_name)),
        _ => {
            let inner = fmt_recexpr_node(recexpr, child, var_name);
            if inner.starts_with('-') {
                format!("-({inner})")
            } else {
                bracketed(recexpr, child, format!("-{inner}"), var_name)
            }
        }
    }
}

pub(crate) fn flatten_mul_terms(recexpr: &RecExpr<RationalLang>, id: Id, out: &mut Vec<Id>) {
    match &recexpr[id] {
        RationalLang::Mul([a, b]) => {
            flatten_mul_terms(recexpr, *a, out);
            flatten_mul_terms(recexpr, *b, out);
        }
        _ => out.push(id),
    }
}

pub(crate) fn flatten_add_terms(recexpr: &RecExpr<RationalLang>, id: Id, out: &mut Vec<Id>) {
    match &recexpr[id] {
        RationalLang::Add([a, b]) => {
            flatten_add_terms(recexpr, *a, out);
            flatten_add_terms(recexpr, *b, out);
        }
        _ => out.push(id),
    }
}

fn fmt_mul_list(
    recexpr: &RecExpr<RationalLang>,
    root: Id,
    var_name: &impl Fn(VarId) -> String,
) -> String {
    let mut buf = Vec::new();
    flatten_mul_terms(recexpr, root, &mut buf);
    buf.iter()
        .map(|c| {
            let s = fmt_recexpr_node(recexpr, *c, var_name);
            bracketed(recexpr, *c, s, var_name)
        })
        .collect::<Vec<_>>()
        .join(" * ")
}

fn fmt_add_list(
    recexpr: &RecExpr<RationalLang>,
    root: Id,
    var_name: &impl Fn(VarId) -> String,
) -> String {
    let mut buf = Vec::new();
    flatten_add_terms(recexpr, root, &mut buf);
    if buf.is_empty() {
        return "0".to_string();
    }
    buf.iter()
        .map(|c| fmt_recexpr_node(recexpr, *c, var_name))
        .collect::<Vec<_>>()
        .join(" + ")
}

fn fmt_div(
    recexpr: &RecExpr<RationalLang>,
    num: Id,
    den: Id,
    var_name: &impl Fn(VarId) -> String,
) -> String {
    let ns = fmt_recexpr_node(recexpr, num, var_name);
    let ds = match &recexpr[den] {
        RationalLang::Mul([a, b]) => {
            let mut v = Vec::new();
            flatten_mul_terms(recexpr, *a, &mut v);
            flatten_mul_terms(recexpr, *b, &mut v);
            if v.len() >= 2 {
                format!("({})", fmt_mul_list(recexpr, den, var_name))
            } else {
                fmt_recexpr_node(recexpr, den, var_name)
            }
        }
        _ => fmt_recexpr_node(recexpr, den, var_name),
    };
    format!("{ns} / {ds}")
}

fn fmt_recexpr_node(
    recexpr: &RecExpr<RationalLang>,
    id: Id,
    var_name: &impl Fn(VarId) -> String,
) -> String {
    match &recexpr[id] {
        RationalLang::Const(c) => format!("{c}"),
        RationalLang::Pi => "pi".to_string(),
        RationalLang::Var(v) => var_name(*v),
        RationalLang::Neg(x) => fmt_neg(recexpr, *x, var_name),
        RationalLang::Add([_, _]) => fmt_add_list(recexpr, id, var_name),
        RationalLang::Mul([_, _]) => fmt_mul_list(recexpr, id, var_name),
        RationalLang::Div([n, d]) => fmt_div(recexpr, *n, *d, var_name),
        RationalLang::Pow([b, e]) => {
            let bs = fmt_recexpr_node(recexpr, *b, var_name);
            let es = fmt_recexpr_node(recexpr, *e, var_name);
            format!("{bs} ^ {es}")
        }
    }
}

/// Pretty-print the extracted expression at `root` (use after [`egg::Extractor::find_best`]).
pub fn format_recexpr_root(
    recexpr: &RecExpr<RationalLang>,
    root: Id,
    var_name: impl Fn(VarId) -> String,
) -> String {
    fmt_recexpr_node(recexpr, root, &var_name)
}

/// Format using an e-graph root via extraction (stable representative for display).
pub fn format_root(
    egraph: &EGraph<RationalLang, ()>,
    root: Id,
    var_name: impl Fn(VarId) -> String,
) -> String {
    use egg::AstSize;
    let ext = egg::Extractor::new(egraph, AstSize);
    let (_, r) = ext.find_best(root);
    format_recexpr_root(&r, r.root(), var_name)
}

/// Variables in the expression at `root` (via best extraction), sorted by [`VarId`].
#[must_use]
pub fn vars_in_root(egraph: &EGraph<RationalLang, ()>, root: Id) -> Vec<VarId> {
    use egg::AstSize;
    let ext = egg::Extractor::new(egraph, AstSize);
    let (_, recexpr) = ext.find_best(root);
    let mut seen = HashSet::new();
    let mut vars = BTreeSet::new();
    collect_vars_recexpr(&recexpr, recexpr.root(), &mut seen, &mut vars);
    vars.into_iter().collect()
}

fn collect_vars_recexpr(
    recexpr: &RecExpr<RationalLang>,
    id: Id,
    seen: &mut HashSet<Id>,
    vars: &mut BTreeSet<VarId>,
) {
    if !seen.insert(id) {
        return;
    }
    match &recexpr[id] {
        RationalLang::Var(v) => {
            vars.insert(*v);
        }
        RationalLang::Const(_) | RationalLang::Pi => {}
        RationalLang::Add([a, b])
        | RationalLang::Mul([a, b])
        | RationalLang::Div([a, b])
        | RationalLang::Pow([a, b]) => {
            collect_vars_recexpr(recexpr, *a, seen, vars);
            collect_vars_recexpr(recexpr, *b, seen, vars);
        }
        RationalLang::Neg(x) => collect_vars_recexpr(recexpr, *x, seen, vars),
    }
}
