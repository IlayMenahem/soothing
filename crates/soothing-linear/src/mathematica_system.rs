//! Load Mathematica-style rule lists into [`PolynomialSystem`].
//!
//! Each rule `lhs -> rhs` becomes `(lhs) - (numerator(rhs)) = 0`, where `numerator(rhs)` is the part
//! before the first `/` at nesting depth 0 in `rhs` (after stripping matching outer parentheses).
//! This matches clearing denominators for `P/Q = 0` when `rhs` is a single top-level fraction.
//! Denominators (e.g. containing `\[Pi]`) are not parsed.
//!
//! **Limitation:** if `rhs` is not a single fraction at the outer level (e.g. `A/B + C` with `+` at
//! depth 0), taking the numerator slice is wrong. The simplified system file is assumed to match the
//! one-fraction shape.

use crate::dag::{Dag, DagBuilder, DagParseError, VarId};
use crate::system::PolynomialSystem;
use regex::Regex;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;
use thiserror::Error;

/// Repo `systems/` directory when built from `crates/soothing-linear` (same layout as Python `systems.py`).
pub fn default_systems_dir_from_manifest() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../systems")
}

#[derive(Debug, Error)]
pub enum MathematicaLoadError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("empty mathematica system")]
    Empty,
    #[error("rule has no separator (->, :>, or ==): {0:?}")]
    NoRuleSeparator(String),
    #[error("dag parse: {0}")]
    DagParse(#[from] DagParseError),
}

fn strip_outer_braces(text: &str) -> &str {
    let text = text.trim();
    if text.starts_with('{') && text.ends_with('}') {
        text[1..text.len() - 1].trim()
    } else {
        text
    }
}

/// Comma-separated top-level segments; nesting depth counts **parentheses only** so `\[Pi]` does not
/// perturb depth inside `(512 \[Pi]^2)`.
fn split_top_level(text: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut buf: Vec<char> = Vec::new();
    let mut depth = 0i32;
    for ch in text.chars() {
        match ch {
            '(' => {
                depth += 1;
                buf.push(ch);
            }
            ')' => {
                depth = (depth - 1).max(0);
                buf.push(ch);
            }
            ',' if depth == 0 => {
                let part: String = buf.iter().collect();
                let part = part.trim().to_string();
                if !part.is_empty() {
                    parts.push(part);
                }
                buf.clear();
            }
            _ => buf.push(ch),
        }
    }
    let tail: String = buf.iter().collect();
    let tail = tail.trim().to_string();
    if !tail.is_empty() {
        parts.push(tail);
    }
    parts
}

fn split_top_level_once<'a>(text: &'a str, op: &str) -> Option<(&'a str, &'a str)> {
    let mut depth = 0i32;
    let mut i = 0;
    while i + op.len() <= text.len() {
        let ch = text[i..].chars().next().unwrap();
        match ch {
            '(' => depth += 1,
            ')' => depth = (depth - 1).max(0),
            _ => {}
        }
        if depth == 0 && text[i..].starts_with(op) {
            let lhs = text[..i].trim();
            let rhs = text[i + op.len()..].trim();
            return Some((lhs, rhs));
        }
        i += ch.len_utf8();
    }
    None
}

fn split_rule(segment: &str) -> Option<(&str, &str)> {
    let s = segment.trim().trim_end_matches(',');
    if s.is_empty() {
        return None;
    }
    split_top_level_once(s, "==")
        .or_else(|| split_top_level_once(s, "->"))
        .or_else(|| split_top_level_once(s, ":>"))
}

fn collapse_ws(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

static IMPLICIT_MUL_PATTERNS: LazyLock<Vec<(Regex, &'static str)>> = LazyLock::new(|| {
    let patterns: &[(&str, &str)] = &[
        (r"\)\s+([a-zA-Z])", r")*$1"),
        (r"\)\s+(\d)", r")*$1"),
        (r"([a-zA-Z_][a-zA-Z0-9_]*)\s+\(", r"$1*("),
        (r"(\d)\s+\(", r"$1*("),
        (r"([a-zA-Z_][a-zA-Z0-9_]*)\s+([a-zA-Z])", r"$1*$2"),
        (r"(\d)\s+([a-zA-Z])", r"$1*$2"),
        (r"([a-zA-Z_][a-zA-Z0-9_]*)\s+(\d)", r"$1*$2"),
        (r"(\d)([a-zA-Z])", r"$1*$2"),
        (r"([a-zA-Z_][a-zA-Z0-9_]*)\(", r"$1*("),
        (r"(\d)\(", r"$1*("),
        (r"\)([a-zA-Z])", r")*$1"),
        (r"\)(\d)", r")*$1"),
        (r"\) \(", r")*("),
        (r"\)\(", r")*("),
    ];
    patterns
        .iter()
        .map(|(p, r)| {
            (
                Regex::new(p).expect("implicit mul regex"),
                *r,
            )
        })
        .collect()
});

fn apply_implicit_multiplication(s: &str) -> String {
    let mut result = s.to_string();
    for (re, repl) in IMPLICIT_MUL_PATTERNS.iter() {
        result = re.replace_all(&result, *repl).to_string();
    }
    result
}

fn outer_parens_match(t: &str) -> bool {
    let mut depth = 0i32;
    for (i, ch) in t.char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 && i != t.len().saturating_sub(1) {
                    return false;
                }
            }
            _ => {}
        }
    }
    depth == 0
}

fn strip_outer_parens(mut s: &str) -> &str {
    loop {
        let t = s.trim();
        if t.len() < 2 || !t.starts_with('(') || !t.ends_with(')') {
            return t;
        }
        if !outer_parens_match(t) {
            return t;
        }
        s = &t[1..t.len() - 1];
    }
}

/// If `s` contains `/` at nesting depth 0, return `(numerator, denominator)` slices.
fn split_fraction_depth_zero(s: &str) -> Option<(&str, &str)> {
    let t = s.trim();
    let mut depth = 0i32;
    for (i, ch) in t.char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => depth = (depth - 1).max(0),
            '/' if depth == 0 => {
                let left = t[..i].trim();
                let right = t[i + ch.len_utf8()..].trim();
                return Some((left, right));
            }
            _ => {}
        }
    }
    None
}

/// `s` starts with `(`; returns `(full_balanced_group, bytes_after_group)`.
fn take_balanced_paren_group(s: &str) -> Option<(&str, &str)> {
    let t = s.trim();
    if !t.starts_with('(') {
        return None;
    }
    let mut depth = 0i32;
    for (i, ch) in t.char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    let end = i + ch.len_utf8();
                    return Some((&t[..end], &t[end..]));
                }
            }
            _ => {}
        }
    }
    None
}

fn peel_negation_and_outer_parens(mut s: &str) -> (&str, bool) {
    let mut neg = false;
    loop {
        s = strip_outer_parens(s.trim());
        let t = s.trim();
        if let Some(rest) = t.strip_prefix('-') {
            neg = !neg;
            s = rest.trim();
            continue;
        }
        break;
    }
    (s, neg)
}

/// Clears a single outer fraction `lhs/den * tail` when `den` is a parenthesized group and `tail`
/// continues the numerator (implicit or explicit `*`). Otherwise behaves like numerator-only of
/// `lhs/den` when `tail` is empty.
fn rhs_numerator_expression(rhs: &str) -> String {
    let (t, neg) = peel_negation_and_outer_parens(rhs);
    let Some((left, right)) = split_fraction_depth_zero(t) else {
        let mut out = strip_outer_parens(t.trim()).trim().to_string();
        if neg {
            out = format!("-({out})");
        }
        return out;
    };
    let right = right.trim();
    let Some((_den, rest)) = take_balanced_paren_group(right) else {
        let mut out = strip_outer_parens(left.trim()).trim().to_string();
        if neg {
            out = format!("-({out})");
        }
        return out;
    };
    let rest = rest.trim();
    if rest.is_empty() {
        let mut out = strip_outer_parens(left.trim()).trim().to_string();
        if neg {
            out = format!("-({out})");
        }
        return out;
    }
    let tail = rest
        .strip_prefix('*')
        .map(str::trim)
        .unwrap_or(rest);
    if tail.is_empty() {
        let mut out = strip_outer_parens(left.trim()).trim().to_string();
        if neg {
            out = format!("-({out})");
        }
        return out;
    }
    let left_t = left.trim();
    let tail_c = collapse_ws(tail);
    let tail_c = apply_implicit_multiplication(&tail_c);
    let mut out = format!("({left_t})*({tail_c})");
    if neg {
        out = format!("-({out})");
    }
    out
}

/// Numerator of `rhs` for building `lhs - num = 0` (denominator dropped).
fn rhs_numerator(rhs: &str) -> String {
    rhs_numerator_expression(rhs)
}

static IDENT_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b").expect("ident regex"));

fn collect_identifiers(expr: &str) -> Vec<String> {
    let mut out: Vec<String> = IDENT_RE
        .captures_iter(expr)
        .filter_map(|c| c.get(1).map(|m| m.as_str().to_string()))
        .collect();
    out.sort();
    out.dedup();
    out
}

fn build_var_map(names: &[String]) -> HashMap<String, VarId> {
    names
        .iter()
        .enumerate()
        .map(|(i, n)| (n.clone(), VarId(i as u32)))
        .collect()
}

fn equation_strings_from_mathematica(text: &str) -> Result<Vec<String>, MathematicaLoadError> {
    let cleaned = strip_outer_braces(text);
    if cleaned.is_empty() {
        return Err(MathematicaLoadError::Empty);
    }
    let mut equations = Vec::new();
    for segment in split_top_level(cleaned) {
        let Some((lhs, rhs)) = split_rule(&segment) else {
            let t = segment.trim();
            if t.is_empty() {
                continue;
            }
            return Err(MathematicaLoadError::NoRuleSeparator(segment));
        };
        let lhs_c = collapse_ws(lhs);
        let lhs_c = apply_implicit_multiplication(&lhs_c);
        let rhs_c = collapse_ws(rhs);
        let rhs_c = apply_implicit_multiplication(&rhs_c);
        let num = rhs_numerator(&rhs_c);
        let num = apply_implicit_multiplication(&num);
        // Avoid `((kappah))-...`: lhs must be a primary expression, not wrapped in extra parens.
        let eq = format!("{lhs_c}-({num})");
        equations.push(eq);
    }
    if equations.is_empty() {
        return Err(MathematicaLoadError::Empty);
    }
    Ok(equations)
}

/// Parse a Mathematica list of rules into a polynomial system (`equation = 0` for each rule).
pub fn load_mathematica_system_from_str(text: &str) -> Result<PolynomialSystem, MathematicaLoadError> {
    let equations = equation_strings_from_mathematica(text)?;
    let mut all_names: Vec<String> = Vec::new();
    for eq in &equations {
        all_names.extend(collect_identifiers(eq));
    }
    all_names.sort();
    all_names.dedup();
    let var_map = build_var_map(&all_names);
    let mut b = DagBuilder::new();
    let mut roots = Vec::with_capacity(equations.len());
    for eq in equations {
        let (dag, root) = Dag::from_rational_string_with_var_map(&eq, &var_map)?;
        roots.push(b.merge_root(&dag, root));
    }
    Ok(PolynomialSystem::new(b.build(), roots))
}

/// Read UTF-8 from `path` and [`load_mathematica_system_from_str`].
pub fn load_mathematica_system_from_path(path: &Path) -> Result<PolynomialSystem, MathematicaLoadError> {
    let text = std::fs::read_to_string(path)?;
    load_mathematica_system_from_str(&text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_level_numerator_takes_lhs_of_fraction() {
        let rhs = "(a + b)/(c + d)";
        assert_eq!(rhs_numerator(rhs), "a + b");
    }

    #[test]
    fn strip_parens_before_slash() {
        let rhs = "((x + 1))/(y)";
        assert_eq!(rhs_numerator(rhs), "x + 1");
    }

    #[test]
    fn fraction_then_tail_folds_into_numerator() {
        let rhs = "-1/(512*x)*3*(a+b)";
        let n = rhs_numerator(rhs);
        assert!(n.contains("3"), "{n}");
        assert!(n.contains("a+b") || n.contains("a + b"), "{n}");
    }
}
