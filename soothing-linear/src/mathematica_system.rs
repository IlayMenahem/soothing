//! Load Mathematica-style rule lists into [`PolynomialSystem`].
//!
//! Each rule `lhs -> rhs` (or `==`, `:>`) becomes `lhs - (rhs) = 0` after whitespace collapse,
//! implicit multiplication, `\[Pi]` → `pi` (DAG constant, pi ≈ 355/113; not a [`VarId`]), and
//! wrapping `integer/(` as `(integer)/(` so rational literals do not swallow parenthesized denominators.
//! The full rational structure of `rhs` is preserved (no numerator-only rewriting).

use crate::dag::{Dag, DagBuilder, DagParseError, NodeId, VarId};
use crate::system::PolynomialSystem;
use regex::Regex;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;
use thiserror::Error;

/// Repo `systems/` directory when built from `crates/soothing-linear` (same layout as Python `systems.py`).
#[must_use] 
pub fn default_systems_dir_from_manifest() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../systems")
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

/// Mathematica `FullForm`-style `\[Pi]` → parseable `pi` (lowers to `ConstPi` in the DAG).
fn normalize_mathematica_symbols(s: &str) -> String {
    s.replace(r"\[Pi]", "pi")
}

static IMPLICIT_MUL_PATTERNS: LazyLock<Vec<(Regex, &'static str)>> = LazyLock::new(|| {
    let patterns: &[(&str, &str)] = &[
        // `pi` is two letters; run before `(\d)\s+([a-zA-Z])` which would break `256 pi`.
        (r"(\d)\s+pi\b", r"${1}*pi"),
        (r"\)\s+pi\b", r")*pi"),
        (r"([a-zA-Z_][a-zA-Z0-9_]*)\s+pi\b", r"${1}*pi"),
        (r"\)\s+([a-zA-Z])", r")*$1"),
        (r"\)\s+(\d)", r")*$1"),
        (r"([a-zA-Z_][a-zA-Z0-9_]*)\s+\(", r"$1*("),
        (r"(\d)\s+\(", r"$1*("),
        // Full identifier after identifier or digit (e.g. `DpDph1 e`, `2 xr`).
        (
            r"([a-zA-Z_][a-zA-Z0-9_]*)\s+([a-zA-Z_][a-zA-Z0-9_]*)",
            r"$1*$2",
        ),
        (r"(\d)\s+([a-zA-Z_][a-zA-Z0-9_]*)", r"${1}*$2"),
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
        .map(|(p, r)| (Regex::new(p).expect("implicit mul regex"), *r))
        .collect()
});

fn apply_implicit_multiplication(s: &str) -> String {
    IMPLICIT_MUL_PATTERNS
        .iter()
        .fold(s.to_string(), |acc, (re, repl)| {
            re.replace_all(&acc, *repl).to_string()
        })
}

/// Integer `/` `(` must become `(integer)/(` or the rational literal grammar treats `1/` as part
/// of a `Q` literal and rejects a parenthesized denominator (e.g. `1/(512*pi^2)`).
fn wrap_int_numerator_paren_denominator(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len() + 16);
    let mut i = 0usize;
    while i < chars.len() {
        if chars[i].is_ascii_digit() {
            let start = i;
            let mut j = i;
            while j < chars.len() && chars[j].is_ascii_digit() {
                j += 1;
            }
            if j < chars.len() && chars[j] == '/' && j + 1 < chars.len() && chars[j + 1] == '(' {
                out.push('(');
                out.extend(chars[start..j].iter());
                out.push(')');
                out.push('/');
                i = j + 1;
                let mut depth = 0i32;
                while i < chars.len() {
                    let c = chars[i];
                    if c == '(' {
                        depth += 1;
                    } else if c == ')' {
                        depth -= 1;
                    }
                    out.push(c);
                    i += 1;
                    if depth == 0 {
                        break;
                    }
                }
                continue;
            }
        }
        out.push(chars[i]);
        i += 1;
    }
    out
}

static IDENT_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b").expect("ident regex"));

fn collect_identifiers(expr: &str) -> Vec<String> {
    let mut out: Vec<String> = IDENT_RE
        .captures_iter(expr)
        .filter_map(|c| c.get(1).map(|m| m.as_str().to_string()))
        .filter(|n| n != "pi")
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
    let equations =
        split_top_level(cleaned)
            .into_iter()
            .try_fold(Vec::new(), |mut equations, segment| {
                let Some((lhs, rhs)) = split_rule(&segment) else {
                    let t = segment.trim();
                    if t.is_empty() {
                        return Ok(equations);
                    }
                    return Err(MathematicaLoadError::NoRuleSeparator(segment));
                };
                let lhs_c = normalize_mathematica_symbols(lhs);
                let lhs_c = collapse_ws(&lhs_c);
                let lhs_c = apply_implicit_multiplication(&lhs_c);
                let lhs_c = wrap_int_numerator_paren_denominator(&lhs_c);
                let rhs_c = normalize_mathematica_symbols(rhs);
                let rhs_c = collapse_ws(&rhs_c);
                let rhs_c = apply_implicit_multiplication(&rhs_c);
                let rhs_c = wrap_int_numerator_paren_denominator(&rhs_c);
                // Parenthesize rhs only; wrapping lhs as `(kappah)` is parsed as a variable named "(kappah)".
                let eq = format!("{lhs_c}-({rhs_c})");
                equations.push(eq);
                Ok(equations)
            })?;
    if equations.is_empty() {
        return Err(MathematicaLoadError::Empty);
    }
    Ok(equations)
}

/// Result of loading a Mathematica rule list: system plus display names for [`VarId`] indices.
///
/// `var_names[i]` is the name for `VarId(i as u32)` (sorted, deduped identifier order).
#[derive(Clone, Debug)]
pub struct LoadedMathematicaSystem {
    pub system: PolynomialSystem,
    pub var_names: Vec<String>,
}

/// Parse a Mathematica list of rules into a polynomial system and variable names (`equation = 0` per rule).
pub fn load_mathematica_system_from_str_with_var_names(
    text: &str,
) -> Result<LoadedMathematicaSystem, MathematicaLoadError> {
    let equations = equation_strings_from_mathematica(text)?;
    let mut all_names: Vec<String> = equations
        .iter()
        .flat_map(|eq| collect_identifiers(eq))
        .collect();
    all_names.sort();
    all_names.dedup();
    let var_map = build_var_map(&all_names);
    let mut b = DagBuilder::new();
    let roots = equations.into_iter().try_fold(
        Vec::new(),
        |mut roots, eq| -> Result<Vec<NodeId>, MathematicaLoadError> {
            let (dag, root) = Dag::from_rational_string_with_var_map(&eq, &var_map)?;
            roots.push(b.merge_root(&dag, root));
            Ok(roots)
        },
    )?;
    Ok(LoadedMathematicaSystem {
        system: PolynomialSystem::new(b.build(), roots),
        var_names: all_names,
    })
}

/// Parse a Mathematica list of rules into a polynomial system (`equation = 0` for each rule).
pub fn load_mathematica_system_from_str(
    text: &str,
) -> Result<PolynomialSystem, MathematicaLoadError> {
    Ok(load_mathematica_system_from_str_with_var_names(text)?.system)
}

/// Read UTF-8 from `path` and [`load_mathematica_system_from_str_with_var_names`].
pub fn load_mathematica_system_from_path_with_var_names(
    path: &Path,
) -> Result<LoadedMathematicaSystem, MathematicaLoadError> {
    let text = std::fs::read_to_string(path)?;
    load_mathematica_system_from_str_with_var_names(&text)
}

/// Read UTF-8 from `path` and [`load_mathematica_system_from_str`].
pub fn load_mathematica_system_from_path(
    path: &Path,
) -> Result<PolynomialSystem, MathematicaLoadError> {
    let text = std::fs::read_to_string(path)?;
    load_mathematica_system_from_str(&text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equation_is_lhs_minus_rhs_with_fraction() {
        let s = "{0 -> (a + b)/(c + d)}";
        let eqs = equation_strings_from_mathematica(s).unwrap();
        assert_eq!(eqs.len(), 1);
        let e = &eqs[0];
        assert!(e.starts_with("0-("), "{e}");
        assert!(
            e.contains("a + b") || e.contains("a+b"),
            "expected numerator a+b in {e}"
        );
        assert!(
            e.contains("c + d") || e.contains("c+d"),
            "expected denominator c+d in {e}"
        );
    }

    #[test]
    fn pi_normalizes_in_equation_string() {
        let s = "{0 -> 1/(256*pi^2)}";
        let eqs = equation_strings_from_mathematica(s).unwrap();
        assert_eq!(eqs[0], "0-((1)/(256*pi^2))");
    }

    #[test]
    fn mathematica_pi_escape_normalizes() {
        let s = r"{0 -> 1/(256 \[Pi]^2)}";
        let eqs = equation_strings_from_mathematica(s).unwrap();
        assert_eq!(eqs[0], "0-((1)/(256*pi^2))");
    }
}
