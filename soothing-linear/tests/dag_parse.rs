#![allow(clippy::many_single_char_names)]

use std::collections::HashMap;

use soothing_linear::{evaluate, format_root, from_rational_string, q, DagParseError, VarId, Q};

#[test]
fn parse_roundtrip_format() {
    let s = "v1 * v0 + 1 / 2";
    let (dag, root) = from_rational_string(s).unwrap();
    let name = |v: VarId| format!("v{}", v.0);
    let out = format_root(&dag, root, name);
    assert_eq!(out, "v1 * v0 + 1 / 2");
}

#[test]
fn format_neg_of_negative_literal_is_single_number() {
    let (dag, root) = from_rational_string("--3").unwrap();
    let out = format_root(&dag, root, |v: VarId| format!("v{}", v.0));
    assert_eq!(out, "3");
}

#[test]
fn parse_rational_div_chain() {
    let (dag, root) = from_rational_string("1 / 2 / 3").unwrap();
    let assign = HashMap::new();
    let got = evaluate(&dag, root, &assign).unwrap();
    assert_eq!(got, q(1, 6));
}

#[test]
fn parse_pow_left_assoc() {
    let (dag, root) = from_rational_string("2 ^ 3 ^ 2").unwrap();
    let assign = HashMap::new();
    let got = evaluate(&dag, root, &assign).unwrap();
    assert_eq!(got, Q::from_integer(64.into()));
}

#[test]
fn parse_named_vars_and_eval() {
    let (dag, root) = from_rational_string("(v1 + v2) * v0").unwrap();
    let mut assign = HashMap::new();
    assign.insert(VarId(0), q(2, 1));
    assign.insert(VarId(1), q(3, 1));
    assign.insert(VarId(2), q(5, 1));
    let got = evaluate(&dag, root, &assign).unwrap();
    assert_eq!(got, q(16, 1));
}

#[test]
fn parse_rejects_trailing_garbage() {
    let e = from_rational_string("1 + 2 )").unwrap_err();
    assert!(matches!(e, DagParseError::TrailingInput { .. }));
}

#[test]
fn parse_zero_minus_neg_one() {
    from_rational_string("(0)-(-1)").expect("(0)-(-1)");
}

#[test]
fn parse_unary_minus_and_precedence() {
    let (dag, root) = from_rational_string("-v0 * v1 + 2").unwrap();
    let mut assign = HashMap::new();
    assign.insert(VarId(0), q(3, 1));
    assign.insert(VarId(1), q(2, 1));
    let got = evaluate(&dag, root, &assign).unwrap();
    assert_eq!(got, q(-4, 1));
}

#[test]
fn parse_trailing_garbage_after_valid_prefix() {
    let e = from_rational_string("1 @ 2").unwrap_err();
    assert!(matches!(e, DagParseError::TrailingInput { .. }));
}

#[test]
fn parse_unexpected_char_at_start() {
    let e = from_rational_string("?").unwrap_err();
    assert!(matches!(
        e,
        DagParseError::UnexpectedChar { ch: '?', pos: 0 } | DagParseError::TrailingInput { pos: 0 }
    ));
}

#[test]
fn parse_named_identifier_with_underscore() {
    let (dag, root) = from_rational_string("foo_bar + 1").unwrap();
    let name = |v: VarId| format!("v{}", v.0);
    let out = format_root(&dag, root, name);
    assert_eq!(out, "v0 + 1");
}
