use std::collections::HashMap;

use soothing_linear::{evaluate, Dag, DagParseError, Q, VarId};

#[test]
fn parse_roundtrip_format() {
    let s = "v1 * v0 + 1 / 2";
    let (dag, root) = Dag::from_rational_string(s).unwrap();
    let name = |v: VarId| format!("v{}", v.0);
    let out = dag.format_root(root, &name);
    assert_eq!(out, "v1 * v0 + 1 / 2");
}

#[test]
fn parse_rational_div_chain() {
    let (dag, root) = Dag::from_rational_string("1 / 2 / 3").unwrap();
    let assign = HashMap::new();
    let q = evaluate(&dag, root, &assign).unwrap();
    assert_eq!(q, Q::new(1, 6));
}

/// `arithmetic-parser` chains `^` with left associativity: `2^3^2` is `(2^3)^2`, not `2^(3^2)`.
#[test]
fn parse_pow_left_assoc() {
    let (dag, root) = Dag::from_rational_string("2 ^ 3 ^ 2").unwrap();
    let assign = HashMap::new();
    let q = evaluate(&dag, root, &assign).unwrap();
    assert_eq!(q, Q::from_integer(64));
}

#[test]
fn parse_named_vars_and_eval() {
    let (dag, root) = Dag::from_rational_string("(v1 + v2) * v0").unwrap();
    let mut assign = HashMap::new();
    assign.insert(VarId(0), Q::new(2, 1));
    assign.insert(VarId(1), Q::new(3, 1));
    assign.insert(VarId(2), Q::new(5, 1));
    let q = evaluate(&dag, root, &assign).unwrap();
    assert_eq!(q, Q::new(16, 1));
}

#[test]
fn parse_rejects_trailing_garbage() {
    let e = Dag::from_rational_string("1 + 2 )").unwrap_err();
    assert!(matches!(e, DagParseError::TrailingInput { .. }));
}

#[test]
fn parse_unary_minus_and_precedence() {
    let (dag, root) = Dag::from_rational_string("-v0 * v1 + 2").unwrap();
    let mut assign = HashMap::new();
    assign.insert(VarId(0), Q::new(3, 1));
    assign.insert(VarId(1), Q::new(2, 1));
    let q = evaluate(&dag, root, &assign).unwrap();
    assert_eq!(q, Q::new(-4, 1));
}

#[test]
fn parse_trailing_garbage_after_valid_prefix() {
    let e = Dag::from_rational_string("1 @ 2").unwrap_err();
    assert!(matches!(e, DagParseError::TrailingInput { .. }));
}

#[test]
fn parse_unexpected_char_at_start() {
    let e = Dag::from_rational_string("?").unwrap_err();
    assert!(matches!(
        e,
        DagParseError::UnexpectedChar { ch: '?', pos: 0 }
            | DagParseError::TrailingInput { pos: 0 }
    ));
}

/// `foo_bar` must parse as one identifier (underscore inside name), not as two tokens.
#[test]
fn parse_named_identifier_with_underscore() {
    let (dag, root) = Dag::from_rational_string("foo_bar + 1").unwrap();
    let name = |v: VarId| format!("v{}", v.0);
    let out = dag.format_root(root, name);
    assert_eq!(out, "v0 + 1");
}
