use std::collections::HashMap;

use soothing_linear::{evaluate, Dag, Q, VarId};

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

#[test]
fn parse_pow_right_assoc() {
    let (dag, root) = Dag::from_rational_string("2 ^ 3 ^ 2").unwrap();
    let assign = HashMap::new();
    let q = evaluate(&dag, root, &assign).unwrap();
    assert_eq!(q, Q::from_integer(512));
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
    assert!(matches!(e, soothing_linear::DagParseError::TrailingInput { .. }));
}
