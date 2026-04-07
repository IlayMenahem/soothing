#![allow(clippy::many_single_char_names)]

use soothing_linear::{simplify, Dag, DagBuilder};

#[test]
fn flatten_nested_mul() {
    let (d, root) = Dag::from_rational_string("x * (y * (z * 3))").unwrap();
    let mut bb = DagBuilder::new();
    let r = simplify(&mut bb, &d, root).unwrap();
    let out = bb.snapshot();
    assert_eq!(
        out.format_root(r, |v| format!("v{}", v.0)),
        "v0 * v1 * v2 * 3"
    );
}

#[test]
fn flatten_nested_add() {
    let (d, root) = Dag::from_rational_string("x + (y + (z + 3))").unwrap();
    let mut bb = DagBuilder::new();
    let r = simplify(&mut bb, &d, root).unwrap();
    let out = bb.snapshot();
    assert_eq!(
        out.format_root(r, |v| format!("v{}", v.0)),
        "v0 + v1 + v2 + 3"
    );
}

#[test]
fn simplify_neg_of_neg() {
    let (d, root) = Dag::from_rational_string("-(-x)").unwrap();
    let mut bb = DagBuilder::new();
    let r = simplify(&mut bb, &d, root).unwrap();
    let out = bb.snapshot();
    assert_eq!(out.format_root(r, |v| format!("v{}", v.0)), "v0");
}

#[test]
fn simplify_triple_neg() {
    let (d, root) = Dag::from_rational_string("-(-(-x))").unwrap();
    let mut bb = DagBuilder::new();
    let r = simplify(&mut bb, &d, root).unwrap();
    let out = bb.snapshot();
    assert_eq!(out.format_root(r, |v| format!("v{}", v.0)), "-v0");
}
