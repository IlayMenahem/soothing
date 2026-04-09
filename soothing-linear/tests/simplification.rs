#![allow(clippy::many_single_char_names)]

use soothing_linear::{q, simplify, Dag, DagBuilder, VarId};

#[test]
fn flatten_nested_mul() {
    let (d, root) = Dag::from_rational_string("x * (y * (z * 3))").unwrap();
    let mut bb = DagBuilder::new();
    let r = simplify(&mut bb, &d, root).unwrap();
    let out = bb.snapshot();
    assert_eq!(
        out.format_root(r, |v| format!("v{}", v.0)),
        "3 * v0 * v1 * v2"
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
        "3 + v0 + v1 + v2"
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

#[test]
fn fold_add_consts() {
    let (d, root) = Dag::from_rational_string("1 + 2 + 3").unwrap();
    let mut bb = DagBuilder::new();
    let r = simplify(&mut bb, &d, root).unwrap();
    let out = bb.snapshot();
    assert_eq!(out.format_root(r, |v| format!("v{}", v.0)), "6");
}

#[test]
fn fold_mul_consts() {
    let (d, root) = Dag::from_rational_string("2 * 3 * 4").unwrap();
    let mut bb = DagBuilder::new();
    let r = simplify(&mut bb, &d, root).unwrap();
    let out = bb.snapshot();
    assert_eq!(out.format_root(r, |v| format!("v{}", v.0)), "24");
}

#[test]
fn fold_mul_and_add_consts() {
    let (d, root) = Dag::from_rational_string("-(1 + 3 * 2)/4").unwrap();
    let mut bb = DagBuilder::new();
    let r = simplify(&mut bb, &d, root).unwrap();
    let out = bb.snapshot();
    assert_eq!(out.format_root(r, |v| format!("v{}", v.0)), "-7/4");
}

#[test]
fn group_partial_add() {
    let (d, root) = Dag::from_rational_string("1 + 2 + x").unwrap();
    let mut bb = DagBuilder::new();
    let r = simplify(&mut bb, &d, root).unwrap();
    let out = bb.snapshot();
    assert_eq!(out.format_root(r, |v| format!("v{}", v.0)), "3 + v0");
}

#[test]
fn fold_neg_const() {
    let (d, root) = Dag::from_rational_string("-5").unwrap();
    let mut bb = DagBuilder::new();
    let r = simplify(&mut bb, &d, root).unwrap();
    let out = bb.snapshot();
    assert_eq!(out.format_root(r, |v| format!("v{}", v.0)), "-5");
}

#[test]
fn fold_div_consts() {
    let (d, root) = Dag::from_rational_string("6 / 2").unwrap();
    let mut bb = DagBuilder::new();
    let r = simplify(&mut bb, &d, root).unwrap();
    let out = bb.snapshot();
    assert_eq!(out.format_root(r, |v| format!("v{}", v.0)), "3");
}

#[test]
fn fold_pow_consts() {
    let (d, root) = Dag::from_rational_string("2 ^ 3").unwrap();
    let mut bb = DagBuilder::new();
    let r = simplify(&mut bb, &d, root).unwrap();
    let out = bb.snapshot();
    assert_eq!(out.format_root(r, |v| format!("v{}", v.0)), "8");
}

#[test]
fn neg_of_a_negative() {
    let mut input = DagBuilder::new();
    let neg_one = input.constant(q(-1, 1));
    let root = input.neg(neg_one);
    let d = input.snapshot();
    let mut bb = DagBuilder::new();
    let r = simplify(&mut bb, &d, root).unwrap();
    let out = bb.snapshot();
    assert_eq!(out.format_root(r, |v| format!("v{}", v.0)), "1");
}

#[test]
fn neg_absorbs_into_mul_leading_const() {
    let (d, root) = Dag::from_rational_string("-(-3 * v0)").unwrap();
    let mut bb = DagBuilder::new();
    let r = simplify(&mut bb, &d, root).unwrap();
    let out = bb.snapshot();
    assert_eq!(out.format_root(r, |v| format!("v{}", v.0)), "3 * v0");
}

#[test]
fn neg_absorbs_when_mul_has_div_chain() {
    let (d, root) = Dag::from_rational_string("-(-3 * v0 * v1 / 256)").unwrap();
    let mut bb = DagBuilder::new();
    let r = simplify(&mut bb, &d, root).unwrap();
    let out = bb.snapshot().format_root(r, |v| format!("v{}", v.0));
    assert!(
        !out.contains("-(-3"),
        "expected unary minus folded into product, got {out}"
    );
}

#[test]
fn triple_neg_over_negative_mul_is_single_neg_product() {
    let (d, root) = Dag::from_rational_string("-(-(-3 * v0))").unwrap();
    let mut bb = DagBuilder::new();
    let r = simplify(&mut bb, &d, root).unwrap();
    let out = bb.snapshot();
    let s = out.format_root(r, |v| format!("v{}", v.0));
    assert_eq!(
        s, "-3 * v0",
        "expected -(-(-3*v)) to simplify to -3*v, got {s}"
    );
}

/// Three unary minuses around `3 * v` (as in some generated DAGs) → `-3 * v`.
#[test]
fn builder_three_negs_around_mul_plus_three() {
    let mut b = DagBuilder::new();
    let v = b.variable(VarId(0));
    let c = b.constant(q(3, 1));
    let m = b.mul(vec![c, v]);
    let n1 = b.neg(m);
    let n2 = b.neg(n1);
    let root = b.neg(n2);
    let d = b.snapshot();
    let mut bb = DagBuilder::new();
    let r = simplify(&mut bb, &d, root).unwrap();
    let s = bb.snapshot().format_root(r, |x| format!("v{}", x.0));
    assert_eq!(s, "-3 * v0", "Neg^3(Mul(3,v)) should be -3*v, got {s}");
}

#[test]
fn nested_div_left_assoc_normalizes() {
    let (d, root) = Dag::from_rational_string("v0 / v1 / v2").unwrap();
    let mut bb = DagBuilder::new();
    let r = simplify(&mut bb, &d, root).unwrap();
    let out = bb.snapshot().format_root(r, |v| format!("v{}", v.0));
    assert_eq!(out, "v0 / (v1 * v2)");
}

#[test]
fn nested_div_in_denominator_normalizes() {
    let (d, root) = Dag::from_rational_string("v0 / (v1 / v2)").unwrap();
    let mut bb = DagBuilder::new();
    let r = simplify(&mut bb, &d, root).unwrap();
    let out = bb.snapshot().format_root(r, |v| format!("v{}", v.0));
    assert_eq!(out, "v0 * v2 / v1");
}

#[test]
fn nested_div_with_compound_subexpressions() {
    let (d, root) = Dag::from_rational_string("(v0 * v1) / v2 / v3").unwrap();
    let mut bb = DagBuilder::new();
    let r = simplify(&mut bb, &d, root).unwrap();
    let out = bb.snapshot().format_root(r, |v| format!("v{}", v.0));
    assert_eq!(out, "v0 * v1 / (v2 * v3)");
}

#[test]
fn nested_expressions() {
    let (d, root) = Dag::from_rational_string("(v0 / (v1 * v2)) / v3").unwrap();
    let mut bb = DagBuilder::new();
    let r = simplify(&mut bb, &d, root).unwrap();
    let out = bb.snapshot().format_root(r, |v| format!("v{}", v.0));
    assert_eq!(out, "v0 / (v1 * v2 * v3)");
}

#[test]
fn xr_simplify() {
    let (d, root) = Dag::from_rational_string("((3 * DpDph1 * e) / (256 * (pi ^ 2))) / (-1 * DpDph1 * e / (256 * (pi ^ 2)) + (-3 * DpDph1 * e / (256 * (pi ^ 2))))").unwrap();
    let mut bb = DagBuilder::new();
    let r = simplify(&mut bb, &d, root).unwrap();
    let out = bb.snapshot().format_root(r, |v| format!("v{}", v.0));
    assert_eq!(
        out,
        "3 * v0 * v1 / (256 * (pi ^ 2) * (-1 * v0 * v1 / (256 * (pi ^ 2)) + -3 * v0 * v1 / (256 * (pi ^ 2))))"
    );
}
