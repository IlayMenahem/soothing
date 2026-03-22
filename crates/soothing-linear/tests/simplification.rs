use soothing_linear::{evaluate, simplify_once, Dag, DagBuilder, VarId, Q};
use std::collections::HashMap;

#[test]
fn mul_by_zero_annihilates() {
    let mut b = DagBuilder::new();
    let x = VarId(0);
    let v = b.variable(x);
    let z = b.constant(Q::new(0, 1));
    let m = b.mul(vec![z, v]);
    let d = b.build();
    let mut bb = DagBuilder::new();
    let r = simplify_once(&mut bb, &d, m);
    let out = bb.snapshot();
    let empty = HashMap::new();
    assert_eq!(evaluate(&out, r, &empty).unwrap(), Q::new(0, 1));
}

#[test]
fn strip_unit_factors_and_zero_summands() {
    let mut b = DagBuilder::new();
    let x = VarId(0);
    let v = b.variable(x);
    let one = b.constant(Q::new(1, 1));
    let zero = b.constant(Q::new(0, 1));
    let prod = b.mul(vec![one, v, one]);
    let sum = b.add(vec![prod, zero]);
    let d = b.build();
    let mut bb = DagBuilder::new();
    let r = simplify_once(&mut bb, &d, sum);
    let out = bb.snapshot();
    let mut m = HashMap::new();
    m.insert(x, Q::new(7, 3));
    assert_eq!(evaluate(&out, r, &m).unwrap(), Q::new(7, 3));
}

#[test]
fn double_neg_and_add_zero() {
    let mut b = DagBuilder::new();
    let x = VarId(0);
    let v = b.variable(x);
    let n1 = b.neg(v);
    let e = b.neg(n1);
    let d = b.build();
    let mut bb = DagBuilder::new();
    let r = simplify_once(&mut bb, &d, e);
    let out = bb.snapshot();
    let mut m = HashMap::new();
    m.insert(x, Q::new(3, 2));
    assert_eq!(evaluate(&out, r, &m).unwrap(), Q::new(3, 2));
}

#[test]
fn affine_div_simplifies_to_one_minus_x() {
    let (d, root) = Dag::from_rational_string("(x + -1) / (x + -1 + -(x + 1 + -1))").unwrap();
    let x = VarId(0);
    let mut bb = DagBuilder::new();
    let r = simplify_once(&mut bb, &d, root);
    let out = bb.snapshot();
    for &val in &[Q::new(0, 1), Q::new(5, 2), Q::new(-3, 7)] {
        let mut m = HashMap::new();
        m.insert(x, val);
        let expected = Q::new(1, 1) - val;
        assert_eq!(evaluate(&out, r, &m).unwrap(), expected);
    }
}
