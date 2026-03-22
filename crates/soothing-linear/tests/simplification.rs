use soothing_linear::{evaluate, simplify_once, DagBuilder, Q, VarId};
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
