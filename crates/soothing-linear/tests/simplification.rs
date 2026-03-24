use soothing_linear::{
    evaluate, simplify_once, Dag, DagBuilder, SimplifyError, VarId, Q,
};
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
    let r = simplify_once(&mut bb, &d, m).unwrap();
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
    let r = simplify_once(&mut bb, &d, sum).unwrap();
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
    let r = simplify_once(&mut bb, &d, e).unwrap();
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
    let r = simplify_once(&mut bb, &d, root).unwrap();
    let out = bb.snapshot();
    for &val in &[Q::new(0, 1), Q::new(5, 2), Q::new(-3, 7)] {
        let mut m = HashMap::new();
        m.insert(x, val);
        let expected = Q::new(1, 1) - val;
        assert_eq!(evaluate(&out, r, &m).unwrap(), expected);
    }
}

#[test]
fn add_cancels_x_and_neg_x() {
    let mut b = DagBuilder::new();
    let x = VarId(0);
    let v = b.variable(x);
    let nv = b.neg(v);
    let sum = b.add(vec![v, nv]);
    let d = b.build();
    let mut bb = DagBuilder::new();
    let r = simplify_once(&mut bb, &d, sum).unwrap();
    let out = bb.snapshot();
    assert_eq!(
        evaluate(&out, r, &HashMap::new()).unwrap(),
        Q::new(0, 1)
    );
}

#[test]
fn flatten_nested_add_strips_inner_zero() {
    let mut b = DagBuilder::new();
    let x = VarId(0);
    let y = VarId(1);
    let vx = b.variable(x);
    let vy = b.variable(y);
    let inner = b.add(vec![vx, vy]);
    let z = b.constant(Q::new(0, 1));
    let root = b.add(vec![inner, z]);
    let d = b.build();
    let mut bb = DagBuilder::new();
    let r = simplify_once(&mut bb, &d, root).unwrap();
    let out = bb.snapshot();
    let mut m = HashMap::new();
    m.insert(x, Q::new(1, 1));
    m.insert(y, Q::new(2, 1));
    assert_eq!(evaluate(&out, r, &m).unwrap(), Q::new(3, 1));
}

#[test]
fn flatten_nested_mul() {
    let mut b = DagBuilder::new();
    let x = VarId(0);
    let vx = b.variable(x);
    let two = b.constant(Q::new(2, 1));
    let inner = b.mul(vec![vx, two]);
    let one = b.constant(Q::new(1, 1));
    let root = b.mul(vec![one, inner]);
    let d = b.build();
    let mut bb = DagBuilder::new();
    let r = simplify_once(&mut bb, &d, root).unwrap();
    let out = bb.snapshot();
    let mut m = HashMap::new();
    m.insert(x, Q::new(3, 1));
    assert_eq!(evaluate(&out, r, &m).unwrap(), Q::new(6, 1));
}

#[test]
fn product_of_fractions_constant() {
    let (d, root) = Dag::from_rational_string("1/2 * 1/3").unwrap();
    let mut bb = DagBuilder::new();
    let r = simplify_once(&mut bb, &d, root).unwrap();
    let out = bb.snapshot();
    assert_eq!(
        evaluate(&out, r, &HashMap::new()).unwrap(),
        Q::new(1, 6)
    );
}

#[test]
fn nested_division_of_fractions() {
    let (d, root) = Dag::from_rational_string("(1/2) / (3/4)").unwrap();
    let mut bb = DagBuilder::new();
    let r = simplify_once(&mut bb, &d, root).unwrap();
    let out = bb.snapshot();
    assert_eq!(
        evaluate(&out, r, &HashMap::new()).unwrap(),
        Q::new(2, 3)
    );
}

#[test]
fn pow_x_to_one_strips() {
    let mut b = DagBuilder::new();
    let x = VarId(0);
    let vx = b.variable(x);
    let one = b.constant(Q::new(1, 1));
    let root = b.pow(vx, one);
    let d = b.build();
    let mut bb = DagBuilder::new();
    let r = simplify_once(&mut bb, &d, root).unwrap();
    let out = bb.snapshot();
    let mut m = HashMap::new();
    m.insert(x, Q::new(5, 1));
    assert_eq!(evaluate(&out, r, &m).unwrap(), Q::new(5, 1));
}

#[test]
fn pow_one_to_n_is_one() {
    let mut b = DagBuilder::new();
    let one = b.constant(Q::new(1, 1));
    let exp = b.constant(Q::new(100, 1));
    let root = b.pow(one, exp);
    let d = b.build();
    let mut bb = DagBuilder::new();
    let r = simplify_once(&mut bb, &d, root).unwrap();
    let out = bb.snapshot();
    assert_eq!(
        evaluate(&out, r, &HashMap::new()).unwrap(),
        Q::new(1, 1)
    );
}

#[test]
fn const_division_by_zero_errors() {
    let mut b = DagBuilder::new();
    let one = b.constant(Q::new(1, 1));
    let zero = b.constant(Q::new(0, 1));
    let bad = b.div(one, zero);
    let d = b.build();
    let mut bb = DagBuilder::new();
    assert_eq!(
        simplify_once(&mut bb, &d, bad),
        Err(SimplifyError::DivisionByZero)
    );
}

#[test]
fn const_pow_noninteger_exponent_errors() {
    let mut b = DagBuilder::new();
    let two = b.constant(Q::new(2, 1));
    let half = b.constant(Q::new(1, 2));
    let root = b.pow(two, half);
    let d = b.build();
    let mut bb = DagBuilder::new();
    assert_eq!(
        simplify_once(&mut bb, &d, root),
        Err(SimplifyError::NonIntegerExponent)
    );
}
