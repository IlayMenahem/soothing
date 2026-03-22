use soothing_linear::{evaluate, Dag, DagBuilder, EvalError, VarId, Q};
use std::collections::HashMap;

#[test]
fn constant_and_variable() {
    let mut b = DagBuilder::new();
    let x = VarId(0);
    let c = b.constant(Q::new(5, 1));
    let v = b.variable(x);
    let d = b.build();
    let mut m = HashMap::new();
    m.insert(x, Q::new(2, 3));
    assert_eq!(evaluate(&d, c, &m).unwrap(), Q::new(5, 1));
    assert_eq!(evaluate(&d, v, &m).unwrap(), Q::new(2, 3));
}

#[test]
fn addition_and_negation() {
    let mut b = DagBuilder::new();
    let x = VarId(0);
    let v = b.variable(x);
    let nv = b.neg(v);
    let sum = b.add(vec![v, nv]);
    let d = b.build();
    let mut m = HashMap::new();
    m.insert(x, Q::new(11, 2));
    assert_eq!(evaluate(&d, sum, &m).unwrap(), Q::new(0, 1));
}

#[test]
fn multiplication_and_shared_nodes() {
    let mut b = DagBuilder::new();
    let x = VarId(0);
    let v = b.variable(x);
    let one = b.constant(Q::new(1, 1));
    let shared = b.add(vec![v, one]);
    let prod = b.mul(vec![shared, shared]);
    let d = b.build();
    let mut m = HashMap::new();
    m.insert(x, Q::new(2, 1));
    assert_eq!(evaluate(&d, prod, &m).unwrap(), Q::new(9, 1));
}

#[test]
fn power_integer_exponent() {
    let mut b = DagBuilder::new();
    let x = VarId(0);
    let v = b.variable(x);
    let three = b.constant(Q::new(3, 1));
    let p = b.pow(v, three);
    let d = b.build();
    let mut m = HashMap::new();
    m.insert(x, Q::new(2, 1));
    assert_eq!(evaluate(&d, p, &m).unwrap(), Q::new(8, 1));
}

#[test]
fn power_non_integer_exponent_errors() {
    let mut b = DagBuilder::new();
    let x = VarId(0);
    let v = b.variable(x);
    let half = b.constant(Q::new(1, 2));
    let p = b.pow(v, half);
    let d = b.build();
    let mut m = HashMap::new();
    m.insert(x, Q::new(4, 1));
    assert_eq!(
        evaluate(&d, p, &m).unwrap_err(),
        EvalError::NonIntegerExponent
    );
}

#[test]
fn missing_variable_errors() {
    let mut b = DagBuilder::new();
    let x = VarId(0);
    let y = VarId(1);
    let vx = b.variable(x);
    let vy = b.variable(y);
    let e = b.add(vec![vx, vy]);
    let d = b.build();
    let mut m = HashMap::new();
    m.insert(x, Q::new(1, 1));
    assert_eq!(
        evaluate(&d, e, &m).unwrap_err(),
        EvalError::MissingVariable(y)
    );
}

#[test]
fn division_by_zero_constant() {
    let (dag, root) = Dag::from_rational_string("1 / (1 - 1)").unwrap();
    let m = HashMap::new();
    assert_eq!(
        evaluate(&dag, root, &m).unwrap_err(),
        EvalError::DivisionByZero
    );
}

#[test]
fn zero_to_negative_integer_power_errors() {
    let (dag, root) = Dag::from_rational_string("0 ^ -3").unwrap();
    let m = HashMap::new();
    assert_eq!(
        evaluate(&dag, root, &m).unwrap_err(),
        EvalError::DivisionByZero
    );
}
