use soothing_linear::{
    assignment_seed, is_linear_in, is_zero, recover_linear_coeffs, DagBuilder, PredicateError, Q,
    SampleConfig, VarId,
};
use num_traits::Zero;

#[test]
fn is_zero_true() {
    let mut b = DagBuilder::new();
    let x = VarId(0);
    let vx = b.variable(x);
    let nx = b.neg(vx);
    let z = b.add(vec![vx, nx]);
    let d = b.build();
    let cfg = SampleConfig {
        zero_samples: 4,
        ..Default::default()
    };
    assert!(is_zero(&d, z, &[x], &cfg).unwrap());
}

#[test]
fn is_zero_false() {
    let mut b = DagBuilder::new();
    let x = VarId(0);
    let e = b.variable(x);
    let d = b.build();
    let cfg = SampleConfig {
        zero_samples: 4,
        ..Default::default()
    };
    assert!(!is_zero(&d, e, &[x], &cfg).unwrap());
}

#[test]
fn linear_true() {
    let mut b = DagBuilder::new();
    let x = VarId(0);
    let y = VarId(1);
    let v = b.variable(x);
    let w = b.variable(y);
    let e = b.add(vec![v, w]); // v, w already distinct nodes
    let d = b.build();
    let cfg = SampleConfig::default();
    assert!(is_linear_in(&d, e, x, &[y], &cfg).unwrap());
}

#[test]
fn linear_false() {
    let mut b = DagBuilder::new();
    let x = VarId(0);
    let vx = b.variable(x);
    let e = b.mul(vec![vx, vx]);
    let d = b.build();
    let cfg = SampleConfig::default();
    assert!(!is_linear_in(&d, e, x, &[], &cfg).unwrap());
}

#[test]
fn recover_coeffs_affine() {
    let mut b = DagBuilder::new();
    let x = VarId(0);
    let y = VarId(1);
    let xv = b.variable(x);
    let c2 = b.constant(Q::new(2, 1));
    let two_x = b.mul(vec![c2, xv]);
    let yv = b.variable(y);
    let e = b.add(vec![two_x, yv]);
    let d = b.build();
    let base = assignment_seed(&[y], 0);
    let c = recover_linear_coeffs(&d, e, x, &base).unwrap();
    assert_eq!(c.a, Q::new(2, 1));
    assert!(!c.b.is_zero());
}

#[test]
fn degenerate_or_insufficient() {
    let mut b = DagBuilder::new();
    let x = VarId(0);
    let e = b.variable(x);
    let d = b.build();
    let cfg = SampleConfig {
        zero_samples: 0,
        ..Default::default()
    };
    let err = is_zero(&d, e, &[x], &cfg).unwrap_err();
    assert!(matches!(err, PredicateError::InsufficientSamples));
}
