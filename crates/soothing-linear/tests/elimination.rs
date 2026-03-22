use soothing_linear::{
    linear_elimination, DagBuilder, PolynomialSystem, Q, SampleConfig, VarId,
};

#[test]
fn single_step() {
    let mut b = DagBuilder::new();
    let x = VarId(0);
    let y = VarId(1);
    let vx = b.variable(x);
    let vy = b.variable(y);
    let e = b.add(vec![vx, vy]);
    let sys = PolynomialSystem::new(b.build(), vec![e]);
    let r = linear_elimination(sys, &SampleConfig::default()).unwrap();
    assert!(r.substitution.contains(x) || r.substitution.contains(y));
}

#[test]
fn recomputes_candidates() {
    let mut b = DagBuilder::new();
    let x = VarId(0);
    let y = VarId(1);
    let vx = b.variable(x);
    let vy = b.variable(y);
    let e1 = b.add(vec![vx, vy]);
    let vx2 = b.variable(x);
    let c = b.constant(Q::new(1, 1));
    let e2 = b.add(vec![vx2, c]);
    let sys = PolynomialSystem::new(b.build(), vec![e1, e2]);
    let r = linear_elimination(sys, &SampleConfig::default()).unwrap();
    assert!(r.substitution.len() >= 1);
}

#[test]
fn removes_zero_equations() {
    let mut b = DagBuilder::new();
    let x = VarId(0);
    let vx = b.variable(x);
    let nx = b.neg(vx);
    let z = b.add(vec![vx, nx]);
    let sys = PolynomialSystem::new(b.build(), vec![z]);
    let r = linear_elimination(sys, &SampleConfig::default()).unwrap();
    assert!(r.system.equations.is_empty());
}

#[test]
fn end_to_end() {
    let mut b = DagBuilder::new();
    let x = VarId(0);
    let y = VarId(1);
    let vx = b.variable(x);
    let vy = b.variable(y);
    let c = b.constant(Q::new(1, 1));
    let e = b.add(vec![vx, vy, c]);
    let sys = PolynomialSystem::new(b.build(), vec![e]);
    let r = linear_elimination(sys, &SampleConfig::default()).unwrap();
    assert!(r.substitution.len() >= 1);
}
