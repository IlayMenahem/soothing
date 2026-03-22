use soothing_linear::{DagBuilder, Q, SubstitutionMap, VarId};

#[test]
fn single_binding() {
    let mut b = DagBuilder::new();
    let x = VarId(0);
    let one = b.constant(Q::new(1, 1));
    let vx = b.variable(x);
    let e = b.add(vec![vx, one]);
    let d = b.build();
    let mut m = SubstitutionMap::new();
    m.insert(x, &d, e);
    assert_eq!(m.len(), 1);
    assert!(m.get(x).is_some());
}

#[test]
fn replay_order() {
    let x = VarId(0);
    let y = VarId(1);
    let mut by = DagBuilder::new();
    let ey = by.constant(Q::new(1, 1));
    let dy = by.build();
    let mut bx = DagBuilder::new();
    let c2 = bx.constant(Q::new(2, 1));
    let vy = bx.variable(y);
    let ex = bx.add(vec![vy, c2]);
    let dx = bx.build();
    let mut m = SubstitutionMap::new();
    m.insert(y, &dy, ey);
    m.insert(x, &dx, ex);
    let order = m.replay_order().unwrap();
    assert_eq!(order, vec![y, x]);
}

#[test]
fn cycle_rejected() {
    let mut b = DagBuilder::new();
    let x = VarId(0);
    let y = VarId(1);
    let ex = b.variable(y);
    let dx = b.build();
    let mut b2 = DagBuilder::new();
    let ey = b2.variable(x);
    let dy = b2.build();
    let mut m = SubstitutionMap::new();
    m.insert(x, &dx, ex);
    m.insert(y, &dy, ey);
    assert!(m.replay_order().is_err());
}

#[test]
fn sampled_consistency() {
    use soothing_linear::{assignment_seed, evaluate};
    use num_traits::Zero;

    let mut b = DagBuilder::new();
    let x = VarId(0);
    let y = VarId(1);
    let c2 = b.constant(Q::new(2, 1));
    let vy = b.variable(y);
    let ex = b.add(vec![vy, c2]);
    let d = b.build();
    let mut m = SubstitutionMap::new();
    m.insert(x, &d, ex);
    let root = m.get(x).unwrap();
    let dag = m.dag();
    let a = assignment_seed(&[y], 0);
    let vx = evaluate(&dag, root, &a).unwrap();
    let mut full = a.clone();
    full.insert(x, vx);
    let mut b2 = DagBuilder::new();
    let xv = b2.variable(x);
    let yv = b2.variable(y);
    let c2 = b2.constant(Q::new(2, 1));
    let y_plus_2 = b2.add(vec![yv, c2]);
    let ny = b2.neg(y_plus_2);
    let orig = b2.add(vec![xv, ny]);
    let d2 = b2.build();
    assert!(evaluate(&d2, orig, &full).unwrap().is_zero());
}
