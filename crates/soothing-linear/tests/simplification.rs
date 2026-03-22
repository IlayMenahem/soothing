use soothing_linear::{evaluate, simplify_once, DagBuilder, Q, VarId};
use std::collections::HashMap;

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
