use soothing_linear::{DagBuilder, PolynomialSystem, VarId, Q};

#[test]
fn crate_smoke_rational_system_and_dag() {
    let mut b = DagBuilder::new();
    let v = VarId(0);
    let r = b.variable(v);
    let d = b.build();
    let s = PolynomialSystem::new(d, vec![r]);
    assert_eq!(s.equations.len(), 1);
    assert_eq!(Q::new(3, 7), Q::new(3, 7));
}
