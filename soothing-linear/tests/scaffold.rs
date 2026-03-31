use soothing_linear::{q, DagBuilder, PolynomialSystem, VarId};

#[test]
fn crate_smoke_rational_system_and_dag() {
    let mut b = DagBuilder::new();
    let v = VarId(0);
    let r = b.variable(v);
    let d = b.build();
    let s = PolynomialSystem::new(d, vec![r]);
    assert_eq!(s.equations.len(), 1);
    assert_eq!(q(3, 7), q(3, 7));
}
