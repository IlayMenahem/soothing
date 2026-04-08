#![allow(clippy::many_single_char_names)]

use soothing_linear::{q, ExprBuilder, PolynomialSystem, VarId};

#[test]
fn crate_smoke_rational_system_and_egraph() {
    let mut b = ExprBuilder::new();
    let v = VarId(0);
    let r = b.variable(v);
    let d = b.into_egraph();
    let s = PolynomialSystem::new(d, vec![r]);
    assert_eq!(s.equations.len(), 1);
    assert_eq!(q(3, 7), q(3, 7));
}
