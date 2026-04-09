//! Polynomial system: each equation is an [`Atom`] interpreted as `p = 0`.

use symbolica::atom::Atom;

#[derive(Clone, Debug)]
pub struct PolynomialSystem {
    pub equations: Vec<Atom>,
}

impl PolynomialSystem {
    #[must_use]
    pub fn new(equations: Vec<Atom>) -> Self {
        Self { equations }
    }
}
