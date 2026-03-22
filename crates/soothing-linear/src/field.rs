//! Scalar field for v1: exact rationals via [`num_rational::Ratio`].
//!
//! Generalization to other fields is out of scope for the first implementation.

use num_rational::Ratio;
use num_traits::{One, Zero};

/// Rational numbers used throughout the crate (`\mathbb{Q}` in the paper).
pub type Q = Ratio<i64>;

/// Zero in `\mathbb{Q}`.
#[inline]
pub fn q_zero() -> Q {
    Q::zero()
}

/// One in `\mathbb{Q}`.
#[inline]
pub fn q_one() -> Q {
    Q::one()
}
