//! Scalar field for v1: exact rationals via [`num_rational::Ratio`].
//!
//! Generalization to other fields is out of scope for the first implementation.

use num_rational::Ratio;
use num_traits::{One, Zero};

/// Rational numbers used throughout the crate.
pub type Q = Ratio<i64>;

/// Rational zero.
#[inline]
pub fn q_zero() -> Q {
    Q::zero()
}

/// Rational one.
#[inline]
pub fn q_one() -> Q {
    Q::one()
}

/// Identifier `pi` / Mathematica `\[Pi]` in the DAG; pi ≈ 355/113.
#[inline]
pub fn pi_approx_q() -> Q {
    Q::new(355, 113)
}
