//! Scalar field for v1: exact rationals [`num_rational::Ratio`] over `i128`.
//!
//! Generalization to other fields is out of scope for the first implementation.

use num_rational::Ratio;
use num_traits::{One, Zero};

/// Rational numbers used throughout the crate.
pub type Q = Ratio<i128>;

/// Rational zero.
#[inline]
#[must_use]
pub fn q_zero() -> Q {
    Q::zero()
}

/// Rational one.
#[inline]
#[must_use]
pub fn q_one() -> Q {
    Q::one()
}

/// `p/q` from small integer parts (typical literals and tests).
#[inline]
#[must_use]
pub fn q(numer: i64, denom: i64) -> Q {
    Q::new(i128::from(numer), i128::from(denom))
}

/// Identifier `pi` / Mathematica `\[Pi]` in the DAG; pi ≈ 355/113.
#[inline]
#[must_use]
pub fn pi_approx_q() -> Q {
    q(355, 113)
}
