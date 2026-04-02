//! Scalar field for v1: exact rationals [`num_rational::Ratio`] over [`num_bigint::BigInt`].
//!
//! Generalization to other fields is out of scope for the first implementation.

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{One, Pow, Zero};

/// Rational numbers used throughout the crate.
pub type Q = Ratio<BigInt>;

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
    Q::new(numer.into(), denom.into())
}

/// Identifier `pi` / Mathematica `\[Pi]` in the DAG; pi ≈ 355/113.
#[inline]
#[must_use] 
pub fn pi_approx_q() -> Q {
    q(355, 113)
}

/// `base` raised to integer exponent `exp` (negative exponents invert `base`).
/// Returns `None` for `0` to a negative integer power.
#[must_use] 
pub fn q_pow_integer(mut base: Q, exp: &BigInt) -> Option<Q> {
    if exp.is_zero() {
        return Some(Q::one());
    }

    let mut exp_pos = exp.clone();

    if exp_pos < BigInt::zero() {
        if base.is_zero() {
            return None;
        }
        base = Q::one() / base;
        exp_pos = -exp_pos;
    }

    let exp_u = exp_pos.to_biguint()?;
    Some(Pow::pow(base, &exp_u))
}
