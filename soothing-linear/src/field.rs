//! Scalar field for v1: exact rationals [`num_rational::Ratio`] over [`num_bigint::BigInt`].
//!
//! Generalization to other fields is out of scope for the first implementation.

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{One, ToPrimitive, Zero};

/// Rational numbers used throughout the crate.
pub type Q = Ratio<BigInt>;

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

/// `p/q` from small integer parts (typical literals and tests).
#[inline]
pub fn q(numer: i64, denom: i64) -> Q {
    Q::new(numer.into(), denom.into())
}

/// Identifier `pi` / Mathematica `\[Pi]` in the DAG; pi ≈ 355/113.
#[inline]
pub fn pi_approx_q() -> Q {
    q(355, 113)
}

/// `base` raised to integer exponent `exp` (negative exponents invert `base`).
/// Returns `None` for `0` to a negative integer power.
pub fn q_pow_integer(base: Q, exp: &BigInt) -> Option<Q> {
    if exp.is_zero() {
        return Some(Q::one());
    }
    let mut exp_pos = exp.clone();
    let mut base = base;
    if exp_pos < BigInt::zero() {
        if base.is_zero() {
            return None;
        }
        base = Q::one() / base;
        exp_pos = -exp_pos;
    }
    if let Some(i) = exp_pos.to_i32() {
        return Some(base.pow(i));
    }
    let mut acc = Q::one();
    let mut b = base;
    let mut e = exp_pos;
    while e > BigInt::zero() {
        if &e & BigInt::from(1u8) != BigInt::zero() {
            acc *= &b;
        }
        b = &b * &b;
        e >>= 1;
    }
    Some(acc)
}
