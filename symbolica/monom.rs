//! Monomial exponent-vector helpers shared by the Gröbner basis algorithms.
//!
//! Monomials are exponent vectors `&[u16]` aligned to a fixed variable order.
//! Divisibility, exact division, and lcm are the primitives every
//! Buchberger-style algorithm needs.

/// `true` iff `divisor` divides `dividend` (exponent-wise `≤`).
pub(crate) fn monom_divides(divisor: &[u16], dividend: &[u16]) -> bool {
    divisor.iter().zip(dividend).all(|(x, y)| x <= y)
}

/// Exponent vector of `dividend / divisor`, assuming `divisor | dividend`.
pub(crate) fn monom_quot(dividend: &[u16], divisor: &[u16]) -> Vec<u16> {
    dividend.iter().zip(divisor).map(|(x, y)| x - y).collect()
}

/// Least common multiple of two monomials (exponent-wise `max`).
pub(crate) fn monom_lcm(a: &[u16], b: &[u16]) -> Vec<u16> {
    a.iter().zip(b).map(|(x, y)| *x.max(y)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn monom_divides_basic() {
        assert!(monom_divides(&[1, 0], &[2, 1]));
        assert!(!monom_divides(&[1, 2], &[2, 1]));
        assert!(monom_divides(&[0, 0], &[5, 7]));
    }

    #[test]
    fn monom_lcm_and_quot() {
        assert_eq!(monom_lcm(&[1, 2, 0], &[0, 3, 4]), vec![1, 3, 4]);
        assert_eq!(monom_quot(&[3, 2, 4], &[1, 0, 4]), vec![2, 2, 0]);
    }
}
