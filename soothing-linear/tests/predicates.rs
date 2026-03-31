//! Predicate tests aligned with `tests/test_system_substitution.py` (`is_zero`, `is_linear_in_variable`).
//! Rust predicates operate on DAGs (substitution is applied when building the DAG elsewhere).

use num_traits::Zero;
use soothing_linear::{
    assignment_seed, expr_finite_at_some_random_sample, is_linear_in, is_zero, q,
    recover_linear_coeffs, Dag, DagBuilder, NodeId, PredicateError, SampleConfig, VarId,
};

/// Stricter [`zero_samples`] than default for multivariate checks (`linear_*` fields are unused by
/// Rust [`is_linear_in`] but kept for struct completeness).
fn cfg_robust() -> SampleConfig {
    SampleConfig {
        zero_samples: 16,
        ..Default::default()
    }
}

fn parse(s: &str) -> (Dag, NodeId) {
    Dag::from_rational_string(s).unwrap()
}

// ── is_zero (cf. test_is_zero) ─────────────────────────────────────────────

#[test]
fn is_zero_true() {
    let mut b = DagBuilder::new();
    let x = VarId(0);
    let vx = b.variable(x);
    let nx = b.neg(vx);
    let z = b.add(vec![vx, nx]);
    let d = b.build();
    let cfg = SampleConfig {
        zero_samples: 4,
        ..Default::default()
    };
    assert!(is_zero(&d, z, &[x], &cfg).unwrap());
}

#[test]
fn is_zero_false() {
    let mut b = DagBuilder::new();
    let x = VarId(0);
    let e = b.variable(x);
    let d = b.build();
    let cfg = SampleConfig {
        zero_samples: 4,
        ..Default::default()
    };
    assert!(!is_zero(&d, e, &[x], &cfg).unwrap());
}

#[test]
fn is_zero_constant_zero() {
    let (d, r) = parse("0");
    let cfg = cfg_robust();
    assert!(is_zero(&d, r, &[], &cfg).unwrap());
}

#[test]
fn is_zero_constant_nonzero() {
    let (d, r) = parse("1");
    let cfg = cfg_robust();
    assert!(!is_zero(&d, r, &[], &cfg).unwrap());
}

#[test]
fn is_zero_tiny_nonzero_rational_is_not_zero() {
    let (d, r) = parse("1 / 10000000000");
    let cfg = cfg_robust();
    assert!(!is_zero(&d, r, &[], &cfg).unwrap());
}

#[test]
fn is_zero_strictly_positive_polynomial_never_zero_on_samples() {
    // `v0 + v1` can vanish when `v1 = -v0` under random sampling; use a sum of squares + 1.
    let (d, r) = parse("v0^2 + v1^2 + 1");
    let cfg = cfg_robust();
    assert!(!is_zero(&d, r, &[VarId(0), VarId(1)], &cfg).unwrap());
}

#[test]
fn is_zero_polynomial_identically_zero() {
    let (d, r) = parse("v0 - v0");
    let cfg = cfg_robust();
    assert!(is_zero(&d, r, &[VarId(0)], &cfg).unwrap());
}

#[test]
fn is_zero_nonzero_polynomial_not_fooled_by_small_integer_grid() {
    let (d, r) = parse("v0 * (v0 - 1) * (v0 - 2) * (v0 + 1) * (v0 + 2)");
    let cfg = cfg_robust();
    assert!(!is_zero(&d, r, &[VarId(0)], &cfg).unwrap());
}

#[test]
fn is_zero_ignores_removable_zero_over_denominator_singularity() {
    let (d, r) = parse("0 / (v0 + 1)");
    let cfg = cfg_robust();
    assert!(is_zero(&d, r, &[VarId(0)], &cfg).unwrap());
}

// ── is_linear_in (cf. test_is_linear_in_variable) ───────────────────────────

#[test]
fn linear_true() {
    let mut b = DagBuilder::new();
    let x = VarId(0);
    let y = VarId(1);
    let v = b.variable(x);
    let w = b.variable(y);
    let e = b.add(vec![v, w]); // v, w already distinct nodes
    let d = b.build();
    let cfg = SampleConfig::default();
    assert!(is_linear_in(&d, e, x, &[y], &cfg).unwrap());
}

#[test]
fn is_linear_plain_affine_in_x() {
    // Poly(3*x + 2*y - 1, x, y)
    let (d, r) = parse("3 * v0 + 2 * v1 - 1");
    let cfg = cfg_robust();
    assert!(is_linear_in(&d, r, VarId(0), &[VarId(1)], &cfg).unwrap());
}

#[test]
fn is_linear_quadratic_in_x() {
    let (d, r) = parse("v0^2 + v1");
    let cfg = cfg_robust();
    assert!(!is_linear_in(&d, r, VarId(0), &[VarId(1)], &cfg).unwrap());
}

#[test]
fn is_linear_var_absent_rust_treats_as_affine_in_x() {
    // Python `is_linear_in_variable` returns False when `var` is not in free_symbols.
    // Here `v0` does not appear: expression is affine in `v0` with slope 0 (constant in `v0`).
    let (d, r) = parse("3 * v1 + 1");
    let cfg = cfg_robust();
    assert!(is_linear_in(&d, r, VarId(0), &[VarId(1)], &cfg).unwrap());
}

#[test]
fn is_linear_coefficient_involves_other_vars() {
    // Poly(y*x + z, x, y, z)  →  v1 * v0 + v2
    let (d, r) = parse("v1 * v0 + v2");
    let cfg = cfg_robust();
    assert!(is_linear_in(&d, r, VarId(0), &[VarId(1), VarId(2)], &cfg).unwrap());
}

#[test]
fn is_linear_product_of_vars_not_linear_in_x() {
    // Poly(x*y + x^2, x, y)
    let (d, r) = parse("v0 * v1 + v0^2");
    let cfg = cfg_robust();
    assert!(!is_linear_in(&d, r, VarId(0), &[VarId(1)], &cfg).unwrap());
}

#[test]
fn is_linear_cubic_in_x() {
    let (d, r) = parse("v0^3 + v1");
    let cfg = cfg_robust();
    assert!(!is_linear_in(&d, r, VarId(0), &[VarId(1)], &cfg).unwrap());
}

#[test]
fn is_linear_high_degree_in_other_vars_still_linear_in_x() {
    // Same shape as Poly(y^k * x + z^m, …): high degree in y,z, linear in x. Exponents kept small so
    // random samples do not overflow `i64` in `Q` powers.
    let (d, r) = parse("v1^3 * v0 + v2^2");
    let cfg = cfg_robust();
    assert!(is_linear_in(&d, r, VarId(0), &[VarId(1), VarId(2)], &cfg).unwrap());
}

#[test]
fn is_linear_constant_is_affine_in_x() {
    // Poly(5, x): Python rejects (var not free); Rust collinearity sees constant `v0`-slice.
    let (d, r) = parse("5");
    let cfg = cfg_robust();
    assert!(is_linear_in(&d, r, VarId(0), &[], &cfg).unwrap());
}

#[test]
fn quadratic_in_x_not_linear_even_with_parameter_y() {
    let mut b = DagBuilder::new();
    let x = VarId(0);
    let y = VarId(1);
    let vx = b.variable(x);
    let vy = b.variable(y);
    let x2 = b.mul(vec![vx, vx]);
    let e = b.add(vec![x2, vy]);
    let d = b.build();
    let cfg = SampleConfig::default();
    assert!(!is_linear_in(&d, e, x, &[y], &cfg).unwrap());
}

#[test]
fn linear_false() {
    let mut b = DagBuilder::new();
    let x = VarId(0);
    let vx = b.variable(x);
    let e = b.mul(vec![vx, vx]);
    let d = b.build();
    let cfg = SampleConfig::default();
    assert!(!is_linear_in(&d, e, x, &[], &cfg).unwrap());
}

// ── Nonlinear in x (cf. test_is_linear_false_positive_nonlinear_after_substitution) ──

#[test]
fn is_linear_detects_quadratic_after_symbolic_substitution_equivalent_dag() {
    // y + z with y -> x^2  becomes  x^2 + z
    let (d, r) = parse("v0^2 + v2");
    let cfg = cfg_robust();
    assert!(!is_linear_in(&d, r, VarId(0), &[VarId(2)], &cfg).unwrap());
}

#[test]
fn is_linear_detects_quadratic_chain_equivalent_dag() {
    // z + w with y = x+1, z = y^2  →  (x+1)^2 + w  (expanded test shape)
    let (d, r) = parse("(v0 + 1)^2 + v3");
    let cfg = cfg_robust();
    assert!(!is_linear_in(&d, r, VarId(0), &[VarId(3)], &cfg).unwrap());
}

#[test]
fn is_linear_detects_mixed_quadratic_linear_in_x() {
    // y + z with y -> x^2 + x  →  x^2 + x + z
    let (d, r) = parse("v0^2 + v0 + v2");
    let cfg = cfg_robust();
    assert!(!is_linear_in(&d, r, VarId(0), &[VarId(2)], &cfg).unwrap());
}

#[test]
fn is_linear_cubic_via_substitution_equivalent_dag() {
    let (d, r) = parse("v0^3 + v3");
    let cfg = cfg_robust();
    assert!(!is_linear_in(&d, r, VarId(0), &[VarId(3)], &cfg).unwrap());
}

#[test]
fn is_linear_nonlinear_coefficient_multiplying_x_squared() {
    // Poly(x^2 * y + x, x, y)
    let (d, r) = parse("v0^2 * v1 + v0");
    let cfg = cfg_robust();
    assert!(!is_linear_in(&d, r, VarId(0), &[VarId(1)], &cfg).unwrap());
}

#[test]
fn is_linear_substituted_expr_makes_coefficient_nonlinear_in_x() {
    // y + w with y -> x^2 * z  →  x^2 * z + w
    let (d, r) = parse("v0^2 * v2 + v3");
    let cfg = cfg_robust();
    assert!(!is_linear_in(&d, r, VarId(0), &[VarId(2), VarId(3)], &cfg).unwrap());
}

#[test]
fn is_linear_multiple_nonlinear_sources() {
    let (d, r) = parse("v0^2 + v0^3 + v3");
    let cfg = cfg_robust();
    assert!(!is_linear_in(&d, r, VarId(0), &[VarId(3)], &cfg).unwrap());
}

// ── Small leading nonlinear term (cf. test_is_linear_numerical_false_positive) ──
// Predicate sampling is still Monte Carlo, but evaluation is exact in `Q`, so even tiny rational
// nonlinear terms should remain detectable as long as sampling hits a witness point.

#[test]
fn is_linear_epsilon_quadratic_exact_rationals_not_linear() {
    let (d, r) = parse("v0^2 / 1000000000 + v0");
    let cfg = cfg_robust();
    assert!(!is_linear_in(&d, r, VarId(0), &[], &cfg).unwrap());
}

#[test]
fn is_linear_epsilon_quadratic_negative_leading_coeff() {
    let (d, r) = parse("-v0^2 / 1000000000 + v0");
    let cfg = cfg_robust();
    assert!(!is_linear_in(&d, r, VarId(0), &[], &cfg).unwrap());
}

#[test]
fn is_linear_large_linear_small_quadratic_still_detected() {
    let (d, r) = parse("v0^2 / 1000000000 + 1000 * v0");
    let cfg = cfg_robust();
    assert!(!is_linear_in(&d, r, VarId(0), &[], &cfg).unwrap());
}

#[test]
fn is_linear_exact_q_rejects_quadratic_hidden_by_large_scale() {
    let (d, r) = parse("v0^2 / 1000000000 + 1000000 * v0");
    let cfg = cfg_robust();
    assert!(!is_linear_in(&d, r, VarId(0), &[], &cfg).unwrap());
}

#[test]
fn is_linear_free_symbol_scales_quadratic() {
    let (d, r) = parse("v0^2 * v1 / 1000000000 + v0");
    let cfg = cfg_robust();
    assert!(!is_linear_in(&d, r, VarId(0), &[VarId(1)], &cfg).unwrap());
}

#[test]
fn is_linear_two_free_symbols_scale_quadratic() {
    let (d, r) = parse("v0^2 * v1 * v3 / 1000000000 + v0");
    let cfg = cfg_robust();
    assert!(!is_linear_in(&d, r, VarId(0), &[VarId(1), VarId(3)], &cfg).unwrap());
}

#[test]
fn is_linear_epsilon_cubic() {
    let (d, r) = parse("v0^3 / 1000000000 + v0");
    let cfg = cfg_robust();
    assert!(!is_linear_in(&d, r, VarId(0), &[], &cfg).unwrap());
}

// ── recover_linear_coeffs ───────────────────────────────────────────────────

#[test]
fn recover_coeffs_rational_slope() {
    let mut b = DagBuilder::new();
    let x = VarId(0);
    let y = VarId(1);
    let xv = b.variable(x);
    let half = b.constant(q(1, 2));
    let half_x = b.mul(vec![half, xv]);
    let yv = b.variable(y);
    let e = b.add(vec![half_x, yv]);
    let d = b.build();
    let base = assignment_seed(&[y], 0);
    let coeffs = recover_linear_coeffs(&d, e, x, &base).unwrap();
    assert_eq!(coeffs.a, q(1, 2));
    assert!(!coeffs.b.is_zero());
}

#[test]
fn recover_coeffs_affine() {
    let mut b = DagBuilder::new();
    let x = VarId(0);
    let y = VarId(1);
    let xv = b.variable(x);
    let c2 = b.constant(q(2, 1));
    let two_x = b.mul(vec![c2, xv]);
    let yv = b.variable(y);
    let e = b.add(vec![two_x, yv]);
    let d = b.build();
    let base = assignment_seed(&[y], 0);
    let c = recover_linear_coeffs(&d, e, x, &base).unwrap();
    assert_eq!(c.a, q(2, 1));
    assert!(!c.b.is_zero());
}

// ── errors / degenerate config ─────────────────────────────────────────────

#[test]
fn degenerate_or_insufficient() {
    let mut b = DagBuilder::new();
    let x = VarId(0);
    let e = b.variable(x);
    let d = b.build();
    let cfg = SampleConfig {
        zero_samples: 0,
        ..Default::default()
    };
    let err = is_zero(&d, e, &[x], &cfg).unwrap_err();
    assert!(matches!(err, PredicateError::InsufficientSamples));
}

#[test]
fn is_linear_in_rejects_zero_zero_samples() {
    let (d, r) = parse("v0 + v1");
    let cfg = SampleConfig {
        zero_samples: 0,
        ..cfg_robust()
    };
    let err = is_linear_in(&d, r, VarId(0), &[VarId(1)], &cfg).unwrap_err();
    assert!(matches!(err, PredicateError::InsufficientSamples));
}

#[test]
fn exact_substitution_definedness_accepts_some_valid_sample() {
    let (d, r) = parse("1 / v0");
    let cfg = cfg_robust();
    assert!(expr_finite_at_some_random_sample(&d, r, &[VarId(0)], &cfg));
}

#[test]
fn exact_substitution_definedness_rejects_always_invalid_expr() {
    let (d, r) = parse("1 / 0");
    let cfg = cfg_robust();
    assert!(!expr_finite_at_some_random_sample(&d, r, &[], &cfg));
}

#[test]
fn is_linear_ignores_removable_zero_over_denominator_singularity() {
    let (d, r) = parse("v0 + (0 / (v0 + 1)) - 1");
    let cfg = cfg_robust();
    assert!(is_linear_in(&d, r, VarId(0), &[], &cfg).unwrap());
}
