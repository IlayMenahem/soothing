//! Sampling-based “is zero” and “is linear in `x`” predicates (paper).

use crate::dag::{Dag, NodeId, VarId};
use crate::eval::{evaluate, evaluate_f64, EvalError};
use crate::field::Q;
use num_traits::{One, Zero};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;
use thiserror::Error;

/// How many samples to use for predicate checks (paper leaves this to implementation).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SampleConfig {
    /// Number of full assignments for the zero test.
    pub zero_samples: usize,
    /// Number of assignments to “other” variables when testing linearity in `x`.
    pub linear_other_assignments: usize,
    /// Minimum distinct `x` values to check collinearity (at least 3 recommended).
    pub linear_x_points: usize,
    /// Seed for [`StdRng`] when drawing random values for free / “other” variables (mixed with
    /// expression identity so repeated calls are not identical).
    pub random_seed: u64,
}

impl Default for SampleConfig {
    fn default() -> Self {
        Self {
            zero_samples: 8,
            linear_other_assignments: 4,
            linear_x_points: 8,
            random_seed: 0xE0B0_5EED_E0B0_5EED,
        }
    }
}

/// Half-width of the uniform range for random **floating-point** samples in [`is_zero`] and
/// [`is_linear_in`] “other” variables (matches `soothing/system_substitution.py` `_RANDOM_VALUE_RANGE`).
pub const RANDOM_SAMPLE_ABS_MAX: f64 = 2.0;

/// Endpoints for the sweep of the linearity-test variable (matches Python `_VAR_RANGE`).
pub const LINEARITY_X_SAMPLE_LOW: f64 = -10.0;
pub const LINEARITY_X_SAMPLE_HIGH: f64 = 10.0;

/// Predicate sampling uses [`RANDOM_SAMPLE_ABS_MAX`] (float). Kept as an `i64` alias for older call sites.
pub const RANDOM_Q_SAMPLE_ABS_MAX: i64 = RANDOM_SAMPLE_ABS_MAX as i64;

/// Samples for numeric finiteness check on a substitution RHS (like ruling out SymPy `nan`/`zoo`).
pub const SUBSTITUTION_NUMERIC_CHECK_SAMPLES: usize = 8;

fn mix_seed(base: u64, a: usize, b: u32) -> u64 {
    base ^ (a as u64).wrapping_mul(0x9E37_79B9_1E35_1B03)
        ^ (b as u64).wrapping_mul(0xC6BC_2796_92B5_C323)
}

fn rng_for_zero(cfg: &SampleConfig, root: NodeId) -> StdRng {
    StdRng::seed_from_u64(mix_seed(cfg.random_seed, root.0, 0x5EED_5EED))
}

fn rng_for_linear(cfg: &SampleConfig, root: NodeId, x: VarId) -> StdRng {
    StdRng::seed_from_u64(mix_seed(cfg.random_seed, root.0, x.0))
}

fn random_assignment_f64(vars: &[VarId], rng: &mut StdRng) -> HashMap<VarId, f64> {
    vars.iter()
        .map(|&v| {
            let x: f64 = rng.gen_range(-RANDOM_SAMPLE_ABS_MAX..=RANDOM_SAMPLE_ABS_MAX);
            (v, x)
        })
        .collect()
}

fn linspace(low: f64, high: f64, n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![low];
    }
    (0..n)
        .map(|i| low + (high - low) * (i as f64) / ((n - 1) as f64))
        .collect()
}

/// True if at least one random assignment yields a finite [`evaluate_f64`] value (numeric analogue
/// of rejecting `nan` / invalid rationals on a substitution RHS).
pub fn expr_finite_at_some_random_sample(
    dag: &Dag,
    root: NodeId,
    free_vars: &[VarId],
    cfg: &SampleConfig,
) -> bool {
    let mut rng = rng_for_zero(cfg, root);
    for _ in 0..SUBSTITUTION_NUMERIC_CHECK_SAMPLES {
        let a = random_assignment_f64(free_vars, &mut rng);
        if let Ok(v) = evaluate_f64(dag, root, &a) {
            if v.is_finite() {
                return true;
            }
        }
    }
    false
}

/// “Is zero” under floating-point evaluation (see [`is_zero`]).
fn is_effectively_zero(v: f64) -> bool {
    v.is_finite() && v.abs() <= PREDICATE_ZERO_EPS
}

/// True iff samples `(x_i, y_i)` lie on one affine line. [`points_on_one_affine_line`] uses uniform
/// spacing in [`LINEARITY_X_SAMPLE_LOW`] … [`LINEARITY_X_SAMPLE_HIGH`]; for affine `y` the second
/// difference `y_{i+2} - 2 y_{i+1} + y_i` is `0` (and for `y = a x^2 + …` it is `2a` up to step scale).
fn affine_points_collinear(_xs: &[f64], ys: &[f64]) -> bool {
    debug_assert_eq!(_xs.len(), ys.len());
    if _xs.len() < 2 {
        return false;
    }
    if _xs.len() == 2 {
        return _xs[1] != _xs[0];
    }
    for i in 0..ys.len() - 2 {
        let d2 = ys[i + 2] - 2.0 * ys[i + 1] + ys[i];
        if !d2.is_finite() {
            return false;
        }
        let ymax = ys[i]
            .abs()
            .max(ys[i + 1].abs())
            .max(ys[i + 2].abs())
            .max(1.0);
        let tol =
            COLLINEAR_SECOND_DIFF_EPS.max(COLLINEAR_SECOND_DIFF_RTOL * ymax);
        if d2.abs() > tol {
            return false;
        }
    }
    true
}

/// Absolute tolerance for reporting “zero” in [`is_zero`] (matches Python `_ZERO_TOL`).
const PREDICATE_ZERO_EPS: f64 = 1e-9;
/// Base max `|y_{i+2} - 2y_{i+1} + y_i|` for uniform `x` (catches small curvatures).
const COLLINEAR_SECOND_DIFF_EPS: f64 = 1.5e-9;
/// Relative slack so affine slices at large `|y|` still pass despite `f64` cancellation (kept small
/// enough that `d2 = 2` for a unit quadratic in `x` still exceeds the threshold when `|y|` is large).
const COLLINEAR_SECOND_DIFF_RTOL: f64 = 5e-14;

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum PredicateError {
    #[error("expression is not affine in the chosen variable")]
    NotLinear,
    #[error("degenerate slope: a = 0 in ax + b")]
    DegenerateSlope,
    #[error("not enough samples for predicate")]
    InsufficientSamples,
    #[error("evaluation failed: missing variable {0:?}")]
    MissingVariable(VarId),
    #[error("evaluation failed: division by zero")]
    DivisionByZero,
    #[error("evaluation failed: exponent must be an integer in Q")]
    NonIntegerExponent,
}

impl From<EvalError> for PredicateError {
    fn from(e: EvalError) -> Self {
        match e {
            EvalError::MissingVariable(v) => PredicateError::MissingVariable(v),
            EvalError::DivisionByZero => PredicateError::DivisionByZero,
            EvalError::NonIntegerExponent => PredicateError::NonIntegerExponent,
        }
    }
}

/// `free_vars`: all variables that may appear in the expression (caller supplies).
///
/// One random draw per sample (same spirit as `soothing/system_substitution.py` `_is_zero_probabilistic`):
/// division, non-finite values, or `|v| \ge` tolerance ⇒ [`Ok(false)`] (does not abort elimination).
pub fn is_zero(
    dag: &Dag,
    root: NodeId,
    free_vars: &[VarId],
    cfg: &SampleConfig,
) -> Result<bool, PredicateError> {
    if cfg.zero_samples == 0 {
        return Err(PredicateError::InsufficientSamples);
    }
    let mut rng = rng_for_zero(cfg, root);
    for _ in 0..cfg.zero_samples {
        let assignment = random_assignment_f64(free_vars, &mut rng);
        let v = match evaluate_f64(dag, root, &assignment) {
            Ok(val) if val.is_finite() => val,
            Ok(_) => return Ok(false),
            Err(EvalError::DivisionByZero) => return Ok(false),
            Err(e) => return Err(e.into()),
        };
        if !is_effectively_zero(v) {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Whether `p` is affine in `x` for random assignments to the other variables (uniform in
/// ± [`RANDOM_SAMPLE_ABS_MAX`]), then collinearity over `x` on an evenly spaced grid in
/// [`LINEARITY_X_SAMPLE_LOW`] … [`LINEARITY_X_SAMPLE_HIGH`] (Python `is_linear_in_variable` style).
pub fn is_linear_in(
    dag: &Dag,
    root: NodeId,
    x: VarId,
    other_vars: &[VarId],
    cfg: &SampleConfig,
) -> Result<bool, PredicateError> {
    if cfg.linear_x_points < 2 || cfg.linear_other_assignments == 0 {
        return Err(PredicateError::InsufficientSamples);
    }
    let mut rng = rng_for_linear(cfg, root, x);
    for _ in 0..cfg.linear_other_assignments {
        let base = random_assignment_f64(other_vars, &mut rng);
        if !points_on_one_affine_line(dag, root, x, &base, cfg.linear_x_points) {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Recover `a`, `b` with `p = a*x + b` in `x` (other variables fixed by `base`).
pub fn recover_linear_coeffs(
    dag: &Dag,
    root: NodeId,
    x: VarId,
    base: &HashMap<VarId, Q>,
) -> Result<LinearCoeffs, PredicateError> {
    let mut a0 = base.clone();
    a0.insert(x, Q::zero());
    let mut a1 = base.clone();
    a1.insert(x, Q::one());
    let b = evaluate(dag, root, &a0)?;
    let at_one = evaluate(dag, root, &a1)?;
    let a = at_one - b;
    Ok(LinearCoeffs { a, b })
}

#[derive(Clone, Debug, PartialEq)]
pub struct LinearCoeffs {
    pub a: Q,
    pub b: Q,
}

fn points_on_one_affine_line(
    dag: &Dag,
    root: NodeId,
    x: VarId,
    base: &HashMap<VarId, f64>,
    num_x: usize,
) -> bool {
    let xs = linspace(LINEARITY_X_SAMPLE_LOW, LINEARITY_X_SAMPLE_HIGH, num_x);
    if xs.len() < 2 {
        return false;
    }
    let mut ys = Vec::with_capacity(num_x);
    for xv in &xs {
        let mut m = base.clone();
        m.insert(x, *xv);
        let y = match evaluate_f64(dag, root, &m) {
            Ok(y) if y.is_finite() => y,
            _ => return false,
        };
        ys.push(y);
    }
    affine_points_collinear(&xs, &ys)
}

/// Deterministic sample assignment for `vars` (for tests and call sites that need a fixed base).
pub fn assignment_seed(vars: &[VarId], seed: usize) -> HashMap<VarId, Q> {
    vars.iter()
        .enumerate()
        .map(|(i, v)| {
            let t = (seed + 1) as i64 * (i as i64 + 1);
            (*v, Q::new(t, 1))
        })
        .collect()
}
