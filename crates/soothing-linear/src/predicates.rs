//! Sampling-based “is zero” and “is linear in `x`” predicates (paper).

use crate::dag::{Dag, NodeId, VarId};
use crate::eval::{evaluate, EvalError};
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
            linear_x_points: 4,
            random_seed: 0xE0B0_5EED_E0B0_5EED,
        }
    }
}

/// Absolute bound for uniform random integer samples in [`is_zero`] and [`is_linear_in`].
pub const RANDOM_Q_SAMPLE_ABS_MAX: i64 = 50_000;

/// Retries when a random sample hits division-by-zero (or similar) during evaluation.
const SAMPLE_EVAL_RETRIES: usize = 64;

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

fn random_assignment(vars: &[VarId], rng: &mut StdRng) -> HashMap<VarId, Q> {
    vars.iter()
        .map(|&v| {
            let n = rng.gen_range(-RANDOM_Q_SAMPLE_ABS_MAX..=RANDOM_Q_SAMPLE_ABS_MAX);
            (v, Q::new(n, 1))
        })
        .collect()
}

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
/// Samples each free variable uniformly at random as an integer in ± [`RANDOM_Q_SAMPLE_ABS_MAX`].
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
        let mut v = None;
        for _ in 0..SAMPLE_EVAL_RETRIES {
            let assignment = random_assignment(free_vars, &mut rng);
            match evaluate(dag, root, &assignment) {
                Ok(val) => {
                    v = Some(val);
                    break;
                }
                Err(EvalError::DivisionByZero) => continue,
                Err(e) => return Err(e.into()),
            }
        }
        let Some(v) = v else {
            return Err(PredicateError::DivisionByZero);
        };
        if v != Q::zero() {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Whether `p` is affine in `x` for random assignments to the other variables (each uniform
/// integer in ± [`RANDOM_Q_SAMPLE_ABS_MAX`]), then collinearity over `x = 0, 1, …`.
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
        let mut ok = None;
        for _ in 0..SAMPLE_EVAL_RETRIES {
            let base = random_assignment(other_vars, &mut rng);
            match points_on_one_affine_line(dag, root, x, &base, cfg.linear_x_points) {
                Ok(collinear) => {
                    ok = Some(collinear);
                    break;
                }
                Err(PredicateError::DivisionByZero) => continue,
                Err(e) => return Err(e),
            }
        }
        let Some(ok) = ok else {
            return Err(PredicateError::DivisionByZero);
        };
        if !ok {
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
    base: &HashMap<VarId, Q>,
    num_x: usize,
) -> Result<bool, PredicateError> {
    let xs: Vec<Q> = (0..num_x as i64).map(|k| Q::new(k, 1)).collect();
    let mut ys = Vec::with_capacity(num_x);
    for xv in &xs {
        let mut m = base.clone();
        m.insert(x, *xv);
        ys.push(evaluate(dag, root, &m)?);
    }
    if xs.len() < 2 {
        return Err(PredicateError::InsufficientSamples);
    }
    let slope = (ys[1] - ys[0]) / (xs[1] - xs[0]);
    for i in 1..xs.len() {
        let expected = ys[0] + slope * (xs[i] - xs[0]);
        if ys[i] != expected {
            return Ok(false);
        }
    }
    Ok(true)
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
