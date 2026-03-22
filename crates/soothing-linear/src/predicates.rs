//! Sampling-based “is zero” and “is linear in `x`” predicates (paper).

use crate::dag::{Dag, NodeId, VarId};
use crate::eval::{evaluate, EvalError};
use crate::field::Q;
use num_traits::{One, Zero};
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
}

impl Default for SampleConfig {
    fn default() -> Self {
        Self {
            zero_samples: 8,
            linear_other_assignments: 4,
            linear_x_points: 4,
        }
    }
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
pub fn is_zero(
    dag: &Dag,
    root: NodeId,
    free_vars: &[VarId],
    cfg: &SampleConfig,
) -> Result<bool, PredicateError> {
    if cfg.zero_samples == 0 {
        return Err(PredicateError::InsufficientSamples);
    }
    for s in 0..cfg.zero_samples {
        let assignment = assignment_seed(free_vars, s);
        let v = evaluate(dag, root, &assignment)?;
        if v != Q::zero() {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Whether `p` is affine in `x` for the sampled assignments to other variables.
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
    for s in 0..cfg.linear_other_assignments {
        let base = assignment_seed(other_vars, s);
        if !points_on_one_affine_line(dag, root, x, &base, cfg.linear_x_points)? {
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

/// Deterministic sample assignment for `vars` (implementation choice for sampling policy).
pub fn assignment_seed(vars: &[VarId], seed: usize) -> HashMap<VarId, Q> {
    vars.iter()
        .enumerate()
        .map(|(i, v)| {
            let t = (seed + 1) as i64 * (i as i64 + 1);
            (*v, Q::new(t, 1))
        })
        .collect()
}
