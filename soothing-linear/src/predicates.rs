//! Sampled “is zero” and “is linear in `x`” predicates on e-graph expressions, evaluated exactly in
//! [`Q`].
//!
//! For polynomial inputs, these predicates implement the paper's evaluation-based tests. The public
//! APIs also accept rational expressions (division / powers) by skipping sampled assignments that
//! hit a division-by-zero pole.

use crate::eval::{evaluate, EvalError};
use crate::expr_build::ExprBuilder;
use crate::expr_lang::{vars_in_root, RationalLang, VarId};
use crate::field::Q;
use egg::{EGraph, Id};
use num_integer::Integer;
use num_traits::{One, Zero};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::collections::HashMap;
use std::hash::BuildHasher;
use thiserror::Error;

/// How many samples to use for predicate checks.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SampleConfig {
    /// Number of full assignments for the zero test and for the linearity-in-`x` test (via the
    /// auxiliary polynomial `q` in the paper).
    pub zero_samples: usize,
    /// Seed for [`StdRng`] when drawing random values for free / “other” variables (mixed with
    /// expression identity so repeated calls are not identical).
    pub random_seed: u64,
}

impl Default for SampleConfig {
    fn default() -> Self {
        Self {
            zero_samples: 8,
            random_seed: 0xE0B0_5EED_E0B0_5EED,
        }
    }
}

/// Half-width of the bounded rational sample set for exact samples in [`is_zero`] and
/// [`is_linear_in`] “other” variables.
pub const RANDOM_SAMPLE_ABS_MAX: i64 = 32;

/// Samples for exact evaluability checks on a substitution RHS.
pub const SUBSTITUTION_NUMERIC_CHECK_SAMPLES: usize = 8;
const ZERO_TEST_ATTEMPT_MULTIPLIER: usize = 2;

fn mix_seed(base: u64, a: usize, b: u32) -> u64 {
    base ^ (a as u64).wrapping_mul(0x9E37_79B9_1E35_1B03)
        ^ u64::from(b).wrapping_mul(0xC6BC_2796_92B5_C323)
}

fn rng_for_zero(cfg: &SampleConfig, root: Id) -> StdRng {
    StdRng::seed_from_u64(mix_seed(cfg.random_seed, usize::from(root), 0x5EED_5EED))
}

fn rng_for_linearity_q(cfg: &SampleConfig, q_root: Id, x: VarId) -> StdRng {
    StdRng::seed_from_u64(mix_seed(cfg.random_seed, usize::from(q_root), x.0))
}

fn random_sample_q(rng: &mut StdRng) -> Q {
    let (numer, denom) = std::iter::repeat_with(|| {
        (
            rng.gen_range(-RANDOM_SAMPLE_ABS_MAX..=RANDOM_SAMPLE_ABS_MAX),
            rng.gen_range(1..=RANDOM_SAMPLE_ABS_MAX),
        )
    })
    .find(|(n, d)| n.abs().gcd(d) == 1)
    .expect("coprime pair exists in bounded range");

    Q::new(numer.into(), denom.into())
}

fn random_assignment_q(vars: &[VarId], rng: &mut StdRng) -> HashMap<VarId, Q> {
    vars.iter().map(|&v| (v, random_sample_q(rng))).collect()
}

fn max_var_id(egraph: &EGraph<RationalLang, ()>, root: Id, other_vars: &[VarId]) -> u32 {
    other_vars
        .iter()
        .map(|v| v.0)
        .chain(vars_in_root(egraph, root).into_iter().map(|v| v.0))
        .max()
        .unwrap_or(0)
}

/// Builds \(q = (p(w)-p(u))(v-u) - (p(v)-p(u))(w-u)\) with fresh `u,v,w` symbols (see paper).
#[allow(clippy::similar_names)]
fn build_linearity_q_polynomial(
    egraph: &EGraph<RationalLang, ()>,
    root: Id,
    x: VarId,
    other_vars: &[VarId],
) -> (EGraph<RationalLang, ()>, Id, VarId, VarId, VarId) {
    let base = max_var_id(egraph, root, other_vars);
    let u_id = VarId(base + 1);
    let v_id = VarId(base + 2);
    let w_id = VarId(base + 3);
    let mut b = ExprBuilder::new();
    let nu = b.variable(u_id);
    let nv = b.variable(v_id);
    let nw = b.variable(w_id);
    let p_u = b.substitute_in_root(egraph, root, x, nu);
    let p_v = b.substitute_in_root(egraph, root, x, nv);
    let p_w = b.substitute_in_root(egraph, root, x, nw);
    let p_wu = b.sub(p_w, p_u);
    let p_vu = b.sub(p_v, p_u);
    let v_minus_u = b.sub(nv, nu);
    let w_minus_u = b.sub(nw, nu);
    let term1 = b.mul2(p_wu, v_minus_u);
    let term2 = b.mul2(p_vu, w_minus_u);
    let q_root = b.sub(term1, term2);
    (b.into_egraph(), q_root, u_id, v_id, w_id)
}

fn random_assignment_with_uvw(
    other_vars: &[VarId],
    u_id: VarId,
    v_id: VarId,
    w_id: VarId,
    rng: &mut StdRng,
) -> HashMap<VarId, Q> {
    let mut m = random_assignment_q(other_vars, rng);

    m.insert(u_id, random_sample_q(rng));
    m.insert(v_id, random_sample_q(rng));
    m.insert(w_id, random_sample_q(rng));

    m
}

/// True if at least one random exact assignment yields a well-defined [`evaluate`] value.
#[must_use]
pub fn expr_finite_at_some_random_sample(
    egraph: &EGraph<RationalLang, ()>,
    root: Id,
    free_vars: &[VarId],
    cfg: &SampleConfig,
) -> bool {
    let max_attempts = SUBSTITUTION_NUMERIC_CHECK_SAMPLES * ZERO_TEST_ATTEMPT_MULTIPLIER;
    (0..max_attempts).into_par_iter().any(|attempt| {
        let mut rng = StdRng::seed_from_u64(mix_seed(
            cfg.random_seed.wrapping_add(attempt as u64),
            usize::from(root),
            u32::try_from(attempt).unwrap(),
        ));
        let a = random_assignment_q(free_vars, &mut rng);
        evaluate(egraph, root, &a).is_ok()
    })
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
    egraph: &EGraph<RationalLang, ()>,
    root: Id,
    free_vars: &[VarId],
    cfg: &SampleConfig,
) -> Result<bool, PredicateError> {
    if cfg.zero_samples == 0 {
        return Err(PredicateError::InsufficientSamples);
    }
    let mut rng = rng_for_zero(cfg, root);
    probabilistic_zero_on_assignments(egraph, root, cfg, &mut rng, |rng| {
        random_assignment_q(free_vars, rng)
    })
}

fn probabilistic_zero_on_assignments(
    egraph: &EGraph<RationalLang, ()>,
    root: Id,
    cfg: &SampleConfig,
    rng: &mut StdRng,
    mut next_assignment: impl FnMut(&mut StdRng) -> HashMap<crate::expr_lang::VarId, Q>,
) -> Result<bool, PredicateError> {
    let mut successes = 0;
    let max_attempts = cfg.zero_samples * ZERO_TEST_ATTEMPT_MULTIPLIER;

    for _ in 0..max_attempts {
        let assignment = next_assignment(rng);
        let v = match evaluate(egraph, root, &assignment) {
            Ok(val) => val,
            Err(EvalError::DivisionByZero) => continue,
            Err(e) => return Err(e.into()),
        };
        if !v.is_zero() {
            return Ok(false);
        }
        successes += 1;
        if successes == cfg.zero_samples {
            return Ok(true);
        }
    }
    Ok(false)
}

pub fn is_linear_in(
    egraph: &EGraph<RationalLang, ()>,
    root: Id,
    x: VarId,
    other_vars: &[VarId],
    cfg: &SampleConfig,
) -> Result<bool, PredicateError> {
    if cfg.zero_samples == 0 {
        return Err(PredicateError::InsufficientSamples);
    }

    let (q_egraph, q_root, u_id, v_id, w_id) =
        build_linearity_q_polynomial(egraph, root, x, other_vars);
    let mut rng = rng_for_linearity_q(cfg, q_root, x);
    probabilistic_zero_on_assignments(&q_egraph, q_root, cfg, &mut rng, |rng| {
        random_assignment_with_uvw(other_vars, u_id, v_id, w_id, rng)
    })
}

pub fn recover_linear_coeffs<S: BuildHasher + Clone>(
    egraph: &EGraph<RationalLang, ()>,
    root: Id,
    x: VarId,
    base: &HashMap<VarId, Q, S>,
) -> Result<LinearCoeffs, PredicateError> {
    let mut a0 = base.clone();
    let mut a1 = base.clone();

    a0.insert(x, Q::zero());
    a1.insert(x, Q::one());

    let b = evaluate(egraph, root, &a0)?;
    let at_one = evaluate(egraph, root, &a1)?;
    let a = at_one - b.clone();

    Ok(LinearCoeffs { a, b })
}

#[derive(Clone, Debug, PartialEq)]
pub struct LinearCoeffs {
    pub a: Q,
    pub b: Q,
}

/// Deterministic sample assignment for `vars` (for tests and call sites that need a fixed base).
#[must_use]
pub fn assignment_seed(vars: &[VarId], seed: usize) -> HashMap<VarId, Q> {
    vars.par_iter()
        .enumerate()
        .map(|(i, v)| {
            let t = (seed + 1) as i64 * (i as i64 + 1);
            (*v, Q::new(t.into(), 1.into()))
        })
        .collect()
}
