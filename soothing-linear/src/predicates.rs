//! Sampling-based “is zero” and “is linear in `x`” predicates on DAG expressions.

use crate::dag::{Dag, DagBuilder, NodeId, VarId};
use crate::eval::{evaluate, evaluate_f64, EvalError};
use crate::field::Q;
use num_traits::{One, Zero};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;
use thiserror::Error;

/// How many samples to use for predicate checks.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SampleConfig {
    /// Number of full assignments for the zero test and for the linearity-in-`x` test (via the
    /// auxiliary polynomial `q` in the paper).
    pub zero_samples: usize,
    /// Legacy field: not used by [`is_linear_in`] (kept for config compatibility).
    pub linear_other_assignments: usize,
    /// Legacy field: not used by [`is_linear_in`] (kept for config compatibility).
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

/// Samples for numeric finiteness check on a substitution RHS (like ruling out SymPy `nan`/`zoo`).
pub const SUBSTITUTION_NUMERIC_CHECK_SAMPLES: usize = 8;

fn mix_seed(base: u64, a: usize, b: u32) -> u64 {
    base ^ (a as u64).wrapping_mul(0x9E37_79B9_1E35_1B03)
        ^ (b as u64).wrapping_mul(0xC6BC_2796_92B5_C323)
}

fn rng_for_zero(cfg: &SampleConfig, root: NodeId) -> StdRng {
    StdRng::seed_from_u64(mix_seed(cfg.random_seed, root.0, 0x5EED_5EED))
}

fn rng_for_linearity_q(cfg: &SampleConfig, q_root: NodeId, x: VarId) -> StdRng {
    StdRng::seed_from_u64(mix_seed(cfg.random_seed, q_root.0, x.0))
}

fn random_assignment_f64(vars: &[VarId], rng: &mut StdRng) -> HashMap<VarId, f64> {
    vars.iter()
        .map(|&v| {
            let x: f64 = rng.gen_range(-RANDOM_SAMPLE_ABS_MAX..=RANDOM_SAMPLE_ABS_MAX);
            (v, x)
        })
        .collect()
}

fn max_var_id(dag: &Dag, root: NodeId, other_vars: &[VarId]) -> u32 {
    let mut m = other_vars.iter().map(|v| v.0).max().unwrap_or(0);
    for v in dag.vars_in_root(root) {
        m = m.max(v.0);
    }
    m
}

/// Builds \(q = (p(w)-p(u))(v-u) - (p(v)-p(u))(w-u)\) with fresh `u,v,w` symbols (see paper).
fn build_linearity_q_polynomial(
    dag: &Dag,
    root: NodeId,
    x: VarId,
    other_vars: &[VarId],
) -> (Dag, NodeId, VarId, VarId, VarId) {
    let base = max_var_id(dag, root, other_vars);
    let u_id = VarId(base + 1);
    let v_id = VarId(base + 2);
    let w_id = VarId(base + 3);
    let mut b = DagBuilder::new();
    let nu = b.variable(u_id);
    let nv = b.variable(v_id);
    let nw = b.variable(w_id);
    let p_u = b.substitute_in_root(dag, root, x, nu);
    let p_v = b.substitute_in_root(dag, root, x, nv);
    let p_w = b.substitute_in_root(dag, root, x, nw);
    let neg_p_u = b.neg(p_u);
    let p_w_minus_p_u = b.add(vec![p_w, neg_p_u]);
    let neg_p_u_b = b.neg(p_u);
    let p_v_minus_p_u = b.add(vec![p_v, neg_p_u_b]);
    let neg_nu = b.neg(nu);
    let v_minus_u = b.add(vec![nv, neg_nu]);
    let w_minus_u = b.add(vec![nw, neg_nu]);
    let term1 = b.mul(vec![p_w_minus_p_u, v_minus_u]);
    let term2 = b.mul(vec![p_v_minus_p_u, w_minus_u]);
    let neg_term2 = b.neg(term2);
    let q_root = b.add(vec![term1, neg_term2]);
    (b.build(), q_root, u_id, v_id, w_id)
}

/// If `v=u` or `w=u`, the auxiliary polynomial vanishes at that point for every `p`; resample `u,v,w`.
const MIN_LINEARITY_UV_SEP: f64 = 1e-6;
const UVW_RESAMPLE_CAP: u32 = 128;

fn random_assignment_with_nondegenerate_uvw(
    other_vars: &[VarId],
    u_id: VarId,
    v_id: VarId,
    w_id: VarId,
    rng: &mut StdRng,
) -> HashMap<VarId, f64> {
    let mut m = random_assignment_f64(other_vars, rng);
    for _ in 0..UVW_RESAMPLE_CAP {
        let u = rng.gen_range(-RANDOM_SAMPLE_ABS_MAX..=RANDOM_SAMPLE_ABS_MAX);
        let v = rng.gen_range(-RANDOM_SAMPLE_ABS_MAX..=RANDOM_SAMPLE_ABS_MAX);
        let w = rng.gen_range(-RANDOM_SAMPLE_ABS_MAX..=RANDOM_SAMPLE_ABS_MAX);
        if (v - u).abs() >= MIN_LINEARITY_UV_SEP && (w - u).abs() >= MIN_LINEARITY_UV_SEP {
            m.insert(u_id, u);
            m.insert(v_id, v);
            m.insert(w_id, w);
            return m;
        }
    }
    m.insert(u_id, 0.0);
    m.insert(v_id, 1.0);
    m.insert(w_id, -1.0);
    m
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

/// Absolute tolerance for reporting “zero” in [`is_zero`] (matches Python `_ZERO_TOL`).
const PREDICATE_ZERO_EPS: f64 = 1e-9;

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
    probabilistic_zero_on_assignments(dag, root, cfg, &mut rng, |rng| {
        random_assignment_f64(free_vars, rng)
    })
}

fn probabilistic_zero_on_assignments(
    dag: &Dag,
    root: NodeId,
    cfg: &SampleConfig,
    rng: &mut StdRng,
    mut next_assignment: impl FnMut(&mut StdRng) -> HashMap<VarId, f64>,
) -> Result<bool, PredicateError> {
    for _ in 0..cfg.zero_samples {
        let assignment = next_assignment(rng);
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

/// Whether `p` is affine in `x`: builds the paper’s auxiliary polynomial `q` and runs the same
/// probabilistic zero test as [`is_zero`] on `q` (see repository `paper/sections/substitution_map.tex`).
pub fn is_linear_in(
    dag: &Dag,
    root: NodeId,
    x: VarId,
    other_vars: &[VarId],
    cfg: &SampleConfig,
) -> Result<bool, PredicateError> {
    if cfg.zero_samples == 0 {
        return Err(PredicateError::InsufficientSamples);
    }
    let (q_dag, q_root, u_id, v_id, w_id) =
        build_linearity_q_polynomial(dag, root, x, other_vars);
    let mut rng = rng_for_linearity_q(cfg, q_root, x);
    probabilistic_zero_on_assignments(&q_dag, q_root, cfg, &mut rng, |rng| {
        random_assignment_with_nondegenerate_uvw(other_vars, u_id, v_id, w_id, rng)
    })
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
