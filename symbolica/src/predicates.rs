//! Sampled zero and linearity predicates over [`Atom`] expressions.

use std::collections::HashMap;
use std::ops::{Mul, Sub};

use num_integer::Integer;
use num_rational::Ratio;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use symbolica::atom::{Atom, AtomCore, Symbol};

use crate::atom_util::{evaluate_atom, parse_identifier, substitute_symbol};
use crate::var_id::VarId;

pub type Q = Ratio<i64>;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SampleConfig {
    pub zero_samples: usize,
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

pub const RANDOM_SAMPLE_ABS_MAX: i64 = 32;
pub const SUBSTITUTION_NUMERIC_CHECK_SAMPLES: usize = 8;
const ZERO_TEST_ATTEMPT_MULTIPLIER: usize = 2;

fn mix_seed(base: u64, a: usize, b: u32) -> u64 {
    base ^ (a as u64).wrapping_mul(0x9E37_79B9_1E35_1B03)
        ^ u64::from(b).wrapping_mul(0xC6BC_2796_92B5_C323)
}

fn rng_for_zero(cfg: &SampleConfig, salt: usize) -> StdRng {
    StdRng::seed_from_u64(mix_seed(cfg.random_seed, salt, 0x5EED_5EED))
}

fn rng_for_linearity_q(cfg: &SampleConfig, salt: usize, x: VarId) -> StdRng {
    StdRng::seed_from_u64(mix_seed(cfg.random_seed, salt, x.0))
}

fn random_sample_q(rng: &mut StdRng) -> Q {
    let (numer, denom) = std::iter::repeat_with(|| {
        let n = rng.gen_range(-RANDOM_SAMPLE_ABS_MAX..=RANDOM_SAMPLE_ABS_MAX);
        let d = rng.gen_range(1..=RANDOM_SAMPLE_ABS_MAX);
        (n, d)
    })
    .find(|(n, d)| Integer::gcd(&n.abs(), d) == 1)
    .expect("coprime pair exists in bounded range");

    Ratio::new(numer, denom)
}

fn random_assignment_q(vars: &[VarId], rng: &mut StdRng) -> HashMap<VarId, Q> {
    vars.iter().map(|&v| (v, random_sample_q(rng))).collect()
}

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum PredicateError {
    #[error("insufficient samples for predicate")]
    InsufficientSamples,
    #[error("evaluation failed: {0}")]
    Eval(String),
}

fn probabilistic_zero_on_assignments(
    expr: &Atom,
    cfg: &SampleConfig,
    rng: &mut StdRng,
    mut next_assignment: impl FnMut(&mut StdRng) -> HashMap<VarId, Q>,
    id_to_sym: &[Symbol],
) -> Result<bool, PredicateError> {
    if cfg.zero_samples == 0 {
        return Err(PredicateError::InsufficientSamples);
    }
    let mut successes = 0;
    let max_attempts = cfg.zero_samples * ZERO_TEST_ATTEMPT_MULTIPLIER;

    for _ in 0..max_attempts {
        let assignment = next_assignment(rng);
        let v = match evaluate_atom(expr, &assignment, id_to_sym) {
            Ok(val) => val,
            Err(_) => continue,
        };
        if v != 0.0 && v.abs() > 1e-12 {
            return Ok(false);
        }
        successes += 1;
        if successes == cfg.zero_samples {
            return Ok(true);
        }
    }
    Ok(false)
}

pub fn is_zero(
    expr: &Atom,
    free_vars: &[VarId],
    cfg: &SampleConfig,
    id_to_sym: &[Symbol],
) -> Result<bool, PredicateError> {
    if cfg.zero_samples == 0 {
        return Err(PredicateError::InsufficientSamples);
    }
    let mut rng = rng_for_zero(cfg, expr.to_plain_string().len());
    probabilistic_zero_on_assignments(
        expr,
        cfg,
        &mut rng,
        |rng| random_assignment_q(free_vars, rng),
        id_to_sym,
    )
}

fn random_assignment_with_uvw(
    other_vars: &[VarId],
    u_vid: VarId,
    v_vid: VarId,
    w_vid: VarId,
    rng: &mut StdRng,
) -> HashMap<VarId, Q> {
    let mut m = random_assignment_q(other_vars, rng);
    m.insert(u_vid, random_sample_q(rng));
    m.insert(v_vid, random_sample_q(rng));
    m.insert(w_vid, random_sample_q(rng));
    m
}

/// Build auxiliary `q` for linearity in `x` (paper), then sample as [`is_zero`].
pub fn is_linear_in(
    expr: &Atom,
    x: VarId,
    other_vars: &[VarId],
    cfg: &SampleConfig,
    id_to_sym: &[Symbol],
) -> Result<bool, PredicateError> {
    if cfg.zero_samples == 0 {
        return Err(PredicateError::InsufficientSamples);
    }

    let x_sym = id_to_sym[x.0 as usize];
    let base = id_to_sym.len();
    let u_name = format!("___lin_u_{base}");
    let v_name = format!("___lin_v_{base}");
    let w_name = format!("___lin_w_{base}");

    let u_atom = parse_identifier(&u_name).map_err(PredicateError::Eval)?;
    let v_atom = parse_identifier(&v_name).map_err(PredicateError::Eval)?;
    let w_atom = parse_identifier(&w_name).map_err(PredicateError::Eval)?;

    let p_u = substitute_symbol(expr, x_sym, u_atom.clone());
    let p_v = substitute_symbol(expr, x_sym, v_atom.clone());
    let p_w = substitute_symbol(expr, x_sym, w_atom.clone());

    let p_wu = p_w.sub(&p_u);
    let p_vu = p_v.sub(&p_u);
    let v_minus_u = v_atom.clone().sub(&u_atom.clone());
    let w_minus_u = w_atom.clone().sub(&u_atom);
    let term1 = p_wu.mul(&v_minus_u);
    let term2 = p_vu.mul(&w_minus_u);
    let q = term1.sub(&term2).expand();

    let u_vid = VarId(base as u32);
    let v_vid = VarId(base as u32 + 1);
    let w_vid = VarId(base as u32 + 2);

    let mut extended_syms: Vec<Symbol> = id_to_sym.to_vec();
    extended_syms.push(
        u_atom
            .get_symbol()
            .ok_or(PredicateError::Eval(u_name))?,
    );
    extended_syms.push(
        v_atom
            .get_symbol()
            .ok_or(PredicateError::Eval(v_name))?,
    );
    extended_syms.push(
        w_atom
            .get_symbol()
            .ok_or(PredicateError::Eval(w_name))?,
    );

    let mut rng = rng_for_linearity_q(cfg, q.to_plain_string().len(), x);
    probabilistic_zero_on_assignments(
        &q,
        cfg,
        &mut rng,
        |rng| random_assignment_with_uvw(other_vars, u_vid, v_vid, w_vid, rng),
        &extended_syms,
    )
}

pub fn expr_finite_at_some_random_sample(
    expr: &Atom,
    free_vars: &[VarId],
    cfg: &SampleConfig,
    id_to_sym: &[Symbol],
) -> bool {
    let max_attempts = SUBSTITUTION_NUMERIC_CHECK_SAMPLES * ZERO_TEST_ATTEMPT_MULTIPLIER;
    (0..max_attempts).any(|attempt| {
        let mut rng = StdRng::seed_from_u64(
            cfg.random_seed
                .wrapping_add(attempt as u64)
                .wrapping_add(expr.to_plain_string().len() as u64),
        );
        let a = random_assignment_q(free_vars, &mut rng);
        evaluate_atom(expr, &a, id_to_sym).is_ok()
    })
}
