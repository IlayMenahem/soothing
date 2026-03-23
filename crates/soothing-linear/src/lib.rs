//! Linear elimination by substitution for polynomial systems over `\mathbb{Q}`.
//!
//! This crate follows the specification in the research paper (DAG storage, sampling-based
//! predicates for zero and linearity, substitution map, iterative elimination). See `paper/`
//! in the repository root for the full write-up.
//!
//! ```
//! use soothing_linear::{linear_elimination, DagBuilder, PolynomialSystem, SampleConfig, VarId};
//!
//! let mut b = DagBuilder::new();
//! let x = VarId(0);
//! let y = VarId(1);
//! let vx = b.variable(x);
//! let vy = b.variable(y);
//! let eq = b.add(vec![vx, vy]);
//! let sys = PolynomialSystem::new(b.build(), vec![eq]);
//! let _ = linear_elimination(sys, &SampleConfig::default()).expect("ok");
//! ```
//!
//! ## Scope (v1)
//!
//! - **Field:** exact rationals [`field::Q`] only. Other fields are future work.
//! - **Predicates:** configurable sampling for “is zero” and “is linear in `x`” as in the paper.
//! - **Unsupported:** the “handling systems without linearities” section is not implemented;
//!   only the linear-elimination pipeline is provided.
//!
//! ## Implementation assumptions (beyond the paper)
//!
//! - Sample counts, [`predicates::RANDOM_SAMPLE_ABS_MAX`], [`predicates::LINEARITY_X_SAMPLE_LOW`]
//!   / [`predicates::LINEARITY_X_SAMPLE_HIGH`], and [`predicates::SampleConfig::random_seed`] control
//!   predicate sampling (aligned with `soothing/system_substitution.py` ranges).
//! - Elimination with recovered coefficients `a = 0` and `b \neq 0` is rejected as
//!   [`elimination::EliminationError::IllegitimateElimination`]; a substitution RHS that never yields a
//!   finite float at random samples is [`elimination::EliminationError::DegenerateSubstitution`].

pub mod dag;
mod dag_parse;
pub mod elimination;
pub mod mathematica_system;
pub mod eval;
pub mod field;
pub mod predicates;
pub mod simplify;
pub mod substitution;
pub mod system;

pub use dag::{Dag, DagBuilder, DagParseError, NodeId, VarId};
pub use elimination::{dfs_linear_elimination, linear_elimination, EliminationError, EliminationResult};
pub use mathematica_system::{
    default_systems_dir_from_manifest, load_mathematica_system_from_path, load_mathematica_system_from_str,
    MathematicaLoadError,
};
pub use eval::{evaluate, evaluate_f64, EvalError};
pub use field::Q;
pub use predicates::assignment_seed;
pub use predicates::{
    expr_finite_at_some_random_sample, is_linear_in, is_zero, recover_linear_coeffs, LinearCoeffs,
    PredicateError, SampleConfig, LINEARITY_X_SAMPLE_HIGH, LINEARITY_X_SAMPLE_LOW,
    RANDOM_Q_SAMPLE_ABS_MAX, RANDOM_SAMPLE_ABS_MAX, SUBSTITUTION_NUMERIC_CHECK_SAMPLES,
};
pub use simplify::simplify_once;
pub use substitution::SubstitutionMap;
pub use system::PolynomialSystem;
