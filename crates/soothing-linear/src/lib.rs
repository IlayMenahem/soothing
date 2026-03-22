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
//! - Sample counts and point selection for predicates are chosen by the caller via
//!   [`predicates::SampleConfig`].
//! - Elimination with recovered coefficients `a = 0` and `b \neq 0` is rejected as
//!   [`elimination::EliminationError::IllegitimateElimination`].

pub mod dag;
pub mod elimination;
pub mod eval;
pub mod field;
pub mod predicates;
pub mod simplify;
pub mod substitution;
pub mod system;

pub use dag::{Dag, DagBuilder, NodeId, VarId};
pub use elimination::{linear_elimination, EliminationError, EliminationResult};
pub use eval::{evaluate, EvalError};
pub use predicates::assignment_seed;
pub use field::Q;
pub use predicates::{
    is_linear_in, is_zero, recover_linear_coeffs, LinearCoeffs, PredicateError, SampleConfig,
};
pub use simplify::simplify_once;
pub use substitution::SubstitutionMap;
pub use system::PolynomialSystem;
