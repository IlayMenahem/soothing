//! Linear elimination by substitution for polynomial systems over `\mathbb{Q}`.
//!
//! E-graph–stored expressions (`egg`), exact rational evaluation, sampled zero/linearity checks,
//! substitution maps, and greedy or DFS elimination.
//!
//! ```
//! use soothing_linear::{linear_elimination_greedy, ExprBuilder, PolynomialSystem, SampleConfig, VarId};
//! use egg::Id;
//!
//! let mut b = ExprBuilder::new();
//! let x = VarId(0);
//! let y = VarId(1);
//! let vx = b.variable(x);
//! let vy = b.variable(y);
//! let eq = b.add(vec![vx, vy]);
//! let sys = PolynomialSystem::new(b.into_egraph(), vec![eq]);
//! let _ = linear_elimination_greedy(&sys, &SampleConfig::default()).expect("ok");
//! ```
//!
//! ## Scope (v1)
//!
//! - **Field:** exact rationals [`field::Q`] only. Other fields are future work.
//! - **Predicates:** configurable sampled checks for “is zero” and “is linear in `x`”, evaluated
//!   exactly over [`field::Q`].
//! - **Unsupported:** no heuristic for systems with no linear eliminations (only the
//!   linear-elimination pipeline).
//!
//! ## Implementation assumptions
//!
//! - Sample counts, [`RANDOM_SAMPLE_ABS_MAX`], and [`SampleConfig::random_seed`] control exact
//!   rational predicate sampling.
//! - Elimination with recovered coefficients `a = 0` and `b \neq 0` is rejected as
//!   [`elimination::EliminationError::IllegitimateElimination`]; a substitution RHS that never yields a
//!   well-defined exact value at random samples is
//!   [`elimination::EliminationError::DegenerateSubstitution`].

pub mod cli_display;
mod dag_parse;
pub mod elimination;
pub mod eval;
pub mod expr_build;
pub mod expr_lang;
pub mod field;
pub mod mathematica_system;
pub mod predicates;
pub mod simplify;
pub mod substitution;
pub mod system;

pub use egg::Id;
pub use elimination::{
    default_is_solution, dfs_linear_elimination, dfs_linear_elimination_solving_symbolic,
    is_solution_nonzero_kappah, linear_elimination_greedy, EliminationError, EliminationResult,
};
pub use eval::{evaluate, EvalError};
pub use expr_build::ExprBuilder;
pub use expr_lang::{
    format_root, from_rational_string, from_rational_string_with_var_map, vars_in_root,
    DagParseError, RationalLang, VarId,
};
pub use field::{q, Q};
pub use mathematica_system::{
    default_systems_dir_from_manifest, load_mathematica_system_from_path,
    load_mathematica_system_from_path_with_var_names, load_mathematica_system_from_str,
    load_mathematica_system_from_str_with_var_names, LoadedMathematicaSystem, MathematicaLoadError,
};
pub use predicates::assignment_seed;
pub use predicates::{
    expr_finite_at_some_random_sample, is_linear_in, is_zero, recover_linear_coeffs, LinearCoeffs,
    PredicateError, SampleConfig, RANDOM_SAMPLE_ABS_MAX, SUBSTITUTION_NUMERIC_CHECK_SAMPLES,
};

pub use cli_display::{print_substitution_bindings, simplify_for_display};
pub use predicates::is_linear_in as is_linear_in_variable;
pub use predicates::is_zero as is_polynomial_zero;
pub use predicates::recover_linear_coeffs as recover_linear_coefficients;
pub use simplify::{simplify, SimplifyError};
pub use substitution::SubstitutionMap;
pub use system::PolynomialSystem;
