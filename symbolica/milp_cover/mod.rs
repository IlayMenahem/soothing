//! MILP-based radical cover (see `problem statement.md`).
//!
//! Split into focused submodules:
//! - [`problem`]: data assembly from preprocessed factor groups.
//! - [`milp`]: `good_lp`/HiGHS model construction and solve.
//! - [`parameterization`]: symbolic substitution from a cover, and exact
//!   correctness checks against the original / preprocessed systems.
//! - [`report`]: human-readable console output.

pub mod certification;
pub mod milp;
pub mod parameterization;
pub mod problem;
pub mod report;

pub use certification::{
    CertificationError, CertificationReport, certify_cover_groebner, print_certification,
};
pub use milp::{Cover, solve_milp};
pub use parameterization::{
    ParameterizationError, ParameterizationReport, PreprocessedCheckReport,
    check_cover_parameterizes_original, check_cover_preprocessed, cover_substitution,
    expand_cover_substitution,
};
pub use problem::{FactorInfo, ProblemData, build_problem};
pub use report::{print_cover, print_variable_sets};
