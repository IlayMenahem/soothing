//! Shared Symbolica helpers and system loading.

pub mod atom_util;
pub mod cover;
pub mod factor_prep;
pub mod gb_core;
pub mod gb_search;
pub mod m4gb;
pub mod mathematica;
pub(crate) mod monom;
pub mod refine;
pub mod sig_groebner;
#[cfg_attr(test, allow(dead_code))]
mod solver_process;
pub mod sparse;
pub mod system;
pub mod var_id;
pub mod variable_zeroing;

pub use atom_util::{parse_expression, substitute_symbol};
pub use factor_prep::{
    PolyExtLex, factor_groups_from_polys, ideal_ext_lex, interreduce_polys,
    polynomials_from_loaded, preprocess, reorder_by_min_fill, try_factor_groups_from_polys,
    variable_order,
};
pub use m4gb::{groebner_m4gb, groebner_m4gb_from_system, groebner_m4gb_from_system_with};
pub use mathematica::{
    LoadedMathematicaSystem, default_systems_dir_from_manifest,
    load_mathematica_system_from_path_with_var_names,
    load_mathematica_system_from_str_with_var_names,
};
pub use sig_groebner::{
    Signature, groebner_f5, groebner_f5_from_system, groebner_f5_from_system_with, groebner_gvw,
    groebner_gvw_from_system, groebner_gvw_from_system_with, ideal_in_order,
};
pub use solver_process::exit_if_solver_worker;
pub use system::PolynomialSystem;
pub use var_id::VarId;
pub use variable_zeroing::{
    PreprocessingFailure, ProjectedComponent, SplitDecision, SplitValueStrategy, UnresolvedBranch,
    UnresolvedReason, VariableZeroingCoverOutcome, VariableZeroingCoverResult,
    VariableZeroingNodeResult, VariableZeroingOptions, VariableZeroingResult, ZeroingVariable,
    solve_by_zeroing,
};
