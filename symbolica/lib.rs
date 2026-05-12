//! Shared Symbolica helpers and system loading.

pub mod atom_util;
pub mod exact;
pub mod factor_prep;
pub mod mathematica;
pub mod milp_cover;
pub mod substitution;
pub mod system;
pub mod var_id;

pub use atom_util::*;
pub use exact::{
    ExactPredicateError, is_numerator_zero_exact, is_zero_exact, rhs_denominator_nonzero_exact,
};
pub use factor_prep::{
    ExtQ, PolyExtLex, Preprocessed, RationalField, ideal_ext_lex, interreduce_polys, preprocess,
    variable_order,
};
pub use mathematica::{
    LoadedMathematicaSystem, default_systems_dir_from_manifest,
    load_mathematica_system_from_path_with_var_names,
    load_mathematica_system_from_str_with_var_names,
};
pub use substitution::SubstitutionMap;
pub use system::PolynomialSystem;
pub use var_id::VarId;
