//! Shared Symbolica helpers and system loading.

pub mod atom_util;
pub mod exact;
pub mod factor_prep;
pub mod groebner;
pub mod mathematica;
pub mod milp_cover;
pub mod sig_groebner;
pub mod substitution;
pub mod system;
pub mod var_id;

pub use atom_util::{parse_expression, substitute_symbol};
pub use exact::{
    ExactPredicateError, is_numerator_zero_exact, is_zero_exact, rhs_denominator_nonzero_exact,
};
pub use factor_prep::{PolyExtLex, ideal_ext_lex, interreduce_polys, preprocess, variable_order};
pub use groebner::{QPolyLex, groebner_basis_from_system_lex_q, groebner_basis_lex_q, ideal_lex_q};
pub use mathematica::{
    LoadedMathematicaSystem, default_systems_dir_from_manifest,
    load_mathematica_system_from_path_with_var_names,
    load_mathematica_system_from_str_with_var_names,
};
pub use sig_groebner::{
    Signature, groebner_f5, groebner_f5_from_system, groebner_f5_from_system_with, groebner_gvw,
    groebner_gvw_from_system, groebner_gvw_from_system_with, ideal_in_order,
};
pub use substitution::SubstitutionMap;
pub use system::PolynomialSystem;
pub use var_id::VarId;
