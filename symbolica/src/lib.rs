//! DFS linear elimination over Symbolica [`symbolica::atom::Atom`] expressions (standalone crate).

pub mod atom_util;
pub mod elimination;
pub mod elimination_exact;
pub mod exact;
pub mod factor_prep;
pub mod mathematica;
pub mod predicates;
pub mod substitution;
pub mod system;
pub mod var_id;

pub use elimination::{
    EliminationError, EliminationResult, default_is_solution, dfs_eliminate, greedy_eliminate,
    is_solution_nonzero_kappah,
};
pub use elimination_exact::{
    EliminationExactError, dfs_eliminate_exact, greedy_eliminate_exact,
    is_solution_nonzero_kappah_exact,
};
pub use exact::{
    ExactPredicateError, is_linear_in_exact, is_zero_exact, rhs_denominator_nonzero_exact,
};
pub use factor_prep::{
    ExtQ, PolyExtGrevLex, PolyExtLex, Preprocessed, RationalField, ideal_ext_grevlex,
    interreduce_polys, preprocess, variable_order,
};
pub use mathematica::{
    LoadedMathematicaSystem, default_systems_dir_from_manifest,
    load_mathematica_system_from_path_with_var_names,
    load_mathematica_system_from_str_with_var_names,
};
pub use predicates::{RANDOM_SAMPLE_ABS_MAX, SampleConfig};
pub use substitution::SubstitutionMap;
pub use system::PolynomialSystem;
pub use var_id::VarId;
