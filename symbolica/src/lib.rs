//! DFS linear elimination over Symbolica [`symbolica::atom::Atom`] expressions (standalone crate).

pub mod atom_util;
pub mod elimination;
pub mod mathematica;
pub mod predicates;
pub mod substitution;
pub mod system;
pub mod var_id;

pub use elimination::{
    default_is_solution, dfs_linear_elimination_solving_symbolic, is_solution_nonzero_kappah,
    EliminationError, EliminationResult,
};
pub use mathematica::{
    default_systems_dir_from_manifest, load_mathematica_system_from_path_with_var_names,
    LoadedMathematicaSystem,
};
pub use predicates::{SampleConfig, RANDOM_SAMPLE_ABS_MAX};
pub use substitution::SubstitutionMap;
pub use system::PolynomialSystem;
pub use var_id::VarId;
