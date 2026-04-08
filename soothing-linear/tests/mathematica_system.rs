#![allow(clippy::many_single_char_names)]

use std::collections::HashSet;
use std::path::Path;

use soothing_linear::{
    default_is_solution, default_systems_dir_from_manifest, dfs_linear_elimination,
    load_mathematica_system_from_path, load_mathematica_system_from_str, vars_in_root,
    SampleConfig,
};

#[test]
fn tiny_system_loads_and_dfs_runs() {
    let s = "{0 -> x + y, 0 -> x - 1}";
    let sys = load_mathematica_system_from_str(s).unwrap();
    assert_eq!(sys.equations.len(), 2);
    let _ =
        dfs_linear_elimination(&sys, &SampleConfig::default(), None, default_is_solution).unwrap();
}

#[test]
fn full_rhs_keeps_denominator_variables() {
    let s = "{0 -> (a + b)/(c + d)}";
    let sys = load_mathematica_system_from_str(s).unwrap();
    assert_eq!(sys.equations.len(), 1);
    let &root = sys.equations.first().unwrap();
    let vars = vars_in_root(&sys.egraph, root);
    let set: HashSet<_> = vars.into_iter().collect();
    assert_eq!(set.len(), 4, "a, b, c, d from (lhs)-(full rhs)");
}

#[test]
fn default_systems_dir_points_at_repo_systems() {
    let d = default_systems_dir_from_manifest();
    assert!(
        d.join("simplified_mathematica_system.txt").is_file(),
        "expected {}",
        d.display()
    );
}

#[test]
fn load_simplified_mathematica_file() {
    let p =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../systems/simplified_mathematica_system.txt");
    let sys =
        load_mathematica_system_from_path(&p).expect("parse simplified_mathematica_system.txt");
    assert_eq!(sys.equations.len(), 27);
}

#[test]
fn simplified_mathematica_dfs_does_not_panic() {
    let p =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../systems/simplified_mathematica_system.txt");
    let sys =
        load_mathematica_system_from_path(&p).expect("parse simplified_mathematica_system.txt");
    let _ = dfs_linear_elimination(&sys, &SampleConfig::default(), None, default_is_solution)
        .expect("dfs elimination");
}
