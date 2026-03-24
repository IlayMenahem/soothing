use std::collections::HashSet;
use std::path::Path;

use soothing_linear::{
    dfs_linear_elimination, default_systems_dir_from_manifest, load_mathematica_system_from_path,
    load_mathematica_system_from_str, Dag, SampleConfig,
};

#[test]
fn tiny_system_loads_and_dfs_runs() {
    let s = "{0 -> x + y, 0 -> x - 1}";
    let sys = load_mathematica_system_from_str(s).unwrap();
    assert_eq!(sys.equations.len(), 2);
    let _ = dfs_linear_elimination(sys, &SampleConfig::default(), Default::default()).unwrap();
}

#[test]
fn numerator_only_drops_denominator_variables() {
    let s = "{0 -> (a + b)/(c + d)}";
    let sys = load_mathematica_system_from_str(s).unwrap();
    assert_eq!(sys.equations.len(), 1);
    let &root = sys.equations.first().unwrap();
    let vars = Dag::vars_in_root(&sys.dag, root);
    let set: HashSet<_> = vars.into_iter().collect();
    assert_eq!(set.len(), 2, "only a and b from numerator");
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
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../systems/simplified_mathematica_system.txt");
    let sys = load_mathematica_system_from_path(&p).expect("parse simplified_mathematica_system.txt");
    assert_eq!(sys.equations.len(), 27);
}
