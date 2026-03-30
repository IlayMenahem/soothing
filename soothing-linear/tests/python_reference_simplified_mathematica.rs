//! Parity with `python/scripts/solving_symbolic.py` on `systems/simplified_mathematica_system.txt`:
//! same DFS outcome as `best_linear_substitutions_order` — remaining equations, elimination count
//! (`n_initial - n_remaining`), free-variable count, and substitution size.

use serde::Deserialize;
use soothing_linear::{
    default_systems_dir_from_manifest, dfs_linear_elimination_solving_symbolic,
    load_mathematica_system_from_path_with_var_names, Dag, SampleConfig,
};
use std::collections::HashSet;
use std::path::PathBuf;
use std::process::Command;

#[derive(Debug, Deserialize, PartialEq, Eq)]
struct PythonEliminationStats {
    n_equations: usize,
    remaining_equations: usize,
    substitutions: usize,
    free_variables: usize,
    eliminations: usize,
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..")
}

fn python_reference_stats() -> PythonEliminationStats {
    let root = repo_root();
    let python_dir = root.join("python");
    let script = r#"
import json
import pathlib
from soothing.systems import load_mathematica_system
from soothing.linear_substitution_order import best_linear_substitutions_order
from soothing.parse_mathematica import mathematica_to_sympy

path = pathlib.Path("../systems/simplified_mathematica_system.txt").resolve()
text = load_mathematica_system(path)
polys, _ = mathematica_to_sympy(text)
cand, stats = best_linear_substitutions_order(polys)
free = cand.equation_symbols - cand.substitution_map.keys()
print(json.dumps({
    "n_equations": len(polys),
    "remaining_equations": len(cand.remaining_equations),
    "substitutions": len(cand.substitution_map),
    "free_variables": len(free),
    "eliminations": stats.eliminations,
}))
"#;

    let out = Command::new("uv")
        .current_dir(&python_dir)
        .args(["run", "python", "-c", script])
        .output()
        .unwrap_or_else(|e| panic!("failed to spawn `uv` in {}: {e}", python_dir.display()));

    assert!(
        out.status.success(),
        "uv run python failed:\nstderr={}\nstdout={}",
        String::from_utf8_lossy(&out.stderr),
        String::from_utf8_lossy(&out.stdout)
    );

    let line = String::from_utf8(out.stdout).expect("utf8");
    let line = line.trim();
    serde_json::from_str(line).unwrap_or_else(|e| panic!("bad json from python: {line:?}: {e}"))
}

fn rust_elimination_stats() -> PythonEliminationStats {
    let path = default_systems_dir_from_manifest().join("simplified_mathematica_system.txt");
    let loaded = load_mathematica_system_from_path_with_var_names(&path).expect("load system");

    let n_eq = loaded.system.equations.len();
    let mut equation_symbols: HashSet<soothing_linear::VarId> = HashSet::new();
    for &r in &loaded.system.equations {
        equation_symbols.extend(Dag::vars_in_root(&loaded.system.dag, r));
    }

    let cfg = SampleConfig::default();
    let res =
        dfs_linear_elimination_solving_symbolic(loaded, &cfg, None).expect("dfs");

    let remaining = res.reduced_system.equations.len();
    let substitutions = res.substitution.len();
    let free_variables = equation_symbols.len().saturating_sub(substitutions);
    let eliminations = n_eq - remaining;

    PythonEliminationStats {
        n_equations: n_eq,
        remaining_equations: remaining,
        substitutions,
        free_variables,
        eliminations,
    }
}

#[test]
fn dfs_matches_python_solving_symbolic_on_simplified_mathematica() {
    let py = python_reference_stats();
    let rust = rust_elimination_stats();

    assert_eq!(
        py, rust,
        "Rust DFS must match Python best_linear_substitutions_order summary stats \
         (see python/scripts/solving_symbolic.py)"
    );
}
