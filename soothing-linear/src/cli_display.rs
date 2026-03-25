//! Human-readable formatting for CLI binaries (substitution lines, simplified expressions).

use crate::dag::{Dag, DagBuilder, NodeId, VarId};
use crate::simplify::simplify_once;
use crate::substitution::SubstitutionMap;

/// Fold local rewrites so [`Dag::format_root`] is readable (DAGs from elimination are not minimal).
pub fn simplify_for_display(dag: &Dag, root: NodeId) -> (Dag, NodeId) {
    let mut cur_dag = dag.clone();
    let mut cur_root = root;
    for _ in 0..16 {
        let mut b = DagBuilder::new();
        cur_root = simplify_once(&mut b, &cur_dag, cur_root).expect("simplify display");
        cur_dag = b.snapshot();
    }
    (cur_dag, cur_root)
}

/// Print one line per binding in insertion order: `  name = expr`.
pub fn print_substitution_bindings<F: Fn(VarId) -> String>(sub: &SubstitutionMap, var_name: &F) {
    let dag = sub.dag();
    for &v in sub.insertion_order() {
        if let Some(root) = sub.get(v) {
            let (disp_dag, disp_root) = simplify_for_display(&dag, root);
            let expr = disp_dag.format_root(disp_root, var_name);
            println!("  {} = {}", var_name(v), expr);
        }
    }
}
