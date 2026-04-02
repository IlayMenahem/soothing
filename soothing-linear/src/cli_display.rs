//! Human-readable formatting for CLI binaries (substitution lines, simplified expressions).

use crate::dag::{Dag, DagBuilder, NodeId, VarId};
use crate::simplify::simplify_once;
use crate::substitution::SubstitutionMap;

/// Fold local rewrites so [`Dag::format_root`] is readable (DAGs from elimination are not minimal).
///
/// # Panics
///
/// Panics if any fixed-point simplification step returns an unexpected [`crate::simplify::SimplifyError`].
#[must_use]
pub fn simplify_for_display(dag: &Dag, root: NodeId) -> (Dag, NodeId) {
    let mut cur_dag = dag.clone();
    let mut cur_root = root;
    (0..16).for_each(|_| {
        let mut b = DagBuilder::new();
        cur_root = simplify_once(&mut b, &cur_dag, cur_root).expect("simplify display");
        cur_dag = b.snapshot();
    });
    (cur_dag, cur_root)
}

/// Print one line per binding in insertion order: `  name = expr`.
pub fn print_substitution_bindings<F: Fn(VarId) -> String>(sub: &SubstitutionMap, var_name: &F) {
    let dag = sub.dag();
    sub.insertion_order().iter().copied().for_each(|v| {
        if let Some(root) = sub.get(v) {
            let (disp_dag, disp_root) = simplify_for_display(&dag, root);
            let expr = disp_dag.format_root(disp_root, var_name);
            println!("  {} = {}", var_name(v), expr);
        }
    });
}
