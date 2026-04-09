//! Human-readable formatting for CLI binaries (substitution lines, simplified expressions).

use crate::dag::{Dag, NodeId, VarId};
use crate::simplify::simplify;
use crate::substitution::SubstitutionMap;

pub fn simplify_for_display(dag: &Dag, root: NodeId) -> (Dag, NodeId) {
    let mut builder = crate::dag::DagBuilder::new();
    let simplified_root = simplify(&mut builder, dag, root).expect("simplify display");
    (builder.build(), simplified_root)
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
