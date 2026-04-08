//! Human-readable formatting for CLI binaries (substitution lines, simplified expressions).

use crate::expr_lang::{format_root, RationalLang, VarId};
use crate::simplify::simplify;
use crate::substitution::SubstitutionMap;
use egg::{EGraph, Id};

pub fn simplify_for_display(
    egraph: &EGraph<RationalLang, ()>,
    root: Id,
) -> (EGraph<RationalLang, ()>, Id) {
    let mut out = EGraph::default();
    let simplified_root = simplify(&mut out, egraph, root).expect("simplify display");
    (out, simplified_root)
}

/// Print one line per binding in insertion order: `  name = expr`.
pub fn print_substitution_bindings<F: Fn(VarId) -> String>(sub: &SubstitutionMap, var_name: &F) {
    let egraph = sub.egraph();
    sub.insertion_order().iter().copied().for_each(|v| {
        if let Some(root) = sub.get(v) {
            let (disp_eg, disp_root) = simplify_for_display(&egraph, root);
            let expr = format_root(&disp_eg, disp_root, var_name);
            println!("  {} = {}", var_name(v), expr);
        }
    });
}
