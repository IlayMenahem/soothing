//! Human-readable console output for the cover result.

use crate::mathematica::LoadedMathematicaSystem;
use crate::milp_cover::milp::Cover;
use crate::milp_cover::problem::{ProblemData, var_name};

pub fn print_cover(loaded: &LoadedMathematicaSystem, data: &ProblemData, cover: &Cover) {
    println!("\nCover (in topological order):");
    for (rank, (fi, vid, t)) in cover.selected.iter().enumerate() {
        let info = &data.factors[*fi];
        println!(
            "  [{rank}] t={t:.2}  x = {:<16}\n        f = {}",
            var_name(loaded, *vid),
            info.poly
        );
    }
    println!(
        "\nPolynomials zeroed by zeroing alone: {}",
        cover.zeroed_polynomials.len()
    );
}

pub fn print_variable_sets(loaded: &LoadedMathematicaSystem, data: &ProblemData, cover: &Cover) {
    let dep: Vec<&str> = cover
        .selected
        .iter()
        .map(|&(_, vid, _)| var_name(loaded, vid))
        .collect();
    let zeroed: Vec<&str> = cover
        .zeroed_locals
        .iter()
        .map(|&i| var_name(loaded, data.v_ids[i]))
        .collect();
    let all_used: Vec<&str> = data
        .v_ids
        .iter()
        .map(|&vid| var_name(loaded, vid))
        .collect();
    let free: Vec<&str> = all_used
        .iter()
        .copied()
        .filter(|name| !dep.contains(name) && !zeroed.contains(name))
        .collect();

    println!("\nDependent variables X = {dep:?}");
    println!("Zeroed variables    W = {zeroed:?}");
    println!("Free parameters    Y = {free:?}");
    println!("(Variables not appearing in any factor are also free parameters unless zeroed.)");
}
