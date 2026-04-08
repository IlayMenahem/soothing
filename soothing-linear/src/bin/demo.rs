//! Example binary: build a small polynomial system, run linear elimination, inspect the result.
//!
//! Run from the workspace root:
//! `cargo run -p soothing-linear --bin demo`

use std::collections::HashMap;

use soothing_linear::{
    assignment_seed, evaluate, from_rational_string, linear_elimination_greedy,
    print_substitution_bindings, simplify_for_display, vars_in_root, ExprBuilder, PolynomialSystem,
    SampleConfig, VarId, Q,
};

#[allow(clippy::many_single_char_names)]
fn main() {
    println!("soothing-linear — linear elimination demo\n");

    let z = VarId(0);
    let x = VarId(1);
    let y = VarId(2);
    let equation_strings = [
        "v1 + v0 - 1",  // x + z - 1
        "v2 - v0",      // y - z
        "v1 * v2 + v0", // x*y + z
    ];
    let mut b = ExprBuilder::new();
    let equation_roots: Vec<_> = equation_strings
        .iter()
        .map(|s| match from_rational_string(s) {
            Ok(p) => Ok(b.merge_root(&p.0, p.1)),
            Err(e) => {
                eprintln!("parse equation {s:?}: {e}");
                Err(())
            }
        })
        .collect::<Result<_, _>>()
        .unwrap_or_else(|()| std::process::exit(1));
    let sys = PolynomialSystem::new(b.into_egraph(), equation_roots);

    println!("Initial system:");
    println!("  x + z - 1 = 0");
    println!("  y - z = 0");
    println!("  x*y + z = 0   (monomial x*y: degree 2, two variables)");
    println!("E-graph nodes: {}\n", sys.egraph.total_number_of_nodes());

    let result = match linear_elimination_greedy(&sys, &SampleConfig::default()) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("elimination failed: {e}");
            std::process::exit(1);
        }
    };

    let sub = &result.substitution;
    println!(
        "After linear elimination: {} substitution(s), {} remaining equation(s)",
        sub.len(),
        result.reduced_system.equations.len()
    );
    println!("Substitutions:");
    let var_name = |vid: VarId| {
        if vid == z {
            "z".to_string()
        } else if vid == x {
            "x".to_string()
        } else if vid == y {
            "y".to_string()
        } else {
            format!("v{}", vid.0)
        }
    };
    print_substitution_bindings(sub, &var_name);

    let egraph = sub.egraph();
    println!("Evaluation:");
    sub.insertion_order().iter().copied().for_each(|v| {
        if let Some(root) = sub.get(v) {
            let label = if v == x {
                "x"
            } else if v == y {
                "y"
            } else if v == z {
                "z"
            } else {
                "?"
            };
            let (disp_eg, disp_root) = simplify_for_display(&egraph, root);
            let free: Vec<VarId> = vars_in_root(&disp_eg, disp_root);
            if free.is_empty() {
                let assign: HashMap<VarId, Q> = HashMap::new();
                match evaluate(&disp_eg, disp_root, &assign) {
                    Ok(q) => println!("  {label}: value (no free vars): {q}"),
                    Err(e) => println!("  {label}: (eval: {e:?})"),
                }
            } else {
                println!(
                    "  {}: parametric (free: {})",
                    label,
                    free.iter()
                        .map(|u| var_name(*u))
                        .collect::<Vec<_>>()
                        .join(", ")
                );
                (0..3).for_each(|sample| {
                    let assign = assignment_seed(&free, sample);
                    let binding: Vec<String> = free
                        .iter()
                        .map(|u| format!("{}={}", var_name(*u), assign.get(u).cloned().unwrap()))
                        .collect();
                    match evaluate(&disp_eg, disp_root, &assign) {
                        Ok(q) => println!("    [{}]  {}", binding.join(", "), q),
                        Err(e) => println!("    (sample {sample}: {e:?})"),
                    }
                });
            }
        }
    });

    println!("\nDone.");
}
