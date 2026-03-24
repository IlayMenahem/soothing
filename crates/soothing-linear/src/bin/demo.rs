//! Example binary: build a small polynomial system, run linear elimination, inspect the result.
//!
//! Run from the workspace root:
//! `cargo run -p soothing-linear --bin demo`

use std::collections::HashMap;

use soothing_linear::{
    assignment_seed, evaluate, linear_elimination_greedy, print_substitution_bindings,
    simplify_for_display, Dag, DagBuilder, PolynomialSystem, SampleConfig, VarId, Q,
};

fn main() {
    println!("soothing-linear — linear elimination demo\n");

    // Three equations in x, y, z. One is nonlinear: monomial x*y (degree 2, two variables in one
    // monomial), affine in z. Two are linear.
    // VarId order: z < x < y so elimination tries z first (see `linear_elimination_greedy`: BTreeSet over
    // VarId). Then z is removed from x + z - 1 with constant denominator before touching x*y + z.
    // Parser: `vN` is VarId(N); here v0=z, v1=x, v2=y.
    //   x + z - 1 = 0
    //   y - z = 0
    //   x*y + z = 0
    let z = VarId(0);
    let x = VarId(1);
    let y = VarId(2);
    let equation_strings = [
        "v1 + v0 - 1",  // x + z - 1
        "v2 - v0",      // y - z
        "v1 * v2 + v0", // x*y + z
    ];
    let mut b = DagBuilder::new();
    let mut equation_roots = Vec::with_capacity(equation_strings.len());
    for s in &equation_strings {
        let (dag, root) = match Dag::from_rational_string(s) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("parse equation {s:?}: {e}");
                std::process::exit(1);
            }
        };
        equation_roots.push(b.merge_root(&dag, root));
    }
    let sys = PolynomialSystem::new(b.build(), equation_roots);

    println!("Initial system:");
    println!("  x + z - 1 = 0");
    println!("  y - z = 0");
    println!("  x*y + z = 0   (monomial x*y: degree 2, two variables)");
    println!("DAG nodes: {}\n", sys.dag.len());

    let result = match linear_elimination_greedy(sys, &SampleConfig::default(), Default::default())
    {
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

    let dag = sub.dag();
    println!("Evaluation:");
    for &v in sub.insertion_order() {
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
            let (disp_dag, disp_root) = simplify_for_display(&dag, root);
            let free: Vec<VarId> = disp_dag.vars_in_root(disp_root);
            if free.is_empty() {
                let assign: HashMap<VarId, Q> = HashMap::new();
                match evaluate(&disp_dag, disp_root, &assign) {
                    Ok(q) => println!("  {}: value (no free vars): {}", label, q),
                    Err(e) => println!("  {}: (eval: {e:?})", label),
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
                for sample in 0..3 {
                    let assign = assignment_seed(&free, sample);
                    let binding: Vec<String> = free
                        .iter()
                        .map(|u| format!("{}={}", var_name(*u), assign.get(u).copied().unwrap()))
                        .collect();
                    match evaluate(&disp_dag, disp_root, &assign) {
                        Ok(q) => println!("    [{}]  {}", binding.join(", "), q),
                        Err(e) => println!("    (sample {}: {:?})", sample, e),
                    }
                }
            }
        }
    }

    println!("\nDone.");
}
