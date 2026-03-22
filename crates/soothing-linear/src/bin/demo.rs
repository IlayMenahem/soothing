//! Example binary: build a small polynomial system, run linear elimination, inspect the result.
//!
//! Run from the workspace root:
//! `cargo run -p soothing-linear --bin demo`

use std::collections::HashMap;

use soothing_linear::{
    assignment_seed, evaluate, linear_elimination, Dag, DagBuilder, NodeId, PolynomialSystem, Q,
    SampleConfig, VarId, simplify_once,
};

/// Fold local rewrites so [`Dag::format_root`] is readable (DAGs from elimination are not minimal).
fn simplify_for_display(dag: &Dag, root: NodeId) -> (Dag, NodeId) {
    let mut cur_dag = dag.clone();
    let mut cur_root = root;
    for _ in 0..16 {
        let mut b = DagBuilder::new();
        cur_root = simplify_once(&mut b, &cur_dag, cur_root);
        cur_dag = b.snapshot();
    }
    (cur_dag, cur_root)
}

fn main() {
    println!("soothing-linear demo\n");

    // Three equations in x, y, z. One is nonlinear: monomial x*y (degree 2, two variables in one
    // monomial), affine in z. Two are linear.
    // VarId order: z < x < y so elimination tries z first (see `linear_elimination`: BTreeSet over
    // VarId). Then z is removed from x + z - 1 with constant denominator before touching x*y + z.
    //   x + z - 1 = 0
    //   y - z = 0
    //   x*y + z = 0
    let mut b = DagBuilder::new();
    let z = VarId(0);
    let x = VarId(1);
    let y = VarId(2);
    let vz = b.variable(z);
    let vx = b.variable(x);
    let vy = b.variable(y);
    let one = b.constant(Q::new(1, 1));
    let neg_one = b.neg(one);
    let neg_vz = b.neg(vz);
    let xy = b.mul(vec![vx, vy]);
    let eq_xy_z = b.add(vec![xy, vz]);
    let eq_x_z = b.add(vec![vx, vz, neg_one]);
    let eq_y_z = b.add(vec![vy, neg_vz]);
    let sys = PolynomialSystem::new(b.build(), vec![eq_x_z, eq_y_z, eq_xy_z]);

    println!("Initial: three equations");
    println!("  x + z - 1 = 0");
    println!("  y - z = 0");
    println!("  x*y + z = 0   (monomial x*y: degree 2, two variables)");
    println!("DAG nodes: {}\n", sys.dag.len());

    let result = match linear_elimination(sys, &SampleConfig::default()) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("elimination failed: {e}");
            std::process::exit(1);
        }
    };

    let sub = &result.substitution;
    println!(
        "After linear elimination: {} substitution(s), {} remaining equation root(s)",
        sub.len(),
        result.system.equations.len()
    );

    let dag = sub.dag();
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
            println!("  eliminated {label} (VarId({}))", v.0);

            let (disp_dag, disp_root) = simplify_for_display(&dag, root);
            let expr = disp_dag.format_root(disp_root, &var_name);
            println!("    expression: {} = {}", label, expr);

            let free: Vec<VarId> = disp_dag.vars_in_root(disp_root);
            if free.is_empty() {
                let assign: HashMap<VarId, Q> = HashMap::new();
                match evaluate(&disp_dag, disp_root, &assign) {
                    Ok(q) => println!("    value (no free vars): {}", q),
                    Err(e) => println!("    (eval: {e:?})"),
                }
            } else {
                println!("    parametric evaluation (free vars: {}):", {
                    free.iter()
                        .map(|u| var_name(*u))
                        .collect::<Vec<_>>()
                        .join(", ")
                });
                for sample in 0..3 {
                    let assign = assignment_seed(&free, sample);
                    let binding: Vec<String> = free
                        .iter()
                        .map(|u| format!("{}={}", var_name(*u), assign.get(u).copied().unwrap()))
                        .collect();
                    match evaluate(&disp_dag, disp_root, &assign) {
                        Ok(q) => println!("      [{}]  {}", binding.join(", "), q),
                        Err(e) => println!("      (sample {}: {:?})", sample, e),
                    }
                }
            }
        }
    }

    println!("\nDone.");
}
