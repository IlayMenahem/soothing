//! Showcase [`soothing_linear::dfs_linear_elimination`]: explores all linear elimination branches
//! at each step (depth-first), returns early if the system is fully eliminated, otherwise the best
//! residual by equation count (tie-break: more free variables).
//!
//! Run from the workspace root:
//! `cargo run -p soothing-linear --bin dfs_demo`

use soothing_linear::{
    dfs_linear_elimination, linear_elimination_greedy, print_substitution_bindings, Dag,
    DagBuilder, PolynomialSystem, SampleConfig, SubstitutionMap, VarId,
};

fn var_name(vid: VarId, z: VarId, x: VarId, y: VarId) -> String {
    if vid == z {
        "z".to_string()
    } else if vid == x {
        "x".to_string()
    } else if vid == y {
        "y".to_string()
    } else {
        format!("v{}", vid.0)
    }
}

fn print_substitution_section(z: VarId, x: VarId, y: VarId, sub: &SubstitutionMap) {
    let vn = |v: VarId| var_name(v, z, x, y);
    println!("Substitutions:");
    print_substitution_bindings(sub, &vn);
}

fn run_three_equation_demo(cfg: &SampleConfig) {
    let z = VarId(0);
    let x = VarId(1);
    let y = VarId(2);
    let equation_strings = ["v1 + v0 - 1", "v2 - v0", "v1 * v2 + v0"];
    let mut b = DagBuilder::new();
    let mut equation_roots = Vec::with_capacity(equation_strings.len());
    for s in &equation_strings {
        let (dag, root) = Dag::from_rational_string(s).unwrap_or_else(|e| {
            eprintln!("parse equation {s:?}: {e}");
            std::process::exit(1);
        });
        equation_roots.push(b.merge_root(&dag, root));
    }
    let sys = PolynomialSystem::new(b.build(), equation_roots);

    println!("System (same as `demo` binary):");
    println!("  x + z - 1 = 0");
    println!("  y - z = 0");
    println!("  x*y + z = 0");
    println!();

    let greedy =
        linear_elimination_greedy(sys.clone(), cfg, Default::default()).unwrap_or_else(|e| {
            eprintln!("greedy elimination failed: {e}");
            std::process::exit(1);
        });
    let dfs = dfs_linear_elimination(sys, cfg, Default::default(), None).unwrap_or_else(|e| {
        eprintln!("DFS elimination failed: {e}");
        std::process::exit(1);
    });

    println!(
        "Greedy linear elimination: {} substitution(s), {} remaining equation(s)",
        greedy.substitution.len(),
        greedy.reduced_system.equations.len()
    );
    print_substitution_section(z, x, y, &greedy.substitution);
    println!();

    println!(
        "DFS linear elimination: {} substitution(s), {} remaining equation(s)",
        dfs.substitution.len(),
        dfs.reduced_system.equations.len()
    );
    print_substitution_section(z, x, y, &dfs.substitution);

    if dfs.reduced_system.equations.len() < greedy.reduced_system.equations.len() {
        println!();
        println!("On this system, DFS left fewer residual equations than the greedy pass.");
    }
    println!();
}

fn run_two_linear_equations_demo(cfg: &SampleConfig) {
    println!("---");
    println!("Smaller example: two affine equations in x, y (fully linear).");
    println!("  x + y - 1 = 0");
    println!("  x - y + 1 = 0");
    println!();

    let x = VarId(0);
    let y = VarId(1);
    let mut b = DagBuilder::new();
    let e0 = {
        let (d, r) = Dag::from_rational_string("v0 + v1 - 1").unwrap_or_else(|e| {
            eprintln!("parse: {e}");
            std::process::exit(1);
        });
        b.merge_root(&d, r)
    };
    let e1 = {
        let (d, r) = Dag::from_rational_string("v0 - v1 + 1").unwrap_or_else(|e| {
            eprintln!("parse: {e}");
            std::process::exit(1);
        });
        b.merge_root(&d, r)
    };
    let sys = PolynomialSystem::new(b.build(), vec![e0, e1]);

    let dfs = dfs_linear_elimination(sys, cfg, Default::default(), None).unwrap_or_else(|e| {
        eprintln!("DFS elimination failed: {e}");
        std::process::exit(1);
    });
    println!(
        "DFS linear elimination: {} substitution(s), {} remaining equation(s)",
        dfs.substitution.len(),
        dfs.reduced_system.equations.len()
    );
    let vn = |v: VarId| {
        if v == x {
            "x".to_string()
        } else if v == y {
            "y".to_string()
        } else {
            format!("v{}", v.0)
        }
    };
    println!("Substitutions:");
    print_substitution_bindings(&dfs.substitution, &vn);
    println!();
}

fn main() {
    println!("soothing-linear — DFS linear elimination showcase\n");
    let cfg = SampleConfig::default();

    println!(
        "DFS at each step enumerates every (equation, variable) pair that is linear, recurses, \
         and keeps the best residual (fewest equations; ties: more distinct variables in the residual). \
         If any branch eliminates all equations, that result is returned immediately.\n"
    );

    run_three_equation_demo(&cfg);
    run_two_linear_equations_demo(&cfg);

    println!("Done.");
}
