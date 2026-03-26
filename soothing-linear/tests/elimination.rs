use soothing_linear::{
    default_is_solution, dfs_linear_elimination, linear_elimination_greedy, Dag, DagBuilder,
    PolynomialSystem, SampleConfig, VarId, Q,
};

#[test]
fn single_step() {
    let mut b = DagBuilder::new();
    let x = VarId(0);
    let y = VarId(1);
    let vx = b.variable(x);
    let vy = b.variable(y);
    let e = b.add(vec![vx, vy]);
    let sys = PolynomialSystem::new(b.build(), vec![e]);
    let r = linear_elimination_greedy(sys, &SampleConfig::default()).unwrap();
    assert!(r.substitution.contains(x) || r.substitution.contains(y));
}

#[test]
fn recomputes_candidates() {
    let mut b = DagBuilder::new();
    let x = VarId(0);
    let y = VarId(1);
    let vx = b.variable(x);
    let vy = b.variable(y);
    let e1 = b.add(vec![vx, vy]);
    let vx2 = b.variable(x);
    let c = b.constant(Q::new(1, 1));
    let e2 = b.add(vec![vx2, c]);
    let sys = PolynomialSystem::new(b.build(), vec![e1, e2]);
    let r = linear_elimination_greedy(sys, &SampleConfig::default()).unwrap();
    assert!(r.substitution.len() >= 1);
}

#[test]
fn removes_zero_equations() {
    let mut b = DagBuilder::new();
    let x = VarId(0);
    let vx = b.variable(x);
    let nx = b.neg(vx);
    let z = b.add(vec![vx, nx]);
    let sys = PolynomialSystem::new(b.build(), vec![z]);
    let r = linear_elimination_greedy(sys, &SampleConfig::default()).unwrap();
    assert!(r.reduced_system.equations.is_empty());
}

#[test]
fn end_to_end() {
    let mut b = DagBuilder::new();
    let x = VarId(0);
    let y = VarId(1);
    let vx = b.variable(x);
    let vy = b.variable(y);
    let c = b.constant(Q::new(1, 1));
    let e = b.add(vec![vx, vy, c]);
    let sys = PolynomialSystem::new(b.build(), vec![e]);
    let r = linear_elimination_greedy(sys, &SampleConfig::default()).unwrap();
    assert!(r.substitution.len() >= 1);
}

#[test]
fn dfs_trace_writes_elapsed_lines() {
    let mut b = DagBuilder::new();
    let x = VarId(0);
    let vx = b.variable(x);
    let nx = b.neg(vx);
    let z = b.add(vec![vx, nx]);
    let sys = PolynomialSystem::new(b.build(), vec![z]);
    let mut log = Vec::<u8>::new();
    let _ = dfs_linear_elimination(
        sys,
        &SampleConfig::default(),
        Some(&mut log),
        default_is_solution,
    )
    .unwrap();
    let s = String::from_utf8(log).expect("utf8 trace");
    assert!(
        s.contains("elapsed:"),
        "expected DFS timing line, got: {s:?}"
    );
}

#[test]
fn dfs_removes_zero_equations() {
    let mut b = DagBuilder::new();
    let x = VarId(0);
    let vx = b.variable(x);
    let nx = b.neg(vx);
    let z = b.add(vec![vx, nx]);
    let sys = PolynomialSystem::new(b.build(), vec![z]);
    let r = dfs_linear_elimination(sys, &SampleConfig::default(), None, default_is_solution)
    .unwrap();
    assert!(r.reduced_system.equations.is_empty());
}

#[test]
fn dfs_not_worse_than_greedy_on_single_linear() {
    let mut b = DagBuilder::new();
    let x = VarId(0);
    let y = VarId(1);
    let vx = b.variable(x);
    let vy = b.variable(y);
    let e = b.add(vec![vx, vy]);
    let sys = PolynomialSystem::new(b.build(), vec![e]);
    let cfg = SampleConfig::default();
    let g = linear_elimination_greedy(sys.clone(), &cfg).unwrap();
    let d = dfs_linear_elimination(sys, &cfg, None, default_is_solution).unwrap();
    assert!(d.reduced_system.equations.len() <= g.reduced_system.equations.len());
}

/// Same nonlinear–linear mix as `dfs_demo`: multiple valid first elimination choices; DFS should
/// still reach a fully eliminated residual when one exists (early exit on empty \(Q\)).
#[test]
fn dfs_eliminates_fully_three_equation_demo_system() {
    let equation_strings = ["v1 + v0 - 1", "v2 - v0", "v1 * v2 + v0"];
    let mut b = DagBuilder::new();
    let mut equation_roots = Vec::with_capacity(equation_strings.len());
    for s in &equation_strings {
        let (dag, root) = Dag::from_rational_string(s).unwrap();
        equation_roots.push(b.merge_root(&dag, root));
    }
    let sys = PolynomialSystem::new(b.build(), equation_roots);
    let cfg = SampleConfig::default();
    let r = dfs_linear_elimination(sys, &cfg, None, default_is_solution).unwrap();
    assert!(
        r.reduced_system.equations.is_empty(),
        "expected full elimination for z+x-1=0, y-z=0, x*y+z=0"
    );
}
