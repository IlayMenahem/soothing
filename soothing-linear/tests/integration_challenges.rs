//! Heavier scenarios aligned with `tests/` Python coverage: multi-step elimination,
//! partial linear systems, residuals, and DAG utilities.

#![allow(clippy::many_single_char_names)]

use num_traits::Zero;

use soothing_linear::{
    evaluate, linear_elimination_greedy, q, Dag, DagBuilder, EliminationError, PolynomialSystem,
    SampleConfig, VarId, Q,
};
use std::collections::HashMap;

fn default_cfg() -> SampleConfig {
    SampleConfig {
        zero_samples: 12,
        ..Default::default()
    }
}

#[test]
fn empty_system_no_work() {
    let mut b = DagBuilder::new();
    let _ = b.constant(q(0, 1));
    let d = b.build();
    let sys = PolynomialSystem::new(d, vec![]);
    let r = linear_elimination_greedy(&sys, &default_cfg()).unwrap();
    assert!(r.reduced_system.equations.is_empty());
    assert!(r.substitution.is_empty());
}

#[test]
fn four_variable_linear_chain_eliminates_all() {
    let mut b = DagBuilder::new();
    let x = VarId(0);
    let y = VarId(1);
    let z = VarId(2);
    let w = VarId(3);
    let vx = b.variable(x);
    let vy = b.variable(y);
    let vz = b.variable(z);
    let vw = b.variable(w);
    let e0 = b.add(vec![vx, vy]);
    let e1 = b.add(vec![vy, vz]);
    let e2 = b.add(vec![vz, vw]);
    let d = b.build();
    let sys = PolynomialSystem::new(d, vec![e0, e1, e2]);
    let r = linear_elimination_greedy(&sys, &default_cfg()).unwrap();
    assert!(r.reduced_system.equations.is_empty());
    assert_eq!(r.substitution.len(), 3);
}

#[test]
fn partial_system_keeps_quadratic() {
    let mut b = DagBuilder::new();
    let x = VarId(0);
    let y = VarId(1);
    let z = VarId(2);
    let vx = b.variable(x);
    let vy = b.variable(y);
    let vz = b.variable(z);
    let c1 = b.constant(q(1, 1));
    let linear = b.add(vec![vx, vy]);
    let x2 = b.mul(vec![vx, vx]);
    let z2 = b.mul(vec![vz, vz]);
    let sum = b.add(vec![x2, z2]);
    let neg_c1 = b.neg(c1);
    let quad = b.add(vec![sum, neg_c1]);
    let d = b.build();
    let sys = PolynomialSystem::new(d, vec![linear, quad]);
    let r = linear_elimination_greedy(&sys, &default_cfg()).unwrap();
    assert_eq!(r.substitution.len(), 1);
    assert_eq!(r.reduced_system.equations.len(), 1);
}

#[test]
fn quadratic_systems_stall_without_linearities() {
    let mut b = DagBuilder::new();
    let x = VarId(0);
    let y = VarId(1);
    let vx = b.variable(x);
    let one = b.constant(q(1, 1));
    let x2 = b.mul(vec![vx, vx]);
    let neg_one = b.neg(one);
    let e0 = b.add(vec![x2, neg_one]);
    let d = b.build();
    let sys = PolynomialSystem::new(d, vec![e0]);
    let r = linear_elimination_greedy(&sys, &default_cfg()).unwrap();
    assert!(r.substitution.is_empty());
    assert_eq!(r.reduced_system.equations.len(), 1);

    let mut b = DagBuilder::new();
    let vx = b.variable(x);
    let vy = b.variable(y);
    let x2 = b.mul(vec![vx, vx]);
    let y2 = b.mul(vec![vy, vy]);
    let sum = b.add(vec![x2, y2]);
    let c1 = b.constant(q(1, 1));
    let minus_one = b.neg(c1);
    let e0 = b.add(vec![sum, minus_one]);
    let minus_y_squared = b.neg(y2);
    let diff = b.add(vec![x2, minus_y_squared]);
    let d = b.build();
    let sys = PolynomialSystem::new(d, vec![e0, diff]);
    let r = linear_elimination_greedy(&sys, &default_cfg()).unwrap();
    assert!(r.substitution.is_empty());
    assert_eq!(r.reduced_system.equations.len(), 2);
}

#[test]
fn illegitimate_elimination_when_equation_independent_of_target() {
    let mut b = DagBuilder::new();
    let x = VarId(0);
    let y = VarId(1);
    let z = VarId(2);
    let vy = b.variable(y);
    let vx = b.variable(x);
    let vz = b.variable(z);
    let e0 = vy;
    let e1 = b.add(vec![vx, vz]);
    let d = b.build();
    let sys = PolynomialSystem::new(d, vec![e0, e1]);
    let res = linear_elimination_greedy(&sys, &default_cfg());
    assert!(matches!(
        res,
        Err(EliminationError::IllegitimateElimination { var: VarId(0) })
    ));
}

#[test]
fn substitution_residuals_vanish_for_single_linear_equation() {
    let mut b = DagBuilder::new();
    let x = VarId(0);
    let y = VarId(1);
    let vx = b.variable(x);
    let vy = b.variable(y);
    let c2 = b.constant(q(2, 1));
    let two_y = b.mul(vec![c2, vy]);
    let eq = b.add(vec![vx, two_y]);
    let d = b.build();
    let sys = PolynomialSystem::new(d, vec![eq]);
    let r = linear_elimination_greedy(&sys, &default_cfg()).unwrap();
    assert_eq!(r.substitution.len(), 1);
    let sub_dag = r.substitution.dag();
    let order = r.substitution.replay_order().unwrap();
    let free = if r.substitution.contains(x) { y } else { x };
    for seed in 0usize..12 {
        let mut a = HashMap::new();
        a.insert(free, q((seed as i64) + 1, 1));
        for &v in &order {
            if let Some(root) = r.substitution.get(v) {
                let val = evaluate(&sub_dag, root, &a).unwrap();
                a.insert(v, val);
            }
        }
        let value_at_x = a.get(&x).unwrap().clone();
        let value_at_y = a.get(&y).unwrap().clone();
        assert_eq!(value_at_x + q(2, 1) * value_at_y, Q::zero());
    }
}

#[test]
fn vars_in_root_collects_nested_shared_subgraphs() {
    let mut b = DagBuilder::new();
    let x = VarId(0);
    let y = VarId(1);
    let vx = b.variable(x);
    let vy = b.variable(y);
    let shared = b.add(vec![vx, vy]);
    let prod = b.mul(vec![shared, shared]);
    let d = b.build();
    let v = d.vars_in_root(prod);
    assert_eq!(v, vec![x, y]);
}

#[test]
fn parse_named_identifiers_roundtrip_under_custom_names() {
    let s = "alpha + beta * alpha";
    let (dag, root) = Dag::from_rational_string(s).unwrap();
    let mut map: HashMap<VarId, &'static str> = HashMap::new();
    map.insert(VarId(0), "alpha");
    map.insert(VarId(1), "beta");
    let name = |v: VarId| map.get(&v).copied().unwrap_or("?").to_string();
    let out = dag.format_root(root, name);
    assert_eq!(out, "alpha + beta * alpha");
    let mut assign = HashMap::new();
    assign.insert(VarId(0), q(3, 1));
    assign.insert(VarId(1), q(2, 1));
    let val = evaluate(&dag, root, &assign).unwrap();
    assert_eq!(val, q(9, 1));
}

#[test]
fn format_parentheses_non_atomic_subexpressions() {
    let mut b = DagBuilder::new();
    let x = VarId(0);
    let vx = b.variable(x);
    let one = b.constant(q(1, 1));
    let sum = b.add(vec![vx, one]);
    let sq = b.mul(vec![sum, sum]);
    let d = b.build();
    let out = d.format_root(sq, |v| format!("v{}", v.0));
    assert_eq!(out, "(v0 + 1) * (v0 + 1)");
}
