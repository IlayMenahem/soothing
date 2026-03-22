# soothing-linear

Standalone Rust implementation of **linear elimination by substitution** for polynomial systems over `\mathbb{Q}`, following the paper in the repository `paper/` directory.

## Scope

- Exact rationals (`num-rational`).
- DAG-stored expressions, topological evaluation, sampling-based zero/linearity predicates.
- Substitution map with dependency-aware replay order.
- Iterative elimination until no remaining equation is linear in any remaining variable.
- Optional local simplifications (appendix-style).
- **Not implemented:** the “handling systems without linearities” heuristic (paper section only names `grevlex`).

## Build / test

```bash
cd crates/soothing-linear
cargo test
```

## Example

```rust
use soothing_linear::{linear_elimination, DagBuilder, PolynomialSystem, SampleConfig, VarId};

let mut b = DagBuilder::new();
let x = VarId(0);
let y = VarId(1);
let vx = b.variable(x);
let vy = b.variable(y);
let eq = b.add(vec![vx, vy]); // x + y = 0
let sys = PolynomialSystem::new(b.build(), vec![eq]);
let out = linear_elimination(sys, &SampleConfig::default()).expect("elimination");
assert!(out.substitution.len() >= 1);
```
