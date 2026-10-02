# Factor-branching benchmark matrix

This report contains fresh measurements of `symbolica/bin/branching.rs`
on the 20 generated algebraic benchmark systems. No physics system is
included. Each cell records solver-reported total search time / wall time
in seconds, followed by first-solution latency and the largest component
dimension. `P a/b` denotes a partial run,
`CAP a/b` a cover-limited run, and `TO` an outer timeout.

| system | `branching-m4gb` | `branching-f5` | `branching-gvw` | `branching-symbolica` | `branching-singular-gb` |
|---|---: | ---: | ---: | ---: | ---:|
| `signed_distance_cycle_n12` | 2.847/2.995; first=0.006; d=1 | 2.669/2.792; first=0.006; d=1 | 2.716/2.837; first=0.006; d=1 | 2.670/2.792; first=0.006; d=1 | 8.460/8.584; first=0.023; d=1 |
| `signed_distance_cycle_n14` | 9.914/10.698; first=0.006; d=1 | 9.894/10.675; first=0.003; d=1 | 9.824/10.610; first=0.006; d=1 | 9.887/10.663; first=0.006; d=1 | 30.975/31.753; first=0.010; d=1 |
| `signed_distance_cycle_n16` | 37.391/45.219; first=0.006; d=1 | 46.356/54.265; first=0.006; d=1 | TO 60.000/62.000 | TO 60.000/62.000 | TO 60.000/62.000 |
| `overlapping_random_multiples_n4_p6_k2_seed31415` | 0.024/0.034; first=0.003; d=2 | 0.014/0.021; first=0.003; d=2 | 0.019/0.026; first=0.003; d=2 | 0.024/0.031; first=0.003; d=2 | 0.041/0.048; first=0.010; d=2 |
| `overlapping_random_multiples_n6_p8_k3_seed31415` | 1.032/1.051; first=0.016; d=3 | 0.160/0.178; first=0.006; d=3 | 0.355/0.374; first=0.008; d=3 | 0.687/0.705; first=0.008; d=3 | 0.181/0.199; first=0.010; d=3 |
| `overlapping_random_multiples_n8_p10_k3_seed31415` | TO 60.000/62.000 | 10.732/10.795; first=0.006; d=5 | 35.113/35.176; first=0.014; d=5 | TO 60.000/62.000 | 2.671/2.734; first=0.013; d=5 |
| `affine_factor_hypergraph_n8_m6_k4_seed42` | 0.180/0.188; first=0.003; d=6 | 0.189/0.196; first=0.003; d=6 | 0.185/0.193; first=0.003; d=6 | 0.209/0.217; first=0.003; d=6 | 0.571/0.580; first=0.010; d=6 |
| `affine_factor_hypergraph_n10_m8_k5_seed42` | 0.717/0.732; first=0.003; d=7 | 0.707/0.722; first=0.003; d=7 | 0.711/0.726; first=0.003; d=7 | 0.703/0.719; first=0.003; d=7 | 2.063/2.078; first=0.010; d=7 |
| `affine_factor_hypergraph_n12_m10_k6_seed42` | 2.404/2.455; first=0.003; d=9 | 2.401/2.453; first=0.003; d=9 | 2.410/2.462; first=0.003; d=9 | 2.398/2.449; first=0.003; d=9 | 7.884/7.936; first=0.010; d=9 |
| `gated_cyclic_5` | 0.011/0.017; first=0.003; d=5 | 0.011/0.016; first=0.003; d=5 | 0.011/0.017; first=0.003; d=5 | 0.011/0.017; first=0.003; d=5 | 0.020/0.026; first=0.010; d=5 |
| `gated_cyclic_6` | P 1/2; 60.007/60.014; first=0.003; d=6 | 5.653/5.659; first=0.003; d=6 | 2.198/2.205; first=0.005; d=6 | P 1/2; 60.007/60.013; first=0.003; d=6 | 0.043/0.049; first=0.010; d=6 |
| `gated_cyclic_7` | P 1/2; 60.005/60.012; first=0.003; d=7 | P 1/2; 60.364/60.371; first=0.003; d=7 | P 1/2; 61.200/61.215; first=0.006; d=7 | P 1/2; 60.009/60.024; first=0.006; d=7 | P 1/2; 60.025/60.032; first=0.023; d=7 |
| `gated_katsura_6` | 0.348/0.355; first=0.005; d=7 | 0.502/0.508; first=0.006; d=7 | 1.377/1.384; first=0.005; d=7 | 2.628/2.635; first=0.003; d=7 | 0.047/0.054; first=0.010; d=7 |
| `gated_katsura_7` | 4.247/4.253; first=0.005; d=8 | 15.338/15.345; first=0.003; d=8 | 42.858/42.865; first=0.003; d=8 | P 1/2; 60.007/60.013; first=0.003; d=8 | 0.283/0.290; first=0.010; d=8 |
| `gated_katsura_8` | 51.074/51.082; first=0.003; d=9 | P 1/2; 60.085/60.092; first=0.003; d=9 | P 1/2; 60.021/60.027; first=0.003; d=9 | P 1/2; 60.005/60.012; first=0.003; d=9 | 2.575/2.582; first=0.010; d=9 |
| `gated_random_dense_quadratic_n5_seed137` | 0.174/0.181; first=0.003; d=5 | 0.074/0.080; first=0.003; d=5 | 0.203/0.210; first=0.003; d=5 | 0.209/0.216; first=0.003; d=5 | 0.025/0.032; first=0.010; d=5 |
| `gated_random_dense_quadratic_n6_seed137` | 4.708/4.714; first=0.003; d=6 | 3.216/3.223; first=0.003; d=6 | 7.477/7.484; first=0.003; d=6 | 21.756/21.763; first=0.003; d=6 | 0.552/0.559; first=0.010; d=6 |
| `gated_random_dense_quadratic_n7_seed137` | P 1/2; 60.008/60.016; first=0.006; d=7 | P 1/2; 60.046/60.054; first=0.003; d=7 | P 1/2; 60.032/60.039; first=0.006; d=7 | P 1/2; 60.004/60.011; first=0.003; d=7 | 42.670/42.677; first=0.010; d=7 |
| `affine_factor_hypergraph_n8_k4_seed42` | 0.532/0.544; first=0.003; d=5 | 0.557/0.567; first=0.003; d=5 | 0.540/0.551; first=0.003; d=5 | 0.557/0.568; first=0.003; d=5 | 1.569/1.580; first=0.010; d=5 |
| `affine_factor_hypergraph_n10_k5_seed42` | 0.940/0.961; first=0.003; d=7 | 0.975/0.993; first=0.003; d=7 | 0.927/0.946; first=0.006; d=7 | 0.920/0.938; first=0.003; d=7 | 2.909/2.928; first=0.010; d=7 |

## Reproduction

```sh
python3 scripts/benchmark_branching.py --algorithm branching --no-build --jobs 1
```

The script generates the benchmark systems in a temporary directory.
It does not read the physics systems in `systems/`.
