# Factor-branching benchmark matrix

This report contains fresh measurements of `symbolica/bin/branching.rs`
on the 20 generated algebraic benchmark systems. No physics system is
included. Each cell records solver-reported total search time / wall time
in seconds, followed by first-solution latency and the largest component
dimension. `P a/b` denotes a partial run,
`CAP a/b` a cover-limited run, and `TO` an outer timeout.

| system | `branching-m4gb` | `branching-f5` | `branching-gvw` | `branching-symbolica` | `branching-singular-gb` |
|---|---: | ---: | ---: | ---: | ---:|
| `signed_distance_cycle_n8` | 0.204/0.789; first=0.003; d=1 | 0.202/0.212; first=0.003; d=1 | 0.201/0.211; first=0.003; d=1 | 0.200/0.209; first=0.003; d=1 | 0.559/0.568; first=0.023; d=1 |
| `signed_distance_cycle_n10` | 0.723/0.749; first=0.003; d=1 | 0.724/0.750; first=0.003; d=1 | 0.720/0.746; first=0.003; d=1 | 0.723/0.749; first=0.003; d=1 | 2.215/2.240; first=0.010; d=1 |
| `signed_distance_cycle_n12` | 2.661/2.801; first=0.003; d=1 | 2.685/2.824; first=0.003; d=1 | 2.693/2.834; first=0.003; d=1 | 2.666/2.806; first=0.003; d=1 | 8.164/8.304; first=0.010; d=1 |
| `overlapping_random_multiples_n4_p6_k2_seed31415` | 0.030/0.038; first=0.006; d=2 | 0.014/0.022; first=0.003; d=2 | 0.024/0.032; first=0.003; d=2 | 0.026/0.034; first=0.003; d=2 | 0.041/0.049; first=0.010; d=2 |
| `overlapping_random_multiples_n6_p8_k3_seed31415` | 1.066/1.119; first=0.016; d=3 | 0.143/0.193; first=0.006; d=3 | 0.296/0.348; first=0.006; d=3 | 0.675/0.725; first=0.008; d=3 | 0.194/0.248; first=0.010; d=3 |
| `overlapping_random_multiples_n8_p10_k3_seed31415` | TO 60.000/62.000 | 10.803/11.030; first=0.006; d=5 | 30.280/30.508; first=0.011; d=5 | TO 60.000/62.000 | 2.719/2.944; first=0.013; d=5 |
| `affine_factor_hypergraph_n8_m6_k4_seed42` | 0.183/0.191; first=0.003; d=6 | 0.190/0.198; first=0.003; d=6 | 0.182/0.190; first=0.003; d=6 | 0.183/0.191; first=0.003; d=6 | 0.523/0.531; first=0.010; d=6 |
| `affine_factor_hypergraph_n10_m8_k5_seed42` | 0.686/0.703; first=0.003; d=7 | 0.684/0.700; first=0.003; d=7 | 0.683/0.700; first=0.003; d=7 | 0.688/0.705; first=0.003; d=7 | 2.391/2.408; first=0.010; d=7 |
| `affine_factor_hypergraph_n12_m10_k6_seed42` | 2.392/2.459; first=0.003; d=9 | 2.389/2.454; first=0.003; d=9 | 2.391/2.455; first=0.003; d=9 | 2.392/2.457; first=0.003; d=9 | 7.922/7.986; first=0.010; d=9 |
| `gated_cyclic_5` | 0.011/0.018; first=0.003; d=5 | 0.011/0.018; first=0.003; d=5 | 0.011/0.018; first=0.003; d=5 | 0.011/0.018; first=0.003; d=5 | 0.021/0.027; first=0.010; d=5 |
| `gated_cyclic_6` | P 1/2; 60.007/60.014; first=0.003; d=6 | 5.950/5.958; first=0.003; d=6 | 2.271/2.278; first=0.003; d=6 | P 1/2; 60.007/60.014; first=0.003; d=6 | 0.087/0.094; first=0.046; d=6 |
| `gated_cyclic_7` | P 1/2; 60.006/60.015; first=0.003; d=7 | P 1/2; 60.443/60.452; first=0.003; d=7 | P 1/2; 61.339/61.354; first=0.003; d=7 | P 1/2; 60.006/60.021; first=0.003; d=7 | P 1/2; 60.025/60.033; first=0.023; d=7 |
| `gated_katsura_6` | 0.364/0.371; first=0.003; d=7 | 0.475/0.482; first=0.003; d=7 | 1.308/1.315; first=0.003; d=7 | 2.516/2.522; first=0.003; d=7 | 0.048/0.055; first=0.010; d=7 |
| `gated_katsura_7` | 4.368/4.375; first=0.003; d=8 | 15.101/15.109; first=0.003; d=8 | 42.025/42.033; first=0.003; d=8 | P 1/2; 60.006/60.013; first=0.003; d=8 | 0.283/0.291; first=0.010; d=8 |
| `gated_katsura_8` | 51.252/51.260; first=0.003; d=9 | P 1/2; 60.097/60.106; first=0.003; d=9 | P 1/2; 60.024/60.033; first=0.003; d=9 | P 1/2; 60.007/60.015; first=0.003; d=9 | 2.595/2.603; first=0.010; d=9 |
| `gated_random_dense_quadratic_n5_seed137` | 0.169/0.176; first=0.003; d=5 | 0.069/0.076; first=0.003; d=5 | 0.193/0.200; first=0.003; d=5 | 0.201/0.207; first=0.005; d=5 | 0.025/0.032; first=0.010; d=5 |
| `gated_random_dense_quadratic_n6_seed137` | 4.676/4.683; first=0.003; d=6 | 3.195/3.202; first=0.003; d=6 | 7.570/7.577; first=0.003; d=6 | 21.904/21.912; first=0.006; d=6 | 0.542/0.549; first=0.010; d=6 |
| `gated_random_dense_quadratic_n7_seed137` | P 1/2; 60.005/60.013; first=0.003; d=7 | P 1/2; 60.050/60.057; first=0.003; d=7 | P 1/2; 60.030/60.038; first=0.003; d=7 | P 1/2; 60.007/60.014; first=0.003; d=7 | 42.886/42.893; first=0.010; d=7 |
| `affine_factor_hypergraph_n8_k4_seed42` | TO 60.000/62.000 | TO 60.000/62.000 | TO 60.000/62.000 | TO 60.000/62.000 | TO 60.000/62.000 |
| `affine_factor_hypergraph_n10_k5_seed42` | 3.101/3.655; first=0.003; d=7 | 2.548/3.103; first=0.003; d=7 | 2.185/2.741; first=0.003; d=7 | 2.233/2.787; first=0.003; d=7 | 6.666/7.211; first=0.010; d=7 |

## Reproduction

```sh
python3 scripts/benchmark_branching.py --algorithm branching --no-build --jobs 1
```

The script generates the benchmark systems in a temporary directory.
It does not read the physics systems in `systems/`.
