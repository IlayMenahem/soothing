# Factor-branching benchmark matrix

This report contains fresh measurements of `symbolica/bin/branching.rs`
on the 20 generated algebraic benchmark systems. No physics system is
included. Each cell records solver-reported total search time / wall time
in seconds, followed by first-solution latency and the largest component
dimension. `P a/b` denotes a partial run,
`CAP a/b` a cover-limited run, and `TO` an outer timeout.

| system | `branching-m4gb` | `branching-f5` | `branching-gvw` | `branching-symbolica` | `branching-singular-gb` |
|---|---: | ---: | ---: | ---: | ---:|
| `signed_distance_cycle_n8` | 0.006/0.651; first=0.001; d=1 | 0.043/0.053; first=0.008; d=1 | 0.116/0.122; first=0.011; d=1 | 0.002/0.008; first=0.000; d=1 | 0.078/0.083; first=0.023; d=1 |
| `signed_distance_cycle_n10` | 0.040/0.046; first=0.003; d=1 | 10.911/10.927; first=2.038; d=1 | 45.782/45.797; first=4.418; d=1 | 0.018/0.025; first=0.002; d=1 | 0.077/0.084; first=0.023; d=1 |
| `signed_distance_cycle_n12` | 0.175/0.182; first=0.003; d=1 | 6.249/6.256; first=0.094; d=1 | 1.833/1.841; first=0.026; d=1 | 0.081/0.087; first=0.002; d=1 | 0.264/0.271; first=0.010; d=1 |
| `overlapping_random_multiples_n4_p6_k2_seed31415` | 0.010/0.017; first=0.000; d=2 | 0.002/0.008; first=0.000; d=2 | 0.005/0.011; first=0.000; d=2 | 0.009/0.015; first=0.000; d=2 | 0.031/0.038; first=0.008; d=2 |
| `overlapping_random_multiples_n6_p8_k3_seed31415` | 1.006/1.053; first=0.011; d=3 | 0.082/0.132; first=0.001; d=3 | 0.229/0.280; first=0.002; d=3 | 0.604/0.657; first=0.003; d=3 | 0.154/0.201; first=0.008; d=3 |
| `overlapping_random_multiples_n8_p10_k3_seed31415` | TO 60.000/62.000 | 10.713/10.937; first=0.001; d=5 | 30.461/30.687; first=0.007; d=5 | TO 60.000/62.000 | 2.707/2.931; first=0.010; d=5 |
| `affine_factor_hypergraph_n8_m6_k4_seed42` | 0.002/0.009; first=0.000; d=6 | 0.002/0.008; first=0.000; d=6 | 0.004/0.010; first=0.000; d=6 | 0.002/0.008; first=0.000; d=6 | 0.490/0.497; first=0.008; d=6 |
| `affine_factor_hypergraph_n10_m8_k5_seed42` | 0.009/0.021; first=0.000; d=7 | 0.008/0.020; first=0.000; d=7 | 0.016/0.028; first=0.000; d=7 | 0.008/0.020; first=0.000; d=7 | 1.858/1.870; first=0.008; d=7 |
| `affine_factor_hypergraph_n12_m10_k6_seed42` | 0.033/0.072; first=0.000; d=9 | 0.028/0.067; first=0.000; d=9 | 0.057/0.096; first=0.000; d=9 | 0.027/0.068; first=0.000; d=9 | 6.459/6.498; first=0.008; d=9 |
| `gated_cyclic_5` | 0.005/0.010; first=0.000; d=5 | 0.005/0.010; first=0.000; d=5 | 0.005/0.011; first=0.000; d=5 | 0.004/0.009; first=0.000; d=5 | 0.016/0.022; first=0.008; d=5 |
| `gated_cyclic_6` | P 1/2; 60.002/60.010; first=0.000; d=6 | 5.944/5.960; first=0.000; d=6 | 2.259/2.275; first=0.000; d=6 | P 1/2; 60.005/60.013; first=0.000; d=6 | 0.038/0.045; first=0.008; d=6 |
| `gated_cyclic_7` | P 1/2; 60.003/60.875; first=0.000; d=7 | P 1/2; 60.004/60.504; first=0.000; d=7 | P 1/2; 60.004/60.955; first=0.000; d=7 | P 1/2; 60.000/60.024; first=0.000; d=7 | P 1/2; 60.094/60.103; first=0.092; d=7 |
| `gated_katsura_6` | 0.360/0.366; first=0.000; d=7 | 0.476/0.483; first=0.000; d=7 | 1.332/1.338; first=0.000; d=7 | 2.533/2.540; first=0.000; d=7 | 0.043/0.049; first=0.008; d=7 |
| `gated_katsura_7` | 4.480/4.487; first=0.000; d=8 | 15.203/15.220; first=0.000; d=8 | 42.095/42.113; first=0.000; d=8 | P 1/2; 60.005/60.012; first=0.000; d=8 | 0.277/0.283; first=0.008; d=8 |
| `gated_katsura_8` | 54.045/54.055; first=0.000; d=9 | P 1/2; 60.005/60.325; first=0.000; d=9 | P 1/2; 60.003/60.047; first=0.000; d=9 | P 1/2; 60.000/60.012; first=0.000; d=9 | 2.648/2.657; first=0.052; d=9 |
| `gated_random_dense_quadratic_n5_seed137` | 0.170/0.177; first=0.000; d=5 | 0.064/0.071; first=0.000; d=5 | 0.194/0.200; first=0.000; d=5 | 0.197/0.204; first=0.000; d=5 | 0.023/0.029; first=0.008; d=5 |
| `gated_random_dense_quadratic_n6_seed137` | 4.709/4.716; first=0.000; d=6 | 3.186/3.194; first=0.000; d=6 | 7.467/7.475; first=0.000; d=6 | 21.491/21.498; first=0.000; d=6 | 0.540/0.546; first=0.008; d=6 |
| `gated_random_dense_quadratic_n7_seed137` | P 1/2; 60.001/60.008; first=0.000; d=7 | P 1/2; 60.001/60.050; first=0.000; d=7 | P 1/2; 60.005/60.041; first=0.000; d=7 | P 1/2; 60.005/60.015; first=0.000; d=7 | 42.011/42.020; first=0.017; d=7 |
| `affine_factor_hypergraph_n8_k4_seed42` | 15.105/15.117; first=0.001; d=5 | TO 60.000/62.000 | TO 60.000/62.000 | 3.082/3.096; first=0.000; d=5 | 0.460/0.470; first=0.023; d=5 |
| `affine_factor_hypergraph_n10_k5_seed42` | 14.726/14.750; first=0.000; d=7 | TO 60.000/62.000 | 47.537/47.561; first=0.000; d=7 | 1.150/1.174; first=0.000; d=7 | 2.525/2.548; first=0.008; d=7 |

## Reproduction

```sh
python3 scripts/benchmark_branching.py --no-build --jobs 1 --rerun-all
```

The script generates the benchmark systems in a temporary directory.
It does not read the physics systems in `systems/`.
