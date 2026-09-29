# Variable-zeroing benchmark matrix

This report contains fresh measurements of `symbolica/bin/variable_zeroing.rs`
on the 20 generated algebraic benchmark systems. No physics system is
included. Every branch receives 3000 ms and the run a
60 s budget. Each cell records solver-reported total time /
wall time / first-solution time in seconds, then `N` recursion nodes,
`E` hitting-set evaluations, `O/S` per-branch timeouts / zeroing splits,
`C/U` components / unresolved branches. `P` marks a partial run,
`TO` an outer timeout and `ERR` a failed run.

| system | `zeroing-m4gb` | `zeroing-f5` | `zeroing-gvw` | `zeroing-symbolica` | `zeroing-singular-gb` |
|---|---: | ---: | ---: | ---: | ---:|
| `signed_distance_cycle_n8` | 0.866/1.381/0.668; N=1; E=70; O/S=0/0; C/U=70/0 | 0.845/0.850/0.648; N=1; E=70; O/S=0/0; C/U=70/0 | 0.844/0.849/0.645; N=1; E=70; O/S=0/0; C/U=70/0 | 0.863/0.886/0.666; N=1; E=70; O/S=0/0; C/U=70/0 | 1.214/1.220/0.668; N=1; E=70; O/S=0/0; C/U=70/0 |
| `signed_distance_cycle_n10` | P 3.001/3.007/--; N=1; E=0; O/S=0/0; C/U=0/1 | P 3.001/3.006/--; N=1; E=0; O/S=0/0; C/U=0/1 | P 3.000/3.006/--; N=1; E=0; O/S=0/0; C/U=0/1 | P 3.000/3.006/--; N=1; E=0; O/S=0/0; C/U=0/1 | P 3.000/3.006/--; N=1; E=0; O/S=0/0; C/U=0/1 |
| `signed_distance_cycle_n12` | P 3.003/3.008/--; N=1; E=0; O/S=0/0; C/U=0/1 | P 3.002/3.007/--; N=1; E=0; O/S=0/0; C/U=0/1 | P 3.003/3.008/--; N=1; E=0; O/S=0/0; C/U=0/1 | P 3.002/3.009/--; N=1; E=0; O/S=0/0; C/U=0/1 | P 3.000/3.006/--; N=1; E=0; O/S=0/0; C/U=0/1 |
| `overlapping_random_multiples_n4_p6_k2_seed31415` | 0.041/0.047/0.020; N=1; E=4; O/S=0/0; C/U=4/0 | 0.031/0.037/0.020; N=1; E=4; O/S=0/0; C/U=4/0 | 0.033/0.038/0.020; N=1; E=4; O/S=0/0; C/U=4/0 | 0.047/0.053/0.023; N=1; E=4; O/S=0/0; C/U=4/0 | 0.055/0.061/0.027; N=1; E=4; O/S=0/0; C/U=4/0 |
| `overlapping_random_multiples_n6_p8_k3_seed31415` | 1.139/1.166/0.096; N=1; E=16; O/S=0/0; C/U=16/0 | 0.212/0.239/0.078; N=1; E=16; O/S=0/0; C/U=16/0 | 0.360/0.387/0.081; N=1; E=16; O/S=0/0; C/U=16/0 | 0.768/0.781/0.074; N=1; E=16; O/S=0/0; C/U=16/0 | 0.264/0.291/0.087; N=1; E=16; O/S=0/0; C/U=16/0 |
| `overlapping_random_multiples_n8_p10_k3_seed31415` | P 60.003/60.096/0.310; N=18; E=37; O/S=13/10; C/U=21/14 | 11.009/11.103/0.244; N=1; E=28; O/S=0/0; C/U=28/0 | P 60.002/60.096/0.252; N=22; E=41; O/S=15/12; C/U=26/14 | P 60.004/60.031/0.194; N=24; E=43; O/S=16/13; C/U=21/14 | 2.945/3.037/0.248; N=1; E=28; O/S=0/0; C/U=28/0 |
| `affine_factor_hypergraph_n8_m6_k4_seed42` | 0.430/0.437/0.222; N=1; E=63; O/S=0/0; C/U=63/0 | 0.430/0.437/0.223; N=1; E=63; O/S=0/0; C/U=63/0 | 0.426/0.433/0.220; N=1; E=63; O/S=0/0; C/U=63/0 | 0.370/0.378/0.191; N=1; E=63; O/S=0/0; C/U=63/0 | 0.793/0.800/0.228; N=1; E=63; O/S=0/0; C/U=63/0 |
| `affine_factor_hypergraph_n10_m8_k5_seed42` | 1.602/1.611/0.817; N=1; E=239; O/S=0/0; C/U=239/0 | 1.612/1.622/0.813; N=1; E=239; O/S=0/0; C/U=239/0 | 1.608/1.617/0.810; N=1; E=239; O/S=0/0; C/U=239/0 | 1.424/1.433/0.730; N=1; E=239; O/S=0/0; C/U=239/0 | 2.990/2.999/0.827; N=1; E=239; O/S=0/0; C/U=239/0 |
| `affine_factor_hypergraph_n12_m10_k6_seed42` | 5.620/5.639/2.828; N=1; E=833; O/S=0/0; C/U=833/0 | 5.608/5.630/2.844; N=1; E=833; O/S=0/0; C/U=833/0 | 4.885/4.906/2.496; N=1; E=833; O/S=0/0; C/U=833/0 | 4.931/4.947/2.512; N=1; E=833; O/S=0/0; C/U=833/0 | 9.681/9.700/2.455; N=1; E=833; O/S=0/0; C/U=833/0 |
| `gated_cyclic_5` | 0.030/0.036/0.023; N=1; E=2; O/S=0/0; C/U=2/0 | 0.028/0.034/0.020; N=1; E=2; O/S=0/0; C/U=2/0 | 0.027/0.033/0.020; N=1; E=2; O/S=0/0; C/U=2/0 | 0.028/0.034/0.020; N=1; E=2; O/S=0/0; C/U=2/0 | 0.035/0.040/0.025; N=1; E=2; O/S=0/0; C/U=2/0 |
| `gated_cyclic_6` | P 9.184/9.189/0.020; N=13; E=6; O/S=3/6; C/U=1/1 | P 9.196/9.202/0.023; N=13; E=6; O/S=3/6; C/U=1/1 | 2.202/2.208/0.020; N=1; E=2; O/S=0/0; C/U=2/0 | P 9.268/9.274/0.020; N=13; E=6; O/S=3/6; C/U=1/1 | 0.060/0.066/0.027; N=1; E=2; O/S=0/0; C/U=2/0 |
| `gated_cyclic_7` | P 15.368/15.374/0.023; N=15; E=10; O/S=5/7; C/U=1/1 | P 16.973/16.979/0.023; N=15; E=10; O/S=5/7; C/U=1/1 | P 15.531/15.537/0.023; N=15; E=10; O/S=5/7; C/U=1/1 | P 15.376/15.382/0.023; N=15; E=10; O/S=5/7; C/U=1/1 | P 15.330/15.337/0.029; N=15; E=10; O/S=5/7; C/U=1/1 |
| `gated_katsura_6` | 0.376/0.382/0.020; N=1; E=2; O/S=0/0; C/U=2/0 | 0.487/0.493/0.020; N=1; E=2; O/S=0/0; C/U=2/0 | 1.310/1.316/0.020; N=1; E=2; O/S=0/0; C/U=2/0 | 2.758/2.765/0.022; N=1; E=2; O/S=0/0; C/U=2/0 | 0.065/0.071/0.028; N=1; E=2; O/S=0/0; C/U=2/0 |
| `gated_katsura_7` | P 60.003/60.008/0.020; N=14; E=9; O/S=8/8; C/U=1/9 | P 60.004/60.010/0.023; N=14; E=9; O/S=8/8; C/U=1/9 | P 60.004/60.010/0.020; N=14; E=9; O/S=8/8; C/U=1/9 | P 60.004/60.011/0.025; N=14; E=9; O/S=8/8; C/U=1/9 | 0.299/0.305/0.030; N=1; E=2; O/S=0/0; C/U=2/0 |
| `gated_katsura_8` | P 60.002/60.009/0.023; N=14; E=9; O/S=8/8; C/U=1/9 | P 60.003/60.010/0.024; N=14; E=9; O/S=8/8; C/U=1/9 | P 60.003/60.009/0.024; N=14; E=9; O/S=8/8; C/U=1/9 | P 60.002/60.009/0.034; N=14; E=9; O/S=8/8; C/U=1/9 | 2.598/2.604/0.032; N=1; E=2; O/S=0/0; C/U=2/0 |
| `gated_random_dense_quadratic_n5_seed137` | 0.186/0.192/0.020; N=1; E=2; O/S=0/0; C/U=2/0 | 0.087/0.093/0.021; N=1; E=2; O/S=0/0; C/U=2/0 | 0.213/0.218/0.021; N=1; E=2; O/S=0/0; C/U=2/0 | 0.238/0.244/0.022; N=1; E=2; O/S=0/0; C/U=2/0 | 0.041/0.047/0.026; N=1; E=2; O/S=0/0; C/U=2/0 |
| `gated_random_dense_quadratic_n6_seed137` | P 57.144/57.150/0.021; N=13; E=8; O/S=7/6; C/U=1/7 | P 57.145/57.151/0.021; N=13; E=8; O/S=7/6; C/U=1/7 | P 57.163/57.170/0.021; N=13; E=8; O/S=7/6; C/U=1/7 | P 57.165/57.171/0.022; N=13; E=8; O/S=7/6; C/U=1/7 | 0.548/0.555/0.026; N=1; E=2; O/S=0/0; C/U=2/0 |
| `gated_random_dense_quadratic_n7_seed137` | P 60.030/60.038/0.021; N=11; E=7; O/S=6/6; C/U=1/7 | P 60.031/60.039/0.021; N=11; E=7; O/S=6/6; C/U=1/7 | P 60.001/60.009/0.021; N=15; E=9; O/S=8/7; C/U=1/8 | P 60.049/60.062/0.021; N=12; E=8; O/S=7/6; C/U=1/7 | P 60.002/60.009/0.026; N=15; E=9; O/S=8/7; C/U=1/8 |
| `affine_factor_hypergraph_n8_k4_seed42` | P 3.002/3.009/--; N=1; E=0; O/S=0/0; C/U=0/1 | P 3.002/3.008/--; N=1; E=0; O/S=0/0; C/U=0/1 | P 3.002/3.008/--; N=1; E=0; O/S=0/0; C/U=0/1 | P 3.002/3.010/--; N=1; E=0; O/S=0/0; C/U=0/1 | P 3.001/3.008/--; N=1; E=0; O/S=0/0; C/U=0/1 |
| `affine_factor_hypergraph_n10_k5_seed42` | P 3.001/3.011/--; N=1; E=0; O/S=0/0; C/U=0/1 | P 3.002/3.012/--; N=1; E=0; O/S=0/0; C/U=0/1 | P 3.002/3.012/--; N=1; E=0; O/S=0/0; C/U=0/1 | P 3.002/3.011/--; N=1; E=0; O/S=0/0; C/U=0/1 | P 3.003/3.012/--; N=1; E=0; O/S=0/0; C/U=0/1 |

## Reproduction

```sh
python3 scripts/benchmark_branching.py --algorithm zeroing --no-build --jobs 1
```

The script generates the benchmark systems in a temporary directory.
It does not read the physics systems in `systems/`.
