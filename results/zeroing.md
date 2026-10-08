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
| `signed_distance_cycle_n12` | P 3.001/3.009/--; N=1; E=0; O/S=0/0; C/U=0/1 | P 3.002/3.008/--; N=1; E=0; O/S=0/0; C/U=0/1 | P 3.002/3.009/--; N=1; E=0; O/S=0/0; C/U=0/1 | P 3.002/3.008/--; N=1; E=0; O/S=0/0; C/U=0/1 | P 3.003/3.010/--; N=1; E=0; O/S=0/0; C/U=0/1 |
| `signed_distance_cycle_n14` | P 3.005/3.013/--; N=1; E=0; O/S=0/0; C/U=0/1 | P 3.006/3.014/--; N=1; E=0; O/S=0/0; C/U=0/1 | P 3.005/3.012/--; N=1; E=0; O/S=0/0; C/U=0/1 | P 3.005/3.012/--; N=1; E=0; O/S=0/0; C/U=0/1 | P 3.005/3.013/--; N=1; E=0; O/S=0/0; C/U=0/1 |
| `signed_distance_cycle_n16` | P 3.027/3.037/--; N=1; E=0; O/S=0/0; C/U=0/1 | P 3.024/3.035/--; N=1; E=0; O/S=0/0; C/U=0/1 | P 3.022/3.033/--; N=1; E=0; O/S=0/0; C/U=0/1 | P 3.024/3.036/--; N=1; E=0; O/S=0/0; C/U=0/1 | P 3.023/3.035/--; N=1; E=0; O/S=0/0; C/U=0/1 |
| `overlapping_random_multiples_n4_p6_k2_seed31415` | 0.055/0.061/0.031; N=1; E=4; O/S=0/0; C/U=4/0 | 0.045/0.051/0.033; N=1; E=4; O/S=0/0; C/U=4/0 | 0.052/0.058/0.032; N=1; E=4; O/S=0/0; C/U=4/0 | 0.055/0.060/0.031; N=1; E=4; O/S=0/0; C/U=4/0 | 0.072/0.078/0.041; N=1; E=4; O/S=0/0; C/U=4/0 |
| `overlapping_random_multiples_n6_p8_k3_seed31415` | 1.166/1.177/0.124; N=1; E=16; O/S=0/0; C/U=16/0 | 0.280/0.292/0.120; N=1; E=16; O/S=0/0; C/U=16/0 | 0.465/0.477/0.120; N=1; E=16; O/S=0/0; C/U=16/0 | 0.823/0.835/0.122; N=1; E=16; O/S=0/0; C/U=16/0 | 0.309/0.320/0.126; N=1; E=16; O/S=0/0; C/U=16/0 |
| `overlapping_random_multiples_n8_p10_k3_seed31415` | P 60.002/60.030/0.320; N=9; E=26; O/S=5/4; C/U=21/11 | 11.980/12.011/0.306; N=1; E=28; O/S=0/0; C/U=28/0 | P 60.006/60.036/0.283; N=11; E=29; O/S=5/5; C/U=24/9 | P 60.006/60.033/0.273; N=11; E=27; O/S=5/5; C/U=22/11 | 3.015/3.043/0.259; N=1; E=28; O/S=0/0; C/U=28/0 |
| `affine_factor_hypergraph_n8_m6_k4_seed42` | 0.657/0.664/0.431; N=1; E=63; O/S=0/0; C/U=63/0 | 0.671/0.678/0.466; N=1; E=63; O/S=0/0; C/U=63/0 | 0.647/0.653/0.445; N=1; E=63; O/S=0/0; C/U=63/0 | 0.635/0.641/0.430; N=1; E=63; O/S=0/0; C/U=63/0 | 1.011/1.018/0.430; N=1; E=63; O/S=0/0; C/U=63/0 |
| `affine_factor_hypergraph_n10_m8_k5_seed42` | 2.406/2.416/1.620; N=1; E=239; O/S=0/0; C/U=239/0 | 2.414/2.422/1.610; N=1; E=239; O/S=0/0; C/U=239/0 | 2.413/2.422/1.623; N=1; E=239; O/S=0/0; C/U=239/0 | 2.419/2.427/1.603; N=1; E=239; O/S=0/0; C/U=239/0 | 3.888/3.897/1.608; N=1; E=239; O/S=0/0; C/U=239/0 |
| `affine_factor_hypergraph_n12_m10_k6_seed42` | P 3.003/3.020/--; N=1; E=0; O/S=0/0; C/U=0/1 | P 3.002/3.018/--; N=1; E=0; O/S=0/0; C/U=0/1 | P 3.004/3.020/--; N=1; E=0; O/S=0/0; C/U=0/1 | P 3.002/3.019/--; N=1; E=0; O/S=0/0; C/U=0/1 | P 3.003/3.019/--; N=1; E=0; O/S=0/0; C/U=0/1 |
| `gated_cyclic_5` | 0.031/0.037/0.021; N=1; E=2; O/S=0/0; C/U=2/0 | 0.030/0.037/0.021; N=1; E=2; O/S=0/0; C/U=2/0 | 0.030/0.036/0.020; N=1; E=2; O/S=0/0; C/U=2/0 | 0.030/0.037/0.020; N=1; E=2; O/S=0/0; C/U=2/0 | 0.040/0.046/0.029; N=1; E=2; O/S=0/0; C/U=2/0 |
| `gated_cyclic_6` | P 9.124/9.130/0.021; N=3; E=2; O/S=1/1; C/U=1/1 | P 9.123/9.130/0.021; N=3; E=2; O/S=1/1; C/U=1/1 | 2.275/2.281/0.020; N=1; E=2; O/S=0/0; C/U=2/0 | P 9.124/9.130/0.020; N=3; E=2; O/S=1/1; C/U=1/1 | 0.064/0.070/0.028; N=1; E=2; O/S=0/0; C/U=2/0 |
| `gated_cyclic_7` | P 36.505/36.512/0.021; N=15; E=10; O/S=5/7; C/U=1/1 | P 37.969/37.977/0.021; N=15; E=10; O/S=5/7; C/U=1/1 | P 36.654/36.660/0.021; N=15; E=10; O/S=5/7; C/U=1/1 | P 36.422/36.429/0.020; N=15; E=10; O/S=5/7; C/U=1/1 | P 36.426/36.433/0.026; N=15; E=10; O/S=5/7; C/U=1/1 |
| `gated_katsura_6` | 0.366/0.373/0.021; N=1; E=2; O/S=0/0; C/U=2/0 | 0.517/0.524/0.020; N=1; E=2; O/S=0/0; C/U=2/0 | 1.390/1.397/0.022; N=1; E=2; O/S=0/0; C/U=2/0 | 2.683/2.690/0.020; N=1; E=2; O/S=0/0; C/U=2/0 | 0.066/0.072/0.026; N=1; E=2; O/S=0/0; C/U=2/0 |
| `gated_katsura_7` | P 60.005/60.012/0.022; N=11; E=8; O/S=7/7; C/U=1/7 | P 60.004/60.011/0.021; N=11; E=8; O/S=7/7; C/U=1/7 | P 60.004/60.011/0.021; N=11; E=8; O/S=7/7; C/U=1/7 | P 60.004/60.011/0.021; N=11; E=8; O/S=7/7; C/U=1/7 | 0.296/0.303/0.026; N=1; E=2; O/S=0/0; C/U=2/0 |
| `gated_katsura_8` | P 60.003/60.010/0.024; N=11; E=8; O/S=7/7; C/U=1/7 | P 60.004/60.011/0.021; N=11; E=8; O/S=7/7; C/U=1/7 | P 60.002/60.009/0.021; N=12; E=8; O/S=7/7; C/U=1/7 | P 60.005/60.012/0.021; N=12; E=9; O/S=8/7; C/U=1/7 | 2.505/2.511/0.031; N=1; E=2; O/S=0/0; C/U=2/0 |
| `gated_random_dense_quadratic_n5_seed137` | 0.180/0.187/0.021; N=1; E=2; O/S=0/0; C/U=2/0 | 0.087/0.093/0.022; N=1; E=2; O/S=0/0; C/U=2/0 | 0.205/0.211/0.021; N=1; E=2; O/S=0/0; C/U=2/0 | 0.210/0.216/0.021; N=1; E=2; O/S=0/0; C/U=2/0 | 0.041/0.047/0.026; N=1; E=2; O/S=0/0; C/U=2/0 |
| `gated_random_dense_quadratic_n6_seed137` | P 14.689/14.695/0.021; N=3; E=2; O/S=1/1; C/U=1/2 | 3.026/3.033/0.021; N=1; E=2; O/S=0/0; C/U=2/0 | P 14.710/14.717/0.021; N=3; E=2; O/S=1/1; C/U=1/2 | P 14.717/14.723/0.021; N=3; E=2; O/S=1/1; C/U=1/2 | 0.541/0.547/0.026; N=1; E=2; O/S=0/0; C/U=2/0 |
| `gated_random_dense_quadratic_n7_seed137` | P 60.022/60.030/0.021; N=9; E=6; O/S=5/5; C/U=1/6 | P 60.021/60.029/0.021; N=9; E=6; O/S=5/5; C/U=1/6 | P 60.022/60.031/0.021; N=9; E=6; O/S=5/5; C/U=1/6 | P 60.020/60.029/0.021; N=9; E=6; O/S=5/5; C/U=1/6 | P 60.022/60.031/0.027; N=9; E=6; O/S=5/5; C/U=1/6 |
| `affine_factor_hypergraph_n8_k4_seed42` | 1.541/1.548/1.036; N=1; E=178; O/S=0/0; C/U=178/0 | 1.514/1.521/1.013; N=1; E=178; O/S=0/0; C/U=178/0 | 1.522/1.528/1.017; N=1; E=178; O/S=0/0; C/U=178/0 | 1.518/1.525/1.015; N=1; E=178; O/S=0/0; C/U=178/0 | 2.411/2.417/1.023; N=1; E=178; O/S=0/0; C/U=178/0 |
| `affine_factor_hypergraph_n10_k5_seed42` | 2.660/2.669/1.777; N=1; E=309; O/S=0/0; C/U=309/0 | 2.653/2.662/1.773; N=1; E=309; O/S=0/0; C/U=309/0 | 2.648/2.657/1.771; N=1; E=309; O/S=0/0; C/U=309/0 | 2.646/2.655/1.764; N=1; E=309; O/S=0/0; C/U=309/0 | 4.147/4.156/1.773; N=1; E=309; O/S=0/0; C/U=309/0 |

## Reproduction

```sh
python3 scripts/baseline_table.py --part zeroing --no-build --jobs 1
```

The script generates the benchmark systems in a temporary directory.
It does not read the physics systems in `systems/`.
