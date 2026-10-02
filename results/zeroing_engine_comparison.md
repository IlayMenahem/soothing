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
| `signed_distance_cycle_n12` | P 3.001/3.635/--; N=1; E=0; O/S=0/0; C/U=0/1 | P 3.000/3.006/--; N=1; E=0; O/S=0/0; C/U=0/1 | P 3.002/3.008/--; N=1; E=0; O/S=0/0; C/U=0/1 | P 3.003/3.009/--; N=1; E=0; O/S=0/0; C/U=0/1 | P 3.001/3.007/--; N=1; E=0; O/S=0/0; C/U=0/1 |
| `signed_distance_cycle_n14` | P 3.001/3.007/--; N=1; E=0; O/S=0/0; C/U=0/1 | P 3.001/3.006/--; N=1; E=0; O/S=0/0; C/U=0/1 | P 3.000/3.007/--; N=1; E=0; O/S=0/0; C/U=0/1 | P 3.002/3.008/--; N=1; E=0; O/S=0/0; C/U=0/1 | P 3.001/3.007/--; N=1; E=0; O/S=0/0; C/U=0/1 |
| `signed_distance_cycle_n16` | P 3.002/3.009/--; N=1; E=0; O/S=0/0; C/U=0/1 | P 3.001/3.007/--; N=1; E=0; O/S=0/0; C/U=0/1 | P 3.000/3.006/--; N=1; E=0; O/S=0/0; C/U=0/1 | P 3.003/3.009/--; N=1; E=0; O/S=0/0; C/U=0/1 | P 3.001/3.008/--; N=1; E=0; O/S=0/0; C/U=0/1 |
| `overlapping_random_multiples_n4_p6_k2_seed31415` | 0.064/0.070/0.039; N=1; E=4; O/S=0/0; C/U=4/0 | 0.051/0.057/0.037; N=1; E=4; O/S=0/0; C/U=4/0 | 0.053/0.059/0.036; N=1; E=4; O/S=0/0; C/U=4/0 | 0.055/0.061/0.036; N=1; E=4; O/S=0/0; C/U=4/0 | 0.070/0.075/0.043; N=1; E=4; O/S=0/0; C/U=4/0 |
| `overlapping_random_multiples_n6_p8_k3_seed31415` | 1.166/1.177/0.141; N=1; E=16; O/S=0/0; C/U=16/0 | 0.274/0.285/0.127; N=1; E=16; O/S=0/0; C/U=16/0 | 0.483/0.495/0.146; N=1; E=16; O/S=0/0; C/U=16/0 | 0.821/0.832/0.133; N=1; E=16; O/S=0/0; C/U=16/0 | 0.345/0.356/0.144; N=1; E=16; O/S=0/0; C/U=16/0 |
| `overlapping_random_multiples_n8_p10_k3_seed31415` | P 60.005/60.031/0.310; N=13; E=30; O/S=9/8; C/U=21/15 | 11.198/11.225/0.252; N=1; E=28; O/S=0/0; C/U=28/0 | P 60.004/60.031/0.262; N=21; E=40; O/S=15/11; C/U=25/14 | P 60.004/60.031/0.261; N=16; E=34; O/S=12/9; C/U=22/15 | 2.931/2.956/0.258; N=1; E=28; O/S=0/0; C/U=28/0 |
| `affine_factor_hypergraph_n8_m6_k4_seed42` | 0.642/0.649/0.435; N=1; E=63; O/S=0/0; C/U=63/0 | 0.647/0.653/0.439; N=1; E=63; O/S=0/0; C/U=63/0 | 0.644/0.651/0.433; N=1; E=63; O/S=0/0; C/U=63/0 | 0.652/0.659/0.446; N=1; E=63; O/S=0/0; C/U=63/0 | 1.014/1.020/0.446; N=1; E=63; O/S=0/0; C/U=63/0 |
| `affine_factor_hypergraph_n10_m8_k5_seed42` | 2.408/2.416/1.608; N=1; E=239; O/S=0/0; C/U=239/0 | 2.419/2.427/1.624; N=1; E=239; O/S=0/0; C/U=239/0 | 2.409/2.417/1.612; N=1; E=239; O/S=0/0; C/U=239/0 | 2.421/2.429/1.619; N=1; E=239; O/S=0/0; C/U=239/0 | 3.808/3.817/1.630; N=1; E=239; O/S=0/0; C/U=239/0 |
| `affine_factor_hypergraph_n12_m10_k6_seed42` | P 3.001/3.017/--; N=1; E=0; O/S=0/0; C/U=0/1 | P 3.002/3.018/--; N=1; E=0; O/S=0/0; C/U=0/1 | P 3.003/3.019/--; N=1; E=0; O/S=0/0; C/U=0/1 | P 3.003/3.018/--; N=1; E=0; O/S=0/0; C/U=0/1 | P 3.003/3.019/--; N=1; E=0; O/S=0/0; C/U=0/1 |
| `gated_cyclic_5` | 0.030/0.036/0.021; N=1; E=2; O/S=0/0; C/U=2/0 | 0.030/0.036/0.020; N=1; E=2; O/S=0/0; C/U=2/0 | 0.030/0.037/0.020; N=1; E=2; O/S=0/0; C/U=2/0 | 0.030/0.035/0.021; N=1; E=2; O/S=0/0; C/U=2/0 | 0.035/0.041/0.026; N=1; E=2; O/S=0/0; C/U=2/0 |
| `gated_cyclic_6` | P 9.404/9.409/0.020; N=13; E=6; O/S=3/6; C/U=1/1 | P 9.412/9.418/0.021; N=13; E=6; O/S=3/6; C/U=1/1 | 2.183/2.189/0.021; N=1; E=2; O/S=0/0; C/U=2/0 | P 9.409/9.415/0.021; N=13; E=6; O/S=3/6; C/U=1/1 | 0.059/0.065/0.026; N=1; E=2; O/S=0/0; C/U=2/0 |
| `gated_cyclic_7` | P 15.543/15.549/0.021; N=15; E=10; O/S=5/7; C/U=1/1 | P 16.938/16.944/0.024; N=15; E=10; O/S=5/7; C/U=1/1 | P 15.709/15.716/0.024; N=15; E=10; O/S=5/7; C/U=1/1 | P 15.567/15.574/0.023; N=15; E=10; O/S=5/7; C/U=1/1 | P 15.531/15.537/0.026; N=15; E=10; O/S=5/7; C/U=1/1 |
| `gated_katsura_6` | 0.353/0.359/0.021; N=1; E=2; O/S=0/0; C/U=2/0 | 0.511/0.517/0.021; N=1; E=2; O/S=0/0; C/U=2/0 | 1.360/1.366/0.021; N=1; E=2; O/S=0/0; C/U=2/0 | 2.587/2.593/0.020; N=1; E=2; O/S=0/0; C/U=2/0 | 0.063/0.069/0.027; N=1; E=2; O/S=0/0; C/U=2/0 |
| `gated_katsura_7` | P 60.002/60.009/0.021; N=14; E=9; O/S=8/8; C/U=1/9 | P 60.004/60.011/0.025; N=14; E=9; O/S=8/8; C/U=1/9 | P 60.004/60.011/0.023; N=14; E=9; O/S=8/8; C/U=1/9 | P 60.004/60.010/0.017; N=14; E=9; O/S=8/8; C/U=1/9 | 0.295/0.301/0.025; N=1; E=2; O/S=0/0; C/U=2/0 |
| `gated_katsura_8` | P 60.002/60.009/0.020; N=14; E=9; O/S=8/8; C/U=1/9 | P 60.003/60.009/0.018; N=14; E=9; O/S=8/8; C/U=1/9 | P 60.004/60.011/0.018; N=14; E=9; O/S=8/8; C/U=1/9 | P 60.004/60.010/0.018; N=14; E=9; O/S=8/8; C/U=1/9 | 2.599/2.605/0.027; N=1; E=2; O/S=0/0; C/U=2/0 |
| `gated_random_dense_quadratic_n5_seed137` | 0.188/0.194/0.017; N=1; E=2; O/S=0/0; C/U=2/0 | 0.088/0.094/0.018; N=1; E=2; O/S=0/0; C/U=2/0 | 0.222/0.228/0.023; N=1; E=2; O/S=0/0; C/U=2/0 | 0.226/0.232/0.020; N=1; E=2; O/S=0/0; C/U=2/0 | 0.049/0.055/0.033; N=1; E=2; O/S=0/0; C/U=2/0 |
| `gated_random_dense_quadratic_n6_seed137` | P 57.136/57.143/0.020; N=13; E=8; O/S=7/6; C/U=1/7 | P 57.147/57.153/0.020; N=13; E=8; O/S=7/6; C/U=1/7 | P 57.141/57.148/0.020; N=13; E=8; O/S=7/6; C/U=1/7 | P 57.141/57.147/0.020; N=13; E=8; O/S=7/6; C/U=1/7 | 0.552/0.558/0.023; N=1; E=2; O/S=0/0; C/U=2/0 |
| `gated_random_dense_quadratic_n7_seed137` | P 60.042/60.051/0.020; N=12; E=8; O/S=7/6; C/U=1/7 | P 60.044/60.053/0.020; N=12; E=8; O/S=7/6; C/U=1/7 | P 60.045/60.054/0.020; N=12; E=8; O/S=7/6; C/U=1/7 | P 60.041/60.050/0.020; N=12; E=8; O/S=7/6; C/U=1/7 | P 60.046/60.055/0.027; N=12; E=8; O/S=7/6; C/U=1/7 |
| `affine_factor_hypergraph_n8_k4_seed42` | 1.532/1.538/1.029; N=1; E=178; O/S=0/0; C/U=178/0 | 1.545/1.551/1.040; N=1; E=178; O/S=0/0; C/U=178/0 | 1.542/1.549/1.040; N=1; E=178; O/S=0/0; C/U=178/0 | 1.538/1.544/1.032; N=1; E=178; O/S=0/0; C/U=178/0 | 2.412/2.418/1.049; N=1; E=178; O/S=0/0; C/U=178/0 |
| `affine_factor_hypergraph_n10_k5_seed42` | 2.678/2.687/1.801; N=1; E=309; O/S=0/0; C/U=309/0 | 2.676/2.685/1.790; N=1; E=309; O/S=0/0; C/U=309/0 | 2.661/2.669/1.784; N=1; E=309; O/S=0/0; C/U=309/0 | 2.659/2.667/1.778; N=1; E=309; O/S=0/0; C/U=309/0 | 4.222/4.230/1.796; N=1; E=309; O/S=0/0; C/U=309/0 |

## Reproduction

```sh
python3 scripts/benchmark_branching.py --algorithm zeroing --no-build --jobs 1
```

The script generates the benchmark systems in a temporary directory.
It does not read the physics systems in `systems/`.
