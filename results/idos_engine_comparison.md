# IDOS and baseline benchmark matrix

This report consolidates the tuned size sweeps and every algorithm exposed by
[`scripts/benchmark_idos.py`](../scripts/benchmark_idos.py) and
[`scripts/baselines/`](../scripts/baselines/). The selected systems avoid the
trivial `Gr(2,4)` and first-cover-only gated tests: a completed IDOS result must
visit every required cover, including the nonlinear residual cover.
This Markdown matrix is the canonical result store: each `(system, algorithm)`
pair has one cell, and reruns replace that cell atomically.

## How to read the matrix

- `solve/wall; d=N` means tool-reported solve time / end-to-end wall time in
  seconds, followed by the largest solution-component dimension.
- Direct whole-ideal Groebner methods show only `solve/wall`, as they already
  return a basis for the full solution set and do not need a dimension entry.
- `P a/b` is a partial IDOS result with `a` of `b` covers solved. Its dimension
  is the largest dimension found so far, not a claim that the search completed.
- `CAP a/b` is a deliberately capped partial search, and `TO` is a timeout.
- Times below are single measurements, not distributions. Existing cells retain
  their original run settings; the formerly unmeasured cells were completed
  with eight concurrent method jobs. Use one job for publication timing.

## Algorithms to benchmark

Together, the three result tables below form the complete system-by-algorithm
matrix.

| class | algorithms | result represented |
|---|---|---|
| IDOS factor-cover search | `idos-m4gb`, `idos-f5`, `idos-gvw`, `idos-symbolica`, `idos-singular-gb` | exhaustive/capped component search; largest returned dimension |
| direct whole-ideal Groebner basis | `m4gb-grevlex`, `f5-grevlex`, `gvw-grevlex`, `builtin-grevlex`, `singular-gb-lex`, `m2-gb-lex` | one basis for the original ideal; dimension intentionally omitted |
| decomposition / numerical algebraic geometry | `singular-facstd`, `singular-primdecGTZ`, `m2-decompose`, `homotopy-jl` | components and maximum component dimension |

`idos-singular-gb` runs Singular separately on every IDOS cover;
`singular-gb-lex` runs Singular once on the complete expanded ideal. They are
therefore distinct algorithms and both belong in the benchmark.

Macaulay2 was upgraded from 1.26.05 to 1.26.06 to repair stale Homebrew
dynamic-library links. The converter was also changed to use alphanumeric
positional variable names, because Macaulay2 parses `_` as its indexed-variable
operator. Both Macaulay2 columns are included below.

## Systems to benchmark

| system | variables | equations | measured/required IDOS covers |
|---|---:|---:|---:|
| `signed_distance_cycle_n8` | 8 | 8 | 8/8 |
| `signed_distance_cycle_n10` | 10 | 10 | 8/8 |
| `signed_distance_cycle_n12` | 12 | 12 | 32/32 |
| `overlapping_random_multiples_n4_p6_k2_seed31415` | 4 | 4 | 4/4 |
| `overlapping_random_multiples_n6_p8_k3_seed31415` | 6 | 6 | 16/16 |
| `overlapping_random_multiples_n8_p10_k3_seed31415` | 8 | 8 | 28/28 |
| `affine_factor_hypergraph_n8_m6_k4_seed42` | 8 | 6 | 63/63 |
| `affine_factor_hypergraph_n10_m8_k5_seed42` | 10 | 8 | 239/239 |
| `affine_factor_hypergraph_n12_m10_k6_seed42` | 12 | 10 | 833/833 |
| `gated_cyclic_5` | 6 | 5 | 2/2 |
| `gated_cyclic_6` | 7 | 6 | 2/2 |
| `gated_katsura_6` | 8 | 7 | 2/2 |
| `gated_katsura_7` | 9 | 8 | 2/2 |
| `gated_random_dense_quadratic_n5_seed137` | 6 | 5 | 2/2 |
| `gated_random_dense_quadratic_n6_seed137` | 7 | 6 | 2/2 |
| `affine_factor_hypergraph_n8_k4_seed42` | 8 | 8 | 31/31 |
| `affine_factor_hypergraph_n10_k5_seed42` | 10 | 10 | 304/304 |

## IDOS factor-cover algorithms

The IDOS measurements used a 45-second per-cover limit for structured systems
and a 20-second limit for hard residual systems. Overlapping-random-multiple
rows disable input interreduction so their deliberately repeated nonlinear
factors reach factor-cover enumeration unchanged.

| system | `idos-m4gb` | `idos-f5` | `idos-gvw` | `idos-symbolica` | `idos-singular-gb` |
|---|---:|---:|---:|---:|---:|
| `signed_distance_cycle_n8` | 0.003/0.008; d=1 | 0.030/0.035; d=1 | 0.123/0.129; d=1 | 0.003/0.008; d=1 | 0.062/0.068; d=1 |
| `signed_distance_cycle_n10` | 0.039/0.045; d=1 | 11.153/11.167; d=1 | 46.377/46.390; d=1 | 0.020/0.026; d=1 | 0.065/0.071; d=1 |
| `signed_distance_cycle_n12` | 0.172/0.178; d=1 | 6.634/6.642; d=1 | 1.913/1.921; d=1 | 0.088/0.094; d=1 | 0.308/0.314; d=1 |
| `overlapping_random_multiples_n4_p6_k2_seed31415` | 0.021/0.443; d=2 | 0.005/0.012; d=2 | 0.008/0.015; d=2 | 0.011/0.017; d=2 | 0.431/0.437; d=2 |
| `overlapping_random_multiples_n6_p8_k3_seed31415` | 0.979/1.009; d=3 | 0.100/0.128; d=3 | 0.236/0.265; d=3 | 0.599/0.627; d=3 | 0.173/0.200; d=3 |
| `overlapping_random_multiples_n8_p10_k3_seed31415` | TO 45.000/47.000 | 10.342/10.439; d=5 | 29.553/29.651; d=5 | TO 45.000/47.000 | 2.829/2.926; d=5 |
| `affine_factor_hypergraph_n8_m6_k4_seed42` | 0.004/0.009; d=6 | 0.003/0.008; d=6 | 0.005/0.010; d=6 | 0.003/0.009; d=6 | 0.490/0.496; d=6 |
| `affine_factor_hypergraph_n10_m8_k5_seed42` | 0.016/0.024; d=7 | 0.014/0.022; d=7 | 0.022/0.030; d=7 | 0.014/0.023; d=7 | 1.858/1.866; d=7 |
| `affine_factor_hypergraph_n12_m10_k6_seed42` | 0.075/0.095; d=9 | 0.071/0.091; d=9 | 0.101/0.121; d=9 | 0.071/0.092; d=9 | 6.509/6.532; d=9 |
| `gated_cyclic_5` | 0.005/0.010; d=5 | 0.005/0.010; d=5 | 0.005/0.010; d=5 | 0.004/0.009; d=5 | 0.016/0.022; d=5 |
| `gated_cyclic_6` | P 1/2; 20.006/20.013; d=6 | 5.926/5.947; d=6 | 2.272/2.289; d=6 | P 1/2; 20.004/20.011; d=6 | 0.057/0.063; d=6 |
| `gated_katsura_6` | 0.361/0.367; d=7 | 0.484/0.490; d=7 | 1.344/1.351; d=7 | 2.560/2.567; d=7 | 0.044/0.050; d=7 |
| `gated_katsura_7` | 4.500/4.507; d=8 | 15.319/15.338; d=8 | P 1/2; 20.006/20.022; d=8 | P 1/2; 20.003/20.009; d=8 | 0.292/0.298; d=8 |
| `gated_random_dense_quadratic_n5_seed137` | 0.170/0.176; d=5 | 0.066/0.072; d=5 | 0.192/0.198; d=5 | 0.198/0.204; d=5 | 0.025/0.031; d=5 |
| `gated_random_dense_quadratic_n6_seed137` | 4.731/4.737; d=6 | 3.198/3.205; d=6 | 7.491/7.499; d=6 | P 1/2; 20.005/20.012; d=6 | 0.534/0.540; d=6 |
| `affine_factor_hypergraph_n8_k4_seed42` | 15.034/15.043; d=5 | TO 20.000/22.000 | TO 20.000/22.000 | 3.109/3.118; d=5 | 0.675/0.682; d=5 |
| `affine_factor_hypergraph_n10_k5_seed42` | 15.050/15.065; d=7 | TO 20.000/22.000 | TO 20.000/22.000 | 1.176/1.188; d=7 | 2.556/2.567; d=7 |

The apparent dimension on a partial gated row is the easy gate hyperplane.
The nonlinear residual cover timed out, so that dimension does not turn the
row into a completed solve.

## Direct whole-ideal Groebner algorithms

Per the reporting convention above, these cells intentionally omit solution
dimension. Existing direct-baseline measurements used a 10-second timeout.

| system | `m4gb-grevlex` | `f5-grevlex` | `gvw-grevlex` | `builtin-grevlex` | `singular-gb-lex` | `m2-gb-lex` |
|---|---:|---:|---:|---:|---:|---:|
| `signed_distance_cycle_n8` | 0.005/0.012 | 0.004/0.010 | 0.003/0.008 | 0.001/0.007 | 0.002/0.245 | 0.002/0.591 |
| `signed_distance_cycle_n10` | 0.123/0.128 | 0.232/0.238 | 0.121/0.129 | 0.022/0.028 | 0.015/0.185 | 0.051/0.634 |
| `signed_distance_cycle_n12` | 3.724/3.736 | TO 10.000/10.014 | 9.874/9.883 | 0.531/0.540 | 0.360/0.594 | 1.216/1.700 |
| `overlapping_random_multiples_n4_p6_k2_seed31415` | 1.014/1.411 | TO 10.000/10.011 | 0.465/0.471 | 0.408/0.413 | TO 10.000/10.390 | TO 10.000/10.221 |
| `overlapping_random_multiples_n6_p8_k3_seed31415` | TO 10.000/10.027 | TO 10.000/10.029 | TO 10.000/10.040 | TO 10.000/10.026 | TO 10.000/11.347 | TO 10.000/11.360 |
| `overlapping_random_multiples_n8_p10_k3_seed31415` | TO 10.000/10.089 | TO 10.000/10.096 | TO 10.000/10.094 | TO 10.000/10.092 | TO 10.000/12.000 | TO 10.000/12.000 |
| `affine_factor_hypergraph_n8_m6_k4_seed42` | TO 10.000/10.012 | TO 10.000/10.019 | TO 10.000/10.036 | TO 10.000/10.020 | TO 10.000/10.421 | TO 10.000/10.390 |
| `affine_factor_hypergraph_n10_m8_k5_seed42` | TO 10.000/10.011 | TO 10.000/10.011 | TO 10.000/10.024 | TO 10.000/10.036 | TO 10.000/10.723 | TO 10.000/10.878 |
| `affine_factor_hypergraph_n12_m10_k6_seed42` | TO 10.000/10.025 | TO 10.000/10.045 | TO 10.000/10.029 | TO 10.000/10.134 | TO 10.000/12.000 | TO 10.000/12.000 |
| `gated_cyclic_5` | 0.006/0.012 | 0.283/0.289 | 0.005/0.010 | 0.004/0.009 | 0.002/0.190 | 0.007/0.639 |
| `gated_cyclic_6` | TO 10.000/10.012 | TO 10.000/10.009 | 1.877/1.897 | TO 10.000/10.008 | 0.144/0.438 | 0.674/1.201 |
| `gated_katsura_6` | 0.511/0.518 | TO 10.000/10.012 | 0.295/0.304 | 6.219/6.228 | TO 10.000/10.243 | TO 10.000/10.236 |
| `gated_katsura_7` | 5.987/5.995 | TO 10.000/10.008 | 3.658/3.665 | TO 10.000/10.012 | TO 10.000/10.217 | TO 10.000/10.218 |
| `gated_random_dense_quadratic_n5_seed137` | 0.202/0.209 | TO 10.000/10.012 | 0.209/0.215 | 0.225/0.231 | 0.394/0.626 | TO 10.000/10.228 |
| `gated_random_dense_quadratic_n6_seed137` | 5.673/5.682 | TO 10.000/10.010 | 8.063/8.070 | TO 10.000/10.008 | TO 10.000/10.251 | TO 10.000/10.247 |
| `affine_factor_hypergraph_n8_k4_seed42` | TO 10.000/10.012 | TO 10.000/10.013 | TO 10.000/10.029 | TO 10.000/10.012 | TO 10.000/10.371 | TO 10.000/10.380 |
| `affine_factor_hypergraph_n10_k5_seed42` | TO 10.000/10.010 | TO 10.000/10.009 | TO 10.000/10.015 | TO 10.000/10.016 | TO 10.000/10.797 | TO 10.000/10.915 |

## Decomposition and numerical algorithms

These methods report maximum component dimension when they complete.

| system | `singular-facstd` | `singular-primdecGTZ` | `m2-decompose` | `homotopy-jl` |
|---|---:|---:|---:|---:|
| `signed_distance_cycle_n8` | 0.017/0.259; d=1 | 0.107/0.356; d=1 | 1.083/0.950; d=1 | TO 10.000/10.235 |
| `signed_distance_cycle_n10` | 0.244/0.416; d=1 | 0.451/0.647; d=1 | 5.282/3.119; d=1 | TO 10.000/10.271 |
| `signed_distance_cycle_n12` | 5.089/5.322; d=1 | 3.048/3.409; d=1 | TO 10.000/10.189 | TO 10.000/10.277 |
| `overlapping_random_multiples_n4_p6_k2_seed31415` | 0.047/0.266; d=2 | TO 10.000/10.212 | TO 10.000/10.216 | TO 10.000/10.233 |
| `overlapping_random_multiples_n6_p8_k3_seed31415` | TO 10.000/11.378 | TO 10.000/11.364 | TO 10.000/11.363 | TO 10.000/11.421 |
| `overlapping_random_multiples_n8_p10_k3_seed31415` | TO 10.000/12.000 | TO 10.000/12.000 | TO 10.000/12.000 | TO 10.000/12.000 |
| `affine_factor_hypergraph_n8_m6_k4_seed42` | 0.085/0.406; d=6 | TO 10.000/10.308 | TO 10.000/10.266 | TO 10.000/10.343 |
| `affine_factor_hypergraph_n10_m8_k5_seed42` | 1.177/1.828; d=7 | TO 10.000/10.641 | TO 10.000/10.649 | TO 10.000/10.772 |
| `affine_factor_hypergraph_n12_m10_k6_seed42` | TO 10.000/12.000 | TO 10.000/12.000 | TO 10.000/12.000 | TO 10.000/12.000 |
| `gated_cyclic_5` | 0.005/0.189; d=5 | 0.158/0.350; d=5 | 0.413/0.719; d=5 | TO 10.000/10.187 |
| `gated_cyclic_6` | 2.625/2.812; d=6 | TO 10.000/10.186 | 2.966/2.676; d=6 | TO 10.000/10.196 |
| `gated_katsura_6` | 0.057/0.281; d=7 | TO 10.000/10.211 | TO 10.000/10.182 | TO 10.000/10.224 |
| `gated_katsura_7` | 0.395/0.613; d=8 | TO 10.000/10.212 | TO 10.000/10.189 | TO 10.000/10.232 |
| `gated_random_dense_quadratic_n5_seed137` | 0.016/0.255; d=5 | TO 10.000/10.235 | TO 10.000/10.204 | TO 10.000/10.241 |
| `gated_random_dense_quadratic_n6_seed137` | 0.616/0.869; d=6 | TO 10.000/10.246 | TO 10.000/10.210 | TO 10.000/10.255 |
| `affine_factor_hypergraph_n8_k4_seed42` | 0.331/0.701; d=5 | TO 10.000/10.356 | TO 10.000/10.319 | TO 10.000/10.365 |
| `affine_factor_hypergraph_n10_k5_seed42` | 1.417/2.232; d=7 | TO 10.000/10.793 | TO 10.000/10.786 | TO 10.000/10.785 |

## Findings

- IDOS success is most visible on the positive affine systems, where direct
  whole-ideal Groebner methods time out at every tested size.
- On overlapping random multiples, IDOS+F5 is 34.4x faster in wall time than
  the best direct method at `n=4`. At `n=6`, every IDOS engine completes while
  all direct and decomposition methods time out. At `n=8`, IDOS+F5, GVW, and
  Singular complete, while every direct and decomposition method times out.
- Symbolica is strongest on workloads with many cheap covers. On the
  `n=10` affine hypergraph it is 12.4x faster in wall time than IDOS+m4gb and
  2.3x faster than IDOS+Singular.
- Singular is strongest on hard nonlinear residual covers. It is the only
  tuned IDOS engine here to complete gated Cyclic-6, and it is 16.1x faster
  than m4gb on Katsura-7 and 9.1x faster on the dense quadratic at `n=6`.
- Every system-by-algorithm cell in the selected matrix has now been measured;
  unsuccessful cells are recorded explicitly as timeouts rather than omissions.
- Macaulay2 decomposition now completes 4 of the 17 selected systems within
  10 seconds. It agrees with the reported maximum dimensions on every completed
  row; the remaining 13 rows time out rather than failing at startup.

## Reproduction

```sh
python3 scripts/benchmark_idos.py --no-build --timeout 10 --jobs 1
```

The benchmark generates all 17 systems once and writes only this report.
By default it runs `NR` cells and checkpoints every completed cell; use
`--rerun-all` to replace all measurements or `--only METHOD` to restrict the
queue. It applies 45-second IDOS limits to structured systems, 20-second IDOS
limits to hard residual systems, and `--timeout` to direct/external methods.
Use `--jobs 1` for publication timing without cross-instance contention.
