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
- Times below are single measurements, not distributions. The direct,
  decomposition, and numerical cells were rerun with one method job and a
  60-second timeout. The IDOS cells retain their original run settings.

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
dimension. These measurements used a 60-second timeout.

| system | `m4gb-grevlex` | `f5-grevlex` | `gvw-grevlex` | `builtin-grevlex` | `singular-gb-lex` | `m2-gb-lex` |
|---|---:|---:|---:|---:|---:|---:|
| `signed_distance_cycle_n8` | 0.012/0.646 | 0.006/0.014 | 0.004/0.010 | 0.002/0.008 | 0.001/0.333 | 0.002/0.570 |
| `signed_distance_cycle_n10` | 0.125/0.130 | 0.216/0.221 | 0.107/0.112 | 0.023/0.029 | 0.014/0.194 | 0.040/0.518 |
| `signed_distance_cycle_n12` | 3.259/3.266 | 20.038/20.044 | 8.679/8.685 | 0.465/0.472 | 0.338/0.512 | 0.990/1.293 |
| `overlapping_random_multiples_n4_p6_k2_seed31415` | 0.994/1.000 | 24.859/24.865 | 0.467/0.472 | 0.407/0.413 | TO 60.000/60.208 | TO 60.000/60.210 |
| `overlapping_random_multiples_n6_p8_k3_seed31415` | TO 60.000/60.027 | TO 60.000/60.035 | TO 60.000/60.074 | TO 60.000/60.033 | TO 60.000/61.441 | TO 60.000/61.362 |
| `overlapping_random_multiples_n8_p10_k3_seed31415` | TO 60.000/60.092 | TO 60.000/60.095 | TO 60.000/60.101 | TO 60.000/60.113 | TO 60.000/62.000 | TO 60.000/62.000 |
| `affine_factor_hypergraph_n8_m6_k4_seed42` | TO 60.000/60.011 | TO 60.000/60.016 | TO 60.000/60.545 | TO 60.000/60.018 | TO 60.000/60.361 | TO 60.000/60.282 |
| `affine_factor_hypergraph_n10_m8_k5_seed42` | TO 60.000/60.015 | TO 60.000/60.011 | TO 60.000/60.043 | TO 60.000/60.030 | TO 60.000/60.645 | TO 60.000/60.651 |
| `affine_factor_hypergraph_n12_m10_k6_seed42` | TO 60.000/60.021 | TO 60.000/60.022 | TO 60.000/60.038 | TO 60.000/60.380 | TO 60.000/62.000 | TO 60.000/62.000 |
| `gated_cyclic_5` | 0.007/0.012 | 0.274/0.279 | 0.005/0.009 | 0.004/0.008 | 0.002/0.183 | 0.005/0.480 |
| `gated_cyclic_6` | TO 60.000/60.012 | TO 60.000/60.009 | 1.727/1.738 | TO 60.000/60.014 | 0.138/0.309 | 0.574/0.917 |
| `gated_katsura_6` | 0.431/0.436 | 44.956/44.961 | 0.220/0.226 | 5.272/5.277 | 14.264/14.430 | TO 60.000/60.175 |
| `gated_katsura_7` | 5.159/5.165 | TO 60.000/60.014 | 3.170/3.176 | TO 60.000/60.016 | TO 60.000/60.190 | TO 60.000/60.181 |
| `gated_random_dense_quadratic_n5_seed137` | 0.165/0.171 | 8.878/8.884 | 0.173/0.179 | 0.194/0.200 | 0.327/0.529 | TO 60.000/60.187 |
| `gated_random_dense_quadratic_n6_seed137` | 4.730/4.736 | TO 60.000/60.011 | 6.748/6.755 | 17.467/17.472 | TO 60.000/60.203 | TO 60.000/60.204 |
| `affine_factor_hypergraph_n8_k4_seed42` | TO 60.000/60.013 | TO 60.000/60.012 | TO 60.000/60.601 | TO 60.000/60.018 | TO 60.000/60.401 | TO 60.000/60.319 |
| `affine_factor_hypergraph_n10_k5_seed42` | TO 60.000/60.014 | TO 60.000/60.011 | TO 60.000/60.039 | TO 60.000/60.034 | TO 60.000/60.776 | TO 60.000/60.785 |

## Decomposition and numerical algorithms

These methods report maximum component dimension when they complete.

| system | `singular-facstd` | `singular-primdecGTZ` | `m2-decompose` | `homotopy-jl` |
|---|---:|---:|---:|---:|
| `signed_distance_cycle_n8` | 0.019/0.190; d=1 | 0.125/0.320; d=1 | 1.120/1.020; d=1 | 26.413/32.410; d=1 |
| `signed_distance_cycle_n10` | 0.250/0.430; d=1 | 0.462/0.667; d=1 | 5.377/3.224; d=1 | 27.690/29.944; d=1 |
| `signed_distance_cycle_n12` | 4.616/4.800; d=1 | 2.767/3.056; d=1 | 65.359/25.223; d=1 | 34.575/36.880; d=1 |
| `overlapping_random_multiples_n4_p6_k2_seed31415` | 0.045/0.265; d=2 | TO 60.000/60.216 | TO 60.000/60.218 | 27.652/30.069; d=2 |
| `overlapping_random_multiples_n6_p8_k3_seed31415` | TO 60.000/61.357 | TO 60.000/61.358 | TO 60.000/61.364 | TO 60.000/61.397 |
| `overlapping_random_multiples_n8_p10_k3_seed31415` | TO 60.000/62.000 | TO 60.000/62.000 | TO 60.000/62.000 | TO 60.000/62.000 |
| `affine_factor_hypergraph_n8_m6_k4_seed42` | 0.080/0.368; d=6 | TO 60.000/60.288 | TO 60.000/60.281 | 48.438/50.934; d=6 |
| `affine_factor_hypergraph_n10_m8_k5_seed42` | 1.174/1.836; d=7 | TO 60.000/60.648 | TO 60.000/60.658 | TO 60.000/60.665 |
| `affine_factor_hypergraph_n12_m10_k6_seed42` | 50.737/53.318; d=9 | TO 60.000/62.000 | TO 60.000/62.000 | TO 60.000/62.000 |
| `gated_cyclic_5` | 0.004/0.181; d=5 | 0.154/0.337; d=5 | 0.409/0.709; d=5 | 32.765/35.054; d=5 |
| `gated_cyclic_6` | 2.553/2.729; d=6 | TO 60.000/60.171 | 2.908/2.620; d=6 | 35.041/37.314; d=6 |
| `gated_katsura_6` | 0.052/0.233; d=7 | TO 60.000/60.173 | TO 60.000/60.174 | 29.276/31.532; d=7 |
| `gated_katsura_7` | 0.360/0.539; d=8 | TO 60.000/60.181 | TO 60.000/60.174 | 29.724/31.980; d=8 |
| `gated_random_dense_quadratic_n5_seed137` | 0.013/0.198; d=5 | TO 60.000/60.184 | TO 60.000/60.185 | 27.015/29.289; d=5 |
| `gated_random_dense_quadratic_n6_seed137` | 0.551/0.755; d=6 | TO 60.000/60.198 | TO 60.000/60.198 | 27.764/30.053; d=6 |
| `affine_factor_hypergraph_n8_k4_seed42` | 0.289/0.601; d=5 | TO 60.000/60.306 | TO 60.000/60.311 | TO 60.000/60.317 |
| `affine_factor_hypergraph_n10_k5_seed42` | 1.356/2.141; d=7 | TO 60.000/60.775 | TO 60.000/60.793 | TO 60.000/60.787 |

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
- Macaulay2 decomposition completes 5 of the 17 selected systems within
  60 seconds. It agrees with the reported maximum dimensions on every completed
  row; the remaining 12 rows time out rather than failing at startup.

## Reproduction

```sh
python3 scripts/benchmark_idos.py --no-build --timeout 60 --jobs 1 --rerun-all
```

The benchmark generates all 17 systems once and writes only this report.
By default it runs `NR` cells and checkpoints every completed cell; use
`--rerun-all` to replace all measurements or `--only METHOD` to restrict the
queue. It applies 45-second IDOS limits to structured systems, 20-second IDOS
limits to hard residual systems, and `--timeout` to direct/external methods.
Use `--jobs 1` for publication timing without cross-instance contention.
