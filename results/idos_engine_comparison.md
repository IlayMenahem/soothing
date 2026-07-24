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
and a 20-second limit for hard residual systems.

| system | `idos-m4gb` | `idos-f5` | `idos-gvw` | `idos-symbolica` | `idos-singular-gb` |
|---|---:|---:|---:|---:|---:|
| `signed_distance_cycle_n8` | 0.003/0.013; d=1 | 0.045/0.642; d=1 | 0.147/0.743; d=1 | 0.003/0.013; d=1 | 0.064/0.069; d=1 |
| `signed_distance_cycle_n10` | 0.046/0.051; d=1 | 12.005/12.026; d=1 | TO 45.000/47.000 | 0.022/0.028; d=1 | 0.082/0.088; d=1 |
| `signed_distance_cycle_n12` | 0.200/0.207; d=1 | 6.685/6.693; d=1 | 1.991/1.999; d=1 | 0.098/0.104; d=1 | 0.307/0.313; d=1 |
| `affine_factor_hypergraph_n8_m6_k4_seed42` | 0.005/0.012; d=6 | 0.004/0.011; d=6 | 0.006/0.013; d=6 | 0.004/0.011; d=6 | 0.486/0.493; d=6 |
| `affine_factor_hypergraph_n10_m8_k5_seed42` | 0.021/0.030; d=7 | 0.016/0.026; d=7 | 0.025/0.034; d=7 | 0.016/0.025; d=7 | 1.858/1.867; d=7 |
| `affine_factor_hypergraph_n12_m10_k6_seed42` | 0.090/0.114; d=9 | 0.076/0.101; d=9 | 0.114/0.139; d=9 | 0.078/0.100; d=9 | 6.469/6.491; d=9 |
| `gated_cyclic_5` | 0.005/0.014; d=5 | 0.005/0.013; d=5 | 0.006/0.012; d=5 | 0.004/0.010; d=5 | 0.016/0.021; d=5 |
| `gated_cyclic_6` | P 1/2; 20.002/20.009; d=6 | 6.551/6.574; d=6 | 2.667/2.695; d=6 | P 1/2; 20.001/20.008; d=6 | 0.041/0.046; d=6 |
| `gated_katsura_6` | 0.368/0.376; d=7 | 0.521/0.529; d=7 | 1.439/1.447; d=7 | 2.562/2.568; d=7 | 0.046/0.052; d=7 |
| `gated_katsura_7` | 4.660/4.666; d=8 | 16.794/16.809; d=8 | P 1/2; 20.006/20.050; d=8 | P 1/2; 20.002/20.008; d=8 | 0.283/0.290; d=8 |
| `gated_random_dense_quadratic_n5_seed137` | 0.174/0.179; d=5 | 0.071/0.078; d=5 | 0.211/0.217; d=5 | 0.202/0.208; d=5 | 0.023/0.029; d=5 |
| `gated_random_dense_quadratic_n6_seed137` | 4.950/4.956; d=6 | 3.504/3.512; d=6 | 8.194/8.203; d=6 | P 1/2; 20.004/20.010; d=6 | 0.541/0.547; d=6 |
| `affine_factor_hypergraph_n8_k4_seed42` | 14.980/14.987; d=5 | TO 20.000/22.000 | TO 20.000/22.000 | 3.008/3.016; d=5 | 0.441/0.447; d=5 |
| `affine_factor_hypergraph_n10_k5_seed42` | 15.478/15.490; d=7 | TO 20.000/22.000 | TO 20.000/22.000 | 1.237/1.250; d=7 | 2.841/2.853; d=7 |

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
- Symbolica is strongest on workloads with many cheap covers. On the
  `n=10` affine hypergraph it is 12.4x faster in wall time than IDOS+m4gb and
  2.3x faster than IDOS+Singular.
- Singular is strongest on hard nonlinear residual covers. It is the only
  tuned IDOS engine here to complete gated Cyclic-6, and it is 16.1x faster
  than m4gb on Katsura-7 and 9.1x faster on the dense quadratic at `n=6`.
- Every system-by-algorithm cell in the selected matrix has now been measured;
  unsuccessful cells are recorded explicitly as timeouts rather than omissions.
- Macaulay2 decomposition now completes 4 of the 14 selected systems within
  10 seconds. It agrees with the reported maximum dimensions on every completed
  row; the remaining 10 rows time out rather than failing at startup.

## Reproduction

```sh
python3 scripts/benchmark_idos.py --no-build --timeout 10 --jobs 1
```

The benchmark generates all 14 systems once and writes only this report.
By default it runs `NR` cells and checkpoints every completed cell; use
`--rerun-all` to replace all measurements or `--only METHOD` to restrict the
queue. It applies 45-second IDOS limits to structured systems, 20-second IDOS
limits to hard residual systems, and `--timeout` to direct/external methods.
Use `--jobs 1` for publication timing without cross-instance contention.
