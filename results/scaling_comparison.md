# Scaling of factor branching on size-graded families

Wall time in seconds of the fastest completed method per class. Within a family,
runs are sequential; families may run in parallel (`--families`), one driver each.
A method that times out or fails at one size is not run at larger sizes.
*Dimensions* lists the distinct largest dimensions reported by completed runs.
A size marked † has at least one run recorded under a noted machine condition (the `condition` field of its rows); treat its times as unreliable.

## rcyclic

| size | FB branches | factor branching | best baseline | ratio | dimensions |
|---|---|---|---|---|---|
| 8 | 3 | 0.02 (branching-symbolica) | 0.01 (gvw-grevlex) | 0.4x | 5 |
| 9 | 3 | 0.02 (branching-symbolica) | 0.02 (gvw-grevlex) | 0.8x | 6 |
| 10 | 3 | 0.02 (branching-symbolica) | 0.04 (m4gb-grevlex) | 2.3x | 7 |
| 11 | 3 | 0.02 (branching-symbolica) | 0.08 (m4gb-grevlex) | 4.7x | 8 |
| 12 | 3 | 0.02 (branching-symbolica) | 0.20 (m4gb-grevlex) | 9.4x | 9 |
| 13 | 3 | 0.03 (branching-symbolica) | 0.34 (singular-gb-dp) | 13.2x | 10 |
| 14 | 3 | 0.03 (branching-symbolica) | 0.68 (singular-gb-dp) | 23.6x | 11 |
| 15 | 3 | 0.04 (branching-symbolica) | 2.20 (singular-gb-dp) | 56.3x | 12 |
| 16 | 3 | 0.06 (branching-singular-gb) | 7.15 (singular-gb-dp) | 115.6x | 13 |
| 17 | 3 | 0.09 (branching-singular-gb) | 19.76 (m2-gb-lex) | 216.8x | 14 |
| 18 | 3 | 0.09 (branching-singular-gb) | 34.60 (m2-gb-lex) | 372.4x | 15 |
| 19 | 3 | 0.11 (branching-singular-gb) | TO | only FB | 16 |

## nash_generic

| size | FB branches | factor branching | best baseline | ratio | dimensions |
|---|---|---|---|---|---|
| 3 | 13 | 0.04 (branching-f5) | 0.01 (f5-grevlex) | 0.1x | 0 |
| 4 | 43 | 0.14 (branching-gvw) | 0.26 (singular-facstd) | 1.9x | 1 |
| 5 | 148 | 1.38 (branching-singular-gb) | TO | only FB | 1 |
| 6 |  | TO | TO | — |  |

## nash_degenerate

| size | FB branches | factor branching | best baseline | ratio | dimensions |
|---|---|---|---|---|---|
| 3 | 13 | 0.04 (branching-m4gb) | 0.01 (gvw-grevlex) | 0.1x | 1 |
| 4 | 28 | 0.09 (branching-f5) | 0.20 (singular-facstd) | 2.4x | 1 |
| 5 | 84 | 0.71 (branching-singular-gb) | TO | only FB | 1 |
| 6 |  | TO | TO | — |  |

## box_qp

| size | FB branches | factor branching | best baseline | ratio | dimensions |
|---|---|---|---|---|---|
| 4 | 81 | 0.24 (branching-symbolica) | 0.01 (f5-grevlex) | 0.0x | 0 |
| 5 | 242 | 0.70 (branching-f5) | 0.22 (singular-gb-dp) | 0.3x | 0 |
| 6 | 729 | 2.17 (branching-m4gb) | 0.03 (f5-grevlex) | 0.0x | 0 |
| 7 | 2187 | 6.49 (branching-f5) | 12.63 (singular-facstd) | 1.9x | 0 |
| 8 | 6534 | 20.42 (branching-f5) | TO | only FB | 0 |
| 9 |  | TO | TO | — |  |

## lotka_volterra

| size | FB branches | factor branching | best baseline | ratio | dimensions |
|---|---|---|---|---|---|
| 6 | 58 | 0.19 (branching-m4gb) | 0.01 (f5-grevlex) | 0.0x | 0 |
| 8 | 240 | 0.81 (branching-symbolica) | 0.46 (singular-facstd) | 0.6x | 0 |
| 10 | 984 | 3.00 (branching-gvw) | 6.43 (singular-facstd) | 2.1x | 0 |
| 12 | 4044 | 13.15 (branching-f5) | TO | only FB | 0 |
| 14 |  | TO | TO | — |  |

