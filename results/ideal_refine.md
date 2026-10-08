# Alg. IdealRefine vs std on the baseline systems

Wall time in seconds and what each completed run found, in the harness of
`scripts/factor_variants.py`. TO = timeout. *exact*: the leaves intersect to I (untimed check).

| system | IdealRefine | std | exact |
|---|---|---|---|
| signed_distance_cycle_n12 | TO | 0.35 (76 gens) | — |
| signed_distance_cycle_n14 | TO | 4.23 (210 gens) | — |
| signed_distance_cycle_n16 | TO | TO | — |
| overlapping_random_multiples_n4_p6_k2_seed31415 | 0.32 (3 leaves) | 0.10 (18 gens) | True |
| overlapping_random_multiples_n6_p8_k3_seed31415 | TO | TO | — |
| overlapping_random_multiples_n8_p10_k3_seed31415 | TO | TO | — |
| affine_factor_hypergraph_n8_m6_k4_seed42 | TO | 16.28 (48 gens) | — |
| affine_factor_hypergraph_n10_m8_k5_seed42 | TO | TO | — |
| affine_factor_hypergraph_n12_m10_k6_seed42 | TO | TO | — |
| gated_cyclic_5 | 0.09 (2 leaves) | 0.06 (20 gens) | True |
| gated_cyclic_6 | 6.11 (2 leaves) | 0.15 (45 gens) | True |
| gated_cyclic_7 | TO | TO | — |
| gated_katsura_6 | 0.35 (2 leaves) | 0.10 (41 gens) | True |
| gated_katsura_7 | 2.37 (2 leaves) | 0.43 (74 gens) | True |
| gated_katsura_8 | 23.12 (2 leaves) | 3.43 (143 gens) | True |
| gated_random_dense_quadratic_n5_seed137 | 0.14 (2 leaves) | 0.07 (21 gens) | True |
| gated_random_dense_quadratic_n6_seed137 | 2.39 (2 leaves) | 0.68 (39 gens) | True |
| gated_random_dense_quadratic_n7_seed137 | TO | 45.17 (67 gens) | — |
| affine_factor_hypergraph_n8_k4_seed42 | TO | TO | — |
| affine_factor_hypergraph_n10_k5_seed42 | TO | TO | — |
