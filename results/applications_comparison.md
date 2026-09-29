# Application systems

Best completed method per class; factor branching and variable zeroing give T/F in seconds, HomotopyContinuation.jl and decompositions give the component count in parentheses.

| System | dim | factor branching | variable zeroing | direct GB | decomposition | numerical |
|---|---|---|---|---|---|---|
| nash_generic_n4 | 1 | branching-m4gb 0.137/0.003 | zeroing-f5 0.603/0.466 | singular-gb-lex 0.657 | singular-facstd 0.074 (26) | homotopy-jl 28.903 (53) |
| nash_generic_n5 | 1 | branching-singular-gb 1.970/0.010 | zeroing-singular-gb 3.450/1.501 | timeout | timeout | homotopy-jl 36.906 (291) |
| nash_generic_n6 | None | timeout | timeout | timeout | timeout | timeout |
| nash_degenerate_n4 | 1 | branching-symbolica 0.107/0.003 | zeroing-symbolica 0.604/0.498 | singular-gb-lex 0.016 | singular-facstd 0.009 (11) | homotopy-jl 28.967 (22) |
| nash_degenerate_n5 | 1 | branching-singular-gb 1.121/0.010 | zeroing-singular-gb 2.737/1.640 | timeout | timeout | homotopy-jl 35.424 (144) |
| nash_degenerate_n6 | None | timeout | timeout | timeout | timeout | timeout |
| box_qp_n6 | 0 | branching-gvw 2.660/0.006 | timeout | f5-grevlex 0.026 | singular-primdecGTZ 0.621 (729) | homotopy-jl 29.105 (726) |
| box_qp_n7 | 0 | branching-gvw 6.466/0.003 | timeout | timeout | singular-facstd 13.037 (1534) | homotopy-jl 40.261 (2083) |
| box_qp_n8 | 0 | branching-symbolica 19.034/0.006 | timeout | timeout | timeout | timeout |
| lotka_volterra_n8 | 0 | branching-symbolica 0.652/0.003 | zeroing-symbolica 2.876/2.171 | f5-grevlex 1.428 | singular-facstd 0.257 (240) | homotopy-jl 28.826 (224) |
| lotka_volterra_n10 | 0 | branching-symbolica 2.989/0.006 | timeout | timeout | singular-facstd 6.284 (984) | homotopy-jl 33.932 (948) |
| lotka_volterra_n12 | 0 | branching-gvw 8.635/0.006 | timeout | timeout | timeout | homotopy-jl 49.536 (4004) |

Completed runs with a wrong dimension, excluded above:

