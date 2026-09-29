# Application systems

Best completed method per class; factor branching and variable zeroing give T/F in seconds, HomotopyContinuation.jl and decompositions give the component count in parentheses.

| System | dim | factor branching | variable zeroing | direct GB | decomposition | numerical |
|---|---|---|---|---|---|---|
| nash_generic_n4 | 1 | branching-m4gb 0.137/0.003 | zeroing-f5 0.603/0.466 | singular-gb-lex 0.657 | singular-facstd 0.074 (26) | homotopy-jl 28.903 (53) |
| nash_generic_n5 | 1 | branching-singular-gb 1.970/0.010 | zeroing-singular-gb 3.450/1.501 | timeout | timeout | homotopy-jl 36.906 (291) |
| nash_generic_n6 | None | timeout | timeout | timeout | timeout | timeout |
| nash_degenerate_n4 | 1 | branching-f5 0.117/0.003 | zeroing-gvw 0.718/0.595 | singular-gb-lex 0.016 | singular-facstd 0.009 (11) | homotopy-jl 28.967 (22) |
| nash_degenerate_n5 | 1 | branching-singular-gb 1.121/0.010 | zeroing-singular-gb 2.737/1.640 | timeout | timeout | homotopy-jl 35.424 (144) |
| nash_degenerate_n6 | None | timeout | timeout | timeout | timeout | timeout |
| box_qp_n6 | 0 | branching-gvw 2.660/0.006 | timeout | f5-grevlex 0.026 | singular-primdecGTZ 0.621 (729) | homotopy-jl 29.105 (726) |
| box_qp_n7 | 0 | branching-gvw 6.466/0.003 | timeout | timeout | singular-facstd 13.037 (1534) | homotopy-jl 40.261 (2083) |
| box_qp_n8 | 0 | branching-gvw 19.406/0.003 | timeout | timeout | timeout | timeout |
| lotka_volterra_n8 | 0 | branching-f5 0.789/0.006 | zeroing-gvw 3.127/2.410 | f5-grevlex 1.428 | singular-facstd 0.257 (240) | homotopy-jl 28.826 (224) |
| lotka_volterra_n10 | 0 | branching-f5 3.049/0.006 | timeout | timeout | singular-facstd 6.284 (984) | homotopy-jl 33.932 (948) |
| lotka_volterra_n12 | 0 | branching-gvw 8.635/0.006 | timeout | timeout | timeout | homotopy-jl 49.536 (4004) |

Completed runs excluded above (Symbolica's F4, or a wrong dimension):

- box_qp_n6 branching-symbolica: dimension 0, reference 0
- box_qp_n6 builtin-grevlex: dimension 0, reference 0
- box_qp_n7 branching-symbolica: dimension 0, reference 0
- box_qp_n8 branching-symbolica: dimension 0, reference 0
- lotka_volterra_n10 branching-symbolica: dimension 0, reference 0
- lotka_volterra_n12 branching-symbolica: dimension 0, reference 0
- lotka_volterra_n8 branching-symbolica: dimension 0, reference 0
- lotka_volterra_n8 zeroing-symbolica: dimension None, reference 0
- nash_degenerate_n4 branching-symbolica: dimension 1, reference 1
- nash_degenerate_n4 builtin-grevlex: dimension 1, reference 1
- nash_degenerate_n4 zeroing-symbolica: dimension None, reference 1
- nash_degenerate_n5 branching-symbolica: dimension 1, reference 1
- nash_generic_n4 branching-symbolica: dimension 1, reference 1
- nash_generic_n4 builtin-grevlex: dimension 1, reference 1
- nash_generic_n4 zeroing-symbolica: dimension None, reference 1
