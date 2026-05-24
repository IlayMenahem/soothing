# Benchmark summary

## Primary: per-baseline status and time (seconds)

| system | n_eqs | n_vars | homotopy-jl | m2-decompose | m2-gb-lex | milp-cover | singular-gb-lex | singular-primdecGTZ | symbolica-f5-lex | symbolica-gb-lex | symbolica-gvw-lex |
|---|---:|---:|---|---|---|---|---|---|---|---|---|
| system | 29 | 22 | timeout | timeout | timeout | ok (0.11s) | timeout | timeout | timeout | timeout | timeout |

## Secondary: dimension findings

| system | ref_dim (Z3+Jac) | method | found_dim | shortfall |
|---|---:|---|---:|---:|
| system | 9 | milp-cover | 2 | 7 |
