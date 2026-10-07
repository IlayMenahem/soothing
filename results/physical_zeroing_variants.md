# Physical system: variable zeroing variants

Entries are `T/F` in seconds (120 s total, 5 s per branch); `sat a/b unit` counts the saturations that returned the unit ideal out of those attempted.

| Backend | rabinowitsch | saturation | rabinowitsch+prune | saturation+prune |
|---|---|---|---|---|
| m4gb | $24.616/0.270$ complete: 13 splits | $63.532/0.290$ complete: 9 splits (sat 1/9 unit) | $24.828/0.288$ complete: 13 splits | $63.652/0.359$ complete: 9 splits (sat 1/9 unit) |
| f5 | $timeout/0.282$ partial: 35 splits, 11 unresolved | $timeout/0.336$ partial: 16 splits, 5 unresolved (sat 3/16 unit) | $timeout/0.381$ partial: 35 splits, 12 unresolved | $timeout/0.404$ partial: 16 splits, 5 unresolved (sat 3/16 unit) |
| gvw | $timeout/0.313$ partial: 35 splits, 11 unresolved | $timeout/0.354$ partial: 16 splits, 5 unresolved (sat 3/16 unit) | $timeout/0.336$ partial: 33 splits, 11 unresolved | $timeout/0.382$ partial: 17 splits, 5 unresolved (sat 3/17 unit) |
| symbolica | $timeout/0.344$ partial: 32 splits, 8 unresolved | $timeout/0.440$ partial: 16 splits, 6 unresolved (sat 1/16 unit) | $timeout/0.308$ partial: 32 splits, 8 unresolved | $timeout/0.400$ partial: 16 splits, 6 unresolved (sat 1/16 unit) |
| singular | $17.816/0.336$ complete: 7 splits | $52.740/0.345$ complete: 7 splits (sat 0/7 unit) | $17.998/0.427$ complete: 7 splits | $52.920/0.441$ complete: 7 splits (sat 0/7 unit) |
| facstd | $1.150/0.327$ complete: 0 splits | $1.179/0.348$ complete: 0 splits | $1.238/0.392$ complete: 0 splits | $1.318/0.452$ complete: 0 splits |
