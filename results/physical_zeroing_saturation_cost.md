# Physical system: cost of saturating the inequality branches

Each row is one inequality branch `x != 0` created by variable zeroing with
M4GB (5 s per branch) and `--ideal-saturation` on `systems/system.txt`, in
the order the run creates them. "Conditions" lists the variables required to
be nonzero (all on the `pph != 0` path except rows 2 and 3, which lie under
`pph = 0`). The ideal is the parent's Rabinowitsch-free generators. Each
entry is the standalone Singular 4.4.1 time in seconds, one process per
program, nine in parallel on the 14-core M4 Pro; `unit` means the result is
the unit ideal, so the branch is empty.

- `sat`: Singular's `sat(I, prod)` (what `--ideal-saturation` runs).
- iterated `sat`: `sat` by one variable at a time.
- elimination: `<I, 1 - t_i x_i> ∩ K[x]` with one `t_i` per condition, in the
  block order `(dp(k), dp(n))`.
- Rabinowitsch std: `std(<I, 1 - t_i x_i>)` in degrevlex with the `t_i` last,
  the separate encoding of the paper. It decides emptiness but does not
  produce the saturation.

| # | Conditions | `sat` | iterated `sat` | elimination | Rabinowitsch std |
|---|---|---|---|---|---|
| 1 | pph | >300 | >60 | >60 | >60 |
| 2 | DpDph2 (pph = 0) | >120 | >60 | 53.36 unit | 0.22 unit |
| 3 | DpDph2, DpDph1 (pph = 0) | 0.58 unit | >60 | >60 | 0.03 unit |
| 4 | pph, DpDph2 | >120 | >60 | >60 | >60 |
| 5 | + xr | >120 | >60 | >60 | >60 |
| 6 | + DpDph1 | >120 | >60 | >60 | 5.30 unit |
| 7 | + DpDphh4 | >120 | >60 | >60 | 0.19 unit |
| 8 | + DpDphh3 | >120 | >60 | >60 | 0.19 unit |
| 9 | + pphh2 | >120 | >60 | >60 | 0.21 unit |

Rows 6–9 are empty, and the Rabinowitsch encoding shows it in at most 5.3 s,
whereas every way of computing the saturated ideal takes more than a minute.
Row 3 is the only saturation that finishes within the 5 s branch limit.
Rows 1, 4 and 5 are the branches on which M4GB also times out with the
Rabinowitsch encoding.
With one Rabinowitsch variable for the product of the seven conditions
(row 9), `std` also needs more than 120 s, as does `eliminate`.
