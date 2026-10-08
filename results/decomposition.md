# Factor-branching variants on the size-graded families

Wall time in seconds with a 60-second limit, and what each completed run found.
Every method runs in the same harness (`scripts/factor_variants.py`): one Python process
driving one Singular 4.4.1 process. *Rust Alg. Branching* is the fastest recorded backend
from `scaling_results.jsonl`. Cells, as (object, decomposition):

- (sqrt I, covering): Alg. Branching vs facstd
- (I, monolithic): IdealReduce + std vs std
- (I, covering): IdealRefine
- (sqrt I, irreducible): PrimeBranching vs minAssGTZ
- (I, irreducible): PrimaryBranching vs primdecGTZ

br = branches or factorizing-GB components, gens = generators, comp = primary components,
emb = embedded components. TO = timeout, — = not run (an earlier size failed).

## rcyclic

| size | Rust Alg. Branching | Alg. Branching | facstd | IdealReduce + std | std | IdealRefine | PrimeBranching | minAssGTZ | PrimaryBranching | primdecGTZ |
|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 0.02 | 0.06 (10 br) | 0.29 (8 br) | 0.06 (7→7 gens) | 0.05 (38 gens) | TO | 0.08 (3 primes) | 10.64 (3 primes) | TO (isolated 0.37: 3 comp, has emb) | TO |
| 9 | 0.02 | 0.07 (14 br) | TO | 0.06 (8→8 gens) | 0.06 (55 gens) | — | 0.09 (3 primes) | TO | TO | — |
| 10 | 0.02 | 0.07 (20 br) | — | 0.07 (9→9 gens) | 0.06 (78 gens) | — | 0.11 (4 primes) | — | — | — |
| 11 | 0.02 | 0.07 (19 br) | — | 0.07 (10→10 gens) | 0.07 (90 gens) | — | 0.18 (3 primes) | — | — | — |
| 12 | 0.02 | 0.07 (11 br) | — | 0.11 (11→11 gens) | 0.11 (124 gens) | — | 1.47 (3 primes) | — | — | — |
| 13 | 0.03 | 0.08 (12 br) | — | 0.20 (12→12 gens) | 0.21 (164 gens) | — | 18.02 (4 primes) | — | — | — |
| 14 | 0.03 | 0.14 (13 br) | — | 0.55 (13→13 gens) | 0.58 (217 gens) | — | 21.22 (3 primes) | — | — | — |
| 15 | 0.04 | 0.24 (14 br) | — | 2.00 (14→14 gens) | 2.22 (278 gens) | — | TO | — | — | — |
| 16 | 0.06 | 0.41 (15 br) | — | 7.02 (15→15 gens) | 7.81 (351 gens) | — | — | — | — | — |
| 17 | 0.09 | 0.90 (16 br) | — | 38.96 (16→16 gens) | 42.88 (437 gens) | — | — | — | — | — |
| 18 | 0.09 | 0.85 (17 br) | — | TO (reduced 0.00: 17→17 gens) | TO | — | — | — | — | — |
| 19 | 0.11 | 1.69 (18 br) | — | — | — | — | — | — | — | — |

## nash_generic

| size | Rust Alg. Branching | Alg. Branching | facstd | IdealReduce + std | std | IdealRefine | PrimeBranching | minAssGTZ | PrimaryBranching | primdecGTZ |
|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 0.04 | 0.06 (13 br) | 0.06 (10 br) | 0.06 (3→3 gens) | 0.05 (10 gens) | 0.17 (13 leaves) | 0.06 (13 primes) | 0.07 (13 primes) | 0.07 (13 comp) | 0.06 (13 comp) |
| 4 | 0.14 | 0.08 (40 br) | 0.13 (26 br) | 0.13 (4→4 gens) | 0.13 (39 gens) | 2.74 (38 leaves) | 0.12 (37 primes) | TO | TO | TO |
| 5 | 1.38 | 0.36 (152 br) | TO | TO (reduced 0.01: 5→5 gens) | TO | TO | TO | — | — | — |
| 6 | TO | TO | — | — | — | — | — | — | — | — |

## nash_degenerate

| size | Rust Alg. Branching | Alg. Branching | facstd | IdealReduce + std | std | IdealRefine | PrimeBranching | minAssGTZ | PrimaryBranching | primdecGTZ |
|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 0.04 | 0.06 (10 br) | 0.06 (7 br) | 0.06 (3→3 gens) | 0.05 (7 gens) | 0.22 (8 leaves) | 0.06 (7 primes) | 0.07 (7 primes) | 0.09 (8 comp, 1 emb) | 0.07 (8 comp, 1 emb) |
| 4 | 0.09 | 0.08 (28 br) | 0.06 (11 br) | 0.06 (4→4 gens) | 0.06 (27 gens) | 0.37 (15 leaves) | 0.09 (16 primes) | TO | 6.12 (18 comp, 2 emb) | 5.00 (18 comp, 2 emb) |
| 5 | 0.71 | 0.18 (104 br) | TO | TO (reduced 0.01: 5→5 gens) | TO | TO | TO | — | TO | TO |
| 6 | TO | TO | — | — | — | — | — | — | — | — |

## box_qp

| size | Rust Alg. Branching | Alg. Branching | facstd | IdealReduce + std | std | IdealRefine | PrimeBranching | minAssGTZ | PrimaryBranching | primdecGTZ |
|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 0.24 | 0.09 (81 br) | 0.06 (77 br) | 0.06 (4→4 gens) | 0.05 (13 gens) | 0.21 (81 leaves) | 0.09 (81 primes) | 0.17 (81 primes) | 0.09 (81 comp) | 0.13 (81 comp) |
| 5 | 0.70 | 0.15 (243 br) | 0.18 (242 br) | 0.10 (5→5 gens) | 0.09 (36 gens) | 7.28 (242 leaves) | 0.17 (242 primes) | 17.00 (242 primes) | 0.72 (242 comp) | 16.20 (242 comp) |
| 6 | 2.17 | 0.36 (729 br) | 1.14 (729 br) | 0.06 (6→6 gens) | 0.06 (24 gens) | 10.29 (729 leaves) | 0.45 (729 primes) | 1.56 (729 primes) | 0.42 (729 comp) | 3.17 (729 comp) |
| 7 | 6.49 | 1.22 (2187 br) | 12.96 (1534 br) | TO (reduced 0.01: 7→7 gens) | TO | TO | 1.56 (2187 primes) | TO | TO | TO |
| 8 | 20.42 | 4.10 (6561 br) | TO | — | — | — | 5.74 (6534 primes) | — | — | — |
| 9 | TO | 15.13 (19683 br) | — | — | — | — | 23.05 (19602 primes) | — | — | — |
| 10 | TO | 53.51 (58932 br) | — | — | — | — | TO | — | — | — |
| 11 | TO | TO | — | — | — | — | — | — | — | — |

## lotka_volterra

| size | Rust Alg. Branching | Alg. Branching | facstd | IdealReduce + std | std | IdealRefine | PrimeBranching | minAssGTZ | PrimaryBranching | primdecGTZ |
|---|---|---|---|---|---|---|---|---|---|---|
| 6 | 0.19 | 0.09 (62 br) | 0.06 (58 br) | 0.06 (6→6 gens) | 0.05 (16 gens) | 0.40 (57 leaves) | 0.10 (58 primes) | 0.16 (58 primes) | 0.10 (58 comp) | 0.13 (58 comp) |
| 8 | 0.81 | 0.22 (248 br) | 0.32 (240 br) | 0.56 (8→8 gens) | 0.56 (84 gens) | 14.47 (225 leaves) | 0.25 (240 primes) | TO | 1.87 (240 comp) | TO |
| 10 | 3.00 | 0.95 (1024 br) | 6.32 (984 br) | TO (reduced 0.01: 10→10 gens) | TO | TO | 1.06 (984 primes) | — | TO | — |
| 12 | 13.15 | 4.18 (4084 br) | TO | — | — | — | 5.29 (4044 primes) | — | — | — |
| 14 | TO | 22.30 (16341 br) | — | — | — | — | 31.61 (16279 primes) | — | — | — |
| 16 | TO | TO | — | — | — | — | TO | — | — | — |

## Agreement

45 variant/baseline pairs finished on the same system; 0 disagree.
Compared: largest dimension (coverings), reduced Gröbner basis (I), minimal primes (sqrt I),
and the number of components with the isolated components (primary decompositions).

