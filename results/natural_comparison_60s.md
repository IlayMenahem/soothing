# Systems not designed for factor branching (60 s limit)

Each Singular tool is compared with factor branching (FB) and variable
zeroing (VZ) that use the same tool as their branch solver. *Rescued*: the
tool alone times out and FB or VZ completes; *faster*/*slower*: the faster of
FB and VZ differs by at least 2x and 1 s;
*lost*: the tool alone completes and both FB and VZ time out. Times of
the tools alone are their internal timers; FB, VZ and Macaulay2 are
timed by process wall time, including factorization and one Singular
start per branch. TO = time limit.
*FB branches* counts the terminal branches of Algorithm Refine; with one
branch, FB differs from the tool alone only by the lex interreduction it
applies first. *[splits]* counts the zero/nonzero splits of VZ.

| Family | systems | std (dp): rescued / faster / slower / lost | facstd: rescued / faster / slower / lost | primdecGTZ: rescued / faster / slower / lost |
|---|---|---|---|---|
| adjacent-minors | 1 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 1 / 0 / 0 / 0 |
| odebase-ss | 3 | 0 / 0 / 2 / 0 | 1 / 1 / 0 / 0 | 2 / 0 / 0 / 0 |
| odebase-ssc | 2 | 1 / 1 / 0 / 0 | 1 / 1 / 0 / 0 | 1 / 0 / 0 / 0 |
| symbolicdata | 17 | 7 / 4 / 0 / 0 | 10 / 1 / 0 / 0 | 4 / 0 / 0 / 0 |

## Fastest method per class (systems run with every method)

| System | FB branches | factor branching | variable zeroing | direct GB | decomposition |
|---|---|---|---|---|---|
| adjacent_minors_4x4 | 1 | branching-m4gb 0.02 | zeroing-m4gb 0.03 | singular-gb-dp 0.00 | singular-facstd 0.01 |
| Behnke | 31 | branching-gvw 3.71 | TO | m2-gb-lex 54.22 | m2-decompose 58.92 |
| Butcher | 1 | branching-symbolica 0.03 | zeroing-symbolica 0.04 | singular-gb-dp 0.04 | singular-facstd 0.62 |
| Geometry.Chou.329_1 | 2 | branching-singular-facstd 2.94 | zeroing-singular-facstd 10.43 | TO | TO |
| Geometry.Chou.337_1 | 1 | TO | zeroing-singular-facstd 8.23 | TO | TO |
| Geometry.Chou.353_1 | 1 | branching-singular-facstd 3.48 | zeroing-singular-gb 19.65 | TO | TO |
| Geometry.Chou.428_1 | 3 | branching-singular-gb 0.44 | zeroing-singular-gb 0.46 | singular-gb-dp 2.70 | singular-facstd 23.73 |
| Geometry.Chou.451_1 | 12 | branching-singular-facstd 0.41 | zeroing-singular-facstd 1.02 | TO | TO |
| Geometry.Chou.459_1 | 1 | branching-singular-gb 29.66 | zeroing-singular-gb 5.65 | TO | TO |
| Geometry.Chou.460_1 | 1 | TO | zeroing-singular-facstd 14.00 | TO | TO |
| Geometry.Chou.468_1 | 1 | branching-singular-gb 23.63 | zeroing-singular-gb 10.80 | singular-gb-dp 28.52 | TO |
| Gerdt-85 | 9 | branching-symbolica 0.04 | zeroing-gvw 0.25 | singular-gb-dp 0.00 | singular-facstd 0.00 |
| Singular.rcyclic_12 | 3 | branching-symbolica 0.03 | TO | singular-gb-dp 0.06 | TO |
| Singular.rcyclic_14 | 3 | branching-m4gb 0.08 | TO | singular-gb-dp 0.52 | TO |
| Singular.rcyclic_16 | 3 | branching-singular-gb 0.30 | TO | singular-gb-dp 7.35 | TO |
| Symmetric_6 | 30 | branching-symbolica 0.14 | TO | m2-gb-lex 5.16 | singular-facstd 0.08 |
| BIOMD0000000220_ss | 320 | branching-symbolica 1.87 | TO | m4gb-grevlex 0.00 | m2-decompose 43.48 |
| BIOMD0000000333_ss | 77 | branching-m4gb 1.79 | TO | singular-gb-dp 1.40 | m2-decompose 8.07 |
| BIOMD0000000335_ss | 42 | branching-symbolica 0.27 | zeroing-symbolica 0.70 | singular-gb-dp 0.02 | singular-facstd 0.57 |
| BIOMD0000000430_ssc | 1 | branching-singular-gb 0.17 | zeroing-singular-gb 0.15 | TO | TO |
| BIOMD0000000651_ssc | 4 | branching-singular-gb 0.15 | zeroing-singular-gb 0.21 | singular-gb-dp 17.72 | singular-facstd 8.52 |

## Systems where FB or VZ beats every tool alone

| System | family | vars/eqs | FB branches | std (dp) alone | FB | VZ [splits] | facstd alone | FB | VZ [splits] | primdecGTZ alone | FB | VZ [splits] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Behnke | symbolicdata | 5/5 | 31 | TO | **9.60** | TO [5] | TO | TO | TO [25] | TO | **10.18** | TO [5] |
| Ellipsoid_-2 | symbolicdata | 10/4 | 1 | 9.88 | **0.05** | 0.06 | TO | 0.73 | **0.68** | TO | TO | TO [14] |
| Geometry.Chou.329_1 | symbolicdata | 19/16 | 2 | TO | TO | TO [5] | TO | **2.94** | 10.43 [1] | TO | TO | TO [27] |
| Geometry.Chou.337_1 | symbolicdata | 15/12 | 1 | TO | TO | **8.27** [1] | TO | TO | **8.23** [1] | TO | TO | TO [23] |
| Geometry.Chou.353_1 | symbolicdata | 13/10 | 1 | TO | **16.93** | 19.65 [3] | TO | **3.48** | TO [2] | TO | TO | TO [22] |
| Geometry.Chou.428_1 | symbolicdata | 11/8 | 3 | 2.70 | **0.44** | 0.46 | 23.73 | **0.65** | 0.69 | TO | TO | TO [21] |
| Geometry.Chou.451_1 | symbolicdata | 17/14 | 12 | TO | **1.73** | 2.30 | TO | **0.41** | 1.02 | TO | TO | TO [24] |
| Geometry.Chou.459_1 | symbolicdata | 16/13 | 1 | TO | 29.66 | **5.65** [1] | TO | TO | TO [25] | TO | TO | TO [29] |
| Geometry.Chou.460_1 | symbolicdata | 15/12 | 1 | TO | TO | **14.01** [3] | TO | TO | **14.00** [3] | TO | TO | TO [20] |
| Geometry.Chou.468_1 | symbolicdata | 13/8 | 1 | 28.52 | 23.63 | **10.80** [3] | TO | TO | **13.00** [3] | TO | TO | TO [24] |
| Singular.rcyclic_16 | symbolicdata | 16/15 | 3 | 7.35 | **0.30** | TO | TO | **0.33** | TO | TO | TO | TO |
| BIOMD0000000430_ssc | odebase-ssc | 27/29 | 1 | TO | 0.17 | **0.15** | TO | 0.58 | **0.57** | TO | TO | TO [45] |
| BIOMD0000000651_ssc | odebase-ssc | 29/42 | 4 | 17.72 | **0.15** | 0.21 | 8.52 | **0.53** | 0.60 | TO | 6.13 | **6.07** |

## Systems where FB or VZ rescues or speeds up a tool

| System | family | vars/eqs | FB branches | std (dp) alone | FB | VZ [splits] | facstd alone | FB | VZ [splits] | primdecGTZ alone | FB | VZ [splits] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| adjacent_minors_4x4 | adjacent-minors | 16/9 | 1 | 0.00 | 0.02 | 0.04 | 0.01 | 0.03 | 0.04 | TO | TO | **6.89** [2] |
| Behnke | symbolicdata | 5/5 | 31 | TO | **9.60** | TO [5] | TO | TO | TO [25] | TO | **10.18** | TO [5] |
| Butcher | symbolicdata | 8/8 | 1 | 0.04 | 0.03 | 0.04 | 0.62 | 0.04 | 0.05 | TO | 19.36 | **7.15** [3] |
| Ellipsoid_-2 | symbolicdata | 10/4 | 1 | 9.88 | **0.05** | 0.06 | TO | 0.73 | **0.68** | TO | TO | TO [14] |
| Geometry.Chou.329_1 | symbolicdata | 19/16 | 2 | TO | TO | TO [5] | TO | **2.94** | 10.43 [1] | TO | TO | TO [27] |
| Geometry.Chou.337_1 | symbolicdata | 15/12 | 1 | TO | TO | **8.27** [1] | TO | TO | **8.23** [1] | TO | TO | TO [23] |
| Geometry.Chou.353_1 | symbolicdata | 13/10 | 1 | TO | **16.93** | 19.65 [3] | TO | **3.48** | TO [2] | TO | TO | TO [22] |
| Geometry.Chou.428_1 | symbolicdata | 11/8 | 3 | 2.70 | **0.44** | 0.46 | 23.73 | **0.65** | 0.69 | TO | TO | TO [21] |
| Geometry.Chou.451_1 | symbolicdata | 17/14 | 12 | TO | **1.73** | 2.30 | TO | **0.41** | 1.02 | TO | TO | TO [24] |
| Geometry.Chou.459_1 | symbolicdata | 16/13 | 1 | TO | 29.66 | **5.65** [1] | TO | TO | TO [25] | TO | TO | TO [29] |
| Geometry.Chou.460_1 | symbolicdata | 15/12 | 1 | TO | TO | **14.01** [3] | TO | TO | **14.00** [3] | TO | TO | TO [20] |
| Geometry.Chou.468_1 | symbolicdata | 13/8 | 1 | 28.52 | 23.63 | **10.80** [3] | TO | TO | **13.00** [3] | TO | TO | TO [24] |
| Gerdt-85 | symbolicdata | 7/13 | 9 | 0.00 | 0.10 | 0.32 | 0.00 | 0.10 | 0.31 | TO | **0.18** | 0.38 |
| Singular.rcyclic_12 | symbolicdata | 12/11 | 3 | 0.06 | 0.05 | TO | TO | **0.05** | TO | TO | TO | TO |
| Singular.rcyclic_14 | symbolicdata | 14/13 | 3 | 0.52 | 0.09 | TO | TO | **0.10** | TO | TO | TO | TO |
| Singular.rcyclic_16 | symbolicdata | 16/15 | 3 | 7.35 | **0.30** | TO | TO | **0.33** | TO | TO | TO | TO |
| Symmetric_6 | symbolicdata | 5/5 | 30 | TO | **0.36** | TO | 0.08 | 0.36 | TO | TO | **0.66** | TO |
| BIOMD0000000220_ss | odebase-ss | 58/58 | 320 | 0.00 | 3.57 | TO | TO | **4.67** | TO | TO | **19.33** | TO |
| BIOMD0000000333_ss | odebase-ss | 54/54 | 77 | 1.40 | 5.06 | TO | 50.70 | **11.20** | TO | TO | TO | TO |
| BIOMD0000000335_ss | odebase-ss | 34/34 | 42 | 0.02 | 0.51 | 0.91 | 0.57 | 0.73 | 1.15 | TO | **5.97** | 6.54 |
| BIOMD0000000430_ssc | odebase-ssc | 27/29 | 1 | TO | 0.17 | **0.15** | TO | 0.58 | **0.57** | TO | TO | TO [45] |
| BIOMD0000000651_ssc | odebase-ssc | 29/42 | 4 | 17.72 | **0.15** | 0.21 | 8.52 | **0.53** | 0.60 | TO | 6.13 | **6.07** |

## Systems where FB and VZ are slower or lose a completed run

| System | family | vars/eqs | FB branches | std (dp) alone | FB | VZ [splits] | facstd alone | FB | VZ [splits] | primdecGTZ alone | FB | VZ [splits] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BIOMD0000000220_ss | odebase-ss | 58/58 | 320 | 0.00 | 3.57 | TO | TO | **4.67** | TO | TO | **19.33** | TO |
| BIOMD0000000333_ss | odebase-ss | 54/54 | 77 | 1.40 | 5.06 | TO | 50.70 | **11.20** | TO | TO | TO | TO |
