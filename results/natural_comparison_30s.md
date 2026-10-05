# Systems not designed for factor branching (30 s limit)

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
| adjacent-minors | 10 | 0 / 0 / 1 / 0 | 0 / 0 / 0 / 0 | 4 / 0 / 0 / 0 |
| binomial-edge | 13 | 0 / 0 / 1 / 0 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |
| odebase-ss | 199 | 0 / 0 / 4 / 0 | 3 / 0 / 0 / 0 | 11 / 3 / 2 / 0 |
| odebase-ssc | 117 | 1 / 6 / 0 / 2 | 2 / 2 / 2 / 2 | 7 / 2 / 0 / 0 |
| permanents | 9 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |
| symbolicdata | 374 | 12 / 14 / 6 / 3 | 22 / 8 / 7 / 3 | 48 / 5 / 16 / 4 |

## Systems where FB or VZ beats every tool alone

| System | family | vars/eqs | FB branches | std (dp) alone | FB | VZ [splits] | facstd alone | FB | VZ [splits] | primdecGTZ alone | FB | VZ [splits] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Behnke | symbolicdata | 5/5 | 31 | TO | **13.23** | TO | TO | TO | TO | TO | **13.00** | TO |
| Ellipsoid_-2 | symbolicdata | 10/4 | 1 | 9.44 | **0.04** | 0.06 | TO | **0.76** | 0.77 | TO | TO | TO [8] |
| Geometry.Brocard_1 | symbolicdata | 13/11 | 1 | TO | **0.03** | 0.05 | 1.34 | **0.03** | 0.06 | TO | TO | TO [19] |
| Geometry.Chou.329_1 | symbolicdata | 19/16 | 2 | TO | TO | **14.97** [3] | TO | **3.48** | 5.63 [1] | TO | TO | TO [15] |
| Geometry.Chou.330_1 | symbolicdata | 19/12 | 1 | 3.06 | **0.66** | 0.70 | 10.71 | 9.83 | 7.79 [2] | TO | TO | TO [7] |
| Geometry.Chou.337_1 | symbolicdata | 15/12 | 1 | TO | TO | **7.18** [1] | TO | TO | **6.60** [1] | TO | TO | TO [11] |
| Geometry.Chou.353_1 | symbolicdata | 13/10 | 1 | TO | 18.03 | **14.75** [3] | TO | **3.87** | 6.93 [1] | TO | TO | TO [8] |
| Geometry.Chou.393_1 | symbolicdata | 11/8 | 1 | 3.74 | **0.44** | 0.47 | TO | TO | **3.70** [1] | TO | TO | TO [11] |
| Geometry.Chou.428_1 | symbolicdata | 11/8 | 3 | 3.23 | **0.49** | 0.56 | 28.70 | **0.70** | 0.80 | TO | TO | TO [11] |
| Geometry.Chou.451_1 | symbolicdata | 17/14 | 12 | TO | **2.00** | 2.56 | TO | **0.45** | 1.00 | TO | TO | TO [11] |
| Geometry.Chou.459_1 | symbolicdata | 16/13 | 1 | TO | TO | **6.02** [1] | TO | TO | TO [8] | TO | TO | TO [16] |
| Geometry.Chou.460_1 | symbolicdata | 15/12 | 1 | TO | TO | **13.98** [3] | TO | TO | **14.10** [3] | TO | TO | TO [7] |
| Geometry.Chou.468_1 | symbolicdata | 13/8 | 1 | 27.52 | 24.01 | **10.95** [3] | TO | TO | **13.47** [3] | TO | TO | TO [10] |
| Geometry.Chou.64_1 | symbolicdata | 12/8 | 2 | 1.19 | **0.17** | 0.20 | 2.15 | TO | 4.88 [1] | TO | TO | TO [13] |
| Paris.berth | symbolicdata | 3/2 | 1 | 1.07 | **0.06** | 0.07 | TO | TO | **3.49** [1] | TO | TO | TO [3] |
| Singular.rcyclic_12 | symbolicdata | 12/11 | 3 | 2.35 | **0.96** | TO | TO | **1.25** | TO | TO | TO | TO |
| Singular.rcyclic_13 | symbolicdata | 13/12 | 3 | 5.28 | **1.27** | TO | TO | **1.43** | TO | TO | TO | TO |
| Singular.rcyclic_14 | symbolicdata | 14/13 | 3 | TO | **2.34** | TO | TO | **2.88** | TO | TO | TO | TO |
| Singular.rcyclic_15 | symbolicdata | 15/14 | 3 | TO | **6.30** | TO | TO | **7.59** | TO | TO | TO | TO |
| Singular.rcyclic_16 | symbolicdata | 16/15 | 3 | TO | **11.32** | TO | TO | **12.83** | TO | TO | TO | TO |
| BIOMD0000000011_ssc | odebase-ssc | 22/29 | 1 | 3.21 | **1.07** | 1.52 | 5.22 | 4.62 | TO [22] | TO | TO | TO [22] |
| BIOMD0000000270_ssc | odebase-ssc | 33/41 | 1 | 15.46 | **1.17** | 1.26 | 19.05 | **3.87** | err | TO | TO | err |
| BIOMD0000000430_ssc | odebase-ssc | 27/29 | 1 | TO | **0.17** | 0.18 | TO | **0.67** | 0.69 | TO | TO | TO [24] |
| BIOMD0000000501_ssc | odebase-ssc | 35/37 | 1 | 5.51 | **1.51** | 1.56 | TO | **14.94** | TO [6] | TO | TO | TO [7] |
| BIOMD0000000651_ssc | odebase-ssc | 29/42 | 4 | 21.54 | **0.18** | 0.24 | 10.15 | **0.63** | 0.67 | TO | **7.18** | 7.25 |

## Systems where FB or VZ rescues or speeds up a tool

| System | family | vars/eqs | FB branches | std (dp) alone | FB | VZ [splits] | facstd alone | FB | VZ [splits] | primdecGTZ alone | FB | VZ [splits] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| adjacent_minors_3x4 | adjacent-minors | 12/6 | 1 | 0.00 | 0.04 | 0.06 | 0.00 | 0.03 | 0.08 | TO | TO | **3.40** [1] |
| adjacent_minors_3x5 | adjacent-minors | 15/8 | 1 | 0.00 | 0.03 | 0.06 | 0.00 | 0.03 | 0.05 | TO | TO | **3.89** [1] |
| adjacent_minors_3x6 | adjacent-minors | 18/10 | 1 | 0.00 | 0.04 | 0.07 | 0.01 | 0.06 | 0.09 | TO | TO | **11.94** [3] |
| adjacent_minors_4x4 | adjacent-minors | 16/9 | 1 | 0.00 | 0.04 | 0.06 | 0.00 | 0.04 | 0.07 | TO | TO | **7.42** [2] |
| Behnke | symbolicdata | 5/5 | 31 | TO | **13.23** | TO | TO | TO | TO | TO | **13.00** | TO |
| Bjoerk_8 | symbolicdata | 8/9 | 1 | 0.18 | 0.10 | 0.17 | TO | **0.26** | 0.33 | 5.58 | 4.23 | TO [8] |
| Butcher | symbolicdata | 8/8 | 1 | 0.05 | 0.04 | 0.06 | 0.75 | 0.04 | 0.08 | TO | 23.23 | **7.29** [3] |
| Czapor-86c | symbolicdata | 12/3 | 1 | 0.00 | 0.18 | 0.41 | 0.00 | 0.14 | 0.50 | TO | **0.38** | 0.75 |
| Czapor-86c_1 | symbolicdata | 9/3 | 1 | 0.00 | 0.17 | 0.38 | 0.02 | 0.26 | 0.44 | TO | TO | **9.99** [2] |
| Czapor-86c_2 | symbolicdata | 6/3 | 1 | 0.00 | 0.23 | 0.40 | 0.01 | 0.22 | 0.60 | TO | TO | **4.85** [1] |
| Ellipsoid_-2 | symbolicdata | 10/4 | 1 | 9.44 | **0.04** | 0.06 | TO | **0.76** | 0.77 | TO | TO | TO [8] |
| Ellipsoid_2 | symbolicdata | 10/4 | 1 | 0.00 | 0.02 | 0.04 | 0.01 | 0.03 | 0.05 | TO | TO | **6.93** [2] |
| Geometry.Biarc | symbolicdata | 8/4 | 1 | 0.00 | 0.02 | 0.04 | 0.00 | 0.02 | 0.04 | TO | TO | **16.15** [6] |
| Geometry.Brocard_1 | symbolicdata | 13/11 | 1 | TO | **0.03** | 0.05 | 1.34 | **0.03** | 0.06 | TO | TO | TO [19] |
| Geometry.Chou.100_1 | symbolicdata | 8/6 | 1 | 0.00 | 0.02 | 0.04 | 0.00 | 0.03 | 0.04 | TO | **0.51** | 0.54 |
| Geometry.Chou.116_1 | symbolicdata | 6/3 | 5 | 0.00 | 0.07 | 0.13 | 0.01 | 0.07 | 0.14 | TO | **0.39** | 0.46 |
| Geometry.Chou.128_1 | symbolicdata | 7/4 | 3 | 0.00 | 0.05 | 0.11 | 0.00 | 0.05 | 0.11 | TO | TO | **11.96** [3] |
| Geometry.Chou.139_1 | symbolicdata | 9/6 | 6 | 1.95 | **0.12** | 0.26 | 0.02 | 0.12 | 0.26 | TO | TO | **10.58** [3] |
| Geometry.Chou.152_1 | symbolicdata | 15/9 | 1 | 0.01 | 0.03 | 0.04 | 3.24 | **0.71** | **0.71** | TO | TO | TO [7] |
| Geometry.Chou.153_1 | symbolicdata | 8/5 | 1 | 0.00 | 0.02 | 0.04 | 0.00 | 0.02 | 0.04 | TO | TO | **3.22** [1] |
| Geometry.Chou.156_1 | symbolicdata | 7/4 | 2 | 0.00 | 0.03 | 0.07 | 0.01 | 0.04 | 0.08 | TO | **0.77** | 0.80 |
| Geometry.Chou.160_1 | symbolicdata | 9/6 | 2 | 0.18 | 0.15 | 0.19 | 3.10 | **0.21** | 0.24 | TO | TO | TO [10] |
| Geometry.Chou.247_1 | symbolicdata | 7/4 | 2 | 0.00 | 0.04 | 0.07 | 0.01 | 0.04 | 0.08 | TO | TO | **3.49** [1] |
| Geometry.Chou.274_2 | symbolicdata | 11/7 | 1 | 0.54 | 0.53 | 0.54 | TO | 8.85 | **4.57** [1] | TO | TO | TO [10] |
| Geometry.Chou.292_1 | symbolicdata | 11/7 | 7 | 0.00 | 0.10 | 0.23 | 0.00 | 0.10 | 0.23 | TO | **0.22** | 0.34 |
| Geometry.Chou.296_1 | symbolicdata | 11/7 | 1 | 0.00 | 0.02 | 0.04 | 0.00 | 0.02 | 0.04 | TO | **0.08** | 0.10 |
| Geometry.Chou.296_2 | symbolicdata | 13/9 | 2 | 0.00 | 0.03 | 0.07 | 0.01 | 0.05 | 0.08 | TO | TO | **10.27** [3] |
| Geometry.Chou.300_1 | symbolicdata | 9/6 | 2 | 0.00 | 0.03 | 0.08 | 0.00 | 0.04 | 0.08 | 25.09 | **0.10** | 0.13 |
| Geometry.Chou.301_2 | symbolicdata | 11/7 | 2 | 0.00 | 0.04 | 0.08 | 0.00 | 0.04 | 0.08 | TO | **0.12** | 0.15 |
| Geometry.Chou.303_1 | symbolicdata | 13/8 | 1 | 0.02 | 0.03 | 0.04 | 3.21 | **0.41** | 0.42 | TO | TO | TO [9] |
| Geometry.Chou.313_1 | symbolicdata | 15/11 | 1 | 0.00 | 0.02 | 0.04 | 0.00 | 0.03 | 0.05 | TO | **1.77** | 6.98 [2] |
| Geometry.Chou.329_1 | symbolicdata | 19/16 | 2 | TO | TO | **14.97** [3] | TO | **3.48** | 5.63 [1] | TO | TO | TO [15] |
| Geometry.Chou.330_1 | symbolicdata | 19/12 | 1 | 3.06 | **0.66** | 0.70 | 10.71 | 9.83 | 7.79 [2] | TO | TO | TO [7] |
| Geometry.Chou.337_1 | symbolicdata | 15/12 | 1 | TO | TO | **7.18** [1] | TO | TO | **6.60** [1] | TO | TO | TO [11] |
| Geometry.Chou.353_1 | symbolicdata | 13/10 | 1 | TO | 18.03 | **14.75** [3] | TO | **3.87** | 6.93 [1] | TO | TO | TO [8] |
| Geometry.Chou.35_1 | symbolicdata | 11/9 | 2 | 0.00 | 0.03 | 0.10 | 0.00 | 0.03 | 0.10 | TO | 4.31 | **3.37** [1] |
| Geometry.Chou.383_1 | symbolicdata | 12/9 | 2 | 1.81 | **0.05** | 0.09 | 0.61 | 0.13 | 0.18 | TO | TO | TO [12] |
| Geometry.Chou.387_1 | symbolicdata | 7/4 | 2 | 0.00 | 0.03 | 0.07 | 0.02 | 0.04 | 0.08 | TO | TO | **10.35** [3] |
| Geometry.Chou.388_1 | symbolicdata | 7/4 | 2 | 0.00 | 0.03 | 0.07 | 0.01 | 0.03 | 0.07 | TO | **0.18** | 0.21 |
| Geometry.Chou.389_1 | symbolicdata | 11/8 | 2 | 25.02 | **0.37** | 0.41 | 0.12 | 0.73 | 0.76 | TO | TO | TO [11] |
| Geometry.Chou.393_1 | symbolicdata | 11/8 | 1 | 3.74 | **0.44** | 0.47 | TO | TO | **3.70** [1] | TO | TO | TO [11] |
| Geometry.Chou.395_1 | symbolicdata | 8/5 | 1 | 0.00 | 0.03 | 0.05 | 0.02 | 0.04 | 0.05 | TO | 10.41 | **8.32** [2] |
| Geometry.Chou.398_1 | symbolicdata | 7/3 | 1 | 0.00 | 0.02 | 0.04 | 0.00 | 0.02 | 0.04 | 21.34 | **0.04** | 0.06 |
| Geometry.Chou.400_1 | symbolicdata | 12/9 | 1 | 0.01 | 0.03 | 0.04 | 0.76 | 0.06 | 0.08 | TO | TO | **12.33** [3] |
| Geometry.Chou.401_1 | symbolicdata | 13/7 | 1 | 0.94 | 0.46 | 0.45 | 5.94 | 2.65 | **2.61** | TO | TO | TO [12] |
| Geometry.Chou.418_1 | symbolicdata | 9/5 | 5 | 0.00 | 0.07 | 0.18 | 0.00 | 0.08 | 0.18 | TO | **0.18** | 0.30 |
| Geometry.Chou.425_1 | symbolicdata | 9/6 | 1 | 0.00 | 0.02 | 0.04 | 0.01 | 0.03 | 0.05 | TO | TO | **6.22** [1] |
| Geometry.Chou.426_1 | symbolicdata | 8/5 | 3 | 0.00 | 0.05 | 0.11 | 0.00 | 0.05 | 0.11 | TO | **0.12** | 0.18 |
| Geometry.Chou.428_1 | symbolicdata | 11/8 | 3 | 3.23 | **0.49** | 0.56 | 28.70 | **0.70** | 0.80 | TO | TO | TO [11] |
| Geometry.Chou.439_1 | symbolicdata | 7/4 | 1 | 0.00 | 0.02 | 0.04 | 0.01 | 0.04 | 0.05 | TO | TO | **3.94** [1] |
| Geometry.Chou.444_1 | symbolicdata | 13/10 | 6 | 0.01 | 0.09 | 0.22 | 0.01 | 0.10 | 0.24 | TO | TO | **10.24** [3] |
| Geometry.Chou.451_1 | symbolicdata | 17/14 | 12 | TO | **2.00** | 2.56 | TO | **0.45** | 1.00 | TO | TO | TO [11] |
| Geometry.Chou.459_1 | symbolicdata | 16/13 | 1 | TO | TO | **6.02** [1] | TO | TO | TO [8] | TO | TO | TO [16] |
| Geometry.Chou.460_1 | symbolicdata | 15/12 | 1 | TO | TO | **13.98** [3] | TO | TO | **14.10** [3] | TO | TO | TO [7] |
| Geometry.Chou.468_1 | symbolicdata | 13/8 | 1 | 27.52 | 24.01 | **10.95** [3] | TO | TO | **13.47** [3] | TO | TO | TO [10] |
| Geometry.Chou.475_1 | symbolicdata | 13/8 | 1 | 0.35 | 0.11 | 0.13 | TO | **0.34** | 0.37 | TO | TO | TO [8] |
| Geometry.Chou.477_1 | symbolicdata | 11/6 | 2 | 0.00 | 0.03 | 0.06 | 0.01 | 0.03 | 0.07 | 1.38 | **0.11** | 0.14 |
| Geometry.Chou.64_1 | symbolicdata | 12/8 | 2 | 1.19 | **0.17** | 0.20 | 2.15 | TO | 4.88 [1] | TO | TO | TO [13] |
| Geometry.Chou.94_2 | symbolicdata | 7/4 | 1 | 0.00 | 0.02 | 0.04 | 0.01 | 0.03 | 0.04 | TO | TO | **3.77** [1] |
| Geometry.GeneralizedFermatPoint_1 | symbolicdata | 8/5 | 1 | 0.00 | 0.02 | 0.04 | 0.01 | 0.04 | 0.06 | TO | TO | **3.82** [1] |
| Geometry.IMO.35_1 | symbolicdata | 9/6 | 4 | 0.00 | 0.06 | 0.10 | 0.01 | 0.07 | 0.12 | TO | TO | **18.76** [8] |
| Geometry.NapoleonTriangle_1 | symbolicdata | 9/6 | 1 | 0.01 | 0.02 | 0.04 | 0.02 | 0.03 | 0.05 | TO | TO | **10.62** [3] |
| Geometry.Simson_4 | symbolicdata | 11/7 | 1 | 0.00 | 0.02 | 0.04 | 0.01 | 0.03 | 0.05 | TO | TO | **3.52** [1] |
| Geometry.Steiner_2 | symbolicdata | 8/5 | 1 | 0.00 | 0.02 | 0.04 | 0.01 | 0.03 | 0.05 | TO | TO | **3.78** [1] |
| Gerdt-85 | symbolicdata | 7/13 | 9 | 0.00 | 0.12 | 0.38 | 0.00 | 0.13 | 0.38 | TO | **0.19** | 0.46 |
| Gerdt-85_1 | symbolicdata | 7/12 | 6 | 0.04 | 0.09 | 0.43 | 0.40 | 0.09 | 0.44 | TO | **0.15** | 0.53 |
| Gerdt-93a | symbolicdata | 6/3 | 1 | 0.00 | 0.02 | 0.05 | 0.00 | 0.02 | 0.05 | TO | TO | **3.21** [1] |
| Gonnet-83 | symbolicdata | 17/19 | 3 | 0.01 | 0.06 | 0.56 | 0.00 | 0.05 | 0.55 | 2.55 | **0.09** | 0.58 |
| Gonnet-83_1 | symbolicdata | 18/19 | 7 | 0.77 | 0.12 | 1.92 | 0.02 | 0.16 | 1.91 | TO | **0.20** | 1.87 |
| Klein | symbolicdata | 5/3 | 1 | 0.05 | 0.03 | 0.05 | TO | **0.06** | 0.08 | TO | **0.05** | 0.06 |
| Milnor1 | symbolicdata | 3/3 | 2 | 2.63 | **0.04** | 0.22 | 0.23 | 0.28 | 0.47 | TO | TO | TO [6] |
| Paris.berth | symbolicdata | 3/2 | 1 | 1.07 | **0.06** | 0.07 | TO | TO | **3.49** [1] | TO | TO | TO [3] |
| Pavelle | symbolicdata | 8/4 | 1 | 0.00 | 0.02 | 0.04 | 0.00 | 0.02 | 0.04 | TO | **0.03** | 0.05 |
| Pfister_1 | symbolicdata | 3/3 | 6 | 0.00 | 0.09 | 0.21 | 0.00 | 0.09 | 0.21 | TO | **4.82** | TO [3] |
| Robot-RR | symbolicdata | 15/16 | 2 | 0.08 | 0.47 | 1.57 | 0.15 | 0.81 | 1.61 | TO | **2.45** | 2.81 |
| Robot-Romin | symbolicdata | 12/7 | 1 | 0.00 | 0.28 | 0.56 | 0.01 | 0.48 | 0.29 | TO | TO | **6.63** [2] |
| Singular.gerhard_1 | symbolicdata | 5/3 | 7 | 0.83 | 1.12 | 4.49 | 4.04 | **1.74** | TO | TO | TO | TO [5] |
| Singular.rcyclic_10 | symbolicdata | 10/9 | 3 | 0.41 | 0.66 | TO | TO | **0.88** | TO | TO | TO | TO |
| Singular.rcyclic_11 | symbolicdata | 11/10 | 3 | 0.77 | 0.57 | TO | TO | **1.03** | TO | TO | TO | TO |
| Singular.rcyclic_12 | symbolicdata | 12/11 | 3 | 2.35 | **0.96** | TO | TO | **1.25** | TO | TO | TO | TO |
| Singular.rcyclic_13 | symbolicdata | 13/12 | 3 | 5.28 | **1.27** | TO | TO | **1.43** | TO | TO | TO | TO |
| Singular.rcyclic_14 | symbolicdata | 14/13 | 3 | TO | **2.34** | TO | TO | **2.88** | TO | TO | TO | TO |
| Singular.rcyclic_15 | symbolicdata | 15/14 | 3 | TO | **6.30** | TO | TO | **7.59** | TO | TO | TO | TO |
| Singular.rcyclic_16 | symbolicdata | 16/15 | 3 | TO | **11.32** | TO | TO | **12.83** | TO | TO | TO | TO |
| Singular.rcyclic_8 | symbolicdata | 8/7 | 3 | 0.00 | 0.49 | TO | 4.62 | **0.61** | TO | TO | **7.00** | TO |
| Singular.rcyclic_9 | symbolicdata | 9/8 | 3 | 0.04 | 0.34 | TO | TO | **0.33** | TO | TO | TO | TO |
| Symmetric_5 | symbolicdata | 5/5 | 31 | 0.11 | 5.29 | TO | TO | **4.50** | TO | 14.61 | 12.82 | TO |
| Symmetric_6 | symbolicdata | 5/5 | 30 | TO | **5.34** | TO | 1.20 | 4.00 | TO | TO | **13.14** | TO |
| TD-89 | symbolicdata | 4/3 | 1 | 0.01 | 0.32 | 0.36 | 0.02 | 0.26 | 0.50 | 2.12 | **0.45** | 0.61 |
| Vermeer | symbolicdata | 5/4 | 2 | 0.01 | 0.20 | 0.75 | 0.18 | 0.20 | 0.63 | TO | **0.98** | 2.06 |
| Vermeer_1 | symbolicdata | 5/4 | 1 | 0.00 | 0.41 | 0.34 | 0.07 | 0.14 | 0.43 | TO | **0.38** | 0.73 |
| Verschelde.virasoro | symbolicdata | 8/8 | 1 | 2.36 | **0.66** | 0.70 | 0.61 | 0.58 | 0.62 | TO | TO | TO [8] |
| BIOMD0000000011_ss | odebase-ss | 22/22 | 1 | 0.00 | 0.20 | 0.44 | 0.14 | 0.24 | 0.50 | TO | TO | **19.33** [1] |
| BIOMD0000000011_ssc | odebase-ssc | 22/29 | 1 | 3.21 | **1.07** | 1.52 | 5.22 | 4.62 | TO [22] | TO | TO | TO [22] |
| BIOMD0000000028_ss | odebase-ss | 16/16 | 1 | 0.04 | 0.34 | 0.42 | 0.11 | 0.27 | 0.54 | 3.59 | 1.59 | **1.32** |
| BIOMD0000000028_ssc | odebase-ssc | 16/19 | 1 | 0.27 | 0.28 | 0.59 | 0.72 | 0.49 | 0.85 | TO | **1.39** | 1.82 |
| BIOMD0000000030_ss | odebase-ss | 18/18 | 1 | 0.00 | 0.14 | 0.54 | 0.07 | 0.45 | 0.61 | 4.77 | **1.14** | 1.42 |
| BIOMD0000000030_ssc | odebase-ssc | 18/21 | 1 | 0.14 | 0.19 | 0.50 | 0.55 | 0.56 | 1.06 | 5.77 | **1.41** | 2.21 |
| BIOMD0000000038_ss | odebase-ss | 17/13 | 1 | 0.01 | 0.46 | 0.10 | 0.01 | 0.80 | 0.10 | TO | TO | **4.75** [1] |
| BIOMD0000000046_ss | odebase-ss | 16/12 | 2 | 0.01 | 0.10 | 0.17 | 0.01 | 0.09 | 0.13 | TO | TO | **4.19** [1] |
| BIOMD0000000102_ss | odebase-ss | 13/13 | 1 | 0.18 | 0.08 | 0.09 | TO | **0.76** | 0.77 | TO | TO | TO [11] |
| BIOMD0000000105_ss | odebase-ss | 39/28 | 0 | 0.00 | 0.01 | 0.05 | 0.00 | 0.01 | 0.04 | TO | **0.02** | 0.05 |
| BIOMD0000000105_ssc | odebase-ssc | 39/32 | 0 | 0.00 | 0.01 | 0.05 | 0.00 | 0.01 | 0.05 | TO | **0.02** | 0.05 |
| BIOMD0000000220_ss | odebase-ss | 58/58 | 320 | 0.00 | 6.79 | TO | TO | **6.96** | TO | TO | **27.12** | TO |
| BIOMD0000000230_ss | odebase-ss | 26/24 | 4 | 0.45 | 0.09 | 0.21 | 0.03 | 0.11 | 0.21 | TO | **0.22** | 0.33 |
| BIOMD0000000230_ssc | odebase-ssc | 26/25 | 4 | 2.90 | **0.12** | 0.26 | 0.05 | 0.13 | 0.26 | TO | TO | TO [10] |
| BIOMD0000000270_ssc | odebase-ssc | 33/41 | 1 | 15.46 | **1.17** | 1.26 | 19.05 | **3.87** | err | TO | TO | err |
| BIOMD0000000291_ssc | odebase-ssc | 7/7 | 1 | 0.04 | 0.03 | 0.05 | 0.07 | 0.04 | 0.06 | TO | **0.05** | 0.06 |
| BIOMD0000000333_ss | odebase-ss | 54/54 | 77 | 0.50 | 2.17 | TO | TO | **8.61** | TO | TO | TO | TO |
| BIOMD0000000335_ss | odebase-ss | 34/34 | 42 | 0.02 | 0.61 | 1.29 | 0.72 | 0.86 | 1.52 | TO | **7.09** | 7.76 |
| BIOMD0000000389_ssc | odebase-ssc | 22/23 | 4 | 0.02 | 0.07 | 0.37 | 0.03 | 0.09 | 0.39 | TO | TO | **4.37** [1] |
| BIOMD0000000430_ssc | odebase-ssc | 27/29 | 1 | TO | **0.17** | 0.18 | TO | **0.67** | 0.69 | TO | TO | TO [24] |
| BIOMD0000000431_ss | odebase-ss | 27/27 | 1 | 0.02 | 0.03 | 0.04 | 0.35 | 0.13 | 0.14 | TO | **0.39** | 0.40 |
| BIOMD0000000475_ssc | odebase-ssc | 23/30 | 4 | 0.06 | 0.07 | 0.18 | 0.05 | 0.09 | 0.20 | TO | 13.01 | **7.01** [2] |
| BIOMD0000000478_ss | odebase-ss | 33/29 | 8 | 0.00 | 0.11 | 0.40 | 0.14 | 0.14 | 0.42 | TO | TO | **18.35** [4] |
| BIOMD0000000478_ssc | odebase-ssc | 33/37 | 3 | 0.01 | 0.05 | 0.26 | 0.00 | 0.05 | 0.27 | TO | **0.08** | 0.30 |
| BIOMD0000000501_ssc | odebase-ssc | 35/37 | 1 | 5.51 | **1.51** | 1.56 | TO | **14.94** | TO [6] | TO | TO | TO [7] |
| BIOMD0000000581_ss | odebase-ss | 28/25 | 1 | 0.01 | 0.04 | 0.06 | 0.05 | 0.06 | 0.08 | 5.25 | **0.21** | 0.24 |
| BIOMD0000000599_ss | odebase-ss | 30/30 | 8 | 0.02 | 0.13 | 0.33 | 0.06 | 0.15 | 0.36 | TO | **0.29** | 0.51 |
| BIOMD0000000599_ssc | odebase-ssc | 30/33 | 3 | 0.04 | 0.07 | 0.14 | 0.08 | 0.08 | 0.15 | 13.82 | **0.13** | 0.21 |
| BIOMD0000000646_ssc | odebase-ssc | 11/19 | 1 | 7.85 | **2.94** | 2.95 | 1.52 | 12.77 | TO [17] | TO | TO | TO [11] |
| BIOMD0000000651_ss | odebase-ss | 29/29 | 1 | 0.00 | 0.03 | 0.05 | 0.00 | 0.03 | 0.05 | TO | **0.25** | 0.27 |
| BIOMD0000000651_ssc | odebase-ssc | 29/42 | 4 | 21.54 | **0.18** | 0.24 | 10.15 | **0.63** | 0.67 | TO | **7.18** | 7.25 |

## Systems where FB and VZ are slower or lose a completed run

| System | family | vars/eqs | FB branches | std (dp) alone | FB | VZ [splits] | facstd alone | FB | VZ [splits] | primdecGTZ alone | FB | VZ [splits] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| adjacent_minors_2x15 | adjacent-minors | 30/14 | 1 | 0.00 | 1.30 | 1.34 | 0.21 | 0.58 | 0.63 | TO | TO | TO [6] |
| binomial_edge_C14 | binomial-edge | 28/14 | 1 | 0.00 | 1.97 | 1.98 | 4.35 | 4.00 | TO [7] | TO | TO | TO [7] |
| Cassou_1 | symbolicdata | 4/4 | 1 | 0.04 | 0.07 | 0.11 | TO | TO | TO [4] | 8.97 | TO | TO [4] |
| Czapor-87_1 | symbolicdata | 4/4 | 1 | 0.00 | 0.32 | 0.46 | 0.02 | 0.31 | 0.46 | 0.09 | 3.40 | 2.50 |
| Czapor-87_2 | symbolicdata | 4/4 | 1 | 0.00 | 0.13 | 0.43 | 0.11 | 0.14 | 0.33 | 0.15 | 25.30 | TO [4] |
| DiscrC2 | symbolicdata | 12/3 | 1 | 1.28 | TO | 4.02 [1] | 7.12 | TO | TO [8] | TO | TO | TO [9] |
| Geometry.Chou.238_1 | symbolicdata | 11/8 | 2 | 2.21 | TO | TO [9] | 1.53 | TO | 3.83 [1] | TO | TO | TO [11] |
| Geometry.Chou.311_1 | symbolicdata | 15/11 | 1 | 0.36 | 1.52 | 1.52 | 0.46 | 0.47 | 0.47 | TO | TO | TO [10] |
| Geometry.Chou.317_1 | symbolicdata | 13/9 | 1 | 0.14 | 0.04 | 0.06 | 1.37 | 7.60 | 3.60 [1] | TO | TO | TO [13] |
| Geometry.Chou.376_1 | symbolicdata | 9/7 | 1 | 0.00 | 0.02 | 0.04 | 0.00 | 0.02 | 0.04 | 0.09 | 16.50 | 3.33 [1] |
| Geometry.Chou.377_1 | symbolicdata | 8/5 | 1 | 0.00 | 0.02 | 0.04 | 0.01 | 0.03 | 0.04 | 3.43 | TO | 8.82 [2] |
| Geometry.Chou.462_1 | symbolicdata | 7/3 | 1 | 0.00 | 0.02 | 0.03 | 0.00 | 0.02 | 0.03 | 11.33 | TO | TO [8] |
| Geometry.Chou.64_1 | symbolicdata | 12/8 | 2 | 1.19 | **0.17** | 0.20 | 2.15 | TO | 4.88 [1] | TO | TO | TO [13] |
| Geometry.Simson_5 | symbolicdata | 9/5 | 1 | 0.00 | 0.02 | 0.04 | 0.00 | 0.02 | 0.04 | 0.39 | 3.10 | 5.96 [1] |
| Hairer_2_1 | symbolicdata | 13/12 | 1 | 0.12 | 0.19 | 0.21 | 0.24 | 1.24 | 1.25 | TO | TO | TO [17] |
| Meintjes | symbolicdata | 12/5 | 1 | 0.10 | TO | TO [10] | 1.07 | TO | TO [11] | TO | TO | TO [11] |
| Pfister_2 | symbolicdata | 13/13 | 1 | 7.55 | TO | TO [5] | TO | TO | TO [7] | TO | TO | TO [6] |
| Roczen | symbolicdata | 10/22 | 24 | 0.01 | 4.38 | TO | 0.06 | 4.60 | TO | 0.97 | 9.16 | TO |
| SignalTheory.f744 | symbolicdata | 12/12 | 1 | 4.83 | 3.83 | 2.94 | 16.33 | TO | TO [12] | TO | TO | TO [12] |
| Singular.ecyclic_7 | symbolicdata | 43/7 | 1 | 1.68 | 3.94 | TO | TO | TO | TO [8] | TO | TO | TO [8] |
| Sym1_311 | symbolicdata | 3/3 | 5 | 0.00 | 0.85 | 1.85 | 0.00 | 0.52 | 2.36 | 0.07 | 1.18 | 2.84 |
| Sym1_321 | symbolicdata | 3/3 | 1 | 0.00 | 0.33 | 0.32 | 0.00 | 0.26 | 0.31 | 0.08 | 1.16 | 1.29 |
| Sym1_411 | symbolicdata | 3/3 | 5 | 0.00 | 0.72 | 2.04 | 0.00 | 0.84 | 2.20 | 0.23 | 1.99 | 2.45 |
| Sym3_3 | symbolicdata | 3/3 | 1 | 0.00 | 0.28 | 0.41 | 0.01 | 0.17 | 0.38 | 0.07 | 28.20 | TO [3] |
| Symmetric_5 | symbolicdata | 5/5 | 31 | 0.11 | 5.29 | TO | TO | **4.50** | TO | 14.61 | 12.82 | TO |
| Symmetric_6 | symbolicdata | 5/5 | 30 | TO | **5.34** | TO | 1.20 | 4.00 | TO | TO | **13.14** | TO |
| Twomat3 | symbolicdata | 18/8 | 1 | 0.20 | 7.38 | 4.96 [1] | 4.79 | TO | 6.32 [1] | TO | TO | TO [4] |
| Verschelde.eco7 | symbolicdata | 7/7 | 1 | 0.00 | 0.03 | 0.04 | 0.01 | 0.05 | 0.08 | 0.26 | 3.88 | 5.14 [3] |
| Wilfried | symbolicdata | 16/11 | 241 | 4.85 | 3.09 | TO | 1.25 | 3.25 | TO | TO | TO | TO |
| ZeroDim.example_12 | symbolicdata | 3/3 | 1 | 0.00 | 0.02 | 0.03 | 0.00 | 0.02 | 0.04 | 0.00 | 9.51 | TO [3] |
| ZeroDim.example_14 | symbolicdata | 4/4 | 1 | 0.00 | 0.02 | 0.03 | 0.00 | 0.02 | 0.04 | 0.01 | 10.74 | TO [4] |
| ZeroDim.example_27 | symbolicdata | 4/4 | 1 | 0.00 | 0.02 | 0.04 | 0.00 | 0.02 | 0.04 | 0.01 | 1.65 | 1.68 |
| ZeroDim.example_59 | symbolicdata | 4/4 | 1 | 0.00 | 0.02 | 0.04 | 0.00 | 0.02 | 0.04 | 0.01 | 1.71 | 1.73 |
| ZeroDim.example_60 | symbolicdata | 3/3 | 2 | 0.00 | 0.03 | 0.07 | 0.00 | 0.04 | 0.07 | 0.02 | TO | TO [3] |
| ZeroDim.example_61 | symbolicdata | 3/3 | 2 | 0.00 | 0.04 | 0.09 | 0.00 | 0.03 | 0.10 | 0.05 | 19.36 | TO [3] |
| ZeroDim.example_62 | symbolicdata | 3/3 | 1 | 0.00 | 0.02 | 0.04 | 0.01 | 0.03 | 0.05 | 0.20 | TO | TO [3] |
| BIOMD0000000002_ssc | odebase-ssc | 13/15 | - | 0.69 | TO | TO | 5.36 | TO | TO | TO | TO | TO |
| BIOMD0000000046_ssc | odebase-ssc | 16/14 | 1 | 0.03 | 0.09 | 0.12 | 0.04 | 1.52 | 1.47 | TO | TO | TO [12] |
| BIOMD0000000220_ss | odebase-ss | 58/58 | 320 | 0.00 | 6.79 | TO | TO | **6.96** | TO | TO | **27.12** | TO |
| BIOMD0000000226_ss | odebase-ss | 24/14 | 1 | 0.00 | 0.02 | 0.06 | 0.00 | 0.03 | 0.05 | 0.07 | TO | 3.27 [1] |
| BIOMD0000000333_ss | odebase-ss | 54/54 | 77 | 0.50 | 2.17 | TO | TO | **8.61** | TO | TO | TO | TO |
| BIOMD0000000362_ss | odebase-ss | 34/34 | 136 | 0.07 | 2.09 | TO | 3.12 | 2.19 | TO | TO | TO | TO |
| BIOMD0000000365_ssc | odebase-ssc | 30/35 | - | 3.21 | TO | TO | 2.60 | TO | TO | TO | TO | TO |
| BIOMD0000000646_ssc | odebase-ssc | 11/19 | 1 | 7.85 | **2.94** | 2.95 | 1.52 | 12.77 | TO [17] | TO | TO | TO [11] |
| BIOMD0000000770_ss | odebase-ss | 4/4 | 1 | 0.00 | 0.02 | 0.04 | 0.00 | 0.02 | 0.04 | 0.01 | 1.35 | 1.36 |
| BIOMD0000000951_ss | odebase-ss | 34/34 | 136 | 0.07 | 1.78 | TO | 3.03 | 2.06 | TO | TO | TO | TO |
