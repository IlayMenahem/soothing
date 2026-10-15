# Primary decomposition of the p-zeroed alpha systems

Systems: `systems/system_gravfermscal_EH_d5_alpha_{1,1_2,1_4,2,4}.txt` with the 17 variables whose names contain `p`, other than KappaRefFinal, RabKappaRefFinal and LambdaOverKappa, set to 0 (DhDphp1, DhDphp2, DhDphp3, DhDphp4, DhDphp5, DhDphp6, DpDph1, DpDph2, DpDphh1, DpDphh2, DpDphh3, DpDphh4, DppDh1, DppDh2, pph, pphh1, pphh2), and without the last equation, the Rabinowitsch equation KappaRefFinal*RabKappaRefFinal = 1. Each has 31 equations in 15 variables (Lg2 occurs only in terms with a zeroed variable and drops out). KappaRefFinal != 0 is imposed by saturation, so the decomposed ideal is J = <P> : KappaRefFinal^oo; the Rabinowitsch and whole-ideal rows reach J's components differently, as described below.

Setup: Singular 4.4.1 and Macaulay2 1.26.06 on an Apple M4 Pro (14 cores, 24 GB), 120 s per run. Each run is one process holding one solver slot of `scripts/_harness.py`, and up to four ran at once. Times are seconds of the whole method inside the session (Singular, via `scripts/factor_variants.py`; for Alg. PrimaryBranching from IdealReduce to the last component) or M2's `cpuTime`, excluding start-up and input conversion. Singular rings are degrevlex with the variables in name order; Alg. PrimaryBranching refines in lex with the min-fill order and computes every Gröbner basis in degrevlex.

## Times

| Method | alpha_1 | alpha_1_2 | alpha_1_4 | alpha_2 | alpha_4 |
|---|---|---|---|---|---|
| Alg. PrimaryBranching | 0.162 | 0.165 | 0.165 | 0.164 | 0.166 |
| Alg. PrimaryBranching, no separators | 0.165 | 0.168 | 0.172 | 0.170 | 0.169 |
| `primdecGTZ`, saturated | 0.006 | 0.006 | 0.006 | 0.006 | 0.006 |
| `primdecSY`, saturated | 0.004 | 0.004 | 0.004 | 0.004 | 0.004 |
| M2 `primaryDecomposition`, saturated | 0.005 | 0.005 | 0.005 | 0.005 | 0.005 |
| `primdecGTZ`, Rabinowitsch | TO | TO | TO | TO | TO |
| `primdecSY`, Rabinowitsch | 0.019 | 0.018 | 0.017 | 0.017 | 0.017 |
| `primdecGTZ` of <P> | TO | TO | TO | TO | TO |
| `primdecSY` of <P> | TO | TO | TO | TO | TO |
| standalone baseline `singular-primdecGTZ` (lex) | TO | TO | TO | TO | TO |

TO = timeout at 120 s. The last row is the standalone baseline of `physical_table.py` (`_harness.py singular --task decompose --nonzero KappaRefFinal`), which saturates and decomposes in lex and was run separately.

Methods:

- **Alg. PrimaryBranching**: IdealReduce with KappaRefFinal nonzero, J = saturation of the reduced system, minimal primes by Alg. Branching with `minAssGTZ`, then the optional separator lines; `primdecGTZ` runs only if the isolated components do not intersect to J.
- **Alg. PrimaryBranching, no separators**: the same without the optional lines, as for the physical system: `primdecGTZ` of J, components marked by the minimal primes.
- **`primdecGTZ`, saturated**: Singular `sat` by KappaRefFinal, then `primdecGTZ`.
- **`primdecSY`, saturated**: Singular `sat` by KappaRefFinal, then `primdecSY`.
- **M2 `primaryDecomposition`, saturated**: Macaulay2 `saturate` by KappaRefFinal, then `primaryDecomposition`.
- **`primdecGTZ`, Rabinowitsch**: `primdecGTZ` of <P, 1 - s*KappaRefFinal>, s eliminated from each component.
- **`primdecSY`, Rabinowitsch**: `primdecSY` of <P, 1 - s*KappaRefFinal>, s eliminated from each component.
- **`primdecGTZ` of <P>**: `primdecGTZ` of <P> itself, keeping the components whose radical omits KappaRefFinal.
- **`primdecSY` of <P>**: `primdecSY` of <P> itself, keeping the components whose radical omits KappaRefFinal.

## Result

30 of 45 runs finished. Every finished run returns the same components as Alg. PrimaryBranching (equal reduced Gröbner bases).

All five systems have the same decomposition. Alg. IdealReduce turns the 31 equations into 27 generators in 2 rounds, and their saturation by KappaRefFinal has 3 generators. The reduced ideal has two minimal primes, one of which contains KappaRefFinal, so J has 1. The separator lines find that the isolated components intersect to J, so `primdecGTZ` does not run.

| # | type | dim | degree of primary | primary component | radical |
|---|---|---|---|---|---|
| 1 | isolated | 12 | 2 | `<Mf,LambdaRefFinal,m^2>` | `<m,Mf,LambdaRefFinal>` |

J is primary but not prime. Its solution set is m = Mf = LambdaRefFinal = 0, with KappaRefFinal != 0 and the other 11 variables free, and m has multiplicity 2.
