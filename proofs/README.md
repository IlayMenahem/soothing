# Lean formalization of the paper's proofs

Lean 4 + Mathlib (`v4.34.1`) proofs of every numbered result in `paper/sections/`.
There is no `sorry` and no custom axiom: `Audit.lean` checks that each result depends only on
`propext`, `Classical.choice` and `Quot.sound`.

```
cd proofs
lake exe cache get        # Mathlib build cache
lake build                # builds FactorBranching/
lake env lean Audit.lean  # prints the axioms of every result
```

## Paper result → Lean

| Paper | Lean | File |
|---|---|---|
| Nullstellensatz, `V(Q)=∅ ⇔ ⟨Q⟩=⟨1⟩` (preliminaries) | `vanishing_V_eq_radical`, `V_eq_empty_iff`, `sat_iff_V_nonempty` | `Basic.lean` |
| Interreduction does not shrink the leading-monomial ideal (preliminaries) | `IsInterRun.lmIdeal_le` | `Interreduce.lean` |
| Interreduction keeps `⟨Q⟩` and gives an interreduced set | `IsInterRun.span_eq`, `IsInterRun.isInterreduced` | `Interreduce.lean` |
| Interreduction terminates (multiset order) | `wellFounded_interStep`, `exists_isInterRun`, `Interreducer.ofTermination` | `InterreduceTermination.lean` |
| Theorem `thm:branch-decomposition` | `Factorizer.V_eq_iUnion_minHitting` (radical form `radical_eq_iInf_minHitting`) | `Factorization.lean` |
| Progress of one refinement step, `μ(Q) ⊊ μ(H)` (§`subsec:branching-theoretical-analysis`) | `Refiner.refinement_step` | `Refine.lean` |
| Theorem `thm:refine-termination` | `Refiner.refine_terminates`, `Refiner.terminal_of_mem_refine` | `Refine.lean` |
| Lemma `lem:recursive-refinement` | `Refiner.iUnion_V_round`, `Refiner.V_eq_iUnion_refine`, `Refiner.radical_eq_iInf_refine` | `Refine.lean` |
| Theorem `thm:branching-correctness` | `Refiner.branching_correct` | `Refine.lean` |
| Pruning keeps the union (§`subsec:pruning`) | `Refiner.iUnion_V_prune` | `Refine.lean` |
| Theorem `thm:prime-branching` | `Refiner.primeBranching_eq_minimalPrimes` | `PrimeBranching.lean` |
| Absorption optimization | `Refiner.primeBranching_with_absorption` | `PrimeBranching.lean` |
| Lemma `lem:sat-branches` | `Factorizer.pow_maxMult_mem_span` (geometric SAT: `pow_maxMult_mem_span'`) | `IdealReduction.lean` |
| Theorem `thm:ideal-reduce` | `Refiner.ideal_reduce_terminates`, `Refiner.ideal_reduce_span` | `IdealReduction.lean` |
| Gröbner bases decide membership (preliminaries) | `mem_iff_remainder_eq_zero`, `exists_isRemainder` | `IdealRefinement.lean` |
| Separators exist (§`subsec:primary-decomposition`) | `exists_separator` | `PrimaryBranching.lean` |
| Saturation keeps components, `J:u^∞ = ⋂_{u∉𝔭_j} 𝔮_j` | `saturation_finsetInf_primary` | `PrimaryBranching.lean` |
| Lemma `lem:isolated-separator` | `isolated_separator` | `PrimaryBranching.lean` |
| Isolated components by separators suffice when they intersect to `⟨P⟩` (§`subsec:primary-decomposition`) | `separatedComponents_unmixed` | `PrimaryBranching.lean` |
| Marking by containment and dimension (§`subsec:primary-decomposition`) | `eq_of_le_of_ringKrullDim_eq` | `PrimaryBranching.lean` |
| Theorem `thm:primary-branching` | `primaryBranching_spec` (without the optional lines), `primaryBranchingSeparated_spec` (with them), `mem_span_iff_reduces_to_zero_components` (membership) | `PrimaryBranching.lean` |
| Rabinowitsch encoding of the nonzero conditions (§`subsec:zeroing-branching`) | `separate_encoding` | `VariableZeroing.lean` |
| Product encoding `1 - s ∏_{x ∈ N} x` (no longer in the paper) | `product_encoding` | `VariableZeroing.lean` |
| Auxiliary generators survive refinement (Rabinowitsch encoding, §`subsec:zeroing-branching`; no longer used by Algorithm `alg:ZeroingBranching`) | `projection_auxGen`, `degree_auxGen_apply`, `auxGen_reduces_none`, `auxGen_not_reduced_by_auxGen`, `reduceTerm_auxGen`, `irreducible_auxGen`, `mem_of_isHitting_auxGen` | `AuxGenerators.lean` |
| Exhaustive split (§`subsec:zeroing-theoretical-analysis`) | `exhaustive_split` | `VariableZeroing.lean` |
| `V(Saturate(H, N)) ∖ V(u_N) = V(H) ∖ V(u_N)` (§`subsec:zeroing-branching`) | `mem_V_iff_of_isSaturation`, `VY_saturate` | `VariableZeroing.lean` |
| `Refine(Q, N)` keeps `V(Q) ∖ V(u_N)` (§`subsec:zeroing-branching`) | `Refiner.toCondRefiner`, `CondRefiner.dropVanishing` | `VariableZeroing.lean` |
| Lemma `lem:zero-nonzero-split` | `zero_nonzero_split` | `VariableZeroing.lean` |
| Equation `eq:zeroing-refinement` | `VY_eq_iUnion_refine` | `VariableZeroing.lean` |
| Theorem `thm:zeroing-correctness` | `zeroing_correct` | `VariableZeroing.lean` |
| Corollary `cor:zeroing-completeness` | `zeroing_complete` | `VariableZeroing.lean` |
| Theorem `thm:zeroing-termination` | `zeroing_terminates` (loop iterations `ZStepFresh`, whose split variables lie outside `Z ∪ N`) | `VariableZeroing.lean` |
| Branch contraction (§`subsec:zeroing-branch-contraction`) | `contraction_zero_unsat`, `contraction_nonzero_unsat` | `VariableZeroing.lean` |

## How the paper's objects are modelled

The algorithms are formalized at the level of their specifications. Every oracle is a structure
that carries exactly what the paper assumes, and each theorem holds for every admissible oracle.

- **Ring and points.** `R = MvPolynomial σ K` with `σ` finite. Solutions lie in any
  algebraically closed `K`-algebra `L` (the paper's `K̄`). Statements that only need a domain
  are stated for one.
- **SAT.** `Sat Q` means `⟨Q⟩ ≠ ⟨1⟩`. By the Nullstellensatz this is `V(Q) ≠ ∅`.
- **Factorization.** `Factorizer`: for `q ≠ 0`, `factors q` is the set of monic irreducible
  divisors of `q`, and `q = c_q ∏ f^{e_{q,f}}`. "Monic" refers to the monomial order.
- **Interreduction.** `InterStep` is one loop iteration of Algorithm `alg:interreduction`.
  `Interreducer` is any procedure whose output comes from such a run. The loop terminates by
  the paper's multiset argument (`wellFounded_interStep`, using the Dershowitz–Manna order), so
  an interreducer exists (`Interreducer.ofTermination`).
- **Refinement.** One round replaces each branch `Q` by `𝓗(Q̃)`, and `refine P` is the work set
  at the first round that changes nothing. Termination is proved by well-founded induction on
  `μ(Q)`, using that `R` is Noetherian, instead of Dickson's and König's lemmas. The radical
  identities come from a prime-ideal form of the branch decomposition (a prime contains `Q` iff
  it contains a minimal hitting set), so they don't need the Nullstellensatz.
- **Solvers.** `BranchOutcome` either gives an exact representation of `V(H)` or certifies it
  empty. `PrimeSolver` returns the minimal primes of `⟨H⟩`. `SatSolver` returns `J : u^∞` with
  an exponent `s` such that `u^s S ⊆ J`. `PrimarySolver` returns a minimal primary
  decomposition. Ideal reduction models its solver as any family `𝓗_sat` that contains every
  SAT minimal hitting set.
- **Primary decompositions** use Mathlib's `Submodule.IsMinimalPrimaryDecomposition`. The
  dimension of a component `q` is `ringKrullDim (R ⧸ √q)`. `TopPartSolver` returns the
  intersection of the components of maximal dimension. That this doesn't depend on which
  decomposition is chosen (uniqueness of isolated components) is part of its specification,
  not proved here. Algorithm `alg:PrimaryBranching` takes any system `Q` that Algorithm
  `alg:IdealReduce` reaches from `P`. With its optional lines (`primaryBranchingSeparated`) it
  returns the components from the separators, all marked isolated, if they intersect to `⟨Q⟩`.
  Otherwise, and always without them (`primaryBranching`), it returns each component of
  `𝓑_prim(Q)` paired with its mark (`true` for isolated).
- **Ideal refinement.** `IdealRefinement.lean` also formalizes an ideal refinement algorithm
  (`IdealRefine`, `saturation_split`, `power_branches`) that is no longer in the paper. The paper
  uses only its saturation and Gröbner-membership lemmas.
- **Variable zeroing.** Every branch lives in the original ring `K[Y]`, without auxiliary
  variables. A point is a full `a : Y → L` with `a_z = 0` for `z ∈ Z`, which is equivalent to
  the paper's restriction to `Y ∖ Z`. `H|_{x=0}` drops equations that become `0`.
  `Saturate(H, N)` is any `G` with `⟨H⟩ ⊆ ⟨G⟩` and `u_N^m g ∈ ⟨H⟩` for each `g ∈ G`
  (`IsSaturation`): a basis of `⟨H⟩ : u_N^∞`, or `H` itself after a timeout
  (`isSaturation_self`). `Refine(Q, N)` is a `CondRefiner`, specified by
  `V(Q) ∖ V(u_N) = ⋃_{H} V(H) ∖ V(u_N)`; Algorithm `alg:Refine` satisfies it, and dropping
  branches whose ideal contains a variable of `N` keeps it. Removing the variables of `N` from
  the factor sets is not modelled. Each terminal branch's fate (solved, certified empty under
  `N`, unresolved, or split on some `x`) is an arbitrary outcome, so the correctness theorem
  covers every possible solver result, saturation, timeout, budget and variable choice.
- **Auxiliary generators** (separate encoding) are proved through the three proof steps in the
  module doc of `AuxGenerators.lean`, plus the projection identity. The reduction step assumes
  that the reducing generator is nonconstant. With a constant generator, `⟨H'⟩ = ⟨1⟩` and the
  branch is UNSAT, a case the paper leaves implicit.

Not formalized: that a `Factorizer` exists (unique factorization in `K[x]`).
