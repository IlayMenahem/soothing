import FactorBranching

/-! Axiom audit: every paper result depends only on Lean's standard axioms
(`propext`, `Classical.choice`, `Quot.sound`). Run `lake env lean Audit.lean`. -/

open FactorBranching

-- Preliminaries
#print axioms FactorBranching.vanishing_V_eq_radical
#print axioms FactorBranching.V_eq_empty_iff
#print axioms FactorBranching.IsInterRun.lmIdeal_le
-- Section "Factor branching framework"
#print axioms FactorBranching.Factorizer.V_eq_iUnion_minHitting
#print axioms FactorBranching.Factorizer.radical_eq_iInf_minHitting
#print axioms FactorBranching.Refiner.refinement_step
#print axioms FactorBranching.Refiner.refine_terminates
#print axioms FactorBranching.Refiner.terminal_of_mem_refine
#print axioms FactorBranching.Refiner.V_eq_iUnion_refine
#print axioms FactorBranching.Refiner.radical_eq_iInf_refine
#print axioms FactorBranching.Refiner.branching_correct
#print axioms FactorBranching.Refiner.iUnion_V_prune
-- Section "Prime factor branching"
#print axioms FactorBranching.Refiner.primeBranching_eq_minimalPrimes
#print axioms FactorBranching.Refiner.primeBranching_with_absorption
-- Section "Ideal-preserving factor branching"
#print axioms FactorBranching.Factorizer.pow_maxMult_mem_span
#print axioms FactorBranching.Factorizer.pow_maxMult_mem_span'
#print axioms FactorBranching.Refiner.ideal_reduce_terminates
#print axioms FactorBranching.Refiner.ideal_reduce_span
#print axioms FactorBranching.power_branches
#print axioms FactorBranching.saturation_split
#print axioms FactorBranching.IdealRefine.exists_round_no_split
#print axioms FactorBranching.IdealRefine.span_eq_inf_leaves
#print axioms FactorBranching.mem_span_iff_reduces_to_zero
-- Section "Primary factor branching"
#print axioms FactorBranching.exists_separator
#print axioms FactorBranching.isolated_separator
#print axioms FactorBranching.primaryBranching_spec
-- Section "Recursive splitting"
#print axioms FactorBranching.exhaustive_split
#print axioms FactorBranching.projection_auxGen
#print axioms FactorBranching.reduceTerm_auxGen
#print axioms FactorBranching.irreducible_auxGen
#print axioms FactorBranching.auxGen_reduces_none
#print axioms FactorBranching.zero_nonzero_split
#print axioms FactorBranching.VY_eq_iUnion_refine
#print axioms FactorBranching.zeroing_correct
#print axioms FactorBranching.zeroing_complete
#print axioms FactorBranching.contraction_zero_unsat
#print axioms FactorBranching.separate_encoding
#print axioms FactorBranching.product_encoding
-- Supporting results named in README.md
#print axioms FactorBranching.sat_iff_V_nonempty
#print axioms FactorBranching.IsInterRun.span_eq
#print axioms FactorBranching.IsInterRun.isInterreduced
#print axioms FactorBranching.Refiner.iUnion_V_round
#print axioms FactorBranching.IdealRefine.mem_span_iff_forall_leaves
#print axioms FactorBranching.mem_iff_remainder_eq_zero
#print axioms FactorBranching.exists_isRemainder
#print axioms FactorBranching.saturation_finsetInf_primary
#print axioms FactorBranching.degree_auxGen_apply
#print axioms FactorBranching.auxGen_not_reduced_by_auxGen
#print axioms FactorBranching.mem_of_isHitting_auxGen
#print axioms FactorBranching.contraction_nonzero_unsat
