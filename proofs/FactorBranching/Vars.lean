import FactorBranching.Refine

/-!
# Refinement does not introduce variables

Factors of `q` and interreductions of `Q` only involve variables of `q` and `Q`. The correctness
proof of variable zeroing (`VariableZeroing.lean`) uses this to keep each pending branch inside
`K[Y ∖ Z][s_y : y ∈ N]`.
-/

open MvPolynomial

noncomputable section

open scoped Classical

namespace FactorBranching

variable {K : Type*} [Field K] {σ : Type*}

/-- All variables of the elements of `Q` lie in `S`. -/
def VarsIn (Q : Finset (MvPolynomial σ K)) (S : Set σ) : Prop :=
  ∀ q ∈ Q, ∀ v ∈ q.vars, v ∈ S

/-- A divisor of a nonzero polynomial only involves its variables. -/
theorem vars_subset_of_dvd {f q : MvPolynomial σ K} (hfq : f ∣ q) (hq : q ≠ 0) :
    f.vars ⊆ q.vars := by
  obtain ⟨g, rfl⟩ := hfq
  intro v hv
  rw [mem_vars_iff_degreeOf_ne_zero] at hv ⊢
  rw [degreeOf_mul_eq (left_ne_zero_of_mul hq) (right_ne_zero_of_mul hq)]
  omega

variable {m : MonomialOrder σ}

theorem vars_cancelFactor_subset {p q : MvPolynomial σ K} {u : σ →₀ ℕ} (hu : u ∈ p.support) :
    (cancelFactor m p q u).vars ⊆ p.vars := by
  intro v hv
  by_cases hc : p.coeff u / m.leadingCoeff q = 0
  · simp [cancelFactor, hc] at hv
  rw [cancelFactor, vars_monomial hc] at hv
  rw [mem_vars_iff_mem_support]
  refine ⟨u, hu, ?_⟩
  rw [Finsupp.mem_support_iff] at hv ⊢
  have := (tsub_le_self : u - m.degree q ≤ u) v
  omega

theorem VarsIn.stepResult {Q : Finset (MvPolynomial σ K)} {S : Set σ} (hQ : VarsIn Q S)
    {p q : MvPolynomial σ K} {u : σ →₀ ℕ} (hp : p ∈ Q) (hq : q ∈ Q) (hu : u ∈ p.support) :
    VarsIn (stepResult m Q p q u) S := by
  have herase : ∀ x ∈ Q.erase p, ∀ v ∈ x.vars, v ∈ S :=
    fun x hx => hQ x (Finset.mem_of_mem_erase hx)
  have hr : ∀ v ∈ (reduceTerm m p q u).vars, v ∈ S := by
    intro v hv
    rcases Finset.mem_union.mp (vars_sub_subset (p := p) (q := cancelFactor m p q u * q) hv) with hv | hv
    · exact hQ p hp v hv
    · rcases Finset.mem_union.mp (vars_mul _ _ hv) with hv | hv
      · exact hQ p hp v (vars_cancelFactor_subset hu hv)
      · exact hQ q hq v hv
  rcases stepResult_eq_or (m := m) Q p q u with ⟨-, h⟩ | ⟨-, h⟩ <;> rw [h]
  · exact herase
  · intro x hx
    rcases Finset.mem_insert.mp hx with rfl | hx
    · exact hr
    · exact herase x hx

theorem VarsIn.normalize {Q : Finset (MvPolynomial σ K)} {S : Set σ} (hQ : VarsIn Q S) :
    VarsIn (FactorBranching.normalize m Q) S := by
  intro x hx v hv
  obtain ⟨p, hp, rfl⟩ := Finset.mem_image.mp hx
  by_cases hc : (m.leadingCoeff p)⁻¹ = 0
  · simp [hc] at hv
  · rw [vars_C_mul _ hc] at hv
    exact hQ p hp v hv

theorem VarsIn.of_isInterRun {Q Q' : Finset (MvPolynomial σ K)} {S : Set σ} (hQ : VarsIn Q S)
    (hr : IsInterRun m Q Q') : VarsIn Q' S := by
  obtain ⟨Q₁, hs, -, rfl⟩ := hr
  refine VarsIn.normalize ?_
  induction hs with
  | refl => exact hQ
  | tail _ hstep ih =>
    obtain ⟨p, hp, q, hq, -, u, hu, -, rfl⟩ := hstep
    exact ih.stepResult hp hq hu

namespace Refiner

variable (ρ : Refiner K m)

theorem VarsIn.tilde {Q : Finset (MvPolynomial σ K)} {S : Set σ} (hQ : VarsIn Q S) :
    VarsIn (ρ.tilde Q) S :=
  VarsIn.of_isInterRun (fun q hq => hQ q (Finset.mem_of_mem_erase hq))
    (ρ.I.isRun _ (zero_notMem_erase_zero Q))

/-- The minimal hitting sets of `Q` only involve variables of `Q`. -/
theorem varsIn_of_mem_minHitting {Q H : Finset (MvPolynomial σ K)} {S : Set σ}
    (hQ : VarsIn Q S) (h0 : (0 : MvPolynomial σ K) ∉ Q) (hH : H ∈ ρ.F.minHitting Q) :
    VarsIn H S := by
  intro h hh v hv
  obtain ⟨q, hq, hhq⟩ :=
    Factorizer.mem_allFactors.mp ((Factorizer.isHitting_of_mem_minHitting hH).1 hh)
  have hq0 : q ≠ 0 := fun h' => h0 (h' ▸ hq)
  exact hQ q hq v (vars_subset_of_dvd (Factorizer.dvd_of_mem hq0 hhq) hq0 hv)

theorem zero_notMem_of_mem_minHitting {Q H : Finset (MvPolynomial σ K)}
    (h0 : (0 : MvPolynomial σ K) ∉ Q) (hH : H ∈ ρ.F.minHitting Q) : (0 : MvPolynomial σ K) ∉ H :=
  fun hH0 => by
    obtain ⟨q, hq, hq0⟩ :=
      Factorizer.mem_allFactors.mp ((Factorizer.isHitting_of_mem_minHitting hH).1 hH0)
    exact (Factorizer.irreducible_of_mem (fun h' => h0 (h' ▸ hq)) hq0).ne_zero rfl

/-- Every work set of Algorithm `alg:Refine` consists of branches without `0` whose variables lie
in those of `P`. -/
theorem varsIn_iterate_round {P : Finset (MvPolynomial σ K)} {S : Set σ} (hP : VarsIn P S)
    (h0 : (0 : MvPolynomial σ K) ∉ P) (k : ℕ) :
    ∀ H ∈ ρ.round^[k] (ρ.F.minHitting P), VarsIn H S ∧ (0 : MvPolynomial σ K) ∉ H := by
  induction k with
  | zero =>
    intro H hH
    exact ⟨ρ.varsIn_of_mem_minHitting hP h0 hH, ρ.zero_notMem_of_mem_minHitting h0 hH⟩
  | succ k ih =>
    intro H hH
    rw [Function.iterate_succ_apply'] at hH
    obtain ⟨Q, hQ, hHQ⟩ := Finset.mem_biUnion.mp hH
    exact ⟨ρ.varsIn_of_mem_minHitting (VarsIn.tilde ρ (ih Q hQ).1) (zero_notMem_tilde Q) hHQ,
      zero_notMem_of_mem_children hHQ⟩

theorem varsIn_of_mem_refine [Finite σ] {P H : Finset (MvPolynomial σ K)} {S : Set σ}
    (hP : VarsIn P S) (h0 : (0 : MvPolynomial σ K) ∉ P) (hH : H ∈ ρ.refine P) :
    VarsIn H S ∧ (0 : MvPolynomial σ K) ∉ H :=
  ρ.varsIn_iterate_round hP h0 _ H hH

end Refiner

end FactorBranching
