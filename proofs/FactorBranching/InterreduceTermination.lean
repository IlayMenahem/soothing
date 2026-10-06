import FactorBranching.Interreduce

/-!
# Termination of interreduction

The remark after Algorithm `alg:interreduction` (Preliminaries): "Each loop iteration cancels one
monomial `u` and introduces only monomials strictly smaller than `u`. The multiset extension of
the monomial order is well-founded, so the loop terminates."

The measure of a set `Q` is the multiset of all monomials of all its elements, ordered by the
Dershowitz–Manna extension of the monomial order. One step replaces the monomials of `p` by
those of `r`, which lack `u` and add only monomials below `u`. As a consequence every finite set
of nonzero polynomials has a terminating run, so an `Interreducer` exists
(`Interreducer.ofTermination`).
-/

open MvPolynomial

noncomputable section

open scoped Classical MonomialOrder

namespace FactorBranching

variable {K : Type*} [Field K] {σ : Type*} {m : MonomialOrder σ}

/-! ### One step cancels `u` and adds only smaller monomials -/

section Step

variable {p q : MvPolynomial σ K} {u : σ →₀ ℕ}

/-- The subtracted multiple has coefficient `c` at `u`, so it cancels the term `c u`. -/
theorem coeff_cancelFactor_mul (hq : q ≠ 0) (hdiv : m.degree q ≤ u) :
    (cancelFactor m p q u * q).coeff u = p.coeff u := by
  rw [cancelFactor, coeff_monomial_mul']
  split_ifs with h
  swap
  · exact absurd tsub_le_self h
  rw [tsub_tsub_cancel_of_le hdiv]
  exact div_mul_cancel₀ _ (m.leadingCoeff_ne_zero_iff.mpr hq)

theorem notMem_support_reduceTerm (hq : q ≠ 0) (hdiv : m.degree q ≤ u) :
    u ∉ (reduceTerm m p q u).support := by
  rw [mem_support_iff, not_not, reduceTerm, coeff_sub, coeff_cancelFactor_mul hq hdiv, sub_self]

/-- Every monomial of `r` is a monomial of `p` or strictly smaller than `u`. -/
theorem mem_support_reduceTerm (hq : q ≠ 0) (hdiv : m.degree q ≤ u) {v : σ →₀ ℕ}
    (hv : v ∈ (reduceTerm m p q u).support) : v ∈ p.support ∨ v ≺[m] u := by
  by_cases hvp : v ∈ p.support
  · exact Or.inl hvp
  right
  have hcoeff : (cancelFactor m p q u * q).coeff v ≠ 0 := by
    rw [mem_support_iff, reduceTerm, coeff_sub] at hv
    rw [notMem_support_iff.mp hvp, zero_sub, neg_ne_zero] at hv
    exact hv
  have hle : v ≼[m] u :=
    (m.le_degree (mem_support_iff.mpr hcoeff)).trans (degree_cancel_le hdiv)
  refine lt_of_le_of_ne hle fun h => ?_
  rw [m.toSyn.injective h] at hv
  exact notMem_support_reduceTerm hq hdiv hv

end Step

/-! ### The multiset measure -/

/-- The monomials of `p`, as elements of the well-ordered type `m.syn`. -/
def synSupport (m : MonomialOrder σ) (p : MvPolynomial σ K) : Finset m.syn :=
  p.support.map m.toSyn.toEquiv.toEmbedding

theorem mem_synSupport {p : MvPolynomial σ K} {v : σ →₀ ℕ} :
    m.toSyn v ∈ synSupport m p ↔ v ∈ p.support := by
  simp [synSupport]

/-- The multiset of all monomials of all elements of `Q`. -/
def termMeasure (m : MonomialOrder σ) (Q : Finset (MvPolynomial σ K)) : Multiset m.syn :=
  ∑ p ∈ Q, (synSupport m p).val

theorem isDershowitzMannaLT_add_left {α : Type*} [Preorder α] {M N : Multiset α}
    (h : Multiset.IsDershowitzMannaLT M N) (A : Multiset α) :
    Multiset.IsDershowitzMannaLT (A + M) (A + N) := by
  obtain ⟨X, Y, Z, hZ, rfl, rfl, hY⟩ := h
  exact ⟨A + X, Y, Z, hZ, (add_assoc _ _ _).symm, (add_assoc _ _ _).symm, hY⟩

/-- Replacing a finite set `P` by `R`, where `R` lacks some `x ∈ P` and every new element is
below `x`, decreases the multiset order. -/
theorem isDershowitzMannaLT_of_replace {α : Type*} [Preorder α] {P R : Finset α} {x : α}
    (hxP : x ∈ P) (hxR : x ∉ R) (hR : ∀ y ∈ R, y ∈ P ∨ y < x) :
    Multiset.IsDershowitzMannaLT R.val P.val := by
  have hcommon : (R.filter (· ∈ P)).val = (P.filter (· ∈ R)).val := by
    congr 1
    ext y
    simp [and_comm]
  refine ⟨(R.filter (· ∈ P)).val, (R.filter (· ∉ P)).val, (P.filter (· ∉ R)).val, ?_, ?_, ?_, ?_⟩
  · intro h
    have : x ∈ (P.filter (· ∉ R)).val := Finset.mem_filter.mpr ⟨hxP, hxR⟩
    rw [h] at this
    exact Multiset.notMem_zero x this
  · rw [Finset.filter_val, Finset.filter_val, Multiset.filter_add_not]
  · rw [hcommon, Finset.filter_val, Finset.filter_val, Multiset.filter_add_not]
  · intro y hy
    obtain ⟨hyR, hyP⟩ := Finset.mem_filter.mp hy
    exact ⟨x, Finset.mem_filter.mpr ⟨hxP, hxR⟩, (hR y hyR).resolve_left hyP⟩

theorem synSupport_val_ne_zero {p : MvPolynomial σ K} (hp : p ≠ 0) :
    (synSupport m p).val ≠ 0 := by
  intro h
  have : m.toSyn (m.degree p) ∈ synSupport m p := mem_synSupport.mpr (m.degree_mem_support hp)
  rw [← Finset.mem_val, h] at this
  exact Multiset.notMem_zero _ this

/-- One loop iteration decreases the measure. -/
theorem InterStep.termMeasure_lt {Q Q' : Finset (MvPolynomial σ K)} (hs : InterStep m Q Q')
    (h0 : (0 : MvPolynomial σ K) ∉ Q) :
    Multiset.IsDershowitzMannaLT (termMeasure m Q') (termMeasure m Q) := by
  obtain ⟨p, hp, q, hq, -, u, hu, hdiv, rfl⟩ := hs
  have hp0 : p ≠ 0 := fun h => h0 (h ▸ hp)
  have hq0 : q ≠ 0 := fun h => h0 (h ▸ hq)
  set E := Q.erase p
  have hQ : termMeasure m Q = termMeasure m E + (synSupport m p).val :=
    (Finset.sum_erase_add Q _ hp).symm
  -- dropping `p` alone decreases the measure
  have hdrop : Multiset.IsDershowitzMannaLT (termMeasure m E) (termMeasure m Q) := by
    rw [hQ]
    exact ⟨termMeasure m E, 0, (synSupport m p).val, synSupport_val_ne_zero hp0, (add_zero _).symm,
      rfl, by simp⟩
  rcases stepResult_eq_or (m := m) Q p q u with ⟨-, h⟩ | ⟨-, h⟩ <;> rw [h]
  · exact hdrop
  · by_cases hrE : reduceTerm m p q u ∈ E
    · rw [Finset.insert_eq_of_mem hrE]
      exact hdrop
    rw [termMeasure, Finset.sum_insert hrE, add_comm, hQ]
    refine isDershowitzMannaLT_add_left (isDershowitzMannaLT_of_replace (x := m.toSyn u)
      (mem_synSupport.mpr hu) (fun h' => notMem_support_reduceTerm hq0 hdiv
        (mem_synSupport.mp h')) fun y hy => ?_) _
    obtain ⟨v, hv, rfl⟩ : ∃ v ∈ (reduceTerm m p q u).support, m.toSyn v = y :=
      ⟨m.toSyn.symm y, by simpa [synSupport] using hy, m.toSyn.apply_symm_apply y⟩
    rcases mem_support_reduceTerm hq0 hdiv hv with h' | h'
    · exact Or.inl (mem_synSupport.mpr h')
    · exact Or.inr h'

/-! ### Termination and existence of an interreducer -/

/-- **Termination of Algorithm `alg:interreduction`.** On finite sets of nonzero polynomials
there is no infinite sequence of loop iterations. -/
theorem wellFounded_interStep :
    WellFounded fun Q' Q : Finset (MvPolynomial σ K) =>
      (0 : MvPolynomial σ K) ∉ Q ∧ InterStep m Q Q' :=
  Subrelation.wf (fun h => h.2.termMeasure_lt h.1)
    (InvImage.wf (termMeasure m) Multiset.wellFounded_isDershowitzMannaLT)

/-- From every finite set of nonzero polynomials the loop reaches a set where no step applies. -/
theorem exists_terminal_run (Q : Finset (MvPolynomial σ K)) (h0 : (0 : MvPolynomial σ K) ∉ Q) :
    ∃ Q₁, Relation.ReflTransGen (InterStep m) Q Q₁ ∧ ∀ Q₂, ¬ InterStep m Q₁ Q₂ := by
  induction Q using (wellFounded_interStep (m := m)).induction with
  | _ Q ih =>
    by_cases hstep : ∃ Q₂, InterStep m Q Q₂
    · obtain ⟨Q₂, hs⟩ := hstep
      obtain ⟨Q₁, hrun, hterm⟩ := ih Q₂ ⟨h0, hs⟩ (hs.zero_notMem h0)
      exact ⟨Q₁, Relation.ReflTransGen.head hs hrun, hterm⟩
    · push Not at hstep
      exact ⟨Q, Relation.ReflTransGen.refl, hstep⟩

/-- Algorithm `alg:interreduction` has an output on every finite set of nonzero polynomials. -/
theorem exists_isInterRun (Q : Finset (MvPolynomial σ K)) (h0 : (0 : MvPolynomial σ K) ∉ Q) :
    ∃ Q', IsInterRun m Q Q' := by
  obtain ⟨Q₁, hrun, hterm⟩ := exists_terminal_run (m := m) Q h0
  exact ⟨normalize m Q₁, Q₁, hrun, hterm, rfl⟩

variable (K m) in
/-- An interreducer built from the termination proof: run the loop until no step applies, then
normalize. -/
def Interreducer.ofTermination : Interreducer K m where
  ired Q := if h : (0 : MvPolynomial σ K) ∉ Q then (exists_isInterRun (m := m) Q h).choose else Q
  isRun Q h0 := by
    split_ifs <;> exact (exists_isInterRun Q h0).choose_spec

instance : Nonempty (Interreducer K m) :=
  ⟨Interreducer.ofTermination K m⟩

end FactorBranching
