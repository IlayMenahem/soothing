import FactorBranching.Monomial

/-!
# Interreduction

Algorithm `alg:interreduction` and Lemma `lem:lm-monotone` (Preliminaries).

One loop iteration (`InterStep`) cancels a term `c u` of `p` using `q` with `LM(q) ∣ u`, replacing
`p` by `r = p - (c u / LT(q)) q` (dropped if zero). A run (`IsInterRun`) iterates steps until none
applies and then makes every element monic. An `Interreducer` is any procedure whose output is such
a run; the paper's termination remark (multiset order) is what makes one exist and is not
formalized here.
-/

open MvPolynomial

noncomputable section

open scoped Classical MonomialOrder

namespace FactorBranching

variable {K : Type*} [Field K] {σ : Type*} (m : MonomialOrder σ)

/-- `Q` is interreduced: its elements are nonzero and monic, and no term of one element is
divisible by the leading monomial of another. -/
def IsInterreduced (Q : Finset (MvPolynomial σ K)) : Prop :=
  (0 : MvPolynomial σ K) ∉ Q ∧ (∀ q ∈ Q, m.Monic q) ∧
    ∀ p ∈ Q, ∀ q ∈ Q, p ≠ q → ∀ u ∈ p.support, ¬ m.degree q ≤ u

/-- The multiple `(c u / LT(q))` of `q` subtracted when cancelling the term at `u` of `p`. -/
def cancelFactor (p q : MvPolynomial σ K) (u : σ →₀ ℕ) : MvPolynomial σ K :=
  monomial (u - m.degree q) (p.coeff u / m.leadingCoeff q)

/-- `r = p - (c u / LT(q)) q`. -/
def reduceTerm (p q : MvPolynomial σ K) (u : σ →₀ ℕ) : MvPolynomial σ K :=
  p - cancelFactor m p q u * q

/-- The set after one loop iteration: `p` is replaced by `r`, or removed if `r = 0`. -/
def stepResult (Q : Finset (MvPolynomial σ K)) (p q : MvPolynomial σ K) (u : σ →₀ ℕ) :
    Finset (MvPolynomial σ K) :=
  if reduceTerm m p q u = 0 then Q.erase p else insert (reduceTerm m p q u) (Q.erase p)

/-- One iteration of the loop of Algorithm `alg:interreduction`. -/
def InterStep (Q Q' : Finset (MvPolynomial σ K)) : Prop :=
  ∃ p ∈ Q, ∃ q ∈ Q, p ≠ q ∧ ∃ u ∈ p.support, m.degree q ≤ u ∧ Q' = stepResult m Q p q u

/-- The final normalization `p ↦ p / LC(p)`. -/
def normalize (Q : Finset (MvPolynomial σ K)) : Finset (MvPolynomial σ K) :=
  Q.image fun p => C (m.leadingCoeff p)⁻¹ * p

/-- `Q'` is a possible output of Algorithm `alg:interreduction` on `Q`. -/
def IsInterRun (Q Q' : Finset (MvPolynomial σ K)) : Prop :=
  ∃ Q₁, Relation.ReflTransGen (InterStep m) Q Q₁ ∧ (∀ Q₂, ¬ InterStep m Q₁ Q₂) ∧
    Q' = normalize m Q₁

/-- An interreduction procedure: on every finite set of nonzero polynomials it returns a possible
output of Algorithm `alg:interreduction`. -/
structure Interreducer (K : Type*) [Field K] {σ : Type*} (m : MonomialOrder σ) where
  ired : Finset (MvPolynomial σ K) → Finset (MvPolynomial σ K)
  isRun : ∀ Q, (0 : MvPolynomial σ K) ∉ Q → IsInterRun m Q (ired Q)

variable {m}

/-! ### One step -/

section Step

variable {Q : Finset (MvPolynomial σ K)} {p q : MvPolynomial σ K} {u : σ →₀ ℕ}

theorem stepResult_eq_or (Q : Finset (MvPolynomial σ K)) (p q : MvPolynomial σ K) (u) :
    (reduceTerm m p q u = 0 ∧ stepResult m Q p q u = Q.erase p) ∨
      (reduceTerm m p q u ≠ 0 ∧ stepResult m Q p q u = insert (reduceTerm m p q u) (Q.erase p)) := by
  unfold stepResult
  split_ifs with h
  · exact Or.inl ⟨h, rfl⟩
  · exact Or.inr ⟨h, rfl⟩

theorem zero_notMem_stepResult (h0 : (0 : MvPolynomial σ K) ∉ Q) :
    (0 : MvPolynomial σ K) ∉ stepResult m Q p q u := by
  rcases stepResult_eq_or (m := m) Q p q u with ⟨-, h⟩ | ⟨hr, h⟩ <;> rw [h]
  · exact fun h' => h0 (Finset.mem_of_mem_erase h')
  · rw [Finset.mem_insert, not_or]
    exact ⟨fun h' => hr h'.symm, fun h' => h0 (Finset.mem_of_mem_erase h')⟩

theorem span_stepResult (hp : p ∈ Q) (hq : q ∈ Q) (hpq : p ≠ q) :
    Ideal.span (↑(stepResult m Q p q u) : Set (MvPolynomial σ K)) = Ideal.span ↑Q := by
  set r := reduceTerm m p q u
  set E : Set (MvPolynomial σ K) := ↑(Q.erase p)
  have hqE : q ∈ E := by simp [E, hq, hpq.symm]
  have hQ : (↑Q : Set (MvPolynomial σ K)) = insert p E := by
    simp only [E, Finset.coe_erase, Set.insert_sdiff_singleton]
    exact (Set.insert_eq_of_mem (Finset.mem_coe.mpr hp)).symm
  have key : Ideal.span (insert r E) = Ideal.span (insert p E) := by
    apply le_antisymm <;> rw [Ideal.span_le, Set.insert_subset_iff] <;>
      refine ⟨?_, fun x hx => Ideal.subset_span (Set.mem_insert_of_mem _ hx)⟩
    · exact Ideal.sub_mem _ (Ideal.subset_span (Set.mem_insert _ _))
        (Ideal.mul_mem_left _ _ (Ideal.subset_span (Set.mem_insert_of_mem _ hqE)))
    · have : p = r + cancelFactor m p q u * q := by simp [r, reduceTerm]
      rw [this]
      exact Ideal.add_mem _ (Ideal.subset_span (Set.mem_insert _ _))
        (Ideal.mul_mem_left _ _ (Ideal.subset_span (Set.mem_insert_of_mem _ hqE)))
  rw [hQ, ← key]
  rcases stepResult_eq_or (m := m) Q p q u with ⟨hr, h⟩ | ⟨-, h⟩ <;> rw [h]
  · rw [show r = 0 from hr, Ideal.span_insert_zero]
  · rw [Finset.coe_insert]

/-- The degree of the subtracted multiple is at most `u`. -/
theorem degree_cancel_le (hdiv : m.degree q ≤ u) :
    m.degree (cancelFactor m p q u * q) ≼[m] u := by
  calc m.toSyn (m.degree (cancelFactor m p q u * q))
      ≤ m.toSyn (m.degree (cancelFactor m p q u) + m.degree q) := m.degree_mul_le
    _ ≤ m.toSyn ((u - m.degree q) + m.degree q) := by
        rw [map_add, map_add]
        exact add_le_add_left (m.degree_monomial_le _) _
    _ = m.toSyn u := by rw [tsub_add_cancel_of_le hdiv]

/-- If the cancelled monomial is not the leading one, `r` keeps the leading monomial of `p`. -/
theorem degree_reduceTerm (hu : u ∈ p.support) (hdiv : m.degree q ≤ u) (hne : u ≠ m.degree p) :
    m.degree (reduceTerm m p q u) = m.degree p ∧ reduceTerm m p q u ≠ 0 := by
  have hlt : u ≺[m] m.degree p :=
    lt_of_le_of_ne (m.le_degree hu) fun h => hne (m.toSyn.injective h)
  have hdeg : m.degree (reduceTerm m p q u) = m.degree p :=
    m.degree_sub_of_lt (lt_of_le_of_lt (degree_cancel_le hdiv) hlt)
  refine ⟨hdeg, fun h0 => ?_⟩
  rw [h0, m.degree_zero] at hdeg
  rw [← hdeg] at hlt
  exact absurd hlt (not_lt.mpr (by simp))

theorem lmIdeal_le_stepResult (hp : p ∈ Q) (hq : q ∈ Q) (hpq : p ≠ q) (hu : u ∈ p.support)
    (hdiv : m.degree q ≤ u) :
    lmIdeal m Q ≤ lmIdeal m (stepResult m Q p q u) := by
  have hsub : ∀ x ∈ Q, x ≠ p → x ∈ stepResult m Q p q u := fun x hx hxp => by
    rcases stepResult_eq_or (m := m) Q p q u with ⟨-, h⟩ | ⟨-, h⟩ <;> rw [h]
    · exact Finset.mem_erase.mpr ⟨hxp, hx⟩
    · exact Finset.mem_insert_of_mem (Finset.mem_erase.mpr ⟨hxp, hx⟩)
  rw [lmIdeal_le_iff]
  intro x hx
  by_cases hxp : x = p
  · subst hxp
    by_cases hlead : u = m.degree x
    · exact ⟨q, hsub q hq (Ne.symm hpq), hlead ▸ hdiv⟩
    · obtain ⟨hdeg, hr⟩ := degree_reduceTerm hu hdiv hlead
      refine ⟨reduceTerm m x q u, ?_, hdeg.le⟩
      rcases stepResult_eq_or (m := m) Q x q u with ⟨h0, -⟩ | ⟨-, h⟩
      · exact absurd h0 hr
      · rw [h]; exact Finset.mem_insert_self _ _
  · exact ⟨x, hsub x hx hxp, le_rfl⟩

end Step

/-! ### Normalization -/

section Normalize

theorem degree_normalize {p : MvPolynomial σ K} (hp : p ≠ 0) :
    m.degree (C (m.leadingCoeff p)⁻¹ * p) = m.degree p := by
  have hc : (m.leadingCoeff p)⁻¹ ≠ 0 := inv_ne_zero (m.leadingCoeff_ne_zero_iff.mpr hp)
  rw [m.degree_mul (C_ne_zero.mpr hc) hp, m.degree_C, zero_add]

theorem monic_normalize {p : MvPolynomial σ K} (hp : p ≠ 0) :
    m.Monic (C (m.leadingCoeff p)⁻¹ * p) := by
  have hc : m.leadingCoeff p ≠ 0 := m.leadingCoeff_ne_zero_iff.mpr hp
  have hne : m.leadingCoeff (C (m.leadingCoeff p)⁻¹) * m.leadingCoeff p ≠ 0 := by
    rw [m.leadingCoeff_C, inv_mul_cancel₀ hc]; exact one_ne_zero
  rw [MonomialOrder.Monic, m.leadingCoeff_mul_of_mul_leadingCoeff_ne_zero hne, m.leadingCoeff_C,
    inv_mul_cancel₀ hc]

theorem normalize_ne_zero {p : MvPolynomial σ K} (hp : p ≠ 0) :
    C (m.leadingCoeff p)⁻¹ * p ≠ 0 :=
  (monic_normalize hp).ne_zero

theorem support_normalize {p : MvPolynomial σ K} (hp : p ≠ 0) :
    (C (m.leadingCoeff p)⁻¹ * p).support = p.support := by
  rw [C_mul', support_smul_eq (inv_ne_zero (m.leadingCoeff_ne_zero_iff.mpr hp))]

theorem span_normalize (Q : Finset (MvPolynomial σ K)) (h0 : (0 : MvPolynomial σ K) ∉ Q) :
    Ideal.span (↑(normalize m Q) : Set (MvPolynomial σ K)) = Ideal.span ↑Q := by
  apply le_antisymm <;> rw [Ideal.span_le]
  · intro x hx
    obtain ⟨p, hp, rfl⟩ := Finset.mem_image.mp hx
    exact Ideal.mul_mem_left _ _ (Ideal.subset_span hp)
  · intro p hp
    have hp0 : p ≠ 0 := fun h => h0 (h ▸ hp)
    have hc : m.leadingCoeff p ≠ 0 := m.leadingCoeff_ne_zero_iff.mpr hp0
    have : p = C (m.leadingCoeff p) * (C (m.leadingCoeff p)⁻¹ * p) := by
      rw [← mul_assoc, ← C_mul, mul_inv_cancel₀ hc, C_1, one_mul]
    rw [this]
    exact Ideal.mul_mem_left _ _ (Ideal.subset_span (Finset.mem_image_of_mem _ hp))

theorem lmIdeal_normalize (Q : Finset (MvPolynomial σ K)) (h0 : (0 : MvPolynomial σ K) ∉ Q) :
    lmIdeal m (normalize m Q) = lmIdeal m Q := by
  apply le_antisymm <;> rw [lmIdeal_le_iff]
  · intro x hx
    obtain ⟨p, hp, rfl⟩ := Finset.mem_image.mp hx
    exact ⟨p, hp, (degree_normalize fun h => h0 (h ▸ hp)).ge⟩
  · intro p hp
    exact ⟨_, Finset.mem_image_of_mem _ hp, (degree_normalize fun h => h0 (h ▸ hp)).le⟩

theorem normalize_of_monic {Q : Finset (MvPolynomial σ K)} (h : ∀ q ∈ Q, m.Monic q) :
    normalize m Q = Q := by
  conv_rhs => rw [← Finset.image_id (s := Q)]
  refine Finset.image_congr fun q hq => ?_
  simp [(h q hq).leadingCoeff_eq_one]

end Normalize

/-! ### Runs -/

section Run

variable {Q Q' : Finset (MvPolynomial σ K)}

theorem InterStep.zero_notMem (hs : InterStep m Q Q') (h0 : (0 : MvPolynomial σ K) ∉ Q) :
    (0 : MvPolynomial σ K) ∉ Q' := by
  obtain ⟨p, -, q, -, -, u, -, -, rfl⟩ := hs
  exact zero_notMem_stepResult h0

theorem InterStep.span_eq (hs : InterStep m Q Q') :
    Ideal.span (↑Q' : Set (MvPolynomial σ K)) = Ideal.span ↑Q := by
  obtain ⟨p, hp, q, hq, hpq, u, -, -, rfl⟩ := hs
  exact span_stepResult hp hq hpq

theorem InterStep.lmIdeal_le (hs : InterStep m Q Q') : lmIdeal m Q ≤ lmIdeal m Q' := by
  obtain ⟨p, hp, q, hq, hpq, u, hu, hdiv, rfl⟩ := hs
  exact lmIdeal_le_stepResult hp hq hpq hu hdiv

theorem reflTransGen_invariants (hs : Relation.ReflTransGen (InterStep m) Q Q')
    (h0 : (0 : MvPolynomial σ K) ∉ Q) :
    (0 : MvPolynomial σ K) ∉ Q' ∧ Ideal.span (↑Q' : Set (MvPolynomial σ K)) = Ideal.span ↑Q ∧
      lmIdeal m Q ≤ lmIdeal m Q' := by
  induction hs with
  | refl => exact ⟨h0, rfl, le_rfl⟩
  | tail _ hstep ih =>
    obtain ⟨h0', hspan, hlm⟩ := ih
    exact ⟨hstep.zero_notMem h0', hstep.span_eq.trans hspan, hlm.trans hstep.lmIdeal_le⟩

/-- Interreduction preserves `⟨Q⟩` (Preliminaries). -/
theorem IsInterRun.span_eq (hr : IsInterRun m Q Q') (h0 : (0 : MvPolynomial σ K) ∉ Q) :
    Ideal.span (↑Q' : Set (MvPolynomial σ K)) = Ideal.span ↑Q := by
  obtain ⟨Q₁, hs, -, rfl⟩ := hr
  obtain ⟨h0', hspan, -⟩ := reflTransGen_invariants hs h0
  rw [span_normalize _ h0', hspan]

/-- **Lemma `lem:lm-monotone` (Leading monomials under interreduction).** If
Algorithm `alg:interreduction` returns `Q'` on input `Q`, then
`⟨LM(q) : q ∈ Q⟩ ⊆ ⟨LM(q) : q ∈ Q'⟩`. -/
theorem IsInterRun.lmIdeal_le (hr : IsInterRun m Q Q') (h0 : (0 : MvPolynomial σ K) ∉ Q) :
    lmIdeal m Q ≤ lmIdeal m Q' := by
  obtain ⟨Q₁, hs, -, rfl⟩ := hr
  obtain ⟨h0', -, hlm⟩ := reflTransGen_invariants hs h0
  rw [lmIdeal_normalize _ h0']
  exact hlm

/-- The output of Algorithm `alg:interreduction` is interreduced. -/
theorem IsInterRun.isInterreduced (hr : IsInterRun m Q Q') (h0 : (0 : MvPolynomial σ K) ∉ Q) :
    IsInterreduced m Q' := by
  obtain ⟨Q₁, hs, hterm, rfl⟩ := hr
  obtain ⟨h0', -, -⟩ := reflTransGen_invariants hs h0
  have hne : ∀ p ∈ Q₁, p ≠ 0 := fun p hp h => h0' (h ▸ hp)
  refine ⟨?_, ?_, ?_⟩
  · intro h
    obtain ⟨p, hp, hp0⟩ := Finset.mem_image.mp h
    exact normalize_ne_zero (hne p hp) hp0
  · intro x hx
    obtain ⟨p, hp, rfl⟩ := Finset.mem_image.mp hx
    exact monic_normalize (hne p hp)
  · intro x hx y hy hxy u hu hdiv
    obtain ⟨p, hp, rfl⟩ := Finset.mem_image.mp hx
    obtain ⟨q, hq, rfl⟩ := Finset.mem_image.mp hy
    have hpq : p ≠ q := fun h => hxy (h ▸ rfl)
    rw [support_normalize (hne p hp)] at hu
    rw [degree_normalize (hne q hq)] at hdiv
    exact hterm _ ⟨p, hp, q, hq, hpq, u, hu, hdiv, rfl⟩

/-- On an interreduced input, Algorithm `alg:interreduction` returns its input. -/
theorem IsInterRun.eq_of_isInterreduced (hr : IsInterRun m Q Q') (hQ : IsInterreduced m Q) :
    Q' = Q := by
  obtain ⟨Q₁, hs, -, rfl⟩ := hr
  have hnostep : ∀ Q₂, ¬ InterStep m Q Q₂ := by
    rintro _ ⟨p, hp, q, hq, hpq, u, hu, hdiv, -⟩
    exact hQ.2.2 p hp q hq hpq u hu hdiv
  obtain rfl : Q₁ = Q := by
    rcases Relation.ReflTransGen.cases_head hs with h | ⟨Q₂, hstep, -⟩
    · exact h.symm
    · exact absurd hstep (hnostep Q₂)
  exact normalize_of_monic hQ.2.1

end Run

namespace Interreducer

variable (I : Interreducer K m) {Q : Finset (MvPolynomial σ K)}

theorem span_ired (h0 : (0 : MvPolynomial σ K) ∉ Q) :
    Ideal.span (↑(I.ired Q) : Set (MvPolynomial σ K)) = Ideal.span ↑Q :=
  (I.isRun Q h0).span_eq h0

theorem lmIdeal_le_ired (h0 : (0 : MvPolynomial σ K) ∉ Q) :
    lmIdeal m Q ≤ lmIdeal m (I.ired Q) :=
  (I.isRun Q h0).lmIdeal_le h0

theorem isInterreduced_ired (h0 : (0 : MvPolynomial σ K) ∉ Q) : IsInterreduced m (I.ired Q) :=
  (I.isRun Q h0).isInterreduced h0

theorem ired_eq_self (hQ : IsInterreduced m Q) : I.ired Q = Q :=
  (I.isRun Q hQ.1).eq_of_isInterreduced hQ

end Interreducer

end FactorBranching
