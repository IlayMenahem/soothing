import FactorBranching.InterreduceTermination
import FactorBranching.Refine

/-!
# Refined branches are simpler

Theorem `thm:refine-simpler` (Section "Factor branching framework"): for a child
`H ∈ 𝓗(Q̃)` with `H ≠ Q̃`, the leading monomials of `H` are smaller than those of `Q` in the
multiset extension of `≺`, yet generate a strictly larger monomial ideal.

The multiset of leading monomials (`lmMeasure`) decreases strictly from `Q̃` to `H`, because each
element of `H` divides a different element of `Q̃`, and it does not increase under
interreduction.
-/

open MvPolynomial

noncomputable section

open scoped Classical MonomialOrder

namespace FactorBranching

variable {K : Type*} [Field K] {σ : Type*} {m : MonomialOrder σ}

/-! ### The multiset order -/

/-- The reflexive closure of the Dershowitz–Manna order. -/
def DMLE {α : Type*} [Preorder α] (M N : Multiset α) : Prop :=
  M = N ∨ Multiset.IsDershowitzMannaLT M N

theorem DMLE.trans {α : Type*} [Preorder α] {M N P : Multiset α} (h₁ : DMLE M N)
    (h₂ : DMLE N P) : DMLE M P := by
  rcases h₁ with rfl | h₁
  · exact h₂
  rcases h₂ with rfl | h₂
  · exact Or.inr h₁
  · exact Or.inr (h₁.trans h₂)

theorem DMLE.lt_of_lt {α : Type*} [Preorder α] {M N P : Multiset α}
    (h₁ : Multiset.IsDershowitzMannaLT M N) (h₂ : DMLE N P) : Multiset.IsDershowitzMannaLT M P := by
  rcases h₂ with rfl | h₂
  · exact h₁
  · exact h₁.trans h₂

/-- A sub-multiset is smaller or equal. -/
theorem dmle_of_le {α : Type*} [Preorder α] {M N : Multiset α} (h : M ≤ N) : DMLE M N := by
  by_cases hMN : M = N
  · exact Or.inl hMN
  refine Or.inr ⟨M, 0, N - M, fun h0 => hMN (le_antisymm h ?_), by simp,
    (add_tsub_cancel_of_le h).symm, by simp⟩
  exact tsub_eq_zero_iff_le.mp h0

/-- If `ι : A → B` is injective with `f a ≤ g (ι a)`, then `f(A) ≤ g(B)` in the multiset order,
strictly unless `ι` is onto and `f a = g (ι a)` for all `a`. -/
theorem isDershowitzMannaLT_of_injOn {α β γ : Type*} [LinearOrder γ] {A : Finset α}
    {B : Finset β} (ι : α → β) (f : α → γ) (g : β → γ) (hmaps : ∀ a ∈ A, ι a ∈ B)
    (hinj : Set.InjOn ι A) (hle : ∀ a ∈ A, f a ≤ g (ι a))
    (hstrict : ¬ ((∀ a ∈ A, f a = g (ι a)) ∧ ∀ b ∈ B, ∃ a ∈ A, ι a = b)) :
    Multiset.IsDershowitzMannaLT (A.val.map f) (B.val.map g) := by
  set E := A.filter fun a => f a = g (ι a)
  set S := A.filter fun a => ¬ f a = g (ι a)
  set IE := E.image ι
  set R := B.filter (· ∉ IE)
  have hE : ∀ {a}, a ∈ E ↔ a ∈ A ∧ f a = g (ι a) := Finset.mem_filter
  -- an element of `A` that `ι` sends into `ι(E)` lies in `E`
  have hIE : ∀ a ∈ A, ι a ∈ IE → a ∈ E := fun a ha h => by
    obtain ⟨e, he, hea⟩ := Finset.mem_image.mp h
    rwa [← hinj (Finset.mem_filter.mp he).1 ha hea]
  refine ⟨E.val.map f, S.val.map f, R.val.map g, ?_, ?_, ?_, ?_⟩
  · intro hZ
    rw [Multiset.empty_eq_zero, Multiset.map_eq_zero] at hZ
    have hR : ∀ b ∈ B, b ∈ IE := fun b hb => by
      by_contra hb'
      have : b ∈ R.val := Finset.mem_val.mpr (Finset.mem_filter.mpr ⟨hb, hb'⟩)
      rw [hZ] at this
      exact Multiset.notMem_zero _ this
    refine hstrict ⟨fun a ha => (hE.mp (hIE a ha (hR _ (hmaps a ha)))).2, fun b hb => ?_⟩
    obtain ⟨e, he, rfl⟩ := Finset.mem_image.mp (hR b hb)
    exact ⟨e, (hE.mp he).1, rfl⟩
  · rw [← Multiset.map_add, Finset.filter_val, Finset.filter_val, Multiset.filter_add_not]
  · have hIEB : B.filter (· ∈ IE) = IE := by
      ext b
      simp only [Finset.mem_filter, and_iff_right_iff_imp]
      intro hb
      obtain ⟨e, he, rfl⟩ := Finset.mem_image.mp hb
      exact hmaps e (hE.mp he).1
    have hB : B.val = IE.val + R.val := by
      conv_lhs => rw [← Multiset.filter_add_not (· ∈ IE) B.val]
      rw [← Finset.filter_val, hIEB, ← Finset.filter_val]
    rw [hB, Multiset.map_add, Finset.image_val_of_injOn (hinj.mono fun a ha => (hE.mp ha).1),
      Multiset.map_map]
    congr 1
    exact Multiset.map_congr rfl fun a ha => (hE.mp (Finset.mem_val.mp ha)).2.symm
  · intro y hy
    obtain ⟨a, ha, rfl⟩ := Multiset.mem_map.mp hy
    obtain ⟨haA, hne⟩ := Finset.mem_filter.mp (Finset.mem_val.mp ha)
    refine ⟨g (ι a), Multiset.mem_map_of_mem g (Finset.mem_val.mpr ?_),
      lt_of_le_of_ne (hle a haA) hne⟩
    exact Finset.mem_filter.mpr ⟨hmaps a haA, fun h => hne (hE.mp (hIE a haA h)).2⟩

/-! ### The multiset of leading monomials -/

/-- The multiset of the leading monomials of the elements of `Q`. -/
def lmMeasure (m : MonomialOrder σ) (Q : Finset (MvPolynomial σ K)) : Multiset m.syn :=
  Q.val.map fun p => m.toSyn (m.degree p)

theorem dmle_lmMeasure_of_subset {Q Q' : Finset (MvPolynomial σ K)} (h : Q' ⊆ Q) :
    DMLE (lmMeasure m Q') (lmMeasure m Q) :=
  dmle_of_le (Multiset.map_le_map (Finset.val_le_iff.mpr h))

/-- One loop iteration of Algorithm `alg:interreduction` does not increase the multiset of
leading monomials. -/
theorem InterStep.lmMeasure_le {Q Q' : Finset (MvPolynomial σ K)} (hs : InterStep m Q Q')
    (h0 : (0 : MvPolynomial σ K) ∉ Q) : DMLE (lmMeasure m Q') (lmMeasure m Q) := by
  obtain ⟨p, hp, q, hq, -, u, hu, hdiv, rfl⟩ := hs
  have hq0 : q ≠ 0 := fun h => h0 (h ▸ hq)
  set E := Q.erase p
  rcases stepResult_eq_or (m := m) Q p q u with ⟨-, h⟩ | ⟨hr, h⟩ <;> rw [h]
  · exact dmle_lmMeasure_of_subset (Finset.erase_subset p Q)
  set r := reduceTerm m p q u
  by_cases hrE : r ∈ E
  · rw [Finset.insert_eq_of_mem hrE]
    exact dmle_lmMeasure_of_subset (Finset.erase_subset p Q)
  -- `LM(r)` is a monomial of `p` or below `u`, hence at most `LM(p)`
  have hrp : m.toSyn (m.degree r) ≤ m.toSyn (m.degree p) := by
    rcases mem_support_reduceTerm hq0 hdiv (m.degree_mem_support hr) with h' | h'
    · exact m.le_degree h'
    · exact h'.le.trans (m.le_degree hu)
  have hQ : lmMeasure m Q = m.toSyn (m.degree p) ::ₘ lmMeasure m E := by
    rw [lmMeasure, lmMeasure, Finset.erase_val]
    conv_lhs => rw [← Multiset.cons_erase (Finset.mem_val.mpr hp)]
    rw [Multiset.map_cons]
  rw [lmMeasure, Finset.insert_val_of_notMem hrE, Multiset.map_cons, ← lmMeasure, hQ]
  rcases hrp.eq_or_lt with heq | hlt
  · exact Or.inl (by rw [heq])
  refine Or.inr ⟨lmMeasure m E, {m.toSyn (m.degree r)}, {m.toSyn (m.degree p)}, by simp, ?_, ?_,
    by simpa using hlt⟩ <;> rw [add_comm, Multiset.singleton_add]

/-- The final normalization does not increase the multiset of leading monomials. -/
theorem lmMeasure_normalize_le {Q : Finset (MvPolynomial σ K)} (h0 : (0 : MvPolynomial σ K) ∉ Q) :
    DMLE (lmMeasure m (normalize m Q)) (lmMeasure m Q) := by
  apply dmle_of_le
  rw [lmMeasure, normalize, Finset.image_val]
  refine (Multiset.map_le_map (Multiset.dedup_le _)).trans (le_of_eq ?_)
  rw [Multiset.map_map, lmMeasure]
  exact Multiset.map_congr rfl fun p hp => by
    simp [Function.comp, degree_normalize fun h => h0 (h ▸ Finset.mem_val.mp hp)]

/-- Interreduction does not increase the multiset of leading monomials. -/
theorem IsInterRun.lmMeasure_le {Q Q' : Finset (MvPolynomial σ K)} (hr : IsInterRun m Q Q')
    (h0 : (0 : MvPolynomial σ K) ∉ Q) : DMLE (lmMeasure m Q') (lmMeasure m Q) := by
  obtain ⟨Q₁, hs, -, rfl⟩ := hr
  have hrun : DMLE (lmMeasure m Q₁) (lmMeasure m Q) := by
    induction hs with
    | refl => exact Or.inl rfl
    | tail hs' hstep ih =>
      exact (hstep.lmMeasure_le (reflTransGen_invariants hs' h0).1).trans ih
  exact (lmMeasure_normalize_le (reflTransGen_invariants hs h0).1).trans hrun

/-! ### Theorem `thm:refine-simpler` -/

namespace Refiner

variable {ρ : Refiner K m}

/-- Each element `h` of a minimal hitting set `H` of `Q̃` is the only element of `H` among the
factors of some `q ∈ Q̃`. -/
theorem exists_private_generator {Q H : Finset (MvPolynomial σ K)} (hH : H ∈ ρ.children Q)
    {h : MvPolynomial σ K} (hh : h ∈ H) :
    ∃ q ∈ ρ.tilde Q, h ∈ ρ.F.factors q ∧ ∀ f ∈ H, f ∈ ρ.F.factors q → f = h := by
  obtain ⟨⟨hsub, hmeet⟩, hmin⟩ := Factorizer.mem_minHitting.mp hH
  have : ∃ q ∈ ρ.tilde Q, ∀ f ∈ H.erase h, f ∉ ρ.F.factors q := by
    by_contra hc
    push Not at hc
    exact hmin _ (Finset.erase_ssubset hh) ⟨(Finset.erase_subset h H).trans hsub, hc⟩
  obtain ⟨q, hq, hnone⟩ := this
  have huniq : ∀ f ∈ H, f ∈ ρ.F.factors q → f = h := fun f hf hfq => by
    by_contra hfh
    exact hnone f (Finset.mem_erase.mpr ⟨hfh, hf⟩) hfq
  obtain ⟨f, hf, hfq⟩ := hmeet q hq
  exact ⟨q, hq, huniq f hf hfq ▸ hfq, huniq⟩

/-- The multiset of leading monomials decreases strictly from `Q̃` to a child `H ≠ Q̃`. -/
theorem lmMeasure_lt_tilde {Q H : Finset (MvPolynomial σ K)} (hH : H ∈ ρ.children Q)
    (hne : H ≠ ρ.tilde Q) :
    Multiset.IsDershowitzMannaLT (lmMeasure m H) (lmMeasure m (ρ.tilde Q)) := by
  have hQi := isInterreduced_tilde (ρ := ρ) Q
  choose! ι hιQ hιf hιu using fun h (hh : h ∈ H) => exists_private_generator hH hh
  have hι0 : ∀ h ∈ H, ι h ≠ 0 := fun h hh h0 => hQi.1 (h0 ▸ hιQ h hh)
  have hdvd : ∀ h ∈ H, h ∣ ι h := fun h hh => Factorizer.dvd_of_mem (hι0 h hh) (hιf h hh)
  -- `h = ι(h)` as soon as their leading monomials agree
  have heq : ∀ h ∈ H, m.toSyn (m.degree h) = m.toSyn (m.degree (ι h)) → h = ι h :=
    fun h hh hdeg => eq_of_dvd_of_degree_le (Factorizer.monic_of_mem (hι0 h hh) (hιf h hh))
      (hQi.2.1 _ (hιQ h hh)) (hdvd h hh) (m.toSyn.injective hdeg).ge
  refine isDershowitzMannaLT_of_injOn ι _ _ hιQ (fun h₁ hh₁ h₂ hh₂ h12 => ?_)
    (fun h hh => m.toSyn_monotone (degree_le_of_dvd (hdvd h hh) (hι0 h hh))) ?_
  · exact (hιu h₁ hh₁ h₂ hh₂ (h12 ▸ hιf h₂ hh₂)).symm
  rintro ⟨hall, honto⟩
  apply hne
  apply le_antisymm
  · intro h hh
    rw [heq h hh (hall h hh)]
    exact hιQ h hh
  · intro q hq
    obtain ⟨h, hh, rfl⟩ := honto q hq
    rw [← heq h hh (hall h hh)]
    exact hh

/-- **Theorem `thm:refine-simpler` (Refined branches are simpler).** Let `H ∈ 𝓗(Q̃)` with
`H ≠ Q̃`. The leading monomials of `H` are smaller than those of `Q` in the multiset extension
of `≺`, yet generate a strictly larger monomial ideal. -/
theorem refine_simpler {Q H : Finset (MvPolynomial σ K)} (h0 : (0 : MvPolynomial σ K) ∉ Q)
    (hH : H ∈ ρ.children Q) (hne : H ≠ ρ.tilde Q) :
    Multiset.IsDershowitzMannaLT (lmMeasure m H) (lmMeasure m Q) ∧
      lmIdeal m Q < lmIdeal m H := by
  have hrun : IsInterRun m Q (ρ.tilde Q) := by
    rw [tilde, Finset.erase_eq_of_notMem h0]
    exact ρ.I.isRun Q h0
  exact ⟨DMLE.lt_of_lt (lmMeasure_lt_tilde hH hne) (hrun.lmMeasure_le h0),
    lt_of_le_of_lt (hrun.lmIdeal_le h0) (lmIdeal_tilde_lt hH hne)⟩

end Refiner

end FactorBranching
