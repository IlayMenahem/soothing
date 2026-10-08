import FactorBranching.VariableZeroing

/-!
# Auxiliary generators under refinement

For a branch that carries the separate encoding of Subsection `subsec:nonzero-encoding`,
refinement keeps exactly one generator containing `s_x`, of the form `c (1 - s_x h_x)` with
`h_x ≡ x` modulo the auxiliary-free generators `H'`; hence
`π(V(H)) = {a ∈ V(H') : a_x ≠ 0, x ∈ N}`. Algorithm `alg:ZeroingBranching` no longer refines
such branches: it saturates instead, and Rabinowitsch generators occur only inside solver calls.

The paper's proof has three steps, each formalized here for an arbitrary monomial order:
* the leading monomial of `c (1 - s h)` contains `s` (`degree_auxGen_apply`), so it divides no
  term of a generator without `s` (`auxGen_reduces_none`): it reduces no other generator, and
  the generator for `s_y` cannot reduce the one for `s_x` (`auxGen_not_reduced_by_auxGen`);
* cancelling a term of `c (1 - s h)` by an auxiliary-free generator `q` gives `c (1 - s h')` with
  `h' - h ∈ ⟨q⟩` (`reduceTerm_auxGen`);
* being linear in `s`, `1 - s h` is irreducible (`irreducible_auxGen`), so it is its own factor set
  up to scaling and lies in every hitting set (`mem_of_isHitting_auxGen`).
The projection identity is `projection_auxGen`.
-/

open MvPolynomial

noncomputable section

open scoped Classical MonomialOrder

namespace FactorBranching

/-- The ring `K[x_y, s_y : y ∈ Y]` of the separate encoding (`Sum.inl y` is `x_y`, `Sum.inr y`
the auxiliary variable `s_y`). -/
abbrev ZPoly (K : Type*) [Field K] (Y : Type*) := MvPolynomial (Y ⊕ Y) K

theorem aeval_eq_of_vars {K : Type*} [Field K] {L : Type*} [Field L] [Algebra K L] {σ : Type*}
    {f g : σ → L} {q : MvPolynomial σ K} (h : ∀ v ∈ q.vars, f v = g v) :
    aeval f q = aeval g q :=
  hom_congr_vars (f₁ := (aeval f : MvPolynomial σ K →ₐ[K] L).toRingHom)
    (f₂ := (aeval g : MvPolynomial σ K →ₐ[K] L).toRingHom)
    (by ext; simp) (fun i hi _ => by simpa using h i hi) rfl

variable {K : Type*} [Field K] {σ : Type*}

/-- `c (1 - s h)`. -/
def auxGen (c : K) (s : σ) (h : MvPolynomial σ K) : MvPolynomial σ K :=
  C c * (1 - X s * h)

/-! ### The leading monomial contains `s` -/

section Degree

variable {m : MonomialOrder σ}

theorem degree_one_sub_X_mul {s : σ} {h : MvPolynomial σ K} (hh : h ≠ 0) :
    m.degree (1 - X s * h) = Finsupp.single s 1 + m.degree h := by
  have hXh : X s * h ≠ 0 := mul_ne_zero (X_ne_zero s) hh
  have hdeg : m.degree (X s * h) = Finsupp.single s 1 + m.degree h := by
    rw [m.degree_mul (X_ne_zero s) hh, m.degree_X]
  have hpos : m.degree (1 : MvPolynomial σ K) ≺[m] m.degree (X s * h) := by
    rw [m.degree_one, hdeg]
    refine lt_of_le_of_ne (by simp) fun h0 => ?_
    have := congrArg (· s) (m.toSyn.injective h0)
    simp at this; omega
  rw [show (1 : MvPolynomial σ K) - X s * h = -(X s * h - 1) by ring, m.degree_neg,
    m.degree_sub_of_lt hpos, hdeg]

/-- The leading monomial of `c (1 - s h)` contains `s`. -/
theorem degree_auxGen_apply {c : K} (hc : c ≠ 0) {s : σ} {h : MvPolynomial σ K} (hh : h ≠ 0) :
    1 ≤ m.degree (auxGen c s h) s := by
  have h1 : (1 : MvPolynomial σ K) - X s * h ≠ 0 := fun h0 => by
    have := congrArg (m.degree) h0
    rw [degree_one_sub_X_mul hh, m.degree_zero] at this
    have := congrArg (· s) this
    simp at this
  rw [auxGen, m.degree_mul (C_ne_zero.mpr hc) h1, m.degree_C, zero_add,
    degree_one_sub_X_mul hh]
  simp

/-- `c (1 - s h)` reduces no generator `q` in which `s` does not occur: its leading monomial
divides no term of `q`. -/
theorem auxGen_reduces_none {c : K} (hc : c ≠ 0) {s : σ} {h q : MvPolynomial σ K} (hh : h ≠ 0)
    (hq : s ∉ q.vars) : ∀ u ∈ q.support, ¬ m.degree (auxGen c s h) ≤ u := by
  intro u hu hle
  have hus : u s = 0 := by
    by_contra hne
    exact hq ((mem_vars_iff_mem_support s).mpr ⟨u, hu, Finsupp.mem_support_iff.mpr hne⟩)
  have := (degree_auxGen_apply (m := m) hc hh).trans (hle s)
  omega

theorem vars_auxGen_subset (c : K) (s : σ) (h : MvPolynomial σ K) :
    (auxGen c s h).vars ⊆ insert s h.vars := by
  intro v hv
  have h1 := vars_mul (C c) (1 - X s * h) hv
  rw [vars_C, Finset.empty_union] at h1
  rcases Finset.mem_union.mp (vars_sub_subset (p := 1) (q := X s * h) h1) with h2 | h2
  · simp at h2
  · rcases Finset.mem_union.mp (vars_mul _ _ h2) with h3 | h3
    · rw [vars_X, Finset.mem_singleton] at h3
      exact h3 ▸ Finset.mem_insert_self _ _
    · exact Finset.mem_insert_of_mem h3

/-- The generator for `s_y` cannot reduce the generator for `s_x` (`y ≠ x`, neither `h` involves
`s_y`). -/
theorem auxGen_not_reduced_by_auxGen {c c' : K} (hc' : c' ≠ 0) {s s' : σ} (hss' : s ≠ s')
    {h h' : MvPolynomial σ K} (hh' : h' ≠ 0) (hfree : s' ∉ h.vars) :
    ∀ u ∈ (auxGen c s h).support, ¬ m.degree (auxGen c' s' h') ≤ u :=
  auxGen_reduces_none hc' hh' fun hv => by
    rcases Finset.mem_insert.mp (vars_auxGen_subset c s h hv) with h1 | h1
    · exact hss' h1.symm
    · exact hfree h1

end Degree

/-! ### Reduction by an auxiliary-free generator keeps the form -/

/-- Cancelling a term of `c (1 - s h)` by a nonconstant generator `q` without `s` gives
`c (1 - s h')` with `h' - h ∈ ⟨q⟩`, and `h'` has no new variables. -/
theorem reduceTerm_auxGen {m : MonomialOrder σ} {c : K} (hc : c ≠ 0) {s : σ}
    {h q : MvPolynomial σ K} (hhs : s ∉ h.vars) (hqs : s ∉ q.vars) (hq0 : m.degree q ≠ 0)
    {u : σ →₀ ℕ} (hu : u ∈ (auxGen c s h).support) (hdiv : m.degree q ≤ u) :
    ∃ h' : MvPolynomial σ K, reduceTerm m (auxGen c s h) q u = auxGen c s h' ∧
      h' - h ∈ Ideal.span {q} ∧ h'.vars ⊆ h.vars ∪ q.vars := by
  -- `u` is not the constant term, so it is `single s 1 + d` with `d` in the support of `h`
  have hu0 : u ≠ 0 := fun h0 => hq0 (le_zero_iff.mp (h0 ▸ hdiv))
  have hu' : u ∈ (X s * h).support := by
    rw [auxGen, C_mul', support_smul_eq hc] at hu
    have := support_sub (σ := σ) (R := K) (1 : MvPolynomial σ K) (X s * h) hu
    rcases Finset.mem_union.mp this with h1 | h1
    · simp [support_one] at h1; exact absurd h1 hu0
    · exact h1
  rw [support_X_mul] at hu'
  obtain ⟨d, hd, rfl⟩ := Finset.mem_map.mp hu'
  have hds : d s = 0 := by
    by_contra hne
    exact hhs ((mem_vars_iff_mem_support s).mpr ⟨d, hd, Finsupp.mem_support_iff.mpr hne⟩)
  have hqs' : m.degree q s = 0 := by
    by_contra hne
    by_cases hq : q = 0
    · simp [hq] at hne
    exact hqs ((mem_vars_iff_mem_support s).mpr
      ⟨m.degree q, m.degree_mem_support hq, Finsupp.mem_support_iff.mpr hne⟩)
  have hsplit : (addLeftEmbedding (Finsupp.single s 1) d) - m.degree q =
      Finsupp.single s 1 + (d - m.degree q) := by
    ext v
    simp only [addLeftEmbedding_apply, Finsupp.coe_tsub, Pi.sub_apply, Finsupp.coe_add,
      Pi.add_apply]
    by_cases hv : v = s
    · subst hv; simp [hds, hqs']
    · simp [Ne.symm hv]
  have hX : ∀ k : K, monomial (Finsupp.single s 1 + (d - m.degree q)) k =
      X s * monomial (d - m.degree q) k := fun k => by
    rw [X, monomial_mul_monomial, one_mul]
  have hcc : C c * C c⁻¹ = (1 : MvPolynomial σ K) := by rw [← C_mul, mul_inv_cancel₀ hc, C_1]
  refine ⟨h + C c⁻¹ * monomial (d - m.degree q)
    ((auxGen c s h).coeff (addLeftEmbedding (Finsupp.single s 1) d) / m.leadingCoeff q) * q,
    ?_, ?_, ?_⟩
  · rw [reduceTerm, cancelFactor, hsplit, hX]
    generalize monomial (d - m.degree q)
      ((auxGen c s h).coeff (addLeftEmbedding (Finsupp.single s 1) d) / m.leadingCoeff q) = M
    rw [auxGen, auxGen]
    linear_combination (X s * M * q) * hcc
  · rw [add_sub_cancel_left]
    exact Ideal.mul_mem_left _ _ (Ideal.mem_span_singleton_self q)
  · intro v hv
    set k := (auxGen c s h).coeff (addLeftEmbedding (Finsupp.single s 1) d) / m.leadingCoeff q
    set t := monomial (d - m.degree q) k
    rcases Finset.mem_union.mp (vars_add_subset _ _ hv) with h1 | h1
    · exact Finset.mem_union_left _ h1
    · rcases Finset.mem_union.mp (vars_mul _ _ h1) with h2 | h2
      · rcases Finset.mem_union.mp (vars_mul _ _ h2) with h3 | h3
        · simp at h3
        · by_cases hk : k = 0
          · simp [t, hk] at h3
          rw [vars_monomial hk] at h3
          refine Finset.mem_union_left _ ((mem_vars_iff_mem_support v).mpr ⟨d, hd, ?_⟩)
          rw [Finsupp.mem_support_iff] at h3 ⊢
          have := (tsub_le_self : d - m.degree q ≤ d) v
          omega
      · exact Finset.mem_union_right _ h2

/-! ### Irreducibility -/

section Irreducible

variable {s : σ} {h : MvPolynomial σ K}

/-- The substitution `s ↦ t`, the identity on the other variables. -/
def substAt (s : σ) (t : MvPolynomial σ K) : MvPolynomial σ K →ₐ[K] MvPolynomial σ K :=
  aeval (Function.update X s t)

theorem substAt_of_notMem {g : MvPolynomial σ K} (hg : s ∉ g.vars) (t : MvPolynomial σ K) :
    substAt s t g = g := by
  conv_rhs => rw [← AlgHom.id_apply (R := K) g, ← aeval_X_left]
  refine hom_congr_vars (f₁ := (substAt s t).toRingHom)
    (f₂ := (aeval X : MvPolynomial σ K →ₐ[K] MvPolynomial σ K).toRingHom) (by ext; simp [substAt])
    (fun i hi _ => ?_) rfl
  have : i ≠ s := fun h' => hg (h' ▸ hi)
  simp [substAt, Function.update_of_ne this]

theorem substAt_one_sub (hs : s ∉ h.vars) (t : MvPolynomial σ K) :
    substAt s t (1 - X s * h) = 1 - t * h := by
  have := substAt_of_notMem (K := K) hs t
  rw [map_sub, map_one, map_mul, this]
  simp [substAt]

/-- **`1 - s h` is irreducible** when `h ≠ 0` does not involve `s`: it is linear in `s`, and
setting `s = 0` sends it to `1`, which forces the factor without `s` to be a unit. -/
theorem irreducible_one_sub_X_mul (hh : h ≠ 0) (hs : s ∉ h.vars) :
    Irreducible (1 - X s * h) := by
  set p : MvPolynomial σ K := 1 - X s * h
  have hp0 : substAt s 0 p = 1 := by simp [p, substAt_one_sub hs]
  have hp1 : substAt s 1 p = 1 - h := by simp [p, substAt_one_sub hs]
  have hpne : p ≠ 0 := fun h0 => by simp [h0] at hp0
  -- a factor not involving `s` is a unit
  have hunit : ∀ f g : MvPolynomial σ K, p = f * g → s ∉ g.vars → IsUnit g := by
    intro f g hfg hg
    have := hp0
    rw [hfg, map_mul, substAt_of_notMem hg] at this
    exact IsUnit.of_mul_eq_one_right _ this
  refine ⟨fun hu => ?_, fun {f g} hfg => ?_⟩
  · obtain ⟨r, -, hr⟩ := isUnit_iff_eq_C_of_isReduced.mp hu
    have h0 : substAt s 0 p = C r := by rw [hr]; exact substAt_of_notMem (by simp) 0
    have h1 : substAt s 1 p = C r := by rw [hr]; exact substAt_of_notMem (by simp) 1
    rw [hp0] at h0
    rw [hp1, ← h0] at h1
    exact hh (by simpa using h1)
  · have hdeg : degreeOf s p ≤ 1 := by
      refine (degreeOf_sub_le s _ _).trans (max_le (by simp [degreeOf_one]) ?_)
      refine (degreeOf_mul_le s _ _).trans ?_
      have : degreeOf s h = 0 := by
        by_contra hne; exact hs ((mem_vars_iff_degreeOf_ne_zero).mpr hne)
      simp [this]
    have hf : f ≠ 0 := left_ne_zero_of_mul (hfg ▸ hpne)
    have hg : g ≠ 0 := right_ne_zero_of_mul (hfg ▸ hpne)
    rw [hfg, degreeOf_mul_eq hf hg] at hdeg
    by_cases hgs : degreeOf s g = 0
    · exact Or.inr (hunit f g hfg fun hv => (mem_vars_iff_degreeOf_ne_zero).mp hv hgs)
    · have hfs : degreeOf s f = 0 := by omega
      exact Or.inl (hunit g f (by rw [hfg, mul_comm]) fun hv =>
        (mem_vars_iff_degreeOf_ne_zero).mp hv hfs)

theorem irreducible_auxGen {c : K} (hc : c ≠ 0) (hh : h ≠ 0) (hs : s ∉ h.vars) :
    Irreducible (auxGen c s h) :=
  (irreducible_isUnit_mul ((Ne.isUnit hc).map C)).mpr (irreducible_one_sub_X_mul hh hs)

/-- An irreducible generator is its own factor set up to scaling, so every hitting set contains
it: refinement keeps the generator for `s_x`. -/
theorem mem_of_isHitting_of_irreducible {m : MonomialOrder σ} (F : Factorizer K m)
    {Q H : Finset (MvPolynomial σ K)} {g : MvPolynomial σ K} (hgQ : g ∈ Q) (hg : Irreducible g)
    (hH : F.IsHitting Q H) : C (m.leadingCoeff g)⁻¹ * g ∈ H := by
  obtain ⟨f, hfH, hfg⟩ := hH.2 g hgQ
  obtain ⟨hfm, hfirr, hfdvd⟩ := (F.mem_factors hg.ne_zero).mp hfg
  have hassoc : Associated f (C (m.leadingCoeff g)⁻¹ * g) :=
    (hfirr.associated_of_dvd hg hfdvd).trans
      (associated_unit_mul_left g _ ((Ne.isUnit (inv_ne_zero
        (m.leadingCoeff_ne_zero_iff.mpr hg.ne_zero))).map C)).symm
  rwa [← eq_of_associated_of_monic hfm (monic_normalize hg.ne_zero) hassoc]

theorem mem_of_isHitting_auxGen {m : MonomialOrder σ} (F : Factorizer K m)
    {Q H : Finset (MvPolynomial σ K)} {c : K} (hc : c ≠ 0) (hh : h ≠ 0) (hs : s ∉ h.vars)
    (hgQ : auxGen c s h ∈ Q) (hH : F.IsHitting Q H) :
    C (m.leadingCoeff (auxGen c s h))⁻¹ * auxGen c s h ∈ H :=
  mem_of_isHitting_of_irreducible F hgQ (irreducible_auxGen hc hh hs) hH

end Irreducible

/-! ### The projection -/

section Projection

variable {Y : Type*} (L : Type*) [Field L] [Algebra K L]

/-- No auxiliary variable occurs. -/
def AuxFree (q : ZPoly K Y) : Prop :=
  ∀ v ∈ q.vars, ∃ y, v = Sum.inl y

variable {L}

theorem aeval_auxFree {q : ZPoly K Y} (hq : AuxFree q) (a : Y → L) (s s' : Y → L) :
    aeval (Sum.elim a s) q = aeval (Sum.elim a s') q :=
  aeval_eq_of_vars fun v hv => by obtain ⟨y, rfl⟩ := hq v hv; rfl

theorem aeval_eq_zero_of_mem_span {Q : Set (ZPoly K Y)} {b : Y ⊕ Y → L} (hb : b ∈ V L Q)
    {p : ZPoly K Y} (hp : p ∈ Ideal.span Q) : aeval b p = 0 := by
  rw [V_eq_zeroLocus] at hb
  exact hb p hp

variable (L)

/-- **Lemma `lem:auxiliary-generators`, projection.** If the auxiliary-free generators are `H'`
and, for each `y ∈ N`, the generator containing `s_y` is `c_y (1 - s_y h_y)` with `h_y`
auxiliary-free and `h_y ≡ x_y` modulo `⟨H'⟩`, then
`π(V(H)) = {a ∈ V(H') : a_y ≠ 0, y ∈ N}`. -/
theorem projection_auxGen (H' : Finset (ZPoly K Y)) (N : Finset Y) (c : Y → K)
    (h : Y → ZPoly K Y) (hc : ∀ y ∈ N, c y ≠ 0) (hH' : ∀ q ∈ H', AuxFree q)
    (hhx : ∀ y ∈ N, h y - X (Sum.inl y) ∈ Ideal.span (↑H' : Set (ZPoly K Y))) :
    {a : Y → L | ∃ s : Y → L, Sum.elim a s ∈
        V L ((↑H' : Set (ZPoly K Y)) ∪ (fun y => auxGen (c y) (Sum.inr y) (h y)) '' ↑N)} =
      {a | Sum.elim a (fun _ => 0) ∈ V L (↑H' : Set (ZPoly K Y)) ∧ ∀ y ∈ N, a y ≠ 0} := by
  have hH'V : ∀ (a s : Y → L), Sum.elim a s ∈ V L (↑H' : Set (ZPoly K Y)) ↔
      Sum.elim a (fun _ => 0) ∈ V L (↑H' : Set (ZPoly K Y)) := fun a s => by
    simp only [mem_V, Finset.mem_coe]
    exact forall₂_congr fun q hq => by rw [aeval_auxFree (hH' q hq) a s (fun _ => 0)]
  -- on `V(H')`, `h_y` takes the value `a_y`
  have hval : ∀ (a s : Y → L), Sum.elim a s ∈ V L (↑H' : Set (ZPoly K Y)) → ∀ y ∈ N,
      aeval (Sum.elim a s) (h y) = a y := fun a s hV y hy => by
    have := aeval_eq_zero_of_mem_span hV (hhx y hy)
    rw [map_sub, aeval_X, Sum.elim_inl, sub_eq_zero] at this
    exact this
  have haux : ∀ (a s : Y → L), Sum.elim a s ∈ V L (↑H' : Set (ZPoly K Y)) → ∀ y ∈ N,
      aeval (Sum.elim a s) (auxGen (c y) (Sum.inr y) (h y)) =
        algebraMap K L (c y) * (1 - s y * a y) :=
    fun a s hV y hy => by simp [auxGen, hval a s hV y hy]
  ext a
  simp only [Set.mem_ofPred_eq]
  constructor
  · rintro ⟨s, hs⟩
    have hV : Sum.elim a s ∈ V L (↑H' : Set (ZPoly K Y)) := fun q hq => hs q (Or.inl hq)
    refine ⟨(hH'V a s).mp hV, fun y hy hay => ?_⟩
    have := hs _ (Or.inr ⟨y, hy, rfl⟩)
    rw [haux a s hV y hy, hay, mul_zero, sub_zero, mul_one] at this
    exact hc y hy ((map_eq_zero_iff _ (algebraMap K L).injective).mp this)
  · rintro ⟨hV0, hN⟩
    refine ⟨fun y => (a y)⁻¹, ?_⟩
    have hV := (hH'V a fun y => (a y)⁻¹).mpr hV0
    rintro q (hq | ⟨y, hy, rfl⟩)
    · exact hV q hq
    · rw [haux a _ hV y hy, inv_mul_cancel₀ (hN y hy), sub_self, mul_zero]

end Projection

end FactorBranching
