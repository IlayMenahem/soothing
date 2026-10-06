import FactorBranching.Refine

/-!
# Saturation split and ideal refinement

Subsection `subsec:ideal-branching-framework`: Lemma `lem:power-branches`,
Lemma `lem:saturation-split`, Algorithm `alg:IdealRefine`, Theorem `thm:ideal-refine`, and
Corollary `cor:ideal-membership` together with the Gröbner-basis membership test it relies on
(Preliminaries, "Monomial orders and Gröbner bases").
-/

open MvPolynomial

noncomputable section

open scoped Classical

namespace FactorBranching

section Saturation

variable {R : Type*} [CommRing R]

/-- The saturation `J : u^∞ = {g : u^n g ∈ J for some n}`. -/
def saturation (J : Ideal R) (u : R) : Ideal R where
  carrier := {g | ∃ n : ℕ, u ^ n * g ∈ J}
  add_mem' := by
    rintro a b ⟨n, hn⟩ ⟨k, hk⟩
    refine ⟨n + k, ?_⟩
    have := J.add_mem (J.mul_mem_left (u ^ k) hn) (J.mul_mem_left (u ^ n) hk)
    convert this using 1
    ring
  zero_mem' := ⟨0, by simp⟩
  smul_mem' := by
    rintro c g ⟨n, hn⟩
    refine ⟨n, ?_⟩
    rw [smul_eq_mul, mul_left_comm]
    exact J.mul_mem_left c hn

theorem mem_saturation {J : Ideal R} {u g : R} : g ∈ saturation J u ↔ ∃ n : ℕ, u ^ n * g ∈ J :=
  Iff.rfl

theorem le_saturation (J : Ideal R) (u : R) : J ≤ saturation J u :=
  fun g hg => ⟨0, by simpa using hg⟩

theorem saturation_eq_top_iff {J : Ideal R} {u : R} : saturation J u = ⊤ ↔ u ∈ J.radical := by
  rw [Ideal.eq_top_iff_one, mem_saturation, Ideal.mem_radical_iff]
  simp

/-- **Lemma `lem:saturation-split` (Saturation split).** Let `S = J : u^∞` and `s ≥ 0` with
`u^s S ⊆ J`. Then `J = S ∩ (J + ⟨u^s⟩)`. -/
theorem saturation_split {J : Ideal R} {u : R} {s : ℕ}
    (hs : ∀ g ∈ saturation J u, u ^ s * g ∈ J) :
    J = saturation J u ⊓ (J ⊔ Ideal.span {u ^ s}) := by
  refine le_antisymm (le_inf (le_saturation J u) le_sup_left) ?_
  rintro g ⟨hgS, hgsup⟩
  obtain ⟨j, hj, y, hy, rfl⟩ := Submodule.mem_sup.mp hgsup
  obtain ⟨r, rfl⟩ := Ideal.mem_span_singleton'.mp hy
  -- `r u^s = g - j ∈ S`, so `r ∈ S`, so `u^s r ∈ J`
  have hrS : r ∈ saturation J u := by
    have : r * u ^ s ∈ saturation J u := by
      have := (saturation J u).sub_mem hgS (le_saturation J u hj)
      simpa using this
    obtain ⟨n, hn⟩ := this
    exact ⟨n + s, by rw [pow_add]; convert hn using 1; ring⟩
  exact J.add_mem hj (by rw [mul_comm]; exact hs r hrS)

end Saturation

variable {K : Type*} [Field K] {σ : Type*} {m : MonomialOrder σ}

/-! ### Lemma `lem:power-branches` -/

/-- **Lemma `lem:power-branches` (Power branches).** Let `I = ⟨P⟩`. There is `k₀` such that for
every `k ≥ k₀`, `I = ⋂_{H ∈ 𝓗(P)} (I + ⟨h^k : h ∈ H⟩)`. -/
theorem power_branches [Finite σ] (F : Factorizer K m) {P : Finset (MvPolynomial σ K)}
    (h0 : (0 : MvPolynomial σ K) ∉ P) :
    ∃ k₀, ∀ k ≥ k₀, Ideal.span (↑P : Set (MvPolynomial σ K)) =
      ⨅ H ∈ F.minHitting P, (Ideal.span (↑P : Set (MvPolynomial σ K)) ⊔
        Ideal.span ((fun h => h ^ k) '' (↑H : Set (MvPolynomial σ K)))) := by
  set I := Ideal.span (↑P : Set (MvPolynomial σ K))
  obtain ⟨t, ht, hprim⟩ := Submodule.isLasker (MvPolynomial σ K) (MvPolynomial σ K) I
  have hIq : ∀ q ∈ t, I ≤ q := fun q hq => ht ▸ Finset.inf_le (f := id) hq
  -- each primary component `q` contains a branch `H_q` raised to any power `k ≥ N_q`
  have hq : ∀ q ∈ t, ∃ H ∈ F.minHitting P, ∃ N : ℕ, ∀ h ∈ H, ∀ k, N ≤ k → h ^ k ∈ q := by
    intro q hqt
    have hrad : (Ideal.radical q).IsPrime := Ideal.isPrime_radical (hprim hqt)
    obtain ⟨H, hH, hHsub⟩ :=
      (F.subset_prime_iff h0 (Ideal.radical q)).mp
        (Ideal.span_le.mp ((hIq q hqt).trans Ideal.le_radical))
    obtain ⟨N, hN⟩ := Ideal.exists_pow_le_of_le_radical_of_fg (le_refl (Ideal.radical q))
      (IsNoetherian.noetherian _)
    refine ⟨H, hH, N, fun h hh k hk => hN ?_⟩
    rw [← pow_mul_pow_sub h hk]
    exact Ideal.mul_mem_right _ _ (Ideal.pow_mem_pow (hHsub hh) N)
  choose! Hq hHq N hN using hq
  refine ⟨t.sup N, fun k hk => le_antisymm (le_iInf₂ fun _ _ => le_sup_left) ?_⟩
  calc (⨅ H ∈ F.minHitting P, (I ⊔ Ideal.span ((fun h => h ^ k) '' (↑H : Set _))))
      ≤ t.inf id := Finset.le_inf fun q hqt => by
        refine (iInf₂_le (Hq q) (hHq q hqt)).trans (sup_le (hIq q hqt) ?_)
        rw [Ideal.span_le]
        rintro _ ⟨h, hh, rfl⟩
        exact hN q hqt h hh k ((Finset.le_sup hqt).trans hk)
    _ = I := ht

/-! ### Algorithm `alg:IdealRefine` and Theorem `thm:ideal-refine` -/

/-- A saturation solver: on `(J, u)` it returns `S = ⟨J⟩ : u^∞` and an exponent `s` with
`u^s S ⊆ ⟨J⟩`. -/
structure SatSolver (K : Type*) [Field K] (σ : Type*) where
  solve : Finset (MvPolynomial σ K) → MvPolynomial σ K → Ideal (MvPolynomial σ K) × ℕ
  sat_eq : ∀ J u, (solve J u).1 = saturation (Ideal.span (↑J : Set (MvPolynomial σ K))) u
  exp_spec : ∀ J u, ∀ g ∈ (solve J u).1,
    u ^ (solve J u).2 * g ∈ Ideal.span (↑J : Set (MvPolynomial σ K))

/-- The state of Algorithm `alg:IdealRefine`: residual products `U`, leaves `𝓕`, and the flag
`split` of the current round. -/
structure IRState (K : Type*) [Field K] (σ : Type*) where
  U : Finset (MvPolynomial σ K)
  leaves : Finset (Ideal (MvPolynomial σ K))
  split : Bool

namespace IdealRefine

variable (F : Factorizer K m) (𝒮 : SatSolver K σ) (P : Finset (MvPolynomial σ K))

/-- `u_H = ∏_{f ∈ C(G) ∖ H} f`, the product of the unchosen factors of the generators `G`. -/
def uH (G H : Finset (MvPolynomial σ K)) : MvPolynomial σ K :=
  ∏ f ∈ F.allFactors G \ H, f

/-- The current ideal `⟨P ∪ U⟩`. -/
def J (U : Finset (MvPolynomial σ K)) : Ideal (MvPolynomial σ K) :=
  Ideal.span (↑(P ∪ U) : Set (MvPolynomial σ K))

/-- The body of the inner loop for one hitting set `H` with saturating polynomial `u`. -/
def innerStep (u : MvPolynomial σ K) (st : IRState K σ) : IRState K σ :=
  let r := 𝒮.solve (P ∪ st.U) u
  ⟨insert (u ^ r.2) st.U, if r.1 = ⊤ then st.leaves else insert r.1 st.leaves,
    st.split || decide (r.1 ≠ ⊤)⟩

/-- One round: the inner loop over `𝓗(P ∪ U)` in a fixed order. -/
def round (U : Finset (MvPolynomial σ K)) (leaves : Finset (Ideal (MvPolynomial σ K))) :
    IRState K σ :=
  ((F.minHitting (P ∪ U)).toList.map (uH F (P ∪ U))).foldl (fun st u => innerStep 𝒮 P u st)
    ⟨U, leaves, false⟩

/-- `(U, 𝓕)` at the start of round `t`. -/
def stateAt : ℕ → Finset (MvPolynomial σ K) × Finset (Ideal (MvPolynomial σ K))
  | 0 => (∅, ∅)
  | t + 1 => let st := round F 𝒮 P (stateAt t).1 (stateAt t).2; (st.U, st.leaves)

variable {P} {𝒮}

theorem J_insert (U : Finset (MvPolynomial σ K)) (x : MvPolynomial σ K) :
    J P (insert x U) = J P U ⊔ Ideal.span {x} := by
  simp only [J, Finset.coe_union, Finset.coe_insert]
  rw [Set.union_insert, Ideal.span_insert, sup_comm]

/-- The invariant of the proof of Theorem `thm:ideal-refine`, maintained by one inner step:
`⟨P⟩ = ⋂ 𝓕 ∩ ⟨P ∪ U⟩` (Lemma `lem:saturation-split`), the radical of the current ideal grows,
and strictly so when the step splits off a leaf. -/
theorem innerStep_spec (u : MvPolynomial σ K) (st : IRState K σ) :
    let st' := innerStep 𝒮 P u st
    st.leaves.inf id ⊓ J P st.U = st'.leaves.inf id ⊓ J P st'.U ∧
      (J P st.U).radical ≤ (J P st'.U).radical ∧
      (st'.split = true → st.split = true ∨ (J P st.U).radical < (J P st'.U).radical) := by
  intro st'
  set S := (𝒮.solve (P ∪ st.U) u).1
  set s := (𝒮.solve (P ∪ st.U) u).2
  have hS : S = saturation (J P st.U) u := 𝒮.sat_eq _ _
  have hsplit : J P st.U = S ⊓ J P st'.U := by
    have := saturation_split (J := J P st.U) (u := u) (s := s)
      (fun g hg => 𝒮.exp_spec _ _ g (by show g ∈ S; rw [hS]; exact hg))
    rw [← hS] at this
    rw [show J P st'.U = J P st.U ⊔ Ideal.span {u ^ s} from J_insert _ _]
    exact this
  have hle : J P st.U ≤ J P st'.U := by
    rw [show J P st'.U = J P st.U ⊔ Ideal.span {u ^ s} from J_insert _ _]
    exact le_sup_left
  refine ⟨?_, Ideal.radical_mono hle, ?_⟩
  · by_cases htop : S = ⊤
    · have : st'.leaves = st.leaves := by simp [st', innerStep, S] at htop ⊢; simp [htop]
      rw [this, hsplit, htop, top_inf_eq]
    · have : st'.leaves = insert S st.leaves := by simp [st', innerStep, S] at htop ⊢; simp [htop]
      rw [this, Finset.inf_insert, id, hsplit, inf_left_comm, inf_assoc]
  · intro h
    by_cases htop : S = ⊤
    · left
      simpa [st', innerStep, S, htop] using h
    · right
      refine lt_of_le_of_ne (Ideal.radical_mono hle) fun heq => ?_
      -- `u ∈ √⟨P ∪ U'⟩` since `u^s ∈ ⟨P ∪ U'⟩`, but `u ∉ √⟨P ∪ U⟩` since `S ≠ ⟨1⟩`
      have hu : u ∈ (J P st'.U).radical :=
        ⟨s, Ideal.subset_span (by simp [st', innerStep, s])⟩
      rw [← heq, ← saturation_eq_top_iff, ← hS] at hu
      exact htop hu

theorem foldl_innerStep_spec (us : List (MvPolynomial σ K)) (st : IRState K σ) :
    let st' := us.foldl (fun st u => innerStep 𝒮 P u st) st
    st.leaves.inf id ⊓ J P st.U = st'.leaves.inf id ⊓ J P st'.U ∧
      (J P st.U).radical ≤ (J P st'.U).radical ∧
      (st'.split = true → st.split = true ∨ (J P st.U).radical < (J P st'.U).radical) := by
  induction us generalizing st with
  | nil => exact ⟨rfl, le_rfl, Or.inl⟩
  | cons u us ih =>
    obtain ⟨h1, h2, h3⟩ := innerStep_spec (𝒮 := 𝒮) (P := P) u st
    obtain ⟨h1', h2', h3'⟩ := ih (innerStep 𝒮 P u st)
    refine ⟨h1.trans h1', h2.trans h2', fun hs => ?_⟩
    rcases h3' hs with hs' | hlt
    · rcases h3 hs' with hs'' | hlt
      · exact Or.inl hs''
      · exact Or.inr (lt_of_lt_of_le hlt h2')
    · exact Or.inr (lt_of_le_of_lt h2 hlt)

theorem round_spec (U : Finset (MvPolynomial σ K)) (leaves : Finset (Ideal (MvPolynomial σ K))) :
    let st' := round F 𝒮 P U leaves
    leaves.inf id ⊓ J P U = st'.leaves.inf id ⊓ J P st'.U ∧
      (st'.split = true → (J P U).radical < (J P st'.U).radical) := by
  obtain ⟨h1, -, h3⟩ := foldl_innerStep_spec (𝒮 := 𝒮) (P := P)
    ((F.minHitting (P ∪ U)).toList.map (uH F (P ∪ U))) ⟨U, leaves, false⟩
  exact ⟨h1, fun hs => (h3 hs).resolve_left (by simp)⟩

theorem invariant_stateAt (t : ℕ) :
    Ideal.span (↑P : Set (MvPolynomial σ K)) =
      (stateAt F 𝒮 P t).2.inf id ⊓ J P (stateAt F 𝒮 P t).1 := by
  induction t with
  | zero => simp [stateAt, J]
  | succ t ih => rw [ih, (round_spec F _ _).1]; rfl

variable [Finite σ]

variable (𝒮 P) in
/-- **Theorem `thm:ideal-refine`, termination.** Some round of Algorithm `alg:IdealRefine` splits
nothing: otherwise `√⟨P ∪ U⟩` would grow strictly forever (the descending chain condition on
varieties, in its algebraic form). -/
theorem exists_round_no_split :
    ∃ t, (round F 𝒮 P (stateAt F 𝒮 P t).1 (stateAt F 𝒮 P t).2).split = false := by
  by_contra! h
  have hlt : ∀ t, (J P (stateAt F 𝒮 P t).1).radical < (J P (stateAt F 𝒮 P (t + 1)).1).radical :=
    fun t => (round_spec F _ _).2 (by simpa using h t)
  let f : ℕ →o Ideal (MvPolynomial σ K) :=
    ⟨fun t => (J P (stateAt F 𝒮 P t).1).radical, monotone_nat_of_le_succ fun t => (hlt t).le⟩
  obtain ⟨n, hn⟩ := monotone_stabilizes_iff_noetherian.mpr inferInstance f
  exact (hlt n).ne (hn (n + 1) (Nat.le_succ n))

variable (𝒮 P) in
/-- The leaves returned by Algorithm `alg:IdealRefine`: `𝓕 ∪ {⟨P ∪ U⟩}` after the first round
that splits nothing. -/
def leaves : Finset (Ideal (MvPolynomial σ K)) :=
  let t := Nat.find (exists_round_no_split F 𝒮 P)
  let st := round F 𝒮 P (stateAt F 𝒮 P t).1 (stateAt F 𝒮 P t).2
  insert (J P st.U) st.leaves

/-- **Theorem `thm:ideal-refine` (Exact ideal refinement).** If `𝒮` returns on every input,
Algorithm `alg:IdealRefine` terminates (`exists_round_no_split`) and returns a finite family
`𝓕` with `⟨P⟩ = ⋂_{S ∈ 𝓕} S`. -/
theorem span_eq_inf_leaves :
    Ideal.span (↑P : Set (MvPolynomial σ K)) = (leaves F 𝒮 P).inf id := by
  rw [leaves, Finset.inf_insert, id, inf_comm, invariant_stateAt F, (round_spec F _ _).1]

/-- Corollary `cor:ideal-membership`, first half: `g ∈ ⟨P⟩` iff `g` lies in every leaf. -/
theorem mem_span_iff_forall_leaves {g : MvPolynomial σ K} :
    g ∈ Ideal.span (↑P : Set (MvPolynomial σ K)) ↔ ∀ S ∈ leaves F 𝒮 P, g ∈ S := by
  rw [span_eq_inf_leaves F, Submodule.mem_finsetInf]
  rfl

end IdealRefine

/-! ### Gröbner bases decide membership -/

section Groebner

variable (m)

/-- `G` is a Gröbner basis of `I`: `G ⊆ I ∖ {0}` and `LM(I) = ⟨LM(g) : g ∈ G⟩`, i.e. the leading
monomial of every nonzero element of `I` is divisible by that of some `g ∈ G`. -/
def IsGroebnerBasis (G : Finset (MvPolynomial σ K)) (I : Ideal (MvPolynomial σ K)) : Prop :=
  (↑G : Set (MvPolynomial σ K)) ⊆ I ∧ (0 : MvPolynomial σ K) ∉ G ∧
    ∀ f ∈ I, f ≠ 0 → ∃ g ∈ G, m.degree g ≤ m.degree f

/-- `r` is a remainder of `g` on division by `G`: `g - r ∈ ⟨G⟩`, and no term of `r` is divisible
by a leading monomial of `G`. `g` reduces to zero modulo `G` when `0` is such a remainder. -/
def IsRemainder (G : Finset (MvPolynomial σ K)) (g r : MvPolynomial σ K) : Prop :=
  g - r ∈ Ideal.span (↑G : Set (MvPolynomial σ K)) ∧
    ∀ c ∈ r.support, ∀ b ∈ G, ¬ m.degree b ≤ c

variable {m}

/-- The division algorithm (Mathlib's `MonomialOrder.div_set`) produces a remainder. -/
theorem exists_isRemainder {G : Finset (MvPolynomial σ K)} (h0 : (0 : MvPolynomial σ K) ∉ G)
    (g : MvPolynomial σ K) : ∃ r, IsRemainder m G g r := by
  obtain ⟨c, r, hgr, -, hr⟩ := m.div_set (B := ↑G)
    (fun b hb => m.isUnit_leadingCoeff.mpr fun h => h0 (h ▸ hb)) g
  refine ⟨r, ?_, hr⟩
  have : g - r = Finsupp.linearCombination (MvPolynomial σ K)
      (fun b : (↑G : Set (MvPolynomial σ K)) => (b : MvPolynomial σ K)) c := by
    rw [hgr]; ring
  rw [this]
  have hrange := Finsupp.range_linearCombination (R := MvPolynomial σ K)
    (v := fun b : (↑G : Set (MvPolynomial σ K)) => (b : MvPolynomial σ K))
  rw [Subtype.range_coe] at hrange
  have hc := LinearMap.mem_range_self (Finsupp.linearCombination (MvPolynomial σ K)
    (fun b : (↑G : Set (MvPolynomial σ K)) => (b : MvPolynomial σ K))) c
  rw [hrange] at hc
  exact hc

/-- If `G` is a Gröbner basis of `I`, then `g ∈ I` iff every remainder of `g` modulo `G` is zero
(Preliminaries; Cox–Little–O'Shea Ch. 2 §6). -/
theorem mem_iff_remainder_eq_zero {G : Finset (MvPolynomial σ K)} {I : Ideal (MvPolynomial σ K)}
    (hG : IsGroebnerBasis m G I) {g r : MvPolynomial σ K} (hr : IsRemainder m G g r) :
    g ∈ I ↔ r = 0 := by
  have hGI : Ideal.span (↑G : Set (MvPolynomial σ K)) ≤ I := Ideal.span_le.mpr hG.1
  constructor
  · intro hg
    by_contra hr0
    have hrI : r ∈ I := by simpa using I.sub_mem hg (hGI hr.1)
    obtain ⟨b, hb, hbr⟩ := hG.2.2 r hrI hr0
    exact hr.2 _ (m.degree_mem_support hr0) b hb hbr
  · rintro rfl
    simpa using hGI hr.1

/-- **Corollary `cor:ideal-membership` (Ideal membership).** If every leaf is given by a Gröbner
basis, then `g ∈ ⟨P⟩` iff `g` reduces to zero modulo the basis of every leaf. -/
theorem mem_span_iff_reduces_to_zero [Finite σ] (F : Factorizer K m) (𝒮 : SatSolver K σ)
    {P : Finset (MvPolynomial σ K)} (basis : Ideal (MvPolynomial σ K) → Finset (MvPolynomial σ K))
    (hbasis : ∀ S ∈ IdealRefine.leaves F 𝒮 P, IsGroebnerBasis m (basis S) S)
    (g : MvPolynomial σ K) :
    g ∈ Ideal.span (↑P : Set (MvPolynomial σ K)) ↔
      ∀ S ∈ IdealRefine.leaves F 𝒮 P, ∀ r, IsRemainder m (basis S) g r → r = 0 := by
  rw [IdealRefine.mem_span_iff_forall_leaves F]
  refine forall₂_congr fun S hS => ⟨fun hg r hr => (mem_iff_remainder_eq_zero (hbasis S hS) hr).mp hg,
    fun h => ?_⟩
  obtain ⟨r, hr⟩ := exists_isRemainder (hbasis S hS).2.1 g
  exact (mem_iff_remainder_eq_zero (hbasis S hS) hr).mpr (h r hr)

end Groebner

end FactorBranching
