import FactorBranching.PrimeBranching
import FactorBranching.IdealRefinement

/-!
# Primary factor branching

Section "Primary factor branching": Algorithm `alg:PrimaryBranching`,
Lemma `lem:isolated-separator` and Theorem `thm:primary-branching`.

Primary decompositions are Mathlib's `Submodule.IsMinimalPrimaryDecomposition` (primary components
with distinct radicals, none redundant); for an ideal the radical `√(q : R)` there is
`(q.colon univ).radical = q.radical`. The dimension of a component `q` is the Krull dimension of
`R ⧸ √q`, the dimension of its variety.
-/

open MvPolynomial

noncomputable section

open scoped Classical

namespace FactorBranching

variable {K : Type*} [Field K] {σ : Type*}


section General

variable {A : Type*} [CommRing A]

theorem saturation_inf (I J : Ideal A) (u : A) :
    saturation (I ⊓ J) u = saturation I u ⊓ saturation J u := by
  ext g
  simp only [Submodule.mem_inf, mem_saturation]
  constructor
  · rintro ⟨n, hI, hJ⟩
    exact ⟨⟨n, hI⟩, ⟨n, hJ⟩⟩
  · rintro ⟨⟨a, ha⟩, ⟨b, hb⟩⟩
    refine ⟨a + b, ?_, ?_⟩
    · have := I.mul_mem_left (u ^ b) ha
      rw [pow_add]; convert this using 1; ring
    · rw [pow_add, mul_assoc]; exact J.mul_mem_left _ hb

theorem saturation_top (u : A) : saturation (⊤ : Ideal A) u = ⊤ :=
  eq_top_iff.mpr (le_saturation _ u)

theorem saturation_finsetInf (t : Finset (Ideal A)) (u : A) :
    saturation (t.inf id) u = t.inf fun q => saturation q u := by
  induction t using Finset.induction_on with
  | empty => simp [saturation_top]
  | insert a t _ ih => rw [Finset.inf_insert, Finset.inf_insert, id, saturation_inf, ih]

/-- Saturating a primary ideal by `u` gives `⟨1⟩` if `u ∈ √q` and `q` otherwise. -/
theorem saturation_of_isPrimary {q : Ideal A} (hq : q.IsPrimary) (u : A) :
    saturation q u = if u ∈ q.radical then ⊤ else q := by
  split_ifs with hu
  · exact saturation_eq_top_iff.mpr hu
  · refine le_antisymm (fun g ⟨n, hn⟩ => ?_) (le_saturation q u)
    rcases (Ideal.isPrimary_iff.mp hq).2 (show g * u ^ n ∈ q by rwa [mul_comm]) with hg | hun
    · exact hg
    · exact absurd ((Ideal.isPrime_radical hq).mem_of_pow_mem n hun) hu

/-- Saturation keeps the components whose radicals do not contain `u`,
`J : u^∞ = ⋂_{u ∉ 𝔭_j} 𝔮_j` (Preliminaries, Atiyah–Macdonald Prop. 4.9). -/
theorem saturation_finsetInf_primary {t : Finset (Ideal A)} (ht : ∀ q ∈ t, q.IsPrimary) (u : A) :
    saturation (t.inf id) u = (t.filter fun q => u ∉ q.radical).inf id := by
  rw [saturation_finsetInf, Finset.inf_congr rfl fun q hq => saturation_of_isPrimary (ht q hq) u,
    Finset.inf_ite, Finset.inf_top, top_inf_eq]
  rfl

theorem colon_univ_radical (q : Ideal A) : (q.colon Set.univ).radical = q.radical := by
  rw [Submodule.colon_univ]

end General

/-! ### Dimension -/

/-- The dimension of the variety of a component `q`: the Krull dimension of `R ⧸ √q`. -/
def dimC (q : Ideal (MvPolynomial σ K)) : WithBot ℕ∞ :=
  ringKrullDim ((MvPolynomial σ K) ⧸ q.radical)

/-- Primes strictly containing `𝔭` have smaller dimension (in `R = K[x₁,…,xₙ]`, which has finite
Krull dimension). -/
theorem ringKrullDim_quotient_lt [Finite σ] {𝔭 𝔭' : Ideal (MvPolynomial σ K)} [𝔭.IsPrime] [h𝔭' : 𝔭'.IsPrime]
    (h : 𝔭 < 𝔭') : ringKrullDim ((MvPolynomial σ K) ⧸ 𝔭') < ringKrullDim ((MvPolynomial σ K) ⧸ 𝔭) := by
  obtain ⟨x, hx𝔭', hx𝔭⟩ := SetLike.exists_of_lt h
  have hr : Ideal.Quotient.mk 𝔭 x ∈ nonZeroDivisors ((MvPolynomial σ K) ⧸ 𝔭) :=
    mem_nonZeroDivisors_of_ne_zero (by rwa [Ne, Ideal.Quotient.eq_zero_iff_mem])
  have hle := ringKrullDim_succ_le_of_surjective (Ideal.Quotient.factor h.le)
    (Ideal.Quotient.factor_surjective h.le) hr
    (by rw [Ideal.Quotient.factor_mk, Ideal.Quotient.eq_zero_iff_mem]; exact hx𝔭')
  have htop : ringKrullDim ((MvPolynomial σ K) ⧸ 𝔭') ≤ (Nat.card σ : WithBot ℕ∞) := by
    refine (ringKrullDim_quotient_le 𝔭').trans ?_
    rw [MvPolynomial.ringKrullDim_of_isNoetherianRing_of_finite, ringKrullDim_eq_zero_of_field,
      zero_add]
  have hbot : (0 : WithBot ℕ∞) ≤ ringKrullDim ((MvPolynomial σ K) ⧸ 𝔭') := ringKrullDim_nonneg_of_nontrivial
  obtain ⟨d, hd⟩ : ∃ d : ℕ, ringKrullDim ((MvPolynomial σ K) ⧸ 𝔭') = d := by
    generalize ringKrullDim ((MvPolynomial σ K) ⧸ 𝔭') = a at htop hbot ⊢
    induction a using WithBot.recBotCoe with
    | bot => exact absurd hbot (by simp)
    | coe e =>
      induction e using ENat.recTopCoe with
      | top => exact absurd htop (by intro h; norm_cast at h)
      | coe d => exact ⟨d, rfl⟩
  rw [hd] at hle ⊢
  refine lt_of_lt_of_le ?_ hle
  exact_mod_cast Nat.lt_succ_self d

/-! ### Separators and Lemma `lem:isolated-separator` -/

/-- `s` is a separator of the minimal prime `𝔭` of `I`: `s ∈ ⋂_{𝔭' ≠ 𝔭} 𝔭' ∖ 𝔭`. -/
def IsSeparator (I 𝔭 : Ideal (MvPolynomial σ K)) (s : (MvPolynomial σ K)) : Prop :=
  s ∉ 𝔭 ∧ ∀ 𝔭' ∈ I.minimalPrimes, 𝔭' ≠ 𝔭 → s ∈ 𝔭'

/-- A separator exists by prime avoidance, because `𝔭` contains no other minimal prime. -/
theorem exists_separator [Finite σ] {I 𝔭 : Ideal (MvPolynomial σ K)} (h𝔭 : 𝔭 ∈ I.minimalPrimes) :
    ∃ s, IsSeparator I 𝔭 s := by
  set T := (I.finite_minimalPrimes_of_isNoetherianRing (MvPolynomial σ K)).toFinset.erase 𝔭
  have hp : 𝔭.IsPrime := h𝔭.1.1
  have hnot : ¬ T.inf id ≤ 𝔭 := by
    intro hle
    obtain ⟨𝔭', h𝔭'T, h𝔭'𝔭⟩ := (Ideal.IsPrime.inf_le' hp).mp hle
    obtain ⟨hne, h𝔭'⟩ := Finset.mem_erase.mp h𝔭'T
    rw [Set.Finite.mem_toFinset] at h𝔭'
    exact hne (le_antisymm h𝔭'𝔭 (h𝔭.2 h𝔭'.1 h𝔭'𝔭))
  obtain ⟨s, hsT, hs𝔭⟩ := SetLike.not_le_iff_exists.mp hnot
  refine ⟨s, hs𝔭, fun 𝔭' h𝔭' hne => ?_⟩
  have : 𝔭' ∈ T := Finset.mem_erase.mpr ⟨hne, (Set.Finite.mem_toFinset _).mpr h𝔭'⟩
  exact Finset.inf_le (f := id) this hsT

/-- The components of a decomposition whose radicals do not contain `s`. -/
def keptComponents (t : Finset (Ideal (MvPolynomial σ K))) (s : (MvPolynomial σ K)) : Finset (Ideal (MvPolynomial σ K)) :=
  t.filter fun q => s ∉ q.radical

/-- The components of maximal dimension. -/
def topPart (t : Finset (Ideal (MvPolynomial σ K))) : Finset (Ideal (MvPolynomial σ K)) :=
  t.filter fun q => ∀ q' ∈ t, dimC q' ≤ dimC q

section Isolated

variable {I 𝔭 : Ideal (MvPolynomial σ K)} {t : Finset (Ideal (MvPolynomial σ K))} {s : (MvPolynomial σ K)}

theorem radical_eq_of_mem (ht : Submodule.IsMinimalPrimaryDecomposition I t) (h𝔭 : 𝔭 ∈ I.minimalPrimes) :
    ∃ q ∈ t, q.radical = 𝔭 := by
  obtain ⟨q, hq, hq𝔭⟩ := Ideal.IsMinimalPrimaryDecomposition.minimalPrimes_subset_image_radical ht h𝔭
  exact ⟨q, hq, hq𝔭⟩

theorem le_of_mem (ht : Submodule.IsMinimalPrimaryDecomposition I t) {q : Ideal (MvPolynomial σ K)} (hq : q ∈ t) : I ≤ q :=
  ht.inf_eq ▸ Finset.inf_le (f := id) hq

theorem radical_injOn (ht : Submodule.IsMinimalPrimaryDecomposition I t) {q q' : Ideal (MvPolynomial σ K)} (hq : q ∈ t)
    (hq' : q' ∈ t) (h : q.radical = q'.radical) : q = q' := by
  by_contra hne
  exact ht.distinct hq hq' hne (by simp only [colon_univ_radical]; exact h)

/-- The decomposition of `I` induces one of `I : s^∞`. -/
theorem isMinimalPrimaryDecomposition_saturation (ht : Submodule.IsMinimalPrimaryDecomposition I t) (s : (MvPolynomial σ K)) :
    Submodule.IsMinimalPrimaryDecomposition (saturation I s) (keptComponents t s) where
  inf_eq := by
    rw [keptComponents, ← saturation_finsetInf_primary (fun q hq => ht.primary hq), ht.inf_eq]
  primary := fun _ hq => ht.primary (Finset.mem_of_mem_filter _ hq)
  distinct := ht.distinct.mono (Finset.filter_subset _ _)
  minimal := fun J hJ hle => ht.minimal (Finset.mem_of_mem_filter _ hJ)
    ((Finset.inf_mono (Finset.erase_subset_erase J (Finset.filter_subset _ t))).trans hle)

/-- **Lemma `lem:isolated-separator` (Isolated components by separators).** Let `𝔭` be a minimal
prime of `I`, `s` a separator of `𝔭`, and `I = ⋂ 𝔮_α` a minimal primary decomposition. Then `𝔭` is
the only minimal prime of `I : s^∞`, and among the components of the induced decomposition of
`I : s^∞` those of maximal dimension consist of the `𝔭`-primary component `𝔮` of `I` alone. -/
theorem isolated_separator [Finite σ] (ht : Submodule.IsMinimalPrimaryDecomposition I t)
    (h𝔭 : 𝔭 ∈ I.minimalPrimes) (hs : IsSeparator I 𝔭 s) :
    (saturation I s).minimalPrimes = {𝔭} ∧
      ∃ q ∈ t, q.radical = 𝔭 ∧ topPart (keptComponents t s) = {q} := by
  have hp : 𝔭.IsPrime := h𝔭.1.1
  obtain ⟨q𝔭, hq𝔭t, hq𝔭⟩ := radical_eq_of_mem ht h𝔭
  set t' := keptComponents t s
  have hq𝔭t' : q𝔭 ∈ t' := Finset.mem_filter.mpr ⟨hq𝔭t, hq𝔭 ▸ hs.1⟩
  -- every kept radical contains `𝔭`
  have hkept : ∀ q ∈ t', 𝔭 ≤ q.radical := by
    intro q hq
    obtain ⟨hqt, hsq⟩ := Finset.mem_filter.mp hq
    have hrad : q.radical.IsPrime := Ideal.isPrime_radical (ht.primary hqt)
    obtain ⟨𝔭'', h𝔭'', h𝔭''q⟩ :=
      Ideal.exists_minimalPrimes_le ((le_of_mem ht hqt).trans Ideal.le_radical)
    by_cases heq : 𝔭'' = 𝔭
    · exact heq ▸ h𝔭''q
    · exact absurd (h𝔭''q (hs.2 𝔭'' h𝔭'' heq)) hsq
  -- the other kept components have strictly smaller dimension
  have hdim : ∀ q ∈ t', q ≠ q𝔭 → dimC q < dimC q𝔭 := by
    intro q hq hne
    have hlt : 𝔭 < q.radical := lt_of_le_of_ne (hkept q hq) fun h =>
      hne (radical_injOn ht (Finset.mem_of_mem_filter _ hq) hq𝔭t (h.symm.trans hq𝔭.symm))
    have := Ideal.isPrime_radical (ht.primary (Finset.mem_of_mem_filter _ hq))
    rw [dimC, dimC, hq𝔭]
    exact ringKrullDim_quotient_lt hlt
  refine ⟨?_, q𝔭, hq𝔭t, hq𝔭, ?_⟩
  · -- `I : s^∞ = ⋂ t'` has `𝔭` as its only minimal prime
    rw [← (isMinimalPrimaryDecomposition_saturation ht s).inf_eq]
    have hle𝔭 : t'.inf id ≤ 𝔭 :=
      (Finset.inf_le (f := id) hq𝔭t').trans (hq𝔭 ▸ Ideal.le_radical)
    have habove : ∀ 𝔮 : Ideal (MvPolynomial σ K), 𝔮.IsPrime → t'.inf id ≤ 𝔮 → 𝔭 ≤ 𝔮 := by
      intro 𝔮 h𝔮 hle
      obtain ⟨q, hq, hq𝔮⟩ := (Ideal.IsPrime.inf_le' h𝔮).mp hle
      exact (hkept q hq).trans (h𝔮.radical_le_iff.mpr hq𝔮)
    ext 𝔮
    rw [Set.mem_singleton_iff]
    constructor
    · intro h𝔮
      exact le_antisymm (h𝔮.2 ⟨hp, hle𝔭⟩ (habove 𝔮 h𝔮.1.1 h𝔮.1.2)) (habove 𝔮 h𝔮.1.1 h𝔮.1.2)
    · rintro rfl
      exact ⟨⟨hp, hle𝔭⟩, fun 𝔮 h𝔮 _ => habove 𝔮 h𝔮.1 h𝔮.2⟩
  · ext q
    simp only [topPart, Finset.mem_filter, Finset.mem_singleton]
    constructor
    · rintro ⟨hq, hmax⟩
      by_contra hne
      exact absurd (hmax q𝔭 hq𝔭t') (not_le_of_gt (hdim q hq hne))
    · rintro rfl
      refine ⟨hq𝔭t', fun q' hq' => ?_⟩
      by_cases h : q' = q
      · exact h ▸ le_rfl
      · exact (hdim q' hq' h).le

end Isolated

/-! ### Algorithm `alg:PrimaryBranching` and Theorem `thm:primary-branching` -/

/-- A primary decomposition algorithm `𝓑_prim`. -/
structure PrimarySolver (K : Type*) [Field K] (σ : Type*) where
  decompose : Finset (MvPolynomial σ K) → Finset (Ideal (MvPolynomial σ K))
  spec : ∀ P : Finset (MvPolynomial σ K),
    Submodule.IsMinimalPrimaryDecomposition (Ideal.span (↑P : Set (MvPolynomial σ K))) (decompose P)

/-- A top-dimensional-part algorithm (e.g. Eisenbud–Huneke–Vasconcelos): it returns the
intersection of the components of maximal dimension of a minimal primary decomposition. That this
does not depend on the decomposition is the uniqueness of isolated components
(Atiyah–Macdonald Cor. 4.11); here it is part of the specification. -/
structure TopPartSolver (K : Type*) [Field K] (σ : Type*) where
  top : Ideal (MvPolynomial σ K) → Ideal (MvPolynomial σ K)
  spec : ∀ J t, Submodule.IsMinimalPrimaryDecomposition J t → top J = (topPart t).inf id

/-- Algorithm `alg:PrimaryBranching`. `ps` are the minimal primes returned by `𝓑_min` and `sep`
chooses a separator for each. -/
def primaryBranching (ps : Finset (Ideal (MvPolynomial σ K))) (sep : Ideal (MvPolynomial σ K) → (MvPolynomial σ K)) (𝒯 : TopPartSolver K σ)
    (𝓑prim : PrimarySolver K σ) (P : Finset (MvPolynomial σ K)) : Finset (Ideal (MvPolynomial σ K)) :=
  let qs := ps.image fun 𝔭 => 𝒯.top (saturation (Ideal.span (↑P : Set (MvPolynomial σ K))) (sep 𝔭))
  if qs.inf id = Ideal.span (↑P : Set (MvPolynomial σ K)) then qs else 𝓑prim.decompose P

/-- **Theorem `thm:primary-branching` (Primary branching).** If `𝓑_min` and `𝓑_prim` return on
every input, Algorithm `alg:PrimaryBranching` returns a minimal primary decomposition of `⟨P⟩`. -/
theorem primaryBranching_spec [Finite σ] (P : Finset (MvPolynomial σ K)) (ps : Finset (Ideal (MvPolynomial σ K)))
    (hps : (↑ps : Set (Ideal (MvPolynomial σ K))) = (Ideal.span (↑P : Set (MvPolynomial σ K))).minimalPrimes)
    (sep : Ideal (MvPolynomial σ K) → (MvPolynomial σ K)) (hsep : ∀ 𝔭 ∈ ps, IsSeparator (Ideal.span (↑P : Set (MvPolynomial σ K))) 𝔭 (sep 𝔭))
    (𝒯 : TopPartSolver K σ) (𝓑prim : PrimarySolver K σ) :
    Submodule.IsMinimalPrimaryDecomposition (Ideal.span (↑P : Set (MvPolynomial σ K)))
      (primaryBranching ps sep 𝒯 𝓑prim P) := by
  set I := Ideal.span (↑P : Set (MvPolynomial σ K))
  rw [primaryBranching]
  split_ifs with hinf
  swap
  · exact 𝓑prim.spec P
  obtain ⟨t, ht⟩ := Submodule.IsLasker.exists_isMinimalPrimaryDecomposition (Submodule.isLasker (MvPolynomial σ K) (MvPolynomial σ K)) I
  have hmin : ∀ 𝔭 ∈ ps, 𝔭 ∈ I.minimalPrimes := fun 𝔭 h𝔭 => hps ▸ h𝔭
  -- each `𝔮_i` is the isolated `𝔭_i`-primary component of `I` (Lemma `lem:isolated-separator`)
  let comp := fun 𝔭 => 𝒯.top (saturation I (sep 𝔭))
  have hcomp : ∀ 𝔭 ∈ ps, comp 𝔭 ∈ t ∧ (comp 𝔭).radical = 𝔭 := by
    intro 𝔭 h𝔭
    obtain ⟨-, q, hqt, hq𝔭, htop⟩ := isolated_separator ht (hmin 𝔭 h𝔭) (hsep 𝔭 h𝔭)
    simp only [comp]
    rw [𝒯.spec _ _ (isMinimalPrimaryDecomposition_saturation ht _), htop, Finset.inf_singleton]
    exact ⟨hqt, hq𝔭⟩
  refine ⟨hinf, fun q hq => ?_, fun q hq q' hq' hne => ?_, fun q hq hle => ?_⟩
  · obtain ⟨𝔭, h𝔭, rfl⟩ := Finset.mem_image.mp hq
    exact ht.primary (hcomp 𝔭 h𝔭).1
  · obtain ⟨𝔭, h𝔭, rfl⟩ := Finset.mem_image.mp hq
    obtain ⟨𝔭', h𝔭', rfl⟩ := Finset.mem_image.mp hq'
    intro h
    apply hne
    have h' : (comp 𝔭).radical = (comp 𝔭').radical := by
      simpa [Function.onFun, colon_univ_radical] using h
    rw [(hcomp 𝔭 h𝔭).2, (hcomp 𝔭' h𝔭').2] at h'
    rw [h']
  · -- dropping `𝔮_i` leaves an intersection not contained in `𝔭_i`, by prime avoidance
    obtain ⟨𝔭, h𝔭, rfl⟩ := Finset.mem_image.mp hq
    have hp : 𝔭.IsPrime := (hmin 𝔭 h𝔭).1.1
    have hc𝔭 : comp 𝔭 ≤ 𝔭 := calc
      comp 𝔭 ≤ (comp 𝔭).radical := Ideal.le_radical
      _ = 𝔭 := (hcomp 𝔭 h𝔭).2
    obtain ⟨q', hq', hq'𝔭⟩ := (Ideal.IsPrime.inf_le' hp).mp (hle.trans hc𝔭)
    obtain ⟨hne, hq'img⟩ := Finset.mem_erase.mp hq'
    obtain ⟨𝔭', h𝔭', rfl⟩ := Finset.mem_image.mp hq'img
    have h𝔭'𝔭 : 𝔭' ≤ 𝔭 := calc
      𝔭' = (comp 𝔭').radical := (hcomp 𝔭' h𝔭').2.symm
      _ ≤ 𝔭 := hp.radical_le_iff.mpr hq'𝔭
    have : 𝔭' = 𝔭 := le_antisymm h𝔭'𝔭 ((hmin 𝔭 h𝔭).2 (hmin 𝔭' h𝔭').1 h𝔭'𝔭)
    exact hne (this ▸ rfl)

end FactorBranching
