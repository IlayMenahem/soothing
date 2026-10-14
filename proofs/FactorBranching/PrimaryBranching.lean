import FactorBranching.PrimeBranching
import FactorBranching.IdealRefinement
import FactorBranching.IdealReduction

/-!
# Primary factor branching

Subsection `subsec:primary-decomposition`: Algorithm `alg:PrimaryBranching`, Theorem
`thm:primary-branching`, and Lemma `lem:isolated-separator`.

Primary factor branching reduces `P` to `Q` with Algorithm `alg:IdealReduce` (`⟨Q⟩ = ⟨P⟩`, Theorem
`thm:ideal-reduce`) and takes the minimal primes returned by `𝓑_min`. In its optional lines it
computes the isolated components from separators and returns them when they intersect to `⟨P⟩`
(`separatedComponents_unmixed`). Otherwise it decomposes `⟨Q⟩` with `𝓑_prim` and marks each
component isolated when its radical is one of the minimal primes, which containment and dimension
decide (`eq_of_le_of_ringKrullDim_eq`).

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

/-! ### Isolated components of a decomposition -/

/-- The isolated components of a decomposition: those whose radical is a minimal prime. -/
def isolatedComponents (I : Ideal (MvPolynomial σ K)) (t : Finset (Ideal (MvPolynomial σ K))) :
    Finset (Ideal (MvPolynomial σ K)) :=
  t.filter fun q => q.radical ∈ I.minimalPrimes

theorem mem_isolatedComponents {I q : Ideal (MvPolynomial σ K)} {t : Finset (Ideal (MvPolynomial σ K))} :
    q ∈ isolatedComponents I t ↔ q ∈ t ∧ q.radical ∈ I.minimalPrimes :=
  Finset.mem_filter

/-! ### Primary factor branching and Theorem `thm:primary-branching` -/

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

/-- Each `𝔮_i` is the isolated `𝔭_i`-primary component of `I` (Lemma `lem:isolated-separator`). -/
theorem top_saturation_separator [Finite σ] {I 𝔭 : Ideal (MvPolynomial σ K)}
    {t : Finset (Ideal (MvPolynomial σ K))} (ht : Submodule.IsMinimalPrimaryDecomposition I t)
    (h𝔭 : 𝔭 ∈ I.minimalPrimes) {s : MvPolynomial σ K} (hs : IsSeparator I 𝔭 s) (𝒯 : TopPartSolver K σ) :
    𝒯.top (saturation I s) ∈ t ∧ (𝒯.top (saturation I s)).radical = 𝔭 := by
  obtain ⟨-, q, hqt, hq𝔭, htop⟩ := isolated_separator ht h𝔭 hs
  rw [𝒯.spec _ _ (isMinimalPrimaryDecomposition_saturation ht _), htop, Finset.inf_singleton]
  exact ⟨hqt, hq𝔭⟩

/-- The components `𝔮_1, …, 𝔮_r` that the optional lines of Algorithm `alg:PrimaryBranching` compute from the minimal
primes `ps` and the separators `sep`. -/
def separatedComponents (ps : Finset (Ideal (MvPolynomial σ K))) (sep : Ideal (MvPolynomial σ K) → MvPolynomial σ K)
    (𝒯 : TopPartSolver K σ) (P : Finset (MvPolynomial σ K)) : Finset (Ideal (MvPolynomial σ K)) :=
  ps.image fun 𝔭 => 𝒯.top (saturation (Ideal.span (↑P : Set (MvPolynomial σ K))) (sep 𝔭))

/-- The computed components are exactly the isolated components of any minimal primary
decomposition of `⟨P⟩`. -/
theorem separatedComponents_eq [Finite σ] {P : Finset (MvPolynomial σ K)} {ps : Finset (Ideal (MvPolynomial σ K))}
    (hps : (↑ps : Set (Ideal (MvPolynomial σ K))) = (Ideal.span (↑P : Set (MvPolynomial σ K))).minimalPrimes)
    {sep : Ideal (MvPolynomial σ K) → MvPolynomial σ K}
    (hsep : ∀ 𝔭 ∈ ps, IsSeparator (Ideal.span (↑P : Set (MvPolynomial σ K))) 𝔭 (sep 𝔭))
    (𝒯 : TopPartSolver K σ) {t : Finset (Ideal (MvPolynomial σ K))}
    (ht : Submodule.IsMinimalPrimaryDecomposition (Ideal.span (↑P : Set (MvPolynomial σ K))) t) :
    separatedComponents ps sep 𝒯 P = isolatedComponents (Ideal.span (↑P : Set (MvPolynomial σ K))) t := by
  have hmin : ∀ 𝔭 ∈ ps, 𝔭 ∈ (Ideal.span (↑P : Set (MvPolynomial σ K))).minimalPrimes :=
    fun 𝔭 h𝔭 => hps ▸ h𝔭
  ext q
  simp only [separatedComponents, isolatedComponents, Finset.mem_image, Finset.mem_filter]
  constructor
  · rintro ⟨𝔭, h𝔭, rfl⟩
    obtain ⟨hqt, hrad⟩ := top_saturation_separator ht (hmin 𝔭 h𝔭) (hsep 𝔭 h𝔭) 𝒯
    exact ⟨hqt, by rw [hrad]; exact hmin 𝔭 h𝔭⟩
  · rintro ⟨hqt, hqmin⟩
    have hq : q.radical ∈ ps := by rw [← Finset.mem_coe, hps]; exact hqmin
    obtain ⟨hct, hcrad⟩ := top_saturation_separator ht hqmin (hsep _ hq) 𝒯
    exact ⟨q.radical, hq, radical_injOn ht hct hqt hcrad⟩


/-- **Isolated components by separators (optional lines).** If the components `𝔮_i` computed from
the separators intersect to `⟨P⟩`, they already form a minimal primary decomposition: their radicals
are the distinct minimal primes, and by prime avoidance no `𝔮_i` is redundant. -/
theorem separatedComponents_unmixed [Finite σ] {P : Finset (MvPolynomial σ K)}
    {ps : Finset (Ideal (MvPolynomial σ K))}
    (hps : (↑ps : Set (Ideal (MvPolynomial σ K))) = (Ideal.span (↑P : Set (MvPolynomial σ K))).minimalPrimes)
    {sep : Ideal (MvPolynomial σ K) → MvPolynomial σ K}
    (hsep : ∀ 𝔭 ∈ ps, IsSeparator (Ideal.span (↑P : Set (MvPolynomial σ K))) 𝔭 (sep 𝔭))
    (𝒯 : TopPartSolver K σ)
    (hinf : (separatedComponents ps sep 𝒯 P).inf id = Ideal.span (↑P : Set (MvPolynomial σ K))) :
    Submodule.IsMinimalPrimaryDecomposition (Ideal.span (↑P : Set (MvPolynomial σ K)))
      (separatedComponents ps sep 𝒯 P) := by
  obtain ⟨t, ht⟩ := Submodule.IsLasker.exists_isMinimalPrimaryDecomposition
    (Submodule.isLasker (MvPolynomial σ K) (MvPolynomial σ K)) (Ideal.span (↑P : Set (MvPolynomial σ K)))
  rw [separatedComponents_eq hps hsep 𝒯 ht] at hinf ⊢
  refine ⟨hinf, fun q hq => ht.primary (mem_isolatedComponents.mp hq).1,
    ht.distinct.mono fun q hq => (mem_isolatedComponents.mp hq).1, fun q hq hle => ?_⟩
  -- dropping `𝔮` leaves an intersection inside `√𝔮`, so another radical lies in this minimal prime
  obtain ⟨hqt, hqmin⟩ := mem_isolatedComponents.mp hq
  have hp : (Ideal.radical q).IsPrime := hqmin.1.1
  obtain ⟨K', hK', hK'q⟩ := (Ideal.IsPrime.inf_le' hp).mp (hle.trans Ideal.le_radical)
  obtain ⟨hne, hK'⟩ := Finset.mem_erase.mp hK'
  obtain ⟨hK't, hK'min⟩ := mem_isolatedComponents.mp hK'
  have hrad : Ideal.radical K' ≤ Ideal.radical q := hp.radical_le_iff.mpr hK'q
  exact hne (radical_injOn ht hK't hqt (le_antisymm hrad (hqmin.2 hK'min.1 hrad)))

/-- The mark needs no equality test: a prime that contains `𝔭` and has its dimension equals `𝔭`,
since a prime that strictly contains `𝔭` has smaller dimension. -/
theorem eq_of_le_of_ringKrullDim_eq [Finite σ] {𝔭 𝔮 : Ideal (MvPolynomial σ K)} [𝔭.IsPrime] [𝔮.IsPrime]
    (hle : 𝔭 ≤ 𝔮)
    (hdim : ringKrullDim ((MvPolynomial σ K) ⧸ 𝔮) = ringKrullDim ((MvPolynomial σ K) ⧸ 𝔭)) : 𝔭 = 𝔮 := by
  by_contra hne
  exact (ringKrullDim_quotient_lt (lt_of_le_of_ne hle hne)).ne hdim

/-- The mark of a component: isolated (`true`) when its radical is one of the minimal primes `ps`. -/
def mark (ps : Finset (Ideal (MvPolynomial σ K))) (q : Ideal (MvPolynomial σ K)) : Bool :=
  decide (q.radical ∈ ps)

/-- Algorithm `alg:PrimaryBranching` without its optional lines. `Q` is the output of Algorithm
`alg:IdealReduce` on `P`, and `ps` are the minimal primes returned by `𝓑_min`. Each component of
`𝓑_prim(Q)` is returned with its mark. -/
def primaryBranching (ps : Finset (Ideal (MvPolynomial σ K))) (𝓑prim : PrimarySolver K σ)
    (Q : Finset (MvPolynomial σ K)) : Finset (Ideal (MvPolynomial σ K) × Bool) :=
  (𝓑prim.decompose Q).image fun q => (q, mark ps q)

variable {m : MonomialOrder σ}

/-- **Theorem `thm:primary-branching` (Primary factor branching), without the optional lines.** Let `Q` be any
system that Algorithm `alg:IdealReduce` reaches from `P` (for any answers of the branch solver `𝓑`),
`ps` the minimal primes of `⟨P⟩`, and `𝓑_prim` a primary decomposition algorithm. Then the
components that Algorithm `alg:PrimaryBranching` returns form a minimal primary decomposition of `⟨P⟩`; a component
is marked isolated exactly when its radical is a minimal prime of `⟨P⟩`, and the radical of every
component marked embedded strictly contains a minimal prime. -/
theorem primaryBranching_spec [Finite σ] (ρ : Refiner K m) {P Q : Finset (MvPolynomial σ K)}
    (h0 : (0 : MvPolynomial σ K) ∉ P) (hrun : Relation.ReflTransGen ρ.ReduceStep (ρ.I.ired P) Q)
    (ps : Finset (Ideal (MvPolynomial σ K)))
    (hps : (↑ps : Set (Ideal (MvPolynomial σ K))) = (Ideal.span (↑P : Set (MvPolynomial σ K))).minimalPrimes)
    (𝓑prim : PrimarySolver K σ) :
    Submodule.IsMinimalPrimaryDecomposition (Ideal.span (↑P : Set (MvPolynomial σ K)))
        ((primaryBranching ps 𝓑prim Q).image Prod.fst) ∧
      ∀ x ∈ primaryBranching ps 𝓑prim Q,
        (x.2 = true ↔ x.1.radical ∈ (Ideal.span (↑P : Set (MvPolynomial σ K))).minimalPrimes) ∧
        (x.2 = false → ∃ 𝔭 ∈ (Ideal.span (↑P : Set (MvPolynomial σ K))).minimalPrimes, 𝔭 < x.1.radical) := by
  -- `⟨Q⟩ = ⟨P⟩` (Theorem `thm:ideal-reduce`), so `𝓑_prim` decomposes `⟨P⟩` itself
  have hD := 𝓑prim.spec Q
  rw [ρ.ideal_reduce_span h0 hrun] at hD
  have hfst : (primaryBranching ps 𝓑prim Q).image Prod.fst = 𝓑prim.decompose Q := by
    simp [primaryBranching, Finset.image_image, Function.comp_def]
  refine ⟨hfst ▸ hD, fun x hx => ?_⟩
  obtain ⟨q, hq, rfl⟩ := Finset.mem_image.mp hx
  have hmem : q.radical ∈ ps ↔ q.radical ∈ (Ideal.span (↑P : Set (MvPolynomial σ K))).minimalPrimes := by
    rw [← Finset.mem_coe, hps]
  simp only [mark, decide_eq_true_eq, decide_eq_false_iff_not]
  refine ⟨hmem, fun hnot => ?_⟩
  -- every radical contains `⟨P⟩`, hence a minimal prime, strictly unless it is one
  have : q.radical.IsPrime := Ideal.isPrime_radical (hD.primary hq)
  obtain ⟨𝔭, h𝔭, h𝔭q⟩ := Ideal.exists_minimalPrimes_le ((le_of_mem hD hq).trans Ideal.le_radical)
  exact ⟨𝔭, h𝔭, lt_of_le_of_ne h𝔭q fun h => hnot (hmem.mpr (by rw [← h]; exact h𝔭))⟩

/-- Algorithm `alg:PrimaryBranching` with its optional lines: if the components `𝔮_i` computed from
the separators intersect to `⟨Q⟩`, they are returned, all marked isolated; otherwise `𝓑_prim` runs. -/
def primaryBranchingSeparated (ps : Finset (Ideal (MvPolynomial σ K)))
    (sep : Ideal (MvPolynomial σ K) → MvPolynomial σ K) (𝒯 : TopPartSolver K σ)
    (𝓑prim : PrimarySolver K σ) (Q : Finset (MvPolynomial σ K)) :
    Finset (Ideal (MvPolynomial σ K) × Bool) :=
  if (separatedComponents ps sep 𝒯 Q).inf id = Ideal.span (↑Q : Set (MvPolynomial σ K)) then
    (separatedComponents ps sep 𝒯 Q).image fun q => (q, true)
  else primaryBranching ps 𝓑prim Q

/-- **Theorem `thm:primary-branching` (Primary factor branching), with the optional lines.** Under the
hypotheses of `primaryBranching_spec`, and with a separator `sep 𝔭` of every minimal prime `𝔭`, the
same conclusion holds. -/
theorem primaryBranchingSeparated_spec [Finite σ] (ρ : Refiner K m) {P Q : Finset (MvPolynomial σ K)}
    (h0 : (0 : MvPolynomial σ K) ∉ P) (hrun : Relation.ReflTransGen ρ.ReduceStep (ρ.I.ired P) Q)
    (ps : Finset (Ideal (MvPolynomial σ K)))
    (hps : (↑ps : Set (Ideal (MvPolynomial σ K))) = (Ideal.span (↑P : Set (MvPolynomial σ K))).minimalPrimes)
    {sep : Ideal (MvPolynomial σ K) → MvPolynomial σ K}
    (hsep : ∀ 𝔭 ∈ ps, IsSeparator (Ideal.span (↑P : Set (MvPolynomial σ K))) 𝔭 (sep 𝔭))
    (𝒯 : TopPartSolver K σ) (𝓑prim : PrimarySolver K σ) :
    Submodule.IsMinimalPrimaryDecomposition (Ideal.span (↑P : Set (MvPolynomial σ K)))
        ((primaryBranchingSeparated ps sep 𝒯 𝓑prim Q).image Prod.fst) ∧
      ∀ x ∈ primaryBranchingSeparated ps sep 𝒯 𝓑prim Q,
        (x.2 = true ↔ x.1.radical ∈ (Ideal.span (↑P : Set (MvPolynomial σ K))).minimalPrimes) ∧
        (x.2 = false → ∃ 𝔭 ∈ (Ideal.span (↑P : Set (MvPolynomial σ K))).minimalPrimes, 𝔭 < x.1.radical) := by
  unfold primaryBranchingSeparated
  split_ifs with hinf
  · -- the optional lines return: `⟨Q⟩ = ⟨P⟩`, so the separated components decompose `⟨P⟩`, all isolated
    have hspan := ρ.ideal_reduce_span h0 hrun
    rw [← hspan] at hps hsep
    obtain ⟨t, ht⟩ := Submodule.IsLasker.exists_isMinimalPrimaryDecomposition
      (Submodule.isLasker (MvPolynomial σ K) (MvPolynomial σ K)) (Ideal.span (↑Q : Set (MvPolynomial σ K)))
    have hfst : ((separatedComponents ps sep 𝒯 Q).image fun q => (q, true)).image Prod.fst =
        separatedComponents ps sep 𝒯 Q := by
      simp [Finset.image_image, Function.comp_def]
    rw [hfst, ← hspan]
    refine ⟨separatedComponents_unmixed hps hsep 𝒯 hinf, fun x hx => ?_⟩
    obtain ⟨q, hq, rfl⟩ := Finset.mem_image.mp hx
    rw [separatedComponents_eq hps hsep 𝒯 ht] at hq
    exact ⟨⟨fun _ => (mem_isolatedComponents.mp hq).2, fun _ => rfl⟩, fun h => by simp at h⟩
  · exact primaryBranching_spec ρ h0 hrun ps hps 𝓑prim

/-- **Theorem `thm:primary-branching` (Primary factor branching), membership.** If every component
of a minimal primary decomposition of `⟨P⟩`, such as the output of Algorithm `alg:PrimaryBranching`, is
given by a Gröbner basis, then `g ∈ ⟨P⟩` iff `g` reduces to zero modulo the basis of every
component. -/
theorem mem_span_iff_reduces_to_zero_components {P : Finset (MvPolynomial σ K)}
    {t : Finset (Ideal (MvPolynomial σ K))}
    (ht : Submodule.IsMinimalPrimaryDecomposition (Ideal.span (↑P : Set (MvPolynomial σ K))) t)
    (basis : Ideal (MvPolynomial σ K) → Finset (MvPolynomial σ K))
    (hbasis : ∀ q ∈ t, IsGroebnerBasis m (basis q) q) (g : MvPolynomial σ K) :
    g ∈ Ideal.span (↑P : Set (MvPolynomial σ K)) ↔
      ∀ q ∈ t, ∀ r, IsRemainder m (basis q) g r → r = 0 := by
  rw [← ht.inf_eq, Submodule.mem_finsetInf]
  refine forall₂_congr fun q hq => ⟨fun hg r hr => (mem_iff_remainder_eq_zero (hbasis q hq) hr).mp hg,
    fun h => ?_⟩
  obtain ⟨r, hr⟩ := exists_isRemainder (hbasis q hq).2.1 g
  exact (mem_iff_remainder_eq_zero (hbasis q hq) hr).mpr (h r hr)

end FactorBranching
