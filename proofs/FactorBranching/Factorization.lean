import FactorBranching.Monomial

/-!
# Factor sets, hitting sets, and one-round branch decomposition

Preliminaries ("Factor sets and hitting sets") and Theorem `thm:branch-decomposition`
(Section "Factor branching framework").

The factorization is an oracle `Factorizer` with the specification the paper fixes: for a nonzero
`q`, `factors q` is the set `C_q` of distinct monic, nonconstant irreducible factors of `q`, and
`q = c_q ∏_{f ∈ C_q} f^{e_{q,f}}` with `c_q ≠ 0` and `e_{q,f} ≥ 1`.
-/

open MvPolynomial

noncomputable section

open scoped Classical

namespace FactorBranching

variable {K : Type*} [Field K] {σ : Type*}

/-- A factorization oracle over `K` for the monomial order `m` ("monic" refers to `m`). -/
structure Factorizer (K : Type*) [Field K] {σ : Type*} (m : MonomialOrder σ) where
  /-- The factor set `C_q`. -/
  factors : MvPolynomial σ K → Finset (MvPolynomial σ K)
  mem_factors : ∀ {q f : MvPolynomial σ K}, q ≠ 0 →
    (f ∈ factors q ↔ m.Monic f ∧ Irreducible f ∧ f ∣ q)
  /-- The exponent `e_{q,f}`. -/
  mult : MvPolynomial σ K → MvPolynomial σ K → ℕ
  mult_pos : ∀ {q f : MvPolynomial σ K}, q ≠ 0 → f ∈ factors q → 1 ≤ mult q f
  /-- The constant `c_q`. -/
  lc : MvPolynomial σ K → K
  lc_ne_zero : ∀ {q : MvPolynomial σ K}, q ≠ 0 → lc q ≠ 0
  eq_prod : ∀ {q : MvPolynomial σ K}, q ≠ 0 → q = C (lc q) * ∏ f ∈ factors q, f ^ mult q f

namespace Factorizer

variable {m : MonomialOrder σ} (F : Factorizer K m)

section FactorFacts

variable {F} {q f : MvPolynomial σ K}

theorem monic_of_mem (hq : q ≠ 0) (hf : f ∈ F.factors q) : m.Monic f :=
  ((F.mem_factors hq).mp hf).1

theorem irreducible_of_mem (hq : q ≠ 0) (hf : f ∈ F.factors q) : Irreducible f :=
  ((F.mem_factors hq).mp hf).2.1

theorem dvd_of_mem (hq : q ≠ 0) (hf : f ∈ F.factors q) : f ∣ q :=
  ((F.mem_factors hq).mp hf).2.2

theorem ne_zero_of_mem (hq : q ≠ 0) (hf : f ∈ F.factors q) : f ≠ 0 :=
  (F.irreducible_of_mem hq hf).ne_zero

/-- A nonzero polynomial in a prime ideal has a factor in it. -/
theorem exists_factor_mem_of_mem_prime (hq : q ≠ 0) {𝔭 : Ideal (MvPolynomial σ K)} [𝔭.IsPrime]
    (hq𝔭 : q ∈ 𝔭) : ∃ f ∈ F.factors q, f ∈ 𝔭 := by
  rw [F.eq_prod hq] at hq𝔭
  rcases Ideal.IsPrime.mem_or_mem ‹_› hq𝔭 with hc | hprod
  · exact absurd (Ideal.eq_top_of_isUnit_mem _ hc ((F.lc_ne_zero hq).isUnit.map C))
      Ideal.IsPrime.ne_top'
  · obtain ⟨f, hf, hf𝔭⟩ := Ideal.IsPrime.prod_mem_iff.mp hprod
    exact ⟨f, hf, Ideal.IsPrime.mem_of_pow_mem ‹_› _ hf𝔭⟩

/-- The factor set of a monic irreducible polynomial is itself. -/
theorem factors_self (hm : m.Monic f) (hirr : Irreducible f) : F.factors f = {f} := by
  ext g
  rw [F.mem_factors hirr.ne_zero, Finset.mem_singleton]
  constructor
  · rintro ⟨hgm, hgirr, hgf⟩
    exact eq_of_associated_of_monic hgm hm (hgirr.associated_of_dvd hirr hgf)
  · rintro rfl
    exact ⟨hm, hirr, dvd_rfl⟩

end FactorFacts

/-- `C(Q) = ⋃_{q ∈ Q} C_q`. -/
def allFactors (Q : Finset (MvPolynomial σ K)) : Finset (MvPolynomial σ K) :=
  Q.biUnion F.factors

/-- `H` is a hitting set of `Q`: `H ⊆ C(Q)` and `H ∩ C_q ≠ ∅` for every `q ∈ Q`. -/
def IsHitting (Q H : Finset (MvPolynomial σ K)) : Prop :=
  H ⊆ F.allFactors Q ∧ ∀ q ∈ Q, ∃ f ∈ H, f ∈ F.factors q

/-- The family `𝓗(Q)` of minimal hitting sets of `Q`. -/
def minHitting (Q : Finset (MvPolynomial σ K)) : Finset (Finset (MvPolynomial σ K)) :=
  (F.allFactors Q).powerset.filter fun H => F.IsHitting Q H ∧ ∀ H' ⊂ H, ¬ F.IsHitting Q H'

variable {F}

theorem mem_allFactors {Q : Finset (MvPolynomial σ K)} {f : MvPolynomial σ K} :
    f ∈ F.allFactors Q ↔ ∃ q ∈ Q, f ∈ F.factors q := by
  simp [allFactors]

theorem mem_minHitting {Q H : Finset (MvPolynomial σ K)} :
    H ∈ F.minHitting Q ↔ F.IsHitting Q H ∧ ∀ H' ⊂ H, ¬ F.IsHitting Q H' := by
  simp only [minHitting, Finset.mem_filter, Finset.mem_powerset, and_iff_right_iff_imp]
  exact fun h => h.1.1

theorem isHitting_of_mem_minHitting {Q H : Finset (MvPolynomial σ K)} (h : H ∈ F.minHitting Q) :
    F.IsHitting Q H :=
  (mem_minHitting.mp h).1

/-- Every hitting set contains a minimal one. -/
theorem exists_minHitting_subset {Q : Finset (MvPolynomial σ K)} :
    ∀ {H : Finset (MvPolynomial σ K)}, F.IsHitting Q H → ∃ H' ∈ F.minHitting Q, H' ⊆ H := by
  intro H
  induction H using Finset.strongInduction with
  | H H ih =>
    intro hH
    by_cases h : ∃ H' ⊂ H, F.IsHitting Q H'
    · obtain ⟨H', hlt, hH'⟩ := h
      obtain ⟨H'', hmem, hsub⟩ := ih H' hlt hH'
      exact ⟨H'', hmem, hsub.trans hlt.subset⟩
    · push Not at h
      exact ⟨H, mem_minHitting.mpr ⟨hH, h⟩, subset_rfl⟩

/-- The prime-ideal form of Theorem `thm:branch-decomposition`: a prime contains `Q` exactly when
it contains a minimal hitting set of `Q`. Points are the special case of evaluation kernels
(`V_eq_iUnion_minHitting`), and radicals the intersection over all primes
(`radical_eq_iInf_minHitting`). -/
theorem subset_prime_iff {Q : Finset (MvPolynomial σ K)} (h0 : (0 : MvPolynomial σ K) ∉ Q)
    (𝔭 : Ideal (MvPolynomial σ K)) [𝔭.IsPrime] :
    (↑Q : Set (MvPolynomial σ K)) ⊆ 𝔭 ↔
      ∃ H ∈ F.minHitting Q, (↑H : Set (MvPolynomial σ K)) ⊆ 𝔭 := by
  constructor
  · intro hQ
    have hhit : F.IsHitting Q ((F.allFactors Q).filter (· ∈ 𝔭)) := by
      refine ⟨Finset.filter_subset _ _, fun q hq => ?_⟩
      have hq0 : q ≠ 0 := fun h => h0 (h ▸ hq)
      obtain ⟨f, hf, hf𝔭⟩ := F.exists_factor_mem_of_mem_prime hq0 (hQ hq)
      exact ⟨f, Finset.mem_filter.mpr ⟨mem_allFactors.mpr ⟨q, hq, hf⟩, hf𝔭⟩, hf⟩
    obtain ⟨H, hH, hsub⟩ := exists_minHitting_subset hhit
    exact ⟨H, hH, fun h hh => (Finset.mem_filter.mp (hsub hh)).2⟩
  · rintro ⟨H, hH, hsub⟩ q hq
    have hq0 : q ≠ 0 := fun h => h0 (h ▸ hq)
    obtain ⟨f, hfH, hfq⟩ := (isHitting_of_mem_minHitting hH).2 q hq
    exact Ideal.mem_of_dvd _ (dvd_of_mem hq0 hfq) (hsub hfH)

/-- **Theorem `thm:branch-decomposition` (One-round branch decomposition).**
For every finite `Q ⊆ R ∖ {0}`, `V(Q) = ⋃_{H ∈ 𝓗(Q)} V(H)`. The constant case is included:
a nonzero constant has no factors, so both sides are empty. -/
theorem V_eq_iUnion_minHitting (L : Type*) [CommRing L] [IsDomain L] [Algebra K L]
    {Q : Finset (MvPolynomial σ K)} (h0 : (0 : MvPolynomial σ K) ∉ Q) :
    V L (↑Q : Set (MvPolynomial σ K)) = ⋃ H ∈ F.minHitting Q, V L (↑H : Set (MvPolynomial σ K)) := by
  ext a
  simp only [Set.mem_iUnion, exists_prop, mem_V_iff_subset_ker]
  exact subset_prime_iff h0 _

/-- Membership in a radical, as the intersection of the primes above it. -/
theorem mem_radical_iff_forall_prime {I : Ideal (MvPolynomial σ K)} {x : MvPolynomial σ K} :
    x ∈ I.radical ↔ ∀ 𝔭 : Ideal (MvPolynomial σ K), 𝔭.IsPrime → I ≤ 𝔭 → x ∈ 𝔭 := by
  rw [Ideal.radical_eq_sInf, Ideal.mem_sInf]
  exact ⟨fun h 𝔭 hp hI => h ⟨hI, hp⟩, fun h 𝔭 hp => h 𝔭 hp.2 hp.1⟩

/-- The radical form of Theorem `thm:branch-decomposition`, proved without the Nullstellensatz:
`√⟨Q⟩ = ⋂_{H ∈ 𝓗(Q)} √⟨H⟩`. -/
theorem radical_eq_iInf_minHitting {Q : Finset (MvPolynomial σ K)}
    (h0 : (0 : MvPolynomial σ K) ∉ Q) :
    (Ideal.span (↑Q : Set (MvPolynomial σ K))).radical =
      ⨅ H ∈ F.minHitting Q, (Ideal.span (↑H : Set (MvPolynomial σ K))).radical := by
  ext x
  simp only [Submodule.mem_iInf, mem_radical_iff_forall_prime, Ideal.span_le]
  constructor
  · intro h H hH 𝔭 hp hH𝔭
    exact h 𝔭 hp ((subset_prime_iff h0 𝔭).mpr ⟨H, hH, hH𝔭⟩)
  · intro h 𝔭 hp hQ
    obtain ⟨H, hH, hH𝔭⟩ := (subset_prime_iff h0 𝔭).mp hQ
    exact h H hH 𝔭 hp hH𝔭

end Factorizer

end FactorBranching
