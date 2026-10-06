import FactorBranching.Refine

/-!
# Prime factor branching

Section "Prime factor branching": Algorithm `alg:PrimeBranching` and
Theorem `thm:prime-branching`, with the absorption optimization.

A prime branch solver `𝓑_min` returns the minimal primes of `⟨H⟩` (Gröbner bases are only a
representation and are not modelled).
-/

open MvPolynomial

noncomputable section

open scoped Classical

namespace FactorBranching

variable {R : Type*} [CommRing R]

/-- The minimal elements of a finite set of ideals under inclusion. -/
def minimalElements (M : Finset (Ideal R)) : Finset (Ideal R) :=
  M.filter fun 𝔭 => ∀ 𝔭' ∈ M, 𝔭' ≤ 𝔭 → 𝔭' = 𝔭

/-- The core of Theorem `thm:prime-branching`: if `M` is a finite set of primes containing `I`
whose intersection is `√I`, its minimal elements are exactly the minimal primes of `I`. -/
theorem minimalElements_eq_minimalPrimes {I : Ideal R} {M : Finset (Ideal R)}
    (hprime : ∀ 𝔭 ∈ M, 𝔭.IsPrime) (hinf : (⨅ 𝔭 ∈ M, 𝔭) = I.radical) :
    (↑(minimalElements M) : Set (Ideal R)) = I.minimalPrimes := by
  have hle : ∀ 𝔭 ∈ M, I ≤ 𝔭 := fun 𝔭 h𝔭 =>
    Ideal.le_radical.trans (hinf ▸ iInf₂_le 𝔭 h𝔭)
  -- every minimal prime of `I` lies in `M`, by prime avoidance
  have hmin_mem : ∀ 𝔮 ∈ I.minimalPrimes, 𝔮 ∈ M := by
    intro 𝔮 h𝔮
    have h𝔮p : 𝔮.IsPrime := h𝔮.1.1
    have : M.inf id ≤ 𝔮 := by
      rw [Finset.inf_eq_iInf]
      simpa [hinf] using (Ideal.radical_mono h𝔮.1.2).trans h𝔮p.radical.le
    obtain ⟨𝔭, h𝔭M, h𝔭𝔮⟩ := (Ideal.IsPrime.inf_le' h𝔮p).mp this
    have h𝔭𝔮' : 𝔭 = 𝔮 := le_antisymm h𝔭𝔮 (h𝔮.2 ⟨hprime 𝔭 h𝔭M, hle 𝔭 h𝔭M⟩ h𝔭𝔮)
    exact h𝔭𝔮' ▸ h𝔭M
  ext 𝔭
  simp only [minimalElements, Finset.coe_filter, Set.mem_ofPred_eq]
  constructor
  · rintro ⟨h𝔭M, hmin⟩
    have := hprime 𝔭 h𝔭M
    obtain ⟨𝔮, h𝔮, h𝔮𝔭⟩ := Ideal.exists_minimalPrimes_le (I := I) (J := 𝔭) (hle 𝔭 h𝔭M)
    exact hmin 𝔮 (hmin_mem 𝔮 h𝔮) h𝔮𝔭 ▸ h𝔮
  · intro h𝔭
    exact ⟨hmin_mem 𝔭 h𝔭, fun 𝔭' h𝔭' hle' =>
      le_antisymm hle' (h𝔭.2 ⟨hprime 𝔭' h𝔭', hle 𝔭' h𝔭'⟩ hle')⟩

variable {K : Type*} [Field K] {σ : Type*}

/-- A prime branch solver: on `H` it returns the minimal primes of `⟨H⟩` (the empty set exactly
when `V(H) = ∅`). -/
structure PrimeSolver (K : Type*) [Field K] (σ : Type*) where
  minPrimes : Finset (MvPolynomial σ K) → Finset (Ideal (MvPolynomial σ K))
  spec : ∀ H, (↑(minPrimes H) : Set (Ideal (MvPolynomial σ K))) =
    (Ideal.span (↑H : Set (MvPolynomial σ K))).minimalPrimes

namespace PrimeSolver

variable (𝓑 : PrimeSolver K σ)

theorem isPrime_of_mem {H : Finset (MvPolynomial σ K)} {𝔭} (h : 𝔭 ∈ 𝓑.minPrimes H) :
    𝔭.IsPrime := by
  have : 𝔭 ∈ (↑(𝓑.minPrimes H) : Set (Ideal (MvPolynomial σ K))) := h
  rw [𝓑.spec] at this
  exact this.1.1

theorem iInf_minPrimes (H : Finset (MvPolynomial σ K)) :
    (⨅ 𝔭 ∈ 𝓑.minPrimes H, 𝔭) = (Ideal.span (↑H : Set (MvPolynomial σ K))).radical := by
  rw [← Ideal.sInf_minimalPrimes, ← 𝓑.spec, sInf_eq_iInf]
  rfl

end PrimeSolver

namespace Refiner

variable {m : MonomialOrder σ} (ρ : Refiner K m) [Finite σ]

/-- The set `M` of found primes in Algorithm `alg:PrimeBranching`. -/
def foundPrimes (𝓑 : PrimeSolver K σ) (P : Finset (MvPolynomial σ K)) :
    Finset (Ideal (MvPolynomial σ K)) :=
  (ρ.refine P).biUnion 𝓑.minPrimes

/-- The output of Algorithm `alg:PrimeBranching`. -/
def primeBranching (𝓑 : PrimeSolver K σ) (P : Finset (MvPolynomial σ K)) :
    Finset (Ideal (MvPolynomial σ K)) :=
  minimalElements (ρ.foundPrimes 𝓑 P)

theorem iInf_foundPrimes (𝓑 : PrimeSolver K σ) {P : Finset (MvPolynomial σ K)}
    (h0 : (0 : MvPolynomial σ K) ∉ P) :
    (⨅ 𝔭 ∈ ρ.foundPrimes 𝓑 P, 𝔭) = (Ideal.span (↑P : Set (MvPolynomial σ K))).radical := by
  rw [ρ.radical_eq_iInf_refine h0, foundPrimes]
  simp_rw [← 𝓑.iInf_minPrimes, Finset.mem_biUnion]
  apply le_antisymm
  · exact le_iInf₂ fun H hH => le_iInf₂ fun 𝔭 h𝔭 => iInf₂_le 𝔭 ⟨H, hH, h𝔭⟩
  · exact le_iInf₂ fun 𝔭 ⟨H, hH, h𝔭⟩ => (iInf₂_le H hH).trans (iInf₂_le 𝔭 h𝔭)

/-- **Theorem `thm:prime-branching` (Prime branching).** If `𝓑_min` returns on every branch it
receives, Algorithm `alg:PrimeBranching` returns exactly the minimal primes of `⟨P⟩`. -/
theorem primeBranching_eq_minimalPrimes (𝓑 : PrimeSolver K σ) {P : Finset (MvPolynomial σ K)}
    (h0 : (0 : MvPolynomial σ K) ∉ P) :
    (↑(ρ.primeBranching 𝓑 P) : Set (Ideal (MvPolynomial σ K))) =
      (Ideal.span (↑P : Set (MvPolynomial σ K))).minimalPrimes :=
  minimalElements_eq_minimalPrimes
    (fun 𝔭 h𝔭 => by
      obtain ⟨H, -, h𝔭H⟩ := Finset.mem_biUnion.mp h𝔭
      exact 𝓑.isPrime_of_mem h𝔭H)
    (ρ.iInf_foundPrimes 𝓑 h0)

/-- The absorption optimization of Subsection `subsec:prime-branching-framework`: branches `H`
skipped because a found prime `𝔭 ⊆ ⟨H⟩` adds no component, so the minimal elements of the primes
found on the remaining branches are still the minimal primes of `⟨P⟩`. -/
theorem primeBranching_with_absorption (𝓑 : PrimeSolver K σ) {P : Finset (MvPolynomial σ K)}
    (h0 : (0 : MvPolynomial σ K) ∉ P) (kept : Finset (Finset (MvPolynomial σ K)))
    (hkept : kept ⊆ ρ.refine P)
    (habsorb : ∀ H ∈ ρ.refine P, H ∉ kept →
      ∃ 𝔭 ∈ kept.biUnion 𝓑.minPrimes, 𝔭 ≤ Ideal.span (↑H : Set (MvPolynomial σ K))) :
    (↑(minimalElements (kept.biUnion 𝓑.minPrimes)) : Set (Ideal (MvPolynomial σ K))) =
      (Ideal.span (↑P : Set (MvPolynomial σ K))).minimalPrimes := by
  refine minimalElements_eq_minimalPrimes (fun 𝔭 h𝔭 => ?_) ?_
  · obtain ⟨H, -, h𝔭H⟩ := Finset.mem_biUnion.mp h𝔭
    exact 𝓑.isPrime_of_mem h𝔭H
  rw [← ρ.iInf_foundPrimes 𝓑 h0]
  apply le_antisymm
  · refine le_iInf₂ fun 𝔭 h𝔭 => ?_
    obtain ⟨H, hH, h𝔭H⟩ := Finset.mem_biUnion.mp h𝔭
    by_cases hHk : H ∈ kept
    · exact iInf₂_le 𝔭 (Finset.mem_biUnion.mpr ⟨H, hHk, h𝔭H⟩)
    · -- a skipped branch: `𝔭' ⊆ ⟨H⟩ ⊆ √⟨H⟩ ⊆ 𝔭`
      obtain ⟨𝔭', h𝔭', h𝔭'H⟩ := habsorb H hH hHk
      refine (iInf₂_le 𝔭' h𝔭').trans (h𝔭'H.trans ?_)
      have : 𝔭 ∈ (↑(𝓑.minPrimes H) : Set (Ideal (MvPolynomial σ K))) := h𝔭H
      rw [𝓑.spec] at this
      exact this.1.2
  · exact le_iInf₂ fun 𝔭 h𝔭 => iInf₂_le 𝔭 (by
      obtain ⟨H, hH, h𝔭H⟩ := Finset.mem_biUnion.mp h𝔭
      exact Finset.mem_biUnion.mpr ⟨H, hkept hH, h𝔭H⟩)

end Refiner

end FactorBranching
