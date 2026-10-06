import FactorBranching.Refine

/-!
# Ideal reduction

Subsection `subsec:sat-reduction`: Algorithm `alg:IdealReduce`, Lemma `lem:sat-branches` and
Theorem `thm:ideal-reduce`.

The branch solver is modelled by the family `𝓗_sat` it leaves uncertified: any subfamily of
`𝓗(Q)` that contains every SAT minimal hitting set is admissible ("unknown counts as SAT").
SAT is `⟨H⟩ ≠ ⟨1⟩`; by the Nullstellensatz (`sat_iff_V_nonempty`) this is `V(H) ≠ ∅`.
-/

open MvPolynomial

noncomputable section

open scoped Classical

namespace FactorBranching

variable {K : Type*} [Field K] {σ : Type*} {m : MonomialOrder σ}

namespace Factorizer

variable (F : Factorizer K m)

/-- `a_h = max {e_{q,h} : h ∈ C_q}`, the largest exponent of `h`. -/
def maxMult (Q : Finset (MvPolynomial σ K)) (h : MvPolynomial σ K) : ℕ :=
  (Q.filter fun q => h ∈ F.factors q).sup fun q => F.mult q h

/-- The cofactor `r_q = q / h^{e_{q,h}}`. -/
def cofactor (q h : MvPolynomial σ K) : MvPolynomial σ K :=
  C (F.lc q) * ∏ f ∈ (F.factors q).erase h, f ^ F.mult q f

variable {F}

theorem eq_pow_mul_cofactor {q h : MvPolynomial σ K} (hq : q ≠ 0) (hh : h ∈ F.factors q) :
    q = h ^ F.mult q h * F.cofactor q h := by
  conv_lhs => rw [F.eq_prod hq, ← Finset.mul_prod_erase _ _ hh]
  rw [cofactor]
  ring

theorem mult_le_maxMult {Q : Finset (MvPolynomial σ K)} {q h : MvPolynomial σ K} (hq : q ∈ Q)
    (hh : h ∈ F.factors q) : F.mult q h ≤ F.maxMult Q h :=
  Finset.le_sup (f := fun q => F.mult q h) (Finset.mem_filter.mpr ⟨hq, hh⟩)

/-- **Lemma `lem:sat-branches` (SAT branches).** Let `𝓗_sat ⊆ 𝓗(Q)` contain every SAT minimal
hitting set, and let `h ∈ C(Q)` lie in every `H ∈ 𝓗_sat`. Then `h^{a_h} ∈ ⟨Q⟩`. -/
theorem pow_maxMult_mem_span {Q : Finset (MvPolynomial σ K)} (h0 : (0 : MvPolynomial σ K) ∉ Q)
    (Hsat : Finset (Finset (MvPolynomial σ K)))
    (hHsat : ∀ H ∈ F.minHitting Q, Sat (↑H : Set (MvPolynomial σ K)) → H ∈ Hsat)
    {h : MvPolynomial σ K} (hh : ∀ H ∈ Hsat, h ∈ H) :
    h ^ F.maxMult Q h ∈ Ideal.span (↑Q : Set (MvPolynomial σ K)) := by
  set a := F.maxMult Q h
  set cofs := (Q.filter fun q => h ∈ F.factors q).image fun q => F.cofactor q h
  set J := Ideal.span ((↑Q : Set (MvPolynomial σ K)) ∪ ↑cofs)
  have hne : ∀ q ∈ Q, q ≠ 0 := fun q hq h' => h0 (h' ▸ hq)
  -- the cofactors of `h` have no common zero on `V(Q)`
  have hJ : J = ⊤ := by
    by_contra hJ
    obtain ⟨M, hM, hJM⟩ := Ideal.exists_le_maximal J hJ
    have hMp := hM.isPrime
    have hhit : F.IsHitting Q ((F.allFactors Q).filter fun f => f ≠ h ∧ f ∈ M) := by
      refine ⟨Finset.filter_subset _ _, fun q hq => ?_⟩
      have hfac : ∀ f ∈ F.factors q, f ≠ h → f ∈ M →
          ∃ f' ∈ (F.allFactors Q).filter (fun f => f ≠ h ∧ f ∈ M), f' ∈ F.factors q :=
        fun f hf hfh hfM =>
          ⟨f, Finset.mem_filter.mpr ⟨mem_allFactors.mpr ⟨q, hq, hf⟩, hfh, hfM⟩, hf⟩
      by_cases hhq : h ∈ F.factors q
      · have hcof : F.cofactor q h ∈ M := hJM (Ideal.subset_span (Or.inr (by
          simp only [cofs, Finset.coe_image, Finset.coe_filter]
          exact ⟨q, ⟨hq, hhq⟩, rfl⟩)))
        rcases hMp.mem_or_mem hcof with hc | hprod
        · exact absurd (Ideal.eq_top_of_isUnit_mem _ hc
            ((F.lc_ne_zero (hne q hq)).isUnit.map C)) hMp.ne_top
        · obtain ⟨f, hf, hfM⟩ := hMp.prod_mem_iff.mp hprod
          obtain ⟨hfh, hf⟩ := Finset.mem_erase.mp hf
          exact hfac f hf hfh (hMp.mem_of_pow_mem _ hfM)
      · obtain ⟨f, hf, hfM⟩ := F.exists_factor_mem_of_mem_prime (hne q hq)
          (hJM (Ideal.subset_span (Or.inl hq)))
        exact hfac f hf (fun hfh => hhq (hfh ▸ hf)) hfM
    obtain ⟨H', hH', hsub⟩ := exists_minHitting_subset hhit
    have hsat : Sat (↑H' : Set (MvPolynomial σ K)) := fun htop => hM.ne_top (by
      rw [eq_top_iff, ← htop, Ideal.span_le]
      exact fun f hf => (Finset.mem_filter.mp (hsub hf)).2.2)
    exact (Finset.mem_filter.mp (hsub (hh H' (hHsat H' hH' hsat)))).2.1 rfl
  -- multiplying by `h^{a_h}` sends every generator of `J` into `⟨Q⟩`
  have hle : J ≤ Submodule.comap (LinearMap.mulLeft (MvPolynomial σ K) (h ^ a))
      (Ideal.span (↑Q : Set (MvPolynomial σ K))) := by
    rw [Ideal.span_le]
    rintro x (hx | hx) <;> simp only [SetLike.mem_coe, Submodule.mem_comap,
      LinearMap.mulLeft_apply]
    · exact Ideal.mul_mem_left _ _ (Ideal.subset_span hx)
    · simp only [cofs, Finset.coe_image, Finset.coe_filter, Set.mem_image,
        Set.mem_ofPred_eq] at hx
      obtain ⟨q, ⟨hq, hhq⟩, rfl⟩ := hx
      rw [← pow_mul_pow_sub h (mult_le_maxMult hq hhq), mul_comm (h ^ F.mult q h), mul_assoc,
        ← eq_pow_mul_cofactor (hne q hq) hhq]
      exact Ideal.mul_mem_left _ _ (Ideal.subset_span hq)
  have := hle (hJ ▸ Submodule.mem_top : (1 : MvPolynomial σ K) ∈ J)
  simpa using this

/-- Lemma `lem:sat-branches` with SAT read geometrically: `𝓗_sat` contains every minimal hitting
set `H` with `V(H) ≠ ∅` over an algebraically closed `L`. -/
theorem pow_maxMult_mem_span' [Finite σ] (L : Type*) [Field L] [Algebra K L] [IsAlgClosed L]
    {Q : Finset (MvPolynomial σ K)} (h0 : (0 : MvPolynomial σ K) ∉ Q)
    (Hsat : Finset (Finset (MvPolynomial σ K)))
    (hHsat : ∀ H ∈ F.minHitting Q, (V L (↑H : Set (MvPolynomial σ K))).Nonempty → H ∈ Hsat)
    {h : MvPolynomial σ K} (hh : ∀ H ∈ Hsat, h ∈ H) :
    h ^ F.maxMult Q h ∈ Ideal.span (↑Q : Set (MvPolynomial σ K)) :=
  pow_maxMult_mem_span h0 Hsat (fun H hH hs => hHsat H hH ((sat_iff_V_nonempty L _).mp hs)) hh

end Factorizer

namespace Refiner

variable (ρ : Refiner K m)

/-- `𝓗_sat` is an admissible answer of the branch solver on `Q`. -/
def AdmissibleSat (Q : Finset (MvPolynomial σ K)) (Hsat : Finset (Finset (MvPolynomial σ K))) :
    Prop :=
  Hsat ⊆ ρ.F.minHitting Q ∧
    ∀ H ∈ ρ.F.minHitting Q, Sat (↑H : Set (MvPolynomial σ K)) → H ∈ Hsat

/-- `N = {h^{a_h} : h ∈ ⋂ 𝓗_sat} ∖ Q`, where the empty intersection is `C(Q)`. -/
def newMembers (Q : Finset (MvPolynomial σ K)) (Hsat : Finset (Finset (MvPolynomial σ K))) :
    Finset (MvPolynomial σ K) :=
  ((ρ.F.allFactors Q).filter fun h => ∀ H ∈ Hsat, h ∈ H).image
    (fun h => h ^ ρ.F.maxMult Q h) \ Q

/-- `Interreduce(N ∪ {q ∈ Q : no element of N divides q})`. -/
def reduceNext (Q N : Finset (MvPolynomial σ K)) : Finset (MvPolynomial σ K) :=
  ρ.I.ired (N ∪ Q.filter fun q => ∀ n ∈ N, ¬ n ∣ q)

/-- One iteration of the loop of Algorithm `alg:IdealReduce` that does not return, for some
admissible answer of the branch solver. -/
def ReduceStep (Q Q' : Finset (MvPolynomial σ K)) : Prop :=
  IsInterreduced m Q ∧ ∃ Hsat, ρ.AdmissibleSat Q Hsat ∧ (ρ.newMembers Q Hsat).Nonempty ∧
    Q' = ρ.reduceNext Q (ρ.newMembers Q Hsat)

variable {ρ}

theorem mem_newMembers {Q : Finset (MvPolynomial σ K)} {Hsat} {n : MvPolynomial σ K} :
    n ∈ ρ.newMembers Q Hsat ↔
      (∃ h ∈ ρ.F.allFactors Q, (∀ H ∈ Hsat, h ∈ H) ∧ h ^ ρ.F.maxMult Q h = n) ∧ n ∉ Q := by
  simp [newMembers, and_assoc]

theorem newMembers_ne_zero {Q : Finset (MvPolynomial σ K)} (h0 : (0 : MvPolynomial σ K) ∉ Q)
    {Hsat} {n : MvPolynomial σ K} (hn : n ∈ ρ.newMembers Q Hsat) : n ≠ 0 := by
  obtain ⟨⟨h, hhC, -, rfl⟩, -⟩ := mem_newMembers.mp hn
  obtain ⟨q, hq, hhq⟩ := Factorizer.mem_allFactors.mp hhC
  exact pow_ne_zero _ (Factorizer.ne_zero_of_mem (fun h' => h0 (h' ▸ hq)) hhq)

theorem zero_notMem_reduceInput {Q N : Finset (MvPolynomial σ K)} (h0 : (0 : MvPolynomial σ K) ∉ Q)
    (hN : ∀ n ∈ N, n ≠ 0) :
    (0 : MvPolynomial σ K) ∉ N ∪ Q.filter fun q => ∀ n ∈ N, ¬ n ∣ q := by
  rw [Finset.mem_union, not_or]
  exact ⟨fun h => hN 0 h rfl, fun h => h0 (Finset.mem_of_mem_filter _ h)⟩

namespace ReduceStep

variable {Q Q' : Finset (MvPolynomial σ K)}

/-- One round keeps the ideal: `N ⊆ ⟨Q⟩` by Lemma `lem:sat-branches`, the dropped generators
are multiples of elements of `N`, and interreduction keeps the ideal. -/
theorem span_eq (hs : ρ.ReduceStep Q Q') :
    Ideal.span (↑Q' : Set (MvPolynomial σ K)) = Ideal.span ↑Q := by
  obtain ⟨hQ, Hsat, ⟨-, hadm⟩, -, rfl⟩ := hs
  set N := ρ.newMembers Q Hsat
  have hN : ∀ n ∈ N, n ≠ 0 := fun n hn => newMembers_ne_zero hQ.1 hn
  rw [reduceNext, ρ.I.span_ired (zero_notMem_reduceInput hQ.1 hN)]
  apply le_antisymm <;> rw [Ideal.span_le]
  · intro x hx
    rcases Finset.mem_union.mp hx with hxN | hxQ
    · obtain ⟨⟨h, -, hh, rfl⟩, -⟩ := mem_newMembers.mp hxN
      exact Factorizer.pow_maxMult_mem_span hQ.1 Hsat hadm hh
    · exact Ideal.subset_span (Finset.mem_of_mem_filter _ hxQ)
  · intro q hq
    by_cases hkeep : ∀ n ∈ N, ¬ n ∣ q
    · exact Ideal.subset_span (Finset.mem_union_right _ (Finset.mem_filter.mpr ⟨hq, hkeep⟩))
    · push Not at hkeep
      obtain ⟨n, hn, hnq⟩ := hkeep
      exact Ideal.mem_of_dvd _ hnq (Ideal.subset_span (Finset.mem_union_left _ hn))

/-- One round strictly enlarges `⟨LM(q) : q ∈ Q⟩`. -/
theorem lmIdeal_lt (hs : ρ.ReduceStep Q Q') : lmIdeal m Q < lmIdeal m Q' := by
  obtain ⟨hQ, Hsat, -, ⟨n, hn⟩, rfl⟩ := hs
  set N := ρ.newMembers Q Hsat
  have hN : ∀ n ∈ N, n ≠ 0 := fun n hn => newMembers_ne_zero hQ.1 hn
  set S := N ∪ Q.filter fun q => ∀ n ∈ N, ¬ n ∣ q
  have hne : ∀ q ∈ Q, q ≠ 0 := fun q hq h => hQ.1 (h ▸ hq)
  have hle : lmIdeal m Q ≤ lmIdeal m S := by
    rw [lmIdeal_le_iff]
    intro q hq
    by_cases hkeep : ∀ n ∈ N, ¬ n ∣ q
    · exact ⟨q, Finset.mem_union_right _ (Finset.mem_filter.mpr ⟨hq, hkeep⟩), le_rfl⟩
    · push Not at hkeep
      obtain ⟨n, hn, hnq⟩ := hkeep
      exact ⟨n, Finset.mem_union_left _ hn, degree_le_of_dvd hnq (hne q hq)⟩
  have hnot : ¬ lmIdeal m S ≤ lmIdeal m Q := by
    intro hge
    rw [lmIdeal_le_iff] at hge
    obtain ⟨⟨h, hhC, -, rfl⟩, hnQ⟩ := mem_newMembers.mp hn
    -- a generator `q` with `e_{q,h} = a_h`, so `h^{a_h} ∣ q`
    have hfil : (Q.filter fun q => h ∈ ρ.F.factors q).Nonempty := by
      obtain ⟨q, hq, hhq⟩ := Factorizer.mem_allFactors.mp hhC
      exact ⟨q, Finset.mem_filter.mpr ⟨hq, hhq⟩⟩
    obtain ⟨q, hqf, hqmax⟩ := Finset.exists_mem_eq_sup _ hfil fun q => ρ.F.mult q h
    obtain ⟨hq, hhq⟩ := Finset.mem_filter.mp hqf
    have hdvd : h ^ ρ.F.maxMult Q h ∣ q := by
      rw [Factorizer.maxMult, hqmax]
      exact ⟨_, Factorizer.eq_pow_mul_cofactor (hne q hq) hhq⟩
    have hmonic : m.Monic (h ^ ρ.F.maxMult Q h) :=
      monic_pow (Factorizer.monic_of_mem (hne q hq) hhq) _
    obtain ⟨q', hq', hq'n⟩ := hge _ (Finset.mem_union_left _ hn)
    have hdeg := degree_le_of_dvd (m := m) hdvd (hne q hq)
    have hq'q : q' = q := by
      by_contra hne'
      exact hQ.2.2 q hq q' hq' (Ne.symm hne') (m.degree q)
        (m.degree_mem_support (hne q hq)) (hq'n.trans hdeg)
    subst hq'q
    exact hnQ (eq_of_dvd_of_degree_le hmonic (hQ.2.1 q' hq) hdvd hq'n ▸ hq)
  exact lt_of_lt_of_le (lt_of_le_not_ge hle hnot)
    (ρ.I.lmIdeal_le_ired (zero_notMem_reduceInput hQ.1 hN))

theorem isInterreduced (hs : ρ.ReduceStep Q Q') : IsInterreduced m Q' := by
  obtain ⟨hQ, Hsat, -, -, rfl⟩ := hs
  exact ρ.I.isInterreduced_ired
    (zero_notMem_reduceInput hQ.1 fun n hn => newMembers_ne_zero hQ.1 hn)

end ReduceStep

variable (ρ)

/-- **Theorem `thm:ideal-reduce` (Ideal reduction), termination.** For any answers of the branch
solver, Algorithm `alg:IdealReduce` makes finitely many rounds: there is no infinite chain of
rounds, since each strictly enlarges `⟨LM(q) : q ∈ Q⟩` and `R` is Noetherian. -/
theorem ideal_reduce_terminates [Finite σ] :
    WellFounded fun Q' Q : Finset (MvPolynomial σ K) => ρ.ReduceStep Q Q' :=
  Subrelation.wf (fun hs => ReduceStep.lmIdeal_lt hs) (InvImage.wf (lmIdeal m) wellFounded_gt)

/-- **Theorem `thm:ideal-reduce` (Ideal reduction), correctness.** Every system reached by
Algorithm `alg:IdealReduce` from `Interreduce(P)`, in particular the returned one, generates
`⟨P⟩`. -/
theorem ideal_reduce_span {P Q : Finset (MvPolynomial σ K)} (h0 : (0 : MvPolynomial σ K) ∉ P)
    (hrun : Relation.ReflTransGen ρ.ReduceStep (ρ.I.ired P) Q) :
    Ideal.span (↑Q : Set (MvPolynomial σ K)) = Ideal.span ↑P := by
  induction hrun with
  | refl => exact ρ.I.span_ired h0
  | tail _ hs ih => exact hs.span_eq.trans ih

end Refiner

end FactorBranching
