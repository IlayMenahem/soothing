import FactorBranching.Factorization
import FactorBranching.Interreduce

/-!
# Refinement and factor branching

Section "Factor branching framework": Algorithm `alg:Refine`, Algorithm `alg:Branching`,
Lemma `lem:refinement-step`, Theorem `thm:refine-termination`,
Lemma `lem:recursive-refinement`, Theorem `thm:branching-correctness`, and the pruning remark of
Subsection `subsec:pruning`.

A branch is a finite set of polynomials. `tilde Q` is `Q ∖ {0}` made monic and interreduced,
`children Q = 𝓗(tilde Q)`, and one round of Algorithm `alg:Refine` replaces every branch of the
work set `W` by its children.
-/

open MvPolynomial

noncomputable section

open scoped Classical

namespace FactorBranching

variable {K : Type*} [Field K] {σ : Type*}

/-- The two oracles that Algorithm `alg:Refine` uses. -/
structure Refiner (K : Type*) [Field K] {σ : Type*} (m : MonomialOrder σ) where
  F : Factorizer K m
  I : Interreducer K m

namespace Refiner

variable {m : MonomialOrder σ} (ρ : Refiner K m)

/-- `Q̃`: the system that one round factors, `Q ∖ {0}` made monic and interreduced. -/
def tilde (Q : Finset (MvPolynomial σ K)) : Finset (MvPolynomial σ K) :=
  ρ.I.ired (Q.erase 0)

/-- The children `𝓗(Q̃)` of a branch. -/
def children (Q : Finset (MvPolynomial σ K)) : Finset (Finset (MvPolynomial σ K)) :=
  ρ.F.minHitting (ρ.tilde Q)

/-- `Q` is terminal if one round leaves it unchanged, `𝓗(Q̃) = {Q}`. -/
def Terminal (Q : Finset (MvPolynomial σ K)) : Prop :=
  ρ.children Q = {Q}

/-- The progress measure `μ(Q) = ⟨LM(q) : q ∈ Q̃⟩`. -/
def μ (Q : Finset (MvPolynomial σ K)) : Ideal (MvPolynomial σ K) :=
  lmIdeal m (ρ.tilde Q)

/-- One round of the loop of Algorithm `alg:Refine`. -/
def round (W : Finset (Finset (MvPolynomial σ K))) : Finset (Finset (MvPolynomial σ K)) :=
  W.biUnion ρ.children

/-! ### Basic facts about one round -/

variable {ρ}

theorem zero_notMem_erase_zero (Q : Finset (MvPolynomial σ K)) :
    (0 : MvPolynomial σ K) ∉ Q.erase 0 :=
  Finset.notMem_erase 0 Q

theorem isInterreduced_tilde (Q : Finset (MvPolynomial σ K)) : IsInterreduced m (ρ.tilde Q) :=
  ρ.I.isInterreduced_ired (zero_notMem_erase_zero Q)

theorem zero_notMem_tilde (Q : Finset (MvPolynomial σ K)) : (0 : MvPolynomial σ K) ∉ ρ.tilde Q :=
  (isInterreduced_tilde Q).1

/-- Removing zero, interreduction and scaling preserve the ideal. -/
theorem span_tilde (Q : Finset (MvPolynomial σ K)) :
    Ideal.span (↑(ρ.tilde Q) : Set (MvPolynomial σ K)) = Ideal.span ↑Q := by
  rw [tilde, ρ.I.span_ired (zero_notMem_erase_zero Q), Finset.coe_erase,
    Ideal.span_sdiff_singleton_zero]

/-- Elements of a child are monic irreducible factors of elements of `Q̃`. -/
theorem monic_irreducible_of_mem_children {Q H : Finset (MvPolynomial σ K)}
    (hH : H ∈ ρ.children Q) {h : MvPolynomial σ K} (hh : h ∈ H) :
    ∃ q ∈ ρ.tilde Q, h ∈ ρ.F.factors q ∧ m.Monic h ∧ Irreducible h := by
  obtain ⟨q, hq, hhq⟩ :=
    Factorizer.mem_allFactors.mp ((Factorizer.isHitting_of_mem_minHitting hH).1 hh)
  have hq0 : q ≠ 0 := fun h0 => zero_notMem_tilde Q (h0 ▸ hq)
  exact ⟨q, hq, hhq, Factorizer.monic_of_mem hq0 hhq, Factorizer.irreducible_of_mem hq0 hhq⟩

theorem zero_notMem_of_mem_children {Q H : Finset (MvPolynomial σ K)} (hH : H ∈ ρ.children Q) :
    (0 : MvPolynomial σ K) ∉ H := fun h0 => by
  obtain ⟨-, -, -, -, hirr⟩ := monic_irreducible_of_mem_children hH h0
  exact hirr.ne_zero rfl

/-- A system of monic irreducible polynomials has itself as its only minimal hitting set. -/
theorem minHitting_of_monic_irreducible {H : Finset (MvPolynomial σ K)}
    (hH : ∀ h ∈ H, m.Monic h ∧ Irreducible h) : ρ.F.minHitting H = {H} := by
  have hfac : ∀ h ∈ H, ρ.F.factors h = {h} := fun h hh =>
    ρ.F.factors_self (hH h hh).1 (hH h hh).2
  have hall : ρ.F.allFactors H = H := by
    ext f
    simp only [Factorizer.mem_allFactors]
    constructor
    · rintro ⟨q, hq, hf⟩
      rw [hfac q hq, Finset.mem_singleton] at hf
      exact hf ▸ hq
    · intro hf
      exact ⟨f, hf, by rw [hfac f hf]; exact Finset.mem_singleton_self f⟩
  have hhit : ∀ H', ρ.F.IsHitting H H' ↔ H' = H := by
    intro H'
    constructor
    · rintro ⟨hsub, hmeet⟩
      refine le_antisymm (hall ▸ hsub) fun h hh => ?_
      obtain ⟨f, hf, hfh⟩ := hmeet h hh
      rw [hfac h hh, Finset.mem_singleton] at hfh
      exact hfh ▸ hf
    · rintro rfl
      exact ⟨hall.symm ▸ subset_rfl, fun h hh => ⟨h, hh, by rw [hfac h hh]; simp⟩⟩
  ext H'
  rw [Factorizer.mem_minHitting, Finset.mem_singleton, hhit]
  constructor
  · exact fun h => h.1
  · rintro rfl
    exact ⟨rfl, fun H'' hlt h => (hlt.ne ((hhit H'').mp h))⟩

/-! ### Lemma `lem:refinement-step` -/

/-- **Lemma `lem:refinement-step` (Progress of one refinement step).**
Let `H ∈ 𝓗(Q̃)`. Then `H` is terminal, or `μ(Q) ⊊ μ(H)`. -/
theorem refinement_step {Q H : Finset (MvPolynomial σ K)} (hH : H ∈ ρ.children Q) :
    ρ.Terminal H ∨ ρ.μ Q < ρ.μ H := by
  have hmi : ∀ h ∈ H, m.Monic h ∧ Irreducible h := fun h hh => by
    obtain ⟨-, -, -, hm, hirr⟩ := monic_irreducible_of_mem_children hH hh
    exact ⟨hm, hirr⟩
  have hH0 : (0 : MvPolynomial σ K) ∉ H := zero_notMem_of_mem_children hH
  have hQi := isInterreduced_tilde (ρ := ρ) Q
  set Q' := ρ.tilde Q with hQ'
  -- `H` meets the factor set of every element of `Q̃`
  have hmeet : ∀ q ∈ Q', ∃ h ∈ H, h ∈ ρ.F.factors q ∧ h ∣ q := fun q hq => by
    obtain ⟨h, hh, hhq⟩ := (Factorizer.isHitting_of_mem_minHitting hH).2 q hq
    have hq0 : q ≠ 0 := fun h0 => hQi.1 (h0 ▸ hq)
    exact ⟨h, hh, hhq, Factorizer.dvd_of_mem hq0 hhq⟩
  by_cases hEq : H = Q'
  · -- `H` is interreduced, so `H̃ = H` and its only minimal hitting set is `H`
    left
    have htilde : ρ.tilde H = H := by
      rw [tilde, Finset.erase_eq_of_notMem hH0, hEq]
      exact ρ.I.ired_eq_self hQi
    rw [Terminal, children, htilde]
    exact minHitting_of_monic_irreducible hmi
  right
  -- first step: `μ(Q) ⊊ ⟨LM(h) : h ∈ H⟩`
  have hle : lmIdeal m Q' ≤ lmIdeal m H := by
    rw [lmIdeal_le_iff]
    intro q hq
    obtain ⟨h, hh, -, hdvd⟩ := hmeet q hq
    exact ⟨h, hh, degree_le_of_dvd hdvd fun h0 => hQi.1 (h0 ▸ hq)⟩
  have hnot : ¬ lmIdeal m H ≤ lmIdeal m Q' := by
    intro hge
    rw [lmIdeal_le_iff] at hge
    -- an element of `H` that divides `q ∈ Q̃` equals `q`
    have key : ∀ h ∈ H, ∀ q ∈ Q', h ∣ q → h = q := by
      intro h hh q hq hdvd
      have hq0 : q ≠ 0 := fun h0 => hQi.1 (h0 ▸ hq)
      obtain ⟨q', hq', hq'h⟩ := hge h hh
      have hhq := degree_le_of_dvd (m := m) hdvd hq0
      have hq'q : q' = q := by
        by_contra hne
        exact hQi.2.2 q hq q' hq' (Ne.symm hne) (m.degree q)
          (m.degree_mem_support hq0) (hq'h.trans hhq)
      subst hq'q
      exact eq_of_dvd_of_degree_le (hmi h hh).1 (hQi.2.1 q' hq) hdvd hq'h
    apply hEq
    apply le_antisymm
    · intro h hh
      obtain ⟨q, hq, hhq, -⟩ := monic_irreducible_of_mem_children hH hh
      have hq0 : q ≠ 0 := fun h0 => hQi.1 (h0 ▸ hq)
      rwa [key h hh q hq (Factorizer.dvd_of_mem hq0 hhq)]
    · intro q hq
      obtain ⟨h, hh, -, hdvd⟩ := hmeet q hq
      rwa [← key h hh q hq hdvd]
  -- second step: `⟨LM(h) : h ∈ H⟩ ⊆ μ(H)` by Lemma `lem:lm-monotone`
  have hstep2 : lmIdeal m H ≤ ρ.μ H := by
    rw [μ, tilde, Finset.erase_eq_of_notMem hH0]
    exact ρ.I.lmIdeal_le_ired hH0
  exact lt_of_lt_of_le (lt_of_le_not_ge hle hnot) hstep2

/-! ### Theorem `thm:refine-termination` -/

section Termination


/-- `W` reaches a fixed point under iterated rounds. -/
def Stabilizes (ρ : Refiner K m) (W : Finset (Finset (MvPolynomial σ K))) : Prop :=
  ∃ k, ∀ j, k ≤ j → ρ.round^[j] W = ρ.round^[k] W

theorem round_biUnion (W : Finset (Finset (MvPolynomial σ K)))
    (g : Finset (MvPolynomial σ K) → Finset (Finset (MvPolynomial σ K))) :
    ρ.round (W.biUnion g) = W.biUnion fun Q => ρ.round (g Q) := by
  simp only [round, Finset.biUnion_biUnion]

theorem iterate_round_eq_biUnion (W : Finset (Finset (MvPolynomial σ K))) (j : ℕ) :
    ρ.round^[j] W = W.biUnion fun Q => ρ.round^[j] {Q} := by
  induction j with
  | zero => simp
  | succ j ih =>
    simp only [Function.iterate_succ_apply', ih, round_biUnion]

theorem round_singleton (Q : Finset (MvPolynomial σ K)) : ρ.round {Q} = ρ.children Q := by
  simp [round]

theorem stabilizes_of_terminal {Q : Finset (MvPolynomial σ K)} (hQ : ρ.Terminal Q) :
    Stabilizes ρ {Q} :=
  ⟨0, fun j _ => by
    rw [Function.iterate_fixed (by rw [round_singleton]; exact hQ)]
    rfl⟩

theorem stabilizes_of_forall {W : Finset (Finset (MvPolynomial σ K))}
    (h : ∀ Q ∈ W, Stabilizes ρ {Q}) : Stabilizes ρ W := by
  choose! k hk using h
  refine ⟨W.sup k, fun j hj => ?_⟩
  rw [iterate_round_eq_biUnion, iterate_round_eq_biUnion]
  refine Finset.biUnion_congr rfl fun Q hQ => ?_
  have hkQ : k Q ≤ W.sup k := Finset.le_sup hQ
  rw [hk Q hQ j (hkQ.trans hj), hk Q hQ _ hkQ]

variable [Finite σ] in
/-- Every branch reaches a fixed point: `μ` strictly increases along chains of nonterminal
branches (Lemma `lem:refinement-step`), and `R` is Noetherian. This replaces the paper's
Dickson/König argument by well-founded induction on `μ`. -/
theorem stabilizes_singleton (Q : Finset (MvPolynomial σ K)) : Stabilizes ρ {Q} := by
  have wf : WellFounded fun A B : Finset (MvPolynomial σ K) => ρ.μ B < ρ.μ A :=
    InvImage.wf ρ.μ wellFounded_gt
  induction Q using wf.induction with
  | _ Q ih =>
    by_cases hQ : ρ.Terminal Q
    · exact stabilizes_of_terminal hQ
    obtain ⟨k, hk⟩ : Stabilizes ρ (ρ.children Q) := stabilizes_of_forall fun H hH =>
      (refinement_step hH).elim stabilizes_of_terminal fun hlt => ih H hlt
    refine ⟨k + 1, fun j hj => ?_⟩
    obtain ⟨j, rfl⟩ : ∃ j', j = j' + 1 := ⟨j - 1, by omega⟩
    rw [Function.iterate_succ_apply, Function.iterate_succ_apply, round_singleton,
      hk j (by omega)]

/-- A fixed point of a round consists of terminal branches. -/
theorem terminal_of_round_eq {W : Finset (Finset (MvPolynomial σ K))} (hW : ρ.round W = W) :
    ∀ Q ∈ W, ρ.Terminal Q := by
  by_contra! hex
  obtain ⟨Q, hQW, hmin⟩ :=
    Finset.exists_minimalFor ρ.μ (W.filter fun Q => ¬ ρ.Terminal Q)
      (by obtain ⟨Q, hQ, hQt⟩ := hex; exact ⟨Q, Finset.mem_filter.mpr ⟨hQ, hQt⟩⟩)
  rw [Finset.mem_filter] at hQW
  obtain ⟨Q', hQ'W, hQQ'⟩ := Finset.mem_biUnion.mp (hW.symm ▸ hQW.1 : Q ∈ ρ.round W)
  by_cases hQ't : ρ.Terminal Q'
  · rw [hQ't, Finset.mem_singleton] at hQQ'
    exact hQW.2 (hQQ' ▸ hQ't)
  · rcases refinement_step hQQ' with ht | hlt
    · exact hQW.2 ht
    · exact absurd (hmin (Finset.mem_filter.mpr ⟨hQ'W, hQ't⟩) hlt.le) (not_le_of_gt hlt)

variable (ρ)

variable [Finite σ]

/-- The work set `W₀ = 𝓗(P)` before the first round. -/
def initial (P : Finset (MvPolynomial σ K)) : Finset (Finset (MvPolynomial σ K)) :=
  ρ.F.minHitting P

/-- **Theorem `thm:refine-termination` (Termination of refinement).** Algorithm `alg:Refine`
terminates on every finite polynomial system: some round leaves the work set unchanged. -/
theorem refine_terminates (P : Finset (MvPolynomial σ K)) :
    ∃ k, ρ.round^[k + 1] (ρ.initial P) = ρ.round^[k] (ρ.initial P) := by
  obtain ⟨k, hk⟩ := stabilizes_of_forall (W := ρ.initial P) fun Q _ => stabilizes_singleton Q
  exact ⟨k, hk (k + 1) (Nat.le_succ k)⟩

/-- `𝓛(P)`: the work set at the first round that leaves it unchanged, which
Algorithm `alg:Refine` returns. -/
def refine (P : Finset (MvPolynomial σ K)) : Finset (Finset (MvPolynomial σ K)) :=
  ρ.round^[Nat.find (ρ.refine_terminates P)] (ρ.initial P)

theorem round_refine (P : Finset (MvPolynomial σ K)) : ρ.round (ρ.refine P) = ρ.refine P := by
  have := Nat.find_spec (ρ.refine_terminates P)
  rwa [Function.iterate_succ_apply'] at this

/-- The output of Algorithm `alg:Refine` is a finite set of terminal branches. -/
theorem terminal_of_mem_refine {P H : Finset (MvPolynomial σ K)} (hH : H ∈ ρ.refine P) :
    ρ.Terminal H :=
  terminal_of_round_eq (ρ.round_refine P) H hH

end Termination

/-! ### Lemma `lem:recursive-refinement` -/

section Preservation

/-- One round preserves which primes contain some branch. -/
theorem exists_subset_prime_round (W : Finset (Finset (MvPolynomial σ K)))
    (𝔭 : Ideal (MvPolynomial σ K)) [𝔭.IsPrime] :
    (∃ Q ∈ W, (↑Q : Set (MvPolynomial σ K)) ⊆ 𝔭) ↔
      ∃ H ∈ ρ.round W, (↑H : Set (MvPolynomial σ K)) ⊆ 𝔭 := by
  have hQ : ∀ Q : Finset (MvPolynomial σ K), (↑Q : Set (MvPolynomial σ K)) ⊆ 𝔭 ↔
      ∃ H ∈ ρ.children Q, (↑H : Set (MvPolynomial σ K)) ⊆ 𝔭 := fun Q => by
    rw [← Ideal.span_le, ← span_tilde, Ideal.span_le]
    exact ρ.F.subset_prime_iff (zero_notMem_tilde Q) 𝔭
  simp only [round, Finset.mem_biUnion]
  constructor
  · rintro ⟨Q, hQW, hQ𝔭⟩
    obtain ⟨H, hH, hH𝔭⟩ := (hQ Q).mp hQ𝔭
    exact ⟨H, ⟨Q, hQW, hH⟩, hH𝔭⟩
  · rintro ⟨H, ⟨Q, hQW, hH⟩, hH𝔭⟩
    exact ⟨Q, hQW, (hQ Q).mpr ⟨H, hH, hH𝔭⟩⟩

variable [Finite σ]

theorem exists_subset_prime_refine {P : Finset (MvPolynomial σ K)}
    (h0 : (0 : MvPolynomial σ K) ∉ P) (𝔭 : Ideal (MvPolynomial σ K)) [𝔭.IsPrime] :
    (↑P : Set (MvPolynomial σ K)) ⊆ 𝔭 ↔
      ∃ H ∈ ρ.refine P, (↑H : Set (MvPolynomial σ K)) ⊆ 𝔭 := by
  rw [ρ.F.subset_prime_iff h0 𝔭]
  change (∃ H ∈ ρ.round^[0] (ρ.initial P), _) ↔ _
  rw [refine]
  generalize Nat.find (ρ.refine_terminates P) = k
  induction k with
  | zero => rfl
  | succ k ih => rw [ih, Function.iterate_succ_apply', exists_subset_prime_round]

omit [Finite σ] in
/-- Every round of Algorithm `alg:Refine` preserves `⋃_{Q ∈ W} V(Q)`. -/
theorem iUnion_V_round (L : Type*) [CommRing L] [IsDomain L] [Algebra K L]
    (W : Finset (Finset (MvPolynomial σ K))) :
    ⋃ Q ∈ W, V L (↑Q : Set (MvPolynomial σ K)) = ⋃ H ∈ ρ.round W, V L (↑H : Set (MvPolynomial σ K)) := by
  ext a
  simp only [Set.mem_iUnion, exists_prop, mem_V_iff_subset_ker]
  exact exists_subset_prime_round W _

/-- **Lemma `lem:recursive-refinement` (Refinement preserves solutions)**, variety form:
`V(P) = ⋃_{H ∈ 𝓛(P)} V(H)`. -/
theorem V_eq_iUnion_refine (L : Type*) [CommRing L] [IsDomain L] [Algebra K L]
    {P : Finset (MvPolynomial σ K)} (h0 : (0 : MvPolynomial σ K) ∉ P) :
    V L (↑P : Set (MvPolynomial σ K)) = ⋃ H ∈ ρ.refine P, V L (↑H : Set (MvPolynomial σ K)) := by
  ext a
  simp only [Set.mem_iUnion, exists_prop, mem_V_iff_subset_ker]
  exact exists_subset_prime_refine h0 _

/-- **Lemma `lem:recursive-refinement`**, ideal form:
`√⟨P⟩ = ⋂_{H ∈ 𝓛(P)} √⟨H⟩`. The paper derives it from the variety form by the Nullstellensatz;
here it follows directly from the prime-ideal form. -/
theorem radical_eq_iInf_refine {P : Finset (MvPolynomial σ K)}
    (h0 : (0 : MvPolynomial σ K) ∉ P) :
    (Ideal.span (↑P : Set (MvPolynomial σ K))).radical =
      ⨅ H ∈ ρ.refine P, (Ideal.span (↑H : Set (MvPolynomial σ K))).radical := by
  ext x
  simp only [Submodule.mem_iInf, Factorizer.mem_radical_iff_forall_prime, Ideal.span_le]
  constructor
  · intro h H hH 𝔭 hp hH𝔭
    exact h 𝔭 hp ((exists_subset_prime_refine h0 𝔭).mpr ⟨H, hH, hH𝔭⟩)
  · intro h 𝔭 hp hP
    obtain ⟨H, hH, hH𝔭⟩ := (exists_subset_prime_refine h0 𝔭).mp hP
    exact h H hH 𝔭 hp hH𝔭

end Preservation

end Refiner

/-! ### Algorithm `alg:Branching` and Theorem `thm:branching-correctness` -/

/-- What an exact branch solver `𝓑` returns on `H`: an exact representation of `V(H)`, or a
certificate that `V(H) = ∅`. A representation is modelled by the set it represents. -/
inductive BranchOutcome (L : Type*) [CommRing L] [Algebra K L] (H : Finset (MvPolynomial σ K))
  | rep (s : Set (σ → L)) (hs : s = V L (↑H : Set (MvPolynomial σ K)))
  | unsat (hV : V L (↑H : Set (MvPolynomial σ K)) = ∅)

/-- The set a solver outcome contributes to `Sol(S)`. -/
def BranchOutcome.sol {L : Type*} [CommRing L] [Algebra K L] {H : Finset (MvPolynomial σ K)} :
    BranchOutcome L H → Set (σ → L)
  | .rep s _ => s
  | .unsat _ => ∅

theorem BranchOutcome.sol_eq {L : Type*} [CommRing L] [Algebra K L] {H : Finset (MvPolynomial σ K)}
    (o : BranchOutcome L H) : o.sol = V L (↑H : Set (MvPolynomial σ K)) := by
  cases o with
  | rep s hs => exact hs
  | unsat hV => exact hV.symm

namespace Refiner

variable {m : MonomialOrder σ} (ρ : Refiner K m) [Finite σ]

/-- `Sol(S)` for the output `S` of Algorithm `alg:Branching`: the union of the represented sets
over the terminal branches. -/
def branchingSol (L : Type*) [CommRing L] [Algebra K L]
    (𝓑 : ∀ H : Finset (MvPolynomial σ K), BranchOutcome L H) (P : Finset (MvPolynomial σ K)) :
    Set (σ → L) :=
  ⋃ H ∈ ρ.refine P, (𝓑 H).sol

/-- **Theorem `thm:branching-correctness` (Factor branching).** If `𝓑` returns on every branch
in `𝓛(P)`, Algorithm `alg:Branching` makes finitely many solver calls (`𝓛(P)` is a `Finset`), and
its output is sound and complete for `P`: `Sol(S) = V(P)`. -/
theorem branching_correct (L : Type*) [CommRing L] [IsDomain L] [Algebra K L]
    (𝓑 : ∀ H : Finset (MvPolynomial σ K), BranchOutcome L H) {P : Finset (MvPolynomial σ K)}
    (h0 : (0 : MvPolynomial σ K) ∉ P) :
    ρ.branchingSol L 𝓑 P = V L (↑P : Set (MvPolynomial σ K)) := by
  simp only [branchingSol, BranchOutcome.sol_eq]
  exact (ρ.V_eq_iUnion_refine L h0).symm

omit [Finite σ] in
/-- Subsection `subsec:pruning`: dropping branches whose solution sets lie in those of kept
branches keeps `⋃_{H ∈ W} V(H)`, so the results above also hold after `Prune`. -/
theorem iUnion_V_prune (L : Type*) [CommRing L] [Algebra K L]
    {W W' : Finset (Finset (MvPolynomial σ K))} (hsub : W' ⊆ W)
    (hcov : ∀ H ∈ W, ∃ H' ∈ W', V L (↑H : Set (MvPolynomial σ K)) ⊆ V L (↑H' : Set (MvPolynomial σ K))) :
    ⋃ H ∈ W', V L (↑H : Set (MvPolynomial σ K)) = ⋃ H ∈ W, V L (↑H : Set (MvPolynomial σ K)) := by
  apply le_antisymm
  · exact Set.biUnion_subset_biUnion_left fun H hH => hsub hH
  · refine Set.iUnion₂_subset fun H hH => ?_
    obtain ⟨H', hH', hV⟩ := hcov H hH
    exact hV.trans (Set.subset_iUnion₂_of_subset H' hH' subset_rfl)

end Refiner

end FactorBranching
