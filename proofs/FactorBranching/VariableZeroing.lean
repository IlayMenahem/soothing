import FactorBranching.Vars

/-!
# Recursive splitting by variable zeroing

Section "Recursive splitting": Theorem `lem:zeroing-split`, Lemma `lem:zero-nonzero-split`,
Equation `eq:zeroing-refinement`, Theorem `thm:zeroing-correctness`,
Corollary `cor:zeroing-completeness`, the branch contraction of
Subsection `subsec:zeroing-branch-contraction`, and the two nonzero encodings of
Subsection `subsec:nonzero-encoding`.

All branches live in one ring `K[x_y, s_y : y ∈ Y]` (`Sum.inl y` is `x_y`, `Sum.inr y` the
auxiliary variable `s_y`). A pending branch `(Q, Z, N)` has its auxiliary variables among
`{s_y : y ∈ N}`; the `x_z` with `z ∈ Z` have been substituted by `0`. The paper restricts points
to `Y ∖ Z`; here a point is a full `a : Y → L` with `a_z = 0` for `z ∈ Z`, which is equivalent.
The UNSAT certificate and the timeouts and budget of Algorithm `alg:ZeroingBranching` are an
arbitrary per-branch outcome, so correctness is proved for every possible run.
-/

open MvPolynomial

noncomputable section

open scoped Classical

namespace FactorBranching

/-- **Theorem `lem:zeroing-split` (Correctness of an exhaustive split).** Let `D_i` be side
conditions such that every point satisfying `A` satisfies some `D_i`. If each child set is
`{a ∈ V_Y(Q; A) : a satisfies D_i}`, then `V_Y(Q; A)` is the union of the child sets. -/
theorem exhaustive_split {α ι : Type*} {VQ : Set α} {A : α → Prop} {D : ι → α → Prop}
    {Vi : ι → Set α} (hVQ : ∀ a ∈ VQ, A a) (hcover : ∀ a, A a → ∃ i, D i a)
    (hVi : ∀ i, Vi i = {a | a ∈ VQ ∧ D i a}) : VQ = ⋃ i, Vi i := by
  ext a
  simp only [Set.mem_iUnion, hVi, Set.mem_ofPred_eq]
  constructor
  · intro ha
    obtain ⟨i, hi⟩ := hcover a (hVQ a ha)
    exact ⟨i, ha, hi⟩
  · rintro ⟨i, ha, -⟩
    exact ha

variable {K : Type*} [Field K] {Y : Type*}

/-- The ring `K[x_y, s_y : y ∈ Y]`. -/
abbrev ZPoly (K : Type*) [Field K] (Y : Type*) := MvPolynomial (Y ⊕ Y) K

/-- `σ_x = 1 - s_x x`, the Rabinowitsch generator for `x ≠ 0`. -/
def sigmaPoly (x : Y) : ZPoly K Y :=
  1 - X (Sum.inr x) * X (Sum.inl x)

/-- The substitution `x ↦ 0`. -/
def zeroSubst (x : Y) : Y ⊕ Y → ZPoly K Y :=
  fun v => if v = Sum.inl x then 0 else X v

/-- `H|_{x=0}`, with the zero equations dropped. -/
def restrictZero (H : Finset (ZPoly K Y)) (x : Y) : Finset (ZPoly K Y) :=
  (H.image (bind₁ (zeroSubst x))).erase 0

/-- `UpdateSaturation(H, N ∪ {x})` in the separate encoding: add `σ_x`. -/
def updateSat (H : Finset (ZPoly K Y)) (x : Y) : Finset (ZPoly K Y) :=
  insert (sigmaPoly x) H

variable (L : Type*) [Field L] [Algebra K L]

/-- `V_Y(Q; Z, N)`: points with `a_z = 0` (`z ∈ Z`), `a_y ≠ 0` (`y ∈ N`), that extend to a
solution of `Q` for some values `s` of the auxiliary variables. -/
def VY (Q : Finset (ZPoly K Y)) (Z N : Finset Y) : Set (Y → L) :=
  {a | (∀ z ∈ Z, a z = 0) ∧ (∀ y ∈ N, a y ≠ 0) ∧
    ∃ s : Y → L, Sum.elim a s ∈ V L (↑Q : Set (ZPoly K Y))}

/-- At the root, `V_Y(P; ∅, ∅) = V(P)`. -/
theorem VY_root (P : Finset (MvPolynomial Y K)) :
    VY L (P.image (rename Sum.inl)) ∅ ∅ = V L (↑P : Set (MvPolynomial Y K)) := by
  ext a
  simp only [VY, Finset.notMem_empty, IsEmpty.forall_iff, implies_true, true_and,
    mem_V, Finset.coe_image, Set.mem_image, Finset.mem_coe, forall_exists_index, and_imp,
    forall_apply_eq_imp_iff₂, aeval_rename, Sum.elim_comp_inl, Set.mem_ofPred_eq]
  exact ⟨fun ⟨_, h⟩ => h, fun h => ⟨fun _ => 0, h⟩⟩

variable {L}

theorem aeval_eq_of_vars {σ : Type*} {f g : σ → L} {q : MvPolynomial σ K}
    (h : ∀ v ∈ q.vars, f v = g v) : aeval f q = aeval g q :=
  hom_congr_vars (f₁ := (aeval f : MvPolynomial σ K →ₐ[K] L).toRingHom)
    (f₂ := (aeval g : MvPolynomial σ K →ₐ[K] L).toRingHom)
    (by ext; simp) (fun i hi _ => by simpa using h i hi) rfl

theorem aeval_zeroSubst {x : Y} {b : Y ⊕ Y → L} (hb : b (Sum.inl x) = 0) (q : ZPoly K Y) :
    aeval b (bind₁ (zeroSubst x) q) = aeval b q := by
  have hfun : (fun v => aeval b (zeroSubst (K := K) x v)) = b := by
    funext v
    by_cases hv : v = Sum.inl x
    · subst hv; simp [zeroSubst, hb]
    · simp [zeroSubst, hv]
  rw [aeval_bind₁, hfun]

theorem aeval_sigmaPoly (b : Y ⊕ Y → L) (x : Y) :
    aeval b (sigmaPoly (K := K) x) = 1 - b (Sum.inr x) * b (Sum.inl x) := by
  simp [sigmaPoly]

/-- `V_Y(H|_{x=0}; Z ∪ {x}, N) = {a ∈ V_Y(H; Z, N) : a_x = 0}`. -/
theorem VY_restrictZero (H : Finset (ZPoly K Y)) (Z N : Finset Y) (x : Y) :
    VY L (restrictZero H x) (insert x Z) N = {a | a ∈ VY L H Z N ∧ a x = 0} := by
  have key : ∀ b : Y ⊕ Y → L, b (Sum.inl x) = 0 →
      (b ∈ V L (↑(restrictZero H x) : Set (ZPoly K Y)) ↔ b ∈ V L (↑H : Set (ZPoly K Y))) := by
    intro b hb
    simp only [mem_V, restrictZero, Finset.coe_erase, Finset.coe_image, Set.mem_sdiff,
      Set.mem_image, Finset.mem_coe, Set.mem_singleton_iff]
    constructor
    · intro h q hq
      by_cases h0 : bind₁ (zeroSubst x) q = 0
      · rw [← aeval_zeroSubst hb, h0, map_zero]
      · rw [← aeval_zeroSubst hb]
        exact h _ ⟨⟨q, hq, rfl⟩, h0⟩
    · rintro h _ ⟨⟨q, hq, rfl⟩, -⟩
      rw [aeval_zeroSubst hb]
      exact h q hq
  ext a
  simp only [VY, Finset.mem_insert, forall_eq_or_imp, Set.mem_ofPred_eq]
  constructor
  · rintro ⟨⟨hx, hZ⟩, hN, s, hs⟩
    exact ⟨⟨hZ, hN, s, (key _ (by simpa using hx)).mp hs⟩, hx⟩
  · rintro ⟨⟨hZ, hN, s, hs⟩, hx⟩
    exact ⟨⟨hx, hZ⟩, hN, s, (key _ (by simpa using hx)).mpr hs⟩

/-- `V_Y(H ∪ {σ_x}; Z, N ∪ {x}) = {a ∈ V_Y(H; Z, N) : a_x ≠ 0}`, provided `s_x` does not occur
in `H` (as it does not when `x ∉ N`). -/
theorem VY_updateSat (H : Finset (ZPoly K Y)) (Z N : Finset Y) (x : Y)
    (hfree : ∀ q ∈ H, Sum.inr x ∉ q.vars) :
    VY L (updateSat H x) Z (insert x N) = {a | a ∈ VY L H Z N ∧ a x ≠ 0} := by
  ext a
  simp only [VY, updateSat, Finset.mem_insert, forall_eq_or_imp, Set.mem_ofPred_eq, mem_V,
    Finset.coe_insert, Set.mem_insert_iff, Finset.mem_coe]
  constructor
  · rintro ⟨hZ, ⟨hx, hN⟩, s, -, hs⟩
    exact ⟨⟨hZ, hN, s, hs⟩, hx⟩
  · rintro ⟨⟨hZ, hN, s, hs⟩, hx⟩
    refine ⟨hZ, ⟨hx, hN⟩, Function.update s x (a x)⁻¹, ?_, fun q hq => ?_⟩
    · rw [aeval_sigmaPoly]
      simp [inv_mul_cancel₀ hx]
    · rw [← hs q hq]
      refine aeval_eq_of_vars fun v hv => ?_
      rcases v with y | y
      · rfl
      · have hyx : y ≠ x := fun h => hfree q hq (h ▸ hv)
        simp [Function.update_of_ne hyx]

/-- **Lemma `lem:zero-nonzero-split`.** Let `H` be a branch in which `s_x` does not occur. Then
`V_Y(H; Z, N) = V_Y(H|_{x=0}; Z ∪ {x}, N) ∪ V_Y(H ∪ {σ_x}; Z, N ∪ {x})`. -/
theorem zero_nonzero_split (H : Finset (ZPoly K Y)) (Z N : Finset Y) (x : Y)
    (hfree : ∀ q ∈ H, Sum.inr x ∉ q.vars) :
    VY L H Z N = VY L (restrictZero H x) (insert x Z) N ∪ VY L (updateSat H x) Z (insert x N) := by
  rw [Set.union_def, exhaustive_split (VQ := VY L H Z N) (A := fun _ => True)
    (D := fun (i : Bool) (a : Y → L) => if i then a x = 0 else a x ≠ 0)
    (Vi := fun i => if i then VY L (restrictZero H x) (insert x Z) N
      else VY L (updateSat H x) Z (insert x N))
    (fun _ _ => trivial) (fun a _ => by by_cases h : a x = 0; exacts [⟨true, h⟩, ⟨false, h⟩])
    (fun i => by cases i <;> simp [VY_restrictZero, VY_updateSat _ _ _ _ hfree])]
  ext a
  simp only [Set.mem_iUnion, Bool.exists_bool, Set.mem_ofPred_eq]
  tauto

/-! ### Refinement inside a pending branch -/

/-- The auxiliary variables of a pending branch with nonzero set `N` are among `{s_y : y ∈ N}`. -/
def auxIn (N : Finset Y) : Set (Y ⊕ Y) :=
  {v | ∀ y, v = Sum.inr y → y ∈ N}

variable [Finite Y] {m : MonomialOrder (Y ⊕ Y)} (ρ : Refiner K m)

/-- **Equation `eq:zeroing-refinement`.** `V_Y(Q; Z, N) = ⋃_{H ∈ 𝓛(Q)} V_Y(H; Z, N)`. -/
theorem VY_eq_iUnion_refine {Q : Finset (ZPoly K Y)} (h0 : (0 : ZPoly K Y) ∉ Q) (Z N : Finset Y) :
    VY L Q Z N = ⋃ H ∈ ρ.refine Q, VY L H Z N := by
  ext a
  simp only [VY, Set.mem_iUnion, Set.mem_ofPred_eq, exists_prop, ρ.V_eq_iUnion_refine L h0]
  constructor
  · rintro ⟨hZ, hN, s, H, hH, hs⟩
    exact ⟨H, hH, hZ, hN, s, hs⟩
  · rintro ⟨H, hH, hZ, hN, s, hs⟩
    exact ⟨hZ, hN, s, H, hH, hs⟩

/-! ### Algorithm `alg:ZeroingBranching` -/

/-- A branch `(Q, Z, N)`. -/
abbrev ZBranch (K : Type*) [Field K] (Y : Type*) := Finset (ZPoly K Y) × Finset Y × Finset Y

variable (L) in
/-- The fate of a terminal branch `H` of a pending branch with nonzero set `N`: the solver returns
a representation, certifies `V(H) = ∅`, or fails, after which `H` is left unresolved or split on
a variable `x ∉ N` (the useful variables exclude `Z ∪ N`; `x ∉ N` is what the proof needs). -/
inductive Outcome (H : Finset (ZPoly K Y)) (N : Finset Y)
  | solved
  | unsat (h : V L (↑H : Set (ZPoly K Y)) = ∅)
  | unresolved
  | split (x : Y) (hx : x ∉ N)

/-- What a terminal branch adds to `S`, `U` and `W`. -/
def Outcome.toS {H : Finset (ZPoly K Y)} {N : Finset Y} (Z : Finset Y) :
    Outcome L H N → Finset (ZBranch K Y)
  | .solved => {(H, Z, N)}
  | _ => ∅

def Outcome.toU {H : Finset (ZPoly K Y)} {N : Finset Y} (Z : Finset Y) :
    Outcome L H N → Finset (ZBranch K Y)
  | .unresolved => {(H, Z, N)}
  | _ => ∅

def Outcome.toW {H : Finset (ZPoly K Y)} {N : Finset Y} (Z : Finset Y) :
    Outcome L H N → Finset (ZBranch K Y)
  | .split x _ => {(restrictZero H x, insert x Z, N), (updateSat H x, Z, insert x N)}
  | _ => ∅

/-- The state `(S, U, W)` of Algorithm `alg:ZeroingBranching`. A representation in `S` stored
with `(Z, N)` for the branch `H` represents `V_Y(H; Z, N)`. -/
structure ZState (K : Type*) [Field K] (Y : Type*) where
  S : Finset (ZBranch K Y)
  U : Finset (ZBranch K Y)
  W : Finset (ZBranch K Y)

variable (L) in
/-- `Sol` of a set of stored branches. -/
def solOf (T : Finset (ZBranch K Y)) : Set (Y → L) :=
  ⋃ b ∈ T, VY L b.1 b.2.1 b.2.2

variable (L) in
/-- One iteration of the while loop: remove `b = (Q, Z, N)` from `W` and process every terminal
branch of `Q` according to its outcome `o`. -/
def ZStep (st st' : ZState K Y) : Prop :=
  ∃ b ∈ st.W, ∃ o : ∀ H : Finset (ZPoly K Y), Outcome L H b.2.2,
    st' = ⟨st.S ∪ (ρ.refine b.1).biUnion (fun H => (o H).toS b.2.1),
      st.U ∪ (ρ.refine b.1).biUnion (fun H => (o H).toU b.2.1),
      st.W.erase b ∪ (ρ.refine b.1).biUnion (fun H => (o H).toW b.2.1)⟩

/-- The initial state `S = U = ∅`, `W = {(P, ∅, ∅)}`. -/
def ZState.init (P : Finset (MvPolynomial Y K)) : ZState K Y :=
  ⟨∅, ∅, {(P.image (rename Sum.inl), ∅, ∅)}⟩

/-- Pending branches have no zero generator and no auxiliary variable outside `N`. -/
def GoodBranch (b : ZBranch K Y) : Prop :=
  (0 : ZPoly K Y) ∉ b.1 ∧ VarsIn b.1 (auxIn b.2.2)

section SolLemmas

variable (L)

omit [Finite Y] in
theorem solOf_union (T T' : Finset (ZBranch K Y)) : solOf L (T ∪ T') = solOf L T ∪ solOf L T' := by
  ext a
  simp only [solOf, Set.mem_union, Set.mem_iUnion, Finset.mem_union, exists_prop, or_and_right,
    exists_or]

omit [Finite Y] in
theorem solOf_biUnion {ι : Type*} (s : Finset ι) (f : ι → Finset (ZBranch K Y)) :
    solOf L (s.biUnion f) = ⋃ i ∈ s, solOf L (f i) := by
  ext a
  simp only [solOf, Finset.mem_biUnion, Set.mem_iUnion, exists_prop]
  tauto

omit [Finite Y] in
theorem solOf_erase_union {T : Finset (ZBranch K Y)} {b : ZBranch K Y} (hb : b ∈ T) :
    solOf L (T.erase b) ∪ VY L b.1 b.2.1 b.2.2 = solOf L T := by
  ext a
  simp only [solOf, Set.mem_union, Set.mem_iUnion, Finset.mem_erase, exists_prop]
  constructor
  · rintro (⟨b', ⟨-, hb'⟩, h⟩ | h)
    · exact ⟨b', hb', h⟩
    · exact ⟨b, hb, h⟩
  · rintro ⟨b', hb', h⟩
    by_cases hbb : b' = b
    · subst hbb
      exact Or.inr h
    · exact Or.inl ⟨b', ⟨hbb, hb'⟩, h⟩

end SolLemmas

omit [Finite Y] in
/-- What one terminal branch adds to `S`, `U` and `W` represents exactly `V_Y(H; Z, N)`. -/
theorem outcome_sol_eq {H : Finset (ZPoly K Y)} (Z N : Finset Y) (hH : VarsIn H (auxIn N))
    (o : Outcome L H N) :
    solOf L (o.toS Z) ∪ solOf L (o.toU Z) ∪ solOf L (o.toW Z) = VY L H Z N := by
  cases o with
  | solved => simp [Outcome.toS, Outcome.toU, Outcome.toW, solOf]
  | unsat h =>
    have hVY : VY L H Z N = ∅ := by
      ext a
      simp only [Set.mem_empty_iff_false, iff_false]
      rintro ⟨-, -, s, hs⟩
      rw [h] at hs
      exact hs
    simp [Outcome.toS, Outcome.toU, Outcome.toW, solOf, hVY]
  | unresolved => simp [Outcome.toS, Outcome.toU, Outcome.toW, solOf]
  | split x hx =>
    have hfree : ∀ q ∈ H, Sum.inr x ∉ q.vars := fun q hq hv => hx (hH q hq _ hv x rfl)
    simp only [Outcome.toS, Outcome.toU, Outcome.toW, solOf, Finset.notMem_empty,
      Set.iUnion_of_empty, Set.iUnion_empty, Set.empty_union, Finset.mem_insert,
      Finset.mem_singleton, Set.iUnion_iUnion_eq_or_left, Set.iUnion_iUnion_eq_left]
    exact (zero_nonzero_split H Z N x hfree).symm

omit [Finite Y] in
theorem good_toW {H : Finset (ZPoly K Y)} {Z N : Finset Y} (hH : VarsIn H (auxIn N))
    (hH0 : (0 : ZPoly K Y) ∉ H) (o : Outcome L H N) : ∀ b ∈ o.toW Z, GoodBranch b := by
  cases o with
  | solved | unsat | unresolved => simp [Outcome.toW]
  | split x hx =>
    simp only [Outcome.toW, Finset.mem_insert, Finset.mem_singleton]
    rintro b (rfl | rfl)
    · refine ⟨Finset.notMem_erase 0 _, fun q hq v hv => ?_⟩
      obtain ⟨p, hp, rfl⟩ := Finset.mem_image.mp (Finset.mem_of_mem_erase hq)
      obtain ⟨i, hi, hvi⟩ := Finset.mem_biUnion.mp (vars_bind₁ _ p hv)
      have : v = i := by
        by_cases h : i = Sum.inl x
        · simp [zeroSubst, h] at hvi
        · simpa [zeroSubst, h] using hvi
      exact this ▸ hH p hp i hi
    · refine ⟨?_, fun q hq v hv => ?_⟩
      · rw [updateSat, Finset.mem_insert, not_or]
        refine ⟨fun h => ?_, hH0⟩
        have := congrArg constantCoeff h
        simp [sigmaPoly] at this
      · intro y hy
        rcases Finset.mem_insert.mp hq with rfl | hq
        · have hv' := vars_sub_subset (p := (1 : ZPoly K Y)) (q := X (Sum.inr x) * X (Sum.inl x)) hv
          simp only [vars_one, Finset.empty_union] at hv'
          rcases Finset.mem_union.mp (vars_mul _ _ hv') with h | h <;> rw [vars_X] at h <;>
            simp only [Finset.mem_singleton] at h <;> subst h
          · cases hy; exact Finset.mem_insert_self _ _
          · cases hy
        · exact Finset.mem_insert_of_mem (hH q hq v hv y hy)

/-- The invariant of the proof of Theorem `thm:zeroing-correctness`. -/
def ZInv (P : Finset (MvPolynomial Y K)) (st : ZState K Y) : Prop :=
  V L (↑P : Set (MvPolynomial Y K)) = solOf L st.S ∪ solOf L st.U ∪ solOf L st.W ∧
    ∀ b ∈ st.W, GoodBranch b

omit [Finite Y] in
theorem zInv_init {P : Finset (MvPolynomial Y K)} (h0 : (0 : MvPolynomial Y K) ∉ P) :
    ZInv (L := L) P (ZState.init P) := by
  refine ⟨?_, ?_⟩
  · simp [ZState.init, solOf, VY_root]
  · simp only [ZState.init, Finset.mem_singleton, forall_eq]
    refine ⟨fun h => ?_, fun q hq v hv y hy => ?_⟩
    · obtain ⟨p, hp, hp0⟩ := Finset.mem_image.mp h
      rw [← map_zero (rename (R := K) (Sum.inl : Y → Y ⊕ Y))] at hp0
      exact h0 (rename_injective _ Sum.inl_injective hp0 ▸ hp)
    · obtain ⟨p, -, rfl⟩ := Finset.mem_image.mp hq
      obtain ⟨i, -, rfl⟩ := Finset.mem_image.mp (vars_rename _ p hv)
      cases hy

theorem ZStep.zInv {P : Finset (MvPolynomial Y K)} {st st' : ZState K Y}
    (hs : ZStep L ρ st st') (hinv : ZInv (L := L) P st) : ZInv (L := L) P st' := by
  obtain ⟨b, hb, o, rfl⟩ := hs
  obtain ⟨hV, hgood⟩ := hinv
  obtain ⟨hb0, hbvars⟩ := hgood b hb
  have hterm : ∀ H ∈ ρ.refine b.1, VarsIn H (auxIn b.2.2) ∧ (0 : ZPoly K Y) ∉ H :=
    fun H hH => ρ.varsIn_of_mem_refine hbvars hb0 hH
  refine ⟨?_, ?_⟩
  · rw [hV]
    simp only [solOf_union, solOf_biUnion]
    rw [← solOf_erase_union L hb, VY_eq_iUnion_refine ρ hb0]
    have hsplit : ⋃ H ∈ ρ.refine b.1, VY L H b.2.1 b.2.2 =
        (⋃ H ∈ ρ.refine b.1, solOf L ((o H).toS b.2.1)) ∪
          (⋃ H ∈ ρ.refine b.1, solOf L ((o H).toU b.2.1)) ∪
          (⋃ H ∈ ρ.refine b.1, solOf L ((o H).toW b.2.1)) := by
      ext a
      simp only [Set.mem_union, Set.mem_iUnion, exists_prop]
      constructor
      · rintro ⟨H, hH, ha⟩
        rw [← outcome_sol_eq _ _ (hterm H hH).1 (o H)] at ha
        rcases ha with (ha | ha) | ha
        · exact Or.inl (Or.inl ⟨H, hH, ha⟩)
        · exact Or.inl (Or.inr ⟨H, hH, ha⟩)
        · exact Or.inr ⟨H, hH, ha⟩
      · rintro ((⟨H, hH, ha⟩ | ⟨H, hH, ha⟩) | ⟨H, hH, ha⟩) <;> refine ⟨H, hH, ?_⟩ <;>
          rw [← outcome_sol_eq _ _ (hterm H hH).1 (o H)] <;> simp [ha]
    rw [hsplit]
    ext a
    simp only [Set.mem_union]
    tauto
  · intro b' hb'
    rcases Finset.mem_union.mp hb' with hb' | hb'
    · exact hgood b' (Finset.mem_of_mem_erase hb')
    · obtain ⟨H, hH, hb'H⟩ := Finset.mem_biUnion.mp hb'
      exact good_toW (hterm H hH).1 (hterm H hH).2 (o H) b' hb'H

/-- **Theorem `thm:zeroing-correctness` (Correctness of recursive splitting).** If
Algorithm `alg:ZeroingBranching` returns `(S, U)` on input `P`, i.e. a run reaches `W = ∅`, then
`V(P) = Sol(S) ∪ ⋃_{(H, Z, N) ∈ U} V_Y(H; Z, N)`. This holds for every choice of solver results,
timeouts, budget and splitting variables. -/
theorem zeroing_correct {P : Finset (MvPolynomial Y K)} (h0 : (0 : MvPolynomial Y K) ∉ P)
    {st : ZState K Y} (hrun : Relation.ReflTransGen (ZStep L ρ) (ZState.init P) st)
    (hW : st.W = ∅) :
    V L (↑P : Set (MvPolynomial Y K)) = solOf L st.S ∪ solOf L st.U := by
  have hinv : ZInv (L := L) P st := by
    clear hW
    induction hrun with
    | refl => exact zInv_init h0
    | tail _ hs ih => exact hs.zInv ρ ih
  rw [hinv.1, hW]
  simp [solOf]

/-- **Corollary `cor:zeroing-completeness` (Completeness of variable zeroing).** If `U = ∅` in
the output, then `Sol(S) = V(P)`. -/
theorem zeroing_complete {P : Finset (MvPolynomial Y K)} (h0 : (0 : MvPolynomial Y K) ∉ P)
    {st : ZState K Y} (hrun : Relation.ReflTransGen (ZStep L ρ) (ZState.init P) st)
    (hW : st.W = ∅) (hU : st.U = ∅) :
    solOf L st.S = V L (↑P : Set (MvPolynomial Y K)) := by
  rw [zeroing_correct ρ h0 hrun hW, hU]
  simp [solOf]

/-! ### Branch contraction (Subsection `subsec:zeroing-branch-contraction`) -/

/-- If the `x = 0` child of a split of `H` has no terminal branches and the `x ≠ 0` child has
exactly one, `H'`, then `V_Y(H; Z, N) = V_Y(H'; Z, N ∪ {x})`. -/
theorem contraction_zero_unsat {H H' : Finset (ZPoly K Y)} {Z N : Finset Y} {x : Y}
    (hfree : ∀ q ∈ H, Sum.inr x ∉ q.vars) (hH0 : (0 : ZPoly K Y) ∉ H)
    (hzero : ρ.refine (restrictZero H x) = ∅) (hone : ρ.refine (updateSat H x) = {H'}) :
    VY L H Z N = VY L H' Z (insert x N) := by
  have h0z : (0 : ZPoly K Y) ∉ restrictZero H x := Finset.notMem_erase 0 _
  have h0s : (0 : ZPoly K Y) ∉ updateSat H x := by
    rw [updateSat, Finset.mem_insert, not_or]
    refine ⟨fun h => ?_, hH0⟩
    have := congrArg constantCoeff h
    simp [sigmaPoly] at this
  rw [zero_nonzero_split H Z N x hfree, VY_eq_iUnion_refine ρ h0z, hzero,
    VY_eq_iUnion_refine ρ h0s, hone]
  simp

/-- If the `x ≠ 0` child has no terminal branches and the `x = 0` child has exactly one, `H'`,
then `V_Y(H; Z, N) = V_Y(H'; Z ∪ {x}, N)`. -/
theorem contraction_nonzero_unsat {H H' : Finset (ZPoly K Y)} {Z N : Finset Y} {x : Y}
    (hfree : ∀ q ∈ H, Sum.inr x ∉ q.vars) (hH0 : (0 : ZPoly K Y) ∉ H)
    (hzero : ρ.refine (restrictZero H x) = {H'}) (hone : ρ.refine (updateSat H x) = ∅) :
    VY L H Z N = VY L H' (insert x Z) N := by
  have h0z : (0 : ZPoly K Y) ∉ restrictZero H x := Finset.notMem_erase 0 _
  have h0s : (0 : ZPoly K Y) ∉ updateSat H x := by
    rw [updateSat, Finset.mem_insert, not_or]
    refine ⟨fun h => ?_, hH0⟩
    have := congrArg constantCoeff h
    simp [sigmaPoly] at this
  rw [zero_nonzero_split H Z N x hfree, VY_eq_iUnion_refine ρ h0z, hzero,
    VY_eq_iUnion_refine ρ h0s, hone]
  simp

/-! ### Nonzero encodings (Subsection `subsec:nonzero-encoding`) -/

section Encodings

variable {σ : Type*} (L)

/-- The separate encoding: `π(V(Q ∪ {σ_x : x ∈ N})) = {a ∈ V(Q) : a_x ≠ 0, x ∈ N}`, for `Q` in
the original variables and one auxiliary `s_x` per `x`. -/
theorem separate_encoding (Q : Finset (MvPolynomial σ K)) (N : Finset σ) :
    {a : σ → L | ∃ s : σ → L, Sum.elim a s ∈ V L
        ((↑(Q.image (rename (Sum.inl : σ → σ ⊕ σ))) : Set (MvPolynomial (σ ⊕ σ) K)) ∪
          (fun x => 1 - X (Sum.inr x) * X (Sum.inl x)) '' ↑N)} =
      {a | a ∈ V L (↑Q : Set (MvPolynomial σ K)) ∧ ∀ x ∈ N, a x ≠ 0} := by
  ext a
  simp only [Set.mem_ofPred_eq, mem_V, Set.mem_union, Finset.coe_image, Set.mem_image,
    Finset.mem_coe]
  constructor
  · rintro ⟨s, hs⟩
    refine ⟨fun q hq => ?_, fun x hx hax => ?_⟩
    · simpa [aeval_rename] using hs _ (Or.inl ⟨q, hq, rfl⟩)
    · have := hs _ (Or.inr ⟨x, hx, rfl⟩)
      simp [hax] at this
  · rintro ⟨hQ, hN⟩
    refine ⟨fun x => (a x)⁻¹, ?_⟩
    rintro _ (⟨q, hq, rfl⟩ | ⟨x, hx, rfl⟩)
    · simpa [aeval_rename] using hQ q hq
    · simp [inv_mul_cancel₀ (hN x hx)]

/-- The product encoding: `π(V(Q ∪ {σ_N})) = {a ∈ V(Q) : a_x ≠ 0, x ∈ N}` with
`σ_N = 1 - s ∏_{x ∈ N} x` and a single auxiliary variable `s`. -/
theorem product_encoding (Q : Finset (MvPolynomial σ K)) (N : Finset σ) :
    {a : σ → L | ∃ s : L, (fun v : Option σ => v.elim s a) ∈ V L
        ((↑(Q.image (rename (some : σ → Option σ))) : Set (MvPolynomial (Option σ) K)) ∪
          {1 - X none * ∏ x ∈ N, X (some x)})} =
      {a | a ∈ V L (↑Q : Set (MvPolynomial σ K)) ∧ ∀ x ∈ N, a x ≠ 0} := by
  ext a
  simp only [Set.mem_ofPred_eq, mem_V, Set.mem_union, Finset.coe_image, Set.mem_image,
    Finset.mem_coe, Set.mem_singleton_iff]
  have hcomp : ∀ s : L, (fun v : Option σ => v.elim s a) ∘ some = a := fun _ => rfl
  constructor
  · rintro ⟨s, hs⟩
    refine ⟨fun q hq => ?_, fun x hx hax => ?_⟩
    · simpa [aeval_rename, hcomp] using hs _ (Or.inl ⟨q, hq, rfl⟩)
    · have := hs _ (Or.inr rfl)
      simp [Finset.prod_eq_zero hx hax] at this
  · rintro ⟨hQ, hN⟩
    refine ⟨(∏ x ∈ N, a x)⁻¹, ?_⟩
    rintro _ (⟨q, hq, rfl⟩ | rfl)
    · simpa [aeval_rename, hcomp] using hQ q hq
    · simp [inv_mul_cancel₀ (Finset.prod_ne_zero_iff.mpr hN)]

end Encodings

end FactorBranching
