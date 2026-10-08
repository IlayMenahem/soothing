import FactorBranching.Vars

/-!
# Recursive splitting by variable zeroing

Section "Recursive splitting": the exhaustive split, Lemma `lem:zero-nonzero-split`,
Equation `eq:zeroing-refinement`, Theorem `thm:zeroing-correctness`,
Corollary `cor:zeroing-completeness`, Theorem `thm:zeroing-termination`, the branch contraction of
Subsection `subsec:zeroing-branch-contraction`, and the two nonzero encodings of
Subsection `subsec:nonzero-encoding`.

Branches live in the original ring `K[Y]`: the `x_z` with `z ∈ Z` have been substituted by `0`,
and the variables of `N` are side conditions, not equations, so no branch has an auxiliary
variable. The paper restricts points to `Y ∖ Z`; here a point is a full `a : Y → L` with
`a_z = 0` for `z ∈ Z`, which is equivalent.

`Saturate(H, N)` is modelled by its defining property: a generating set `G` with
`⟨H⟩ ⊆ ⟨G⟩` and `u_N^m g ∈ ⟨H⟩` for every `g ∈ G` (`IsSaturation`). A basis of
`⟨H⟩ : u_N^∞` has it, and so has the fallback `G = H` used when the saturation times out.
`Refine(Q, N)` is a specification (`CondRefiner`) satisfied by Algorithm `alg:Refine` and preserved
by dropping branches on which a nonzero condition vanishes. The branch solver's verdict, the
timeouts and the budget of Algorithm `alg:ZeroingBranching` are an arbitrary per-branch outcome,
so correctness is proved for every possible run.
-/

open MvPolynomial

noncomputable section

open scoped Classical

namespace FactorBranching

/-- **Correctness of an exhaustive split** (Subsection `subsec:zeroing-theoretical-analysis`).
Let `D_i` be side conditions such that every point satisfying `A` satisfies some `D_i`. If each
child set is `{a ∈ V_Y(Q; A) : a satisfies D_i}`, then `V_Y(Q; A)` is the union of the child
sets. -/
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

/-- `u_N = ∏_{y ∈ N} y`. -/
def uN (N : Finset Y) : MvPolynomial Y K :=
  ∏ y ∈ N, X y

/-- The substitution `x ↦ 0`. -/
def zeroSubst (x : Y) : Y → MvPolynomial Y K :=
  fun v => if v = x then 0 else X v

/-- `H|_{x=0}`, with the zero equations dropped. -/
def restrictZero (H : Finset (MvPolynomial Y K)) (x : Y) : Finset (MvPolynomial Y K) :=
  (H.image (bind₁ (zeroSubst x))).erase 0

theorem zero_notMem_restrictZero (H : Finset (MvPolynomial Y K)) (x : Y) :
    (0 : MvPolynomial Y K) ∉ restrictZero H x :=
  Finset.notMem_erase 0 _

/-- `G` is a valid result of `Saturate(H, N)`: `⟨H⟩ ⊆ ⟨G⟩ ⊆ ⟨H⟩ : u_N^∞`. -/
def IsSaturation (H G : Finset (MvPolynomial Y K)) (N : Finset Y) : Prop :=
  (∀ h ∈ H, h ∈ Ideal.span (↑G : Set (MvPolynomial Y K))) ∧
    ∀ g ∈ G, ∃ m : ℕ, uN N ^ m * g ∈ Ideal.span (↑H : Set (MvPolynomial Y K))

/-- The fallback of `Saturate(H, N)`, returning `H` itself, is valid. -/
theorem isSaturation_self (H : Finset (MvPolynomial Y K)) (N : Finset Y) :
    IsSaturation H H N :=
  ⟨fun _ hh => Ideal.subset_span hh, fun _ hg => ⟨0, by simpa using Ideal.subset_span hg⟩⟩

variable (L : Type*) [Field L] [Algebra K L]

/-- `V_Y(Q; Z, N)`: points with `a_z = 0` (`z ∈ Z`) and `a_y ≠ 0` (`y ∈ N`) that solve `Q`. -/
def VY (Q : Finset (MvPolynomial Y K)) (Z N : Finset Y) : Set (Y → L) :=
  {a | (∀ z ∈ Z, a z = 0) ∧ (∀ y ∈ N, a y ≠ 0) ∧ a ∈ V L (↑Q : Set (MvPolynomial Y K))}

/-- At the root, `V_Y(P; ∅, ∅) = V(P)`. -/
theorem VY_root (P : Finset (MvPolynomial Y K)) :
    VY L P ∅ ∅ = V L (↑P : Set (MvPolynomial Y K)) := by
  ext a
  simp [VY]

variable {L}

theorem aeval_mem_span_eq_zero {S : Set (MvPolynomial Y K)} {a : Y → L} (ha : a ∈ V L S)
    {p : MvPolynomial Y K} (hp : p ∈ Ideal.span S) : aeval a p = 0 := by
  rw [V_eq_zeroLocus] at ha
  exact ha p hp

theorem aeval_uN (a : Y → L) (N : Finset Y) : aeval a (uN (K := K) N) = ∏ y ∈ N, a y := by
  simp [uN]

/-- A valid saturation has the same points as `H` where every variable in `N` is nonzero:
`V(Saturate(H, N)) ∖ V(u_N) = V(H) ∖ V(u_N)`. -/
theorem mem_V_iff_of_isSaturation {H G : Finset (MvPolynomial Y K)} {N : Finset Y}
    (hG : IsSaturation H G N) {a : Y → L} (ha : ∀ y ∈ N, a y ≠ 0) :
    a ∈ V L (↑G : Set (MvPolynomial Y K)) ↔ a ∈ V L (↑H : Set (MvPolynomial Y K)) := by
  constructor
  · intro haG h hh
    exact aeval_mem_span_eq_zero haG (hG.1 h hh)
  · intro haH g hg
    obtain ⟨m, hm⟩ := hG.2 g hg
    have hprod := aeval_mem_span_eq_zero haH hm
    rw [map_mul, map_pow, aeval_uN] at hprod
    exact (mul_eq_zero.mp hprod).resolve_left (pow_ne_zero m (Finset.prod_ne_zero_iff.mpr ha))

theorem aeval_zeroSubst {x : Y} {a : Y → L} (ha : a x = 0) (q : MvPolynomial Y K) :
    aeval a (bind₁ (zeroSubst x) q) = aeval a q := by
  have hfun : (fun v => aeval a (zeroSubst (K := K) x v)) = a := by
    funext v
    by_cases hv : v = x
    · subst hv; simp [zeroSubst, ha]
    · simp [zeroSubst, hv]
  rw [aeval_bind₁, hfun]

/-- `V_Y(H|_{x=0}; Z ∪ {x}, N) = {a ∈ V_Y(H; Z, N) : a_x = 0}`. -/
theorem VY_restrictZero (H : Finset (MvPolynomial Y K)) (Z N : Finset Y) (x : Y) :
    VY L (restrictZero H x) (insert x Z) N = {a | a ∈ VY L H Z N ∧ a x = 0} := by
  have key : ∀ a : Y → L, a x = 0 →
      (a ∈ V L (↑(restrictZero H x) : Set (MvPolynomial Y K)) ↔
        a ∈ V L (↑H : Set (MvPolynomial Y K))) := by
    intro a ha
    simp only [mem_V, restrictZero, Finset.coe_erase, Finset.coe_image, Set.mem_sdiff,
      Set.mem_image, Finset.mem_coe, Set.mem_singleton_iff]
    constructor
    · intro h q hq
      by_cases h0 : bind₁ (zeroSubst x) q = 0
      · rw [← aeval_zeroSubst ha, h0, map_zero]
      · rw [← aeval_zeroSubst ha]
        exact h _ ⟨⟨q, hq, rfl⟩, h0⟩
    · rintro h _ ⟨⟨q, hq, rfl⟩, -⟩
      rw [aeval_zeroSubst ha]
      exact h q hq
  ext a
  simp only [VY, Finset.mem_insert, forall_eq_or_imp, Set.mem_ofPred_eq]
  constructor
  · rintro ⟨⟨hx, hZ⟩, hN, hH⟩
    exact ⟨⟨hZ, hN, (key a hx).mp hH⟩, hx⟩
  · rintro ⟨⟨hZ, hN, hH⟩, hx⟩
    exact ⟨⟨hx, hZ⟩, hN, (key a hx).mpr hH⟩

/-- `V_Y(Saturate(H, N ∪ {x}); Z, N ∪ {x}) = {a ∈ V_Y(H; Z, N) : a_x ≠ 0}`. -/
theorem VY_saturate {H G : Finset (MvPolynomial Y K)} {Z N : Finset Y} {x : Y}
    (hG : IsSaturation H G (insert x N)) :
    VY L G Z (insert x N) = {a | a ∈ VY L H Z N ∧ a x ≠ 0} := by
  ext a
  simp only [VY, Set.mem_ofPred_eq]
  constructor
  · rintro ⟨hZ, hN, hGa⟩
    have hN' : ∀ y ∈ N, a y ≠ 0 := fun y hy => hN y (Finset.mem_insert_of_mem hy)
    exact ⟨⟨hZ, hN', (mem_V_iff_of_isSaturation hG hN).mp hGa⟩,
      hN x (Finset.mem_insert_self x N)⟩
  · rintro ⟨⟨hZ, hN, hH⟩, hx⟩
    have hN' : ∀ y ∈ insert x N, a y ≠ 0 := fun y hy => by
      rcases Finset.mem_insert.mp hy with rfl | hy
      exacts [hx, hN y hy]
    exact ⟨hZ, hN', (mem_V_iff_of_isSaturation hG hN').mpr hH⟩

/-- **Lemma `lem:zero-nonzero-split`.** For a valid `Q_1 = Saturate(H, N ∪ {x})`,
`V_Y(H; Z, N) = V_Y(H|_{x=0}; Z ∪ {x}, N) ∪ V_Y(Q_1; Z, N ∪ {x})`. -/
theorem zero_nonzero_split {H G : Finset (MvPolynomial Y K)} (Z N : Finset Y) (x : Y)
    (hG : IsSaturation H G (insert x N)) :
    VY L H Z N = VY L (restrictZero H x) (insert x Z) N ∪ VY L G Z (insert x N) := by
  rw [Set.union_def, exhaustive_split (VQ := VY L H Z N) (A := fun _ => True)
    (D := fun (i : Bool) (a : Y → L) => if i then a x = 0 else a x ≠ 0)
    (Vi := fun i => if i then VY L (restrictZero H x) (insert x Z) N else VY L G Z (insert x N))
    (fun _ _ => trivial) (fun a _ => by by_cases h : a x = 0; exacts [⟨true, h⟩, ⟨false, h⟩])
    (fun i => by cases i <;> simp [VY_restrictZero, VY_saturate hG])]
  ext a
  simp only [Set.mem_iUnion, Bool.exists_bool, Set.mem_ofPred_eq]
  tauto

/-! ### Refinement under the nonzero conditions -/

variable (K Y L) in
/-- `Refine(Q, N)`: refinement that treats the variables of `N` as nonzero. It may drop a
branch, or a factor, only where some variable of `N` vanishes, so it keeps
`V(Q) ∖ V(u_N) = ⋃_{H ∈ 𝓛(Q, N)} V(H) ∖ V(u_N)`. -/
structure CondRefiner where
  refine : Finset (MvPolynomial Y K) → Finset Y → Finset (Finset (MvPolynomial Y K))
  zero_notMem : ∀ {Q : Finset (MvPolynomial Y K)} {N : Finset Y} {H : Finset (MvPolynomial Y K)},
    (0 : MvPolynomial Y K) ∉ Q → H ∈ refine Q N → (0 : MvPolynomial Y K) ∉ H
  mem_V_iff : ∀ {Q : Finset (MvPolynomial Y K)} {N : Finset Y} {a : Y → L},
    (0 : MvPolynomial Y K) ∉ Q → (∀ y ∈ N, a y ≠ 0) →
      (a ∈ V L (↑Q : Set (MvPolynomial Y K)) ↔
        ∃ H ∈ refine Q N, a ∈ V L (↑H : Set (MvPolynomial Y K)))

variable (L) in
/-- Algorithm `alg:Refine`, which ignores `N`, satisfies the specification. -/
def Refiner.toCondRefiner [Finite Y] {m : MonomialOrder Y} (ρ : Refiner K m) :
    CondRefiner K Y L where
  refine Q _ := ρ.refine Q
  zero_notMem h0 hH := (ρ.varsIn_of_mem_refine (S := Set.univ) (fun _ _ _ _ => trivial) h0 hH).2
  mem_V_iff {Q} _ {a} h0 _ := by
    rw [ρ.V_eq_iUnion_refine L h0]
    simp only [Set.mem_iUnion, exists_prop]

/-- Dropping every branch in whose ideal some variable of `N` lies, as `Refine(Q, N)` does once
the variable reduces to zero modulo the branch, keeps the specification. -/
def CondRefiner.dropVanishing (R : CondRefiner K Y L) : CondRefiner K Y L where
  refine Q N := (R.refine Q N).filter fun H =>
    ∀ y ∈ N, X y ∉ Ideal.span (↑H : Set (MvPolynomial Y K))
  zero_notMem h0 hH := R.zero_notMem h0 (Finset.mem_filter.mp hH).1
  mem_V_iff {Q N a} h0 hN := by
    rw [R.mem_V_iff h0 hN]
    constructor
    · rintro ⟨H, hH, haH⟩
      refine ⟨H, Finset.mem_filter.mpr ⟨hH, fun y hy hyH => hN y hy ?_⟩, haH⟩
      simpa using aeval_mem_span_eq_zero haH hyH
    · rintro ⟨H, hH, haH⟩
      exact ⟨H, (Finset.mem_filter.mp hH).1, haH⟩

variable (R : CondRefiner K Y L)

/-- **Equation `eq:zeroing-refinement`.** `V_Y(Q; Z, N) = ⋃_{H ∈ 𝓛(Q, N)} V_Y(H; Z, N)`. -/
theorem VY_eq_iUnion_refine {Q : Finset (MvPolynomial Y K)} (h0 : (0 : MvPolynomial Y K) ∉ Q)
    (Z N : Finset Y) : VY L Q Z N = ⋃ H ∈ R.refine Q N, VY L H Z N := by
  ext a
  simp only [VY, Set.mem_iUnion, Set.mem_ofPred_eq, exists_prop]
  constructor
  · rintro ⟨hZ, hN, hQ⟩
    obtain ⟨H, hH, haH⟩ := (R.mem_V_iff h0 hN).mp hQ
    exact ⟨H, hH, hZ, hN, haH⟩
  · rintro ⟨H, hH, hZ, hN, haH⟩
    exact ⟨hZ, hN, (R.mem_V_iff h0 hN).mpr ⟨H, hH, haH⟩⟩

/-! ### Algorithm `alg:ZeroingBranching` -/

/-- A branch `(Q, Z, N)`. -/
abbrev ZBranch (K : Type*) [Field K] (Y : Type*) :=
  Finset (MvPolynomial Y K) × Finset Y × Finset Y

variable (L) in
/-- The fate of a terminal branch `H` of a pending branch with nonzero set `N`: the solver
returns a representation of `V(H) ∖ V(u_N)`, certifies that it is empty, or fails, after which
`H` is left unresolved or split on a variable `x`, with `G` the result of
`Saturate(H, N ∪ {x})`. -/
inductive Outcome (H : Finset (MvPolynomial Y K)) (N : Finset Y)
  | solved
  | unsat (h : ∀ a : Y → L, (∀ y ∈ N, a y ≠ 0) → a ∉ V L (↑H : Set (MvPolynomial Y K)))
  | unresolved
  | split (x : Y) (G : Finset (MvPolynomial Y K)) (hG : IsSaturation H G (insert x N))
      (hG0 : (0 : MvPolynomial Y K) ∉ G)

/-- What a terminal branch adds to `S`, `U` and `W`. -/
def Outcome.toS {H : Finset (MvPolynomial Y K)} {N : Finset Y} (Z : Finset Y) :
    Outcome L H N → Finset (ZBranch K Y)
  | .solved => {(H, Z, N)}
  | _ => ∅

def Outcome.toU {H : Finset (MvPolynomial Y K)} {N : Finset Y} (Z : Finset Y) :
    Outcome L H N → Finset (ZBranch K Y)
  | .unresolved => {(H, Z, N)}
  | _ => ∅

def Outcome.toW {H : Finset (MvPolynomial Y K)} {N : Finset Y} (Z : Finset Y) :
    Outcome L H N → Finset (ZBranch K Y)
  | .split x G _ _ => {(restrictZero H x, insert x Z, N), (G, Z, insert x N)}
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

/-- One iteration of the while loop: remove `b = (Q, Z, N)` from `W` and process every terminal
branch of `Refine(Q, N)` according to its outcome `o`. -/
def ZStep (st st' : ZState K Y) : Prop :=
  ∃ b ∈ st.W, ∃ o : ∀ H : Finset (MvPolynomial Y K), Outcome L H b.2.2,
    st' = ⟨st.S ∪ (R.refine b.1 b.2.2).biUnion (fun H => (o H).toS b.2.1),
      st.U ∪ (R.refine b.1 b.2.2).biUnion (fun H => (o H).toU b.2.1),
      st.W.erase b ∪ (R.refine b.1 b.2.2).biUnion (fun H => (o H).toW b.2.1)⟩

/-- The initial state `S = U = ∅`, `W = {(P, ∅, ∅)}`. -/
def ZState.init (P : Finset (MvPolynomial Y K)) : ZState K Y :=
  ⟨∅, ∅, {(P, ∅, ∅)}⟩

section SolLemmas

variable (L)

theorem solOf_union (T T' : Finset (ZBranch K Y)) : solOf L (T ∪ T') = solOf L T ∪ solOf L T' := by
  ext a
  simp only [solOf, Set.mem_union, Set.mem_iUnion, Finset.mem_union, exists_prop, or_and_right,
    exists_or]

theorem solOf_biUnion {ι : Type*} (s : Finset ι) (f : ι → Finset (ZBranch K Y)) :
    solOf L (s.biUnion f) = ⋃ i ∈ s, solOf L (f i) := by
  ext a
  simp only [solOf, Finset.mem_biUnion, Set.mem_iUnion, exists_prop]
  tauto

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

/-- What one terminal branch adds to `S`, `U` and `W` represents exactly `V_Y(H; Z, N)`. -/
theorem outcome_sol_eq {H : Finset (MvPolynomial Y K)} (Z N : Finset Y) (o : Outcome L H N) :
    solOf L (o.toS Z) ∪ solOf L (o.toU Z) ∪ solOf L (o.toW Z) = VY L H Z N := by
  cases o with
  | solved => simp [Outcome.toS, Outcome.toU, Outcome.toW, solOf]
  | unsat h =>
    have hVY : VY L H Z N = ∅ := by
      ext a
      simp only [Set.mem_empty_iff_false, iff_false]
      rintro ⟨-, hN, hH⟩
      exact h a hN hH
    simp [Outcome.toS, Outcome.toU, Outcome.toW, solOf, hVY]
  | unresolved => simp [Outcome.toS, Outcome.toU, Outcome.toW, solOf]
  | split x G hG _ =>
    simp only [Outcome.toS, Outcome.toU, Outcome.toW, solOf, Finset.notMem_empty,
      Set.iUnion_of_empty, Set.iUnion_empty, Set.empty_union, Finset.mem_insert,
      Finset.mem_singleton, Set.iUnion_iUnion_eq_or_left, Set.iUnion_iUnion_eq_left]
    exact (zero_nonzero_split Z N x hG).symm

/-- Pending branches have no zero generator. -/
theorem zero_notMem_toW {H : Finset (MvPolynomial Y K)} {Z N : Finset Y} (o : Outcome L H N) :
    ∀ b ∈ o.toW Z, (0 : MvPolynomial Y K) ∉ b.1 := by
  cases o with
  | solved | unsat | unresolved => simp [Outcome.toW]
  | split x G _ hG0 =>
    simp only [Outcome.toW, Finset.mem_insert, Finset.mem_singleton]
    rintro b (rfl | rfl)
    exacts [zero_notMem_restrictZero H x, hG0]

/-- The invariant of the proof of Theorem `thm:zeroing-correctness`. -/
def ZInv (P : Finset (MvPolynomial Y K)) (st : ZState K Y) : Prop :=
  V L (↑P : Set (MvPolynomial Y K)) = solOf L st.S ∪ solOf L st.U ∪ solOf L st.W ∧
    ∀ b ∈ st.W, (0 : MvPolynomial Y K) ∉ b.1

theorem zInv_init {P : Finset (MvPolynomial Y K)} (h0 : (0 : MvPolynomial Y K) ∉ P) :
    ZInv (L := L) P (ZState.init P) := by
  refine ⟨?_, ?_⟩
  · simp [ZState.init, solOf, VY_root]
  · simpa [ZState.init] using h0

theorem ZStep.zInv {P : Finset (MvPolynomial Y K)} {st st' : ZState K Y}
    (hs : ZStep R st st') (hinv : ZInv (L := L) P st) : ZInv (L := L) P st' := by
  obtain ⟨b, hb, o, rfl⟩ := hs
  obtain ⟨hV, hgood⟩ := hinv
  have hb0 := hgood b hb
  refine ⟨?_, ?_⟩
  · rw [hV]
    simp only [solOf_union, solOf_biUnion]
    rw [← solOf_erase_union L hb, VY_eq_iUnion_refine R hb0]
    have hsplit : ⋃ H ∈ R.refine b.1 b.2.2, VY L H b.2.1 b.2.2 =
        (⋃ H ∈ R.refine b.1 b.2.2, solOf L ((o H).toS b.2.1)) ∪
          (⋃ H ∈ R.refine b.1 b.2.2, solOf L ((o H).toU b.2.1)) ∪
          (⋃ H ∈ R.refine b.1 b.2.2, solOf L ((o H).toW b.2.1)) := by
      ext a
      simp only [Set.mem_union, Set.mem_iUnion, exists_prop]
      constructor
      · rintro ⟨H, hH, ha⟩
        rw [← outcome_sol_eq _ _ (o H)] at ha
        rcases ha with (ha | ha) | ha
        · exact Or.inl (Or.inl ⟨H, hH, ha⟩)
        · exact Or.inl (Or.inr ⟨H, hH, ha⟩)
        · exact Or.inr ⟨H, hH, ha⟩
      · rintro ((⟨H, hH, ha⟩ | ⟨H, hH, ha⟩) | ⟨H, hH, ha⟩) <;> refine ⟨H, hH, ?_⟩ <;>
          rw [← outcome_sol_eq _ _ (o H)] <;> simp [ha]
    rw [hsplit]
    ext a
    simp only [Set.mem_union]
    tauto
  · intro b' hb'
    rcases Finset.mem_union.mp hb' with hb' | hb'
    · exact hgood b' (Finset.mem_of_mem_erase hb')
    · obtain ⟨H, -, hb'H⟩ := Finset.mem_biUnion.mp hb'
      exact zero_notMem_toW (o H) b' hb'H

/-- **Theorem `thm:zeroing-correctness` (Correctness of recursive splitting).** If
Algorithm `alg:ZeroingBranching` returns `(S, U)` on input `P`, i.e. a run reaches `W = ∅`, then
`V(P) = Sol(S) ∪ ⋃_{(H, Z, N) ∈ U} V_Y(H; Z, N)`. This holds for every choice of solver results,
saturations, timeouts, budget and splitting variables. -/
theorem zeroing_correct {P : Finset (MvPolynomial Y K)} (h0 : (0 : MvPolynomial Y K) ∉ P)
    {st : ZState K Y} (hrun : Relation.ReflTransGen (ZStep R) (ZState.init P) st)
    (hW : st.W = ∅) :
    V L (↑P : Set (MvPolynomial Y K)) = solOf L st.S ∪ solOf L st.U := by
  have hinv : ZInv (L := L) P st := by
    clear hW
    induction hrun with
    | refl => exact zInv_init h0
    | tail _ hs ih => exact hs.zInv R ih
  rw [hinv.1, hW]
  simp [solOf]

/-- **Corollary `cor:zeroing-completeness` (Completeness of variable zeroing).** If `U = ∅` in
the output, then `Sol(S) = V(P)`. -/
theorem zeroing_complete {P : Finset (MvPolynomial Y K)} (h0 : (0 : MvPolynomial Y K) ∉ P)
    {st : ZState K Y} (hrun : Relation.ReflTransGen (ZStep R) (ZState.init P) st)
    (hW : st.W = ∅) (hU : st.U = ∅) :
    solOf L st.S = V L (↑P : Set (MvPolynomial Y K)) := by
  rw [zeroing_correct R h0 hrun hW, hU]
  simp [solOf]

/-! ### Termination (Theorem `thm:zeroing-termination`) -/

section Termination

/-- A split of a terminal branch of `(Q, Z, N)` uses a variable outside `Z ∪ N`, since
`ChooseVariable` picks it from `vars(H) ∖ (Z ∪ N)`. -/
def Outcome.Fresh {H : Finset (MvPolynomial Y K)} {N : Finset Y} (Z : Finset Y) :
    Outcome L H N → Prop
  | .split x _ _ _ => x ∉ Z ∧ x ∉ N
  | _ => True

/-- One iteration of the while loop as Algorithm `alg:ZeroingBranching` performs it: a `ZStep`
whose split variables lie outside `Z ∪ N`. -/
def ZStepFresh (st st' : ZState K Y) : Prop :=
  ∃ b ∈ st.W, ∃ o : ∀ H : Finset (MvPolynomial Y K), Outcome L H b.2.2,
    (∀ H, (o H).Fresh b.2.1) ∧
    st' = ⟨st.S ∪ (R.refine b.1 b.2.2).biUnion (fun H => (o H).toS b.2.1),
      st.U ∪ (R.refine b.1 b.2.2).biUnion (fun H => (o H).toU b.2.1),
      st.W.erase b ∪ (R.refine b.1 b.2.2).biUnion (fun H => (o H).toW b.2.1)⟩

/-- Theorem `thm:zeroing-correctness` covers these iterations. -/
theorem ZStepFresh.zStep {st st' : ZState K Y} (hs : ZStepFresh R st st') : ZStep R st st' := by
  obtain ⟨b, hb, o, -, h⟩ := hs
  exact ⟨b, hb, o, h⟩

/-- A multiset below one that is smaller in the Dershowitz–Manna order is smaller too. -/
theorem isDershowitzMannaLT_of_le {α : Type*} [Preorder α] {M N P : Multiset α} (hle : M ≤ N)
    (hlt : Multiset.IsDershowitzMannaLT N P) : Multiset.IsDershowitzMannaLT M P := by
  obtain ⟨D, rfl⟩ := Multiset.le_iff_exists_add.mp hle
  by_cases hD : D = 0
  · simpa [hD] using hlt
  · exact Multiset.IsDershowitzMannaLT.trans ⟨M, 0, D, hD, (add_zero M).symm, rfl, by simp⟩ hlt

variable [Fintype Y]

/-- `|Y ∖ (Z ∪ N)|`, the number of variables a pending branch `(Q, Z, N)` can still split on. -/
def freeVars (b : ZBranch K Y) : ℕ :=
  (b.2.1 ∪ b.2.2)ᶜ.card

/-- One iteration decreases the multiset of `|Y ∖ (Z ∪ N)|` over the pending branches: it removes
one pending branch and adds pending branches with one more decided variable. -/
theorem ZStepFresh.freeVars_lt {st st' : ZState K Y} (hs : ZStepFresh R st st') :
    Multiset.IsDershowitzMannaLT (st'.W.val.map freeVars) (st.W.val.map freeVars) := by
  obtain ⟨b, hb, o, hfresh, rfl⟩ := hs
  dsimp only
  set C := (R.refine b.1 b.2.2).biUnion fun H => (o H).toW b.2.1
  have hC : ∀ c ∈ C, freeVars c < freeVars b := by
    intro c hc
    obtain ⟨H, -, hcH⟩ := Finset.mem_biUnion.mp hc
    have hf := hfresh H
    generalize o H = oH at hcH hf
    cases oH with
    | solved | unsat | unresolved => simp [Outcome.toW] at hcH
    | split x G _ _ =>
      obtain ⟨hxZ, hxN⟩ := hf
      have hlt : (insert x (b.2.1 ∪ b.2.2))ᶜ.card < (b.2.1 ∪ b.2.2)ᶜ.card :=
        Finset.card_lt_card (Finset.compl_ssubset_compl.mpr
          (Finset.ssubset_insert (Finset.notMem_union.mpr ⟨hxZ, hxN⟩)))
      simp only [Outcome.toW, Finset.mem_insert, Finset.mem_singleton] at hcH
      rcases hcH with rfl | rfl
      · show (insert x b.2.1 ∪ b.2.2)ᶜ.card < (b.2.1 ∪ b.2.2)ᶜ.card
        rw [Finset.insert_union]
        exact hlt
      · show (b.2.1 ∪ insert x b.2.2)ᶜ.card < (b.2.1 ∪ b.2.2)ᶜ.card
        rw [Finset.union_insert]
        exact hlt
  have hW : st.W.val.map freeVars = (st.W.erase b).val.map freeVars + {freeVars b} := by
    conv_lhs => rw [← Multiset.cons_erase (Finset.mem_val.mpr hb)]
    rw [Multiset.map_cons, Finset.erase_val, ← Multiset.singleton_add, add_comm]
  have hle : (st.W.erase b ∪ C).val.map freeVars ≤
      (st.W.erase b).val.map freeVars + C.val.map freeVars := by
    rw [← Multiset.map_add, Finset.union_val]
    exact Multiset.map_le_map (Multiset.union_le_add _ _)
  refine isDershowitzMannaLT_of_le hle ?_
  rw [hW]
  refine ⟨_, _, {freeVars b}, by simp, rfl, rfl, fun y hy =>
    ⟨freeVars b, Multiset.mem_singleton_self _, ?_⟩⟩
  obtain ⟨c, hc, rfl⟩ := Multiset.mem_map.mp hy
  exact hC c (Finset.mem_val.mp hc)

/-- **Theorem `thm:zeroing-termination` (Termination of recursive splitting).** With finitely many
variables, Algorithm `alg:ZeroingBranching` makes finitely many iterations of its while loop,
for every choice of solver results, saturations, timeouts, budget and splitting variables. -/
theorem zeroing_terminates : WellFounded fun st' st : ZState K Y => ZStepFresh R st st' :=
  Subrelation.wf (fun hs => ZStepFresh.freeVars_lt R hs)
    (InvImage.wf (fun st : ZState K Y => st.W.val.map freeVars)
      Multiset.wellFounded_isDershowitzMannaLT)

end Termination

/-! ### Branch contraction (Subsection `subsec:zeroing-branch-contraction`) -/

/-- If the `x = 0` child of a split of `H` has no terminal branches and the `x ≠ 0` child
`Q_1 = Saturate(H, N ∪ {x})` has exactly one, `H'`, then `V_Y(H; Z, N) = V_Y(H'; Z, N ∪ {x})`. -/
theorem contraction_zero_unsat {H G H' : Finset (MvPolynomial Y K)} {Z N : Finset Y} {x : Y}
    (hG : IsSaturation H G (insert x N)) (hG0 : (0 : MvPolynomial Y K) ∉ G)
    (hzero : R.refine (restrictZero H x) N = ∅) (hone : R.refine G (insert x N) = {H'}) :
    VY L H Z N = VY L H' Z (insert x N) := by
  rw [zero_nonzero_split Z N x hG, VY_eq_iUnion_refine R (zero_notMem_restrictZero H x), hzero,
    VY_eq_iUnion_refine R hG0, hone]
  simp

/-- If the `x ≠ 0` child has no terminal branches and the `x = 0` child has exactly one, `H'`,
then `V_Y(H; Z, N) = V_Y(H'; Z ∪ {x}, N)`. -/
theorem contraction_nonzero_unsat {H G H' : Finset (MvPolynomial Y K)} {Z N : Finset Y} {x : Y}
    (hG : IsSaturation H G (insert x N)) (hG0 : (0 : MvPolynomial Y K) ∉ G)
    (hzero : R.refine (restrictZero H x) N = {H'}) (hone : R.refine G (insert x N) = ∅) :
    VY L H Z N = VY L H' (insert x Z) N := by
  rw [zero_nonzero_split Z N x hG, VY_eq_iUnion_refine R (zero_notMem_restrictZero H x), hzero,
    VY_eq_iUnion_refine R hG0, hone]
  simp

/-! ### Nonzero encodings (Subsection `subsec:nonzero-encoding`)

The branch solver receives the nonzero conditions of a branch and may encode them with
auxiliary variables; these identities show that both encodings describe `V(Q) ∖ V(u_N)`. -/

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
