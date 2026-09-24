# Review of `paper/paper.tex`

**Reviewed:** commit `37e949f` ("Mention Grabe's method in the introduction"), all files under `paper/sections/`, `paper/figs/`, `references.bib`, and the compiled 17-page `paper.pdf`. Numbers were also checked against `results/physical_system_120s_results.json`.
**Criteria:** the `scientific-article-writing` house style, plus the Journal of Symbolic Computation (JSC) requirements. I assumed JSC because the paper uses the Elsevier CAS template and cites mostly JSC work. Skip §G if the paper is going somewhere else.
**Line references** are `file:line`, relative to `paper/`.

---

## Summary

The core idea is clear and worth publishing. The paper interreduces the system and factors every equation. It then takes the minimal hitting sets of the factor sets as branches, recurses, and passes each terminal branch to any exact solver. When a branch times out, it splits on x = 0 versus x ≠ 0. The proofs for Algorithm 2 are mostly correct. Appendix A is a strong feature: it lists every measurement.

Seven issues most affect whether a referee would accept the paper:

1. The paper never states how it differs from its closest prior work, factorizing Gröbner bases (Gräbe; Singular's `facstd`) (A1).
2. The abstract promises irreducible components, but the method returns a possibly redundant union of branch varieties (A2).
3. All 20 benchmark systems are built to factor, and each method class is represented by its best configuration per system. The fair comparison with a fixed backend is already in the appendix and is stronger, but the paper doesn't show it (A4).
4. The physical-system results contradict each other in ways the text doesn't explain: 33 versus 4 root hitting sets, and 32 SAT branches versus 6 versus 1 components. The key standalone baseline, `facstd`, is also not reported (A5).
5. The evidence for variable zeroing is one system and one run, and the paper's wording claims more than that supports (A6).
6. Timing method: single runs, a startup cost of about 30 s in the numerical baseline, and solver threads that keep running after their timeout (A7).
7. `UpdateSaturation`, `FactorBranchingWithTimeout`, and branch contraction are never defined. Theorem 6's correctness depends on them (B1–B3).

---

## A. Major concerns: argument and scientific accuracy

### A1. Novelty against the closest prior work is not stated
- `sections/introduction.tex:20-22` gives factorizing Gröbner-basis methods one sentence. That sentence says they "preserve the solution set" and "interreduce and prune redundant branches", so it lists what they share with this work, not what separates them.
- Readers must work out the novelty on their own. The likely differences are:
  - (i) splitting over **all** equations at once through minimal hitting sets, instead of one polynomial at a time inside Buchberger's loop;
  - (ii) factoring only between solver calls, so any solver can be plugged in;
  - (iii) recursive refinement with a termination proof;
  - (iv) variable zeroing when a branch times out.
- The gap is also hard to see because `facstd` is both a baseline and the only branch solver that completes the physical system (Table 3). FB+`facstd` stacks this framework on top of a factorizing Gröbner basis, and the paper doesn't say what the outer layer contributes.
- **Fix:** add a paragraph comparing the method to Gräbe's factorized Gröbner bases and `facstd`. Say what it improves, under which conditions, and at what cost; for example, the number of hitting sets can grow exponentially. Put this comparison in the introduction's contribution statement and repeat it in the conclusion.
- Related lines of work that a JSC referee will expect to see discussed:
  - triangular and regular-chain decompositions, which also split on "initial = 0 or ≠ 0";
  - comprehensive Gröbner systems, which split on parameter conditions of the form = 0 / ≠ 0;
  - SMT solvers for nonlinear arithmetic, since the paper borrows SAT/UNSAT and conflict-driven learning terminology.
  
  See F1 for candidate references.

### A2. What the method outputs versus what the abstract promises
- The abstract (`paper.tex:52-53`) and introduction (`introduction.tex:7-9`) motivate the work with the dimension and the **irreducible components** of the solution set.
- What the framework actually returns is a union of branch varieties, one exact representation per terminal hitting set. Branches can overlap or be contained in one another. They are not irreducible, and the nonzero branches are locally closed, not closed.
- Dimension follows as the maximum over branches; Table 3's `d` appears to be exactly that. Irreducible components need further decomposition of each branch and removal of redundant ones.
- **Fix:** state exactly what the output is, for example "a finite list of pairs (Gröbner basis, nonzero conditions) whose varieties cover V(P)". Say which global properties can be read off it directly. Either scale back the claim about components or describe the post-processing step.

### A3. The complexity argument does not support the method, and the authors' own comment says so
- `introduction.tex:45` holds the TODO "Needs a better justification…".
- `introduction.tex:46` and `method.tex:8` cite Mayr–Meyer's doubly exponential bound in the **number of variables**. Factor branching keeps every variable; only zeroing removes any. So this bound doesn't explain why the branches are easier.
- **Fix:** argue in terms of degree, component count, or structure instead. Factors have lower degree, so standard degree-based bounds on Gröbner-basis computation drop. Each branch ideal also has fewer components. Alternatively, drop the complexity argument and rest the claim on the experiments.
- The cost the method adds is not analyzed. The number of minimal hitting sets can be exponential; it is at most ∏|Cᵢ|. The paper also never names the enumeration algorithm used. Cite the standard minimal-transversal enumeration literature (Berge; Eiter–Gottlob; Fredman–Khachiyan) and state what the implementation does.

### A4. Benchmark design and comparison method favor the method
- **The benchmark is selected to fit the method.** Every family is built from products of factors: signed-distance cycles, affine hypergraphs, overlapping random multiples, and "gated" systems where every equation is multiplied by t. Gating makes V = {t = 0} ∪ V(original), which is trivial for any method that factors. For the gated rows, factor branching's running time is essentially the time to solve the original system.
  - **Fix:** say plainly that the families were designed to have factor structure. Also report (i) the ungated systems, (ii) standard benchmarks that don't factor, to measure the overhead, and (iii) systems from applications that factor naturally.
- **Each class is scored by its best configuration per system.** Table 2 picks the fastest of 5 factor-branching backends against the fastest of 6 direct configurations, 3 decomposition methods, and 1 numerical method. Classes with more configurations gain from this selection.
- **A fairer comparison is already in the appendix.** With one fixed backend, FB+Singular (Table 4, last column) completes **19/20**. Standalone `facstd` completes 17/20 (Table 7) and Singular's direct Gröbner basis 7/20 (Table 6). FB+Singular is faster than `facstd` on 16 of the 19 systems both can compare. It is slower on positive affine (8,6,4), positive affine (10,8,5), and square affine (10,5).
  - **Fix:** put this head-to-head in the main text. It is a stronger and fairer claim than "#fastest (ties counted)". One caveat: FB+Singular runs a persistent worker, so check that per-process startup is counted the same way on both sides.
- **The methods compute different things.** A Gröbner basis of the whole ideal, a primary decomposition (`primdecGTZ`), a factorized Gröbner basis, a numerical solution set, and a union of branches are different outputs. Wall times across them are not directly comparable.
  - **Fix:** list what each baseline computes, including its Singular or Macaulay2 command, monomial order, and characteristic. For HomotopyContinuation.jl, name the function. The gated systems are positive-dimensional, so if the function was `solve`, it returns only isolated roots and is not comparable.

### A5. Physical-system results: unexplained contradictions and unsupported claims (`experiments.tex:133-178`, Table 3)
- **The two pipelines disagree about the same system.** The results JSON shows that factor branching enumerated **33** minimal hitting sets: `facstd` found 32 SAT and 1 UNSAT. The variable-zeroing run had only **4** hitting sets at the root: `covers=4`, which matches the 4 root children in Fig. 3. With **0 splits**, the variable-zeroing `facstd` run should be the same computation as the factor-branching one. Instead it reports "6 comp." in 0.856 s, against "32 SAT, 1 UNSAT" in 1.341 s. Variable zeroing with M4GB and with Singular reports "1 comp.".
  - **Fix:** explain which refinement settings differ between the two pipelines; interreduction is the likely candidate. Also explain what "comp." counts. Then show that all the complete runs describe the same variety, for example the same dimension and degree for each branch.
- **Claim with no support (`experiments.tex:168-170`).** "Factor branching recovers all solutions quickly with every backend." When one hitting set has timed out, a run cannot know it has all the solutions. Only the `facstd` run, which found that branch UNSAT, shows this after the fact. Reword the sentence to say that.
- **Standalone baselines are not reported.** "All standalone baselines timed out" gives neither the list of baselines nor their time limit. Table 3 has no row for them; the setup says 60 s, but this experiment uses 120 s. Because FB+`facstd` finishes in 1.3 s, the reader's first question will be how long standalone `facstd` takes. Report it explicitly. If it times out, that is the paper's best result and belongs in the abstract.
- **`d = 14` is never defined.** In the results file it is `largest_dimension`. Define it in the text.
- **Time limits are unclear.** The caption says "5-second inner timeout and 120-second total limit", which describes variable zeroing only. The factor-branching column uses 120 s per hitting set, and its overall limit is not stated.
- **The system is described in one sentence.** Give the number of variables and equations, the degrees, the number of terms, the coefficient field, and the physical meaning of each variable (Fig. 3 uses `pph`, `DpDph2`, and `xr`). Also state which paper the equations come from; currently a master's thesis is cited. The phrase "constraining graviton couplings … to remain consistent with QFT" is vague. If these are fixed-point (beta-function) equations of a truncated RG flow, say so, and say whether the physics question needs real solutions.
- **Heading.** "Physical Systems" is plural, but there is one system. It should also be in sentence case, like "Baseline systems".

### A6. The evidence for variable zeroing is thin, and the wording claims more (`variable_zeroing.tex:24-25, 89-97`; `introduction.tex:53`)
- **Scope.** On the 20 baseline systems, variable zeroing never helped. Every row has 0 splits except Cyclic-7, which stayed partial, and dense n=7, which failed with an error (Table 5). Its only positive result is two extra backends completing on one physical system. The best configuration there, FB+`facstd`, needs no splits.
  - **Fix:** frame the result as "makes completion less dependent on which backend is used, on this system". Remove "proves particularly effective" (`introduction.tex:53`).
- **The ablation is in the wrong section and under-reported.** The x = 1 and random-a comparisons sit in the method section, before the experimental setup, and give no backend, per-hitting-set timeout, seed, or numbers. "Demonstrate that zero/nonzero splits are substantially more effective … on these structured systems" rests on a single system with one run per variant.
  - **Fix:** move the comparison to §5.3, report it in a table, and restrict the claim to that one system.
- **"Our experience with the physical systems considered here"** uses the plural, but there is one system.
- **"A variable is often always zero or always nonzero" (`variable_zeroing.tex:24-25`)** is imprecise. On any irreducible component, a variable either vanishes identically or is nonzero on a dense open subset. That holds for every system, so it cannot be what makes zeroing work here.
  - **Fix:** state the actual mechanism. Setting x = 0 deletes every monomial containing x, which can make generators factor. Setting x ≠ 0 saturates by x, which removes the components lying in {x = 0} and lets interreduction prune any hitting set that contains x.
- **Dense n=7 shows how zeroing can hurt.** Factor branching with Singular solves this system in 42 s. Variable zeroing, with a 3-s inner timeout, started splitting and crashed. When a branch is slow but solvable, a short inner timeout turns it into many splits. Discuss this failure mode.
- **The splitting-variable heuristic (§4.2) is not tested.** There is no comparison with random choice or with each component of the score used alone.

### A7. Measurement method (`experiments.tex:20-36`, appendix)
- **Every result is a single run, reported to 3 decimals.** The same M4GB configuration took 39.762 s in Table 3 and 36.314 s in the Fig. 3 run, about 10% apart. Report the median of at least 3–5 runs, or state the variability. Entries of `0.000` can't be distinguished from each other.
- **The numerical baseline has a startup cost of about 30 s.** HomotopyContinuation.jl takes 29–37 s even on n = 5 systems, which points to Julia startup and compilation. Say whether this startup time is included, and apply the same rule to Singular and Macaulay2 process startup. The in-process Rust backends pay no startup cost.
- **Threads.** The hardware has 14 cores, but the paper never says whether backends run on one thread or many. It also doesn't say whether factor branching solves branches in parallel. Macaulay2 and Julia may use several threads by default.
- **Threads that outlive their timeout.** `appendix.tex:104-107` says the in-process backends (M4GB, F5, GVW, F4) keep running after a per-hitting-set timeout. So in every variable-zeroing run with these backends, including the F5, GVW, and F4 rows of Table 3, timed-out work competes with later branches for CPU and memory. This confounds those results. It also breaks the assumption of Theorem 7 (see B9). Say so in the main text, not only in the appendix.
- **The inner timeout was tuned on the test systems.** The 3-s value was chosen on the same hard gated systems it is evaluated on (`experiments.tex:22-26`). Acknowledge this.
- **What F measures.** F is measured from the start of the hitting-set search, so it excludes factorization. State why F is of interest; it measures how soon the first solution appears. Also note that the baselines have no comparable number.
- **What the backends are.** The results file labels the "F4" backend `symbolica`. State which implementations are the authors' own (apparently M4GB, F5, and GVW) and which come from libraries. A slow custom implementation affects the "direct GB" class.

---

## B. Correctness and precision of the theory

### B1. `UpdateSaturation` is undefined, and the product encoding may make it unsound (`variable_zeroing.tex:48, 62-68, 106`)
- §2.4 gives two encodings. Line 106 ("appended to the product in Equation (2)") suggests the implementation uses the single-variable product σ_N = 1 − s∏x.
- Interreduction can rewrite σ_N. Its leading term s∏x can be reduced by other generators, which puts s into other polynomials of the terminal branch R. If `UpdateSaturation` then just replaces σ_N with σ_{N∪{x}}, the generators that already contain s now refer to a different s, and Q₁ no longer encodes "Q and x ≠ 0".
- Equation (3) and Theorem 6 depend on π(V(Q₁)) = {a ∈ π(V(Q)) : a_x ≠ 0}.
- **Fix:** define `UpdateSaturation` exactly. Either eliminate s before updating, use a fresh auxiliary variable for each split, or prove that the replacement is sound. Also state which encoding the experiments use, and drop the other one if it isn't used.

### B2. Branch contraction (§4.3)
- "There is no need to run the branch solver on H" (`variable_zeroing.tex:134-135`) is a heuristic, not a consequence of the theory. H could still be solvable within τ. Say that the step skips a solver call H would probably fail, and that correctness is unaffected.
- Contraction is not in Algorithm 4, even though Fig. 3 shows contracted nodes. Add it to the algorithm, or label it as an optimization in the implementation.
- "Up to exchanging the two children" leaves only one of the two cases displayed. If the x ≠ 0 child is the UNSAT one, the right-hand side becomes V_Y(H; Z∪{x}, N).
- "Let 𝓗₀ and 𝓗₁ be the minimal hitting sets" should read "the families of minimal hitting sets".

### B3. Algorithm 4 does not match the procedure that was run or analyzed
- `FactorBranchingWithTimeout`, `ChooseVariable`, `Substitute`, and `Variables` have no definitions and no pseudocode. Equation (4) rests on the first of these.
- §4.1 says splitting runs "until … the computation budget is exhausted", and the experiments use a 120-s total limit. Algorithm 4 has no global budget. Branches cut off by the budget must go into U, or Theorem 6 does not cover the reported runs.
- `ChooseVariable(R, X)` can pick a variable that doesn't occur in R, which gives two useless children. The appendix mentions "useful splitting variables" (`appendix.tex:109`), but the algorithm does not define them. Restrict the choice to X ∩ vars(R), or define "useful".

### B4. The role of the set A in Algorithm 2 (`method.tex:23, 31`)
- A holds "normalized systems visited on the current recursion path", but "normalized" is never defined. The termination proof (Theorem 2) never uses A: it shows that M strictly increases whenever H ≠ Q̃.
- **Fix:** explain what A adds, such as pruning repeated systems early, or remove it. Define the normalization used when comparing H with Q̃: monic, sorted, as sets.

### B5. Theorems 3 and 4
- "Sound and complete" is used but never defined. Define it: every represented point is a solution, and every solution is represented.
- Theorem 3 assumes that 𝓑 returns on every branch. Algorithm 3 has no path for a call that never returns, so state this assumption in the theorem.
- Theorem 4 then follows immediately and can be a corollary or merged into Theorem 3.
- Lemma 1's hypothesis ("Suppose refinement … terminates") is redundant once Theorem 2 is proved. Put Theorem 2 first and state Lemma 1 without the hypothesis.
- The proof of Theorem 3 repeats the reasoning of Lemma 1 (`method.tex:200-202`). Cite the lemma instead.

### B6. The remark about algebraically closed fields is confused (`method.tex:101-102`)
V is already defined over K̄ (§2.1), and nothing assumes K is algebraically closed. The actual point is that the decomposition identities hold for points over any extension L ⊇ K, and that 1 ∈ ⟨H⟩ rules out solutions over every such L. Say that, and add one sentence on real solutions if the physics application needs them.

### B7. Theorem 5 (`variable_zeroing.tex:146-159`)
- "Qᵢ encodes Q together with Dᵢ" is informal. Define it as V_Y(Qᵢ; A∪{Dᵢ}) = {a ∈ V_Y(Q; A) : a satisfies Dᵢ}.
- For the x = 0 child, Q|_{x=0} lives in fewer variables. Say that the projection onto Y puts x = 0 back.
- The label `lem:zeroing-split` names a theorem. The result is short enough to be a lemma.

### B8. The proof of Theorem 6 is a sketch
"Applying this argument recursively proves the result" should be an explicit induction on the finite recursion tree, like the proof of Lemma 1.

### B9. Theorem 7 (`variable_zeroing.tex:204-215`)
- "Returns after τ" should be "returns within time τ, with either a result or a timeout".
- As noted in A7, the in-process backends do not meet this assumption in practice.
- The proof should also note that each node has finitely many children, since Algorithm 2 returns finitely many hitting sets.

### B10. Edge cases worth one sentence each
- A constant generator has an empty factor set, so it has no hitting sets and the branch closes as UNSAT during refinement, before any solver runs. This is also how interreduction to {1} prunes hitting sets under the nonzero encoding. It is a useful point to make.
- Line 1 of Algorithm 2 removes zeros. If the system becomes empty, its only hitting set is ∅, and V = K̄ⁿ.

### B11. Notation clashes
- **R** is the ring K[x] in §2, the output of Refine in Algorithm 3 (line 53), and a timed-out branch in Algorithm 4.
- **T** is the terminal set (Algorithm 2), the timed-out set (Algorithm 4), and end-to-end time (§5).
- **N** is the set of nonzero variables and also the node count in Table 5.
- **P** is the input system and also "partial" in the appendix tables.
- **F** is a factor set Fᵢ (proof of Theorem 1) and time to first SAT (§5). Also, Cᵢ in §2.3 and Fᵢ in Theorem 1 name the same object.
- 𝓗(P) is defined in §2.3 and again, as 𝓗(Q), at `method.tex:104-105`.
- The same object is called a "hitting set", "branch", "subproblem", "cover" (Fig. 3 legend: "no covers"), and "hitting-set computation" (`experiments.tex:22`, where it is unclear whether enumeration or solving is meant).
- **Fix:** separate the combinatorial hitting set H from its branch system. Use one word for the branch throughout, and use "cover" nowhere.

---

## C. Structure and exposition

### C1. Abstract (`paper.tex:50-68`, 208 words, within the JSC limit of 250)
- Delete the leftover template comments (`% [solving large sets …]`, `% [broadest strongest claim]`, …) before submission.
- `:57` "variants of Buchberger's algorithm with syzygy, resultants, and cylindrical algebraic decomposition": "with syzygy" is unclear (syzygy criteria? signatures?). CAD and resultants don't come up anywhere else in the paper. Meanwhile numerical homotopy, which is a baseline, is missing. List the methods the paper actually compares against.
- `:59` "even the most modern approaches still often fail on practical problems or slight perturbations of standard benchmarks" has no support. The gated results do support it, so connect the two: direct Gröbner bases fail on gated Katsura-8.
- There are no numbers. Add the main results, for example "completes 19 of 20 benchmarks with a single backend, versus 17 for `facstd`" and the physical-system result with its time limit.
- `:65` "no standard method in our comparison succeeds": give the time limit (120 s).
- `:67` The closing sentence ("offer a path toward…") is vague. Replace it with a concrete outlook.

### C2. Introduction
- The gap (`:35`) is a one-sentence paragraph: "large systems can remain inaccessible". Name the structure the paper exploits, which is equations that are products of shared factors, as in fixed-point equations from physics. Then say why existing methods struggle with it: direct Gröbner bases expand the products, `facstd` splits one polynomial at a time, and numerical methods can't certify completeness.
- The machine-learning paragraph (`:28-32`) doesn't connect to the gap. Either link it (those methods could serve as branch solvers) or cut it.
- Put a concrete contribution statement (see A1) with key numbers before the method description. At present, `:56-61` announces the physical-system result but no baseline result.
- Delete the TODO comment at `:45`.

### C3. Add a small worked example (the highest-value addition)
There is no example anywhere in the paper. A 3–4 equation system would help readers far more than Figs. 1–2. It should show: the factor sets, the minimal hitting sets, one refinement step where interreduction creates new factors, a terminal branch, and one zero/nonzero split. Reuse the same example for Theorem 1 and for §4.3.

### C4. Preliminaries contain unused or misplaced material
- The UNSAT-core definitions (`preliminaries.tex:85-87`) serve only §3.2, which reports a negative result. Remove them, or move them to where they are used.
- §2.4 (nonzero encodings) is needed only in §4. Move it there, and keep only the encoding the implementation uses (B1). Name it as the Rabinowitsch trick and cite it.
- Algorithm 1 (interreduction) is textbook material. A citation and one sentence could replace it, leaving only the facts used later: the generated ideal is preserved and M does not decrease.
- The Berge citation (`:103`) is attached to the branch claim. It belongs with the definition of a hitting set (minimal transversal).

### C5. §3.2 UNSAT-core learning
It reports a negative result with no data. Either give the numbers (in the appendix) and say why learned cores are rarely reused (branches are usually disjoint), or reduce it to one sentence in the discussion.

### C6. Section order
- §4.1 contains an experiment (the ablation) and refers ahead to Fig. 3 and §5.3 before the experimental setup exists. Keep §4 limited to the method and theory.
- Fig. 3 comes in §4 but shows the result of §5.3. Move it to §5.3.

### C7. No discussion or limitations section
The conclusion (`conclusion.tex`) is 20 lines and just restates results. Add a discussion covering:
- when the method helps: factor structure, and branches much easier than the whole system;
- when it doesn't: Cyclic-7, and the exponential number of hitting sets;
- that the output may be redundant;
- the dependence on the backend;
- that the benchmark was designed for the method;
- that timeouts are not cancellable;
- that there is one physical system and one run per result;
- the most important piece of future work.

"Future work will test this benefit on other systems" is too vague.

### C8. Implementation details a reader needs to reproduce the work
- Which algorithm enumerates minimal hitting sets.
- The factorization routine, over ℚ using Symbolica.
- The monomial order.
- Whether interreduction is on in each experiment, including the physical system (see A5).
- Whether the fewest-generators-first ordering (`method.tex:67-70`) is used. The results file says "fewest-factors-first", so reconcile "generators" with "factors".
- Which implementations are the authors' own (A7).

### C9. Reproducibility of the benchmarks
- The families are not defined well enough to regenerate: what is a "signed-distance cycle" or a "positive affine hypergraph", and what do (n,p,k) and (n,m,k) mean?
- The bullet list at `experiments.tex:43-47` is not parallel. The first bullet covers two families, and the last bullet describes gating rather than a family.
- **Fix:** give defining formulas, or a pointer to the generator code together with a data/code availability statement (required by JSC, see G).

---

## D. Figures and tables

### D1. Figure 1 (`figs/branching_flowchart.tex`)
- The first step ("Initialize") is on the right and points left to "Refine", so the diagram reads right to left.
- There is no explicit "more hitting sets?" decision. The single rail from SAT/UNSAT splits into "done" and "next H" without saying what decides between them.
- The text is in saturated pure blue, red, green, and violet. Muted colors would look calmer.
- The figure repeats the 9 lines of Algorithm 3 and adds little. Consider replacing it with the worked example from C3.

### D2. Figure 2 (`figs/variable_zeroing_flowchart.tex`)
The recursion is drawn as a loop back to the decision node, so a reader cannot tell that each child re-runs the whole procedure. The label "Unresolved R ∈ T?" should read "Another unresolved R ∈ T?".

### D3. Figure 3 (`figs/variable_zeroing_tree.tex`)
- The labels under timed-out nodes (`pph`, `DpDph2`, `xr`, `DpDphh3`) are the chosen splitting variables, but neither the text nor the caption says so, and the names are never defined. They are set in math italic.
- The legend's "no covers" uses a term that appears nowhere in the text. Say "no hitting sets (UNSAT)" instead.
- The caption should open with the main point, for example "Eight zero/nonzero splits resolve the timed-out branch of the QFT system." It should define a node and an edge, point to §4.3 for "contracted", and mark where the single SAT branch is.
- The run takes 36.314 s, while Table 3 reports 39.762 s. Use the same run, or say that this is a repeat run and why the times differ (see A7).
- The sloped "= 0" and "≠ 0" edge labels are small. Check that they are legible in print.

### D4. Table 1
The column "Variables/equations; hitting sets" packs three numbers into slash- and semicolon-separated lists. Split it into separate columns (n, m, #minimal hitting sets), one row per instance.

### D5. Table 2
- The caption needs to define T/F, the 60-s limit, the abbreviations `primdec` (= `primdecGTZ` in Table 7) and `M2` (which Macaulay2 function?), and what "Timeout" means. Partial runs (P in Table 4) show up as "timeout" in the factor-branching column.
- "Baseline" is ambiguous: it can mean the baseline systems or the baseline methods.
- Also see A4 on the per-class best-configuration selection.

### D6. Table 3
- `$timeout/0.005$` puts "timeout" in math italic, so it reads as a product of variables. Use `\text{timeout}` or plain text.
- `d` is undefined (A5).
- Five crowded columns are shrunk by `\resizebox`. Group them under "Factor branching" and "Variable zeroing" headers (`\cmidrule`), with sub-columns T, F, SAT/UNSAT/timeout, splits, and unresolved.
- The caption's limits describe variable zeroing only (A5).

### D7. Font sizes in tables are inconsistent
- `\resizebox{\textwidth}` scales each table by a different factor. Tables 1–3 end up smaller than the caption text, and appendix Tables 4–7 end up **larger than the body text** (pp. 14–17). Use `\small` with `tabular*`, `tabularx`, or `booktabs` at natural width.
- Numbers typed as `$0.008$` come out in the serif math font, while the CAS table text is sans-serif. Use text-mode numbers or `siunitx`.

### D8. Appendix layout and consistency
- The headings A.3 and A.4 (p. 13) have no text under them, their tables float to pp. 16–17, and pp. 14–17 are mostly empty. Add a sentence under each heading and use `[H]`, `\FloatBarrier`, or `\clearpage` sensibly.
- "P marks a partial run and TO an outer timeout" (`appendix.tex:7-8`): both are 60-s timeouts. Define P as "timed out after finding at least one SAT branch". Factor branching has no inner timeout in this experiment, so "outer" is confusing there.
- Table 5 is introduced as "the fastest completed variable zeroing configuration" (`:54-55`), but it includes a partial row and an error row. Its caption says "Selected". Make the two agree.
- **Hitting-set counts disagree.** For positive affine, Table 1 gives 63 / 239 / 833 hitting sets, while Table 5's E gives 4 / 242 / 511 evaluations. §5.2 (`experiments.tex:127-128`) relies on "833 hitting sets". Also, for positive affine (8,6,4), variable zeroing's best time is 0.181 s (Singular), while factor branching with F5 takes 0.008 s. With zero splits the two should be nearly the same. Explain both differences, which probably come from different refinement settings, as in A5.

### D9. Table caption position
Elsevier normally puts table captions **above** the table. The PDF has them below.

---

## E. Prose (line level)

| Location | Issue | Suggestion |
|---|---|---|
| `introduction.tex:20` | "Factorizing Gröbner-basis methods, split on…": stray comma | Remove the comma |
| `introduction.tex:21-22` | "earlier methods also interreduce…": earlier than what? | Name the methods, or cut "earlier" |
| `introduction.tex:12-15` | Claims about resultants have no citation | Cite (e.g., Cox–Little–O'Shea, *Using Algebraic Geometry*) |
| `introduction.tex:60-61` | "whereas standard standalone solvers timed out": tense change, and the solvers aren't named | "whereas every standalone baseline times out within 120 s" |
| `method.tex:14-15` | "Then again do the same on each hitting set, until the subproblems can't be further broken down.": imperative fragment, contraction | "Refinement then recurses on each hitting set until no branch changes." |
| `method.tex:17` | "Here is the pseudocode…" is informal | "Algorithm 2 gives the refinement procedure." |
| `method.tex:192` | "allows the use of Algorithm 3 without loss of correctness" is vague | State what is guaranteed |
| `variable_zeroing.tex:8-9` | One-sentence paragraph | Merge with the next |
| `variable_zeroing.tex:18-20` | "simplify the system for the solver input and work well with Algorithm 2" is vague | Say what "work well" means (e.g., creates new factors) |
| `variable_zeroing.tex:96` | "zero/nonzero" uses a slash | "zero–nonzero splits" or "splits on x = 0 and x ≠ 0" |
| `experiments.tex:13-18` | "three decomposition methods from Singular, Macaulay2…" reads as three from Singular | "two from Singular (`facstd`, `primdecGTZ`), one from Macaulay2 (`…`)" |
| `experiments.tex:22` | "each hitting-set computation": enumeration or solving? | "each branch-solver call" |
| `experiments.tex:36` | "stored in the appendix" | "Appendix A lists the complete measurements." |
| `experiments.tex:118, 126` | Leftover template comments | Delete |
| `experiments.tex:127` | "support this explanation": F shows early SAT, not easier branches | Make the link explicit, or soften it |
| `experiments.tex:172-175` | 45-word sentence | Split it |
| `experiments.tex:164` vs `:142` | "5-second" in one place, "five-second" in the other | Use one style throughout |
| `conclusion.tex:15-16` | "from `facstd` alone" reads as standalone `facstd` | "from the `facstd` backend alone" |
| throughout | "Subsection 2.4" vs "Section 5.3" for subsections | Use one style |
| throughout | "factor branching" / "factor-branching" / "Factor branching" | Hyphenate only when it modifies a noun, and be consistent |

---

## F. References

### F1. Closest related work is missing
These are candidates to **verify before citing**:
- **Factorizing Gröbner bases:**
  - Czapor, "Solving algebraic equations: combining Buchberger's algorithm with multivariate factorization", *JSC* 7 (1989).
  - Melenk, Möller & Neun, "Symbolic solution of large stationary chemical kinetics problems" (1989).
- **Triangular and regular-chain decomposition:**
  - Wang, *JSC* 16 (1993).
  - Kalkbrener, *JSC* 15 (1993).
  - Chen & Moreno Maza, *JSC* 47 (2012).
- **Comprehensive Gröbner systems:**
  - Weispfenning, *JSC* 14 (1992).
  - Kapur, Sun & Wang (ISSAC 2010).
- **Enumerating minimal transversals:**
  - Eiter & Gottlob, *SIAM J. Comput.* 24 (1995).
  - Fredman & Khachiyan, *J. Algorithms* 21 (1996).
- **Rabinowitsch trick:** Rabinowitsch, *Math. Ann.* 102 (1930).
- **CAD (named in the abstract):** Collins (1975).
- **msolve** (Berthomieu, Eder & Safey El Din, ISSAC 2021): a state-of-the-art F4 solver, and a likely referee request as a baseline.
- **Nonlinear SMT:** Jovanović & de Moura, "Solving non-linear arithmetic" (IJCAR 2012). It is relevant because the paper uses SAT/CDCL framing.
- **Numerical irreducible decomposition tooling:** Bertini (Bates et al., 2013), if the numerical baseline is meant to handle positive-dimensional sets.

### F2. `\let\cite\citep` (`paper.tex:25`) makes every citation parenthetical
This produces "Peifer et al. trained … (Peifer et al., 2020)" (`introduction.tex:29-30`, and likewise Kera) and "as in Section 5 of (Sun et al., 2016)" (`method.tex:90-91`). Use `\citet{peifer2020learning} trained…` and `\citep[Sect.~5]{sun2016}`.

### F3. Bibliography casing
`vacca2010asymptotic` prints as "einstein". Protect it: `{Einstein}`.

### F4. Sources to check
- **Katsura:** check that Böge–Gebauer–Kredel (1986) is the right source. The family is usually credited to Katsura (1990) or the PoSSo test suite.
- **Symbolica:** the bibliography entry is dated 2023 but cites version 1.5. Use the year of that version.
- **QFT system:** add the primary physics source for the equations, not only a master's thesis.

---

## G. Build and JSC submission checks

- **LaTeX warnings:**
  - `Overfull \hbox (117pt too wide)` at `\maketitle` (`paper.tex:75`).
  - Font shape `T1/cmss/m/sc` is not available.
  - An empty "ORCID(s):" footnote on p. 1. Add the ORCIDs or remove the footnote.
- **Abstract and keywords:** the abstract has 208 words (limit 250), with no citations and with SAT/UNSAT defined. The 5 keywords are within the allowed 1–7. Highlights are optional; if you supply them, give 3–5 bullets of at most 85 characters each.
- **Research data and code:** JSC's data policy covers code, algorithms, and benchmark generators. Deposit them, then cite and link them, or explain why not.
- **Declarations are missing:** CRediT author contributions, competing interests, funding and acknowledgments, and a generative-AI use declaration if one applies.
- **Leftover drafting comments** in `paper.tex` (abstract), `introduction.tex`, `method.tex`, `variable_zeroing.tex`, and `experiments.tex`. They don't appear in the PDF, but they will reach the editor with the source files.

---

## Suggested order of work
1. Settle the physical-system inconsistencies and report standalone `facstd` (A5). This may change the headline claim.
2. Define `UpdateSaturation`, `FactorBranchingWithTimeout`, contraction, and the budget handling, and check that B1 is sound (B1–B3).
3. Write the comparison with factorizing Gröbner bases and the contribution statement, and fix what the abstract says the output is (A1, A2, C1, C2).
4. Add the fixed-backend comparison, a disclosure of how the benchmark was built, and repeated runs (A4, A7).
5. Add the worked example, the discussion and limitations section, and move the ablation to §5 (C3, C6, C7, A6).
6. Fix the tables, figures, references, and prose (D, E, F, G).
