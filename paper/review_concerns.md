# Review of `paper/paper.tex`

**Criteria:** the `scientific-article-writing` house style, plus the Journal of Symbolic Computation (JSC) requirements. I assumed JSC because the paper uses the Elsevier CAS template and cites mostly JSC work.

---

## Summary

The core idea is clear and worth publishing. The paper interreduces the system and factors every equation. It then takes the minimal hitting sets of the factor sets as branches, recurses, and passes each terminal branch to any exact solver. When a branch times out, it splits on x = 0 versus x ≠ 0. The proofs for Algorithm 2 are mostly correct. Appendix A is a strong feature: it lists every measurement.

Six issues most affect whether a referee would accept the paper:

1. The paper never states how it differs from its closest prior work, factorizing Gröbner bases (Gräbe; Singular's `facstd`) (A1).
2. The abstract promises irreducible components, but the method returns a possibly redundant union of branch varieties (A2).
3. Each method class is represented by its best configuration per system. The fair comparison with a fixed backend is already in the appendix and is stronger, but the paper doesn't show it (A4).
4. The evidence for variable zeroing is one system and one run, and the paper's wording claims more than that supports (A6).
5. Timing method: single runs and a startup cost of about 30 s in the numerical baseline (A7).
6. Several proofs need tightening: undefined terms, unstated assumptions, and an informal induction (B1).

---

## A. Major concerns: argument and scientific accuracy

### A1. Novelty against the closest prior work is not stated
- `introduction.tex:20-22` gives factorizing Gröbner-basis methods one sentence. That sentence says they "preserve the solution set" and "interreduce and prune redundant branches", so it lists what they share with this work, not what separates them.
- Readers must work out the novelty on their own. The likely differences are:
  - (i) splitting over **all** equations at once through minimal hitting sets, instead of one polynomial at a time inside Buchberger's loop;
  - (ii) factoring only between solver calls, so any solver can be plugged in;
  - (iii) recursive refinement with a termination proof;
  - (iv) variable zeroing when a branch times out.
- The gap is also hard to see because `facstd` is both a baseline and the only branch solver that completes the physical system without splits (Table 3). FB+`facstd` stacks this framework on top of a factorizing Gröbner basis, and the paper doesn't say what the outer layer contributes.
- **Fix:** add a paragraph comparing the method to Gräbe's factorized Gröbner bases and `facstd`. Say what it improves, under which conditions, and at what cost; for example, the number of hitting sets can grow exponentially. Put this comparison in the introduction's contribution statement (C2) and repeat it in the conclusion.
- Also mention the related lines of work a JSC referee will expect: triangular and regular-chain decompositions ("initial = 0 or ≠ 0"), comprehensive Gröbner systems (parameter conditions = 0 / ≠ 0), and nonlinear SMT (the paper borrows SAT/UNSAT terminology). See F1 for candidate references.

### A2. What the method outputs versus what the abstract promises
- The abstract (`paper.tex:79`) and introduction (`introduction.tex:7-9`) motivate the work with the dimension and the **irreducible components** of the solution set.
- What the framework actually returns is a union of branch varieties, one exact representation per terminal hitting set. Branches can overlap or be contained in one another. They are not irreducible, and the nonzero branches are locally closed, not closed.
- **Fix:** state exactly what the output is, for example "a finite list of pairs (Gröbner basis, nonzero conditions) whose varieties cover V(P)". Say which global properties can be read off it directly (dimension, as the maximum over branches). Either scale back the claim about components or describe the post-processing step.

### A3. The complexity argument does not support the method, and the authors' own comment says so
- `introduction.tex:45` holds the TODO "Needs a better justification…".
- `introduction.tex:46` and `method.tex:8` cite Mayr–Meyer's doubly exponential bound in the **number of variables**. Factor branching keeps every variable; only zeroing removes any. So this bound doesn't explain why the branches are easier.
- **Fix:** argue in terms of degree, component count, or structure instead. Factors have lower degree, so degree-based bounds on Gröbner-basis computation drop, and each branch ideal has fewer components. Alternatively, drop the complexity argument and rest the claim on the experiments.
- The cost the method adds is not analyzed. The number of minimal hitting sets can be exponential; it is at most ∏|Cᵢ|. The paper also never names the enumeration algorithm used. State what the implementation does and cite the minimal-transversal enumeration literature (Berge; Eiter–Gottlob; Fredman–Khachiyan).

### A4. Comparison method
- **Benchmark selection.** Every family is built from products of factors, and gating (multiplying every equation by t) makes V = {t = 0} ∪ V(original), which is trivial for any method that factors. **Fix:** say plainly that the families were designed to have factor structure (see also C7).
- **Each class is scored by its best configuration per system.** Table 2 picks the fastest of 5 factor-branching backends against the fastest of 6 direct configurations, 3 decomposition methods, and 1 numerical method. Classes with more configurations gain from this selection.
- **A fairer comparison is already in the appendix.** With one fixed backend, FB+Singular (Table 4, last column) completes **19/20**. Standalone `facstd` completes 17/20 (Table 7) and Singular's direct Gröbner basis 7/20 (Table 6). FB+Singular is faster than `facstd` on 16 of the 19 systems both can compare. It is slower on positive affine (8,6,4), positive affine (10,8,5), and square affine (10,5).
  - **Fix:** put this head-to-head in the main text. It is a stronger and fairer claim than "#fastest (ties counted)". One caveat: FB+Singular runs a persistent worker, so check that per-process startup is counted the same way on both sides.
- **The methods compute different things.** A Gröbner basis of the whole ideal, a primary decomposition (`primdecGTZ`), a factorized Gröbner basis, a numerical solution set, and a union of branches are different outputs. Wall times across them are not directly comparable.
  - **Fix:** list what each baseline computes, including its Singular or Macaulay2 command, monomial order, and characteristic. Table 2's abbreviations `primdec` (= `primdecGTZ`) and `M2` (which function?) should come from this list. For HomotopyContinuation.jl, name the function: the gated systems are positive-dimensional, so if it was `solve`, it returns only isolated roots and is not comparable.

### A6. The evidence for variable zeroing is thin, and the wording claims more (`variable_zeroing.tex:25, 108-116`; `introduction.tex:53`)
- **Scope.** On the 20 baseline systems, variable zeroing never helped. Every row has 0 splits except Cyclic-7, which stayed partial, and dense n=7, which failed with an error (Table 5). Its only positive result is two extra backends completing on one physical system. The best configuration there, FB+`facstd`, needs no splits.
  - **Fix:** frame the result as "makes completion less dependent on which backend is used, on this system". Remove "proves particularly effective" (`introduction.tex:53`).
- **The ablation is in the wrong section and under-reported.** The x = 1 and random-a comparisons sit in §4.1, before the experimental setup, and give no backend, per-hitting-set timeout, seed, or numbers. Fig. 3 likewise sits in §4 but shows the result of §5.3. "Substantially more effective … on these structured systems" rests on a single system with one run per variant.
  - **Fix:** move the comparison and Fig. 3 to §5.3, report the comparison in a table, restrict the claim to that one system, and keep §4 limited to the method and theory.
- **"A variable is often always zero or always nonzero" (`variable_zeroing.tex:25`)** is imprecise. On any irreducible component, a variable either vanishes identically or is nonzero on a dense open subset. That holds for every system, so it cannot be what makes zeroing work here.
  - **Fix:** state the actual mechanism. Setting x = 0 deletes every monomial containing x, which can make generators factor. Setting x ≠ 0 saturates by x, which removes the components lying in {x = 0} and lets interreduction prune any hitting set that contains x.

### A7. Measurement method (`experiments.tex:20-36`, appendix)
- **The numerical baseline has a startup cost of about 30 s.** HomotopyContinuation.jl takes 29–37 s even on n = 5 systems, which points to Julia startup and compilation. Say whether this startup time is included, and apply the same rule to Singular and Macaulay2 process startup. The in-process Rust backends pay no startup cost.
- **Threads.** The hardware has 14 cores, but the paper never says whether backends run on one thread or many, or whether factor branching solves branches in parallel. Macaulay2 and Julia may use several threads by default.
- **The inner timeout was tuned on the test systems.** The 3-s value was chosen on the same hard gated systems it is evaluated on (`experiments.tex:22-26`). Acknowledge this.
- **What F measures.** F is measured from the start of the hitting-set search, so it excludes factorization. State why F is of interest (how soon the first solution appears) and that the baselines have no comparable number.

---

## B. Correctness and precision of the theory

### B1. Proof fixes (single task)

**The set A in Algorithm 2** (`method.tex:23, 31`)
- A holds "normalized systems visited on the current recursion path", but "normalized" is never defined. The termination proof (Theorem 2) never uses A: it shows that M strictly increases whenever H ≠ Q̃.
- **Fix:** explain what A adds, such as pruning repeated systems early, or remove it. Define the normalization used when comparing H with Q̃: monic, sorted, as sets.

**Theorems 2–4**
- Open the proof of Theorem 2 with a one-sentence strategy: the invariant M strictly increases whenever refinement changes the system, and why that forces termination.
- "Sound and complete" (`method.tex:193`) is used but never defined. Define it: every represented point is a solution, and every solution is represented.
- Theorem 3 assumes that 𝓑 returns on every branch. Algorithm 3 has no path for a call that never returns, so state this assumption in the theorem.
- Theorem 4 then follows immediately and can be a corollary or merged into Theorem 3.
- Lemma 1's hypothesis ("Suppose refinement … terminates", `method.tex:127`) is redundant once Theorem 2 is proved. Put Theorem 2 first and state Lemma 1 without the hypothesis.
- The proof of Theorem 3 repeats the reasoning of Lemma 1 (`method.tex:200-202`). Cite the lemma instead.

**Theorem 5** (`variable_zeroing.tex:164-178`)
- "Qᵢ encodes Q together with Dᵢ" is informal. Define it as V_Y(Qᵢ; A∪{Dᵢ}) = {a ∈ V_Y(Q; A) : a satisfies Dᵢ}.
- For the x = 0 child, Q|_{x=0} lives in fewer variables. Say that the projection onto Y puts x = 0 back.

**Equation (4) and Theorems 6–7** (`variable_zeroing.tex:189-230`)
- Equation (4) cites the proof of Theorem 3, which assumes 𝓑 returns on every branch. Say that T holds both timed-out hitting sets and those left unrun when β runs out, and that the partition covers both.
- "Applying this argument recursively proves the result" (`:213`) should be an explicit induction on the finite recursion tree, like the proof of Lemma 1.
- "Returns after τ" (`:225`) should be "returns within time τ, with either a result or a timeout".

**Edge case**
- A constant generator has an empty factor set, so it has no hitting sets and the branch closes as UNSAT during refinement. This is how interreduction to {1} prunes hitting sets under the nonzero encoding; one sentence would make it explicit.

### B11. Notation clashes
- **R** is the ring K[x] in §2, the output of Refine in Algorithm 3, and a timed-out branch in Algorithm 4.
- **T** is the terminal set (Algorithm 2), the timed-out set (Algorithm 4), and end-to-end time (§5).
- **F** is a factor set Fᵢ (proof of Theorem 1) and time to first SAT (§5). Also, Cᵢ in §2.3 and Fᵢ in Theorem 1 name the same object.
- 𝓗(P) is defined in §2.3 and again, as 𝓗(Q), at `method.tex:102`.
- The same object is called a "hitting set", "branch", "subproblem", and "cover" (Fig. 3 legend: "no covers").
- **Fix:** separate the combinatorial hitting set H from its branch system. Use one word for the branch throughout, and use "cover" nowhere.

---

## C. Structure and exposition

### C1. Abstract
- At about 288 words, it exceeds JSC's 250-word limit.
- `paper.tex:83` "variants of Buchberger's algorithm with syzygy, resultants, and cylindrical algebraic decomposition": "with syzygy" is unclear (syzygy criteria? signatures?). CAD doesn't come up anywhere else in the paper. Meanwhile numerical homotopy, which is a baseline, is missing. List the methods the paper actually compares against.
- `:85` "even the most modern approaches still often fail on practical problems or slight perturbations of standard benchmarks" has no support. The gated results do support it, so connect the two: direct Gröbner bases fail on gated Katsura-8.
- There are no numbers. Add the main results, for example "completes 19 of 20 benchmarks with a single backend, versus 17 for `facstd`" and the physical-system result.
- `:91` "no standard method in our comparison succeeds": give the time limit (120 s).
- `:93` The closing sentence ("offer a path toward…") is vague. Replace it with a concrete outlook.

### C2. Introduction
- The gap (`introduction.tex:35`) is a one-sentence paragraph: "large systems can remain inaccessible". Name the structure the paper exploits, which is equations that are products of shared factors, as in fixed-point equations from physics. Then say why existing methods struggle with it: direct Gröbner bases expand the products, `facstd` splits one polynomial at a time, and numerical methods can't certify completeness.
- The machine-learning paragraph (`:28-32`) doesn't connect to the gap. Either link it (those methods could serve as branch solvers) or cut it.
- Put the contribution statement (A1) with key numbers before the method description. At present, `:56-61` announces the physical-system result but no baseline result.

### C3. Add a small worked example (the highest-value addition)
There is no example anywhere in the paper. A 3–4 equation system would help readers far more than Figs. 1–2. It should show: the factor sets, the minimal hitting sets, one refinement step where interreduction creates new factors, a terminal branch, and one zero/nonzero split. Reuse the same example for Theorem 1 and for §4.3.

### C4. Unused or misplaced material
- §3.2 (UNSAT-core learning, `method.tex:83`) reports a negative result with no data, and the UNSAT-core definitions (`preliminaries.tex:85-87`) serve only it. Reduce §3.2 to one sentence in the discussion and remove the definitions, or give the numbers in the appendix.
- Name the nonzero encoding of §2.4 as the Rabinowitsch trick and cite it.
- The Berge citation (`preliminaries.tex:103`) is attached to the branch claim. It belongs with the definition of a hitting set (minimal transversal).

### C7. No discussion or limitations section
The conclusion (`conclusion.tex`) is 20 lines and just restates results. Add a discussion covering:
- when the method helps: factor structure, and branches much easier than the whole system;
- when it doesn't: Cyclic-7, and the exponential number of hitting sets;
- that the output may be redundant (A2);
- the dependence on the backend;
- that the benchmark was designed for the method (A4);
- that there is one physical system and one run per result;
- the most important piece of future work.

"Future work will test this benefit on other systems" is too vague.

### C8. Implementation details a reader needs to reproduce the work
- The factorization routine, over ℚ using Symbolica.
- The monomial order.
- Whether interreduction is on in each experiment, including the physical system.
- Which implementations are the authors' own (apparently M4GB, F5, and GVW) and which come from libraries. The results file labels the "F4" backend `symbolica`. A slow custom implementation affects the "direct GB" class.

### C9. Reproducibility of the benchmarks
- The families are not defined well enough to regenerate: what is a "signed-distance cycle" or a "positive affine hypergraph", and what do (n,p,k) and (n,m,k) mean?
- The bullet list at `experiments.tex:42-47` is not parallel. The first bullet covers two families, and the last bullet describes gating rather than a family.
- **Fix:** give defining formulas, or point to the generator script in the repository from the data availability statement.

---

## D. Figures and tables

### D1. Figure 1 (`figs/branching_flowchart.tex`)
- The first step ("Initialize") is on the right and points left to "Refine", so the diagram reads right to left.
- There is no explicit "more hitting sets?" decision. The single rail from SAT/UNSAT splits into "done" and "next H" without saying what decides between them.
- The text is in saturated pure blue, red, green, and violet. Use muted colors, and check Figs. 1–3 for readers with impaired color vision (a JSC requirement).
- The figure repeats the 9 lines of Algorithm 3 and adds little; the worked example (C3) could replace it.

### D2. Figure 2 (`figs/variable_zeroing_flowchart.tex`)
The recursion is drawn as a loop back to the decision node, so a reader cannot tell that each child re-runs the whole procedure. The label "Unresolved R ∈ T?" should read "Another unresolved R ∈ T?".

### D3. Figure 3 (`figs/variable_zeroing_tree.tex`)
- The labels under timed-out nodes (`pph`, `DpDph2`, `DpDph1`, `xr`) are the chosen splitting variables, but neither the text nor the caption says so, and the names are never defined. They are set in `\mathit`.
- The legend's "no covers" uses a term that appears nowhere in the text. Say "no hitting sets (UNSAT)" instead.
- The caption should open with the main point, for example "Eight zero/nonzero splits resolve the timed-out branch of the QFT system." It should define a node and an edge and mark where the single SAT branch is.
- The caption gives 33.768 s, while Table 3 reports 34.053 s for M4GB with variable zeroing. Use the same run, or say that this is a repeat run.

### D4. Table 1
The column "Variables/equations; hitting sets" packs three numbers into slash- and semicolon-separated lists. Split it into separate columns (n, m, #minimal hitting sets), one row per instance.

### D5. Table 2
- The caption needs to define T/F, the 60-s limit, and what "Timeout" means. Partial runs (P in Table 4) show up as "timeout" in the factor-branching column. For `primdec` and `M2`, see A4.
- "Baseline" is ambiguous: it can mean the baseline systems or the baseline methods.

### D6. Table 3
`$timeout/0.005$` puts "timeout" in math italic, so it reads as a product of variables. Use `\text{timeout}` or plain text.

### D7. Font sizes in tables are inconsistent
`\resizebox{\textwidth}` scales each table by a different factor. Tables 1–3 end up smaller than the caption text, and appendix Tables 4–7 end up **larger than the body text**. Use `\small` with `tabular*`, `tabularx`, or `booktabs` at natural width.

### D8. Appendix layout and consistency
- The headings A.3 and A.4 have no text under them, and their tables float away from them. Add a sentence under each heading and use `[H]`, `\FloatBarrier`, or `\clearpage` sensibly.
- "P marks a partial run and TO an outer timeout" (`appendix.tex:7-8`): both are 60-s timeouts. Define P as "timed out after finding at least one SAT branch". Factor branching has no inner timeout in this experiment, so "outer" is confusing there.
- Table 5 is introduced as "the fastest completed variable zeroing configuration" (`:54-55`), but it includes a partial row and an error row. Its caption says "Selected". Make the two agree.
- **Hitting-set counts disagree.** For positive affine, Table 1 gives 63 / 239 / 833 hitting sets, while Table 5's E gives 4 / 242 / 511 evaluations. §5.2 (`experiments.tex:128`) relies on "833 hitting sets". Also, for positive affine (8,6,4), variable zeroing's best time is 0.181 s (Singular), while factor branching with F5 takes 0.008 s. With zero splits the two should be nearly the same. Explain both differences.

### D9. Table caption position
Elsevier normally puts table captions **above** the table. The PDF has them below.

---

## E. Prose (line level)

| Location | Issue | Suggestion |
|---|---|---|
| `introduction.tex:20` | "Factorizing Gröbner-basis methods, split on…": stray comma | Remove the comma |
| `introduction.tex:21` | "earlier methods also interreduce…": earlier than what? | Name the methods, or cut "earlier" |
| `introduction.tex:61` | "whereas standard standalone solvers timed out": tense change, and the solvers aren't named | "whereas every standalone baseline times out within 120 s" |
| `method.tex:14-15` | "Then again do the same on each hitting set, until the subproblems can't be further broken down.": imperative fragment, contraction | "Refinement then recurses on each hitting set until no branch changes." |
| `method.tex:17` | "Here is the pseudocode…" is informal | "Algorithm 2 gives the refinement procedure." |
| `method.tex:190` | "allows the use of Algorithm 3 without loss of correctness" is vague | State what is guaranteed |
| `variable_zeroing.tex:8-9` | One-sentence paragraph | Merge with the next |
| `variable_zeroing.tex:18-20` | "simplify the system for the solver input and work well with Algorithm 2" is vague | Say what "work well" means (e.g., creates new factors) |
| `experiments.tex:15` | "three decomposition methods from Singular, Macaulay2…" reads as three from Singular | "two from Singular (`facstd`, `primdecGTZ`), one from Macaulay2 (`…`)" |
| `experiments.tex:22` | "each hitting-set computation": enumeration or solving? | "each branch-solver call" |
| `experiments.tex:36` | "stored in the appendix" | "Appendix A lists the complete measurements." |
| `experiments.tex:127` | "support this explanation": F shows early SAT, not easier branches | Make the link explicit, or soften it |
| `experiments.tex:143` vs `:165` | "five-second" in one place, "5-second" in the other | Use one style throughout |
| `conclusion.tex:15-16` | "from `facstd` alone" reads as standalone `facstd` | "from the `facstd` backend alone" |
| throughout | "Subsection 2.4" vs "Section 5.3" for subsections | Use one style |
| throughout | "factor branching" / "factor-branching" / "Factor branching" | Hyphenate only when it modifies a noun, and be consistent |

---

## F. References

### F1. Closest related work is missing
These are candidates to **verify before citing** (see A1, A3, C4):
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
- **Nonlinear SMT:** Jovanović & de Moura, "Solving non-linear arithmetic" (IJCAR 2012).

### F2. `\let\cite\citep` (`paper.tex:25`) makes every citation parenthetical
This produces "Peifer et al. trained … (Peifer et al., 2020)" (`introduction.tex:29-30`, and likewise Kera) and "as in Section 5 of (Sun et al., 2016)" (`method.tex:90-91`). Use `\citet{peifer2020learning} trained…` and `\citep[Sect.~5]{sun2016}`.

### F3. Bibliography casing
`vacca2010asymptotic` prints "Einstein" in lowercase. Protect it: `{Einstein}`.

### F4. Sources to check
- **Katsura:** check that Böge–Gebauer–Kredel (1986) is the right source. The family is usually credited to Katsura (1990) or the PoSSo test suite.
- **Symbolica:** the bibliography entry is dated 2023 but cites version 1.5. Use the year of that version.
- **HomotopyContinuation.jl:** `experiments.tex:16` gives versions for Singular, Macaulay2, and Julia but not for HomotopyContinuation.jl. JSC asks for software versions.
- **DOIs:** 8 of the 28 entries in `references.bib` have none. Add them where they exist.

---

## Suggested order of work
1. Fix the proofs (B1).
2. Write the comparison with factorizing Gröbner bases and the contribution statement, and fix what the abstract says the output is (A1, A2, C1, C2).
3. Add the fixed-backend comparison, the benchmark-design disclosure, and the measurement details (A4, A7).
4. Add the worked example and the discussion section, and move the ablation to §5 (C3, C7, A6).
5. Fix the tables, figures, references, and prose (D, E, F).
