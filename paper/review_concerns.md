# Review of `paper/paper.tex`

**Criteria:** the `scientific-article-writing` house style, plus the Journal of Symbolic Computation (JSC) requirements. I assumed JSC because the paper uses the Elsevier CAS template and cites mostly JSC work.

---

## Summary

The core idea is clear and worth publishing: factor every equation, branch on the minimal hitting sets of the factor sets, recurse, and pass each terminal branch to any exact solver, splitting on x = 0 versus x ≠ 0 when a branch times out. The proofs for Algorithm 2 are mostly correct, and Appendix A lists every measurement.

Five issues most affect acceptance:

1. No stated difference from factorizing Gröbner bases (Gräbe; Singular's `facstd`) (A1).
2. The abstract promises irreducible components; the method returns a possibly redundant union of branch varieties (A2).
3. The main comparison uses each class's best configuration per system; the fairer fixed-backend comparison sits unused in the appendix (A4).
4. Variable zeroing rests on one system and one run, and the wording overclaims (A6).
5. Timing: single runs, and how process startup is counted (A7).

---

## A. Major concerns: argument and scientific accuracy

### A1. Novelty against the closest prior work is not stated
- Better describe what makes this work novel in comparison to factorizing Gröbner-basis methods.
- **Fix:** add a paragraph comparing the method to Gräbe's factorized Gröbner bases and `facstd`: what improves, when, and at what cost (e.g., exponentially many hitting sets). Also mention triangular/regular-chain decompositions, comprehensive Gröbner systems, and nonlinear SMT (see F1).

### A2. What the method outputs versus what the abstract promises
- State exactly what the output is, and how it depends on the branch solver.

### A3. The complexity argument does not support the method
- `introduction.tex:46` and `method.tex:8` cite Mayr–Meyer's doubly exponential bound in the **number of variables**, but factor branching keeps every variable, so the bound doesn't explain why branches are easier (`introduction.tex:45` TODO).
- The added cost is not analyzed: the number of minimal hitting sets can reach ∏|Cᵢ|, and the enumeration algorithm is never named. **Fix:** state what the implementation does (Berge multiplication, `symbolica/cover.rs`) and cite the minimal-transversal enumeration literature (Berge; Eiter–Gottlob; Fredman–Khachiyan — the latter two are already in `references.bib`).

### A4. Comparison method
- **Benchmark selection.** Every family is built from products of factors, and gating makes V = {t = 0} ∪ V(original), which is trivial for any method that factors. **Fix:** say plainly that the families were designed to have factor structure.
- **Best configuration per class.** Table 2 pits the fastest of 5 FB backends against the fastest of 6 direct configurations and 3 decomposition methods; classes with more configurations gain from the selection.
- **A fairer comparison is already in the appendix.** With one fixed backend, FB+Singular completes 18/20 (Table 4), versus 15/20 for `facstd` (Table 7) and 6/20 for Singular's direct GB (Table 6). FB+Singular beats `facstd` on 10 of the 15 systems both complete.
  - **Fix:** put this head-to-head in the main text. Process startup is counted on both sides: FB+Singular starts a new Singular process per terminal branch, `facstd` once per system.
- **The methods compute different outputs** (GB, primary decomposition, factorized GB, union of branches), so wall times are not directly comparable.
  - **Fix:** list what each baseline computes, with its command, monomial order, and characteristic. From `scripts/baselines/`: M4GB/F5/GVW/F4 grevlex; Singular `std` and Macaulay2 `gb` lex; `facstd` after `interred` in `dp`; `primdecGTZ` and M2 `decompose` lex; all over ℚ.

### A6. The evidence for variable zeroing is thin, and the wording claims more (`variable_zeroing.tex:108-116`; `introduction.tex:55`)
- **Scope.** On the 20 baseline systems zeroing never helped (0 splits everywhere; Cyclic-7 partial, dense n=7 partial; Table 5). Its only positive result is two extra backends completing one physical system, where the best configuration, FB+`facstd`, needs no splits.
  - **Fix:** frame the result as "makes completion less dependent on the backend, on this system", and remove "proves particularly effective" (`introduction.tex:55`).
- **Ablation misplaced and under-reported.** The x = 1 versus random-a comparison sits in §4.1 with no backend, timeout, seed, or numbers, and Fig. 3 sits in §4 but shows a §5.3 result. "Substantially more effective" rests on one system and one run per variant.
  - **Fix:** move both to §5.3, report the comparison in a table, and restrict the claim to that system. Data in `results/split_value_comparison.json`: M4GB, 5 s per branch, 600 s total, seed 0; x = 1 → 89 splits, 36 unresolved; random a → 85 splits, 50 unresolved.

### A7. Measurement method (`experiments.tex:20-36`, appendix)
- **Startup cost.** Baseline times are process wall time, so they include startup; FB+Singular starts a Singular process per branch. Say so.
- **Threads.** Say whether backends and branch solving use one thread or many. Runs were sequential (`--jobs 1`) with each tool's default thread count.
- **Timeout tuning.** The 3 s inner timeout was tuned on the same systems it is evaluated on (`experiments.tex:22-26`). Acknowledge this.
- **F.** F excludes factorization. State why it is of interest and that the baselines have no comparable number.

---

## C. Structure and exposition

### C1. Abstract
- At about 288 words it exceeds JSC's 250-word limit; cut, don't add.
- `paper.tex:86`: "even the most modern approaches still often fail…" is unsupported in the abstract. Cut it.
- `:94`: the closing sentence ("offer a path toward…") is vague. Cut it.
- Drop the irreducible-components claim (A2).

### C2. Introduction
- The contribution statement (`:58-63`) gives no baseline result; add key numbers (A1).

### C3. Add a small worked example (highest-value addition)
The paper has no example. A 3–4 equation system should show the factor sets, minimal hitting sets, one refinement step that creates new factors, a terminal branch, and one zero/nonzero split. Reuse it for Theorem 1 and §4.3.

### C4. Unused or misplaced material
- §3.2 (UNSAT-core learning, `method.tex:81`) reports a negative result with no data; the UNSAT-core definitions (`preliminaries.tex:85-87`) serve only it.

### C7. No discussion or limitations section
The 20-line conclusion only restates results. Add a discussion covering when the method helps (factor structure) and when it doesn't (Cyclic-7, exponential hitting sets), redundant output (A2), backend dependence, benchmark design (A4), one physical system and one run per result, and the most important future work. "Future work will test this benefit on other systems" is too vague.

### C8. Implementation details needed to reproduce the work
- Factorization routine (Symbolica, over ℚ) and monomial order (interreduction in lex, branch GBs in grevlex).
- Whether interreduction is on in each experiment, including the physical system (it is: no script passes `--no-interreduce`).
- Which backends are the authors' own (M4GB, F5, GVW) and which are library code; the "F4" backend is Symbolica's built-in routine, labelled `symbolica` in the results.

### C9. Reproducibility of the benchmarks
- The families cannot be regenerated from the text: "signed-distance cycle", "positive affine hypergraph", and the parameters (n,p,k), (n,m,k) are undefined (definitions in `scripts/benchmark_idos.py`).

---

## D. Figures and tables

### D1. Figure 1 (`figs/branching_flowchart.tex`)
- Check Figs. 1–3 for color-vision accessibility (JSC requirement).

### D2. Figure 2 (`figs/variable_zeroing_flowchart.tex`)
The recursion is drawn as a loop, hiding that each child re-runs the whole procedure. Relabel "Unresolved R ∈ T?" as "Another unresolved R ∈ T?".

### D3. Figure 3 (`figs/variable_zeroing_tree.tex`)
- Labels under timed-out nodes (`pph`, `DpDph2`, …) are splitting variables, but this is never stated and the names are undefined.
- Define node and edge in the caption, and mark the SAT branch.
- Caption gives 33.113 s, Table 3 gives 24.611 s for the same configuration: the figure is a separate run (source JSON `results/physical_system_m4gb_5s_figure_run.json` is missing). Reconcile, ideally by regenerating the figure from the Table 3 run.

### D5. Table 2
- Define T/F, the 60 s limit, and "Timeout" in the caption; partial runs (P in Table 4) appear as "timeout".
- "Baseline" is ambiguous (systems or methods).

### D6. Table 3
`$timeout/0.004$` sets "timeout" in math italic. Use `timeout$/0.004$`.

### D7. Inconsistent table font sizes
`\resizebox{\textwidth}` still scales Tables 1–3 (main text) differently from the appendix tables. Use `\small` at natural width, as in the appendix.

### D8. Appendix consistency
- Positive affine (8,6,4) takes 0.543 s with zeroing (0 splits) but 0.177 s with FB+F5. Explain the overhead.

### D9. Table caption position
Elsevier puts table captions **above** the table; Tables 1–3 in the main text still have them below.

---

## Suggested order of work
1. Comparison with factorizing Gröbner bases, contribution statement, and abstract cuts (A1, A2, C1, C2).
2. Fixed-backend comparison, benchmark-design disclosure, measurement details (A4, A7).
3. Worked example, discussion section, ablation moved to §5 (C3, C7, A6).
4. Tables (D)
