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
5. Timing: single runs, and a ~30 s startup cost in the numerical baseline (A7).

---

## A. Major concerns: argument and scientific accuracy

### A1. Novelty against the closest prior work is not stated
- Better describe what makes this work novel in comparison to factorizing Gröbner-basis methods.
- **Fix:** add a paragraph comparing the method to Gräbe's factorized Gröbner bases and `facstd`: what improves, when, and at what cost (e.g., exponentially many hitting sets). Also mention triangular/regular-chain decompositions, comprehensive Gröbner systems, and nonlinear SMT (see F1).

### A2. What the method outputs versus what the abstract promises
- The output is a union of branch varieties that may overlap or contain one another; they are not irreducible, and the nonzero branches are only locally closed.
- **Fix:** state exactly what the output is, and how it depends on the branch solver.

### A3. The complexity argument does not support the method (the authors' own TODO says so)
- `introduction.tex:46` and `method.tex:8` cite Mayr–Meyer's doubly exponential bound in the **number of variables**, but factor branching keeps every variable, so the bound doesn't explain why branches are easier (`introduction.tex:45` TODO).
- The added cost is not analyzed: the number of minimal hitting sets can reach ∏|Cᵢ|, and the enumeration algorithm is never named. **Fix:** state what the implementation does and cite the minimal-transversal enumeration literature (Berge; Eiter–Gottlob; Fredman–Khachiyan).

### A4. Comparison method
- **Benchmark selection.** Every family is built from products of factors, and gating makes V = {t = 0} ∪ V(original), which is trivial for any method that factors. **Fix:** say plainly that the families were designed to have factor structure.
- **Best configuration per class.** Table 2 pits the fastest of 5 FB backends against the fastest of 6 direct configurations, 3 decomposition methods, and 1 numerical method; classes with more configurations gain from the selection.
- **A fairer comparison is already in the appendix.** With one fixed backend, FB+Singular completes 19/20 (Table 4), versus 17/20 for `facstd` (Table 7) and 7/20 for Singular's direct GB (Table 6). FB+Singular beats `facstd` on 16 of 19 comparable systems.
  - **Fix:** put this head-to-head in the main text, checking that process startup is counted the same way on both sides (FB+Singular uses a persistent worker).
- **The methods compute different outputs** (GB, primary decomposition, factorized GB, numerical solutions, union of branches), so wall times are not directly comparable.
  - **Fix:** list what each baseline computes, with its command, monomial order, and characteristic, and name the Macaulay2 and HomotopyContinuation.jl functions. If the latter was `solve`, it returns only isolated roots of these positive-dimensional systems.

### A6. The evidence for variable zeroing is thin, and the wording claims more (`variable_zeroing.tex:25, 108-116`; `introduction.tex:53`)
- **Scope.** On the 20 baseline systems zeroing never helped (0 splits everywhere; Cyclic-7 partial, dense n=7 error; Table 5). Its only positive result is two extra backends completing one physical system, where the best configuration, FB+`facstd`, needs no splits.
  - **Fix:** frame the result as "makes completion less dependent on the backend, on this system", and remove "proves particularly effective" (`introduction.tex:53`).
- **Ablation misplaced and under-reported.** The x = 1 versus random-a comparison sits in §4.1 with no backend, timeout, seed, or numbers, and Fig. 3 sits in §4 but shows a §5.3 result. "Substantially more effective" rests on one system and one run per variant.
  - **Fix:** move both to §5.3, report the comparison in a table, and restrict the claim to that system.
- **"A variable is often always zero or always nonzero" (`variable_zeroing.tex:25`)** holds on every irreducible component of every system, so it cannot explain why zeroing works here.
  - **Fix:** state the actual mechanism (x = 0 deletes monomials, which can make generators factor; x ≠ 0 saturates away components in {x = 0}).

### A7. Measurement method (`experiments.tex:20-36`, appendix)
- **Startup cost.** HomotopyContinuation.jl takes 29–37 s even on n = 5, pointing to Julia startup. Say whether startup is included, and apply the same rule to Singular and Macaulay2.
- **Threads.** Say whether backends and branch solving use one thread or many (14-core machine; Macaulay2 and Julia may multithread by default).
- **Timeout tuning.** The 3 s inner timeout was tuned on the same systems it is evaluated on (`experiments.tex:22-26`). Acknowledge this.
- **F.** F excludes factorization. State why it is of interest and that the baselines have no comparable number.

---

## C. Structure and exposition

### C1. Abstract
- At about 288 words it exceeds JSC's 250-word limit; cut, don't add.
- `paper.tex:83`: "with syzygy" is unclear and CAD appears nowhere else in the paper. Cut both.
- `:85`: "even the most modern approaches still often fail…" is unsupported in the abstract. Cut it.
- `:93`: the closing sentence ("offer a path toward…") is vague. Cut it.
- Drop the irreducible-components claim (A2).

### C2. Introduction
- The gap (`introduction.tex:35`) is one vague sentence; name specific examples.
- The machine-learning paragraph (`:28-32`) doesn't connect to the gap.
- The contribution statement (`:56-61`) gives no baseline result; add key numbers (A1).

### C3. Add a small worked example (highest-value addition)
The paper has no example. A 3–4 equation system should show the factor sets, minimal hitting sets, one refinement step that creates new factors, a terminal branch, and one zero/nonzero split. Reuse it for Theorem 1 and §4.3.

### C4. Unused or misplaced material
- §3.2 (UNSAT-core learning, `method.tex:83`) reports a negative result with no data; the UNSAT-core definitions (`preliminaries.tex:85-87`) serve only it.
- Name the nonzero encoding of §2.4 as the Rabinowitsch trick and cite it.

### C7. No discussion or limitations section
The 20-line conclusion only restates results. Add a discussion covering when the method helps (factor structure) and when it doesn't (Cyclic-7, exponential hitting sets), redundant output (A2), backend dependence, benchmark design (A4), one physical system and one run per result, and the most important future work. "Future work will test this benefit on other systems" is too vague.

### C8. Implementation details needed to reproduce the work
- Factorization routine (over ℚ, Symbolica) and monomial order.
- Whether interreduction is on in each experiment, including the physical system.
- Which backends are the authors' own (apparently M4GB, F5, GVW) and which are library code; the "F4" backend is labelled `symbolica` in the results.

### C9. Reproducibility of the benchmarks
- The families cannot be regenerated from the text: "signed-distance cycle", "positive affine hypergraph", and the parameters (n,p,k), (n,m,k) are undefined.
---

## D. Figures and tables

### D1. Figure 1 (`figs/branching_flowchart.tex`)
- "Initialize" is on the right and points left, so the diagram reads right to left.
- No explicit "more hitting sets?" decision between "done" and "next H".
- Check Figs. 1–3 for color-vision accessibility (JSC requirement).

### D2. Figure 2 (`figs/variable_zeroing_flowchart.tex`)
The recursion is drawn as a loop, hiding that each child re-runs the whole procedure. Relabel "Unresolved R ∈ T?" as "Another unresolved R ∈ T?".

### D3. Figure 3 (`figs/variable_zeroing_tree.tex`)
- Labels under timed-out nodes (`pph`, `DpDph2`, …) are splitting variables, but this is never stated and the names are undefined.
- Legend "no covers" → "no hitting sets (UNSAT)".
- Define node and edge in the caption, and mark the SAT branch.
- Caption gives 33.768 s, Table 3 gives 34.053 s. Reconcile.

### D5. Table 2
- Define T/F, the 60 s limit, and "Timeout" in the caption; partial runs (P in Table 4) appear as "timeout".
- "Baseline" is ambiguous (systems or methods).

### D6. Table 3
`$timeout/0.005$` sets "timeout" in math italic. Use `\text{timeout}`.

### D7. Inconsistent table font sizes
`\resizebox{\textwidth}` scales each table differently: Tables 1–3 are smaller than the captions, Tables 4–7 larger than the body text. Use `\small` at natural width.

### D8. Appendix layout and consistency
- A.3 and A.4 have no text and their tables float away. Add a sentence under each and control float placement.
- "P marks a partial run and TO an outer timeout" (`appendix.tex:7-8`): both are 60 s timeouts. Define P as "timed out after at least one SAT branch".
- Table 5 is introduced as "the fastest completed" configuration (`:54-55`) but includes partial and error rows, and its caption says "Selected". Make them agree.
- **Hitting-set counts disagree.** Positive affine: Table 1 gives 63 / 239 / 833 hitting sets, Table 5 gives 4 / 242 / 511 evaluations, and §5.2 (`experiments.tex:128`) relies on 833. Also, positive affine (8,6,4) takes 0.181 s with zeroing (0 splits) but 0.008 s with FB+F5. Explain both.

### D9. Table caption position
Elsevier puts table captions **above** the table; the PDF has them below.

---

## E. Prose (line level)

| Location | Issue | Suggestion |
|---|---|---|
| `introduction.tex:20` | "Factorizing Gröbner-basis methods, split on…": stray comma | Remove the comma |
| `introduction.tex:21` | "earlier methods also interreduce…": earlier than what? | — |
| `introduction.tex:61` | "whereas standard standalone solvers timed out": tense change, solvers unnamed | "whereas every standalone baseline times out within 120 s" |
| `method.tex:14-15` | Imperative fragment, contraction | "Refinement then recurses on each hitting set until no branch changes." |
| `method.tex:17` | "Here is the pseudocode…" is informal | "Algorithm 2 gives the refinement procedure." |
| `method.tex:190` | "without loss of correctness" is vague | State what is guaranteed |
| `variable_zeroing.tex:18-20` | "work well with Algorithm 2" is vague | Say what it means (e.g., creates new factors) |
| `experiments.tex:15` | Reads as three methods from Singular | "two from Singular (`facstd`, `primdecGTZ`), one from Macaulay2 (`…`)" |
| `experiments.tex:22` | "each hitting-set computation": enumeration or solving? | "each branch-solver call" |
| `experiments.tex:127` | "support this explanation": F shows early SAT, not easier branches | — |
| `experiments.tex:143` vs `:165` | "five-second" vs "5-second" | Use one style |
| `conclusion.tex:15-16` | "from `facstd` alone" reads as standalone `facstd` | "from the `facstd` backend alone" |
| throughout | "Subsection 2.4" vs "Section 5.3" | Use one style |
| throughout | "factor branching" / "factor-branching" | Hyphenate only as a modifier |

---

## F. References

### F1. Closest related work is missing
Candidates to **verify before citing** (A1, A3, C4):
- **Factorizing Gröbner bases:** Czapor, *JSC* 7 (1989); Melenk, Möller & Neun (1989).
- **Triangular and regular-chain decomposition:** Wang, *JSC* 16 (1993); Kalkbrener, *JSC* 15 (1993); Chen & Moreno Maza, *JSC* 47 (2012).
- **Comprehensive Gröbner systems:** Weispfenning, *JSC* 14 (1992); Kapur, Sun & Wang (ISSAC 2010).
- **Minimal-transversal enumeration:** Eiter & Gottlob, *SIAM J. Comput.* 24 (1995); Fredman & Khachiyan, *J. Algorithms* 21 (1996).
- **Rabinowitsch trick:** Rabinowitsch, *Math. Ann.* 102 (1930).
- **Nonlinear SMT:** Jovanović & de Moura (IJCAR 2012).

### F2. `\let\cite\citep` (`paper.tex:25`) makes every citation parenthetical
This yields "Peifer et al. trained … (Peifer et al., 2020)" (`introduction.tex:29-30`, likewise Kera) and "Section 5 of (Sun et al., 2016)" (`method.tex:90-91`). Use `\citet{…}` and `\citep[Sect.~5]{sun2016}`.

### F3. Bibliography casing
`vacca2010asymptotic` prints "einstein". Protect it: `{Einstein}`.

### F4. Sources to check
- **Katsura:** usually credited to Katsura (1990) or PoSSo, not Böge–Gebauer–Kredel (1986).
- **Symbolica:** entry dated 2023 but cites version 1.5; use that version's year.
- **HomotopyContinuation.jl:** version missing (`experiments.tex:16`).
- **DOIs:** 8 of 28 entries lack one.

---

## Suggested order of work
1. Comparison with factorizing Gröbner bases, contribution statement, and abstract cuts (A1, A2, C1, C2).
2. Fixed-backend comparison, benchmark-design disclosure, measurement details (A4, A7).
3. Worked example, discussion section, ablation moved to §5 (C3, C7, A6).
4. Tables, figures, references, prose (D, E, F).
