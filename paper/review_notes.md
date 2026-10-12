# Review of `paper.tex`: issues outside Section 4

Review date: 2026-10-12. Issues related to Section 4 (`sections/method_ideal_covering.tex`) were fixed in the
source. The items below were left for the authors. Line numbers refer to the files as of this review.

## A. Related to Section 4 but not fixable in the text alone

1. **The primdecSY comparison on the physical system is confounded** (`experiments.tex:410-413`, echoed in
   `introduction.tex:100-103`). The 213 s primary-branching run decomposes the reduced system with
   κ_h ≠ 0 imposed by *saturation*. The >30 min primdecSY baseline ran on the *unreduced* system with κ_h ≠ 0
   imposed by a *Rabinowitsch generator*. Two things change at once, so the text cannot attribute the speed-up to
   ideal reduction. The fair pair from 2026-10-07 (both with a Rabinowitsch generator, lex min-fill ring) was
   315 s against >30 min. Either run primdecSY on the 28 unreduced equations saturated by κ_h, or report the
   older controlled pair.
2. **Placeholder outline with no prose** (`experiments.tex:346`):
   `% [IdealReduce: no factor lies in every SAT branch here; with skipped checks it matches std]`. The claim
   "with skipped checks it matches std" has no number in the paper. Either add the measured cost of
   Algorithm 4 relative to `std` on Table 4's systems, or leave the outline empty on purpose.
3. **"The implementations use the optimizations described after each algorithm"** (`experiments.tex:231-232`)
   is broader than what the code does. For example, `scripts/factor_variants.py::prime_branching` does not
   skip the hitting sets that ideal reduction already certified as UNSAT. Section 4 now calls that skip
   optional ("need not be revisited"). The detailed list that follows the sentence is accurate, so the
   sentence can simply be dropped.

## B. Argument and structure

1. **Closest prior work is not compared concretely** (`introduction.tex:38-44`). Factorizing Gröbner bases
   (Gräbe; Singular `facstd`) and triangular or regular-chain decompositions are the nearest methods. The
   introduction should say what factor branching does differently. For example, it branches on minimal
   hitting sets over all equations at once rather than on one factorization at a time, and it is
   solver-agnostic. The introduction should also say when this helps. "Earlier methods also interreduce and
   prune redundant branches" (`introduction.tex:40-41`) does not say which methods.
2. **The complexity argument does not support the method** (`introduction.tex:68`). The Mayr–Meyer bound is
   doubly exponential in the number of variables, but branches keep all the variables. The real argument is
   lower degrees and fewer generators per branch, as Theorem 3 ("Refined branches are simpler") shows. Cite
   that theorem here instead of, or alongside, the variable count.
3. **The gap statement is one sentence about the authors' own system** (`introduction.tex:55-56`). Add which
   kinds of systems remain out of reach and why, such as large systems with factor structure from
   applications, and give the evidence from the baselines.
4. **The abstract has no quantitative result** (`paper.tex:91`). The 19/20 and 27/31 counts and the QFT
   result (all nine baselines time out, factor branching < 1 s) are the strongest evidence. There is room:
   the abstract is about 200 of 250 words.
5. **The abstract makes an unsupported claim** (`paper.tex:86`): "even the most modern approaches still often
   fail on practical problems or slight perturbations of standard benchmarks". The gated Cyclic and Katsura
   systems could support the second half, but the paper never states this link.
6. **The conclusion lacks limitations and a concrete outlook** (`conclusion.tex`). The experiments already
   show the limits: Cyclic-7, where one branch keeps the full difficulty; `facstd` alone is faster on small
   systems; Table 2 shows only single runs. "Future work will test this benefit on other systems" is weak.
   The conclusion also omits the size-graded results (27 of 31) and the Lean formalization.
7. **Terminology for the physical system varies.** The abstract says "quantum field theory", the
   introduction "quantum gravity" (`introduction.tex:56`), and the experiments "graviton couplings in
   asymptotic-safety models". Pick one description and use it throughout.
8. **The introduction singles out `facstd`** (`introduction.tex:99-100`: "with the `facstd` backend completes
   in under a second"), but all six backends finish in under 0.9 s (Table 5 and the conclusion). Say "with
   every backend".

## C. Methods (Sections 2–3)

1. **Dimension is never defined.** Sections 3 and 4 use "dimension", "zero-dimensional", and
   "top-dimensional". Add one sentence to Subsection 2.2: the dimension of J is the Krull dimension of R/J,
   which equals dim V(J).
2. **An uncited fact is used three times.** "A prime that strictly contains 𝔭′ has smaller dimension"
   (`method.tex:242-244`) is reused by Lemma 4 and by the marking paragraph of Section 4. Cite a source, for
   example Cox–Little–O'Shea, Ch. 9. Check the exact proposition before citing it.
3. **Notation clash.** The proof of Theorem 5 (`method.tex:266-281`) uses 𝔮 for a *minimal prime*, while
   Subsection 2.2 and Section 4 use 𝔮 for *primary components*. Rename it, e.g. to 𝔭*.
4. **Algorithm 2 (Factor branching) has no `\Ensure`**, and its `\Require` reads "A set of polynomials
   P = {p_1, …, p_m}" where the other algorithms read "A finite polynomial system P" (`method.tex:60`).
5. **Theorem 4 (Factor branching) indexes the wrong set** (`method.tex:203-212`). Algorithm 2 puts the
   solver's representations into S, but Theorem 4 writes ∪_{H∈S} V(H) as if S held the branches. Either
   index over 𝓛(P), or say that each element of S represents V(H).
6. **Prune is defined ambiguously** (`method.tex:44`): it "returns the branches whose solution sets it cannot
   certify to be contained in each other". Say instead that it removes each branch whose solution set it
   certifies to lie in that of another kept branch. Also say what happens to equal solution sets.
7. **The interreduce-first paragraph gives no criterion** (`method.tex:46-48`). Experiments interreduce first
   for prime branching. State the rule the implementation uses, or the observed effect.
8. **The branch-order paragraph is speculative** (`method.tex:79-81`): "likely … probably lower
   dimensional". Also, "First … Secondly" is not parallel. Either support the claim with the observed
   effect, or shorten it to the rule the implementation uses.
9. **Minimal-hitting-set enumeration is never described or cited.** The bibliography contains
   Fredman–Khachiyan, Eiter–Gottlob, and König, but none is cited. Say how the code enumerates 𝓗(Q) and cite
   the algorithm.
10. **The Berge citation is misplaced** (`preliminaries.tex:152`). It is attached to "defines a branch". It
    belongs on the hitting-set (transversal) definition. The uniqueness of the reduced Gröbner basis
    (`preliminaries.tex:119-120`) has no citation (Cox–Little–O'Shea, Ch. 2, §7).

## D. Experiments (Section 5)

1. **π is a ring variable in the Rust runs.** The paper says "its coefficients involve π"
   (`experiments.tex:356`). However, the Rust loader turns 1/π² into a ring variable, so Table 5 solves a
   system over ℚ with one more variable. The Singular primary-decomposition runs work over ℚ(π). SAT verdicts
   over ℚ[π] can differ from those over ℚ(π). State this, and say whether the 10 SAT branches were checked
   over ℚ(π).
2. **The setup does not cover the Python/Singular runs.** Subsection 5.1 says "We implemented Algorithm 2 in
   Rust" (`experiments.tex:7`), but Table 4 and the physical primary decomposition run through Python driving
   Singular (`scripts/factor_variants.py`). Name both implementations in 5.1.
3. **Backends and baselines are not listed in 5.1** (`experiments.tex:12-18`). `facstd` is used as a backend
   (Tables 4 and 5) but is missing from the list. The "six direct whole-ideal Gröbner-basis configurations
   and three decomposition methods" are named only in Subsection 5.4.
4. **Timing conditions are not stated.** The scripts can run experiments concurrently, which slows them by
   25–50%. State that the reported times are from sequential runs, if they are.
5. **The baseline family list is broken** (`experiments.tex:33-39`). It promises six families, but the fourth
   bullet ("This produces one easy selector branch…") continues the gated bullet rather than naming a family.
   Signed-distance cycles and the affine hypergraphs get no construction or reference. Table 1 merges "Gated
   Katsura and dense quadratics" into one row.
6. **The size-graded list promises five families but has four bullets** (`experiments.tex:132-150`). Nash
   generic and Nash degenerate count as two families. Say so.
7. **Some cited numbers appear in no table** (`experiments.tex:319-325`): "factor branching with `std` in the
   same harness … 15.13 seconds", and rcyclic-13's "18 seconds". Add them to a table or the results
   directory, or cut them.
8. **Section titles use inconsistent case.** "Physical System" (`experiments.tex:348`) is title case, while the
   other subsections use sentence case.
9. **Nonzero handling in factor branching has no theory.** Section 4 now states the nonzero variant of ideal
   reduction. The refinement rules in `experiments.tex:364-368` ("removes κ_h from every factor set and drops
   a branch once κ_h reduces to zero") have no counterpart in Section 3. Add one sentence there, or a
   pointer to `symbolica/nonzero.rs`.

## E. References and submission

1. **Many entries lack DOIs**, including `berge1989hypergraphs`, `hofbauer1998evolutionary`,
   `peifer2020learning`, `kera2024learning`, and `graebe1995factorized`. JSC asks for DOIs where they exist.
   (DOIs were added for `shimoyama1996localization` and `eisenbud1992direct`.)
2. **The bibliography has uncited entries**: `sun2016`, `marques1999grasp`, `marques2009cdcl`,
   `breiding2018homotopy`, `bezanson2017julia`, `eiter1995transversals`, `fredman1996dualization`, and
   `konig1927schlussweise`. natbib omits them from the output, but some belong in the text (see C.9).
3. **Optional JSC items.** A highlights file (3–5 bullets, ≤ 85 characters each) is encouraged.
