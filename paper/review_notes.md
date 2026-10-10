# Review notes on `paper/` (2026-10-08)

Scope: the whole paper was read. The numbers in the main text were checked
against Tables 2–4 and `results/`, and every count and ratio holds. Table 3's
decomposition total of 13 relies on rcyclic-9, which is not shown; `facstd`
completes it in 2.1 s.

Applied and kept, in `experiments.tex` (Algorithm 6):

- The Table 4 caption now says that only the rcyclic sizes of Table 3 are shown.
- The Nash $n=4$ sentence now reads "computing its isolated components does
  not finish, although `std` does". Before, it said "its isolated components
  do not finish either", but `std` does finish there.

Everything below is open. Proposed wording is given where it helps.

## 1. Argument and structure (highest priority)

1. **No proofs in the paper.** The introduction says "We prove that every
   method is exact" and "Every numbered result is *also* formalized in Lean".
   But no theorem or lemma has a proof in the paper. Section 3 (Theorems 1–4,
   Lemma 1) has neither proofs nor sketches, and Lemma 3, Theorem 6 and
   Corollary 1 have one-line arguments at most. JSC referees will expect
   proof sketches, in the text or an appendix. Minimum fix: drop "also", for
   example "Every numbered result is proved in Lean 4 with Mathlib, …", so the
   text does not promise proofs that are not there.

2. **The abstract leaves out Sections 4–5.** It describes only factor branching
   and variable zeroing. It does not mention minimal primes, ideal reduction,
   ideal refinement, primary decomposition, the Lean formalization, or any
   number. This proposal keeps every `%` comment and is 250 words by `wc`:
   - After the SAT/UNSAT sentence, add: "Variants of the framework compute the
     minimal primes, smaller generators of the same ideal, ideals that
     intersect to it, and a minimal primary decomposition."
   - Replace "In our experiments, we use Gröbner-basis solvers and apply the
     method to a system …" with: "With Gröbner-basis branch solvers, factor
     branching completes 19 of 20 generated and 27 of 31 size-graded systems
     within 60 seconds; no baseline class completes more than 16 and 20. On a
     system from quantum field theory, all nine standalone baselines time out,
     whereas factor branching with a factorizing Gröbner-basis branch solver
     finishes in under a second, and the primary-decomposition variant
     decomposes it in about five minutes." The under-a-second claim must name
     the factorizing backend (`facstd`), since the Gröbner-basis backends time
     out on this system.
   - Trims to stay within 250 words:
     - "Beyond finding individual" → "Beyond individual"
     - drop "structural properties and"
     - "large polynomial systems" → "large systems"
     - "these global properties" → "these properties"
     - "even the most modern approaches still often fail" → "even modern
       approaches often fail"
     - "the solver used on each branch" → "the branch solver"
   - "Leading approaches include variants of Buchberger's algorithm, and
     resultants." has a stray comma. Use "Leading approaches include
     resultants and variants of Buchberger's algorithm."
   - Keywords: add "Primary decomposition".

3. **The conclusion is out of date.** It covers only factor branching and
   variable zeroing ("We proved that refinement terminates …"), and leaves out
   Sections 4–5 and the primary decomposition of the physical system.

4. **The gap is vague.** The introduction says "large systems such as a system
   that came up in our research of quantum gravity can remain inaccessible".
   Proposal, using only facts already in the paper: "Despite this progress,
   large systems can remain inaccessible. On a system of 29 equations in 22
   variables from our research in quantum gravity, all nine standalone solvers
   in our comparison time out after 120 seconds, although interreducing and
   factoring its equations splits it into seven branches
   (Section~\ref{subsec:physical-system})."

5. **The closest prior work is never compared.** Factorizing Gröbner bases are
   the nearest method, but the introduction never says how factor branching
   differs. After "… combines the results into its output.", add: "Unlike
   factorizing Gröbner bases, which split on one factorization at a time
   inside a Gröbner-basis computation, the methods split on the factorizations
   of all equations at once." In the history paragraph, "earlier methods also
   interreduce and prune redundant branches" has no referent for "earlier";
   use "they also interreduce …".

6. **The complexity argument doesn't follow.** "Doubly exponential in the
   number of variables, so breaking into smaller branches can be faster" does
   not hold as written, because branches keep the variables. The staged diff
   deleted the `%` note that raised this. Proposed bridge before that
   sentence: "The generators of a branch are factors of the equations, so
   their degrees are no higher, and a linear factor eliminates a variable."

7. **A comparison in the introduction is ambiguous.** "Against at most 16 and
   20 for the direct Gröbner-basis and decomposition methods" reads as direct
   = 16 and decomposition = 20. In fact 16 is decomposition (baseline set) and
   20 is direct (size-graded set). Use "No class of direct Gröbner-basis or
   decomposition methods completes more than 16 and 20, respectively."

8. **A result sits in the method section.** The comparison with $x=1$ and with
   random affine splits (Section 6.1) is an experimental result, and it comes
   before the physical system is described in Section 7.4. Move it to Section
   7.4, after the Rabinowitsch paragraph, opening with "For comparison, we
   repeat the M4GB run of variable zeroing twice with affine side conditions:
   …". Replace "zero/nonzero splits" with "splits into $x=0$ and $x\neq0$".

9. **A forward reference.** Subsection 6.3 (branch contraction) uses
   $V_Y(Q;Z,N)$ and Equation `eq:zeroing-split`, which are both defined in
   6.4. Add "With the notation $V_Y$ of
   Subsection~\ref{subsec:zeroing-theoretical-analysis}," or move 6.3 after
   6.4.

## 2. Experiments

1. **The baseline comparison leaves out a stronger baseline.** Table 2's direct
   Gröbner-basis class has no Singular `std` in degrevlex. Yet the
   IdealRefine paragraph says `std` completes 12 of the 20 baseline systems,
   against 9 for the class in Table 2. Add it or re-run; the "#fastest" counts
   may change. Also state which monomial order Table 2's "Singular" uses.
2. **The appendix is orphaned.** `paper.tex` no longer inputs `appendix.tex`,
   but Section 7.1 still describes how the 3-second variable-zeroing cutoff
   was tuned on the 20 systems. Those results now appear nowhere in the
   paper. Re-include the appendix or cut the paragraph.
3. "In all experiments, we solve the branches with a Gröbner-basis
   computation" is false, because Tables 4–5 use `facstd` and `minAssGTZ` as
   branch solvers. Use "Unless stated otherwise, …".
4. **The engines aren't attributed.** Say that M4GB, F5 and GVW are your own
   Rust implementations and that F4 is Symbolica's `GroebnerBasis::new`, for
   example "our Rust implementations of M4GB, F5, and GVW, Symbolica's F4, or
   Singular 4.4.1".
5. **Three benchmark families are never defined, so a reader can't reproduce
   them.** These definitions follow the generator in `scripts/`:
   - A signed-distance cycle on $n$ vertices has the equations
     $(z_{i+1}-z_i-1)(z_{i+1}-z_i+1)=0$, indices modulo $n$, so every edge has
     signed length $\pm1$.
   - An affine-factor hypergraph $(n,m,k)$ has $m$ equations in $n$
     variables. Each is the product of $k$ sparse affine-linear forms drawn
     from a seeded pool of $2n$. The positive family has $m<n$; the square
     family has $m=n$.
   - An overlapping random multiple $(n,p,k)$ has $n$ equations in $n$
     variables. Each is the product of $k$ factors drawn from a pool of $p$
     seeded dense quadratics. The factors recur across equations, but no
     factor divides every equation.
   - The fourth bullet ("This produces one easy selector branch …") continues
     the gated bullet; it is not a separate item.
6. **A claim undercounts.** "Only factor branching completes rcyclic-19 and
   box-QP with $n=8$" leaves out that factor branching is also the only method
   to complete both Nash games at $n=5$ and Lotka–Volterra at $n=12$. Use
   "Only factor branching completes the largest solved size of each family:
   rcyclic-19, the Nash games with $n=5$, box-QP with $n=8$, and
   Lotka–Volterra with $n=12$. Rcyclic and box-QP reach this point
   differently."
7. rcyclic-13 is not in Table 4, so "a single branch takes almost all of the
   18 seconds" refers to a number the reader can't see. Use "On rcyclic-13,
   which is not listed, a single branch takes almost all of the 18-second
   run".
8. **The IdealRefine numbers appear in no table:** 9 against 15 listed
   systems, 8 against 12 baseline systems, and slowdowns of 3–180× and
   1.6–40×. Add a column or a short table, or cite the results file.
9. Physical system: "Every backend solves the same 6 of the 7 terminal
   branches … and times out on the last one" is wrong for `facstd`, which
   doesn't time out. Use "Every other backend".
10. **How π is handled isn't stated.** The setup says the arithmetic is over
    the rationals, but the physical system has π in its coefficients, and only
    the primary-decomposition paragraph mentions $\mathbb Q(\pi)$. State how
    the Rust backends treat π. If they treat it as a variable, UNSAT
    certificates remain valid, but a SAT branch may be SAT only at special
    values of π.
11. **Two run times disagree.** Figure 7's caption gives 64.154 s and Table 5
    gives 64.166 s for M4GB variable zeroing. They are two different runs; say
    so or make them agree.
12. The subsection title "Physical System" should be "Physical system"; the
    other subsection titles use sentence case.
13. The outline comment `% [IdealReduce: no factor lies in every SAT branch
    here; with skipped checks it matches std]` has no text under it.

## 3. Figures

1. **Figure 1 is missing two methods.** The caption says "Crosses mark the
   methods of this paper", but ideal reduction (Algorithm 4, cell one piece ×
   $I$) and variable zeroing (Algorithm 7, cell cover × $\sqrt I$) have no
   cross. Labels such as "Algorithm 5" force a lookup; use method names. The
   saturated red hatching could be gray.
