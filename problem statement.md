# Problem Statement

## Problem

There is a set of multivariate polynomials $p_1, \ldots, p_m$ we do the following preprocessing:
1. zero out selected variables. (Optionally)
2. interreduce the polynomials with lex order
3. find the square-free factorization of each polynomial into $f_{1, 1} \cdot \ldots \cdot f_{1, n_1}, \ldots, f_{m, 1} \cdot \ldots \cdot f_{m, n_m}$.
4. remove constant factors from the polynomials, and clear zero polynomials.

Taking the preprocessed polynomials as given, We would like to find a set of tuples of factors and variables $(f_1, x_1), (f_2, x_2), \ldots, (f_k, x_k)$ such that:
1. each polynomial has a factor in the set of tuples, A factor $f_i$ covers every polynomial it appears in.
2. for each tuple $(f_i, x_i)$, the equation $f_i = 0$ is linear in $x_i$, and thus provides a rational parametrization of the variable $x_i$.
3. the variables $x_1, \ldots, x_k$ are pairwise distinct.
4. the directed graph $G = (V, E)$ with $V = \{x_1, \ldots, x_k\}$ and $E = \{(x_j, x_i) \mid j \neq i \text{ and } x_j \text{ appears in } f_i\}$ is acyclic.
5. selecting a tuple $(f_i, x_i)$ can't imply that another tuple $(f_j, x_j)$ is not valid; if we only allow for tuples such that $f_m$ is linear in $x_m$, that means that the solution of $f_i = 0$ doesn't imply that we get $g(v) = 0$ for $f_j = g(v) * x_j + h(v)$.

Claim: If we find tuples $(f_1, x_1), (f_2, x_2), \ldots, (f_k, x_k)$ such that the conditions are met, we can find a parametrization in radicals of a non-empty component of the solution set of the system $p_1 = \cdots = p_m = 0$, with the variables outside $\{x_1, \ldots, x_k\}$ acting as free parameters.

Proof (TBA):

## Choosing the lex order (optional and unimportant for now)

The lex order used for interreduction in step 2 affects which variables get eliminated from which polynomials, and hence how much structure the MILP in the next section has to work with. 
We want an order that, after interreduction, leaves polynomials sharing as few variables as possible — this both reduces the size of $A$ and makes the acyclicity constraint (C4) easier to satisfy.

Interreduction under lex behaves like sparse elimination: when a variable $v$ is high in the order, one polynomial is picked as its "definer" (the one whose leading term involves $v$) and $v$ is removed from every other polynomial by reduction. 
Choosing the order is therefore a variable-elimination ordering problem, well studied in sparse linear algebra (Markowitz / minimum-degree heuristics). 

## Solving the problem

We can solve the problem by formulating it as a mixed-integer linear program (MILP).

### Data

Let
- $F$ — the set of all distinct square-free factors appearing across $p_1, \ldots, p_m$ after preprocessing.
- $F_j \subseteq F$ — the factors of $p_j$.
- $V$ — the set of all variables occurring in any $p_j$, with $n = |V|$.
- $\mathrm{vars}(f) \subseteq V$ — the variables appearing in $f$.
- $\deg_x f$ — the degree of $f$ in the variable $x$.
- $c_{f,x}$ — the coefficient of $x$ when $f$ is viewed as a univariate polynomial in $x$.

Define the set of *admissible pairs*
$$A = \{\,(f, x) \mid f \in F,\ x \in \mathrm{vars}(f),\ \deg_x f = 1\,\}.$$
Condition 2 is enforced by restricting our choices to $A$. For every $(f,x)\in A$, write
$$f = c_{f,x} x + r_{f,x}, \qquad x \notin \mathrm{vars}(c_{f,x}) \cup \mathrm{vars}(r_{f,x}).$$

Define the incompatibility relation
$$I = \{\,((f_i,x_i),(f_j,x_j)) \in A^2 \mid (f_i,x_i) \ne (f_j,x_j),\ f_i \text{ divides } c_{f_j,x_j}\,\}.$$
This records when solving $f_i=0$ would force the linear coefficient of another selected tuple to vanish.

### Decision variables

- $s_{f,x} \in \{0, 1\}$ for every $(f, x) \in A$ — set to $1$ iff the tuple $(f, x)$ is selected.
- $z_x \in \{0, 1\}$ for every $x \in V$ — set to $1$ iff $x$ is a dependent variable (appears as the second coordinate of some selected tuple).
- $t_x \in [0, n]$ (continuous) for every $x \in V$ — topological-order stamp for $x$.

### Linking constraints

$$z_x = \sum_{f\,:\,(f,x) \in A} s_{f,x} \qquad \forall x \in V. \tag{L}$$

### Encoding the five conditions

**Condition 1 — coverage.** Each polynomial must have a selected factor:
$$\sum_{f \in F_j} \ \sum_{x\,:\,(f,x)\in A} s_{f,x} \ \ge\ 1 \qquad \forall j = 1, \ldots, m. \tag{C1}$$

**Condition 2 — linear solvability.** Already baked into $A$ by the degree restriction $\deg_x f = 1$.

**Condition 3 — distinct dependent variables.** By (L), $z_x \in \{0, 1\}$ already forces at most one tuple per $x$:
$$\sum_{f\,:\,(f,x)\in A} s_{f,x} \ \le\ 1 \qquad \forall x \in V. \tag{C3}$$
Optionally add $\sum_{x\,:\,(f,x)\in A} s_{f,x} \le 1$ for all $f$ to forbid reusing the same factor (never useful — one use of $f$ already covers every polynomial it divides).

**Condition 4 — acyclicity.** Use the topological stamps $t_x$ as a Miller–Tucker–Zemlin-style certificate. For every $(f, x) \in A$ and every $x' \in \mathrm{vars}(f) \setminus \{x\}$, if the tuple $(f, x)$ is selected and $x'$ is itself a dependent variable, then $x'$ must precede $x$ in the topological order:
$$t_{x'} + 1 \ \le\ t_x + M\,\bigl(2 - s_{f,x} - z_{x'}\bigr), \qquad M = n + 1. \tag{C4}$$
Explanation: when $s_{f,x} = z_{x'} = 1$ the big-M term vanishes and we get the strict predecessor constraint $t_{x'} + 1 \le t_x$; otherwise the RHS is $\ge t_x + M$ and the constraint is trivial. When $x'$ is not a dependent variable ($z_{x'} = 0$) it is a free parameter in $Y$ and imposes no ordering.

**Condition 5 — preserving selected linear solves.** No selected tuple may force another selected tuple's linear coefficient to vanish:
$$s_{f_i,x_i} + s_{f_j,x_j} \le 1 \qquad \forall ((f_i,x_i),(f_j,x_j)) \in I. \tag{C5}$$
Since all selected tuples are linear, this is the required non-implication condition: the component chosen by one selected factor must not make another selected equation lose the variable it is supposed to solve for.

### Objective

The problem as stated is a feasibility problem; any feasible assignment yields a valid cover. A natural minimization objective is the size of the cover,
$$\min \ \sum_{(f,x) \in A} s_{f,x}, \tag{Obj}$$

### Summary

$$
\begin{aligned}
\min \quad & \sum_{(f,x) \in A} s_{f,x} \\
\text{s.t.} \quad
& \sum_{f \in F_j} \sum_{x\,:\,(f,x)\in A} s_{f,x} \ge 1 && \forall j, \\
& z_x = \sum_{f\,:\,(f,x)\in A} s_{f,x} && \forall x \in V, \\
& t_{x'} + 1 \le t_x + (n+1)\,(2 - s_{f,x} - z_{x'}) && \forall (f,x)\in A,\ x' \in \mathrm{vars}(f)\setminus\{x\}, \\
& s_{f_i,x_i} + s_{f_j,x_j} \le 1 && \forall ((f_i,x_i),(f_j,x_j)) \in I, \\
& s_{f,x} \in \{0,1\},\ z_x \in \{0,1\},\ t_x \in [0, n].
\end{aligned}
$$

Any optimal (or merely feasible) solution of this MILP gives a valid set of tuples $\{(f, x) : s_{f,x} = 1\}$, and by the claim proven above, a parametrization in radicals of a component of the variety of $\{p_1, \ldots, p_m\}$ can then be read off in topological order of the $t_x$ stamps.


## Zeroing variables

Zeroing a variable $x$ means committing to $x = 0$ in the parametrization. Since $x = 0$ is trivially expressible in radicals, this is always admissible (degree-1 equation), and it interacts with the rest of the MILP in three useful ways:

1. **Fully zeroing a factor.** If every monomial of a factor $f$ contains at least one zeroed variable, then $f$ vanishes identically on the parametrized component. Any polynomial containing $f$ is automatically covered, with no selection cost.
2. **Easing acyclicity.** A zeroed variable has constant value $0$ and needs no topological predecessor: it can appear inside any chosen factor without inducing an edge in $G$.
3. **Changing the selected equation.** Zeroing can erase the monomials that contain the variable a selected tuple intends to solve for. For example, selecting $(k + xe, x)$ is not a valid way to solve for $x$ if $e$ is also zeroed, because the residual equation is $k = 0$.
4. **Inducing new factors.** After substitution the residual polynomials $\tilde p_j = p_j|_{x \to 0}$ may square-free-factor more finely than $p_j$ did (e.g. $xy + yz \to z$ after zeroing $x$ and clearing the $y$). Exploiting this fully requires re-running the preprocessing; the formulation below captures only effects (1), (2), and the invalid residuals from (3), which are the ones representable in a single MILP without refactorization.

### Modified data

For each factor $f \in F$ let $M(f)$ be its set of monomials and $\mathrm{vars}(m) \subseteq V$ the variables of monomial $m \in M(f)$. Admissible pairs $A$ and the incompatibility relation $I$ are unchanged.

### Additional decision variables

- $w_x \in \{0, 1\}$ for every $x \in V$ — set to $1$ iff $x$ is zeroed.
- $y_f \in \{0, 1\}$ for every $f \in F$ — set to $1$ iff $f$ is zero-killed by the chosen zeroing set.

### New constraints

**Zero-kill certificate.** Factor $f$ is zero-killed iff every monomial has at least one zeroed variable:
$$y_f \ \le\ \sum_{x \in \mathrm{vars}(m)} w_x \qquad \forall f \in F,\ \forall m \in M(f). \tag{Z1}$$
This is the standard hitting-set linearization; the MILP will freely set $y_f = 1$ whenever the RHS permits.

**Zeroing excludes parametrization.** A zeroed variable is already determined and should not also be selected as the dependent of some tuple:
$$z_x + w_x \ \le\ 1 \qquad \forall x \in V. \tag{Z2}$$

**Extended coverage.** A polynomial is now covered if some factor is either selected or zero-killed:
$$\sum_{f \in F_j}\Bigl(\ \sum_{x\,:\,(f,x) \in A} s_{f,x}\ +\ y_f\ \Bigr)\ \ge\ 1 \qquad \forall j = 1, \ldots, m. \tag{C1$'$}$$
This replaces (C1).

**Relaxed acyclicity.** Zeroed variables drop out of the dependency graph. Rewrite (C4) to ignore an in-neighbour $x'$ when $w_{x'} = 1$:
$$t_{x'} + 1 \ \le\ t_x + M\,\bigl(3 - s_{f,x} - z_{x'} - (1 - w_{x'})\bigr) \ =\ t_x + M\,\bigl(2 - s_{f,x} - z_{x'} + w_{x'}\bigr), \qquad M = n + 1. \tag{C4$'$}$$
By (Z2) the term $z_{x'} - w_{x'}$ never exceeds $1$, so the RHS dominates $t_x$ precisely when $s_{f,x} = 1$ and $x'$ is a non-zeroed dependent.

**Residual solvability after zeroing.** For every candidate tuple $(f,x)$ and zeroed variable $u$, forbid selecting $(f,x)$ together with $u=0$ when $u$ appears in every monomial of $f$ that contains $x$, but not in every monomial of $f$:
$$s_{f,x} + w_u \le 1. \tag{Z3}$$
This is the single-variable zeroing certificate for the case where zeroing $u$ removes all $x$-terms while leaving a nonzero residual constraint in other variables. Such a residual does not solve for $x$ and must not make those other variables look free.

**Selected-pair compatibility.** The original compatibility condition remains active:
$$s_{f_i,x_i} + s_{f_j,x_j} \le 1 \qquad \forall ((f_i,x_i),(f_j,x_j)) \in I. \tag{C5}$$

### Objective

Zeroing commits the parametrized component to a subvariety, so it should have real cost. A simple objective minimizes the number of commitments:
$$\min \quad \sum_{(f,x) \in A} s_{f,x}\ +\ \sum_{x \in V} w_x. \tag{Obj$'$}$$
If selected radical solves should be preferred over zeroing in ties, add a small secondary penalty to $\sum_x w_x$ or solve lexicographically.

### Summary

$$
\begin{aligned}
\min \quad & \sum_{(f,x)\in A} s_{f,x} + \sum_{x \in V} w_x \\
\text{s.t.}\quad
& \sum_{f \in F_j}\Bigl(\sum_{x\,:\,(f,x)\in A} s_{f,x} + y_f\Bigr) \ge 1 && \forall j, \\
& z_x = \sum_{f\,:\,(f,x)\in A} s_{f,x} && \forall x \in V, \\
& z_x + w_x \le 1 && \forall x \in V, \\
& y_f \le \sum_{x \in \mathrm{vars}(m)} w_x && \forall f \in F,\ \forall m \in M(f), \\
& s_{f,x} + w_u \le 1 && \forall (f,x)\in A,\ u \text{ invalidates } (f,x), \\
& s_{f_i,x_i} + s_{f_j,x_j} \le 1 && \forall ((f_i,x_i),(f_j,x_j)) \in I, \\
& t_{x'} + 1 \le t_x + (n+1)\,(2 - s_{f,x} - z_{x'} + w_{x'}) && \forall (f,x)\in A,\ x' \in \mathrm{vars}(f)\setminus\{x\}, \\
& s_{f,x},\ z_x,\ w_x,\ y_f \in \{0,1\},\ t_x \in [0, n].
\end{aligned}
$$

### Correctness

The claim of the original section extends: read off the solution by setting $x \leftarrow 0$ for every $x$ with $w_x = 1$, then apply the inductive parametrization to the remaining tuples $\{(f, x) : s_{f,x} = 1\}$ in order of $t_x$. For every selected tuple, (Z3) rules out the case where zeroing removes all terms in the chosen solve variable while leaving a residual constraint in other variables. For every $j$, (C1$'$) gives either a selected factor — which vanishes by the original argument after the valid zero substitutions — or a zero-killed factor $f$ with $y_f = 1$, which vanishes because every monomial of $f$ contains a zeroed variable. In both cases $p_j$ vanishes at the parametrized point.

### Practical notes

- (Z1) adds one inequality per monomial; for dense factors this is the dominant growth term. Restricting $w_x$ to a candidate set (e.g. variables that actually appear as minors in some $F_j$, or variables flagged by the ordering pass as entangled tails) keeps the MILP small.
- If the unzeroed MILP is already feasible, the zeroing variant can be run as a warm-start with the previous solution fixed and $w \equiv 0$ relaxed — the LP relaxation quickly reveals whether any zeroing improves the objective.
- Effect (3) — genuinely new factors after substitution — is recoverable by iterating: solve the MILP, apply the chosen zeroing to the $p_j$, re-run preprocessing, solve again. Convergence is guaranteed since each iteration either strictly decreases the number of uncovered polynomials or fixes $w \equiv 0$.