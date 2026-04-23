# Problem Statement

## Problem

There is a set of multivariate polynomials $p_1, \ldots, p_m$ we do the following preprocessing:
1. interreduce the polynomials with grevlex order
2. find the square-free factorization of each polynomial into $f_{1, 1} \cdot \ldots \cdot f_{1, n_1}, \ldots, f_{m, 1} \cdot \ldots \cdot f_{m, n_m}$.
3. remove constant factors from the polynomials.
4. clear zero polynomials.

Taking the preprocessed polynomials as given, We would like to find a set of tuples of factors and variables $(f_1, x_1), (f_2, x_2), \ldots, (f_k, x_k)$ such that:
1. each polynomial has a factor in the set of tuples, A factor $f_i$ covers every polynomial it appears in.
2. for each tuple $(f_i, x_i)$, the equation $f_i = 0$ is linear in $x_i$ or is a polynomial in the variable $x_i$ in a degree less than 5, and thus provides a parametrization in radicals of the variable $x_i$.
3. the variables $x_1, \ldots, x_k$ are pairwise distinct.
4. the directed graph $G = (V, E)$ with $V = \{x_1, \ldots, x_k\}$ and $E = \{(x_j, x_i) \mid j \neq i \text{ and } x_j \text{ appears in } f_i\}$ is acyclic. (We write "$x_j$ appears in $f_i$" to mean $x_j$ occurs as a variable of the polynomial $f_i$.)

Claim: If we find tuples $(f_1, x_1), (f_2, x_2), \ldots, (f_k, x_k)$ such that the conditions are met, we can find a parametrization in radicals of a non-empty component of the solution set of the system $p_1 = \cdots = p_m = 0$, with the variables outside $\{x_1, \ldots, x_k\}$ acting as free parameters.

Proof:

Let $X = \{x_1, \ldots, x_k\}$ be the *dependent* variables and let $Y$ be the set of all other variables appearing in the $p_j$; the elements of $Y$ will play the role of free parameters.

Since $G$ is acyclic (condition 4) it admits a topological order. Re-index the tuples so that $(x_1, x_2, \ldots, x_k)$ is such an order, i.e. whenever $x_j$ appears in $f_i$ with $j \neq i$ we have $j < i$. Condition 3 guarantees the $x_i$ are distinct, so this re-indexing is well-defined.

We construct the parametrization inductively. Assume that for every $\ell < i$ we have already produced a radical expression $\varphi_\ell$ in the free parameters $Y$ such that substituting $x_\ell \leftarrow \varphi_\ell$ satisfies $f_\ell = 0$. Consider the equation $f_i = 0$. By the topological order, every variable of $f_i$ lies in $Y \cup \{x_1, \ldots, x_i\}$: any $x_j \in X$ appearing in $f_i$ with $j \neq i$ is a predecessor of $x_i$ in $G$, hence $j < i$. Substituting $\varphi_1, \ldots, \varphi_{i-1}$ for $x_1, \ldots, x_{i-1}$ in $f_i$ yields a univariate polynomial
$$\tilde f_i(x_i) \in \overline{\mathbb{Q}(Y)}[x_i]$$
whose coefficients are radical expressions in $Y$. By condition 2, $\tilde f_i$ is either linear in $x_i$ or has degree $\leq 4$ in $x_i$. In both cases the roots are given by radical formulas (the linear formula, or the Cardano/Ferrari formulas for degrees 2, 3, 4); pick any root and call it $\varphi_i$. Inlining the $\varphi_\ell$ for $\ell < i$, the expression $\varphi_i$ is a radical in $Y$ alone.

(Degenerate case: substitution might make $\tilde f_i$ the zero polynomial, in which case any value — e.g. $\varphi_i = 0$ — satisfies it. It might also drop to a lower nonzero degree, which is still solvable by radicals.)

After $k$ steps we obtain radical expressions $\varphi_1, \ldots, \varphi_k$ in $Y$ with $f_i(\varphi_1, \ldots, \varphi_k, Y) = 0$ for every $i = 1, \ldots, k$.

It remains to show that every $p_j$ also vanishes at this point. Fix $j$. By condition 1, some chosen factor $f_i$ covers $p_j$, meaning $f_i$ appears in the square-free factorization $p_j = f_{j,1} \cdots f_{j, n_j}$. Therefore $f_i \mid p_j$ in the polynomial ring, and since $f_i$ vanishes at $(\varphi_1, \ldots, \varphi_k, Y)$ so does $p_j$.

Thus $(x_1, \ldots, x_k) = (\varphi_1(Y), \ldots, \varphi_k(Y))$ is a parametrization in radicals, with free parameters $Y$, of a non-empty component of the zero set of $\{p_1, \ldots, p_m\}$. $\blacksquare$

## Solving the problem

We can solve the problem by formulating it as a mixed-integer linear program (MILP).

### Data

Let
- $F$ — the set of all distinct square-free factors appearing across $p_1, \ldots, p_m$ after preprocessing.
- $F_j \subseteq F$ — the factors of $p_j$.
- $V$ — the set of all variables occurring in any $p_j$, with $n = |V|$.
- $\mathrm{vars}(f) \subseteq V$ — the variables appearing in $f$.
- $\deg_x f$ — the degree of $f$ in the variable $x$.

Define the set of *admissible pairs*
$$A = \{\,(f, x) \mid f \in F,\ x \in \mathrm{vars}(f),\ 1 \le \deg_x f \le 4\,\}.$$
Condition 2 is enforced by restricting our choices to $A$.

### Decision variables

- $s_{f,x} \in \{0, 1\}$ for every $(f, x) \in A$ — set to $1$ iff the tuple $(f, x)$ is selected.
- $z_x \in \{0, 1\}$ for every $x \in V$ — set to $1$ iff $x$ is a dependent variable (appears as the second coordinate of some selected tuple).
- $t_x \in [0, n]$ (continuous) for every $x \in V$ — topological-order stamp for $x$.

### Linking constraints

$$z_x = \sum_{f\,:\,(f,x) \in A} s_{f,x} \qquad \forall x \in V. \tag{L}$$

### Encoding the four conditions

**Condition 1 — coverage.** Each polynomial must have a selected factor:
$$\sum_{f \in F_j} \ \sum_{x\,:\,(f,x)\in A} s_{f,x} \ \ge\ 1 \qquad \forall j = 1, \ldots, m. \tag{C1}$$

**Condition 2 — solvability by radicals.** Already baked into $A$ by the degree restriction $1 \le \deg_x f \le 4$ (linear is the case $\deg_x f = 1$).

**Condition 3 — distinct dependent variables.** By (L), $z_x \in \{0, 1\}$ already forces at most one tuple per $x$:
$$\sum_{f\,:\,(f,x)\in A} s_{f,x} \ \le\ 1 \qquad \forall x \in V. \tag{C3}$$
Optionally add $\sum_{x\,:\,(f,x)\in A} s_{f,x} \le 1$ for all $f$ to forbid reusing the same factor (never useful — one use of $f$ already covers every polynomial it divides).

**Condition 4 — acyclicity.** Use the topological stamps $t_x$ as a Miller–Tucker–Zemlin-style certificate. For every $(f, x) \in A$ and every $x' \in \mathrm{vars}(f) \setminus \{x\}$, if the tuple $(f, x)$ is selected and $x'$ is itself a dependent variable, then $x'$ must precede $x$ in the topological order:
$$t_{x'} + 1 \ \le\ t_x + M\,\bigl(2 - s_{f,x} - z_{x'}\bigr), \qquad M = n + 1. \tag{C4}$$
Explanation: when $s_{f,x} = z_{x'} = 1$ the big-M term vanishes and we get the strict predecessor constraint $t_{x'} + 1 \le t_x$; otherwise the RHS is $\ge t_x + M$ and the constraint is trivial. When $x'$ is not a dependent variable ($z_{x'} = 0$) it is a free parameter in $Y$ and imposes no ordering.

### Objective

The problem as stated is a feasibility problem; any feasible assignment yields a valid cover. A natural minimization objective is the size of the cover,
$$\min \ \sum_{(f,x) \in A} s_{f,x}, \tag{Obj}$$
which produces the smallest set of tuples (and hence the fewest radical extensions in the parametrization). Alternatives include minimizing total algebraic degree $\sum_{(f,x) \in A} \deg_x(f)\, s_{f,x}$ to favor lower-degree (ideally linear) parametrizations.

### Summary

$$
\begin{aligned}
\min \quad & \sum_{(f,x) \in A} s_{f,x} \\
\text{s.t.} \quad
& \sum_{f \in F_j} \sum_{x\,:\,(f,x)\in A} s_{f,x} \ge 1 && \forall j, \\
& z_x = \sum_{f\,:\,(f,x)\in A} s_{f,x} && \forall x \in V, \\
& t_{x'} + 1 \le t_x + (n+1)\,(2 - s_{f,x} - z_{x'}) && \forall (f,x)\in A,\ x' \in \mathrm{vars}(f)\setminus\{x\}, \\
& s_{f,x} \in \{0,1\},\ z_x \in \{0,1\},\ t_x \in [0, n].
\end{aligned}
$$

Any optimal (or merely feasible) solution of this MILP gives a valid set of tuples $\{(f, x) : s_{f,x} = 1\}$, and by the claim proven above, a parametrization in radicals of a component of the variety of $\{p_1, \ldots, p_m\}$ can then be read off in topological order of the $t_x$ stamps.

