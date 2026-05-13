# Problem Statement

## Problem

There is a set of multivariate polynomials $p_1, \ldots, p_m$ we do the following preprocessing:
1. interreduce the polynomials with lex order getting new polynomials $p'_1, \ldots, p'_{m}$
2. clear zero polynomials
3. find the factorization of each polynomial into $\{(f, \{p'_j \mid f \text{ appears in } p'_j\}) \mid f \text{ is an irreducible factor of at least one polynomial}\}$.
4. remove constant factors from the polynomials

Taking the preprocessed polynomials as given, we would like to find a set of tuples of factors and variables $(f_1, x_1), (f_2, x_2), \ldots, (f_k, x_k)$ and a set of variables to zero $v_1, \ldots, v_l$, maximizing the number of preprocessed polynomials that become identically zero after zeroing $v_1, \ldots, v_l$, subject to the following conditions:
1. each polynomial has a factor $f$ such that $s_{f,x} = 1$ for some variable $x$ (i.e., $f$ is selected as the algebraic definer for some variable), or has a factor that becomes identically zero after zeroing $v_1, \ldots, v_l$.
2. for each tuple $(f_i, x_i)$, after zeroing the variables $v_1, \ldots, v_l$, $f_i$ has positive degree in $x_i$ (i.e. some monomial of $f_i$ of positive degree in $x_i$ survives the zeroing).
3. The dependency graph $G = (V, E)$ with $V = \{x_1, \ldots, x_k\}$ and $E = \{(x_j, x_i) \mid j \neq i \text{ and } x_j \text{ appears in } f_i\}$ is acyclic.
4. The variables $x_1, \ldots, x_k, v_1, \ldots, v_l$ are pairwise distinct.

### Implication of system being solved

Claim: If we find tuples $(f_1, x_1), (f_2, x_2), \ldots, (f_k, x_k)$ and variables to zero $v_1, \ldots, v_l$ such that the conditions are met, we can find an algebraic parametrization of a non-empty component of the solution set of the system $p_1 = \cdots = p_m = 0$, with the variables outside $\{x_1, \ldots, x_k, v_1, \ldots, v_l\}$ acting as free parameters.

Proof: The proof will be of two parts, first showing the construction of the algebraic parametrization that zeros each of the $f_i$, and the second showing that the resulting parametrization zeros each of the $p_j$.

First part - By condition 2, for each $i$, after zeroing $v_1, \ldots, v_l$ the factor $f_i$ has positive degree $d_i \ge 1$ in $x_i$, so we can write $f_i = \sum_{j=0}^{d_i} c_j^{(i)} x_i^j$ where the coefficients $c_j^{(i)}$ are polynomials in the variables other than $x_i$, with $c_{d_i}^{(i)} \not\equiv 0$. Hence $f_i$ defines $x_i$ as an algebraic function of the remaining variables — its zero locus is a finite cover of degree $d_i$ over the locus where $c_{d_i}^{(i)} \ne 0$. (For $d_i \in \{1,2,3,4\}$ this algebraic function is expressible by radicals via the linear formula, the quadratic formula, Cardano's cubic formula, or Ferrari's quartic formula; for $d_i \ge 5$ the algebraic function still exists but generically requires non-radical representation. The existence of the parametrization does not depend on a radical form.)
Because the dependency graph is acyclic, we can topologically sort the variables $x_1, \ldots, x_k$ and substitute the algebraic expressions in order, starting with the variables with no incoming edges.

Second part - By the coverage condition, for each $p'_j$, either it has a factor that becomes identically zero due to zeroing $v_1, \ldots, v_l$, or it has a factor $f_i$ that is used to algebraically define $x_i$. 
In the latter case, the substitution of the algebraic expression for $x_i$ forces $f_i = 0$. In either case, the entire substituted polynomial $p'_j$ evaluates to zero. 
We can also note that because $p'_1, \ldots, p'_m$ are the result of interreducing the original system of equations $p_1, \ldots, p_m$, for each original equation $p_i$, there are some polynomials $h_{i1}, \ldots, h_{im}$ such that $p_i = h_{i1}p'_1 + \ldots + h_{im}p'_m$. 
Thus, a solution to the system of equations $p'_1, \ldots, p'_m$ is also a solution to the system of equations $p_1, \ldots, p_m$.

### Optional symbolic certification

The argument above shows existence of the parametrization wherever each leading coefficient $c_{d_i}^{(i)}$ is nonzero. To certify that this open locus is non-empty (i.e. to rule out the case where the parametrization collapses on the entire parameter space due to spurious components), one can perform the following symbolic check after the MILP returns a solution:

1. Substitute $v_j = 0$ for every zeroed variable into $\{f_1, \ldots, f_k\}$.
2. Compute the lex Gröbner basis of $\{f_1, \ldots, f_k\}$ with variable order $x_k > x_{k-1} > \cdots > x_1 > \text{free parameters}$ (e.g. via `symbolica/groebner.rs::groebner_basis_from_system_lex_q`).
3. By acyclicity, the input is already a triangular set, so the resulting lex GB has (generically) the shape $\{g_1(x_1, \text{params}),\, g_2(x_1, x_2, \text{params}),\, \ldots,\, g_k(x_1, \ldots, x_k, \text{params})\}$, with each $g_i$ of positive degree in $x_i$ up to a leading coefficient $\ell_i$ in the lower variables and parameters.
4. The parametrization is valid on the open set where every $\ell_i \neq 0$. Non-emptiness reduces to checking that the saturation $\langle f_1,\ldots,f_k\rangle : (\prod_i \ell_i)^\infty$ is not the unit ideal — equivalently, that the GB contains no nonzero element of the parameter subring.

This step is independent of the degree of any $f_i$ in $x_i$ and replaces the role formerly played by the degree $\le 4$ restriction (which guaranteed solvability by radicals).

### Measuring capability

If the method finds tuples $(f_1, x_1), (f_2, x_2), \ldots, (f_k, x_k)$ and variables to zero $v_1, \ldots, v_l$ such that the conditions are met, we can take a point in the solution $p$ and find the full dimension of the solution by computing #variables - rank of the Jacobian of the system at $p$. A witness point $p$ can be obtained from the triangular set $\{f_1, \ldots, f_k\}$ (after substituting $v_j = 0$) by numerical homotopy or regular-chain real-root isolation; closed-form radical evaluation of the $x_i$ is not required.

## Solving the problem

We formulate the problem as a mixed-integer linear program (MILP).

### Notation

Let $P$ be the set of preprocessed polynomials, $F$ the set of factors and $V$ the set of variables. For $p \in P$, let $F_p \subseteq F$ be the factors of $p$. For $f \in F$, let $\mathrm{mon}(f)$ be its monomials and $\mathrm{vars}(f) \subseteq V$ the variables appearing in $f$. For a monomial $m$, let $\mathrm{vars}(m)$ be its variables and $\deg_x(m)$ its degree in $x$. Let $M^{\ge 1}_{f,x} = \{m \in \mathrm{mon}(f) : \deg_x(m) \ge 1\}$.

### Variables

Decision variables:

- $s_{f, x} \in \{0, 1\}$ for each $(f, x) \in F \times V$. $s_{f, x} = 1$ iff the factor $f$ is selected as the algebraic definer of the variable $x$.
- $z_v \in \{0, 1\}$ for each $v \in V$. $z_v = 1$ iff the variable $v$ is zeroed.

Auxiliary indicator variables (continuous in $[0, 1]$; pinned to $\{0,1\}$ by the constraints below):

- $\zeta_f \in [0, 1]$ for each $f \in F$. $\zeta_f = 1$ iff $f$ becomes identically zero after zeroing $\{v : z_v = 1\}$.
- $y_p \in [0, 1]$ for each $p \in P$. $y_p = 1$ iff $p$ becomes identically zero after zeroing.
- $a_{f, x, m} \in [0, 1]$ for each $(f, x) \in F \times V$ and each $m \in M^{\ge 1}_{f, x}$. $a_{f, x, m} = 1$ indicates that monomial $m$ (which has positive degree in $x$) survives the zeroing.
- $t_x \in [0, |V|]$ for each $x \in V$. Topological position of $x$, used to enforce acyclicity.

### Factor-zeroing definition

A factor $f$ is identically zero after zeroing iff every monomial of $f$ contains at least one zeroed variable. Encoded as, for each $f \in F$ and each $m \in \mathrm{mon}(f)$:

$$\zeta_f \le \sum_{v \in \mathrm{vars}(m)} z_v.$$

(Maximization in the objective drives $\zeta_f$ to its upper bound, which is $1$ exactly when every monomial has a zeroed variable.)

### Constraints

**(C1) Coverage.** Every preprocessed polynomial $p \in P$ must have either a factor that is selected as an algebraic definer for some variable, or a factor that becomes identically zero after zeroing:

$$\sum_{f \in F_p} \left( \zeta_f + \sum_{x \in V} s_{f, x} \right) \ge 1 \qquad \forall\, p \in P.$$

**(C2′) Non-degenerate definer.** For each selected pair $(f, x)$, after zeroing, $f$ must have positive degree in $x$: at least one monomial of $f$ of positive degree in $x$ must survive. This is exactly the condition $c_{d_i}^{(i)} \not\equiv 0$ in the proof, and replaces the former $1 \le d_i \le 4$ bounded-degree restriction (the upper bound was needed only for radical solvability, not for the existence of the parametrization).

(C2′a) A monomial $m \in M^{\ge 1}_{f, x}$ survives only if none of its non-$x$ variables are zeroed. For each $(f, x) \in F \times V$, each $m \in M^{\ge 1}_{f, x}$, and each $v \in \mathrm{vars}(m) \setminus \{x\}$:

$$a_{f, x, m} \le 1 - z_v.$$

(C2′b) If $f$ is selected for $x$, at least one monomial of positive degree in $x$ must survive. For each $(f, x) \in F \times V$:

$$s_{f, x} \le \sum_{m \in M^{\ge 1}_{f, x}} a_{f, x, m}.$$

(If $M^{\ge 1}_{f, x} = \emptyset$ — including the case $x \notin \mathrm{vars}(f)$ — this forces $s_{f, x} = 0$.)

**(C3) Acyclicity.** Use a topological-order encoding on the potential vertices. For each $(f, x) \in F \times V$ and each $x' \in \mathrm{vars}(f) \setminus \{x\}$:

$$t_{x'} + 1 \le t_x + (|V| + 1)\,(1 - s_{f, x}).$$

When $s_{f, x} = 1$ this enforces $t_{x'} < t_x$ on every potential predecessor edge $x' \to x$ of the dependency graph; when $s_{f, x} = 0$ the big-$M$ term deactivates the constraint. A consistent assignment of $\{t_x\}$ exists iff the induced graph on the selected variables is acyclic.

**(C4) Pairwise distinct.** No variable is both selected (as some $x_i$) and zeroed (as some $v_j$), and each variable is selected at most once. For each $x \in V$:

$$\sum_{f \in F} s_{f, x} + z_x \le 1.$$

**(C5) Monomial factors.** A factor $f$ with $|\mathrm{mon}(f)| = 1$ (i.e. $f$ is a single monomial) cannot algebraically define any variable: the equation $f = 0$ merely forces some variable in the product to zero but does not single out any one variable. Therefore:

$$s_{f, x} = 0 \qquad \forall\, x \in V, \text{ whenever } |\mathrm{mon}(f)| = 1.$$

In practice, such factors are excluded from the admissible set entirely, so no $s_{f,x}$ variable is ever created for them.

### Objective

A polynomial $p$ becomes identically zero after zeroing iff at least one of its factors does. Link the objective indicator to the factor indicators: for each $p \in P$,

$$y_p \le \sum_{f \in F_p} \zeta_f.$$

The primary objective is to maximize the number of preprocessed polynomials that become identically zero. As a secondary (tiebreaking) objective, we maximize the number of **free variables** — variables that are neither selected ($x_i$) nor zeroed ($v_j$), i.e., the variables outside $\{x_1,\ldots,x_k,v_1,\ldots,v_l\}$ that act as free parameters in the algebraic parametrization.

The two objectives are combined with a weight $\varepsilon = \frac{1}{|V|+1}$ on the secondary term so that the secondary never overrides a unit difference in the primary (since the primary is integer-valued and the secondary term is bounded by $|V| \cdot \varepsilon < 1$):

$$\max \; \sum_{p \in P} y_p \;-\; \frac{1}{|V|+1}\!\left(\sum_{(f,x)\,\in\,\text{admissible}} s_{f,x} + \sum_{v \in V} z_v\right).$$

Equivalently, ties in the primary objective are broken by minimizing the total number of selected and zeroed variables, i.e., maximizing the number of free parameters.
