# Problem Statement

## Problem

There is a set of multivariate polynomials $p_1, \ldots, p_m$ we do the following preprocessing:
1. interreduce the polynomials with lex order getting new polynomials $p'_1, \ldots, p'_{m}$
2. clear zero polynomials
3. find the factorization of each polynomial into $\{(f, \{p'_j \mid f \text{ appears in } p'_j\}) \mid f \text{ is an irreducible factor of at least one polynomial}\}$.
4. remove constant factors from the polynomials

Taking the preprocessed polynomials as given, we would like to find a set of tuples of factors and variables $(f_1, x_1), (f_2, x_2), \ldots, (f_k, x_k)$ and a set of variables to zero $v_1, \ldots, v_l$, maximizing the number of preprocessed polynomials that become identically zero after zeroing $v_1, \ldots, v_l$, subject to the following conditions:
1. each polynomial has a factor $f$ such that $s_{f,x} = 1$ for some variable $x$ (i.e., $f$ is selected as the linear definer for some variable), or has a factor that becomes identically zero after zeroing $v_1, \ldots, v_l$.
2. for each tuple $(f_i, x_i)$, after zeroing the variables $v_1, \ldots, v_l$, the $f_i$ is of degree 1 in $x_i$.
3. The dependency graph $G = (V, E)$ with $V = \{x_1, \ldots, x_k\}$ and $E = \{(x_j, x_i) \mid j \neq i \text{ and } x_j \text{ appears in } f_i\}$ is acyclic.
4. The variables $x_1, \ldots, x_k, v_1, \ldots, v_l$ are pairwise distinct.

### Implication of system being solved

Claim: If we find tuples $(f_1, x_1), (f_2, x_2), \ldots, (f_k, x_k)$ and variables to zero $v_1, \ldots, v_l$ such that the conditions are met, we can find a rational parametrization of a non-empty component of the solution set of the system $p_1 = \cdots = p_m = 0$, with the variables outside $\{x_1, \ldots, x_k, v_1, \ldots, v_l\}$ acting as free parameters.

Proof: The proof will be of two parts, first showing the construction of the rational parametrization that zeros each of the $f_i$, and the second showing that the resulting parametrization zeros each of the $p_j$.

First part - By condition 2, for each $i$, the factor $f_i$ is exactly of degree 1 in $x_i$ after zeroing $v_1, \ldots, v_l$, so we can write $f_i = a_i x_i + b_i$ where $a_i$ and $b_i$ are rational functions in the variables other than $x_i$.
$x_i = -\frac{b_i}{a_i}$ is a rational parametrization of the variety defined by $f_i = 0$.
Because the dependency graph is acyclic, we can topologically sort the variables $x_1, \ldots, x_k$ and substitute the rational parametrizations in order, starting with the variables with no incoming edges.

Second part - By the coverage condition, for each $p'_j$, either it has a factor that becomes identically zero due to zeroing $v_1, \ldots, v_l$, or it has a factor $f_i$ that is used to rationally define $x_i$. 
In the latter case, the substitution $x_i = -\frac{b_i}{a_i}$ forces $f_i = 0$. In either case, the entire substituted polynomial $p'_j$ evaluates to zero. 
We can also note that because $p'_1, \ldots, p'_m$ are the result of interreducing the original system of equations $p_1, \ldots, p_m$, for each original equation $p_i$, there are some polynomials $h_{i1}, \ldots, h_{im}$ such that $p_i = h_{i1}p'_1 + \ldots + h_{im}p'_m$. 
Thus, a solution to the system of equations $p'_1, \ldots, p'_m$ is also a solution to the system of equations $p_1, \ldots, p_m$.


## Solving the problem

We formulate the problem as a mixed-integer linear program (MILP).

### Notation

Let $P$ be the set of preprocessed polynomials, $F$ the set of factors and $V$ the set of variables. For $p \in P$, let $F_p \subseteq F$ be the factors of $p$. For $f \in F$, let $\mathrm{mon}(f)$ be its monomials and $\mathrm{vars}(f) \subseteq V$ the variables appearing in $f$. For a monomial $m$, let $\mathrm{vars}(m)$ be its variables and $\deg_x(m)$ its degree in $x$. Let $M^d_{f,x} = \{m \in \mathrm{mon}(f) : \deg_x(m) = d\}$.

### Variables

Decision variables:

- $s_{f, x} \in \{0, 1\}$ for each $(f, x) \in F \times V$. $s_{f, x} = 1$ iff the factor $f$ is selected as the linear definer of the variable $x$.
- $z_v \in \{0, 1\}$ for each $v \in V$. $z_v = 1$ iff the variable $v$ is zeroed.

Auxiliary indicator variables (continuous in $[0, 1]$; pinned to $\{0,1\}$ by the constraints below):

- $\zeta_f \in [0, 1]$ for each $f \in F$. $\zeta_f = 1$ iff $f$ becomes identically zero after zeroing $\{v : z_v = 1\}$.
- $y_p \in [0, 1]$ for each $p \in P$. $y_p = 1$ iff $p$ becomes identically zero after zeroing.
- $a_{f, x, m} \in [0, 1]$ for each $(f, x) \in F \times V$ and each $m \in M^1_{f, x}$. $a_{f, x, m} = 1$ indicates that monomial $m$ (which has degree 1 in $x$) survives the zeroing.
- $t_x \in [0, |V|]$ for each $x \in V$. Topological position of $x$, used to enforce acyclicity.

### Factor-zeroing definition

A factor $f$ is identically zero after zeroing iff every monomial of $f$ contains at least one zeroed variable. Encoded as, for each $f \in F$ and each $m \in \mathrm{mon}(f)$:

$$\zeta_f \le \sum_{v \in \mathrm{vars}(m)} z_v.$$

(Maximization in the objective drives $\zeta_f$ to its upper bound, which is $1$ exactly when every monomial has a zeroed variable.)

### Constraints

**(C1) Coverage.** Every preprocessed polynomial $p \in P$ must have either a factor that is selected as a linear definer for some variable, or a factor that becomes identically zero after zeroing:

$$\sum_{f \in F_p} \left( \zeta_f + \sum_{x \in V} s_{f, x} \right) \ge 1 \qquad \forall\, p \in P.$$

**(C2) Linearity.** For each selected pair $(f, x)$, after zeroing, $f$ must be of degree at most $1$ in $x$ with at least one degree-$1$ monomial surviving.

(C2a) Every monomial of $f$ of degree $\ge 2$ in $x$ must be killed by some zeroed variable other than $x$. For each $(f, x) \in F \times V$ and each $m \in \mathrm{mon}(f)$ with $\deg_x(m) \ge 2$:

$$s_{f, x} \le \sum_{v \in \mathrm{vars}(m) \setminus \{x\}} z_v.$$

(C2b) A monomial $m$ of degree exactly $1$ in $x$ survives only if none of its non-$x$ variables are zeroed. For each $(f, x) \in F \times V$, each $m \in M^1_{f, x}$, and each $v \in \mathrm{vars}(m) \setminus \{x\}$:

$$a_{f, x, m} \le 1 - z_v.$$

(C2c) If $f$ is selected for $x$, at least one degree-$1$-in-$x$ monomial must survive. For each $(f, x) \in F \times V$:

$$s_{f, x} \le \sum_{m \in M^1_{f, x}} a_{f, x, m}.$$

(If $M^1_{f, x} = \emptyset$ — including the case $x \notin \mathrm{vars}(f)$ — this forces $s_{f, x} = 0$.)

**(C3) Acyclicity.** Use a topological-order encoding on the potential vertices. For each $(f, x) \in F \times V$ and each $x' \in \mathrm{vars}(f) \setminus \{x\}$:

$$t_{x'} + 1 \le t_x + (|V| + 1)\,(1 - s_{f, x}).$$

When $s_{f, x} = 1$ this enforces $t_{x'} < t_x$ on every potential predecessor edge $x' \to x$ of the dependency graph; when $s_{f, x} = 0$ the big-$M$ term deactivates the constraint. A consistent assignment of $\{t_x\}$ exists iff the induced graph on the selected variables is acyclic.

**(C4) Pairwise distinct.** No variable is both selected (as some $x_i$) and zeroed (as some $v_j$), and each variable is selected at most once. For each $x \in V$:

$$\sum_{f \in F} s_{f, x} + z_x \le 1.$$

**(C5) Monomial factors.** A factor $f$ with $|\mathrm{mon}(f)| = 1$ (i.e. $f$ is a single monomial) cannot rationally define any variable: the equation $f = 0$ merely forces some variable in the product to zero but does not single out any one variable. Therefore:

$$s_{f, x} = 0 \qquad \forall\, x \in V, \text{ whenever } |\mathrm{mon}(f)| = 1.$$

In practice, such factors are excluded from the admissible set entirely, so no $s_{f,x}$ variable is ever created for them.

### Objective

A polynomial $p$ becomes identically zero after zeroing iff at least one of its factors does. Link the objective indicator to the factor indicators: for each $p \in P$,

$$y_p \le \sum_{f \in F_p} \zeta_f.$$

Maximize the number of preprocessed polynomials that become identically zero:

$$\max \; \sum_{p \in P} y_p.$$
