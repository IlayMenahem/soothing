# Problem Statement

## Problem

There is a set of multivariate polynomials $p_1, \ldots, p_m$ we do the following preprocessing:
1. interreduce the polynomials with lex order
2. find the square-free factorization of each polynomial into $f_{1, 1} \cdot \ldots \cdot f_{1, n_1}, \ldots, f_{m, 1} \cdot \ldots \cdot f_{m, n_m}$.
3. remove constant factors from the polynomials.
4. clear zero polynomials.
5. solve each factor by each variable we can solve for getting a parametrization of the variable, we will solve for all factors that are of degree 1 or 2 in the variable.
6. match the parametrizations to the polynomials they solve for.

Taking the parametrizations as given, we would like to find a set of parameterizations $x_1(\bar{v}_1), \ldots, x_k(\bar{v}_k)$ such that:
1. each polynomial is covered by a parametrization.
2. each variable has at most one parametrization.
3. the directed graph $G = (V, E)$ with $V = \{x_1, \ldots, x_k\}$ and $E = \{(x_j, x_i) \mid x_j \text{ appears in } f_i\}$ is acyclic.

Claim: If we find parameterizations $x_1(\bar{v}_1), \ldots, x_k(\bar{v}_k)$ such that the conditions are met, this parametrization is a valid solution to the system $p_1 = \cdots = p_m = 0$.

Proof:

Let $q_1, \ldots, q_r$ be the non-constant, non-zero polynomials that remain after interreduction, square-free factorization, and removal of constants and zero equations. Each preprocessing step preserves the common zero set in the following sense:
1. replacing a polynomial by its remainder modulo the others preserves the generated ideal, hence preserves the common zeros.
2. removing a non-zero constant factor does not change the equation $q = 0$.
3. removing the zero polynomial only removes a vacuous equation.
4. if $q = f_1 \cdots f_n$, then making one factor $f_i$ vanish is sufficient for making $q$ vanish.

By condition 1, every remaining polynomial $q_j$ has at least one selected factor $f_{j, \ell}$ together with a parametrization for some variable $x_i$. This parametrization was constructed to solve that factor, so after substituting
$$
x_i = x_i(\bar{v}_i)
$$
we have $f_{j, \ell} = 0$, and therefore $q_j = 0$.

It remains to show that the substitutions can be applied consistently. Condition 2 says that no variable receives two different definitions. Condition 3 says that the dependency graph between defined variables is acyclic, so it has a topological order. Evaluate the selected parametrizations in that order: whenever a variable $x_i$ is assigned, every defined variable that appears in $\bar{v}_i$ has already been assigned, and every remaining variable in $\bar{v}_i$ is free. Thus the selected parametrizations define a consistent family of values for the dependent variables in terms of the free variables. Under this substitution every $q_j$ vanishes, and therefore the original system $p_1 = \cdots = p_m = 0$ vanishes as well.

## Choosing the lex order

The lex order used for interreduction in step 1 affects which variables get eliminated from which polynomials, and hence how much structure the MILP in the next section has to work with. 
We want an order that, after interreduction, leaves polynomials sharing as few variables as possible — this both reduces the size of $A$ and makes the acyclicity constraint (C4) easier to satisfy.

Interreduction under lex behaves like sparse elimination: when a variable $v$ is high in the order, one polynomial is picked as its "definer" (the one whose leading term involves $v$) and $v$ is removed from every other polynomial by reduction. 
Choosing the order is therefore a variable-elimination ordering problem, well studied in sparse linear algebra (Markowitz / minimum-degree heuristics). 

## Solving the problem

We can solve the problem by formulating it as a mixed-integer linear program (MILP) or a SAT problem. 
The MILP is convenient because it can express both the discrete choices and the acyclicity constraint directly.

Let $F$ be the set of distinct square-free factors after preprocessing. For each preprocessed polynomial $q_j$, let $F_j \subseteq F$ be the factors whose product is $q_j$. Let $X$ be the set of variables that occur in the factors, and let
$$ A = \{(f, x) \in F \times X \mid f \text{ can be solved for } x\} $$
be the set of admissible parametrizations.

For each $(f, x) \in A$, introduce a binary variable
$$ s_{f,x} \in \{0, 1\} $$
which means that we select the parametrization obtained by solving $f = 0$ for $x$. For each variable $x \in X$, introduce a binary variable
$$ d_x \in \{0, 1\} $$
which means that $x$ is a dependent variable, and a continuous or integer variable
$$ t_x \in [0, |X|] $$
which is its topological rank.

The constraints are:

**C1. Cover every polynomial.**
For every preprocessed polynomial $q_j$,
$$ \sum_{(f,x) \in A,\ f \in F_j} s_{f,x} \ge 1 $$
At least one factor of each polynomial must be made zero.

**C2. Select at most one parametrization per variable.**
For every variable $x$,
$$ \sum_{(f,x) \in A} s_{f,x} \le 1 $$
This prevents conflicting definitions of the same variable.

**C3. Link selected parametrizations to dependent variables.**
For every variable $x$,
$$ d_x = \sum_{(f,x) \in A} s_{f,x} $$
Thus $d_x = 1$ exactly when $x$ is solved by one selected factor.

**C4. Enforce acyclicity.**
For every admissible pair $(f,x) \in A$ and every variable $y \ne x$ that appears in $f$, impose
$$ t_y + 1 \le t_x + M(2 - s_{f,x} - d_y) $$
where $M = |X| + 1$.

If $s_{f,x} = 1$ and $y$ is also a dependent variable, this becomes $t_y + 1 \le t_x$, so the dependency edge $y \to x$ must point forward in the topological order. If either the pair $(f,x)$ is not selected or $y$ is free, the constraint is relaxed by the big-$M$ term.

Any feasible solution to C1-C4 gives exactly the kind of parametrization required by the claim. The objective is not needed for correctness, but it is useful for choosing among feasible solutions. Natural objectives are:
1. minimize the number of selected parametrizations,
2. minimize the total degree of the selected equations in their solved variables,
3. combine both objectives lexicographically.

A SAT encoding uses the same binary selection variables $s_{f,x}$. C1 becomes an OR clause for each polynomial, C2 becomes an at-most-one constraint for each variable, and acyclicity can be encoded either by Boolean order variables $o_{y,x}$ or by forbidding directed cycles among the selected dependency edges. The SAT formulation is purely discrete, while the MILP formulation is often simpler because the ranks $t_x$ express the topological order with linear inequalities.

## Zeroing variables - TBA