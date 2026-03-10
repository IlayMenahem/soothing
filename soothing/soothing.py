import jax.numpy as jnp
from jaxtyping import Array
from sympy import (
    Expr,
    Integer,
    Poly,
    Symbol,
    fraction,
    symbols,
    sympify,
    together,
    zoo,
)

from soothing.numerical import newton_raphson
from soothing.symbolic import try_zero_variables
from soothing.utils import (
    collect_gens,
    interreduce_polynomials,
    solve_linear_for_var,
    sympy_to_jax_system,
)


def _get_degree(poly: Poly, var: Symbol) -> int:
    """
    Get the degree of a variable in a polynomial, handling missing generators.

    Args:
        poly (Poly): The polynomial.
        var (Symbol): The variable to get the degree of.

    Returns:
        int: The degree of var in poly, or 0 if var is not a generator.
    """
    if var not in poly.gens:
        return 0

    degree = int(poly.degree(var))  # type: ignore[arg-type]

    return degree


def _is_linear_in_var(poly: Poly, var: Symbol) -> bool:
    """
    Check if a polynomial is linear in a given variable.

    A polynomial is linear in var if the degree of var is exactly 1.

    Args:
        poly (Poly): The polynomial to check.
        var (Symbol): The variable to check linearity for.

    Returns:
        bool: True if poly is linear in var, False otherwise.
    """
    is_linear: bool = _get_degree(poly, var) == 1

    return is_linear


def _find_linear_pairs(
    polynomials: list[Poly], variables: set[Symbol]
) -> list[tuple[Symbol, Poly]]:
    """
    Find all (variable, polynomial) pairs where the polynomial is linear in the variable.

    Args:
        polynomials (list[Poly]): List of polynomials to search.
        variables (set[Symbol]): Set of variables to consider.

    Returns:
        list[tuple[Symbol, Poly]]: List of (variable, polynomial) pairs where polynomial is linear in variable.
    """
    pairs: list[tuple[Symbol, Poly]] = []

    for poly in polynomials:
        for var in variables:
            if var in poly.free_symbols and _is_linear_in_var(poly, var):
                pairs.append((var, poly))

    return pairs


def _count_occurrences(var: Symbol, polynomials: list[Poly]) -> int:
    """
    Count how many polynomials contain the given variable.

    Args:
        var (Symbol): Variable to count occurrences of.
        polynomials (list[Poly]): List of polynomials to search.

    Returns:
        int: Number of polynomials containing var.
    """
    count: int = sum(1 for p in polynomials if var in p.free_symbols)

    return count


def _clear_denominators(polynomials: list[Poly], variables: set[Symbol]) -> list[Poly]:
    """
    Clear denominators in a polynomial system by multiplying each equation
    by its denominator and keeping the numerator.

    Args:
        polynomials (list[Poly]): Polynomial system.
        variables (set[Symbol]): Variables remaining in the system.

    Returns:
        list[Poly]: Polynomials with denominators cleared.
    """
    if not polynomials:
        return []

    gens = collect_gens(polynomials, variables)
    cleared: list[Poly] = []

    for p in polynomials:
        expr = p.as_expr()
        num, den = fraction(together(expr))
        if den != 1:
            expr = num

        if expr == 0:
            continue

        if gens:
            cleared.append(Poly(expr, *gens))
        else:
            cleared.append(Poly(expr))

    return cleared


def _simplify_system(polynomials: list[Poly], variables: set[Symbol]) -> list[Poly]:
    """
    Simplify a polynomial system by clearing denominators, interreducing,
    and making polynomials monic.

    Args:
        polynomials (list[Poly]): Polynomial system.
        variables (set[Symbol]): Variables remaining in the system.

    Returns:
        list[Poly]: Simplified polynomial system.
    """
    if not polynomials:
        return []

    cleared = _clear_denominators(polynomials, variables)
    interreduced = interreduce_polynomials(cleared, variables)

    monic_polys: list[Poly] = []
    for p in interreduced:
        if p.is_zero:
            continue
        if p.total_degree() > 0:
            monic_polys.append(p.monic())
        else:
            monic_polys.append(p)

    return monic_polys


def _total_degree_in_others(var: Symbol, poly: Poly, polynomials: list[Poly]) -> int:
    """
    Calculate total degree of var across all other polynomials.

    Args:
        var (Symbol): Variable to check.
        poly (Poly): The polynomial used for elimination (excluded from count).
        polynomials (list[Poly]): All polynomials.

    Returns:
        int: Sum of degrees of var in all polynomials except poly.
    """
    total_degree: int = sum(_get_degree(p, var) for p in polynomials if p != poly)

    return total_degree


def _select_best_pair(
    pairs: list[tuple[Symbol, Poly]], polynomials: list[Poly]
) -> tuple[Symbol, Poly]:
    """
    Select the best (variable, polynomial) pair for elimination.

    Heuristic: Choose the variable with lowest total degree in other equations.
    This tends to minimize expression growth after substitution.

    Args:
        pairs (list[tuple[Symbol, Poly]]): List of candidate (variable, polynomial) pairs.
        polynomials (list[Poly]): All polynomials in the system.

    Returns:
        tuple[Symbol, Poly]: The best (variable, polynomial) pair to use for elimination.
    """
    if not pairs:
        raise ValueError("No pairs to select from")

    def score(pair: tuple[Symbol, Poly]) -> tuple[int, int]:
        """
        Score a pair first by the total degree in other equations and then by the number of occurrences.

        Args:
            pair (tuple[Symbol, Poly]): The pair to score.

        Returns:
            tuple[int, int]: The score of the pair.
        """
        var, poly = pair
        degree_score = _total_degree_in_others(var, poly, polynomials)
        occurrence_score = _count_occurrences(var, polynomials)

        score: tuple[int, int] = (degree_score, occurrence_score)

        return score

    def shortest_polynomial(pair: tuple[Symbol, Poly]) -> int:
        """
        Score a pair by the length of the polynomial.
        """
        var, poly = pair
        length = len(poly.as_expr().args)
        print(f"Length of {var} polynomial: {length}")
        return length

    best_pair: tuple[Symbol, Poly] = min(pairs, key=score)

    return best_pair


def linear_elimination(
    polynomial_system: list[Poly],
    variables: set[Symbol],
    simplify: bool = True,
    verbose: bool = True,
) -> tuple[list[Poly], dict[Symbol, Expr]]:
    r"""
    Linear elimination by substitution.

    Algorithm outline from "Parametric solutions via linear elimination and rationalization" section:

    Input: Polynomial system $P = {p_1, ..., p_k} \subseteq K[x_1, ..., x_m]$
    Output: Reduced system $Q$ where no $q \in Q$ is linear in any remaining variable; and a substitution map $\Sigma$ to recover eliminated variables

    Heuristics for selecting which variable to eliminate:
    - Lowest degree in the other equations
    - Lowest degree after substitution
    - Greatest or least number of occurrences

    Args:
    - polynomial_system (list[Poly]): List of polynomials $P = {p_1, ..., p_k}$
    - variables (set[Symbol]): Set of variables $V = {x_1, ..., x_m}$
    - simplify (bool): If True, apply optional simplification steps
      (clear denominators, interreduce, make monic).
    - verbose (bool): If True, print progress information

    Returns:
    - Tuple of (reduced_system (list[Poly]), substitution_map (dict))
    """

    def substitute_var(
        q: Poly, var: Symbol, expr: Expr, all_gens_tuple: tuple[Symbol, ...]
    ) -> Poly:
        """
        Substitute var -> expr in a polynomial.

        Args:
            q (Poly): The polynomial to substitute in.
            var (Symbol): The variable to substitute.
            expr (Expr): The expression to substitute var with.
            all_gens_tuple (tuple[Symbol, ...]): The generators of the polynomials.

        Returns:
            Poly: The polynomial with var substituted with expr.
        """
        new_expr: Expr = q.as_expr().subs(var, expr)
        num, den = fraction(together(new_expr))
        if den != 1:
            new_expr = num
        if new_expr != 0 and all_gens_tuple:
            return Poly(new_expr, *all_gens_tuple)
        elif new_expr != 0:
            return q
        return q

    gens = tuple(sorted(variables, key=str))
    Q: list[Poly] = []
    for p in list[Poly](polynomial_system):
        if isinstance(p, Poly):
            Q.append(p)
        else:
            expr = sympify(p)
            if gens:
                Q.append(Poly(expr, *gens))
            else:
                Q.append(Poly(expr))
    V = set[Symbol](variables)
    substitution_map: dict[Symbol, Expr] = {}

    if simplify:
        Q = _simplify_system(Q, V)

    while True:
        linear_pairs: list[tuple[Symbol, Poly]] = _find_linear_pairs(Q, V)

        if not linear_pairs:
            break

        var, poly = _select_best_pair(linear_pairs, Q)

        if verbose:
            print(f"Eliminating variable: {var}")

        expr: Expr = solve_linear_for_var(poly, var)
        substitution_map[var] = expr
        Q = [q for q in Q if q != poly]

        all_gens_tuple: tuple[Symbol, ...] = collect_gens(Q, V)

        Q = [substitute_var(q, var, expr, all_gens_tuple) for q in Q]
        Q = [q for q in Q if not q.is_zero]
        V.discard(var)

        if simplify:
            Q = _simplify_system(Q, V)

    return Q, substitution_map


def _count_linear_pairs_after_zeroing(
    polynomials: list[Poly], variables: set[Symbol], var_to_zero: Symbol
) -> int:
    """
    Count how many linear pairs would exist if var_to_zero were set to 0.

    This is used as a heuristic to select which variable to zero:
    we prefer zeroing variables that create more linear elimination opportunities.

    Args:
    - polynomials (list[Poly]): Current polynomial system.
    - variables (set[Symbol]): Set of remaining variables.
    - var_to_zero (Symbol): Variable to hypothetically set to 0.

    Returns:
    - int: Number of (var, poly) linear pairs that would exist after zeroing.
    """
    # Substitute var_to_zero = 0 in all polynomials
    hypothetical_polys = []
    for p in polynomials:
        new_expr = p.as_expr().subs(var_to_zero, 0)
        if new_expr != 0:
            # Get remaining generators
            remaining_gens = [g for g in p.gens if g != var_to_zero]
            if remaining_gens:
                hypothetical_polys.append(Poly(new_expr, *remaining_gens))

    remaining_vars = variables - {var_to_zero}
    pairs = _find_linear_pairs(hypothetical_polys, remaining_vars)

    return len(pairs)


def _select_best_to_zero(
    zeroable: list[tuple[int, Array, float]],
    polynomials: list[Poly],
    var_list: list[Symbol],
    variables: set[Symbol],
) -> list[tuple[int, Symbol, Array, float, int]]:
    """
    Select the best variable to zero based on heuristics.

    Heuristic: Choose the variable that creates the most linear pairs
    after zeroing, to maximize opportunities for further linear elimination.

    Args:
    - zeroable (list[tuple[int, Array, float]]): List of (idx, new_solution, error) from try_zero_variables.
    - polynomials (list[Poly]): Current polynomial system.
    - var_list (list[Symbol]): Ordered list of variables (maps indices to symbols).
    - variables (set[Symbol]): Set of remaining variables.

    Returns:
    - list[tuple[int, Symbol, Array, float, int]]: List of tuples containing (index, symbol, new_solution, error, score) for the best variable to zero.
    """
    ranking: list[tuple[int, Symbol, Array, float, int]] = []
    scores: list[tuple[int, Symbol, Array, float, int]] = []

    for idx, new_solution, error in zeroable:
        var = var_list[idx]
        score = _count_linear_pairs_after_zeroing(polynomials, variables, var)
        scores.append((idx, var, new_solution, error, score))

    ranking = sorted(scores, key=lambda x: (-x[4], x[3]))

    return ranking


def soothing(
    polynomial_system: list[Poly],
    variables: set[Symbol],
    tol: float = 1e-10,
    max_iterations: int = 100,
    verbose: bool = True,
    simplify: bool = True,
) -> tuple[list[Poly], dict]:
    r"""
    Parametric solution via linear elimination and variable zeroing.

    Algorithm outline from "Parametric solutions via linear elimination and rationalization" section:

    Input: Polynomial system $P = {p_1, ..., p_k} \subseteq \mathbb{K}[x_1, ..., x_m]$
    Output: Parametric solution $Q$ with some variables expressed symbolically in terms of free variables

    The algorithm alternates between:
    1. Linear elimination: symbolically eliminate variables that appear linearly
    2. Variable zeroing: numerically find which variables can be set to 0

    Heuristics for choosing which variable to zero:
    - Most linear variables after substitution: choose the variable that will lead to
      the most equations being linear in some variable after substitution

    Args:
    - polynomial_system (list[Poly]): List of polynomials $P = {p_1, ..., p_k}$
    - variables (set[Symbol]): Set of variables $V = {x_1, ..., x_m}$
    - tol (float): Tolerance for considering a zeroing successful
    - max_iterations (int): Maximum number of zeroing iterations
    - verbose (bool): If True, print progress information
    - simplify (bool): If True, apply optional simplification steps
      (clear denominators, interreduce, make monic).

    Returns:
    - Tuple of (parametric_solution (list[Poly]), substitution_map (dict))
    """

    def add_to_substitution_map(
        subs_map: dict[Symbol, Expr], var: Symbol, expr: Expr
    ) -> dict[Symbol, Expr]:
        """
        Add a new substitution to the substitution map and update existing substitutions.

        Args:
            subs_map (dict[Symbol, Expr]): Current substitution map.
            var (Symbol): Variable being substituted.
            expr (Expr): Expression that var is substituted with.

        Returns:
            dict[Symbol, Expr]: Updated substitution map with var -> expr and all existing substitutions updated to reflect this new substitution.
        """
        new_map = {var: expr}
        for existing_var, existing_expr in subs_map.items():
            if var in existing_expr.free_symbols:
                new_expr = existing_expr.subs(var, expr)
                new_map[existing_var] = new_expr
            else:
                new_map[existing_var] = existing_expr

        return new_map

    # Step 1: Initial linear elimination
    Q, substitution_map = linear_elimination(
        polynomial_system, variables, simplify=simplify
    )
    V = variables - set[Symbol](substitution_map.keys())

    if verbose:
        print("After initial linear elimination:")
        print(
            f"  Eliminated {len(substitution_map)} variables: {list[Symbol](substitution_map.keys())}"
        )
        print(f"  Remaining equations: {Q}")
        print(f"  Remaining variables: {V}")

    if not Q or not V:
        return Q, substitution_map

    # Create ordered variable list for index mapping
    var_list = sorted(list[Symbol](V), key=lambda x: str(x))

    for iteration in range(max_iterations):
        if verbose:
            print(f"\nIteration {iteration + 1}:")

        # Convert current system to JAX for numerical operations
        jax_system = sympy_to_jax_system(Q, var_list)

        # Find a numerical solution using Newton-Raphson
        initial_guess = jnp.ones(len(var_list))
        solution, errors, _ = newton_raphson(jax_system, initial_guess, tol=tol)

        final_error = (
            float(errors[errors < jnp.inf].min())
            if jnp.any(errors < jnp.inf)
            else float("inf")
        )
        if verbose:
            print(f"  Found numerical solution with error: {final_error:.2e}")
            print(f"  Solution: {solution}")

        # Try to zero each variable
        zeroable = try_zero_variables(
            solution, jax_system, variable_indices=list(range(len(var_list))), tol=tol
        )

        if verbose:
            print(f"  Zeroable variables: {len(zeroable)}")
            for idx, _, err in zeroable:
                print(f"    {var_list[idx]}: error = {err:.2e}")

        if not zeroable:
            if verbose:
                print("  No variables can be zeroed. Stopping.")
            break

        # Select the best variable to zero
        ranking = _select_best_to_zero(zeroable, Q, var_list, V)

        valid_zeroing_found = False
        var_to_zero = ranking[0][1]
        for idx, var_to_zero_cand, new_solution, _, _ in ranking:
            causes_zoo = False
            new_substitution = add_to_substitution_map(
                substitution_map, var_to_zero_cand, Integer(0)
            )
            for var, expr in new_substitution.items():
                if expr.has(zoo):
                    causes_zoo = True
                    break

            if not causes_zoo:
                var_to_zero = var_to_zero_cand
                valid_zeroing_found = True
                break

        if not valid_zeroing_found:
            if verbose:
                print("  All zeroable variables cause sympy.zoo. Stopping.")
            break

        if verbose:
            print(f"  Zeroing variable: {var_to_zero}")

        # Record the zeroing in substitution map
        substitution_map[var_to_zero] = Integer(0)

        # Update the rest of the substitution map to substitute var_to_zero = 0
        for var, expr in substitution_map.items():
            if var_to_zero in expr.free_symbols:
                new_expr = expr.subs(var_to_zero, 0)
                substitution_map[var] = new_expr

        # Substitute z=0 in all polynomials
        Q = [
            Poly(
                q.as_expr().subs(var_to_zero, 0),
                *[g for g in q.gens if g != var_to_zero],
            )
            for q in Q
        ]

        # Remove zero polynomials
        Q = list[Poly](filter(lambda q: not q.is_zero, Q))

        # Update variable set and list
        V.discard(var_to_zero)
        var_list = [v for v in var_list if v != var_to_zero]

        if not Q or not V:
            if verbose:
                print("  System fully solved or no variables remain.")
            break

        if simplify:
            Q = _simplify_system(Q, V)

        # Re-run linear elimination on the simplified system
        Q, new_sigma = linear_elimination(Q, V, simplify=simplify)

        if new_sigma:
            if verbose:
                print(
                    f"  Linear elimination found {len(new_sigma)} more eliminations: {list[Symbol](new_sigma.keys())}"
                )
            substitution_map.update(new_sigma)
            V = V - set[Symbol](new_sigma.keys())
            var_list = [v for v in var_list if v not in new_sigma]

        if verbose:
            print(f"  Remaining equations: {Q}")
            print(f"  Remaining variables: {V}")

        if not Q or not V:
            if verbose:
                print("  System fully solved after linear elimination.")
            break

    return Q, substitution_map


if __name__ == "__main__":
    import jax
    from sympy import Poly, symbols

    from soothing.soothing import soothing

    jax.config.update("jax_enable_x64", True)

    x0, x1, x2, x3, x4 = symbols("x0, x1, x2, x3, x4")
    system: list[Poly] = [
        Poly(x0**3 - 2 * x1 + x2 + x3 - 2 * x4**3 - 1),
        Poly(x1**2 + x2**2 - x0 * x3 + 4 * x4**2 - 7),
    ]

    # solve using soothing
    solution, substitution_map = soothing(system, {x0, x1, x2, x3, x4})
    print("Solution:", solution)
    print("\nSubstitution map:")
    for key, value in substitution_map.items():
        print(f"{key} -> {value}")
