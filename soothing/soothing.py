from sympy import Symbol, Poly, symbols, solve, Expr, Integer, fraction, together, sympify
from sympy.polys.polytools import reduced
import jax.numpy as jnp
from jaxtyping import Array

from soothing.numerical import newton_raphson
from soothing.symbolic import try_zero_variables
from soothing.utils import sympy_to_jax_system


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


def _find_linear_pairs(polynomials: list[Poly], variables: set[Symbol]) -> list[tuple[Symbol, Poly]]:
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


def _solve_linear_for_var(poly: Poly, var: Symbol) -> Expr:
    """
    Solve a polynomial that is linear in var for var.
    
    Given poly = a*var + b = 0 (where a, b may depend on other variables),
    returns var = -b/a.
    
    Args:
        poly (Poly): Polynomial linear in var.
        var (Symbol): Variable to solve for.
        
    Returns:
        Expr: Expression for var in terms of other variables.
    """
    expr: Expr = poly.as_expr()
    solutions: list[Expr] = solve(expr, var)
    
    if not solutions:
        raise ValueError(f"Could not solve {poly} for {var}")
    
    return solutions[0]


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


def _collect_gens(polynomials: list[Poly], variables: set[Symbol]) -> tuple[Symbol, ...]:
    """
    Collect all generators from the polynomials and variable set.
    
    Args:
        polynomials (list[Poly]): The polynomials to collect generators from.
        variables (set[Symbol]): Set of variables considered in the system.
        
    Returns:
        tuple[Symbol, ...]: The generators sorted by name.
    """
    gens: set[Symbol] = set[Symbol](variables)
    
    for p in polynomials:
        gens.update(set[Symbol](p.gens))  # type: ignore[arg-type]
    
    return tuple[Symbol, ...](sorted(gens, key=str))


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
    
    gens = _collect_gens(polynomials, variables)
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


def _interreduce_polynomials(polynomials: list[Poly], variables: set[Symbol]) -> list[Poly]:
    """
    Interreduce a polynomial system using polynomial reduction (grevlex order).
    
    Args:
        polynomials (list[Poly]): Polynomial system.
        variables (set[Symbol]): Variables remaining in the system.
        
    Returns:
        list[Poly]: Interreduced polynomial system.
    """
    if not polynomials:
        return []
    
    gens = _collect_gens(polynomials, variables)
    if not gens:
        return polynomials
    
    reduced_polys: list[Poly] = []
    
    for idx, poly in enumerate[Poly](polynomials):
        others: list[Expr] = [p.as_expr() for j, p in enumerate[Poly](polynomials) if j != idx]
        if others:
            _, reduced_expr = reduced(poly.as_expr(), others, *gens, order="grevlex")
        else:
            reduced_expr = poly.as_expr()
        
        if reduced_expr != 0:
            reduced_polys.append(Poly(reduced_expr, *gens))
    
    return reduced_polys


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
    interreduced = _interreduce_polynomials(cleared, variables)
    
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


def _select_best_pair(pairs: list[tuple[Symbol, Poly]], polynomials: list[Poly]) -> tuple[Symbol, Poly]:
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
    def substitute_var(q: Poly, var: Symbol, expr: Expr, all_gens_tuple: tuple[Symbol, ...]) -> Poly:
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
            print(f"Selecting best pair: {var} -> {poly}")

        expr: Expr = _solve_linear_for_var(poly, var)
        if verbose:
            print(f"Solved {var} = {expr}")
        substitution_map[var] = expr
        Q = [q for q in Q if q != poly]
        
        all_gens_tuple: tuple[Symbol, ...] = _collect_gens(Q, V)
        
        Q = [substitute_var(q, var, expr, all_gens_tuple) for q in Q]
        Q = [q for q in Q if not q.is_zero]
        V.discard(var)
        
        if simplify:
            Q = _simplify_system(Q, V)
    
    return Q, substitution_map


def _count_linear_pairs_after_zeroing(
    polynomials: list[Poly], 
    variables: set[Symbol], 
    var_to_zero: Symbol
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
) -> tuple[int, Symbol, Array]:
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
    - tuple[int, Symbol, Array]: Tuple of (index, symbol, new_solution) for the best variable to zero.
    """
    if not zeroable:
        raise ValueError("No zeroable variables")
    
    best: tuple[int, Symbol, Array, float] | None = None
    best_score = -1
    
    for idx, new_solution, error in zeroable:
        var = var_list[idx]
        score = _count_linear_pairs_after_zeroing(polynomials, variables, var)
        
        if score > best_score or (score == best_score and (best is None or error < best[3])):
            best = (idx, var, new_solution, error)
            best_score = score

    if best is None:
        raise ValueError("No best variable to zero")
    
    idx, var, new_solution, error = best
    return idx, var, new_solution


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
    # Step 1: Initial linear elimination
    Q, substitution_map = linear_elimination(polynomial_system, variables, simplify=simplify)
    V = variables - set[Symbol](substitution_map.keys())
    
    if verbose:
        print(f"After initial linear elimination:")
        print(f"  Eliminated {len(substitution_map)} variables: {list[Symbol]   (substitution_map.keys())}")
        print(f"  Remaining equations: {len(Q)}")
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
        
        final_error = float(errors[errors < jnp.inf].min()) if jnp.any(errors < jnp.inf) else float('inf')
        if verbose:
            print(f"  Found numerical solution with error: {final_error:.2e}")
            print(f"  Solution: {solution}")
        
        # Try to zero each variable
        zeroable = try_zero_variables(
            solution, jax_system, 
            variable_indices=list(range(len(var_list))),
            tol=tol
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
        idx, var_to_zero, new_solution = _select_best_to_zero(
            zeroable, Q, var_list, V
        )
        
        if verbose:
            print(f"  Zeroing variable: {var_to_zero}")
        
        # Record the zeroing in substitution map
        substitution_map[var_to_zero] = Integer(0)
        
        # Substitute z=0 in all polynomials
        Q = [Poly(q.as_expr().subs(var_to_zero, 0), *[g for g in q.gens if g != var_to_zero]) 
             for q in Q]
        
        # Remove zero polynomials
        Q = [q for q in Q if not q.is_zero]
        
        # Update variable set and list
        V.discard(var_to_zero)
        var_list = [v for v in var_list if v != var_to_zero]
        
        # If no equations or variables remain, we're done
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
                print(f"  Linear elimination found {len(new_sigma)} more eliminations: {list[Symbol](new_sigma.keys())}")
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
        Poly(x0**3 - 2*x1 + x2 + x3 - 2*x4**3 - 1),
        Poly(x1**2 + x2**2 - x0*x3 + 4*x4**2 - 7),
    ]

    # solve using soothing
    solution, substitution_map = soothing(system, {x0, x1, x2, x3, x4})
    print("Solution:", solution)
    print("\nSubstitution map:")
    for key, value in substitution_map.items():
        print(f"{key} -> {value}")