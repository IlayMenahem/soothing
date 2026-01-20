from sympy import Symbol, Poly, symbols, solve, Expr, Integer
import jax.numpy as jnp

from soothing.numerical import newton_raphson
from soothing.symbolic import simple_sparsification, try_zero_variables
from soothing.utils import sympy_to_jax_system


def _get_degree(poly: Poly, var: Symbol) -> int:
    """Get the degree of a variable in a polynomial, handling missing generators.
    
    Args:
        poly: The polynomial.
        var: The variable to get the degree of.
        
    Returns:
        The degree of var in poly, or 0 if var is not a generator.
    """
    if var not in poly.gens:
        return 0
    return int(poly.degree(var))  # type: ignore[arg-type]


def _is_linear_in_var(poly: Poly, var: Symbol) -> bool:
    """Check if a polynomial is linear in a given variable.
    
    A polynomial is linear in var if the degree of var is exactly 1.
    
    Args:
        poly: The polynomial to check.
        var: The variable to check linearity for.
        
    Returns:
        True if poly is linear in var, False otherwise.
    """
    return _get_degree(poly, var) == 1


def _find_linear_pairs(polynomials: list[Poly], variables: set[Symbol]) -> list[tuple[Symbol, Poly]]:
    """Find all (variable, polynomial) pairs where the polynomial is linear in the variable.
    
    Args:
        polynomials: List of polynomials to search.
        variables: Set of variables to consider.
        
    Returns:
        List of (variable, polynomial) pairs where polynomial is linear in variable.
    """
    pairs = []
    for poly in polynomials:
        for var in variables:
            if var in poly.free_symbols and _is_linear_in_var(poly, var):
                pairs.append((var, poly))
    return pairs


def _solve_linear_for_var(poly: Poly, var: Symbol) -> Expr:
    """Solve a polynomial that is linear in var for var.
    
    Given poly = a*var + b = 0 (where a, b may depend on other variables),
    returns var = -b/a.
    
    Args:
        poly: Polynomial linear in var.
        var: Variable to solve for.
        
    Returns:
        Expression for var in terms of other variables.
    """
    expr = poly.as_expr()
    solutions = solve(expr, var)
    if not solutions:
        raise ValueError(f"Could not solve {poly} for {var}")
    return solutions[0]


def _count_occurrences(var: Symbol, polynomials: list[Poly]) -> int:
    """Count how many polynomials contain the given variable.
    
    Args:
        var: Variable to count occurrences of.
        polynomials: List of polynomials to search.
        
    Returns:
        Number of polynomials containing var.
    """
    return sum(1 for p in polynomials if var in p.free_symbols)


def _total_degree_in_others(var: Symbol, poly: Poly, polynomials: list[Poly]) -> int:
    """Calculate total degree of var across all other polynomials.
    
    Args:
        var: Variable to check.
        poly: The polynomial used for elimination (excluded from count).
        polynomials: All polynomials.
        
    Returns:
        Sum of degrees of var in all polynomials except poly.
    """
    return sum(_get_degree(p, var) for p in polynomials if p != poly)


def _select_best_pair(
    pairs: list[tuple[Symbol, Poly]], 
    polynomials: list[Poly]
) -> tuple[Symbol, Poly]:
    """Select the best (variable, polynomial) pair for elimination.
    
    Heuristic: Choose the variable with lowest total degree in other equations.
    This tends to minimize expression growth after substitution.
    
    Args:
        pairs: List of candidate (variable, polynomial) pairs.
        polynomials: All polynomials in the system.
        
    Returns:
        The best (variable, polynomial) pair to use for elimination.
    """
    if not pairs:
        raise ValueError("No pairs to select from")
    
    # Score each pair: lower is better
    # Primary: lowest total degree in other equations
    # Secondary: fewest occurrences (to minimize substitution work)
    def score(pair: tuple[Symbol, Poly]) -> tuple[int, int]:
        var, poly = pair
        degree_score = _total_degree_in_others(var, poly, polynomials)
        occurrence_score = _count_occurrences(var, polynomials)
        return (degree_score, occurrence_score)
    
    return min(pairs, key=score)


def linear_elimination(polynomial_system: list[Poly], variables: set[Symbol]) -> tuple[list[Poly], dict]:
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
    
    Returns:
    - Tuple of (reduced_system (list[Poly]), substitution_map (dict))
    """
    Q = list(polynomial_system)
    V = set(variables)
    sigma = {}  # substitution map: var -> expression
    
    while True:
        # Find all (var, poly) pairs where poly is linear in var
        linear_pairs = _find_linear_pairs(Q, V)
        
        if not linear_pairs:
            break
        
        # Select the best pair using heuristics
        var, poly = _select_best_pair(linear_pairs, Q)
        
        # Solve for var: poly = 0 => var = f(other_vars)
        expr = _solve_linear_for_var(poly, var)
        
        # Record the substitution
        sigma[var] = expr
        
        # Remove the equation used for elimination
        Q = [q for q in Q if q != poly]
        
        # Collect all generators from remaining polynomials (excluding the eliminated var)
        all_gens: set[Symbol] = set()
        for q in Q:
            all_gens.update(q.gens)
        all_gens.discard(var)
        all_gens_tuple = tuple(sorted(all_gens, key=str))
        
        # Substitute var -> expr in all remaining equations
        new_Q = []
        for q in Q:
            new_expr = q.as_expr().subs(var, expr)
            if new_expr != 0 and all_gens_tuple:
                new_Q.append(Poly(new_expr, *all_gens_tuple))
            elif new_expr != 0:
                # If no generators left, keep as constant check
                pass  # This would mean system is inconsistent or fully solved
        Q = new_Q
        
        # Remove zero polynomials that may result from substitution
        Q = [q for q in Q if not q.is_zero]
        
        # Remove var from the variable set
        V.discard(var)
    
    return Q, sigma


def _count_linear_pairs_after_zeroing(
    polynomials: list[Poly], 
    variables: set[Symbol], 
    var_to_zero: Symbol
) -> int:
    """Count how many linear pairs would exist if var_to_zero were set to 0.
    
    This is used as a heuristic to select which variable to zero:
    we prefer zeroing variables that create more linear elimination opportunities.
    
    Args:
        polynomials: Current polynomial system.
        variables: Set of remaining variables.
        var_to_zero: Variable to hypothetically set to 0.
        
    Returns:
        Number of (var, poly) linear pairs that would exist after zeroing.
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
    zeroable: list[tuple[int, jnp.ndarray, float]],
    polynomials: list[Poly],
    var_list: list[Symbol],
    variables: set[Symbol],
) -> tuple[int, Symbol, jnp.ndarray]:
    """Select the best variable to zero based on heuristics.
    
    Heuristic: Choose the variable that creates the most linear pairs
    after zeroing, to maximize opportunities for further linear elimination.
    
    Args:
        zeroable: List of (idx, new_solution, error) from try_zero_variables.
        polynomials: Current polynomial system.
        var_list: Ordered list of variables (maps indices to symbols).
        variables: Set of remaining variables.
        
    Returns:
        Tuple of (index, symbol, new_solution) for the best variable to zero.
    """
    if not zeroable:
        raise ValueError("No zeroable variables")
    
    best: tuple[int, Symbol, jnp.ndarray, float] | None = None
    best_score = -1
    
    for idx, new_solution, error in zeroable:
        var = var_list[idx]
        score = _count_linear_pairs_after_zeroing(polynomials, variables, var)
        
        # Prefer more linear pairs; break ties by lower error
        if score > best_score or (score == best_score and (best is None or error < best[3])):
            best = (idx, var, new_solution, error)
            best_score = score
    
    # This assertion is always true since zeroable is non-empty
    assert best is not None
    return best[0], best[1], best[2]


def soothing(
    polynomial_system: list[Poly], 
    variables: set[Symbol],
    tol: float = 1e-10,
    max_iterations: int = 100,
    verbose: bool = True,
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
    
    Returns:
    - Tuple of (parametric_solution (list[Poly]), substitution_map (dict))
    """
    # Step 1: Initial linear elimination
    Q, sigma = linear_elimination(polynomial_system, variables)
    V = variables - set(sigma.keys())
    
    if verbose:
        print(f"After initial linear elimination:")
        print(f"  Eliminated {len(sigma)} variables: {list(sigma.keys())}")
        print(f"  Remaining equations: {len(Q)}")
        print(f"  Remaining variables: {V}")
    
    # If no equations remain or no variables remain, we're done
    if not Q or not V:
        return Q, sigma
    
    # Create ordered variable list for index mapping
    var_list = sorted(list(V), key=lambda x: str(x))
    
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
        sigma[var_to_zero] = Integer(0)
        
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
        
        # Re-run linear elimination on the simplified system
        Q, new_sigma = linear_elimination(Q, V)
        
        if new_sigma:
            if verbose:
                print(f"  Linear elimination found {len(new_sigma)} more eliminations: {list(new_sigma.keys())}")
            sigma.update(new_sigma)
            V = V - set(new_sigma.keys())
            var_list = [v for v in var_list if v not in new_sigma]
        
        if not Q or not V:
            if verbose:
                print("  System fully solved after linear elimination.")
            break
    
    return Q, sigma


def linear_simplification(system: list[Poly], variables: set[Symbol]) -> tuple[list[Poly], dict]:
    """
    Simplify a system using elimination of variables in a linear fashion.

    This is an alias for linear_elimination.
    
    Args:
    - system (list[Poly]): List of polynomials $P = {p_1, ..., p_k}$
    - variables (set[Symbol]): Set of variables $V = {x_1, ..., x_m}$
    

    Returns:
    - Tuple of (simplified_system (list[Poly]), substitution_map (dict))
    """
    return linear_elimination(system, variables)
