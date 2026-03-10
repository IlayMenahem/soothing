from dataclasses import dataclass
from graphlib import TopologicalSorter
from typing import cast

import numpy as np
from sympy import Expr, Poly, Symbol, lambdify


@dataclass(frozen=True)
class SystemSubstitutionMap:
    """
    A class to represent a system of polynomial equations that has a substitution map.

    The substitutions in substitution_map do not have cycles, meaning that if x is substituted by an expression that contains y, then y is not substituted by an expression that contains x.
    """

    remaining_equations: list[Poly]
    substitution_map: dict[Symbol, Expr]
    free_variables: set[Symbol]
    substituted_variables: set[Symbol]


_NUM_COLLINEARITY_POINTS = 8
_RANDOM_SEED = 42
_RANDOM_VALUE_RANGE = (-5.0, 5.0)
_VAR_RANGE = (-10.0, 10.0)
_COLLINEARITY_TOL = 1e-8


def _are_collinear(x: np.ndarray, y: np.ndarray, tol: float) -> bool:
    """
    Determine whether the points (x[i], y[i]) are collinear within tolerance.

    Fits a degree-1 polynomial through the first and last point, then checks
    that all intermediate points lie on that line.

    Args:
        x: x-coordinates of the points.
        y: y-coordinates of the points.
        tol: Maximum allowed deviation from the fitted line.

    Returns:
        True if all points are collinear within `tol`, False otherwise.
    """
    coeffs = np.polyfit(x[[0, -1]], y[[0, -1]], deg=1)
    y_fitted = np.polyval(coeffs, x)
    return bool(np.all(np.abs(y - y_fitted) < tol))


def _get_independent_expr(
    poly: Poly, var: Symbol, substitution_map: dict[Symbol, Expr]
) -> Expr:
    """
    Return the expression for `poly` with all symbols that transitively depend
    on `var` through `substitution_map` substituted out.

    This ensures the resulting expression contains no symbol whose definition
    (through the substitution chain) references `var`, which is required both
    for correctly solving for `var` and for correctly testing linearity in `var`.

    Only symbols that transitively depend on `var` are substituted; others in
    `substitution_map` are left as-is to keep the expression as simple as possible.

    Args:
        poly: The polynomial to rewrite.
        var: The variable whose dependents should be eliminated from the expression.
        substitution_map: Map from substituted symbols to their defining expressions.

    Returns:
        The rewritten expression, free of any symbol that depends on `var`.
    """

    def depends_on_var(sym: Symbol, visited: frozenset[Symbol]) -> bool:
        """Transitively check if sym depends on var through substitution_map."""
        if sym == var:
            return True
        if sym in visited or sym not in substitution_map:
            return False
        return any(
            depends_on_var(cast(Symbol, s), visited | {sym})
            for s in substitution_map[sym].free_symbols
        )

    vars_depending_on_var = {
        sym
        for sym in substitution_map
        if sym != var and depends_on_var(sym, frozenset())
    }

    dependency_graph: dict[Symbol, set[Symbol]] = {
        sym: cast(set[Symbol], substitution_map[sym].free_symbols)
        & vars_depending_on_var
        for sym in vars_depending_on_var
    }

    substitution_order = reversed(
        list(TopologicalSorter(dependency_graph).static_order())
    )
    substitutions = [(sym, substitution_map[sym]) for sym in substitution_order]

    return poly.as_expr().subs(substitutions)


def is_linear_in_variable(
    sub_map: SystemSubstitutionMap, poly_index: int, var: Symbol
) -> bool:
    """
    Check if the polynomial at `poly_index` is linear in `var`, after
    back-substituting all symbols that transitively depend on `var`.

    Uses a probabilistic collinearity test: fix all truly-free symbols at random
    values, then evaluate the back-substituted expression at
    `_NUM_COLLINEARITY_POINTS` random values of `var`. The expression is linear
    in `var` if and only if those points are collinear (almost surely over the
    random choices, avoiding a measure-zero set of bad samples).

    Testing the back-substituted expression (rather than the raw polynomial) is
    essential: a symbol like `x1 = f(x0)` in the substitution map makes the raw
    polynomial appear linear in `x0` even though the true expression is not.

    Args:
        sub_map: The system substitution map containing the remaining equations.
        poly_index: Index into `sub_map.remaining_equations` of the polynomial to test.
        var: The variable to test linearity in.

    Returns:
        True if the back-substituted expression is linear in `var`, False otherwise.
    """
    poly = sub_map.remaining_equations[poly_index]

    if var not in poly.free_symbols:
        return False

    if var in sub_map.substitution_map:
        return False

    independent_expr = _get_independent_expr(poly, var, sub_map.substitution_map)

    # _get_independent_expr only substitutes symbols that transitively depend on var.
    # Other substituted symbols (independent of var) may still appear in the expression
    # and must also be resolved before numeric evaluation.
    remaining_substituted = {
        cast(Symbol, s)
        for s in independent_expr.free_symbols
        if s != var and s in sub_map.substitution_map
    }
    if remaining_substituted:
        remaining_dep_graph: dict[Symbol, set[Symbol]] = {
            sym: cast(set[Symbol], sub_map.substitution_map[sym].free_symbols)
            & remaining_substituted
            for sym in remaining_substituted
        }
        remaining_order = reversed(
            list(TopologicalSorter(remaining_dep_graph).static_order())
        )
        remaining_subs = [
            (sym, sub_map.substitution_map[sym]) for sym in remaining_order
        ]
        independent_expr = independent_expr.subs(remaining_subs)

    rng = np.random.default_rng(_RANDOM_SEED)

    # All symbols remaining in independent_expr are truly free (not substituted).
    free_symbols_in_expr: set[Symbol] = {
        cast(Symbol, s) for s in independent_expr.free_symbols
    }

    lo, hi = _RANDOM_VALUE_RANGE
    fixed_substitutions: dict[Symbol, float] = {
        sym: float(rng.uniform(lo, hi)) for sym in free_symbols_in_expr - {var}
    }

    non_var_symbols = [s for s in free_symbols_in_expr if s != var]
    expr_func = lambdify([var] + non_var_symbols, independent_expr, modules=["numpy"])
    non_var_args = [fixed_substitutions[s] for s in non_var_symbols]

    lo, hi = _VAR_RANGE
    var_values = np.linspace(lo, hi, _NUM_COLLINEARITY_POINTS)
    expr_values = np.array([float(expr_func(v, *non_var_args)) for v in var_values])

    if not np.all(np.isfinite(expr_values)):
        return False

    return _are_collinear(var_values, expr_values, tol=_COLLINEARITY_TOL)


def add_linear_substitution(
    sub_map: SystemSubstitutionMap, poly_index: int, var: Symbol
) -> SystemSubstitutionMap:
    """
    Add a linear substitution for `var` based on the polynomial at `poly_index`.

    Args:
        sub_map: The system substitution map containing the remaining equations.
        poly_index: Index into `sub_map.remaining_equations` of the polynomial to use for the substitution.
        var: The variable the polynomial is linear in, which will be substituted for.

    Returns:
        A new SystemSubstitutionMap with the substitution added, and the polynomial at `poly_index` removed from the remaining equations.
    """

    poly = sub_map.remaining_equations[poly_index]
    independent_expr = _get_independent_expr(poly, var, sub_map.substitution_map)

    # Solve the linear equation a*var + b = 0 for var (i.e. var = -b/a)
    b = independent_expr.subs(var, 0)
    a = independent_expr.diff(var)
    var_substitution = -b / a

    remaining_equations = (
        sub_map.remaining_equations[:poly_index]
        + sub_map.remaining_equations[poly_index + 1 :]
    )
    new_substitution_map = {**sub_map.substitution_map, var: var_substitution}
    new_free_variables = sub_map.free_variables - {var}
    new_substituted_variables = sub_map.substituted_variables | {var}

    return SystemSubstitutionMap(
        remaining_equations=remaining_equations,
        substitution_map=new_substitution_map,
        free_variables=new_free_variables,
        substituted_variables=new_substituted_variables,
    )


def best_linear_substitutions_order(equations: list[Poly]) -> SystemSubstitutionMap:
    """
    Find the substitution ordering that eliminates the most variables by performing
    a DFS over all possible sequences of linear substitutions.

    Each node in the search tree is a SystemSubstitutionMap. Its children are produced
    by picking any (poly_index, var) pair where the polynomial is linear in that variable
    and applying add_linear_substitution. A node with no such pairs is a leaf.

    The returned node is the leaf with the fewest remaining equations (i.e. the deepest
    reachable node in terms of substitutions applied).

    Args:
        equations: the system of polynomial equations each of which is equated to zero.

    Returns:
        A SystemSubstitutionMap with the least number of remaining equations achievable
        by any sequence of linear substitutions.
    """
    initial = SystemSubstitutionMap(
        remaining_equations=equations,
        substitution_map={},
        free_variables={
            cast(Symbol, sym) for poly in equations for sym in poly.free_symbols
        },
        substituted_variables=set(),
    )

    def linear_substitution_candidates(
        sub_map: SystemSubstitutionMap,
    ) -> list[tuple[int, Symbol]]:
        """Return all (poly_index, var) pairs where the polynomial is linear in that variable."""
        return [
            (poly_index, var)
            for poly_index in range(len(sub_map.remaining_equations))
            for var in cast(
                set[Symbol], sub_map.remaining_equations[poly_index].free_symbols
            )
            if is_linear_in_variable(sub_map, poly_index, var)
        ]

    def dfs(sub_map: SystemSubstitutionMap) -> SystemSubstitutionMap:
        """
        Recursively explore all substitution branches and return the leaf
        with the fewest remaining equations.
        """
        candidates = linear_substitution_candidates(sub_map)

        if not candidates:
            return sub_map

        child_results = [
            dfs(add_linear_substitution(sub_map, poly_index, var))
            for poly_index, var in candidates
        ]

        return min(child_results, key=lambda m: len(m.remaining_equations))

    return dfs(initial)
