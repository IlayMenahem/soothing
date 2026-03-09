from dataclasses import dataclass
from graphlib import TopologicalSorter
from typing import cast

import numpy as np
from sympy import Expr, Poly, Symbol, lambdify

from soothing.utils import solve_linear_for_var


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


_NUM_COLLINEARITY_POINTS = 5
_RANDOM_SEED = 42
_RANDOM_VALUE_RANGE = (-10.0, 10.0)
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


def _evaluate_substituted_vars(
    poly: Poly,
    var: Symbol,
    substitution_map: dict[Symbol, Expr],
    fixed_substitutions: dict[Symbol, float],
) -> dict[Symbol, float]:
    """
    Numerically evaluate the substituted variables that appear in `poly` (excluding `var`),
    resolving chains through `substitution_map` in topological order.

    Only variables that appear in the polynomial and are present in `substitution_map`
    are evaluated. Their expressions may themselves depend on other substituted variables,
    which are resolved first via topological sort.

    Args:
        poly: The polynomial whose non-`var` symbols need numeric values.
        var: The variable being tested for linearity (excluded from evaluation).
        substitution_map: Map from substituted symbols to their defining expressions.
        fixed_substitutions: Already-fixed numeric values for free variables.

    Returns:
        A dict of all substituted variable values needed, merged with `fixed_substitutions`.
    """
    substituted_in_poly = {
        cast(Symbol, sym)
        for sym in poly.free_symbols
        if sym != var and sym in substitution_map
    }

    # Collect all transitive substituted dependencies, not just the ones directly in poly.
    def collect_dependencies(syms: set[Symbol]) -> set[Symbol]:
        closure: set[Symbol] = set()
        frontier = set(syms)
        while frontier:
            sym = frontier.pop()
            if sym in closure or sym not in substitution_map:
                continue
            closure.add(sym)
            frontier |= (
                cast(set[Symbol], substitution_map[sym].free_symbols)
                & substitution_map.keys()
            )
        return closure

    all_deps = collect_dependencies(substituted_in_poly)

    dependency_graph: dict[Symbol, set[Symbol]] = {
        sym: cast(set[Symbol], substitution_map[sym].free_symbols) & all_deps
        for sym in all_deps
    }

    all_values: dict[Symbol, float] = dict(fixed_substitutions)

    for sym in TopologicalSorter(dependency_graph).static_order():
        expr = substitution_map[sym]
        free_syms = list(cast(set[Symbol], expr.free_symbols))
        f = lambdify(free_syms, expr, modules=["numpy"])
        args = [all_values[s] for s in free_syms]
        all_values[sym] = float(f(*args))

    return all_values


def is_linear_in_variable(
    sub_map: SystemSubstitutionMap, poly_index: int, var: Symbol
) -> bool:
    """
    Check if the polynomial at `poly_index` is linear in `var`.

    Uses a probabilistic collinearity test: fix all free variables other than
    `var` at random values, then evaluate the polynomial at
    `_NUM_COLLINEARITY_POINTS` random values of `var`.  The polynomial is
    linear in `var` if and only if those points are collinear (almost surely
    over the random choices, avoiding a measure-zero set of bad samples).

    Args:
        sub_map: The system substitution map containing the remaining equations.
        poly_index: Index into `sub_map.remaining_equations` of the polynomial to test.
        var: The variable to test linearity in.

    Returns:
        True if the polynomial is linear in `var`, False otherwise.
    """
    poly = sub_map.remaining_equations[poly_index]

    if var not in poly.free_symbols:
        return False

    rng = np.random.default_rng(_RANDOM_SEED)
    lo, hi = _RANDOM_VALUE_RANGE

    other_free_symbols = sub_map.free_variables - {var}
    fixed_substitutions: dict[Symbol, float] = {
        sym: float(rng.uniform(lo, hi)) for sym in other_free_symbols
    }

    all_values = _evaluate_substituted_vars(
        poly, var, sub_map.substitution_map, fixed_substitutions
    )

    non_var_symbols = [s for s in cast(set[Symbol], poly.free_symbols) if s != var]
    poly_func = lambdify([var] + non_var_symbols, poly.as_expr(), modules=["numpy"])
    non_var_args = [all_values[s] for s in non_var_symbols]

    var_values = rng.uniform(lo, hi, size=_NUM_COLLINEARITY_POINTS)
    poly_values = np.array([float(poly_func(v, *non_var_args)) for v in var_values])

    if not np.all(np.isfinite(poly_values)):
        return False

    return _are_collinear(var_values, poly_values, tol=_COLLINEARITY_TOL)


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

    def get_independent_poly(
        poly: Poly, var: Symbol, substitution_map: dict[Symbol, Expr]
    ) -> Poly:
        """
        gives the polynomial such that there is no variable in the polynomial that depends on var through the substitution map. This is necessary to ensure that solving for var linearly in the polynomial is correct.

        only variables that depend on var through the substitution map are substituted, and other variables are not substituted, even if they are in the substitution map, this is to minimize the number of substitutions and avoid unnecessary substitutions that can make the polynomial more complicated.

        Args:
            poly: the polynomial to get the independent version of.
            var: the variable to get the independent version of.
            substitution_map: the substitution map to use to get the independent version of the polynomial.
        """

        def depends_on_var(sym: Symbol, visited: frozenset[Symbol]) -> bool:
            """Transitively check if sym depends on var through substitution_map."""
            if sym == var:
                return True
            if sym in visited or sym not in substitution_map:
                return False
            expr = substitution_map[sym]
            return any(
                depends_on_var(cast(Symbol, s), visited | {sym})
                for s in expr.free_symbols
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

        independent_expr = poly.as_expr().subs(substitutions)
        gens = poly.gens if var in poly.gens else (*poly.gens, var)
        return Poly(independent_expr, *gens)

    poly = sub_map.remaining_equations[poly_index]
    independent_poly = get_independent_poly(poly, var, sub_map.substitution_map)
    var_substitution = solve_linear_for_var(independent_poly, var)

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
