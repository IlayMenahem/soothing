from dataclasses import dataclass
from graphlib import TopologicalSorter
from typing import cast

import numpy as np
from sympy import Expr, Symbol, lambdify, S, nan, simplify, zoo

__all__ = [
    "SystemSubstitutionMap",
    "is_linear_in_variable",
    "is_zero",
    "validate_substitution_map",
    "add_linear_substitution",
]

@dataclass(frozen=True)
class SystemSubstitutionMap:
    """
    A system of polynomial equations with a substitution map.

    The substitutions in substitution_map are acyclic: if x is substituted by an expression containing y, then y is not substituted by an expression containing x.
    """

    remaining_equations: list[Expr]
    substitution_map: dict[Symbol, Expr]
    free_variables: set[Symbol]
    substituted_variables: set[Symbol]


_NUM_COLLINEARITY_POINTS = 8
_NUM_ZERO_SAMPLES = 12
_RANDOM_SEED = 42
_RANDOM_VALUE_RANGE = (-2.0, 2.0)
_VAR_RANGE = (-10.0, 10.0)
_COLLINEARITY_TOL = 1e-8
_ZERO_TOL = 1e-9


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


def _depends_on_var(
    sym: Symbol,
    var: Symbol,
    substitution_map: dict[Symbol, Expr],
    visited: frozenset[Symbol],
) -> bool:
    """Transitively check if sym depends on var through substitution_map."""
    if sym == var:
        return True
    if sym in visited or sym not in substitution_map:
        return False
    return any(
        _depends_on_var(cast(Symbol, s), var, substitution_map, visited | {sym})
        for s in substitution_map[sym].free_symbols
    )


def _get_independent_expr(
    poly: Expr, var: Symbol, substitution_map: dict[Symbol, Expr]
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
    vars_depending_on_var = {
        sym
        for sym in substitution_map
        if sym != var and _depends_on_var(sym, var, substitution_map, frozenset())
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


def _back_substitute(
    expr: Expr,
    substitution_map: dict[Symbol, Expr],
    exclude: set[Symbol] | None = None,
) -> Expr:
    """
    Substitute into expr every symbol that appears in substitution_map and not in
    exclude, in topological order of their dependencies.

    Args:
        expr: Expression to rewrite.
        substitution_map: Map from substituted symbols to their defining expressions.
        exclude: Symbols to leave unsubstituted. If None, substitute all mapped symbols.

    Returns:
        The expression after back-substitution.
    """
    exclude_set = exclude if exclude is not None else set()

    while True:
        remaining = {
            cast(Symbol, s)
            for s in expr.free_symbols
            if s in substitution_map and s not in exclude_set
        }
        if not remaining:
            return expr

        dep_graph: dict[Symbol, set[Symbol]] = {
            sym: cast(
                set[Symbol],
                getattr(substitution_map[sym], "free_symbols", set()) & remaining,
            )
            for sym in remaining
        }
        order = reversed(list(TopologicalSorter(dep_graph).static_order()))
        expr = expr.subs([(sym, substitution_map[sym]) for sym in order])


def _propagate_constants(
    substitution_map: dict[Symbol, Expr],
) -> dict[Symbol, Expr]:
    """
    Repeatedly substitute constant-valued entries into all other entries
    until no further constants can be propagated (fixed-point iteration).

    A constant entry is one whose value has no free symbols.
    """
    result = dict(substitution_map)
    while True:
        constants = {k: v for k, v in result.items() if not v.free_symbols}
        if not constants:
            return result
        new_result = {
            k: v if k in constants else v.subs(list(constants.items()))
            for k, v in result.items()
        }
        if new_result == result:
            return result
        result = new_result


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
    independent_expr = _back_substitute(independent_expr, sub_map.substitution_map, exclude={var})

    rng = np.random.default_rng(_RANDOM_SEED)

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
    try:
        expr_values = np.array(
            [float(expr_func(v, *non_var_args)) for v in var_values]
        )
    except OverflowError:
        return False

    if not np.all(np.isfinite(expr_values)):
        return False

    return _are_collinear(var_values, expr_values, tol=_COLLINEARITY_TOL)

def _is_zero_probabilistic(expr: Expr, substitution_map: dict[Symbol, Expr]) -> bool:
    expr = _back_substitute(expr, substitution_map)
    free_symbols_in_expr: tuple[Symbol, ...] = tuple(
        sorted((cast(Symbol, s) for s in expr.free_symbols), key=lambda s: s.name)
    )

    if not free_symbols_in_expr:
        try:
            val = complex(expr.evalf())
        except (TypeError, ValueError):
            return False
        return bool(np.isfinite(val) and abs(val) < _ZERO_TOL)

    rng = np.random.default_rng(_RANDOM_SEED)
    lo, hi = _RANDOM_VALUE_RANGE
    expr_func = lambdify(free_symbols_in_expr, expr, modules=["numpy"])

    for _ in range(_NUM_ZERO_SAMPLES):
        args = [float(rng.uniform(lo, hi)) for _ in free_symbols_in_expr]
        try:
            raw = expr_func(*args)
            v = float(np.asarray(raw, dtype=np.float64).reshape(-1)[0])
        except (TypeError, ValueError, OverflowError):
            return False
        if not np.isfinite(v) or abs(v) >= _ZERO_TOL:
            return False

    return True


def is_zero(sub_map: SystemSubstitutionMap, poly_index: int) -> bool:
    """
    Check if the polynomial at `poly_index` is zero.

    Back-substitutes the substitution map, then evaluates the result at
    `_NUM_ZERO_SAMPLES` random points (one draw per sample over all free
    symbols in the expression). Returns True iff every evaluation is within
    `_ZERO_TOL` of zero (probabilistic: a nonzero polynomial vanishes on a
    lower-dimensional set, so random points almost surely detect it).

    Args:
        sub_map: The system substitution map.
        poly_index: Index into `sub_map.remaining_equations` of the polynomial to test.

    Returns:
        True if the polynomial at `poly_index` is zero, False otherwise.
    """
    poly = sub_map.remaining_equations[poly_index]

    return _is_zero_probabilistic(poly.as_expr(), sub_map.substitution_map)


def validate_substitution_map(
    system: list[Expr], substitution_map: dict[Symbol, Expr]
) -> bool:
    """
    Validate that the substitution map is a valid solution to the system of equations.

    Args:
        system: List of equations.
        substitution_map: Dictionary of substitutions.

    Returns:
        True if the substitution map is a valid solution to the system of equations,
        False otherwise.
    """
    return all(
        _back_substitute(equation, substitution_map).equals(S.Zero)
        for equation in system
    )


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
    independent_expr = _back_substitute(independent_expr, sub_map.substitution_map, exclude={var})

    # Solve a*var + b = 0 → var = -b/a
    b = independent_expr.subs(var, 0)
    a = independent_expr.diff(var)
    var_substitution = -b / a

    a_is_zero = _is_zero_probabilistic(a, sub_map.substitution_map)
    has_invalid_value = var_substitution.has(nan) or var_substitution.has(zoo)
    if a_is_zero or has_invalid_value:
        raise ValueError("degenerate substitution")

    merged = {**sub_map.substitution_map, var: var_substitution}
    propagated = _propagate_constants(merged)
    substitution_map = {k: simplify(v) for k, v in propagated.items()}
    if any(expr.has(nan) or expr.has(zoo) for expr in substitution_map.values()):
        raise ValueError("degenerate substitution")

    return SystemSubstitutionMap(
        remaining_equations=(
            sub_map.remaining_equations[:poly_index]
            + sub_map.remaining_equations[poly_index + 1 :]
        ),
        substitution_map=substitution_map,
        free_variables=sub_map.free_variables - {var},
        substituted_variables=sub_map.substituted_variables | {var},
    )
