from dataclasses import dataclass
from graphlib import TopologicalSorter
from typing import cast

import numpy as np
from sympy import Expr, Symbol, lambdify, S, nan, simplify


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
_RANDOM_SEED = 42
_RANDOM_VALUE_RANGE = (-2.0, 2.0)
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


def is_zero(sub_map: SystemSubstitutionMap, poly_index: int) -> bool:
    """
    Check if the polynomial at `poly_index` is zero.

    Substitutes all symbols from the substitution map into the polynomial (in
    dependency order), then returns True iff the resulting expression is zero.

    Args:
        sub_map: The system substitution map.
        poly_index: Index into `sub_map.remaining_equations` of the polynomial to test.

    Returns:
        True if the polynomial at `poly_index` is zero, False otherwise.
    """
    poly = sub_map.remaining_equations[poly_index]
    expr = _back_substitute(poly.as_expr(), sub_map.substitution_map)
    return expr.equals(S.Zero)

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

    a_is_zero = a == 0 or a.equals(S.Zero)
    is_nan = var_substitution.has(nan)
    if a_is_zero or is_nan:
        raise ValueError("degenerate substitution")

    merged = {**sub_map.substitution_map, var: var_substitution}
    propagated = _propagate_constants(merged)
    substitution_map = {k: simplify(v) for k, v in propagated.items()}

    return SystemSubstitutionMap(
        remaining_equations=(
            sub_map.remaining_equations[:poly_index]
            + sub_map.remaining_equations[poly_index + 1 :]
        ),
        substitution_map=substitution_map,
        free_variables=sub_map.free_variables - {var},
        substituted_variables=sub_map.substituted_variables | {var},
    )


def _drop_zero_equations(sub_map: SystemSubstitutionMap) -> SystemSubstitutionMap:
    """
    Return a new SystemSubstitutionMap with polynomials that are zero
    (after substitution) removed from remaining_equations.
    """
    non_zero = [
        poly
        for i, poly in enumerate(sub_map.remaining_equations)
        if not is_zero(sub_map, i)
    ]
    if len(non_zero) == len(sub_map.remaining_equations):
        return sub_map
    return SystemSubstitutionMap(
        remaining_equations=non_zero,
        substitution_map=sub_map.substitution_map,
        free_variables=sub_map.free_variables,
        substituted_variables=sub_map.substituted_variables,
    )


def _linear_substitution_candidates(
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


def _equation_linearity_count(sub_map: SystemSubstitutionMap, poly_index: int) -> int:
    """Return the number of variables the equation at poly_index is linear in."""
    poly = sub_map.remaining_equations[poly_index]
    return sum(
        1
        for v in cast(set[Symbol], poly.free_symbols)
        if is_linear_in_variable(sub_map, poly_index, v)
    )


def _greedy_ordered_candidates(
    sub_map: SystemSubstitutionMap, candidates: list[tuple[int, Symbol]]
) -> list[tuple[int, Symbol]]:
    """
    Return candidates filtered and sorted by the greedy heuristic.

    The best polynomial index and variable pair to substitute for is the one that leaves the most equations linear in some variable.

    Args:
        sub_map: The current substitution state.
        candidates: All available (poly_index, var) pairs from
            _linear_substitution_candidates.

    Returns:
        The best polynomial index and variable pair to substitute for.
    """
    raise NotImplementedError("greedy_ordered_candidates not implemented")


def _dfs(
    sub_map: SystemSubstitutionMap,
    memo: dict[frozenset[Symbol], SystemSubstitutionMap],
) -> SystemSubstitutionMap:
    """
    Recursively explore all substitution branches and return the leaf with the
    fewest remaining equations.

    Memoizes by the frozenset of substituted variables: two paths that have
    eliminated the same set of variables reach an equivalent state, so the best
    result from that state only needs to be computed once.
    """
    is_kappah_zero = Symbol('kappah') in sub_map.substitution_map.keys() and sub_map.substitution_map[Symbol('kappah')] == 0
    if len(sub_map.remaining_equations) == 0 and not is_kappah_zero:
        print(len(sub_map.free_variables))
        for sym, expr in sub_map.substitution_map.items():
            print(sym, '=', expr)
        print('\n')

    state_key = frozenset(sub_map.substituted_variables)
    if state_key in memo:
        return memo[state_key]

    candidates = _linear_substitution_candidates(sub_map)

    if not candidates:
        memo[state_key] = sub_map
        return sub_map

    child_results = []
    for poly_index, var in candidates:
        try:
            child_map = add_linear_substitution(sub_map, poly_index, var)
            child_map = _drop_zero_equations(child_map)
            child_results.append(_dfs(child_map, memo))
        except ValueError:
            continue

    if not child_results:
        memo[state_key] = sub_map
        return sub_map

    best = min(child_results, key=lambda m: len(m.remaining_equations))
    memo[state_key] = best
    return best


def best_linear_substitutions_order(
    equations: list[Expr], greedy: bool = False
) -> SystemSubstitutionMap:
    """
    Find the substitution ordering that eliminates the most variables.

    Two search strategies are available:

    - DFS (default, ``greedy=False``): exhaustively explores all sequences of
      linear substitutions, returning the leaf with the fewest remaining
      equations. Memoizes by the frozenset of substituted variables so that
      two paths reaching the same eliminated set are not explored twice.

    - Greedy (``greedy=True``): at each step calls ``_greedy_candidate`` to
      pick the variable that the fewest equations are linear in, breaking ties
      by choosing the equation that is linear in the fewest variables. Runs in
      linear time in the number of substitutions but may not find the global
      optimum.

    Args:
        equations: the system of polynomial equations each of which is equated to zero.
        greedy: If True, use the greedy heuristic instead of exhaustive DFS.
            Defaults to False.

    Returns:
        A SystemSubstitutionMap with the fewest remaining equations achievable
        (exactly minimal for DFS, heuristically minimal for greedy).
    """
    initial = SystemSubstitutionMap(
        remaining_equations=equations,
        substitution_map={},
        free_variables={
            cast(Symbol, sym) for poly in equations for sym in poly.free_symbols
        },
        substituted_variables=set(),
    )

    if greedy:
        sub_map = initial
        while True:
            candidates = _linear_substitution_candidates(sub_map)
            if not candidates:
                return sub_map
            ordered = _greedy_ordered_candidates(sub_map, candidates)
            for poly_index, var in ordered:
                try:
                    sub_map = add_linear_substitution(sub_map, poly_index, var)
                    sub_map = _drop_zero_equations(sub_map)
                    break
                except ValueError:
                    continue
            else:
                return sub_map

    return _dfs(initial, {})
