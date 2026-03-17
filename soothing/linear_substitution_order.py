from typing import cast
from sympy import Expr, Symbol

from .system_substitution import (
    SystemSubstitutionMap,
    is_linear_in_variable,
    add_linear_substitution,
    _drop_zero_equations,
)

__all__ = ["best_linear_substitutions_order"]


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
