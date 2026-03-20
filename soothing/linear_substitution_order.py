from dataclasses import dataclass
from math import sqrt
from typing import Iterable, cast

from sympy import Expr, Symbol, default_sort_key, symbols

from .system_substitution import (
    is_zero,
    SystemSubstitutionMap,
    add_linear_substitution,
    is_linear_in_variable,
)

__all__ = ["LinearSubstitutionOrderStats", "best_linear_substitutions_order"]

_MCTS_DEFAULT_ITERATIONS = 4
_MCTS_DEFAULT_ROLLOUT_DEPTH = 4
_MCTS_DEFAULT_EXPLORATION_WEIGHT = sqrt(2.0)
_MCTS_DEFAULT_RANDOM_SEED = 0

LinearSubstitutionAction = tuple[int, Symbol]


@dataclass(frozen=True, slots=True)
class LinearSubstitutionOrderStats:
    """
    Run statistics for :func:`best_linear_substitutions_order`.

    ``eliminations`` is ``len(input equations) - len(remaining_equations)`` in
    the returned map.

    ``successful_substitutions`` counts every successful linear substitution
    application, including speculative tries (greedy ordering,
    child enumeration in DFS/MCTS).

    ``search_nodes_visited`` depends on ``strategy``: DFS counts recursive
    ``_dfs`` entries; greedy counts decision steps where at least one candidate
    exists; MCTS sums ``len(path)`` over simulations (tree nodes traversed
    per iteration, including the root).
    """

    eliminations: int
    successful_substitutions: int
    search_nodes_visited: int


class _RunCounters:
    __slots__ = ("successful_substitutions", "search_nodes_visited")

    def __init__(self) -> None:
        self.successful_substitutions = 0
        self.search_nodes_visited = 0


def _linear_substitution_candidates(
    sub_map: SystemSubstitutionMap,
) -> Iterable[LinearSubstitutionAction]:
    """Return all (poly_index, var) pairs where the polynomial is linear in that variable."""
    yield from (
        (poly_index, var)
        for poly_index, poly in enumerate(sub_map.remaining_equations)
        for var in sorted(cast(set[Symbol], poly.free_symbols), key=default_sort_key)
        if is_linear_in_variable(sub_map, poly_index, var)
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


def _try_apply_action(
    sub_map: SystemSubstitutionMap,
    action: LinearSubstitutionAction,
    counters: _RunCounters | None = None,
) -> SystemSubstitutionMap | None:
    """Apply one substitution action, returning None when it is degenerate."""
    poly_index, var = action
    try:
        child_map = add_linear_substitution(sub_map, poly_index, var)
    except ValueError as error:
        if str(error) != "degenerate substitution":
            raise
        return None
    out = _drop_zero_equations(child_map)
    if counters is not None:
        counters.successful_substitutions += 1
    return out


def _state_priority(sub_map: SystemSubstitutionMap) -> tuple[int, int]:
    """Compare states by primary objective, then by eliminated variables."""
    return (len(sub_map.remaining_equations), -len(sub_map.substituted_variables))


def _dfs(
    sub_map: SystemSubstitutionMap,
    memo: dict[frozenset[Symbol], SystemSubstitutionMap],
    counters: _RunCounters,
) -> SystemSubstitutionMap:
    """
    Recursively explore all substitution branches and return the leaf with the
    fewest remaining equations.

    Memoizes by the frozenset of substituted variables: two paths that have
    eliminated the same set of variables reach an equivalent state, so the best
    result from that state only needs to be computed once.
    """
    counters.search_nodes_visited += 1
    
    kappa = symbols("kappah")
    solved = len(sub_map.remaining_equations) == 0
    is_kappah_zero = kappa in sub_map.substituted_variables and sub_map.substitution_map[kappa] == 0
    if solved and not is_kappah_zero:
        print("solved and kappa is not zero")
        return sub_map

    state_key = frozenset(sub_map.substituted_variables)
    if state_key in memo:
        return memo[state_key]

    child_results = []
    for action in _linear_substitution_candidates(sub_map):
        child_map = _try_apply_action(sub_map, action, counters)
        if child_map is None:
            continue

        child_result = _dfs(child_map, memo, counters)
        child_solved = len(child_result.remaining_equations) == 0
        child_kappa_is_zero = (
            kappa in child_result.substituted_variables
            and child_result.substitution_map[kappa] == 0
        )
        if child_solved and not child_kappa_is_zero:
            memo[state_key] = child_result
            return child_result
        child_results.append(child_result)
    
    if not child_results:
        memo[state_key] = sub_map
        return sub_map

    best = min(child_results, key=_state_priority)
    memo[state_key] = best
    return best


def best_linear_substitutions_order(equations: list[Expr]) -> tuple[SystemSubstitutionMap, LinearSubstitutionOrderStats]:
    """
    Find a substitution ordering that minimizes the remaining equations.

    Args:
        equations: The polynomial equations, each understood as equal to zero.

    Returns:
        The chosen ``SystemSubstitutionMap`` and run statistics for the search.
    """
    initial = SystemSubstitutionMap(
        remaining_equations=equations,
        substitution_map={},
        free_variables={
            cast(Symbol, sym) for poly in equations for sym in poly.free_symbols
        },
        substituted_variables=set(),
    )

    counters = _RunCounters()
    result = _dfs(initial, {}, counters)

    stats = LinearSubstitutionOrderStats(
        eliminations=len(equations) - len(result.remaining_equations),
        successful_substitutions=counters.successful_substitutions,
        search_nodes_visited=counters.search_nodes_visited,
    )
    
    return result, stats