from dataclasses import dataclass, field
from math import log, sqrt
from random import Random
from typing import cast

from sympy import Expr, Symbol, default_sort_key, symbols

from .system_substitution import (
    is_zero,
    SystemSubstitutionMap,
    add_linear_substitution,
    is_linear_in_variable,
)

__all__ = ["best_linear_substitutions_order"]

_MCTS_DEFAULT_ITERATIONS = 4
_MCTS_DEFAULT_ROLLOUT_DEPTH = 4
_MCTS_DEFAULT_EXPLORATION_WEIGHT = sqrt(2.0)
_MCTS_DEFAULT_RANDOM_SEED = 0

LinearSubstitutionAction = tuple[int, Symbol]


@dataclass(slots=True)
class _MCTSNode:
    state: SystemSubstitutionMap
    unexpanded_children: list[tuple[LinearSubstitutionAction, SystemSubstitutionMap]]
    parent: "_MCTSNode | None" = None
    action: LinearSubstitutionAction | None = None
    visits: int = 0
    total_value: float = 0.0
    children: list["_MCTSNode"] = field(default_factory=list)


def _linear_substitution_candidates(
    sub_map: SystemSubstitutionMap,
) -> list[LinearSubstitutionAction]:
    """Return all (poly_index, var) pairs where the polynomial is linear in that variable."""
    candidates: list[LinearSubstitutionAction] = []
    for poly_index, poly in enumerate(sub_map.remaining_equations):
        variables = sorted(cast(set[Symbol], poly.free_symbols), key=default_sort_key)
        for var in variables:
            if is_linear_in_variable(sub_map, poly_index, var):
                candidates.append((poly_index, var))
    return candidates


def _equation_linearity_count(sub_map: SystemSubstitutionMap, poly_index: int) -> int:
    """Return the number of variables the equation at poly_index is linear in."""
    poly = sub_map.remaining_equations[poly_index]
    return sum(
        1
        for v in cast(set[Symbol], poly.free_symbols)
        if is_linear_in_variable(sub_map, poly_index, v)
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
    sub_map: SystemSubstitutionMap, action: LinearSubstitutionAction
) -> SystemSubstitutionMap | None:
    """Apply one substitution action, returning None when it is degenerate."""
    poly_index, var = action
    try:
        child_map = add_linear_substitution(sub_map, poly_index, var)
    except ValueError as error:
        if str(error) != "degenerate substitution":
            raise
        return None
    return _drop_zero_equations(child_map)


def _valid_action_results(
    sub_map: SystemSubstitutionMap,
) -> list[tuple[LinearSubstitutionAction, SystemSubstitutionMap]]:
    """Return valid actions together with the states they produce."""
    results: list[tuple[LinearSubstitutionAction, SystemSubstitutionMap]] = []
    for action in _linear_substitution_candidates(sub_map):
        child_map = _try_apply_action(sub_map, action)
        if child_map is not None:
            results.append((action, child_map))
    return results


def _state_priority(sub_map: SystemSubstitutionMap) -> tuple[int, int]:
    """Compare states by primary objective, then by eliminated variables."""
    return (len(sub_map.remaining_equations), -len(sub_map.substituted_variables))


def _state_reward(
    sub_map: SystemSubstitutionMap,
    *,
    initial_equation_count: int,
    initial_variable_count: int,
) -> float:
    """Return a bounded reward where larger is better."""
    if initial_equation_count == 0:
        elimination_score = 1.0
    else:
        elimination_score = (
            initial_equation_count - len(sub_map.remaining_equations)
        ) / initial_equation_count

    if initial_variable_count == 0:
        substitution_score = 0.0
    else:
        substitution_score = len(sub_map.substituted_variables) / initial_variable_count

    return elimination_score + 0.01 * substitution_score


def _greedy_ordered_candidates(
    sub_map: SystemSubstitutionMap,
    candidates: list[LinearSubstitutionAction],
) -> list[LinearSubstitutionAction]:
    """
    Return candidates filtered and sorted by the greedy heuristic.

    The best action is the one that leaves the richest next frontier: first the
    most remaining equations that are linear in some variable, then the largest
    number of available substitution actions, then the simplest source equation.

    Args:
        sub_map: The current substitution state.
        candidates: All available (poly_index, var) pairs from
            _linear_substitution_candidates.

    Returns:
        The candidate actions ordered from best to worst according to the
        heuristic.
    """
    scored: list[
        tuple[int, int, int, tuple[object, ...], LinearSubstitutionAction]
    ] = []
    for action in candidates:
        child_map = _try_apply_action(sub_map, action)
        if child_map is None:
            continue
        future_candidates = _linear_substitution_candidates(child_map)
        linear_equations_after_move = len({poly_index for poly_index, _ in future_candidates})
        scored.append(
            (
                -linear_equations_after_move,
                -len(future_candidates),
                _equation_linearity_count(sub_map, action[0]),
                default_sort_key(action[1]),
                action,
            )
        )
    scored.sort()
    return [action for *_, action in scored]


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
    state_key = frozenset(sub_map.substituted_variables)
    if state_key in memo:
        return memo[state_key]

    child_results = [
        _dfs(child_map, memo) for _, child_map in _valid_action_results(sub_map)
    ]
    if not child_results:
        memo[state_key] = sub_map
        return sub_map

    best = min(child_results, key=_state_priority)
    memo[state_key] = best
    return best


def _new_mcts_node(
    state: SystemSubstitutionMap,
    *,
    parent: _MCTSNode | None = None,
    action: LinearSubstitutionAction | None = None,
) -> _MCTSNode:
    unexpanded_children = _valid_action_results(state)
    
    return _MCTSNode(
        state=state,
        unexpanded_children=unexpanded_children,
        parent=parent,
        action=action,
    )


def _select_mcts_child(
    node: _MCTSNode, exploration_weight: float
) -> _MCTSNode:
    """Select a child using the UCT rule."""

    def uct_score(child: _MCTSNode) -> float:
        if child.visits == 0:
            return float("inf")
        exploitation = child.total_value / child.visits
        exploration = exploration_weight * sqrt(log(node.visits) / child.visits)
        return exploitation + exploration

    return max(node.children, key=uct_score)


def _rollout(
    start_state: SystemSubstitutionMap,
    *,
    rollout_depth: int,
    rng: Random,
) -> SystemSubstitutionMap:
    """Simulate a bounded random playout from start_state."""
    state = start_state
    for _ in range(rollout_depth):
        action_results = _valid_action_results(state)
        if not action_results:
            return state
        _, state = action_results[rng.randrange(len(action_results))]
    return state


def _mcts(
    initial: SystemSubstitutionMap,
    *,
    iterations: int,
    rollout_depth: int,
    exploration_weight: float,
    random_seed: int,
) -> SystemSubstitutionMap:
    """Run a bounded Monte Carlo tree search and return the best discovered state."""
    if iterations <= 0:
        raise ValueError("mcts_iterations must be positive")
    if rollout_depth < 0:
        raise ValueError("mcts_rollout_depth must be non-negative")
    if exploration_weight < 0:
        raise ValueError("mcts_exploration_weight must be non-negative")

    root = _new_mcts_node(initial)
    best_state = initial
    best_priority = _state_priority(initial)
    initial_equation_count = len(initial.remaining_equations)
    initial_variable_count = len(initial.free_variables)
    rng = Random(random_seed)
    
    for _ in range(iterations):
        node = root
        path = [node]


        while not node.unexpanded_children and node.children:
            node = _select_mcts_child(node, exploration_weight)
            path.append(node)

        if node.unexpanded_children:
            action, child_state = node.unexpanded_children.pop(0)
            child_node = _new_mcts_node(child_state, parent=node, action=action)
            node.children.append(child_node)
            node = child_node
            path.append(node)
            if _state_priority(child_state) < best_priority:
                best_state = child_state
                best_priority = _state_priority(child_state)

        rollout_state = _rollout(node.state, rollout_depth=rollout_depth, rng=rng)
        if _state_priority(rollout_state) < best_priority:
            best_state = rollout_state
            best_priority = _state_priority(rollout_state)

        reward = _state_reward(
            rollout_state,
            initial_equation_count=initial_equation_count,
            initial_variable_count=initial_variable_count,
        )
        for path_node in path:
            path_node.visits += 1
            path_node.total_value += reward

    return best_state


def _greedy_search(initial: SystemSubstitutionMap) -> SystemSubstitutionMap:
    """Repeatedly apply the locally best valid substitution."""
    sub_map = initial
    while True:
        candidates = _linear_substitution_candidates(sub_map)
        if not candidates:
            return sub_map

        for action in _greedy_ordered_candidates(sub_map, candidates):
            child_map = _try_apply_action(sub_map, action)
            if child_map is not None:
                sub_map = child_map
                break
        else:
            return sub_map


def best_linear_substitutions_order(
    equations: list[Expr],
    strategy: str = "greedy",
    *,
    mcts_iterations: int = _MCTS_DEFAULT_ITERATIONS,
    mcts_rollout_depth: int = _MCTS_DEFAULT_ROLLOUT_DEPTH,
    mcts_exploration_weight: float = _MCTS_DEFAULT_EXPLORATION_WEIGHT,
    random_seed: int = _MCTS_DEFAULT_RANDOM_SEED,
) -> SystemSubstitutionMap:
    """
    Find a substitution ordering that minimizes the remaining equations.

    Search strategies:

    - ``dfs``: exhaustively explores all valid substitution sequences and
      returns the best reachable leaf.
    - ``greedy``: repeatedly chooses the locally best action according to a
      one-step heuristic.
    - ``mcts``: uses a bounded Monte Carlo tree search with deterministic
      random seed control.

    The legacy ``greedy`` boolean is still accepted as a compatibility shim.

    Args:
        equations: The polynomial equations, each understood as equal to zero.
        strategy: Explicit search strategy. One of ``\"dfs\"``, ``\"greedy\"``,
            or ``\"mcts\"``. When omitted, the value of ``greedy`` is used.
        mcts_iterations: Number of MCTS iterations to perform.
        mcts_rollout_depth: Maximum rollout depth per MCTS iteration.
        mcts_exploration_weight: Exploration coefficient in the UCT score.
        random_seed: Seed used by the MCTS rollout RNG.

    Returns:
        A SystemSubstitutionMap chosen by the requested strategy.
    """
    initial = SystemSubstitutionMap(
        remaining_equations=equations,
        substitution_map={},
        free_variables={
            cast(Symbol, sym) for poly in equations for sym in poly.free_symbols
        },
        substituted_variables=set(),
    )

    if strategy == "dfs":
        return _dfs(initial, {})
    elif strategy == "greedy":
        return _greedy_search(initial)
    elif strategy == "mcts":
        return _mcts(
            initial,
            iterations=mcts_iterations,
            rollout_depth=mcts_rollout_depth,
            exploration_weight=mcts_exploration_weight,
            random_seed=random_seed,
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")