import pytest
from sympy import Expr, Poly, S, simplify, symbols

from soothing.linear_substitution_order import (
    LinearSubstitutionOrderStats,
    _valid_action_results,
    best_linear_substitutions_order,
)
from soothing.system_substitution import SystemSubstitutionMap

x, y, z, w = symbols("x y z w")


def _as_exprs(*polys: Poly) -> list[Expr]:
    return [poly.as_expr() for poly in polys]


def test_best_linear_substitutions_order_no_linear_equations() -> None:
    poly = Poly(x**2 + y**2 - 1, x, y)
    result, stats = best_linear_substitutions_order(_as_exprs(poly), strategy="greedy")
    assert len(result.remaining_equations) == 1
    assert result.substitution_map == {}
    assert result.substituted_variables == set()
    assert stats.eliminations == 0


def test_best_linear_substitutions_order_single_linear_equation() -> None:
    poly = Poly(x + y, x, y)
    result, stats = best_linear_substitutions_order(_as_exprs(poly), strategy="greedy")
    assert len(result.remaining_equations) == 0
    assert len(result.substituted_variables) == 1
    assert stats.eliminations == 1
    assert stats.successful_substitutions >= 1
    assert stats.search_nodes_visited == 1


def test_best_linear_substitutions_order_fully_linear_system() -> None:
    poly0 = Poly(x + y, x, y)
    poly1 = Poly(y + z, y, z)
    result, _ = best_linear_substitutions_order(_as_exprs(poly0, poly1), strategy="greedy")
    assert len(result.remaining_equations) == 0
    assert len(result.substituted_variables) == 2


def test_best_linear_substitutions_order_partial_system() -> None:
    poly0 = Poly(x + y, x, y)
    poly1 = Poly(x**2 + z**2 - 1, x, z)
    result, _ = best_linear_substitutions_order(_as_exprs(poly0, poly1), strategy="greedy")
    assert len(result.remaining_equations) == 1
    assert len(result.substituted_variables) == 1


def test_best_linear_substitutions_order_three_variable_chain() -> None:
    poly0 = Poly(x + y, x, y)
    poly1 = Poly(y + z, y, z)
    poly2 = Poly(z + w, z, w)
    result, _ = best_linear_substitutions_order(
        _as_exprs(poly0, poly1, poly2), strategy="greedy"
    )
    assert len(result.remaining_equations) == 0
    assert len(result.substituted_variables) == 3


def test_best_linear_substitutions_order_substitution_map_solves_equations() -> None:
    poly = Poly(x + 2 * y, x, y)
    result, _ = best_linear_substitutions_order(_as_exprs(poly), strategy="greedy")
    substituted_var = next(iter(result.substituted_variables))
    substitution_expr = result.substitution_map[substituted_var]
    residual = poly.as_expr().subs(substituted_var, substitution_expr)
    assert simplify(residual) == 0


def test_best_linear_substitutions_order_variable_sets_are_disjoint() -> None:
    poly0 = Poly(x + y, x, y)
    poly1 = Poly(y + z, y, z)
    result, _ = best_linear_substitutions_order(_as_exprs(poly0, poly1), strategy="greedy")
    assert result.free_variables.isdisjoint(result.substituted_variables)
    assert result.free_variables | result.substituted_variables == {x, y, z}


def test_best_linear_substitutions_order_prefers_deeper_branch() -> None:
    a, b_sym, c_sym = symbols("a b c")
    poly0 = Poly(a + b_sym, a, b_sym, c_sym)
    poly1 = Poly(a + c_sym, a, b_sym, c_sym)
    result, _ = best_linear_substitutions_order(_as_exprs(poly0, poly1), strategy="greedy")
    assert len(result.remaining_equations) == 0
    assert len(result.substituted_variables) == 2


def test_best_linear_substitutions_order_non_linear_system_unchanged_equations() -> None:
    poly0 = Poly(x**2 - 1, x)
    poly1 = Poly(y**2 - 4, y)
    result, stats = best_linear_substitutions_order(_as_exprs(poly0, poly1), strategy="greedy")
    assert len(result.remaining_equations) == 2
    assert result.substitution_map == {}
    assert stats.eliminations == 0


def test_best_linear_substitutions_order_returns_system_substitution_map() -> None:
    poly = Poly(x + y, x, y)
    result, stats = best_linear_substitutions_order(_as_exprs(poly), strategy="greedy")
    assert isinstance(result, SystemSubstitutionMap)
    assert isinstance(stats, LinearSubstitutionOrderStats)


def test_best_linear_substitutions_order_empty_system() -> None:
    result, stats = best_linear_substitutions_order([])
    assert result.remaining_equations == []
    assert result.substitution_map == {}
    assert result.substituted_variables == set()
    assert stats.eliminations == 0
    assert stats.search_nodes_visited == 0


def test_best_linear_substitutions_order_defaults_to_dfs() -> None:
    poly0 = Poly(x + y, x, y)
    poly1 = Poly(y + z, y, z)
    default_result, _ = best_linear_substitutions_order(
        _as_exprs(poly0, poly1), strategy="greedy"
    )
    explicit_result, _ = best_linear_substitutions_order(
        _as_exprs(poly0, poly1), strategy="dfs"
    )
    assert default_result.remaining_equations == explicit_result.remaining_equations
    assert default_result.substitution_map == explicit_result.substitution_map
    assert default_result.free_variables == explicit_result.free_variables
    assert default_result.substituted_variables == explicit_result.substituted_variables


def test_best_linear_substitutions_order_accepts_explicit_dfs_strategy() -> None:
    poly = Poly(x + y, x, y)
    result, stats = best_linear_substitutions_order(_as_exprs(poly), strategy="dfs")
    assert len(result.remaining_equations) == 0
    assert len(result.substituted_variables) == 1
    assert stats.eliminations == 1
    assert stats.search_nodes_visited >= 1


def test_best_linear_substitutions_order_rejects_unknown_strategy() -> None:
    poly = Poly(x + y, x, y)
    with pytest.raises(ValueError, match="Unknown strategy"):
        best_linear_substitutions_order(_as_exprs(poly), strategy="unknown")


def test_best_linear_substitutions_order_accepts_explicit_greedy_strategy() -> None:
    poly = Poly(x + y, x, y)
    result, _ = best_linear_substitutions_order(_as_exprs(poly), strategy="greedy")
    assert len(result.remaining_equations) == 0
    assert len(result.substituted_variables) == 1


def test_best_linear_substitutions_order_mcts_is_deterministic_with_seed() -> None:
    a, b_sym, c_sym = symbols("a b c")
    poly0 = Poly(a + b_sym, a, b_sym, c_sym)
    poly1 = Poly(a + c_sym, a, b_sym, c_sym)
    first, stats_a = best_linear_substitutions_order(
        _as_exprs(poly0, poly1),
        strategy="mcts",
        random_seed=7,
        mcts_iterations=32,
        mcts_rollout_depth=2,
    )
    second, stats_b = best_linear_substitutions_order(
        _as_exprs(poly0, poly1),
        strategy="mcts",
        random_seed=7,
        mcts_iterations=32,
        mcts_rollout_depth=2,
    )
    assert first.remaining_equations == second.remaining_equations
    assert first.substitution_map == second.substitution_map
    assert first.free_variables == second.free_variables
    assert first.substituted_variables == second.substituted_variables
    assert stats_a == stats_b
    assert stats_a.search_nodes_visited >= 32


def test_best_linear_substitutions_order_mcts_matches_dfs_on_trivial_system() -> None:
    poly = Poly(x + y, x, y)
    dfs_result, _ = best_linear_substitutions_order(_as_exprs(poly), strategy="dfs")
    mcts_result, _ = best_linear_substitutions_order(
        _as_exprs(poly),
        strategy="mcts",
        random_seed=5,
        mcts_iterations=8,
        mcts_rollout_depth=1,
    )
    assert mcts_result.remaining_equations == dfs_result.remaining_equations
    assert mcts_result.substitution_map == dfs_result.substitution_map
    assert mcts_result.free_variables == dfs_result.free_variables
    assert mcts_result.substituted_variables == dfs_result.substituted_variables


def test_best_linear_substitutions_order_mcts_reaches_optimum_on_small_branch() -> None:
    a, b_sym, c_sym = symbols("a b c")
    poly0 = Poly(a + b_sym, a, b_sym, c_sym)
    poly1 = Poly(a * b_sym + c_sym, a, b_sym, c_sym)
    result, _ = best_linear_substitutions_order(
        _as_exprs(poly0, poly1),
        strategy="mcts",
        random_seed=11,
        mcts_iterations=64,
        mcts_rollout_depth=2,
    )
    assert len(result.remaining_equations) == 0
    assert len(result.substituted_variables) == 2


def test_valid_action_results_filters_degenerate_actions() -> None:
    sub_map = SystemSubstitutionMap(
        remaining_equations=[Poly(y * x + 1, x, y).as_expr(), Poly(x + z, x, z).as_expr()],
        substitution_map={y: S.Zero},
        free_variables={x, z},
        substituted_variables={y},
    )
    valid_actions = _valid_action_results(sub_map)
    assert [(poly_index, str(var)) for (poly_index, var), _ in valid_actions] == [
        (1, "x"),
        (1, "z"),
    ]


def test_best_linear_substitutions_order_mcts_skips_degenerate_and_stays_valid() -> None:
    from sympy import nan, zoo

    poly0 = Poly(y, x, y, z)
    poly1 = Poly(y * x + 1, x, y, z)
    poly2 = Poly(x + z, x, y, z)
    result, _ = best_linear_substitutions_order(
        _as_exprs(poly0, poly1, poly2),
        strategy="mcts",
        random_seed=3,
        mcts_iterations=64,
        mcts_rollout_depth=3,
    )
    assert len(result.substitution_map) >= 2
    for expr in result.substitution_map.values():
        assert not (hasattr(expr, "has") and (expr.has(zoo) or expr.has(nan)))


def test_best_linear_substitutions_order_skips_degenerate_and_continues() -> None:
    from sympy import nan, zoo

    poly0 = Poly(x + y, x, y)
    poly1 = Poly(z * x + z, x, z)
    result, _ = best_linear_substitutions_order(_as_exprs(poly0, poly1), strategy="greedy")
    assert len(result.substitution_map) >= 1
    for expr in result.substitution_map.values():
        assert not (hasattr(expr, "has") and (expr.has(zoo) or expr.has(nan)))


def test_best_linear_substitutions_order_full_system_no_nan_zoo() -> None:
    from soothing.parse_mathematica import mathematica_to_sympy
    from sympy import nan, zoo

    system = r"""0 -> (DpDph1 e (-3 + 2 xr))/(256 \[Pi]^2), kappah -> (DpDph1^3 e^3 (-1 + xr))/(768 \[Pi]^2)"""
    polys, _ = mathematica_to_sympy(system)
    result, _ = best_linear_substitutions_order([poly.as_expr() for poly in polys], strategy="greedy")
    for expr in result.substitution_map.values():
        assert not (hasattr(expr, "has") and (expr.has(zoo) or expr.has(nan)))
