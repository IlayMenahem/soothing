from sympy import Expr, Poly, simplify, symbols

from soothing.linear_substitution_order import best_linear_substitutions_order

x, y, z, w = symbols("x y z w")


def _as_exprs(*polys: Poly) -> list[Expr]:
    return [poly.as_expr() for poly in polys]


def test_best_linear_substitutions_order_no_linear_equations() -> None:
    poly = Poly(x**2 + y**2 - 1, x, y)
    result, stats = best_linear_substitutions_order(_as_exprs(poly))
    assert len(result.remaining_equations) == 1
    assert result.substitution_map == {}
    assert result.substituted_variables == set()
    assert stats.eliminations == 0


def test_best_linear_substitutions_order_fully_linear_system() -> None:
    poly0 = Poly(x + y, x, y)
    poly1 = Poly(y + z, y, z)
    result, _ = best_linear_substitutions_order(_as_exprs(poly0, poly1))
    assert len(result.remaining_equations) == 0
    assert len(result.substituted_variables) == 2


def test_best_linear_substitutions_order_partial_system() -> None:
    poly0 = Poly(x + y, x, y)
    poly1 = Poly(x**2 + z**2 - 1, x, z)
    result, _ = best_linear_substitutions_order(_as_exprs(poly0, poly1))
    assert len(result.remaining_equations) == 1
    assert len(result.substituted_variables) == 1


def test_best_linear_substitutions_order_three_variable_chain() -> None:
    poly0 = Poly(x + y, x, y)
    poly1 = Poly(y + z, y, z)
    poly2 = Poly(z + w, z, w)
    result, _ = best_linear_substitutions_order(_as_exprs(poly0, poly1, poly2))
    assert len(result.remaining_equations) == 0
    assert len(result.substituted_variables) == 3


def test_best_linear_substitutions_order_substitution_map_solves_equations() -> None:
    poly = Poly(x + 2 * y, x, y)
    result, _ = best_linear_substitutions_order(_as_exprs(poly))
    substituted_var = next(iter(result.substituted_variables))
    substitution_expr = result.substitution_map[substituted_var]
    residual = poly.as_expr().subs(substituted_var, substitution_expr)
    assert simplify(residual) == 0


def test_best_linear_substitutions_order_variable_sets_are_disjoint() -> None:
    poly0 = Poly(x + y, x, y)
    poly1 = Poly(y + z, y, z)
    result, _ = best_linear_substitutions_order(_as_exprs(poly0, poly1))
    assert result.free_variables.isdisjoint(result.substituted_variables)
    assert result.free_variables | result.substituted_variables == {x, y, z}


def test_best_linear_substitutions_order_prefers_deeper_branch() -> None:
    a, b_sym, c_sym = symbols("a b c")
    poly0 = Poly(a + b_sym, a, b_sym, c_sym)
    poly1 = Poly(a + c_sym, a, b_sym, c_sym)
    result, _ = best_linear_substitutions_order(_as_exprs(poly0, poly1))
    assert len(result.remaining_equations) == 0
    assert len(result.substituted_variables) == 2


def test_best_linear_substitutions_order_non_linear_system_unchanged_equations() -> None:
    poly0 = Poly(x**2 - 1, x)
    poly1 = Poly(y**2 - 4, y)
    result, stats = best_linear_substitutions_order(_as_exprs(poly0, poly1))
    assert len(result.remaining_equations) == 2
    assert result.substitution_map == {}
    assert stats.eliminations == 0


def test_best_linear_substitutions_order_empty_system() -> None:
    result, stats = best_linear_substitutions_order([])
    assert result.remaining_equations == []
    assert result.substitution_map == {}
    assert result.substituted_variables == set()
    assert stats.eliminations == 0
    assert stats.search_nodes_visited == 0


def test_best_linear_substitutions_order_accepts_explicit_dfs_strategy() -> None:
    poly = Poly(x + y, x, y)
    result, stats = best_linear_substitutions_order(_as_exprs(poly))
    assert len(result.remaining_equations) == 0
    assert len(result.substituted_variables) == 1
    assert stats.eliminations == 1
    assert stats.search_nodes_visited >= 1


def test_best_linear_substitutions_order_skips_degenerate_and_continues() -> None:
    from sympy import nan, zoo

    poly0 = Poly(x + y, x, y)
    poly1 = Poly(z * x + z, x, z)
    result, _ = best_linear_substitutions_order(_as_exprs(poly0, poly1))
    assert len(result.substitution_map) >= 1
    for expr in result.substitution_map.values():
        assert not (hasattr(expr, "has") and (expr.has(zoo) or expr.has(nan)))


def test_best_linear_substitutions_order_full_system_no_nan_zoo() -> None:
    from soothing.parse_mathematica import mathematica_to_sympy
    from sympy import nan, zoo

    system = r"""0 -> (DpDph1 e (-3 + 2 xr))/(256 \[Pi]^2), kappah -> (DpDph1^3 e^3 (-1 + xr))/(768 \[Pi]^2)"""
    polys, _ = mathematica_to_sympy(system)
    result, _ = best_linear_substitutions_order([poly.as_expr() for poly in polys])
    for expr in result.substitution_map.values():
        assert not (hasattr(expr, "has") and (expr.has(zoo) or expr.has(nan)))
