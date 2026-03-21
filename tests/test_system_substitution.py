from typing import cast

import pytest
from sympy import Expr, Poly, Rational, S, Symbol, symbols

from soothing.system_substitution import (
    SystemSubstitutionMap,
    _back_substitute,
    _propagate_constants,
    add_linear_substitution,
    is_linear_in_variable,
    is_zero,
    validate_substitution_map,
)

x, y, z, w = symbols("x y z w")


def make_sub_map(
    equations: list[Expr | Poly],
    substitution_map: dict[Symbol, Expr] | None = None,
    free_variables: set[Symbol] | None = None,
    substituted_variables: set[Symbol] | None = None,
) -> SystemSubstitutionMap:
    normalized_sub_map = substitution_map or {}
    equations_expr = [
        equation.as_expr() if isinstance(equation, Poly) else equation
        for equation in equations
    ]
    all_syms = {
        cast(Symbol, sym) for equation in equations_expr for sym in equation.free_symbols
    }
    keys = set(normalized_sub_map.keys())
    free = free_variables if free_variables is not None else all_syms - keys
    sub = substituted_variables if substituted_variables is not None else keys
    equation_symbols = frozenset(free | sub)

    return SystemSubstitutionMap(
        remaining_equations=equations_expr,
        substitution_map=normalized_sub_map,
        equation_symbols=equation_symbols,
    )


def sub_map_from_poly(poly: Poly, substitution_map: dict[Symbol, Expr]) -> SystemSubstitutionMap:
    """Build map with equation_symbols = symbols appearing in poly, keys, and substitution values."""
    syms: set[Symbol] = {cast(Symbol, s) for s in poly.free_symbols}
    for key, val in substitution_map.items():
        syms.add(key)
        syms.update(cast(Symbol, s) for s in val.free_symbols)
    return SystemSubstitutionMap(
        remaining_equations=[poly.as_expr()],
        substitution_map=substitution_map,
        equation_symbols=frozenset(syms),
    )


# ── _back_substitute ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("expr", "substitution_map", "exclude", "expected"),
    [
        (x + y, {y: z + 1, z: w + 2}, None, x + w + 3),
        (y + z, {y: x + 1, z: y + 2}, {y}, 2 * y + 2),
        (y + z, {y: x + 1, z: y + 2}, {y, z}, y + z),
    ],
    ids=["chains", "respects_exclude", "all_excluded"],
)
def test_back_substitute(
    expr: Expr,
    substitution_map: dict[Symbol, Expr],
    exclude: set[Symbol] | None,
    expected: Expr,
) -> None:
    kwargs = {} if exclude is None else {"exclude": exclude}
    assert _back_substitute(expr, substitution_map, **kwargs) == expected


# ── is_linear_in_variable ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("poly", "expected"),
    [
        (Poly(3 * x + 2 * y - 1, x, y), True),
        (Poly(x**2 + y, x, y), False),
        (Poly(3 * y + 1, x, y), False),
        (Poly(y * x + z, x, y, z), True),
        (Poly(x * y + x**2, x, y), False),
        (Poly(x**3 + y, x, y), False),
        (Poly(y**10 * x + z**5, x, y, z), True),
        (Poly(5, x), False),
    ],
    ids=[
        "plain_linear",
        "quadratic",
        "var_absent",
        "linear_with_coeff",
        "product_of_vars",
        "cubic",
        "high_degree_other_vars",
        "constant_poly",
    ],
)
def test_is_linear_in_variable(poly: Poly, expected: bool) -> None:
    assert is_linear_in_variable(make_sub_map([poly]), 0, x) is expected


@pytest.mark.parametrize(
    ("poly_index", "expected"),
    [(0, False), (1, True)],
)
def test_is_linear_selects_correct_poly_by_index(poly_index: int, expected: bool) -> None:
    poly0 = Poly(x**2 + y, x, y)
    poly1 = Poly(x + y, x, y)
    assert is_linear_in_variable(make_sub_map([poly0, poly1]), poly_index, x) is expected


def test_is_linear_with_substituted_variable_in_poly() -> None:
    poly = Poly(x + z, x, z)
    assert is_linear_in_variable(sub_map_from_poly(poly, {z: 2 * w}), 0, x) is True


def test_is_linear_with_chained_substitution_not_obscuring_linearity() -> None:
    poly = Poly(x + y, x, y)
    assert is_linear_in_variable(sub_map_from_poly(poly, {z: w, y: z}), 0, x) is True


@pytest.mark.parametrize(
    ("poly", "substitution_map"),
    [
        (Poly(y + z, y, z), {y: x**2}),
        (Poly(z + w, z, w), {y: x + 1, z: y**2}),
        (Poly(y + z, y, z), {y: x**2 + x}),
        (Poly(y + w, y, w), {y: x**3}),
        (Poly(z + w, z, w), {y: x**2 + 1, z: y - 1}),
        (Poly(x**2 * y + x, x, y), {}),
        (Poly(y + w, y, w), {y: x**2 * z}),
        (Poly(y + z + w, y, z, w), {y: x**2, z: x**3}),
        (Poly(y + w, y, w), {w: z + 1, y: x**2}),
    ],
    ids=[
        "quadratic_via_direct_substitution",
        "chained_quadratic_substitution",
        "product_hidden_by_substitution",
        "cubic_via_substitution",
        "two_hop_chain_squared",
        "nonlinear_coefficient_on_var",
        "var_nonlinear_in_substituted_expr",
        "multiple_substituted_nonlinear",
        "quadratic_masked_by_independent_substitution",
    ],
)
def test_is_linear_false_positive_nonlinear_after_substitution(
    poly: Poly, substitution_map: dict[Symbol, Expr]
) -> None:
    sm = sub_map_from_poly(poly, substitution_map)
    assert is_linear_in_variable(sm, 0, x) is False


def test_is_linear_false_positive_three_hop_nonlinear_chain() -> None:
    a, b_sym, c_sym = symbols("a b c")
    poly = Poly(c_sym + z, c_sym, z)
    sm = sub_map_from_poly(poly, {a: x**2, b_sym: a + 1, c_sym: b_sym - 1})
    assert is_linear_in_variable(sm, 0, x) is False


@pytest.mark.parametrize(
    ("poly", "substitution_map"),
    [
        (Poly(Rational(1, 10**9) * x**2 + x, x), {}),
        (Poly(-Rational(1, 10**9) * x**2 + x, x), {}),
        (Poly(y + x, y, x), {y: Rational(1, 10**9) * x**2}),
        (Poly(Rational(1, 10**9) * x**2 + 10**6 * x, x), {}),
        (
            Poly(y + z + x, y, z, x),
            {y: (1 + Rational(1, 10**9)) * x**2, z: -(x**2)},
        ),
        (Poly(Rational(1, 10**9) * y * x**2 + x, x, y), {}),
        (Poly(Rational(1, 10**9) * y * w * x**2 + x, x, y, w), {}),
        (Poly(Rational(1, 10**9) * x**3 + x, x), {}),
    ],
    ids=[
        "epsilon_quadratic_direct",
        "epsilon_quadratic_negative_coeff",
        "epsilon_quadratic_via_substitution",
        "large_linear_term",
        "chained_cancellation_epsilon_quadratic",
        "free_symbol_scales_quadratic",
        "two_free_symbols_scale_quadratic",
        "epsilon_cubic",
    ],
)
def test_is_linear_numerical_false_positive(
    poly: Poly, substitution_map: dict[Symbol, Expr]
) -> None:
    sm = sub_map_from_poly(poly, substitution_map)
    assert is_linear_in_variable(sm, 0, x) is False


def test_is_linear_numerical_false_positive_two_hop_chain_epsilon_quadratic() -> None:
    a, b_sym = symbols("a b")
    poly = Poly(b_sym + x, b_sym, x)
    sm = sub_map_from_poly(poly, {a: x**2, b_sym: Rational(1, 10**9) * a})
    assert is_linear_in_variable(sm, 0, x) is False


# ── add_linear_substitution ───────────────────────────────────────────────────


def test_add_linear_substitution_x_plus_y_removes_equation_and_maps_x() -> None:
    poly = Poly(x + y, x, y)
    sub_map = make_sub_map([poly])
    result = add_linear_substitution(sub_map, 0, x)
    assert result.substitution_map[x] == -y
    assert len(result.remaining_equations) == 0
    assert result.equation_symbols == sub_map.equation_symbols


def test_add_linear_substitution_fraction_and_coefficient() -> None:
    assert add_linear_substitution(make_sub_map([Poly(x * y + z, x, y, z)]), 0, x).substitution_map[
        x
    ] == -z / y
    assert add_linear_substitution(make_sub_map([Poly(2 * x + y, x, y)]), 0, x).substitution_map[
        x
    ] == -y / 2


def test_add_linear_substitution_preserves_other_equation_and_index() -> None:
    poly0 = Poly(x + y, x, y).as_expr()
    poly1 = Poly(y + z, y, z).as_expr()
    r0 = add_linear_substitution(make_sub_map([poly0, poly1]), 0, x)
    assert r0.remaining_equations == [poly1]

    poly_a = Poly(x**2 + y, x, y).as_expr()
    poly_b = Poly(x + z, x, z).as_expr()
    r1 = add_linear_substitution(make_sub_map([poly_a, poly_b]), 1, x)
    assert r1.remaining_equations == [poly_a]


@pytest.mark.parametrize(
    ("poly", "substitution_map", "expected_x"),
    [
        (Poly(x + z, x, z), {z: w}, -w),
        (Poly(y + z, y, z), {y: x + 1}, -(z + 1)),
        (Poly(z + w, z, w), {y: x + 1, z: y + 2}, -(w + 3)),
        (
            Poly(y + z + w, y, z, w),
            {y: x + 1, z: 2 * x},
            -(w + 1) / 3,
        ),
    ],
    ids=["independent_var", "var_dependent", "chained_var_dependent", "multiple_dependencies"],
)
def test_add_linear_substitution_resolves_x(
    poly: Poly, substitution_map: dict[Symbol, Expr], expected_x: Expr
) -> None:
    assert add_linear_substitution(sub_map_from_poly(poly, substitution_map), 0, x).substitution_map[
        x
    ] == expected_x


def test_add_linear_substitution_preserves_existing_map_and_immutability() -> None:
    poly = Poly(x + y, x, y)
    existing = {z: w + 1}
    sub_map = SystemSubstitutionMap(
        remaining_equations=[poly.as_expr()],
        substitution_map=existing,
        equation_symbols=frozenset({x, y, z}),
    )
    result = add_linear_substitution(sub_map, 0, x)
    assert result.substitution_map[z] == w + 1

    base = make_sub_map([poly])
    add_linear_substitution(base, 0, x)
    assert len(base.remaining_equations) == 1 and x not in base.substitution_map


def test_add_linear_substitution_complex_coefficients() -> None:
    poly = Poly((y**2 + z) * x + (w - 1), x, y, z, w)
    r = add_linear_substitution(make_sub_map([poly]), 0, x)
    assert r.substitution_map[x] == -(w - 1) / (y**2 + z)


def test_add_linear_substitution_raises_when_denominator_zero() -> None:
    poly = Poly(y * x + y, x, y)
    sm = sub_map_from_poly(poly, {y: y * 0})
    with pytest.raises(ValueError, match="degenerate"):
        add_linear_substitution(sm, 0, x)


# ── is_zero ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("polys", "poly_index", "expected"),
    [
        ([Poly(0, x, y)], 0, True),
        ([Poly(1, x)], 0, False),
        ([Poly(x + y, x, y)], 0, False),
        ([Poly(0, x), Poly(x + 1, x)], 0, True),
        ([Poly(0, x), Poly(x + 1, x)], 1, False),
        ([Poly(x**2 + y, x, y)], 0, False),
    ],
    ids=[
        "zero_poly",
        "nonzero_constant",
        "nonzero_poly",
        "index0_zero",
        "index1_nonzero",
        "no_substitution_nonzero",
    ],
)
def test_is_zero(polys: list[Expr | Poly], poly_index: int, expected: bool) -> None:
    assert is_zero(make_sub_map(polys), poly_index) is expected


@pytest.mark.parametrize(
    ("poly", "substitution_map", "expected"),
    [
        (Poly(y - 1, y), {y: S.One}, True),
        (Poly(y - 1, y), {y: S(2)}, False),
        (Poly(z, z), {y: S.Zero, z: y}, True),
        (Poly(x - y, x, y), {x: y}, True),
    ],
    ids=["subst_to_zero", "subst_nonzero", "chained_to_zero", "expr_simplifies_to_zero"],
)
def test_is_zero_with_substitution(
    poly: Poly, substitution_map: dict[Symbol, Expr], expected: bool
) -> None:
    assert is_zero(sub_map_from_poly(poly, substitution_map), 0) is expected


# ── _propagate_constants ───────────────────────────────────────────────────────


def test_propagate_constants() -> None:
    assert _propagate_constants({}) == {}

    assert _propagate_constants({x: y + 1, y: z}) == {x: y + 1, y: z}

    r = _propagate_constants({x: Rational(3), y: x + 1})
    assert r[x] == Rational(3) and r[y] == Rational(4)

    assert _propagate_constants({x: Rational(5), y: x * 2})[x] == Rational(5)

    r2 = _propagate_constants({x: Rational(2), y: x + 1, z: y * 3})
    assert r2 == {x: Rational(2), y: Rational(3), z: Rational(9)}

    a, b, c, d = symbols("a b c d")
    r3 = _propagate_constants({
        a: Rational(1),
        b: a + 1,
        c: b + 1,
        d: c + 1,
    })
    assert r3 == {a: Rational(1), b: Rational(2), c: Rational(3), d: Rational(4)}

    r4 = _propagate_constants({x: Rational(2), y: z + x})
    assert r4[x] == Rational(2) and r4[y] == z + 2 and z not in r4

    assert _propagate_constants({x: Rational(1), y: Rational(2), z: x + y})[z] == Rational(3)

    initial = {x: Rational(3), y: x + 1, z: y * 2}
    once = _propagate_constants(initial)
    assert once == _propagate_constants(once)

    a2, b2, c2, d2, e = symbols("a b c d e")
    r5 = _propagate_constants({
        a2: Rational(2),
        b2: a2 + 1,
        c2: b2 + d2,
        e: c2 + b2,
    })
    assert r5[a2] == Rational(2) and r5[b2] == Rational(3)
    assert r5[c2] == d2 + 3 and r5[e] == c2 + 3


# ── validate_substitution_map ────────────────────────────────────────────────


@pytest.mark.parametrize("z_val, ok", [(2, True), (3, False)])
def test_validate_substitution_map_nested(z_val: int, ok: bool) -> None:
    xa, ya, za = symbols("x y z")
    system = [xa - ya - 1, ya - za, za - 2]
    substitution_map = {xa: ya + 1, ya: za, za: z_val}
    assert validate_substitution_map(system, substitution_map) is ok

