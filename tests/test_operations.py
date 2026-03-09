from sympy import Poly, symbols

from soothing.operations import (
    SystemSubstitutionMap,
    add_linear_substitution,
    is_linear_in_variable,
)

x, y, z, w = symbols("x y z w")


# ── Helpers ───────────────────────────────────────────────────────────────────


def make_sub_map(
    equations: list[Poly],
    substitution_map: dict = {},
    free_variables: set = None,
    substituted_variables: set = None,
) -> SystemSubstitutionMap:
    all_syms = {sym for poly in equations for sym in poly.free_symbols}
    return SystemSubstitutionMap(
        remaining_equations=equations,
        substitution_map=substitution_map,
        free_variables=free_variables
        if free_variables is not None
        else all_syms - substitution_map.keys(),
        substituted_variables=substituted_variables
        if substituted_variables is not None
        else set(substitution_map.keys()),
    )


# ── is_linear_in_variable ─────────────────────────────────────────────────────


def test_is_linear_returns_true_for_plain_linear_poly() -> None:
    # 3x + 2y - 1
    poly = Poly(3 * x + 2 * y - 1, x, y)
    sub_map = make_sub_map([poly])

    assert is_linear_in_variable(sub_map, 0, x) is True


def test_is_linear_returns_false_for_quadratic_poly() -> None:
    # x^2 + y
    poly = Poly(x**2 + y, x, y)
    sub_map = make_sub_map([poly])

    assert is_linear_in_variable(sub_map, 0, x) is False


def test_is_linear_returns_false_when_var_absent() -> None:
    # 3y + 1  — x does not appear
    poly = Poly(3 * y + 1, x, y)
    sub_map = make_sub_map([poly])

    assert is_linear_in_variable(sub_map, 0, x) is False


def test_is_linear_returns_true_for_linear_with_coefficient() -> None:
    # y*x + z  — linear in x (coefficient y is a free variable)
    poly = Poly(y * x + z, x, y, z)
    sub_map = make_sub_map([poly])

    assert is_linear_in_variable(sub_map, 0, x) is True


def test_is_linear_returns_false_for_product_of_vars() -> None:
    # x*y + x^2  — quadratic in x
    poly = Poly(x * y + x**2, x, y)
    sub_map = make_sub_map([poly])

    assert is_linear_in_variable(sub_map, 0, x) is False


def test_is_linear_selects_correct_poly_by_index() -> None:
    # polys[0]: x^2 + y  (non-linear in x)
    # polys[1]: x + y    (linear in x)
    poly0 = Poly(x**2 + y, x, y)
    poly1 = Poly(x + y, x, y)
    sub_map = make_sub_map([poly0, poly1])

    assert is_linear_in_variable(sub_map, 0, x) is False
    assert is_linear_in_variable(sub_map, 1, x) is True


def test_is_linear_with_substituted_variable_in_poly() -> None:
    # substitution_map: z -> 2*w  (z is a substituted variable)
    # poly: x + z  — x appears linearly; z's value is resolved numerically
    poly = Poly(x + z, x, z)
    sub_map = SystemSubstitutionMap(
        remaining_equations=[poly],
        substitution_map={z: 2 * w},
        free_variables={x, w},
        substituted_variables={z},
    )

    assert is_linear_in_variable(sub_map, 0, x) is True


def test_is_linear_with_chained_substitution_not_obscuring_linearity() -> None:
    # substitution_map: z -> w,  y -> z  (chain: y -> z -> w)
    # poly: x + y  — linear in x regardless of how y resolves
    poly = Poly(x + y, x, y)
    sub_map = SystemSubstitutionMap(
        remaining_equations=[poly],
        substitution_map={z: w, y: z},
        free_variables={x, w},
        substituted_variables={y, z},
    )

    assert is_linear_in_variable(sub_map, 0, x) is True


def test_is_linear_returns_false_for_cubic() -> None:
    poly = Poly(x**3 + y, x, y)
    sub_map = make_sub_map([poly])

    assert is_linear_in_variable(sub_map, 0, x) is False


# ── add_linear_substitution ───────────────────────────────────────────────────


def test_add_linear_substitution_removes_poly_from_remaining() -> None:
    # x + y = 0  =>  x -> -y
    poly = Poly(x + y, x, y)
    sub_map = make_sub_map([poly])

    result = add_linear_substitution(sub_map, 0, x)

    assert len(result.remaining_equations) == 0


def test_add_linear_substitution_solves_correctly() -> None:
    # x + y = 0  =>  x -> -y
    poly = Poly(x + y, x, y)
    sub_map = make_sub_map([poly])

    result = add_linear_substitution(sub_map, 0, x)

    assert result.substitution_map[x] == -y


def test_add_linear_substitution_updates_variable_sets() -> None:
    poly = Poly(x + y, x, y)
    sub_map = make_sub_map([poly])

    result = add_linear_substitution(sub_map, 0, x)

    assert x not in result.free_variables
    assert x in result.substituted_variables


def test_add_linear_substitution_preserves_other_equations() -> None:
    poly0 = Poly(x + y, x, y)
    poly1 = Poly(y + z, y, z)
    sub_map = make_sub_map([poly0, poly1])

    result = add_linear_substitution(sub_map, 0, x)

    assert len(result.remaining_equations) == 1
    assert result.remaining_equations[0] == poly1


def test_add_linear_substitution_removes_poly_by_index() -> None:
    poly0 = Poly(x**2 + y, x, y)
    poly1 = Poly(x + z, x, z)
    sub_map = make_sub_map([poly0, poly1])

    result = add_linear_substitution(sub_map, 1, x)

    assert len(result.remaining_equations) == 1
    assert result.remaining_equations[0] == poly0


def test_add_linear_substitution_with_coefficient() -> None:
    # 2*x + y = 0  =>  x -> -y/2
    poly = Poly(2 * x + y, x, y)
    sub_map = make_sub_map([poly])

    result = add_linear_substitution(sub_map, 0, x)

    assert result.substitution_map[x] == -y / 2


def test_add_linear_substitution_does_not_substitute_independent_vars() -> None:
    # substitution_map has z -> w, but z does not depend on x
    # poly: x + z = 0  =>  x -> -z  (z should NOT be expanded to w)
    poly = Poly(x + z, x, z)
    sub_map = SystemSubstitutionMap(
        remaining_equations=[poly],
        substitution_map={z: w},
        free_variables={x, w},
        substituted_variables={z},
    )

    result = add_linear_substitution(sub_map, 0, x)

    # x -> -z, not -w, because z is independent of x
    assert result.substitution_map[x] == -z


def test_add_linear_substitution_expands_var_dependent_substitutions() -> None:
    # substitution_map: y -> x + 1  (y depends on x)
    # poly: y + z = 0  =>  must substitute y first: (x + 1) + z = 0  =>  x -> -(z + 1)
    poly = Poly(y + z, y, z)
    sub_map = SystemSubstitutionMap(
        remaining_equations=[poly],
        substitution_map={y: x + 1},
        free_variables={x, z},
        substituted_variables={y},
    )

    result = add_linear_substitution(sub_map, 0, x)

    assert result.substitution_map[x] == -(z + 1)


def test_add_linear_substitution_chained_var_dependent_substitutions() -> None:
    # chain: y -> x + 1, z -> y + 2 = (x + 1) + 2 = x + 3
    # poly: z + w = 0  =>  substitute z (which depends on x via y):
    #   (x + 3) + w = 0  =>  x -> -(w + 3)
    poly = Poly(z + w, z, w)
    sub_map = SystemSubstitutionMap(
        remaining_equations=[poly],
        substitution_map={y: x + 1, z: y + 2},
        free_variables={x, w},
        substituted_variables={y, z},
    )

    result = add_linear_substitution(sub_map, 0, x)

    assert result.substitution_map[x] == -(w + 3)


def test_add_linear_substitution_preserves_existing_substitution_map() -> None:
    poly = Poly(x + y, x, y)
    existing = {z: w + 1}
    sub_map = SystemSubstitutionMap(
        remaining_equations=[poly],
        substitution_map=existing,
        free_variables={x, y},
        substituted_variables={z},
    )

    result = add_linear_substitution(sub_map, 0, x)

    assert result.substitution_map[z] == w + 1


def test_add_linear_substitution_does_not_mutate_original() -> None:
    poly = Poly(x + y, x, y)
    sub_map = make_sub_map([poly])

    add_linear_substitution(sub_map, 0, x)

    assert len(sub_map.remaining_equations) == 1
    assert x not in sub_map.substitution_map
