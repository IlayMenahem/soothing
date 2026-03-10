from sympy import Poly, Rational, Symbol, simplify, symbols

from soothing.operations import (
    SystemSubstitutionMap,
    add_linear_substitution,
    best_linear_substitutions_order,
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


def test_add_linear_substitution_substitutes_independent_vars() -> None:
    # substitution_map has z -> w, but z does not depend on x
    # poly: x + z = 0  =>  after substituting z->w: x + w = 0  =>  x -> -w
    # We substitute var-independent symbols for consistency with is_linear_in_variable.
    poly = Poly(x + z, x, z)
    sub_map = SystemSubstitutionMap(
        remaining_equations=[poly],
        substitution_map={z: w},
        free_variables={x, w},
        substituted_variables={z},
    )

    result = add_linear_substitution(sub_map, 0, x)

    assert result.substitution_map[x] == -w


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


def test_is_linear_with_high_degree_other_vars() -> None:
    # y^10 * x + z^5  — linear in x despite high degrees of y and z
    poly = Poly(y**10 * x + z**5, x, y, z)
    sub_map = make_sub_map([poly])

    assert is_linear_in_variable(sub_map, 0, x) is True


def test_is_linear_constant_poly() -> None:
    # 5  — x does not appear
    poly = Poly(5, x)
    sub_map = make_sub_map([poly])

    assert is_linear_in_variable(sub_map, 0, x) is False


def test_add_linear_substitution_complex_coefficients() -> None:
    # (y^2 + z)*x + (w - 1) = 0  =>  x -> -(w - 1)/(y^2 + z)
    poly = Poly((y**2 + z) * x + (w - 1), x, y, z, w)
    sub_map = make_sub_map([poly])

    result = add_linear_substitution(sub_map, 0, x)

    assert result.substitution_map[x] == -(w - 1) / (y**2 + z)


def test_add_linear_substitution_multiple_dependencies() -> None:
    # substitution_map: y -> x + 1, z -> 2*x
    # poly: y + z + w = 0  =>  substitute y and z:
    #   (x + 1) + 2*x + w = 0  =>  3*x + w + 1 = 0  =>  x -> -(w + 1)/3
    poly = Poly(y + z + w, y, z, w)
    sub_map = SystemSubstitutionMap(
        remaining_equations=[poly],
        substitution_map={y: x + 1, z: 2 * x},
        free_variables={x, w},
        substituted_variables={y, z},
    )

    result = add_linear_substitution(sub_map, 0, x)

    assert result.substitution_map[x] == -(w + 1) / 3


# ── best_linear_substitutions_order ──────────────────────────────────────────


def test_best_linear_substitutions_order_no_linear_equations() -> None:
    # x^2 + y^2 - 1 = 0  — no variable appears linearly, nothing to substitute
    poly = Poly(x**2 + y**2 - 1, x, y)

    result = best_linear_substitutions_order([poly])

    assert len(result.remaining_equations) == 1
    assert result.substitution_map == {}
    assert result.substituted_variables == set()


def test_best_linear_substitutions_order_single_linear_equation() -> None:
    # x + y = 0  =>  one variable eliminated, no remaining equations
    poly = Poly(x + y, x, y)

    result = best_linear_substitutions_order([poly])

    assert len(result.remaining_equations) == 0
    assert len(result.substituted_variables) == 1


def test_best_linear_substitutions_order_fully_linear_system() -> None:
    # x + y = 0
    # y + z = 0
    # Both equations are linear; a full chain of substitutions eliminates all of them.
    poly0 = Poly(x + y, x, y)
    poly1 = Poly(y + z, y, z)

    result = best_linear_substitutions_order([poly0, poly1])

    assert len(result.remaining_equations) == 0
    assert len(result.substituted_variables) == 2


def test_best_linear_substitutions_order_partial_system() -> None:
    # x + y = 0          — linear, can be eliminated
    # x^2 + z^2 - 1 = 0  — non-linear, must remain
    poly0 = Poly(x + y, x, y)
    poly1 = Poly(x**2 + z**2 - 1, x, z)

    result = best_linear_substitutions_order([poly0, poly1])

    assert len(result.remaining_equations) == 1
    assert len(result.substituted_variables) == 1


def test_best_linear_substitutions_order_three_variable_chain() -> None:
    # x + y = 0   =>  x -> -y
    # y + z = 0   =>  y -> -z
    # z + w = 0   =>  z -> -w
    # All three equations are eliminated by chaining substitutions.
    poly0 = Poly(x + y, x, y)
    poly1 = Poly(y + z, y, z)
    poly2 = Poly(z + w, z, w)

    result = best_linear_substitutions_order([poly0, poly1, poly2])

    assert len(result.remaining_equations) == 0
    assert len(result.substituted_variables) == 3


def test_best_linear_substitutions_order_substitution_map_solves_equations() -> None:
    # x + 2*y = 0  =>  x -> -2*y
    # Verify the substitution actually satisfies the original equation.
    poly = Poly(x + 2 * y, x, y)

    result = best_linear_substitutions_order([poly])

    substituted_var = next(iter(result.substituted_variables))
    substitution_expr = result.substitution_map[substituted_var]
    residual = poly.as_expr().subs(substituted_var, substitution_expr)
    assert simplify(residual) == 0


def test_best_linear_substitutions_order_variable_sets_are_disjoint() -> None:
    # free_variables and substituted_variables must be disjoint and together
    # cover all original variables.
    poly0 = Poly(x + y, x, y)
    poly1 = Poly(y + z, y, z)

    result = best_linear_substitutions_order([poly0, poly1])

    assert result.free_variables.isdisjoint(result.substituted_variables)
    original_vars = {x, y, z}
    assert result.free_variables | result.substituted_variables == original_vars


def test_best_linear_substitutions_order_prefers_deeper_branch() -> None:
    # Two possible first moves from poly0 = a + b = 0:
    #   (a) eliminate a  =>  a -> -b.  poly1 = a*b + b = 0 becomes (-b)*b + b = -b^2 + b = 0,
    #       which is non-linear in b  =>  1 equation remains, no further move.
    #   (b) eliminate b  =>  b -> -a.  poly1 = a*b + b = 0 becomes a*(-a) + (-a) = -a^2 - a = 0,
    #       which is non-linear in a  =>  1 equation remains, no further move.
    # Both branches stall at 1 remaining equation, so the minimum achievable is 1.
    #
    # A system where branching order matters:
    #   poly0: a + b = 0              (linear in both a and b)
    #   poly1: a + c = 0              (linear in both a and c)
    # Any first substitution removes one equation and makes the remaining one still linear,
    # so the optimal result is 0 remaining equations regardless of order chosen.
    # This verifies the DFS explores branches until all equations are consumed.
    a, b_sym, c_sym = symbols("a b c")
    poly0 = Poly(a + b_sym, a, b_sym, c_sym)
    poly1 = Poly(a + c_sym, a, b_sym, c_sym)

    result = best_linear_substitutions_order([poly0, poly1])

    assert len(result.remaining_equations) == 0
    assert len(result.substituted_variables) == 2


def test_best_linear_substitutions_order_non_linear_system_unchanged_equations() -> (
    None
):
    # x^2 - 1 = 0 and y^2 - 4 = 0 — purely quadratic, no linear substitutions possible.
    poly0 = Poly(x**2 - 1, x)
    poly1 = Poly(y**2 - 4, y)

    result = best_linear_substitutions_order([poly0, poly1])

    assert len(result.remaining_equations) == 2
    assert result.substitution_map == {}


def test_best_linear_substitutions_order_returns_system_substitution_map() -> None:
    poly = Poly(x + y, x, y)

    result = best_linear_substitutions_order([poly])

    assert isinstance(result, SystemSubstitutionMap)


def test_best_linear_substitutions_order_empty_system() -> None:
    result = best_linear_substitutions_order([])

    assert result.remaining_equations == []
    assert result.substitution_map == {}
    assert result.substituted_variables == set()


# ── is_linear_in_variable: false-positive induction ──────────────────────────
#
# Each test below constructs a scenario where the underlying expression is
# *non-linear* in `var` and asserts that `is_linear_in_variable` correctly
# returns False.  The goal is to probe paths that could mistakenly return True.


def test_is_linear_false_positive_quadratic_via_direct_substitution() -> None:
    # substitution_map: y -> x**2
    # poly: y + z  — looks trivially linear in x on the raw polynomial,
    # but after back-substitution it becomes x**2 + z, which is quadratic in x.
    poly = Poly(y + z, y, z)
    sub_map = SystemSubstitutionMap(
        remaining_equations=[poly],
        substitution_map={y: x**2},
        free_variables={x, z},
        substituted_variables={y},
    )

    assert is_linear_in_variable(sub_map, 0, x) is False


def test_is_linear_false_positive_chained_quadratic_substitution() -> None:
    # substitution_map: z -> y**2,  y -> x + 1
    # chain: z = (x+1)**2 — quadratic in x
    # poly: z + w  — raw polynomial is linear in z but non-linear in x after expansion.
    poly = Poly(z + w, z, w)
    sub_map = SystemSubstitutionMap(
        remaining_equations=[poly],
        substitution_map={y: x + 1, z: y**2},
        free_variables={x, w},
        substituted_variables={y, z},
    )

    assert is_linear_in_variable(sub_map, 0, x) is False


def test_is_linear_false_positive_product_hidden_by_substitution() -> None:
    # substitution_map: y -> x * w   (bilinear; non-linear in x when w is free)
    # poly: y + z  — raw polynomial is linear in y, but x * w + z is non-linear in x
    # because the coefficient of x itself depends on w (which is free), making the
    # relationship linear.  However if the substitution were y -> x**2 the same
    # structure would be quadratic.  This variant keeps y -> x*w to confirm the
    # function correctly identifies it *as* linear (coefficient w is a free symbol).
    # The *non-linear* twin below uses y -> x**2 + x to be non-linear.
    poly = Poly(y + z, y, z)
    sub_map = SystemSubstitutionMap(
        remaining_equations=[poly],
        substitution_map={y: x**2 + x},
        free_variables={x, z},
        substituted_variables={y},
    )

    # x**2 + x + z is quadratic in x — must not be a false positive.
    assert is_linear_in_variable(sub_map, 0, x) is False


def test_is_linear_false_positive_cubic_via_substitution() -> None:
    # substitution_map: y -> x**3
    # poly: y + w  — after back-substitution: x**3 + w, cubic in x.
    poly = Poly(y + w, y, w)
    sub_map = SystemSubstitutionMap(
        remaining_equations=[poly],
        substitution_map={y: x**3},
        free_variables={x, w},
        substituted_variables={y},
    )

    assert is_linear_in_variable(sub_map, 0, x) is False


def test_is_linear_false_positive_two_hop_chain_squared() -> None:
    # substitution_map: y -> x**2 + 1,  z -> y - 1  =>  z = x**2
    # poly: z + w  — after full expansion: x**2 + w, quadratic in x.
    poly = Poly(z + w, z, w)
    sub_map = SystemSubstitutionMap(
        remaining_equations=[poly],
        substitution_map={y: x**2 + 1, z: y - 1},
        free_variables={x, w},
        substituted_variables={y, z},
    )

    assert is_linear_in_variable(sub_map, 0, x) is False


def test_is_linear_false_positive_nonlinear_coefficient_on_var() -> None:
    # poly: x**2 * y + x  — this is *not* linear in x (degree-2 term exists),
    # even though x appears with coefficient y.  No substitution map involvement.
    poly = Poly(x**2 * y + x, x, y)
    sub_map = make_sub_map([poly])

    assert is_linear_in_variable(sub_map, 0, x) is False


def test_is_linear_false_positive_var_appears_in_substituted_expr_non_linearly() -> (
    None
):
    # substitution_map: y -> x**2 * z   (quadratic in x, scaled by free var z)
    # poly: y + w  =>  x**2 * z + w  — quadratic in x.
    poly = Poly(y + w, y, w)
    sub_map = SystemSubstitutionMap(
        remaining_equations=[poly],
        substitution_map={y: x**2 * z},
        free_variables={x, z, w},
        substituted_variables={y},
    )

    assert is_linear_in_variable(sub_map, 0, x) is False


def test_is_linear_false_positive_multiple_substituted_vars_both_nonlinear() -> None:
    # substitution_map: y -> x**2,  z -> x**3
    # poly: y + z + w  =>  x**2 + x**3 + w  — non-linear in x.
    poly = Poly(y + z + w, y, z, w)
    sub_map = SystemSubstitutionMap(
        remaining_equations=[poly],
        substitution_map={y: x**2, z: x**3},
        free_variables={x, w},
        substituted_variables={y, z},
    )

    assert is_linear_in_variable(sub_map, 0, x) is False


def test_is_linear_false_positive_quadratic_masked_by_independent_substitution() -> (
    None
):
    # substitution_map: w -> 7  (independent of x),  y -> x**2  (depends on x)
    # poly: y + w  =>  after full resolution: x**2 + 7  — quadratic in x.
    # The independent substitution (w -> 7) must not distract from detecting
    # the non-linearity introduced by y -> x**2.
    poly = Poly(y + w, y, w)
    sub_map = SystemSubstitutionMap(
        remaining_equations=[poly],
        substitution_map={w: z + 1, y: x**2},
        free_variables={x, z},
        substituted_variables={y, w},
    )

    assert is_linear_in_variable(sub_map, 0, x) is False


def test_is_linear_false_positive_three_hop_nonlinear_chain() -> None:
    # chain: a -> x**2,  b -> a + 1 = x**2 + 1,  c -> b - 1 = x**2
    # poly: c + z  =>  x**2 + z  — quadratic in x despite 3-hop chain.
    a, b_sym, c_sym = symbols("a b c")
    poly = Poly(c_sym + z, c_sym, z)
    sub_map = SystemSubstitutionMap(
        remaining_equations=[poly],
        substitution_map={a: x**2, b_sym: a + 1, c_sym: b_sym - 1},
        free_variables={x, z},
        substituted_variables={a, b_sym, c_sym},
    )

    assert is_linear_in_variable(sub_map, 0, x) is False


# ── is_linear_in_variable: numerical false-positive induction ─────────────────
#
# The collinearity test has a fixed absolute tolerance of 1e-8.  Any expression
# whose quadratic (or higher) curvature produces point-to-chord deviations
# smaller than that tolerance will be misclassified as linear.
#
# The maximum deviation of eps*x^n + linear from its chord on the seeded sample
# points is eps * max_i|(xi - x0)(xi - x_last)|  (for n=2; similar for n=3).
# With the default RNG seed and range [0.5, 2.0] that maximum factor is ≈ 1.058,
# so any |eps| < ~9.45e-9 sneaks through.  All attacks below use eps = 1e-9,
# which is comfortably inside that window regardless of free-symbol scaling.
#
# Each test documents the mechanism, derives why it fools the test, and asserts
# that is_linear_in_variable incorrectly returns True.


def test_is_linear_numerical_false_positive_epsilon_quadratic_direct() -> None:
    # Mechanism: the quadratic coefficient is smaller than the collinearity
    # tolerance, so the 8 sample points fall within 1e-8 of a straight line.
    #
    # Expression: 1e-9 * x**2 + x
    # Max point-to-chord deviation ≈ 1e-9 * 1.058 ≈ 1.06e-9 < 1e-8  ✓ fools test
    poly = Poly(Rational(1, 10**9) * x**2 + x, x)
    sub_map = SystemSubstitutionMap(
        remaining_equations=[poly],
        substitution_map={},
        free_variables={x},
        substituted_variables=set(),
    )

    assert is_linear_in_variable(sub_map, 0, x) is False


def test_is_linear_numerical_false_positive_epsilon_quadratic_negative_coeff() -> None:
    # Same mechanism as above but with a negative quadratic coefficient, confirming
    # the tolerance breach is symmetric.
    #
    # Expression: -1e-9 * x**2 + x
    poly = Poly(-Rational(1, 10**9) * x**2 + x, x)
    sub_map = SystemSubstitutionMap(
        remaining_equations=[poly],
        substitution_map={},
        free_variables={x},
        substituted_variables=set(),
    )

    assert is_linear_in_variable(sub_map, 0, x) is False


def test_is_linear_numerical_false_positive_epsilon_quadratic_via_substitution() -> (
    None
):
    # Mechanism: the non-linearity is hidden in a substituted variable, so the
    # raw polynomial looks like a plain symbol.  After back-substitution the
    # expression is 1e-9*x^2 + x — identical to the direct case above.
    #
    # substitution_map: y -> 1e-9 * x**2
    # poly: y + x  =>  back-substituted: 1e-9*x^2 + x
    poly = Poly(y + x, y, x)
    sub_map = SystemSubstitutionMap(
        remaining_equations=[poly],
        substitution_map={y: Rational(1, 10**9) * x**2},
        free_variables={x},
        substituted_variables={y},
    )

    assert is_linear_in_variable(sub_map, 0, x) is False


def test_is_linear_numerical_false_positive_large_linear_term_drowns_curvature() -> (
    None
):
    # Mechanism: a dominant linear term (1e6 * x) does NOT increase the
    # point-to-chord deviation of the quadratic component.  The deviation formula
    # for eps*x^2 + L*x is eps*(xi-x0)(xi-x_last), independent of L.
    # So scaling up the linear coefficient provides no protection.
    #
    # Expression: 1e-9 * x**2 + 1e6 * x
    # Max deviation ≈ 1e-9 * 1.058 ≈ 1.06e-9 < 1e-8  ✓ fools test
    poly = Poly(Rational(1, 10**9) * x**2 + 10**6 * x, x)
    sub_map = SystemSubstitutionMap(
        remaining_equations=[poly],
        substitution_map={},
        free_variables={x},
        substituted_variables=set(),
    )

    assert is_linear_in_variable(sub_map, 0, x) is False


def test_is_linear_numerical_false_positive_chained_cancellation_produces_epsilon_quadratic() -> (
    None
):
    # Mechanism: two substituted variables whose quadratic parts nearly cancel,
    # leaving only an epsilon-quadratic residual.  SymPy's exact arithmetic
    # simplifies the sum symbolically, so the epsilon survives cleanly into
    # the numeric evaluation.
    #
    # substitution_map: y -> (1 + 1e-9) * x**2,  z -> -x**2
    # poly: y + z + x  =>  back-substituted: (1 + 1e-9)*x^2 - x^2 + x = 1e-9*x^2 + x
    poly = Poly(y + z + x, y, z, x)
    sub_map = SystemSubstitutionMap(
        remaining_equations=[poly],
        substitution_map={y: (1 + Rational(1, 10**9)) * x**2, z: -(x**2)},
        free_variables={x},
        substituted_variables={y, z},
    )

    assert is_linear_in_variable(sub_map, 0, x) is False


def test_is_linear_numerical_false_positive_two_hop_chain_epsilon_quadratic() -> None:
    # Mechanism: epsilon-quadratic produced at the end of a two-hop substitution
    # chain.  _get_independent_expr must fully expand both hops; the result is
    # still 1e-9*x^2 + x and is still below the collinearity tolerance.
    #
    # chain: a -> x**2,  b -> 1e-9 * a = 1e-9 * x**2
    # poly: b + x  =>  back-substituted: 1e-9*x^2 + x
    a, b_sym = symbols("a b")
    poly = Poly(b_sym + x, b_sym, x)
    sub_map = SystemSubstitutionMap(
        remaining_equations=[poly],
        substitution_map={a: x**2, b_sym: Rational(1, 10**9) * a},
        free_variables={x},
        substituted_variables={a, b_sym},
    )

    assert is_linear_in_variable(sub_map, 0, x) is False


def test_is_linear_numerical_false_positive_free_symbol_scales_quadratic_below_threshold() -> (
    None
):
    # Mechanism: the quadratic coefficient is the product of epsilon and a free
    # symbol fixed by the RNG.  With the default seed the free symbol y is fixed
    # to ≈ 1.661, so the effective coefficient ≈ 1.661e-9 < 1e-8.
    #
    # Expression: 1e-9 * y * x**2 + x,  y free (fixed to ≈ 1.661 by seeded RNG)
    # Effective quadratic coefficient ≈ 1.661e-9 < 9.45e-9 threshold  ✓ fools test
    poly = Poly(Rational(1, 10**9) * y * x**2 + x, x, y)
    sub_map = SystemSubstitutionMap(
        remaining_equations=[poly],
        substitution_map={},
        free_variables={x, y},
        substituted_variables=set(),
    )

    assert is_linear_in_variable(sub_map, 0, x) is False


def test_is_linear_numerical_false_positive_two_free_symbols_scale_quadratic_below_threshold() -> (
    None
):
    # Mechanism: two free symbols y and w are both fixed by the RNG before the
    # var samples are drawn.  With the default seed their values are ≈ 1.661 and
    # ≈ 1.158, giving a product ≈ 1.924.  The effective quadratic coefficient is
    # 1e-9 * 1.924 ≈ 1.924e-9 < 9.45e-9 threshold, so the test is still fooled.
    #
    # Expression: 1e-9 * y * w * x**2 + x,  y & w both free
    poly = Poly(Rational(1, 10**9) * y * w * x**2 + x, x, y, w)
    sub_map = SystemSubstitutionMap(
        remaining_equations=[poly],
        substitution_map={},
        free_variables={x, y, w},
        substituted_variables=set(),
    )

    assert is_linear_in_variable(sub_map, 0, x) is False


def test_is_linear_numerical_false_positive_epsilon_cubic() -> None:
    # Mechanism: a cubic term with a sufficiently small coefficient.
    # The maximum point-to-chord deviation of x^3 on the seeded sample points
    # is ≈ 4.213, so the threshold epsilon for a cubic is 1e-8 / 4.213 ≈ 2.37e-9.
    # Using 1e-9 keeps the coefficient well inside that window.
    #
    # Expression: 1e-9 * x**3 + x
    # Max deviation ≈ 1e-9 * 4.213 ≈ 4.21e-9 < 1e-8  ✓ fools test
    poly = Poly(Rational(1, 10**9) * x**3 + x, x)
    sub_map = SystemSubstitutionMap(
        remaining_equations=[poly],
        substitution_map={},
        free_variables={x},
        substituted_variables=set(),
    )

    assert is_linear_in_variable(sub_map, 0, x) is False


# ── Degenerate substitutions and evaluation failures ─────────────────────────


def test_add_linear_substitution_raises_when_denominator_zero() -> None:
    """add_linear_substitution raises ValueError when a=0 (degenerate)."""
    import pytest

    # y*x + y = 0 with y -> 0: independent_expr becomes 0*x + 0 = 0, a=0
    poly = Poly(y * x + y, x, y)
    sub_map = SystemSubstitutionMap(
        remaining_equations=[poly],
        substitution_map={y: y * 0},
        free_variables={x},
        substituted_variables={y},
    )

    with pytest.raises(ValueError, match="degenerate"):
        add_linear_substitution(sub_map, 0, x)


def test_best_linear_substitutions_order_skips_degenerate_and_continues() -> None:
    """best_linear_substitutions_order skips degenerate candidates and uses valid ones."""
    poly0 = Poly(x + y, x, y)
    poly1 = Poly(z * x + z, x, z)
    result = best_linear_substitutions_order([poly0, poly1], greedy=True)

    assert len(result.substitution_map) >= 1
    from sympy import nan, zoo

    for sym, expr in result.substitution_map.items():
        assert not (hasattr(expr, "has") and (expr.has(zoo) or expr.has(nan)))


def test_best_linear_substitutions_order_full_system_no_nan_zoo() -> None:
    """best_linear_substitutions_order on the Mathematica system produces no nan/zoo."""
    from soothing.parse_mathematica import mathematica_to_sympy
    from sympy import nan, zoo

    system = r"""0 -> (DpDph1 e (-3 + 2 xr))/(256 \[Pi]^2), kappah -> (DpDph1^3 e^3 (-1 + xr))/(768 \[Pi]^2)"""
    polys, _ = mathematica_to_sympy(system)
    result = best_linear_substitutions_order(polys, greedy=True)

    for sym, expr in result.substitution_map.items():
        assert not (hasattr(expr, "has") and (expr.has(zoo) or expr.has(nan)))
