from sympy import Symbol, pi

from soothing.parse_mathematica import mathematica_to_sympy


# ── Rule parsing: LHS variable is retained ────────────────────────────────────


def test_arrow_rule_lhs_symbol_present() -> None:
    """-> rule must yield a polynomial that contains the LHS variable."""
    polys, symbols = mathematica_to_sympy(r"alpha1LambdaOverKappa -> 3 e / (256 \[Pi]^2)")
    lhs_sym = Symbol("alpha1LambdaOverKappa")
    all_free = {s for p in polys for s in p.free_symbols}
    assert lhs_sym in all_free, "LHS variable missing from parsed polynomial"


def test_arrow_rule_lhs_coefficient_is_nonzero() -> None:
    """-> rule produces (lhs - rhs) * denominator = 0 after clearing fractions.

    The LHS variable's coefficient is the cleared denominator (256*pi**2 here),
    not 1.  We only assert it is non-zero, confirming the LHS is present linearly.
    """
    polys, _ = mathematica_to_sympy(r"alpha1LambdaOverKappa -> 3 e / (256 \[Pi]^2)")
    assert len(polys) == 1
    poly = polys[0]
    lhs_sym = Symbol("alpha1LambdaOverKappa")
    expr = poly.as_expr()
    coeff = expr.coeff(lhs_sym)
    assert coeff != 0, "Expected non-zero coefficient for LHS variable after denominator clearing"


def test_arrow_rule_multiple_equations_all_lhs_present() -> None:
    """Multiple -> rules must each contribute their LHS variable."""
    system = r"alpha1 -> e, alpha2 -> e^2"
    polys, all_symbols = mathematica_to_sympy(system)
    assert Symbol("alpha1") in all_symbols
    assert Symbol("alpha2") in all_symbols
    assert len(polys) == 2


# ── alpha5Dl multiline expression ─────────────────────────────────────────────


_ALPHA5DL_EXPR = r"""alpha5Dl -> -1/(512 \[Pi]^2)
    3 (16 DhDphp2 DpDph1 - 16 DhDphp3 DpDph1 + 16 DhDphp5 DpDph1 +
      3 DpDph1^3 - 16 DpDph1 DpDphh3 - 4 DpDph1^2 DppDh1 +
      32 DpDphh3 DppDh1 - 4 DpDph1^2 DppDh2 +
      32 DpDphh3 DppDh2) e^3 (-1 + xr)"""


def test_alpha5dl_lhs_symbol_present() -> None:
    """alpha5Dl multiline rule must retain the LHS variable."""
    polys, all_symbols = mathematica_to_sympy(_ALPHA5DL_EXPR)
    assert Symbol("alpha5Dl") in all_symbols


def test_alpha5dl_polynomial_is_nonzero() -> None:
    """alpha5Dl parsed polynomial must be non-trivially non-zero."""
    polys, _ = mathematica_to_sympy(_ALPHA5DL_EXPR)
    assert len(polys) == 1
    assert polys[0].as_expr() != 0
