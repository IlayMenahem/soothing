#!/usr/bin/env python3
"""
Generate classical polynomial benchmark systems as Mathematica-style rule lists.

Output matches Mathematica format.

References (per family):
  * **cyclic** — same as ``PolynomialTestSystems.cyclic(n)`` (Julia), Björck–Fröberg.
  * **katsura** — same as ``PolynomialTestSystems.katsura(n)`` (PoSSo / Katsura).
  * **noonburg** — Noonburg neural-network benchmark (e.g. Chen thesis; common form below).
  * **eco** — Morgan economics models, Verschelde demos ``eco5``, ``eco6``, …
  * **nash** — 2×2 game; row/column indifference (linear polynomials in ``p``, ``q``).
  * **adjacent_minors** — vanishing adjacent 2×2 minors of an ``r×c`` matrix.
  * **sparse_toric** — sparse binomial fewnomial system ``x_i^2 * x_{i+1} - 1`` (cyclic indices).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Callable, Sequence

from sympy import Expr, Integer, Rational, Symbol, expand, prod, signsimp, symbols


def sympy_to_dag_string(expr: Expr) -> str:
    """Emit syntax compatible with ``soothing-linear`` rational DAG: ``^`` for powers, ``/`` for divide."""
    e = signsimp(expand(expr))
    # str() avoids sstr's ``a + -b`` formatting; still replace ``**`` with ``^`` for the DAG parser.
    raw = str(e).replace("**", "^")
    # sstr sometimes emits SymPy's ``S(n)`` wrappers; the DAG parser expects plain integers.
    return re.sub(r"S\((-?\d+)\)", r"\1", raw)


def rules_from_equations(equations: Sequence[Expr]) -> str:
    """Comma-separated ``0 -> (rhs)`` rules; rhs is the expression that must be zero."""

    parts = []

    for eq in equations:
        rhs = sympy_to_dag_string(expand(eq))
        parts.append(f"0 -> ({rhs})")
    
    return ", ".join(parts)


def cyclic(n: int) -> list[Expr]:
    """Cyclic n-roots style sums (Julia PolynomialTestSystems)."""
    if n < 2:
        raise ValueError("cyclic requires n >= 2")

    z = [Symbol(f"z{i}") for i in range(1, n + 1)]
    eqs: list[Expr] = []
    
    for m in range(0, n - 1):
        s: Expr = Integer(0)
        for j in range(1, n + 1):
            term: Expr = Integer(1)
            for jj in range(m + 1):
                k_julia = j + jj
                idx = (k_julia - 1) % n
                term *= z[idx]
            s += term
        eqs.append(s)
    eqs.append(prod(z) - 1)
    
    return eqs


def katsura(n: int) -> list[Expr]:
    """Katsura-(n) PoSSo formulation: variables x0..xn (n+1 vars), n+1 equations."""
    if n < 1:
        raise ValueError("katsura requires n >= 1")
    
    x = [Symbol(f"x{i}") for i in range(0, n + 1)]
    eqs: list[Expr] = []
    
    for m in range(0, n):
        s: Expr = Integer(0)
        for l in range(-n, n + 1):
            if abs(m - l) <= n:
                s += x[abs(l)] * x[abs(m - l)]
        eqs.append(s - x[m])
    eqs.append(x[0] + 2 * sum(x[i] for i in range(1, n + 1)) - 1)
    
    return eqs


def noonburg(k: int) -> list[Expr]:
    """Noonburg benchmark: x_i * (sum_j x_j^2 - x_i^2 - 11/10) + 1 = 0."""
    if k < 1:
        raise ValueError("noonburg requires k >= 1")
    
    xs = [Symbol(f"x{i}") for i in range(1, k + 1)]
    sum_sq = sum(t**2 for t in xs)
    return [x_i * (sum_sq - x_i**2 - Rational(11, 10)) + 1 for x_i in xs]


def eco(n: int) -> list[Expr]:
    """Morgan economics Eco-n (Verschelde demo eco5, eco6, …). Variables x1..xn."""
    if n < 2:
        raise ValueError("eco requires n >= 2")
    
    xs = [Symbol(f"x{i}") for i in range(1, n + 1)]
    eqs: list[Expr] = []
    
    for e in range(0, n - 2):
        s = xs[e]
        for j in range(n - 2 - e):
            s += xs[j] * xs[e + j + 1]
        prod_s = s * xs[-1]
        if e == 0:
            eqs.append(Integer(n) * prod_s - 1)
        else:
            eqs.append(prod_s - Integer(e + 1))
    eqs.append(xs[n - 2] * xs[n - 1] - Integer(n - 1))
    eqs.append(sum(xs[0 : n - 1]) + Integer(1))
    
    return eqs


def nash_2x2() -> list[Expr]:
    """
    2×2 bimatrix mixed-strategy indifference (interior conditions).
    Row: R00*q + R01*(1-q) = R10*q + R11*(1-q)
    Col: C00*p + C10*(1-p) = C01*p + C11*(1-p)

    Payoffs: matching pennies (non-degenerate mixed equilibrium at p=q=1/2).
    """
    p, q = symbols("p q")
    r = ((1, -1), (-1, 1))
    c = ((-1, 1), (1, -1))
    eq_row = r[0][0] * q + r[0][1] * (1 - q) - (r[1][0] * q + r[1][1] * (1 - q))
    eq_col = c[0][0] * p + c[1][0] * (1 - p) - (c[0][1] * p + c[1][1] * (1 - p))
    
    return [expand(eq_row), expand(eq_col)]


def adjacent_minors(rows: int, cols: int) -> list[Expr]:
    """Vanishing adjacent 2×2 minors; variables m_i_j for entries."""
    if rows < 2 or cols < 2:
        raise ValueError("adjacent_minors requires rows >= 2 and cols >= 2")
    
    m = [[Symbol(f"m_{i}_{j}") for j in range(1, cols + 1)] for i in range(1, rows + 1)]
    
    eqs: list[Expr] = []
    for i in range(rows - 1):
        for j in range(cols - 1):
            a, b = m[i][j], m[i][j + 1]
            c2, d = m[i + 1][j], m[i + 1][j + 1]
            eqs.append(expand(a * d - b * c2))
    
    return eqs


def sparse_toric(n: int) -> list[Expr]:
    """Sparse binomial cycle: x_i^2 * x_{i+1} - 1 (indices mod n)."""
    if n < 2:
        raise ValueError("sparse_toric requires n >= 2")
    
    xs = [Symbol(f"t{i}") for i in range(1, n + 1)]
    eqs: list[Expr] = []
    
    for i in range(n):
        eqs.append(xs[i] ** 2 * xs[(i + 1) % n] - 1)
    
    return eqs


FAMILY_HANDLERS: dict[str, Callable[[int], list[Expr]]] = {
    "cyclic": cyclic,
    "katsura": katsura,
    "noonburg": noonburg,
    "eco": eco,
    "sparse_toric": sparse_toric,
}

ALL_FAMILIES: tuple[str, ...] = tuple(sorted((*FAMILY_HANDLERS.keys(), "nash", "adjacent_minors")))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Emit one benchmark polynomial system as a Mathematica rule list.",
    )
    
    p.add_argument(
        "--family",
        required=True,
        choices=ALL_FAMILIES,
        help="Benchmark family name",
    )
    
    p.add_argument(
        "--n",
        type=int,
        default=None,
        help="Primary size parameter (cyclic, katsura, noonburg, eco, sparse_toric). "
        "Ignored for nash.",
    )
    
    p.add_argument(
        "--rows",
        type=int,
        default=None,
        help="Matrix row count for adjacent_minors (>=2).",
    )
    
    p.add_argument(
        "--cols",
        type=int,
        default=None,
        help="Matrix column count for adjacent_minors (>=2).",
    )
    
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write to this file instead of stdout.",
    )
    
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    fam = args.family

    if fam == "nash":
        eqs = nash_2x2()
    elif fam == "adjacent_minors":
        if args.rows is None or args.cols is None:
            print("adjacent_minors requires --rows and --cols", file=sys.stderr)
            sys.exit(2)
        eqs = adjacent_minors(args.rows, args.cols)
    else:
        if args.n is None:
            print(f"{fam} requires --n", file=sys.stderr)
            sys.exit(2)
        eqs = FAMILY_HANDLERS[fam](args.n)

    text = rules_from_equations(eqs)
    if args.output is not None:
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        sys.stdout.write(text + "\n")

