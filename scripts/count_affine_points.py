#!/usr/bin/env python3
"""Exact number of solutions of a system whose equations are products of affine factors.

Every solution satisfies one affine factor from each equation, so the solution
set is the union, over all factor choices, of the solutions of a square affine
system.  Each square system is solved over Q; a consistent singular choice is
counted separately, since then the solution set is not finite.

Used for the box-QP and Lotka--Volterra systems of benchmark_applications.py:

    python3 scripts/benchmark_applications.py --report-only --output-dir /tmp/apps
    python3 scripts/count_affine_points.py /tmp/apps/box_qp_n*.txt /tmp/apps/lotka_volterra_n*.txt
"""

from __future__ import annotations

import itertools
import re
import sys
from fractions import Fraction
from pathlib import Path

import sympy

AffineRow = tuple[tuple[Fraction, ...], Fraction]


def read_polynomials(path: Path) -> list[sympy.Expr]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if "->" in line]
    return [sympy.sympify(re.sub(r"^\s*0\s*->\s*", "", line).rstrip(",")) for line in lines]


def affine_factors(polynomial: sympy.Expr, symbols: list[sympy.Symbol]) -> list[AffineRow]:
    """Distinct affine factors a.x + b of ``polynomial`` as (a, -b)."""
    _, factors = sympy.factor_list(polynomial, *symbols)
    rows: list[AffineRow] = []
    for factor, _ in factors:
        poly = sympy.Poly(factor, *symbols)
        if poly.total_degree() != 1:
            raise SystemExit(f"non-affine factor {factor}")
        coefficients = tuple(Fraction(str(poly.coeff_monomial(symbol))) for symbol in symbols)
        rows.append((coefficients, -Fraction(str(poly.coeff_monomial(1)))))
    return rows


def solve(rows: tuple[AffineRow, ...]) -> tuple[Fraction, ...] | str | None:
    """Unique solution, "positive-dimensional", or None when inconsistent."""
    size = len(rows[0][0])
    matrix = [[*coefficients, rhs] for coefficients, rhs in rows]
    pivots: list[int] = []
    for column in range(size):
        pivot = next((r for r in range(len(pivots), len(matrix)) if matrix[r][column]), None)
        if pivot is None:
            continue
        top = len(pivots)
        matrix[top], matrix[pivot] = matrix[pivot], matrix[top]
        lead = matrix[top][column]
        matrix[top] = [value / lead for value in matrix[top]]
        matrix = [
            row if r == top or not row[column]
            else [a - row[column] * b for a, b in zip(row, matrix[top], strict=True)]
            for r, row in enumerate(matrix)
        ]
        pivots.append(column)
    if any(all(value == 0 for value in row[:-1]) and row[-1] != 0 for row in matrix):
        return None
    if len(pivots) < size:
        return "positive-dimensional"
    return tuple(matrix[index][-1] for index in range(size))


def count_points(path: Path) -> str:
    polynomials = read_polynomials(path)
    symbols = sorted(
        set().union(*(polynomial.free_symbols for polynomial in polynomials)),
        key=lambda symbol: int(str(symbol)[1:]),
    )
    choices = [affine_factors(polynomial, symbols) for polynomial in polynomials]
    outcomes = [solve(rows) for rows in itertools.product(*choices)]
    points = {outcome for outcome in outcomes if isinstance(outcome, tuple)}
    positive = sum(outcome == "positive-dimensional" for outcome in outcomes)
    return (
        f"{path.stem}: choices={len(outcomes)} distinct_points={len(points)} "
        f"positive_dim_choices={positive}"
    )


def main() -> None:
    for argument in sys.argv[1:]:
        print(count_points(Path(argument)), flush=True)


if __name__ == "__main__":
    main()
