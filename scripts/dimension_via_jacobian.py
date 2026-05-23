"""
Compute the dimension of the variety V(p_1, ..., p_m) at a solution point.

Following the procedure in auxilary/main.tex (Section "Measuring Capability"):

    dim V = #variables - rank(J_p),

where J_p is the Jacobian of the system at a point p in V.

The point p is obtained by solving the polynomial system with Z3 (over the
reals) via `scripts/z3_solver.py`.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
from pathlib import Path

import sympy as sp
import z3  # type: ignore[import-untyped]

from parse_mathematica import mathematica_to_sympy
from z3_solver import SolveResult, solve_system_text


def z3_value_to_sympy(value: z3.ExprRef) -> sp.Rational:
    if isinstance(value, z3.RatNumRef):
        return sp.Rational(value.numerator_as_long(), value.denominator_as_long())
    if isinstance(value, z3.AlgebraicNumRef):
        # Fall back to a high-precision rational approximation.
        decimal = value.as_decimal(40)
        return sp.Rational(Fraction(decimal.rstrip("?")))
    raise TypeError(f"Unsupported Z3 value type: {type(value).__name__}")


def jacobian_rank_at_point(
    polynomials: list[sp.Basic],
    symbols_in_order: list[sp.Symbol],
    point: dict[sp.Symbol, sp.Rational],
) -> int:
    jacobian_rows = [
        [sp.diff(poly, sym) for sym in symbols_in_order] for poly in polynomials
    ]
    jacobian = sp.Matrix(jacobian_rows).subs(point)
    return jacobian.rank()


def compute_dimension(
    system_text: str,
    *,
    max_solutions: int,
    timeout_ms: int,
) -> tuple[SolveResult, int | None, int | None]:
    polynomials, symbols = mathematica_to_sympy(system_text)
    symbols_in_order = sorted(symbols, key=str)

    solve_result = solve_system_text(
        system_text, max_solutions=max_solutions, timeout_ms=timeout_ms
    )

    if not solve_result.solutions:
        return solve_result, None, None

    first_solution = solve_result.solutions[0]
    point = {
        sym: z3_value_to_sympy(first_solution[str(sym)]) for sym in symbols_in_order
    }
    rank = jacobian_rank_at_point(polynomials, symbols_in_order, point)
    dimension = len(symbols_in_order) - rank
    return solve_result, rank, dimension


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Solve a benchmark polynomial system with Z3 over the reals and "
            "report the dimension of the variety at the witness point via "
            "the Jacobian rank trick."
        )
    )
    parser.add_argument("system_path", type=Path)
    parser.add_argument("--max-solutions", type=int, default=1)
    parser.add_argument("--timeout-ms", type=int, default=120_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    system_text = args.system_path.read_text(encoding="utf-8")

    solve_result, rank, dimension = compute_dimension(
        system_text,
        max_solutions=args.max_solutions,
        timeout_ms=args.timeout_ms,
    )

    print(f"z3 status: {solve_result.status}")
    if solve_result.reason_unknown:
        print(f"reason unknown: {solve_result.reason_unknown}")
    if solve_result.reached_limit:
        print(f"solution limit reached: {len(solve_result.solutions)}")
    print(f"solutions found: {len(solve_result.solutions)}")

    if rank is None or dimension is None:
        print("no witness point — cannot compute Jacobian rank")
        return 1

    polynomials, symbols = mathematica_to_sympy(system_text)
    print(f"polynomials: {len(polynomials)}")
    print(f"variables: {len(symbols)}")
    print(f"jacobian rank at witness: {rank}")
    print(f"dimension of variety at witness: {dimension}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
