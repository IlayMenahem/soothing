from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from typing import Sequence

import sympy as sp
import z3  # type: ignore[import-untyped]

from soothing.parse_mathematica import mathematica_to_sympy


_NUMERIC_CONSTANT_PRECISION = 80


@dataclass(frozen=True)
class SystemConstraints:
    variables: dict[str, z3.ArithRef]
    constraints: list[z3.BoolRef]


@dataclass(frozen=True)
class SolveResult:
    status: z3.CheckSatResult
    solutions: list[dict[str, z3.ExprRef]]
    reached_limit: bool
    reason_unknown: str | None = None


def _multiply_z3(terms: Sequence[z3.ArithRef]) -> z3.ArithRef:
    return reduce(lambda left, right: left * right, terms, z3.RealVal(1))


def _pow_z3(base: z3.ArithRef, exponent: int) -> z3.ArithRef:
    if exponent < 0:
        raise ValueError("Z3 conversion supports only non-negative integer powers")
    return _multiply_z3([base] * exponent)


def sympy_to_z3(expr: sp.Basic, variables: dict[str, z3.ArithRef]) -> z3.ArithRef:
    if expr.is_Symbol:
        return variables[str(expr)]

    if expr in {sp.pi, sp.E}:
        return z3.RealVal(str(sp.N(expr, _NUMERIC_CONSTANT_PRECISION)))

    if expr.is_Integer:
        return z3.RealVal(str(expr))

    if expr.is_Rational:
        return z3.RealVal(str(expr))

    if expr.is_Add:
        return sum((sympy_to_z3(arg, variables) for arg in expr.args), z3.RealVal(0))

    if expr.is_Mul:
        return _multiply_z3([sympy_to_z3(arg, variables) for arg in expr.args])

    if expr.is_Pow:
        base, exponent = expr.args
        if not exponent.is_Integer:
            raise ValueError("Z3 conversion supports only non-negative integer powers")
        return _pow_z3(sympy_to_z3(base, variables), int(str(exponent)))

    raise ValueError(f"Unsupported SymPy expression for Z3 conversion: {expr!s}")


def constraints_from_system_text(system_text: str) -> SystemConstraints:
    polynomials, symbols = mathematica_to_sympy(system_text)
    variables = {str(symbol): z3.Real(str(symbol)) for symbol in sorted(symbols, key=str)}
    constraints = [
        sympy_to_z3(polynomial, variables) == z3.RealVal(0)
        for polynomial in polynomials
    ]
    return SystemConstraints(variables=variables, constraints=constraints)


def load_system_constraints(path_or_text: str | Path) -> SystemConstraints:
    path = Path(path_or_text)
    if path.exists():
        return constraints_from_system_text(path.read_text(encoding="utf-8"))
    return constraints_from_system_text(str(path_or_text))


def solve_system_text(
    system_text: str,
    *,
    max_solutions: int = 10,
    timeout_ms: int | None = 30_000,
) -> SolveResult:
    if max_solutions < 1:
        raise ValueError("max_solutions must be at least 1")

    system = constraints_from_system_text(system_text)
    solver = z3.Solver()
    if timeout_ms is not None:
        solver.set(timeout=timeout_ms)
    solver.add(system.constraints)

    solutions: list[dict[str, z3.ExprRef]] = []
    ordered_variables = sorted(system.variables.items())

    while len(solutions) < max_solutions:
        status = solver.check()
        if status == z3.unsat:
            return SolveResult(status=status, solutions=solutions, reached_limit=False)
        if status == z3.unknown:
            return SolveResult(
                status=status,
                solutions=solutions,
                reached_limit=False,
                reason_unknown=solver.reason_unknown(),
            )

        model = solver.model()
        solution = {
            name: model.eval(variable, model_completion=True)
            for name, variable in ordered_variables
        }
        solutions.append(solution)

        if not ordered_variables:
            return SolveResult(status=z3.unsat, solutions=solutions, reached_limit=False)

        solver.add(
            z3.Or(
                [
                    variable != model.eval(variable, model_completion=True)
                    for _, variable in ordered_variables
                ]
            )
        )

    return SolveResult(status=z3.sat, solutions=solutions, reached_limit=True)


def solve_system_file(
    system_path: Path,
    *,
    max_solutions: int = 10,
    timeout_ms: int | None = 30_000,
) -> SolveResult:
    return solve_system_text(
        system_path.read_text(encoding="utf-8"),
        max_solutions=max_solutions,
        timeout_ms=timeout_ms,
    )


def format_z3_value(value: z3.ExprRef, decimal_places: int) -> str:
    if isinstance(value, z3.RatNumRef):
        exact = f"{value.numerator_as_long()}/{value.denominator_as_long()}"
    else:
        exact = value.sexpr()

    as_decimal = getattr(value, "as_decimal", None)
    if as_decimal is None:
        return exact

    decimal = as_decimal(decimal_places)
    if decimal == exact:
        return exact
    return f"{exact} (~{decimal})"


def format_result(result: SolveResult, *, decimal_places: int = 20) -> str:
    lines = [f"status: {result.status}"]
    if result.reason_unknown:
        lines.append(f"reason: {result.reason_unknown}")

    for index, solution in enumerate(result.solutions, start=1):
        lines.append(f"solution {index}:")
        for name, value in sorted(solution.items()):
            lines.append(f"  {name} = {format_z3_value(value, decimal_places)}")

    if result.reached_limit:
        lines.append(f"solution limit reached: {len(result.solutions)}")

    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Solve a benchmark polynomial system with Z3 over the reals."
    )
    parser.add_argument("system_path", type=Path)
    parser.add_argument("--max-solutions", type=int, default=10)
    parser.add_argument("--timeout-ms", type=int, default=30_000)
    parser.add_argument("--decimal-places", type=int, default=20)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = solve_system_file(
        args.system_path,
        max_solutions=args.max_solutions,
        timeout_ms=args.timeout_ms,
    )
    print(format_result(result, decimal_places=args.decimal_places))
    return 0 if result.status != z3.unknown else 2
