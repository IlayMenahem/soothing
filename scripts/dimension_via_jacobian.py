"""
Compute the dimension of the variety V(p_1, ..., p_m) at a solution point.

Following the procedure in auxilary/main.tex (Section "Measuring Capability"):

    dim V = #variables - rank(J_p),

where J_p is the Jacobian of the system at a point p in V.

By default the witness point is found by Z3. Alternatively, ``--input-point``
loads an externally-supplied point (JSON dict of ``{var_name: value}`` where
each value is a SymPy-parseable string, e.g. ``"1/2"`` or ``"sqrt(2)"``).
This makes the probe reusable for points produced by other tools (the MILP
triangular set, HomotopyContinuation.jl witness sets, etc.).

``--output-json`` emits a single JSON line on stdout in the
``BaselineResult`` shape used elsewhere in the benchmark, so the Rust
orchestrator can consume it directly.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
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
    point: dict[sp.Symbol, sp.Basic],
) -> int:
    jacobian_rows = [
        [sp.diff(poly, sym) for sym in symbols_in_order] for poly in polynomials
    ]
    jacobian = sp.Matrix(jacobian_rows).subs(point)
    return jacobian.rank()


def _point_from_json(
    point_path: Path, symbols_in_order: list[sp.Symbol]
) -> dict[sp.Symbol, sp.Basic]:
    """Load ``{var_name: value}`` from JSON and convert values to SymPy."""
    raw = json.loads(point_path.read_text(encoding="utf-8"))
    by_name = {str(sym): sym for sym in symbols_in_order}
    point: dict[sp.Symbol, sp.Basic] = {}
    for name, value in raw.items():
        sym = by_name.get(name)
        if sym is None:
            # Tolerate extra entries (e.g. from a witness set with parameter values).
            continue
        point[sym] = sp.sympify(value)
    missing = [str(s) for s in symbols_in_order if s not in point]
    if missing:
        raise ValueError(f"point JSON is missing variables: {missing}")
    return point


def compute_dimension_z3(
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


def compute_dimension_at_supplied_point(
    system_text: str, point_path: Path
) -> tuple[int, int, int]:
    polynomials, symbols = mathematica_to_sympy(system_text)
    symbols_in_order = sorted(symbols, key=str)
    point = _point_from_json(point_path, symbols_in_order)
    rank = jacobian_rank_at_point(polynomials, symbols_in_order, point)
    n_vars = len(symbols_in_order)
    dimension = n_vars - rank
    return n_vars, rank, dimension


# ---------- JSON result shape (matches scripts/baselines/_common.py) ----------


@dataclass
class DimensionResult:
    baseline: str
    status: str
    elapsed_secs: float = 0.0
    found_dim: int | None = None
    rank: int | None = None
    n_vars: int | None = None
    source: str = ""
    error: str | None = None

    def emit(self) -> None:
        sys.stdout.write(json.dumps({k: v for k, v in asdict(self).items() if v is not None}) + "\n")
        sys.stdout.flush()


def _emit_json_result(
    *,
    elapsed: float,
    source: str,
    n_vars: int | None,
    rank: int | None,
    dimension: int | None,
    status: str,
    error: str | None,
) -> None:
    DimensionResult(
        baseline="jacobian-dim",
        status=status,
        elapsed_secs=elapsed,
        found_dim=dimension,
        rank=rank,
        n_vars=n_vars,
        source=source,
        error=error,
    ).emit()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute the dimension of a variety at a witness point via the "
            "Jacobian rank trick. Point source: Z3 (default), or an external "
            "JSON file supplied with --input-point."
        )
    )
    parser.add_argument("system_path", type=Path)
    parser.add_argument(
        "--input-point",
        type=Path,
        default=None,
        help="JSON file with {var_name: value} for an externally-found point.",
    )
    parser.add_argument("--max-solutions", type=int, default=1)
    parser.add_argument("--timeout-ms", type=int, default=120_000)
    parser.add_argument(
        "--output-json",
        action="store_true",
        help="Emit a single JSON line on stdout instead of human-readable text.",
    )
    return parser.parse_args()


def _run_human(args: argparse.Namespace) -> int:
    system_text = args.system_path.read_text(encoding="utf-8")
    if args.input_point is not None:
        n_vars, rank, dimension = compute_dimension_at_supplied_point(
            system_text, args.input_point
        )
        print(f"point source: {args.input_point}")
        print(f"variables: {n_vars}")
        print(f"jacobian rank at witness: {rank}")
        print(f"dimension of variety at witness: {dimension}")
        return 0

    solve_result, rank, dimension = compute_dimension_z3(
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


def _run_json(args: argparse.Namespace) -> int:
    system_text = args.system_path.read_text(encoding="utf-8")
    start = time.perf_counter()
    try:
        if args.input_point is not None:
            n_vars, rank, dimension = compute_dimension_at_supplied_point(
                system_text, args.input_point
            )
            _emit_json_result(
                elapsed=time.perf_counter() - start,
                source=f"file:{args.input_point.name}",
                n_vars=n_vars,
                rank=rank,
                dimension=dimension,
                status="ok",
                error=None,
            )
            return 0

        solve_result, rank, dimension = compute_dimension_z3(
            system_text,
            max_solutions=args.max_solutions,
            timeout_ms=args.timeout_ms,
        )
        if rank is None or dimension is None:
            _emit_json_result(
                elapsed=time.perf_counter() - start,
                source="z3",
                n_vars=None,
                rank=None,
                dimension=None,
                status="no_witness",
                error=str(solve_result.status),
            )
            return 0
        _, symbols = mathematica_to_sympy(system_text)
        _emit_json_result(
            elapsed=time.perf_counter() - start,
            source="z3",
            n_vars=len(symbols),
            rank=rank,
            dimension=dimension,
            status="ok",
            error=None,
        )
        return 0
    except Exception as exc:  # noqa: BLE001 — any failure becomes a structured error result.
        _emit_json_result(
            elapsed=time.perf_counter() - start,
            source="z3" if args.input_point is None else f"file:{args.input_point.name}",
            n_vars=None,
            rank=None,
            dimension=None,
            status="error",
            error=str(exc),
        )
        return 0


def main() -> int:
    args = parse_args()
    return _run_json(args) if args.output_json else _run_human(args)


if __name__ == "__main__":
    raise SystemExit(main())
