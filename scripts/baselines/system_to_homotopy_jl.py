"""Run HomotopyContinuation.jl as a numerical-NAG baseline.

This is the only baseline in the suite that operates numerically rather than
symbolically. It computes a numerical irreducible decomposition and reports
the number of components and the maximum-dimensional component.

The Julia subprocess incurs ~1–3 seconds of startup; we report only the
algorithm-internal time in ``elapsed_secs`` (the orchestrator separately
measures wall time for its kill watchdog).
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

import sympy as sp

from _common import BaselineResult, load_system, require_tool, run_subprocess, tail

_BASELINE_LABEL = "homotopy-jl"

_JULIA_BUILTIN_COLLISIONS = {"pi", "e", "E", "I", "Inf", "NaN", "im"}


def _safe_name(s: str) -> str:
    return f"v_{s}" if s in _JULIA_BUILTIN_COLLISIONS else s


def _poly_to_julia(poly: sp.Basic, rename: dict[str, str]) -> str:
    text = sp.sstr(sp.expand(poly))
    text = text.replace("**", "^")
    # rename in a single pass with regex to avoid double-substitution
    if rename:
        pattern = re.compile(r"\b(" + "|".join(re.escape(k) for k in rename) + r")\b")
        text = pattern.sub(lambda m: rename[m.group(1)], text)
    return text


def _julia_program(polys: list[sp.Basic], symbols: list[sp.Symbol]) -> str:
    rename = {str(s): _safe_name(str(s)) for s in symbols if str(s) in _JULIA_BUILTIN_COLLISIONS}
    var_names = [_safe_name(str(s)) for s in symbols]
    var_decl = "@var " + " ".join(var_names)
    var_list = ", ".join(var_names)
    polys_jl = ",\n  ".join(_poly_to_julia(p, rename) for p in polys)
    return (
        'using HomotopyContinuation\n'
        'using Random\n'
        'Random.seed!(42)\n'
        f'{var_decl}\n'
        f'F = System([\n  {polys_jl}\n], variables=[{var_list}])\n'
        't0 = time_ns()\n'
        'nid = numerical_irreducible_decomposition(F)\n'
        'dt_ms = (time_ns() - t0) / 1_000_000\n'
        'wsets = nid.Witness_Sets\n'
        'num_components = sum(length(v) for v in values(wsets); init=0)\n'
        'max_dim = isempty(wsets) ? -1 : maximum(keys(wsets))\n'
        'num_solutions = haskey(wsets, 0) ? sum(length(solutions(w)) for w in wsets[0]; init=0) : 0\n'
        'println("SOLVE_MS ", round(Int, dt_ms))\n'
        'println("NUM_COMPONENTS ", num_components)\n'
        'println("MAX_DIM ", max_dim)\n'
        'println("NUM_SOLUTIONS ", num_solutions)\n'
        'println("DONE")\n'
    )


def _parse_julia_output(stdout: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for line in stdout.splitlines():
        m = re.match(
            r"^(SOLVE_MS|NUM_COMPONENTS|MAX_DIM|NUM_SOLUTIONS)\s+(-?\d+)\s*$",
            line.strip(),
        )
        if m:
            out[m.group(1)] = int(m.group(2))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("system_path", type=Path)
    parser.add_argument("--timeout-secs", type=float, default=300.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tool = require_tool("julia", _BASELINE_LABEL)
    if tool is None:
        return 0

    try:
        polys, symbols = load_system(args.system_path)
    except Exception as exc:  # noqa: BLE001
        BaselineResult(baseline=_BASELINE_LABEL, status="parse_error", error=str(exc)).emit()
        return 0

    if not polys:
        BaselineResult(
            baseline=_BASELINE_LABEL, status="ok", elapsed_secs=0.0, num_components=0
        ).emit()
        return 0

    program = _julia_program(polys, symbols)

    try:
        rc, out, err, wall = run_subprocess(
            [tool, "--startup-file=no", "-e", program],
            timeout=args.timeout_secs,
        )
    except subprocess.TimeoutExpired:
        BaselineResult(
            baseline=_BASELINE_LABEL, status="timeout", elapsed_secs=args.timeout_secs
        ).emit()
        return 0

    parsed = _parse_julia_output(out)
    if "SOLVE_MS" not in parsed or "DONE" not in out:
        BaselineResult(
            baseline=_BASELINE_LABEL,
            status="error",
            elapsed_secs=wall,
            error=f"julia rc={rc}",
            tool_stdout_tail=tail(out + "\n--- stderr ---\n" + err),
        ).emit()
        return 0

    BaselineResult(
        baseline=_BASELINE_LABEL,
        status="ok",
        elapsed_secs=parsed["SOLVE_MS"] / 1000.0,
        num_components=parsed.get("NUM_COMPONENTS"),
        max_dim=parsed.get("MAX_DIM"),
        num_solutions=parsed.get("NUM_SOLUTIONS"),
    ).emit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
