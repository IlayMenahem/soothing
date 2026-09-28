"""Run Singular as a baseline on a Mathematica-style polynomial system.

Tasks:
    gb         — lex Gröbner basis via ``std(I)``; reports ``basis_len``.
    facstd     — factorizing Gröbner bases via Gräbe's ``facstd``;
                 reports ``num_components`` and ``max_dim``.
    decompose  — primary decomposition via ``primdec.lib``'s ``primdecGTZ``;
                 reports ``num_components`` and ``max_dim``.

The historical ``gb`` and ``decompose`` modes treat every free symbol as a
polynomial-ring variable.  To match the Rust loader (`mathematica.rs`), the
``facstd`` mode instead treats ``pi`` as a transcendental coefficient in
``Q(pi)`` and all other free symbols (including ``e``) as ring variables.

Emits a single JSON line on stdout in the ``BaselineResult`` shape from
``_common.py``.
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

import sympy as sp

from _common import BaselineResult, load_system, require_tool, run_subprocess, tail

_BASELINE_LABEL_BY_TASK = {
    "gb": "singular-gb-lex",
    "facstd": "singular-facstd",
    "decompose": "singular-primdecGTZ",
}


def _poly_to_singular(poly: sp.Basic) -> str:
    """Convert a SymPy polynomial expression to Singular syntax."""
    text = sp.sstr(sp.expand(poly))
    return text.replace("**", "^")


def _singular_program(polys: list[sp.Basic], symbols: list[sp.Symbol], task: str) -> str:
    # The Rust loader recognizes Mathematica's \[Pi] as the
    # coefficient ``pi`` and excludes it from the polynomial variable order.
    # Match that interpretation for facstd with the transcendental coefficient
    # field Q(pi).  The older baseline modes retain their historical behavior.
    coefficient_symbols = [s for s in symbols if task == "facstd" and str(s) == "pi"]
    polynomial_symbols = [s for s in symbols if s not in coefficient_symbols]
    coefficients = (
        f"(0,{','.join(str(s) for s in coefficient_symbols)})"
        if coefficient_symbols
        else "0"
    )
    ordering = "dp" if task == "facstd" else "lp"
    ring_vars = ",".join(str(s) for s in polynomial_symbols)
    ideal_body = ",\n    ".join(_poly_to_singular(p) for p in polys)
    # Singular's `printf` only accepts a literal format with no substitution;
    # we use `print("LABEL " + string(value))` instead, which auto-newlines.
    if task == "gb":
        body = (
            'option(redSB);\n'
            'int t0 = rtimer;\n'
            'ideal G = std(I);\n'
            'int dt = rtimer - t0;\n'
            'print("SOLVE_MS " + string(dt));\n'
            'print("BASIS_LEN " + string(size(G)));\n'
        )
    elif task == "facstd":
        body = (
            'int t0 = rtimer;\n'
            'ideal FI = interred(I);\n'
            'list F = facstd(FI);\n'
            'int dt = rtimer - t0;\n'
            'print("SOLVE_MS " + string(dt));\n'
            'print("NUM_COMPONENTS " + string(size(F)));\n'
            'int i;\n'
            'int maxd = -1;\n'
            'ideal J;\n'
            'for (i = 1; i <= size(F); i = i + 1) {\n'
            '  J = F[i];\n'
            '  int d = dim(J);\n'
            '  if (d > maxd) { maxd = d; }\n'
            '}\n'
            'print("MAX_DIM " + string(maxd));\n'
        )
    elif task == "decompose":
        body = (
            'LIB "primdec.lib";\n'
            'int t0 = rtimer;\n'
            'list pd = primdecGTZ(I);\n'
            'int dt = rtimer - t0;\n'
            'print("SOLVE_MS " + string(dt));\n'
            'print("NUM_COMPONENTS " + string(size(pd)));\n'
            'int i;\n'
            'int maxd = -1;\n'
            'for (i = 1; i <= size(pd); i = i + 1) {\n'
            '  int d = dim(std(pd[i][2]));\n'
            '  if (d > maxd) { maxd = d; }\n'
            '}\n'
            'print("MAX_DIM " + string(maxd));\n'
        )
    else:
        raise ValueError(f"unknown task: {task}")
    return (
        # rtimer ticks per second; default is 1 (whole seconds) — bump to 1000
        # so SOLVE_MS is in milliseconds, matching the other baselines.
        'system("--ticks-per-sec", 1000);\n'
        f"ring R = {coefficients}, ({ring_vars}), {ordering};\n"
        f"ideal I = {ideal_body};\n"
        f"{body}"
        f'print("DONE");\n'
        f"exit;\n"
    )


def _parse_singular_output(stdout: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for line in stdout.splitlines():
        m = re.match(r"^(SOLVE_MS|BASIS_LEN|NUM_COMPONENTS|MAX_DIM)\s+(-?\d+)\s*$", line.strip())
        if m:
            out[m.group(1)] = int(m.group(2))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("system_path", type=Path)
    parser.add_argument("--task", choices=list(_BASELINE_LABEL_BY_TASK.keys()), required=True)
    parser.add_argument("--timeout-secs", type=float, default=120.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    label = _BASELINE_LABEL_BY_TASK[args.task]

    tool = require_tool("Singular", label)
    if tool is None:
        return 0  # tool_missing already emitted

    try:
        polys, symbols = load_system(args.system_path)
    except Exception as exc:  # noqa: BLE001 — any parse failure becomes an emitted result.
        BaselineResult(baseline=label, status="parse_error", error=str(exc)).emit()
        return 0

    if not polys:
        BaselineResult(baseline=label, status="ok", elapsed_secs=0.0, basis_len=0).emit()
        return 0

    program = _singular_program(polys, symbols, args.task)

    try:
        rc, out, err, wall = run_subprocess(
            [tool, "-q", "--no-tty", "-c", program],
            timeout=args.timeout_secs,
        )
    except subprocess.TimeoutExpired:
        BaselineResult(baseline=label, status="timeout", elapsed_secs=args.timeout_secs).emit()
        return 0

    parsed = _parse_singular_output(out)
    if "SOLVE_MS" not in parsed or "DONE" not in out:
        BaselineResult(
            baseline=label,
            status="error",
            elapsed_secs=wall,
            error=f"Singular rc={rc}",
            tool_stdout_tail=tail(out + "\n--- stderr ---\n" + err),
        ).emit()
        return 0

    result = BaselineResult(
        baseline=label,
        status="ok",
        elapsed_secs=parsed["SOLVE_MS"] / 1000.0,
    )
    if args.task == "gb":
        result.basis_len = parsed.get("BASIS_LEN")
    elif args.task in {"facstd", "decompose"}:
        result.num_components = parsed.get("NUM_COMPONENTS")
        result.max_dim = parsed.get("MAX_DIM")
    result.emit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
