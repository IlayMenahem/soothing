"""Run Macaulay2 as a baseline on a Mathematica-style polynomial system.

Tasks:
    gb         — lex Gröbner basis via ``gb(I)``; reports ``basis_len``.
    decompose  — minimal-prime decomposition via ``decompose I`` (from the
                 ``PrimaryDecomposition`` package); reports ``num_components``
                 and ``max_dim``.

Like the Singular driver, all free symbols are made polynomial-ring variables,
including ``pi`` and physics constants. Emits a single ``BaselineResult`` JSON
line on stdout.
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

import sympy as sp

from _common import BaselineResult, load_system, require_tool, run_subprocess, tail

_BASELINE_LABEL_BY_TASK = {
    "gb": "m2-gb-lex",
    "decompose": "m2-decompose",
}


def _poly_to_m2(poly: sp.Basic, rename: dict[str, str]) -> str:
    text = sp.sstr(sp.expand(poly))
    text = text.replace("**", "^")
    if rename:
        pattern = re.compile(r"\b(" + "|".join(re.escape(k) for k in rename) + r")\b")
        text = pattern.sub(lambda m: rename[m.group(1)], text)
    return text


def _m2_program(polys: list[sp.Basic], symbols: list[sp.Symbol], task: str) -> str:
    # M2 predefines several lower-case identifiers as mathematical constants
    # (e, pi, ii) and treating them as ring variables raises "expected strings,
    # integers, or symbols". Prefix every variable to make collisions impossible.
    rename = {str(s): f"v_{s}" for s in symbols}
    ring_vars = ",".join(rename[str(s)] for s in symbols)
    ideal_body = ",\n  ".join(_poly_to_m2(p, rename) for p in polys)
    # cpuTime() returns floating-point seconds; ``currentTime()`` is integer-second
    # Unix epoch and rounds to 0 for fast operations.
    if task == "gb":
        body = (
            'startTime = cpuTime();\n'
            'G = gens gb I;\n'
            'dt = cpuTime() - startTime;\n'
            '<< "SOLVE_MS " << floor(dt * 1000) << "\\n";\n'
            '<< "BASIS_LEN " << numColumns G << "\\n";\n'
        )
    elif task == "decompose":
        body = (
            'needsPackage "PrimaryDecomposition";\n'
            'startTime = cpuTime();\n'
            'comps = decompose I;\n'
            'dt = cpuTime() - startTime;\n'
            '<< "SOLVE_MS " << floor(dt * 1000) << "\\n";\n'
            '<< "NUM_COMPONENTS " << #comps << "\\n";\n'
            'maxd = -1;\n'
            'scan(comps, c -> (d = dim c; if d > maxd then maxd = d));\n'
            '<< "MAX_DIM " << maxd << "\\n";\n'
        )
    else:
        raise ValueError(f"unknown task: {task}")
    return (
        f'R = QQ[{ring_vars}, MonomialOrder => Lex];\n'
        f'I = ideal(\n  {ideal_body}\n);\n'
        f'{body}'
        f'<< "DONE\\n";\n'
        f'exit 0;\n'
    )


def _parse_m2_output(stdout: str) -> dict[str, int]:
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

    tool = require_tool("M2", label)
    if tool is None:
        return 0

    try:
        polys, symbols = load_system(args.system_path)
    except Exception as exc:  # noqa: BLE001
        BaselineResult(baseline=label, status="parse_error", error=str(exc)).emit()
        return 0

    if not polys:
        BaselineResult(baseline=label, status="ok", elapsed_secs=0.0, basis_len=0).emit()
        return 0

    program = _m2_program(polys, symbols, args.task)

    try:
        rc, out, err, wall = run_subprocess(
            [tool, "--no-readline", "--silent", "--stop", "-q", "-e", program],
            timeout=args.timeout_secs,
        )
    except subprocess.TimeoutExpired:
        BaselineResult(baseline=label, status="timeout", elapsed_secs=args.timeout_secs).emit()
        return 0

    parsed = _parse_m2_output(out)
    if "SOLVE_MS" not in parsed or "DONE" not in out:
        BaselineResult(
            baseline=label,
            status="error",
            elapsed_secs=wall,
            error=f"M2 rc={rc}",
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
    elif args.task == "decompose":
        result.num_components = parsed.get("NUM_COMPONENTS")
        result.max_dim = parsed.get("MAX_DIM")
    result.emit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
