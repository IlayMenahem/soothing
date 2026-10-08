#!/usr/bin/env python3
"""Run the variable-zeroing figure experiment on the physical system.

Runs ``variable_zeroing`` with the M4GB backend, ideal saturation, a 5 s budget
per branch and a 120 s overall budget, and stores the summary together with the raw TikZ tree
in ``results/physical_system_m4gb_5s_figure_run.json``. The paper figure
``paper/figs/variable_zeroing_tree.tex`` is derived from that raw TikZ.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

from benchmark_idos import REPO, SYMBOLICA, run

BINARY = SYMBOLICA / "target" / "release" / "variable_zeroing"
SYSTEM = REPO / "systems" / "system.txt"
OUTPUT = REPO / "results" / "physical_system_m4gb_5s_figure_run.json"
OUTER_TIMEOUT_SECS = 125.0


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary_line(stdout: str) -> str:
    return next(
        (line for line in stdout.splitlines() if line.startswith("ZEROING_SUMMARY ")),
        "",
    )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="zeroing_figure_") as directory:
        plot = Path(directory) / "variable_zeroing_tree.tex"
        argv: list[str] = [
            str(BINARY),
            str(SYSTEM),
            "--gb",
            "m4gb",
            "--timeout",
            "5000",
            "--max-runtime",
            "120000",
            "--ideal-saturation",
            "--summary",
            "--plot",
            str(plot),
        ]
        stdout, stderr, wall, killed, returncode = run(argv, OUTER_TIMEOUT_SECS)
        raw_tikz = plot.read_text(encoding="utf-8") if plot.exists() else ""

    result: dict[str, object] = {
        "argv": argv[:-2],
        "binary_sha256": sha256(BINARY),
        "input_sha256": sha256(SYSTEM),
        "raw_tikz": raw_tikz,
        "returncode": returncode,
        "killed": killed,
        "wall_secs": wall,
        "stderr": stderr,
        "stdout": stdout,
        "summary": summary_line(stdout),
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(result["summary"])
    print(f"saved {OUTPUT}")
    return 0 if returncode == 0 and not killed else 1


if __name__ == "__main__":
    raise SystemExit(main())
