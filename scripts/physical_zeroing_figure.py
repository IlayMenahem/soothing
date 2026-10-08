#!/usr/bin/env python3
"""Figure ``fig:variable_zeroing_tree``: the variable-zeroing tree of the physical system.

Runs ``variable_zeroing`` with the M4GB backend, ideal saturation, a 5 s budget
per branch and a 120 s overall budget, and stores the summary together with the raw TikZ tree
in ``results/physical_system_m4gb_5s_figure_run.json``. The paper figure
``paper/figs/variable_zeroing_tree.tex`` is then derived from that raw TikZ with the
plot-only changes of the current figure: a source header, "no hitting sets" in place
of "no covers", the SAT/UNSAT legend, and no unused macros.

Examples:

    python3 scripts/physical_zeroing_figure.py
    python3 scripts/physical_zeroing_figure.py --derive-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
from pathlib import Path

from _harness import BINARIES, PHYSICAL_SYSTEM, REPO, RESULTS, build, run

BINARY = BINARIES["variable_zeroing"]
OUTPUT = RESULTS / "physical_system_m4gb_5s_figure_run.json"
FIGURE = REPO / "paper" / "figs" / "variable_zeroing_tree.tex"
OUTER_TIMEOUT_SECS = 125.0


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary_line(stdout: str) -> str:
    return next(
        (line for line in stdout.splitlines() if line.startswith("ZEROING_SUMMARY ")),
        "",
    )


def run_figure() -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="zeroing_figure_") as directory:
        plot = Path(directory) / "variable_zeroing_tree.tex"
        argv: list[str] = [
            str(BINARY),
            str(PHYSICAL_SYSTEM),
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

    return {
        "argv": argv[:-2],
        "binary_sha256": sha256(BINARY),
        "input_sha256": sha256(PHYSICAL_SYSTEM),
        "raw_tikz": raw_tikz,
        "returncode": returncode,
        "killed": killed,
        "wall_secs": wall,
        "stderr": stderr,
        "stdout": stdout,
        "summary": summary_line(stdout),
    }


def derive_figure(raw: str, old_figure: str) -> str:
    """The paper figure from the raw TikZ, keeping the legend of the current figure at the raw legend's position."""
    legend = next(line for line in old_figure.splitlines() if "Branch outcome:" in line)
    raw_legend = next(line for line in raw.splitlines() if "Cover outcome:" in line)
    position = re.search(r"at \(([^)]*)\)", raw_legend)
    if position is None:
        raise ValueError("the raw legend has no position")
    legend = re.sub(r"at \([^)]*\)", f"at ({position.group(1)})", legend, count=1)

    first_line, *rest = raw.splitlines()
    figure_lines = [
        first_line,
        "% Source: results/physical_system_m4gb_5s_figure_run.json",
        "% Plot-only changes: omit unused macros; label outcomes SAT/UNSAT.",
        *(
            legend if "Cover outcome:" in line else line.replace(r"\vzNoCovers\ no covers}", r"\vzNoCovers\ no hitting sets}")
            for line in rest
            if not line.startswith(r"\providecommand{\vzPruned}")
        ),
    ]
    return "\n".join(figure_lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--derive-only", action="store_true", help="rederive the figure from the stored run")
    parser.add_argument("--no-build", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.derive_only:
        if not args.no_build:
            build("variable_zeroing")
        result = run_figure()
        OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(result["summary"])
        print(f"saved {OUTPUT}")
        if result["returncode"] != 0 or result["killed"]:
            return 1
    stored = json.loads(OUTPUT.read_text(encoding="utf-8"))
    FIGURE.write_text(derive_figure(str(stored["raw_tikz"]), FIGURE.read_text(encoding="utf-8")), encoding="utf-8")
    print(f"updated {FIGURE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
