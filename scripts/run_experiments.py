#!/usr/bin/env python3
"""Run the experiments of ``paper/sections/experiments.tex`` all at once.

Every experiment script runs as its own process, and all of them run concurrently.
Each solver run holds a solver slot of ``_harness.solver_slot``, so together they use
at most ``RESOURCE_SHARE`` of the cores, and no run starts while the running ones use
``RESOURCE_SHARE`` of the physical memory or more, swapped memory included. The Rust binaries are built once first. Each script
resumes from its store in ``results/``. One log per experiment goes to ``--logs``.
Concurrent runs slow one another, so their times are not comparable to runs made alone.

Examples:

    python3 scripts/run_experiments.py --logs /tmp/experiment_logs
    python3 scripts/run_experiments.py --only physical-table,scaling --logs /tmp/experiment_logs
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

from _harness import REPO, build

SCRIPTS = REPO / "scripts"
# Experiment name -> script and arguments; "{systems}" is the directory for generated systems.
EXPERIMENTS: dict[str, tuple[str, ...]] = {
    "physical-table": ("physical_table.py", "--no-build"),
    "physical-zeroing-figure": ("physical_zeroing_figure.py", "--no-build"),
    "physical-primary": ("physical_primary_decomposition.py",),
    "baselines": ("baseline_table.py", "--part", "baselines", "--no-build"),
    "branching": ("baseline_table.py", "--part", "branching", "--no-build"),
    "zeroing": ("baseline_table.py", "--part", "zeroing", "--no-build"),
    "cutoff-sweep": ("baseline_table.py", "--part", "cutoff-sweep", "--no-build"),
    "scaling": ("scaling_table.py", "--no-build", "--output-dir", "{systems}/scaling"),
    "decomposition": ("scaling_decomposition_table.py", "--output-dir", "{systems}/decomposition"),
    "ideal-refine": ("scaling_decomposition_table.py", "--systems", "baseline", "--output-dir", "{systems}/ideal_refine"),
}
# Experiments whose report reads the scaling store, rendered again once everything has finished.
READS_SCALING: frozenset[str] = frozenset({"decomposition", "ideal-refine"})


def command(name: str, systems: Path) -> list[str]:
    script, *arguments = EXPERIMENTS[name]
    return [sys.executable, "-u", str(SCRIPTS / script), *(a.format(systems=systems) for a in arguments)]


def launch(name: str, systems: Path, logs: Path) -> subprocess.Popen[bytes]:
    """Start one experiment with its output in ``logs/NAME.log``."""
    with (logs / f"{name}.log").open("w", encoding="utf-8") as log:
        return subprocess.Popen(command(name, systems), stdout=log, stderr=subprocess.STDOUT, cwd=REPO)


def run_all(names: list[str], systems: Path, logs: Path) -> dict[str, int]:
    """Start every experiment, wait for all of them, and return their exit codes."""
    processes = {name: launch(name, systems, logs) for name in names}
    print(f"started {', '.join(names)}; logs in {logs}", flush=True)
    return {name: process.wait() for name, process in processes.items()}


def render_scaling_readers() -> int:
    """Render the decomposition reports again from the final stores."""
    report = [sys.executable, str(SCRIPTS / "scaling_decomposition_table.py"), "--report-only"]
    return subprocess.run(report, cwd=REPO, check=False).returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--logs", type=Path, required=True, help="directory for one log per experiment")
    parser.add_argument("--only", default="", help=f"comma-separated experiments of: {', '.join(EXPERIMENTS)}")
    parser.add_argument("--systems", type=Path, help="directory for generated systems; default: a temporary one")
    parser.add_argument("--no-build", action="store_true")
    args = parser.parse_args()
    unknown = {name for name in args.only.split(",") if name} - EXPERIMENTS.keys()
    if unknown:
        parser.error(f"unknown experiment(s): {', '.join(sorted(unknown))}")
    return args


def main() -> int:
    args = parse_args()
    names = [name for name in args.only.split(",") if name] or list(EXPERIMENTS)
    if not args.no_build:
        build("sig_groebner", "branching", "variable_zeroing")
    args.logs.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="experiment_systems_") as temporary:
        exit_codes = run_all(names, args.systems or Path(temporary), args.logs)
    if READS_SCALING & set(names):
        exit_codes["reports"] = render_scaling_readers()
    for name, code in exit_codes.items():
        print(f"{name}: {'ok' if code == 0 else f'exit {code}'}")
    return 0 if all(code == 0 for code in exit_codes.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
