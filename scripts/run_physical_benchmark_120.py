#!/usr/bin/env python3
"""Run the physical system benchmark for all backends with a 120 s limit.

Every backend runs factor branching and variable zeroing (5 s per branch)
sequentially, so the timings are comparable, and the raw results are saved in
``results/physical_system_120s_results.json``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
OUTPUT = REPO / "results" / "physical_system_120s_results.json"

BACKENDS: tuple[str, ...] = ("m4gb", "f5", "gvw", "symbolica", "singular", "facstd")
HARNESSES: dict[str, tuple[str, str]] = {
    "branching": ("_run_parallel_branching_benchmark.py", "BRANCHING_SUMMARY "),
    "zeroing": ("_run_zeroing_benchmark.py", "ZEROING_SUMMARY "),
}


def run_harness(algorithm: str, backend: str) -> dict[str, object]:
    script, summary_prefix = HARNESSES[algorithm]
    completed = subprocess.run(  # noqa: S603
        [sys.executable, str(SCRIPTS / script), backend],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    lines = completed.stdout.splitlines()
    summary = next((line for line in lines if line.startswith(summary_prefix)), "")
    harness: dict[str, object] = next(
        (
            json.loads(line.removeprefix("HARNESS_SUMMARY "))
            for line in lines
            if line.startswith("HARNESS_SUMMARY ")
        ),
        {},
    )
    return {
        "type": algorithm,
        "backend": backend,
        "summary": summary,
        "harness": harness,
        "stdout": completed.stdout,
    }


def run_algorithm(algorithm: str) -> dict[str, dict[str, object]]:
    print(f"=== {algorithm} (120 s limit) ===")
    results: dict[str, dict[str, object]] = {}
    for backend in BACKENDS:
        results[backend] = run_harness(algorithm, backend)
        print(f"[{algorithm}] {backend}: {results[backend]['summary']}")
    return results


def main() -> int:
    """Run every harness, or only those named on the command line, keeping the
    stored results of the others."""
    selected = sys.argv[1:] or list(HARNESSES)
    unknown = set(selected) - HARNESSES.keys()
    if unknown:
        raise SystemExit(f"unknown harness(es): {', '.join(sorted(unknown))}")
    results: dict[str, dict[str, dict[str, object]]] = (
        json.loads(OUTPUT.read_text(encoding="utf-8")) if OUTPUT.exists() else {}
    )
    results.update({algorithm: run_algorithm(algorithm) for algorithm in selected})
    OUTPUT.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"saved {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
