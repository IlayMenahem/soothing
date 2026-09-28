#!/usr/bin/env python3
"""Compare zeroing split values on the physical system.

Runs ``variable_zeroing`` with the M4GB backend, a 5 s budget per branch and a
600 s overall budget, once splitting on ``x = 1`` and once on a random value
(seed 0), and stores both runs in ``results/split_value_comparison.json``.
"""

from __future__ import annotations

import json

from benchmark_idos import REPO, SYMBOLICA, run

BINARY = SYMBOLICA / "target" / "release" / "variable_zeroing"
SYSTEM = REPO / "systems" / "system.txt"
OUTPUT = REPO / "results" / "split_value_comparison.json"
MAX_RUNTIME_MS = 600_000
OUTER_TIMEOUT_SECS = MAX_RUNTIME_MS / 1000 + 5
SPLIT_VALUES: dict[str, list[str]] = {
    "one": ["--split-value", "1"],
    "random": ["--split-value", "random", "--random-seed", "0"],
}


def run_split(split_args: list[str]) -> dict[str, object]:
    argv: list[str] = [
        str(BINARY),
        str(SYSTEM),
        "--gb",
        "m4gb",
        "--timeout",
        "5000",
        "--max-runtime",
        str(MAX_RUNTIME_MS),
        *split_args,
        "--summary",
    ]
    stdout, stderr, wall, killed, returncode = run(argv, OUTER_TIMEOUT_SECS)
    summary = next(
        (line for line in stdout.splitlines() if line.startswith("ZEROING_SUMMARY ")),
        "",
    )
    return {
        "argv": argv,
        "summary": summary,
        "wall_secs": wall,
        "killed": killed,
        "returncode": returncode,
        "stderr_tail": stderr[-600:],
    }


def main() -> int:
    results: dict[str, dict[str, object]] = {}
    for name, split_args in SPLIT_VALUES.items():
        results[name] = run_split(split_args)
        print(f"[{name}] {results[name]['summary'] or 'no summary'}")
    OUTPUT.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"saved {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
