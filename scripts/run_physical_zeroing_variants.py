#!/usr/bin/env python3
"""Variable zeroing on the physical system with each nonzero encoding and pruning.

Reruns the setup of Table ``tab:physical-system`` (120 s total, 5 s per
branch, six backends) for the variants below, sequentially so the timings are
comparable, and saves the raw summaries in
``results/physical_zeroing_variants.json`` and a table in
``results/physical_zeroing_variants.md``.

Variants:
- ``rabinowitsch``: one equation ``1 - s_x x`` per nonzero variable (paper).
- ``saturation``: the nonzero child is ``<H> : (prod N)^oo`` (Singular ``sat``).
- ``rabinowitsch+prune`` / ``saturation+prune``: also skip covers inside a
  solved component and drop solved components inside another.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from benchmark_idos import REPO, SYMBOLICA, run

BINARY = SYMBOLICA / "target" / "release" / "variable_zeroing"
SYSTEM = REPO / "systems" / "system.txt"
JSON_OUTPUT = REPO / "results" / "physical_zeroing_variants.json"
TABLE_OUTPUT = REPO / "results" / "physical_zeroing_variants.md"
BACKENDS: tuple[str, ...] = ("m4gb", "f5", "gvw", "symbolica", "singular", "facstd")
VARIANTS: dict[str, tuple[str, ...]] = {
    "rabinowitsch": (),
    "saturation": ("--ideal-saturation",),
    "rabinowitsch+prune": ("--prune",),
    "saturation+prune": ("--ideal-saturation", "--prune"),
}
PER_BRANCH_MS = 5_000
TOTAL_MS = 120_000


def summary_fields(stdout: str) -> dict[str, str]:
    line = next(
        (line for line in stdout.splitlines() if line.startswith("ZEROING_SUMMARY ")),
        "",
    )
    pairs = (field.split("=", 1) for field in line.split()[1:])
    return {key: value for key, value in pairs}


def run_variant(backend: str, flags: tuple[str, ...]) -> dict[str, object]:
    argv = [
        str(BINARY),
        str(SYSTEM),
        "--gb",
        backend,
        "--timeout",
        str(PER_BRANCH_MS),
        "--max-runtime",
        str(TOTAL_MS),
        "--summary",
        *flags,
    ]
    stdout, stderr, wall, killed, returncode = run(argv, TOTAL_MS / 1000 + 15)
    return {
        "argv": argv,
        "summary": summary_fields(stdout),
        "wall_secs": wall,
        "killed": killed,
        "returncode": returncode,
        "stderr_tail": stderr[-600:],
    }


def table_cell(result: dict[str, object]) -> str:
    summary: dict[str, str] = result["summary"]  # type: ignore[assignment]
    if not summary:
        return f"error (exit {result['returncode']})"
    complete = summary["complete"] == "true"
    time = f"{float(summary['elapsed_secs']):.3f}" if complete else "timeout"
    first = summary["first_solution_secs"]
    first = f"{float(first):.3f}" if first != "unknown" else "-"
    outcome = (
        f"complete: {summary['splits']} splits"
        if complete
        else f"partial: {summary['splits']} splits, {summary['unresolved']} unresolved"
    )
    extras = [
        f"sat {summary['saturation_units']}/{summary['saturations']} unit"
        if summary["saturations"] != "0"
        else "",
        f"{summary['pruned_covers']} pruned" if summary["pruned_covers"] != "0" else "",
        f"{summary['redundant_components']} redundant"
        if summary["redundant_components"] != "0"
        else "",
    ]
    detail = "; ".join(extra for extra in extras if extra)
    return f"${time}/{first}$ {outcome}" + (f" ({detail})" if detail else "")


def markdown(results: dict[str, dict[str, dict[str, object]]]) -> str:
    header = "| Backend | " + " | ".join(VARIANTS) + " |"
    rule = "|---" * (len(VARIANTS) + 1) + "|"
    rows = [
        f"| {backend} | "
        + " | ".join(table_cell(results[variant][backend]) for variant in VARIANTS)
        + " |"
        for backend in BACKENDS
    ]
    return "\n".join(
        [
            "# Physical system: variable zeroing variants",
            "",
            f"Entries are `T/F` in seconds ({TOTAL_MS // 1000} s total, "
            f"{PER_BRANCH_MS // 1000} s per branch); `sat a/b unit` counts the "
            "saturations that returned the unit ideal out of those attempted.",
            "",
            header,
            rule,
            *rows,
            "",
        ]
    )


def main() -> int:
    if not BINARY.exists():
        print(f"build {BINARY} first (cargo build --release)", file=sys.stderr)
        return 1
    results: dict[str, dict[str, dict[str, object]]] = {}
    for variant, flags in VARIANTS.items():
        results[variant] = {}
        for backend in BACKENDS:
            results[variant][backend] = run_variant(backend, flags)
            print(f"[{variant}] {backend}: {table_cell(results[variant][backend])}", flush=True)
    JSON_OUTPUT.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    TABLE_OUTPUT.write_text(markdown(results), encoding="utf-8")
    print(f"saved {JSON_OUTPUT} and {TABLE_OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
