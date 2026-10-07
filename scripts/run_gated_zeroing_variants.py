#!/usr/bin/env python3
"""Variable zeroing variants on the generated systems on which zeroing splits.

Runs the four variants of ``run_physical_zeroing_variants.py`` with the
Table ``tab:baseline-comparison`` limits (3 s per branch, 60 s per run) on the
generated systems whose residual branch times out, and saves
``results/gated_zeroing_variants.json`` and ``results/gated_zeroing_variants.md``.
"""

from __future__ import annotations

import concurrent.futures
import json
import tempfile
from pathlib import Path

from benchmark_idos import REPO, generate_matrix, run, solver_args_with_engine
from run_physical_zeroing_variants import BINARY, VARIANTS, summary_fields, table_cell

JSON_OUTPUT = REPO / "results" / "gated_zeroing_variants.json"
TABLE_OUTPUT = REPO / "results" / "gated_zeroing_variants.md"
SYSTEMS: tuple[str, ...] = (
    "overlapping_random_multiples_n8_p10_k3_seed31415",
    "gated_cyclic_6",
    "gated_katsura_7",
    "gated_random_dense_quadratic_n6_seed137",
)
ENGINES: tuple[str, ...] = ("m4gb", "f5", "gvw", "symbolica", "singular-gb")
PER_BRANCH_MS = 3_000
TOTAL_MS = 60_000
JOBS = 2


def run_one(path: Path, solver_args: list[str], engine: str, flags: tuple[str, ...]) -> dict[str, object]:
    argv = [
        str(BINARY),
        str(path),
        *solver_args_with_engine(solver_args, engine),
        "--timeout",
        str(PER_BRANCH_MS),
        "--max-runtime",
        str(TOTAL_MS),
        "--summary",
        *flags,
    ]
    stdout, stderr, wall, killed, returncode = run(argv, TOTAL_MS / 1000 + 15)
    return {
        "argv": argv[1:],
        "summary": summary_fields(stdout),
        "wall_secs": wall,
        "killed": killed,
        "returncode": returncode,
        "stderr_tail": stderr[-600:],
    }


def markdown(results: dict[str, dict[str, dict[str, dict[str, object]]]]) -> str:
    lines = [
        "# Generated systems: variable zeroing variants",
        "",
        f"Entries are `T/F` in seconds ({TOTAL_MS // 1000} s total, "
        f"{PER_BRANCH_MS // 1000} s per branch), as in "
        "`run_physical_zeroing_variants.py`.",
        "",
    ]
    for system in SYSTEMS:
        lines += [
            f"## `{system}`",
            "",
            "| Engine | " + " | ".join(VARIANTS) + " |",
            "|---" * (len(VARIANTS) + 1) + "|",
            *(
                f"| {engine} | "
                + " | ".join(table_cell(results[system][variant][engine]) for variant in VARIANTS)
                + " |"
                for engine in ENGINES
            ),
            "",
        ]
    return "\n".join(lines)


def main() -> int:
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory)
        instances = {instance.name: instance for instance in generate_matrix(output)}
        tasks = [
            (system, variant, engine)
            for system in SYSTEMS
            for variant in VARIANTS
            for engine in ENGINES
        ]
        results: dict[str, dict[str, dict[str, dict[str, object]]]] = {
            system: {variant: {} for variant in VARIANTS} for system in SYSTEMS
        }

        def execute(task: tuple[str, str, str]) -> tuple[str, str, str, dict[str, object]]:
            system, variant, engine = task
            instance = instances[system]
            row = run_one(output / instance.path, instance.solver_args, engine, VARIANTS[variant])
            return system, variant, engine, row

        with concurrent.futures.ThreadPoolExecutor(max_workers=JOBS) as pool:
            for system, variant, engine, row in pool.map(execute, tasks):
                results[system][variant][engine] = row
                print(f"[{system}] {variant} {engine}: {table_cell(row)}", flush=True)
    JSON_OUTPUT.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    TABLE_OUTPUT.write_text(markdown(results), encoding="utf-8")
    print(f"saved {JSON_OUTPUT} and {TABLE_OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
