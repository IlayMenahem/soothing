#!/usr/bin/env python3
"""Table ``tab:physical-system``: the physical system with a 120 s limit per run.

The system is ``systems/system.txt`` without its last equation, the Rabinowitsch
equation ``kappah*rRabKappah = 1`` (``_harness.physical_system``). Every run imposes
``kappah != 0`` by saturation instead: Alg. Branching and Alg. ZeroingBranching start
from the saturation by ``kappah`` (5 s limit, else the system itself) and keep
``kappah`` as a nonzero side condition; Singular and Macaulay2 saturate before their
task; M4GB, F5, GVW and F4 cannot saturate and add ``1 - s*kappah``.

Harnesses; all their runs start at once, each holding a solver slot of
``_harness.solver_slot`` shared with every other experiment script:

    branching   Alg. Branching with each of six backends, no limit per branch
    zeroing     Alg. ZeroingBranching with each of six backends, ideal saturation
                and 5 s per branch
    baselines   the nine standalone methods of the text, which all time out: M4GB,
                F5, GVW and F4 in grevlex, Singular and Macaulay2 lex bases,
                facstd, primdecGTZ and Macaulay2 decompose

The raw results go to ``results/physical.json``, one key per
harness. Naming harnesses reruns only those and keeps the stored results of the
others. ``--report-only`` prints the rows of the table and the baseline outcomes.

Examples:

    python3 scripts/physical_table.py
    python3 scripts/physical_table.py zeroing
    python3 scripts/physical_table.py --report-only
"""

from __future__ import annotations

import argparse
import json
import re
from functools import partial
from pathlib import Path

from _harness import (
    BINARIES,
    PHYSICAL_NONZERO,
    RESULTS,
    build,
    concurrently,
    nonzero_arguments,
    physical_system,
    run,
    run_direct,
    run_external,
)

OUTPUT = RESULTS / "physical.json"
TOTAL_MS = 120_000
PER_BRANCH_MS = 5_000
OUTER_TIMEOUT_SECS = 125.0
BACKENDS: dict[str, str] = {
    "m4gb": "M4GB",
    "f5": "F5",
    "gvw": "GVW",
    "symbolica": "F4",
    "singular": "Singular",
    "facstd": "\\texttt{facstd}",
}
BASELINES: tuple[str, ...] = (
    "m4gb-grevlex",
    "f5-grevlex",
    "gvw-grevlex",
    "builtin-grevlex",
    "singular-gb-lex",
    "m2-gb-lex",
    "singular-facstd",
    "singular-primdecGTZ",
    "m2-decompose",
)
HARNESSES: tuple[str, ...] = ("branching", "zeroing", "baselines")
SUMMARY_FIELD = re.compile(r"(\w+)=(\S+)")


def solver_argv(algorithm: str, backend: str, system: Path) -> list[str]:
    if algorithm == "branching":
        return [
            str(BINARIES["branching"]),
            str(system),
            "--gb",
            backend,
            "--timeout",
            str(TOTAL_MS),
            "--saturation-timeout",
            str(PER_BRANCH_MS),
            *nonzero_arguments(PHYSICAL_NONZERO),
            "--summary",
        ]
    return [
        str(BINARIES["variable_zeroing"]),
        str(system),
        "--gb",
        backend,
        "--timeout",
        str(PER_BRANCH_MS),
        "--max-runtime",
        str(TOTAL_MS),
        "--ideal-saturation",
        *nonzero_arguments(PHYSICAL_NONZERO),
        "--summary",
    ]


def kept_line(algorithm: str, line: str) -> bool:
    """The stdout lines stored with a run: the summary, and for Alg. Branching the per-cover outcomes."""
    if algorithm == "branching":
        return line.startswith(("Branching over ", "  cover #", "BRANCHING_SUMMARY "))
    return line.startswith("ZEROING_SUMMARY ")


def run_solver(algorithm: str, backend: str, system: Path) -> dict[str, object]:
    argv = solver_argv(algorithm, backend, system)
    stdout, stderr, wall, killed, returncode = run(argv, OUTER_TIMEOUT_SECS)
    fields: dict[str, object] = {
        "backend": backend,
        "argv": argv,
        "wall_secs": wall,
        "killed": killed,
        "returncode": returncode,
        "stderr_lines": len(stderr.splitlines()),
        **({"stderr_tail": stderr[-600:]} if algorithm == "zeroing" else {}),
    }
    harness = dict(sorted(fields.items()))
    kept = [line for line in stdout.splitlines() if kept_line(algorithm, line)]
    summary_prefix = "BRANCHING_SUMMARY " if algorithm == "branching" else "ZEROING_SUMMARY "
    return {
        "type": algorithm,
        "backend": backend,
        "summary": next((line for line in kept if line.startswith(summary_prefix)), ""),
        "harness": harness,
        "stdout": "\n".join([*kept, "HARNESS_SUMMARY " + json.dumps(harness, sort_keys=True)]) + "\n",
    }


def run_baseline(label: str, system: Path) -> dict[str, object]:
    seconds = TOTAL_MS // 1000
    if label.endswith("-grevlex"):
        return run_direct(system, label.removesuffix("-grevlex"), seconds, PHYSICAL_NONZERO)
    return run_external(label, system, seconds, PHYSICAL_NONZERO)


def run_one(system: Path, harness: str, name: str) -> dict[str, object]:
    result = run_baseline(name, system) if harness == "baselines" else run_solver(harness, name, system)
    outcome = result["status"] if harness == "baselines" else result["summary"]
    print(f"[{harness}] {name}: {outcome}", flush=True)
    return result


def run_harness(system: Path, harness: str) -> dict[str, dict[str, object]]:
    """Every backend or baseline of `harness` concurrently, 120 s each."""
    names = BASELINES if harness == "baselines" else tuple(BACKENDS)
    return dict(zip(names, concurrently(partial(run_one, system, harness), names), strict=True))


def summary_fields(summary: str) -> dict[str, str]:
    return dict(SUMMARY_FIELD.findall(summary))


def timing(fields: dict[str, str]) -> str:
    """``$T/F$``, or ``timeout$/F$`` for a run the total limit stopped. Alg. Branching
    times its search after the saturation by ``kappah``, so its ``saturation_secs`` is added."""
    saturation = float(fields.get("saturation_secs", "0"))
    first = fields.get("first_solution_secs", "unknown")
    first_shown = "--" if first == "unknown" else f"{float(first) + saturation:.3f}"
    if fields.get("complete") != "true":
        return f"timeout$/{first_shown}$"
    return f"${float(fields['elapsed_secs']) + saturation:.3f}/{first_shown}$"


def branching_result(fields: dict[str, str]) -> str:
    if fields.get("complete") == "true":
        return f"complete: {fields['satisfiable']} SAT, {fields['unsatisfiable']} UNSAT"
    return f"partial: {fields['satisfiable']} SAT, {fields['timeouts']} timeout"


def zeroing_result(fields: dict[str, str]) -> str:
    if fields.get("complete") == "true":
        return f"complete: {fields['splits']} splits"
    return f"partial: {fields['splits']} splits, {fields['unresolved']} unresolved"


def report(results: dict[str, dict[str, dict[str, object]]]) -> list[str]:
    rows: list[str] = ["% tab:physical-system"]
    for backend, name in BACKENDS.items():
        branching = summary_fields(str(results.get("branching", {}).get(backend, {}).get("summary", "")))
        zeroing = summary_fields(str(results.get("zeroing", {}).get(backend, {}).get("summary", "")))
        cells = [
            timing(branching) if branching else "--",
            branching_result(branching) if branching else "--",
            timing(zeroing) if zeroing else "--",
            zeroing_result(zeroing) if zeroing else "--",
        ]
        rows.append(f"{name} & " + " & ".join(cells) + " \\\\")
    baselines = results.get("baselines", {})
    rows += ["", "% standalone baselines"]
    rows += [
        f"% {label}: {baselines[label]['status']} after {float(str(baselines[label]['wall_secs'])):.1f} s"
        if label in baselines
        else f"% {label}: not run"
        for label in BASELINES
    ]
    return rows


def load_results() -> dict[str, dict[str, dict[str, object]]]:
    return json.loads(OUTPUT.read_text(encoding="utf-8")) if OUTPUT.exists() else {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("harnesses", nargs="*", help=f"any of {', '.join(HARNESSES)}; default: all")
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    unknown = set(args.harnesses) - set(HARNESSES)
    if unknown:
        parser.error(f"unknown harness(es): {', '.join(sorted(unknown))}")
    return args


def main() -> int:
    args = parse_args()
    if not args.report_only:
        if not args.no_build:
            build("sig_groebner", "branching", "variable_zeroing")
        system = physical_system()
        results = load_results()
        harnesses = args.harnesses or list(HARNESSES)
        results.update(zip(harnesses, concurrently(partial(run_harness, system), harnesses), strict=True))
        OUTPUT.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"saved {OUTPUT}")
    print(*report(load_results()), sep="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
