#!/usr/bin/env python3
"""Table ``tab:scaling-classes``: grow each size-graded family until every method times out.

Families: reduced cyclic n-roots from SymbolicData (``Singular.rcyclic_n``)
and the generators of ``_systems.py`` (Nash
equilibria of binary games, box-QP faces, Lotka--Volterra supports).

Families run concurrently. Within a family the sizes run in order and the methods
of one size concurrently; a method that times out or fails at one size is not run
at larger sizes. Each run holds a solver slot of ``_harness.solver_slot``, shared
with every other experiment script. Every time is process wall time, so factor
branching and the tools alone both pay their process startup.

Examples:

    python3 scripts/scaling_table.py --timeout 60 --output-dir /tmp/scaling
    python3 scripts/scaling_table.py --families box_qp --output-dir /tmp/scaling
    python3 scripts/scaling_table.py --report-only --output-dir /tmp/scaling
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from functools import partial
from pathlib import Path

from _harness import (
    RESULTS,
    append_row,
    build,
    concurrently,
    load_rows,
    run_branching,
    run_direct,
    run_external,
)
from _systems import box_qp, lotka_volterra, nash_game, symbolicdata_system

RESULTS_FILE = RESULTS / "scaling.jsonl"
REPORT_FILE = RESULTS / "scaling.md"
BRANCH_ENGINES: tuple[str, ...] = (
    "m4gb",
    "f5",
    "gvw",
    "symbolica",
    "singular-gb",
    "singular-facstd",
    "singular-primdecGTZ",
    "m2-decompose",
)
FACTOR_BRANCHING: tuple[str, ...] = tuple(f"branching-{engine}" for engine in BRANCH_ENGINES)
BASELINES: tuple[str, ...] = (
    "m4gb-grevlex",
    "f5-grevlex",
    "gvw-grevlex",
    "singular-gb-dp",
    "singular-gb-lex",
    "singular-facstd",
    "singular-primdecGTZ",
    "m2-gb-lex",
    "m2-decompose",
)

SystemMaker = Callable[[Path, int], Path]


def rcyclic(output: Path, size: int) -> Path:
    path = symbolicdata_system(output, f"Singular.rcyclic_{size}.xml")
    if path is None:
        raise SystemExit(f"SymbolicData has no Singular.rcyclic_{size}")
    return path


def from_instance(output: Path, instance_path: str) -> Path:
    return output / instance_path


FAMILIES: dict[str, tuple[SystemMaker, range, tuple[str, ...]]] = {
    "rcyclic": (rcyclic, range(8, 20), BASELINES),
    "nash_generic": (
        lambda output, n: from_instance(output, nash_game(output, n, 1000 + n, 9).path),
        range(3, 9),
        BASELINES,
    ),
    "nash_degenerate": (
        lambda output, n: from_instance(output, nash_game(output, n, 2000 + n, 1).path),
        range(3, 9),
        BASELINES,
    ),
    "box_qp": (
        lambda output, n: from_instance(output, box_qp(output, n, 3000 + n, 0.5).path),
        range(4, 13),
        BASELINES,
    ),
    "lotka_volterra": (
        lambda output, n: from_instance(output, lotka_volterra(output, n, 4000 + n, 0.4).path),
        range(6, 21, 2),
        BASELINES,
    ),
}


def run_method(method: str, path: Path, timeout: int) -> dict[str, object]:
    if method.startswith("branching-"):
        return run_branching(path, [], timeout, method.removeprefix("branching-"))
    if method.endswith("-grevlex"):
        return run_direct(path, method.removesuffix("-grevlex"), timeout)
    return run_external(method, path, timeout)


def finished(row: dict[str, object]) -> bool:
    return row["status"] == "ok" and row.get("complete", True) is not False


def measure(
    family: str, size: int, path: Path, timeout: int, tag: dict[str, str], method: str
) -> dict[str, object]:
    """Run one method on one system of a family and store its row."""
    row = run_method(method, path, timeout)
    row.pop("error", None)
    record = {"family": family, "size": size, "system": path.stem, "label": method, "timeout": timeout, **tag, **row}
    append_row(RESULTS_FILE, record)
    print(f"{path.stem} {method} {row['status']} {row['wall_secs']:.3f}", flush=True)
    return record


def benchmark_family(family: str, output: Path, timeout: int, condition: str | None) -> None:
    make, sizes, baselines = FAMILIES[family]
    done = {(row["size"], row["label"]): row for row in load_rows(RESULTS_FILE) if row["family"] == family}
    # The smallest size at which each method timed out or failed; it is not run above it.
    failed_at: dict[str, int] = {}
    for (size, label), row in sorted(done.items()):
        if not finished(row):
            failed_at.setdefault(str(label), int(str(size)))
    tag: dict[str, str] = {} if condition is None else {"condition": condition}
    methods = (*FACTOR_BRANCHING, *baselines)
    for size in sizes:
        pending = [m for m in methods if failed_at.get(m, size) >= size and (size, m) not in done]
        if not pending:
            continue
        path = make(output, size)
        for record in concurrently(partial(measure, family, size, path, timeout, tag), pending):
            if not finished(record):
                failed_at[str(record["label"])] = min(failed_at.get(str(record["label"]), size), size)


def best(rows: list[dict[str, object]], labels: tuple[str, ...]) -> tuple[float, str] | None:
    times = [(float(str(row["wall_secs"])), str(row["label"])) for row in rows if row["label"] in labels and finished(row)]
    return min(times) if times else None


def dimension(row: dict[str, object]) -> object:
    """The largest dimension a run reports; each tool names the field differently."""
    return next((row[key] for key in ("largest_dimension", "max_dim", "dimension") if row.get(key) is not None), None)


def cell(entry: tuple[float, str] | None) -> str:
    return "TO" if entry is None else f"{entry[0]:.2f} ({entry[1]})"


def render_report() -> str:
    rows = load_rows(RESULTS_FILE)
    lines = [
        "# Scaling of factor branching on size-graded families",
        "",
        "Wall time in seconds of the fastest completed method per class. Within a family,",
        "sizes run in order and the methods of one size concurrently; families run concurrently.",
        "A method that times out or fails at one size is not run at larger sizes.",
        "*Dimensions* lists the distinct largest dimensions reported by completed runs.",
        ("A size marked † has at least one run recorded under a noted machine condition"
         " (the `condition` field of its rows); treat its times as unreliable."),
        "",
    ]
    for family, (_, sizes, baselines) in FAMILIES.items():
        family_rows = [row for row in rows if row["family"] == family]
        if not family_rows:
            continue
        lines += [
            f"## {family}",
            "",
            "| size | FB branches | factor branching | best baseline | ratio | dimensions |",
            "|---|---|---|---|---|---|",
        ]
        for size in sizes:
            size_rows = [row for row in family_rows if row["size"] == size]
            if not size_rows:
                continue
            ours = best(size_rows, FACTOR_BRANCHING)
            theirs = best(size_rows, baselines)
            branches = sorted({int(str(row["total"])) for row in size_rows if row["label"] in FACTOR_BRANCHING and "total" in row})
            dimensions = sorted({str(dimension(row)) for row in size_rows if finished(row) and dimension(row) is not None})
            ratio = (
                f"{theirs[0] / ours[0]:.1f}x" if ours and theirs else "only FB" if ours else "only baseline" if theirs else "—"
            )
            marker = " †" if any("condition" in row for row in size_rows) else ""
            lines.append(
                f"| {size}{marker} | {', '.join(map(str, branches))} | {cell(ours)} | {cell(theirs)} | {ratio} | {', '.join(dimensions)} |"
            )
        lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--families", help="comma-separated families")
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--condition", help="machine condition recorded on every new row, e.g. 'swap pressure'")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    families = args.families.split(",") if args.families else list(FAMILIES)
    unknown = set(families) - FAMILIES.keys()
    if unknown:
        raise SystemExit(f"unknown families: {', '.join(sorted(unknown))}")
    if not args.report_only:
        if not args.no_build:
            build("sig_groebner", "branching")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        concurrently(lambda family: benchmark_family(family, args.output_dir, args.timeout, args.condition), families)
    REPORT_FILE.write_text(render_report() + "\n", encoding="utf-8")
    print(f"updated {REPORT_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
