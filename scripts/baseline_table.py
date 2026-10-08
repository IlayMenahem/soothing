#!/usr/bin/env python3
"""Tables ``tab:benchmark-families`` and ``tab:baseline-comparison``: the 20 generated systems.

Parts, each a resumable Markdown matrix of system x method in ``results/``:

    baselines     the nine standalone methods: M4GB, F5, GVW and F4 in grevlex,
                  Singular and Macaulay2 lex bases, facstd, primdecGTZ and
                  Macaulay2 decompose (``idos_engine_comparison.md``)
    branching     Alg. Branching with five backends (``branching_engine_comparison.md``)
    zeroing       Alg. ZeroingBranching with five backends and 3 s per branch
                  (``zeroing_engine_comparison.md``)
    cutoff-sweep  Alg. ZeroingBranching with F4 on the gated systems at 1, 2, 3 and
                  5 s per branch, the choice of the 3 s limit in the setup
                  (``zeroing_f4_cutoff_sweep.json``)

Only cells marked NR run, unless ``--rerun-all``. ``--report-only`` prints the rows
of ``tab:baseline-comparison`` and ``tab:appendix-zeroing`` from the stored results.
The systems are generated in a temporary directory unless ``--output-dir`` is given.

Examples:

    python3 scripts/baseline_table.py --part baselines --jobs 1
    python3 scripts/baseline_table.py --part branching --only branching-symbolica
    python3 scripts/baseline_table.py --part cutoff-sweep
    python3 scripts/baseline_table.py --generate-only --output-dir systems/benchmarks
    python3 scripts/baseline_table.py --report-only
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from _harness import (
    RESULTS,
    ZEROING_BRANCH_TIMEOUT_MS,
    build,
    run_branching,
    run_direct,
    run_external,
    run_zeroing,
    subprocess_limit,
)
from _systems import MATRIX_REQUIRED_COVERS, Instance, generate_matrix

ENGINES: tuple[str, ...] = ("m4gb", "f5", "gvw", "symbolica", "singular-gb")
DIRECT: tuple[str, ...] = (
    "m4gb-grevlex",
    "f5-grevlex",
    "gvw-grevlex",
    "builtin-grevlex",
    "singular-gb-lex",
    "m2-gb-lex",
)
DECOMPOSITION: tuple[str, ...] = ("singular-facstd", "singular-primdecGTZ", "m2-decompose")
METHODS: dict[str, tuple[str, ...]] = {
    "baselines": (*DIRECT, *DECOMPOSITION),
    "branching": tuple(f"branching-{engine}" for engine in ENGINES),
    "zeroing": tuple(f"zeroing-{engine}" for engine in ENGINES),
}
MATRICES: dict[str, Path] = {
    "baselines": RESULTS / "idos_engine_comparison.md",
    "branching": RESULTS / "branching_engine_comparison.md",
    "zeroing": RESULTS / "zeroing_engine_comparison.md",
}
SWEEP_FILE = RESULTS / "zeroing_f4_cutoff_sweep.json"
SWEEP_BRANCH_TIMEOUTS_MS: tuple[int, ...] = (1000, 2000, 3000, 5000)
SWEEP_ENGINE = "symbolica"
PARTS: tuple[str, ...] = (*METHODS, "cutoff-sweep")
BINARIES: dict[str, tuple[str, ...]] = {
    "baselines": ("sig_groebner",),
    "branching": ("branching",),
    "zeroing": ("variable_zeroing",),
    "cutoff-sweep": ("variable_zeroing",),
}
NAMES: dict[str, str] = {
    "m4gb": "M4GB",
    "f5": "F5",
    "gvw": "GVW",
    "symbolica": "F4",
    "singular-gb": "Singular",
    "builtin": "F4",
    "singular-gb-lex": "Singular",
    "m2-gb-lex": "M2",
    "singular-facstd": "facstd",
    "singular-primdecGTZ": "primdec",
    "m2-decompose": "M2",
}
LABELS: dict[str, str] = {
    "signed_distance_cycle_n12": "Signed distance, $n=12$",
    "signed_distance_cycle_n14": "Signed distance, $n=14$",
    "signed_distance_cycle_n16": "Signed distance, $n=16$",
    "overlapping_random_multiples_n4_p6_k2_seed31415": "Overlapping $(4,6,2)$",
    "overlapping_random_multiples_n6_p8_k3_seed31415": "Overlapping $(6,8,3)$",
    "overlapping_random_multiples_n8_p10_k3_seed31415": "Overlapping $(8,10,3)$",
    "affine_factor_hypergraph_n8_m6_k4_seed42": "Positive affine $(8,6,4)$",
    "affine_factor_hypergraph_n10_m8_k5_seed42": "Positive affine $(10,8,5)$",
    "affine_factor_hypergraph_n12_m10_k6_seed42": "Positive affine $(12,10,6)$",
    "gated_cyclic_5": "Gated Cyclic-5",
    "gated_cyclic_6": "Gated Cyclic-6",
    "gated_cyclic_7": "Gated Cyclic-7",
    "gated_katsura_6": "Gated Katsura-6",
    "gated_katsura_7": "Gated Katsura-7",
    "gated_katsura_8": "Gated Katsura-8",
    "gated_random_dense_quadratic_n5_seed137": "Gated dense, $n=5$",
    "gated_random_dense_quadratic_n6_seed137": "Gated dense, $n=6$",
    "gated_random_dense_quadratic_n7_seed137": "Gated dense, $n=7$",
    "affine_factor_hypergraph_n8_k4_seed42": "Square affine $(8,4)$",
    "affine_factor_hypergraph_n10_k5_seed42": "Square affine $(10,5)$",
}

Runner = Callable[[Instance, Path, str, int], dict[str, object]]
Formatter = Callable[[str, str, dict[str, object]], str]


def initial_report(part: str, instances: list[Instance], timeout: int) -> list[str]:
    methods = " | ".join(f"`{method}`" for method in METHODS[part])
    separator = " | ".join("---:" for _ in METHODS[part])
    header = {
        "baselines": [
            "# Standalone baseline matrix",
            "",
            f"Measurements of the standalone methods on the {len(instances)} generated systems",
            f"with a {timeout} s limit. Each cell records solver-reported time / wall time in",
            "seconds; decompositions add the largest component dimension `d`.",
            "`TO` marks an outer timeout and `ERR` a failed run.",
        ],
        "branching": [
            "# Factor-branching benchmark matrix",
            "",
            "This report contains fresh measurements of `symbolica/bin/branching.rs`",
            f"on the {len(instances)} generated algebraic benchmark systems. No physics system is",
            "included. Each cell records solver-reported total search time / wall time",
            "in seconds, followed by first-solution latency and the largest component",
            "dimension. `P a/b` denotes a partial run,",
            "`CAP a/b` a cover-limited run, and `TO` an outer timeout.",
        ],
        "zeroing": [
            "# Variable-zeroing benchmark matrix",
            "",
            "This report contains fresh measurements of `symbolica/bin/variable_zeroing.rs`",
            f"on the {len(instances)} generated algebraic benchmark systems. No physics system is",
            f"included. Every branch receives {ZEROING_BRANCH_TIMEOUT_MS} ms and the run a",
            f"{timeout} s budget. Each cell records solver-reported total time /",
            "wall time / first-solution time in seconds, then `N` recursion nodes,",
            "`E` hitting-set evaluations, `O/S` per-branch timeouts / zeroing splits,",
            "`C/U` components / unresolved branches. `P` marks a partial run,",
            "`TO` an outer timeout and `ERR` a failed run.",
        ],
    }[part]
    return [
        *header,
        "",
        f"| system | {methods} |",
        f"|---|{separator}|",
        *(f"| `{instance.name}` | " + " | ".join("NR" for _ in METHODS[part]) + " |" for instance in instances),
        "",
        "## Reproduction",
        "",
        "```sh",
        f"python3 scripts/baseline_table.py --part {part} --no-build --jobs 1",
        "```",
        "",
        "The script generates the benchmark systems in a temporary directory.",
        "It does not read the physics systems in `systems/`.",
    ]


class MatrixReport:
    """A Markdown matrix that is also the resumable result store: one row per system, one column per method."""

    def __init__(self, path: Path, methods: tuple[str, ...], initial: list[str]) -> None:
        self.path = path
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("\n".join(initial) + "\n", encoding="utf-8")
        self.lines = path.read_text(encoding="utf-8").splitlines()
        self.cells: dict[tuple[str, str], tuple[int, int]] = {}
        columns: list[str] = []
        for line_number, line in enumerate(self.lines):
            cells = self.split(line)
            if cells and cells[0] == "system":
                columns = [cell.strip("`") for cell in cells[1:]]
                continue
            if not cells or not columns or cells[0].startswith("---"):
                continue
            instance = cells[0].strip("`")
            for column, method in enumerate(columns, start=1):
                if method not in methods:
                    continue
                key = (instance, method)
                if key in self.cells:
                    raise ValueError(f"duplicate matrix result cell: {key}")
                self.cells[key] = (line_number, column)

    @staticmethod
    def split(line: str) -> list[str]:
        if not line.startswith("|"):
            return []
        return [cell.strip() for cell in line.strip().strip("|").split("|")]

    def value(self, instance: str, method: str) -> str:
        line_number, column = self.cells[(instance, method)]
        return self.split(self.lines[line_number])[column]

    def tasks(self, rerun_all: bool) -> list[tuple[str, str]]:
        return [key for key in self.cells if rerun_all or self.value(*key) == "NR"]

    def update(self, instance: str, method: str, value: str) -> None:
        line_number, column = self.cells[(instance, method)]
        cells = self.split(self.lines[line_number])
        cells[column] = value
        self.lines[line_number] = "| " + " | ".join(cells) + " |"
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text("\n".join(self.lines) + "\n", encoding="utf-8")
        temporary.replace(self.path)


def run_baseline(instance: Instance, path: Path, method: str, timeout: int) -> dict[str, object]:
    if method.endswith("-grevlex"):
        return run_direct(path, method.removesuffix("-grevlex"), timeout)
    return run_external(method, path, timeout)


def run_branching_cell(instance: Instance, path: Path, method: str, timeout: int) -> dict[str, object]:
    return run_branching(path, instance.solver_args, timeout, method.removeprefix("branching-"))


def run_zeroing_cell(instance: Instance, path: Path, method: str, timeout: int) -> dict[str, object]:
    return run_zeroing(path, instance.solver_args, timeout, method.removeprefix("zeroing-"))


def format_baseline(instance: str, method: str, row: dict[str, object]) -> str:
    status = str(row["status"])
    solve = float(str(row.get("solve_secs", 0)))
    wall = float(str(row.get("wall_secs", 0)))
    if status == "timeout":
        return f"TO {solve:.3f}/{wall:.3f}"
    if status in {"error", "tool_unavailable"}:
        return "ERR" if status == "error" else "UNAVAILABLE"

    value = f"{solve:.3f}/{wall:.3f}"
    dimension = row.get("largest_dimension", row.get("max_dim"))
    if method in DECOMPOSITION and dimension not in (None, ""):
        value += f"; d={dimension}"
    return value


def format_branching(instance: str, method: str, row: dict[str, object]) -> str:
    status = str(row["status"])
    solve = float(str(row.get("solve_secs", 0)))
    wall = float(str(row.get("wall_secs", 0)))
    if status == "timeout":
        return f"TO {solve:.3f}/{wall:.3f}"
    if status == "error":
        return "ERR"

    value = f"{solve:.3f}/{wall:.3f}"
    required = MATRIX_REQUIRED_COVERS[instance]
    if status == "partial":
        completed = int(str(row.get("satisfiable", 0))) + int(str(row.get("unsatisfiable", 0)))
        prefix = "P" if row.get("cover_timeouts") else "CAP"
        value = f"{prefix} {completed}/{required}; {value}"
    dimension = row.get("largest_dimension")
    first_solution = row.get("first_solution_secs")
    if first_solution is not None:
        value += f"; first={float(str(first_solution)):.3f}"
    if dimension is not None:
        value += f"; d={dimension}"
    return value


def format_zeroing(instance: str, method: str, row: dict[str, object]) -> str:
    status = str(row["status"])
    solve = float(str(row.get("solve_secs", 0)))
    wall = float(str(row.get("wall_secs", 0)))
    if status == "timeout":
        return f"TO {solve:.3f}/{wall:.3f}"
    if status == "error":
        return "ERR"
    first_solution = row.get("first_solution_secs")
    first = "--" if first_solution is None else f"{float(str(first_solution)):.3f}"
    prefix = "P " if status == "partial" else ""
    return (
        f"{prefix}{solve:.3f}/{wall:.3f}/{first}; N={row['nodes']}; E={row['evaluations']}; "
        f"O/S={row['branch_timeouts']}/{row['splits']}; "
        f"C/U={row['components']}/{row['unresolved']}"
    )


RUNNERS: dict[str, tuple[Runner, Formatter]] = {
    "baselines": (run_baseline, format_baseline),
    "branching": (run_branching_cell, format_branching),
    "zeroing": (run_zeroing_cell, format_zeroing),
}


def benchmark_matrix(
    part: str,
    instances: list[Instance],
    output: Path,
    timeout: int,
    jobs: int,
    rerun_all: bool,
    only: set[str],
) -> None:
    report = MatrixReport(MATRICES[part], METHODS[part], initial_report(part, instances, timeout))
    unknown = only - set(METHODS[part])
    if unknown:
        raise ValueError(f"unknown {part} method(s): {', '.join(sorted(unknown))}")
    tasks = [task for task in report.tasks(rerun_all) if not only or task[1] in only]
    by_name = {instance.name: instance for instance in instances}
    missing_systems = {instance for instance, _ in tasks} - by_name.keys()
    if missing_systems:
        raise ValueError(f"matrix report has unknown system(s): {', '.join(sorted(missing_systems))}")
    runner, formatter = RUNNERS[part]

    def execute(task: tuple[str, str]) -> dict[str, object]:
        instance = by_name[task[0]]
        return runner(instance, output / instance.path, task[1], timeout)

    print(f"running {len(tasks)} {part} checks with {jobs} workers")
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {executor.submit(execute, task): task for task in tasks}
        for completed, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            instance_name, method = futures[future]
            try:
                value = formatter(instance_name, method, future.result())
            except Exception as error:  # noqa: BLE001 -- preserve progress and make failures explicit.
                value = "ERR"
                print(f"[{completed}/{len(tasks)}] {instance_name} {method}: {error}")
            else:
                print(f"[{completed}/{len(tasks)}] {instance_name} {method}: {value}")
            report.update(instance_name, method, value)
    print(f"updated {MATRICES[part]}")


def write_sweep(results: dict[str, dict[str, dict[str, object]]]) -> None:
    temporary = SWEEP_FILE.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    temporary.replace(SWEEP_FILE)


def cutoff_sweep(instances: list[Instance], output: Path, timeout: int, jobs: int, rerun_all: bool) -> None:
    results: dict[str, dict[str, dict[str, object]]] = (
        json.loads(SWEEP_FILE.read_text(encoding="utf-8")) if SWEEP_FILE.exists() else {}
    )
    tasks = [
        (instance, branch_ms)
        for instance in instances
        if instance.name.startswith("gated_")
        for branch_ms in SWEEP_BRANCH_TIMEOUTS_MS
        if rerun_all or str(branch_ms) not in results.get(instance.name, {})
    ]

    def execute(task: tuple[Instance, int]) -> dict[str, object]:
        instance, branch_ms = task
        return run_zeroing(output / instance.path, instance.solver_args, timeout, SWEEP_ENGINE, branch_ms)

    print(f"running {len(tasks)} cutoff-sweep checks with {jobs} workers")
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {executor.submit(execute, task): task for task in tasks}
        for completed, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            instance, branch_ms = futures[future]
            row = future.result()
            results.setdefault(instance.name, {})[str(branch_ms)] = row
            write_sweep(results)
            print(f"[{completed}/{len(tasks)}] {instance.name} {branch_ms} ms: {format_zeroing(instance.name, '', row)}")
    print(f"updated {SWEEP_FILE}")


@dataclass(frozen=True)
class Entry:
    """The fastest completed method of one class on one system, as the table prints it."""

    method: str
    seconds: float
    shown: str


BRANCHING_CELL = re.compile(r"^(?P<total>[0-9.]+)/[0-9.]+; first=(?P<first>[0-9.]+)")
BASELINE_CELL = re.compile(r"^[0-9.]+/(?P<wall>[0-9.]+)")


def branching_entry(method: str, value: str) -> Entry | None:
    """A complete factor-branching run, timed by its total search time T and first SAT branch F."""
    match = BRANCHING_CELL.match(value)
    if match is None:
        return None
    return Entry(NAMES[method.removeprefix("branching-")], float(match["total"]), f"{match['total']}/{match['first']}")


def baseline_entry(method: str, value: str) -> Entry | None:
    """A complete standalone run, timed by its wall time."""
    match = BASELINE_CELL.match(value)
    if match is None:
        return None
    return Entry(NAMES[method.removesuffix("-grevlex")], float(match["wall"]), match["wall"])


def fastest(entries: list[Entry | None]) -> Entry | None:
    completed = [entry for entry in entries if entry is not None]
    return min(completed, key=lambda entry: entry.seconds) if completed else None


def show(entry: Entry | None, bold: bool) -> str:
    if entry is None:
        return "timeout"
    return f"\\textbf{{{entry.method}}} $\\mathbf{{{entry.shown}}}$" if bold else f"{entry.method} ${entry.shown}$"


def comparison_rows() -> list[str]:
    """The rows of tab:baseline-comparison: the fastest completed method per class, the fastest in bold."""
    branching = MatrixReport(MATRICES["branching"], METHODS["branching"], [])
    baselines = MatrixReport(MATRICES["baselines"], METHODS["baselines"], [])
    solved = [0, 0, 0]
    wins = [0, 0, 0]
    rows: list[str] = []
    for system, label in LABELS.items():
        best = [
            fastest([branching_entry(method, branching.value(system, method)) for method in METHODS["branching"]]),
            fastest([baseline_entry(method, baselines.value(system, method)) for method in DIRECT]),
            fastest([baseline_entry(method, baselines.value(system, method)) for method in DECOMPOSITION]),
        ]
        winner = min((entry.seconds for entry in best if entry is not None), default=None)
        bold = [entry is not None and entry.seconds == winner for entry in best]
        solved = [count + (entry is not None) for count, entry in zip(solved, best, strict=True)]
        wins = [count + won for count, won in zip(wins, bold, strict=True)]
        rows.append(f"{label} & " + " & ".join(show(e, b) for e, b in zip(best, bold, strict=True)) + " \\\\")
    rows.append("\\#systems solved & " + " & ".join(f"${count}$" for count in solved) + " \\\\")
    rows.append("\\#fastest (ties counted) & " + " & ".join(f"${count}$" for count in wins) + " \\\\")
    return rows


ZEROING_CELL = re.compile(
    r"(?P<partial>P )?(?P<total>[0-9.]+)/[0-9.]+/(?P<first>[0-9.]+|--); N=(?P<nodes>\d+); "
    r"E=(?P<evaluations>\d+); O/S=(?P<timeouts>\d+)/(?P<splits>\d+); "
    r"C/U=(?P<components>\d+)/(?P<unresolved>\d+)"
)


@dataclass(frozen=True)
class ZeroingRun:
    backend: str
    complete: bool
    total: float
    first: str
    nodes: int
    evaluations: int
    timeouts: int
    splits: int
    components: int
    unresolved: int

    def key(self) -> tuple[int, int, float]:
        return (0 if self.complete else 1, 0 if self.complete else self.unresolved, self.total)

    def row(self, label: str) -> str:
        first = f"{float(self.first):.3f}" if self.first != "--" else "--"
        timing = f"${self.total:.3f}/{first}$" if first != "--" else f"${self.total:.3f}$/--"
        result = "complete" if self.complete else "partial"
        return (
            f"{label} & {self.backend} & {timing} & {self.nodes} & {self.evaluations} & "
            f"${self.timeouts}/{self.splits}$ & ${self.components}/{self.unresolved}$ & {result} \\\\"
        )


def zeroing_run(method: str, value: str) -> ZeroingRun | None:
    match = ZEROING_CELL.search(value)
    if match is None:
        return None
    return ZeroingRun(
        backend=NAMES[method.removeprefix("zeroing-")],
        complete=match["partial"] is None,
        total=float(match["total"]),
        first=match["first"],
        nodes=int(match["nodes"]),
        evaluations=int(match["evaluations"]),
        timeouts=int(match["timeouts"]),
        splits=int(match["splits"]),
        components=int(match["components"]),
        unresolved=int(match["unresolved"]),
    )


def appendix_zeroing_rows() -> list[str]:
    """The rows of tab:appendix-zeroing: per system, the fastest complete variable-zeroing run or,
    if none completes, the partial run with the fewest unresolved branches (ties broken by time)."""
    zeroing = MatrixReport(MATRICES["zeroing"], METHODS["zeroing"], [])
    rows: list[str] = []
    for system, label in LABELS.items():
        runs = [run for method in METHODS["zeroing"] if (run := zeroing_run(method, zeroing.value(system, method)))]
        rows.append(min(runs, key=ZeroingRun.key).row(label) if runs else f"{label} & -- & -- \\\\")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--part", choices=PARTS)
    parser.add_argument("--timeout", type=int, default=60, help="seconds per run")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument(
        "--rerun-all",
        action="store_true",
        help="rerun populated cells too; by default only NR cells run",
    )
    parser.add_argument("--only", default="", help="comma-separated method labels of the part, e.g. m2-decompose")
    args = parser.parse_args()
    if args.part is None and not (args.generate_only or args.report_only):
        parser.error("--part is required unless --generate-only or --report-only")
    if args.timeout < 1:
        parser.error("--timeout must be at least 1")
    if args.jobs < 1:
        parser.error("--jobs must be at least 1")
    limit = subprocess_limit()
    if args.jobs > limit:
        print(f"capping --jobs from {args.jobs} to {limit} (available CPUs minus two)", file=sys.stderr)
        args.jobs = limit
    if args.generate_only and args.output_dir is None:
        parser.error("--generate-only requires --output-dir so generated systems persist")
    return args


def run_part(args: argparse.Namespace, root: Path) -> None:
    instances = generate_matrix(root)
    print(f"generated {len(instances)} benchmark instances in {root}")
    if args.generate_only:
        return
    if not args.no_build:
        build(*BINARIES[args.part])
    if args.part == "cutoff-sweep":
        cutoff_sweep(instances, root, args.timeout, args.jobs, args.rerun_all)
        return
    only = {item.strip() for item in args.only.split(",") if item.strip()}
    benchmark_matrix(args.part, instances, root, args.timeout, args.jobs, args.rerun_all, only)


def main() -> int:
    args = parse_args()
    if args.report_only:
        print("% tab:baseline-comparison", *comparison_rows(), "", "% tab:appendix-zeroing", *appendix_zeroing_rows(), sep="\n")
        return 0
    if args.output_dir is not None:
        run_part(args, args.output_dir)
        return 0
    with tempfile.TemporaryDirectory(prefix="baseline_benchmarks_") as directory:
        run_part(args, Path(directory))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
