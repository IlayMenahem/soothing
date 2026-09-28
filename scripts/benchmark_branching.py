#!/usr/bin/env python3
"""Generate and run the factor-branching or variable-zeroing benchmark matrix.

The benchmark reuses the 20 synthetic systems from ``benchmark_idos.py`` and
does not load any of the physics systems under ``systems/``. Results are
checkpointed atomically in a Markdown matrix, so interrupted runs can resume.

Examples:

    python3 scripts/benchmark_branching.py --jobs 1
    python3 scripts/benchmark_branching.py --algorithm zeroing --jobs 1
    python3 scripts/benchmark_branching.py --only branching-symbolica
    python3 scripts/benchmark_branching.py --rerun-all
    python3 scripts/benchmark_branching.py --generate-only --output-dir systems/branching
"""

from __future__ import annotations

import argparse
import concurrent.futures
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from benchmark_idos import (
    MATRIX_REQUIRED_COVERS,
    SYMBOLICA,
    Instance,
    generate_matrix,
    solver_args_with_engine,
    run,
    subprocess_limit,
)

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
BINARIES: dict[str, Path] = {
    "branching": SYMBOLICA / "target" / "release" / "branching",
    "zeroing": SYMBOLICA / "target" / "release" / "variable_zeroing",
}
MATRIX_REPORTS: dict[str, Path] = {
    "branching": RESULTS / "branching_engine_comparison.md",
    "zeroing": RESULTS / "zeroing_engine_comparison.md",
}
METHOD_TIMEOUT_SECS = 60
ZEROING_BRANCH_TIMEOUT_MS = 3000

SUMMARY_RE = re.compile(
    r"BRANCHING_SUMMARY total=(\d+) satisfiable=(\d+) "
    r"unsatisfiable=(\d+) timeouts=(\d+) complete=(true|false) "
    r"largest_dimension=(\d+|unknown) first_solution_secs=([0-9.]+|unknown) "
    r"elapsed_secs=([0-9.]+)"
)

ZEROING_SUMMARY_RE = re.compile(
    r"ZEROING_SUMMARY nodes=(\d+) covers=\d+ evaluations=(\d+) components=(\d+) "
    r"inconsistent=\d+ timeouts=(\d+) splits=(\d+) factorization_timeouts=\d+ "
    r"factorization_errors=\d+ unresolved=(\d+) complete=(true|false) "
    r"largest_depth=\d+ first_solution_secs=([0-9.]+|unknown) elapsed_secs=([0-9.]+)"
)

ENGINES = ("m4gb", "f5", "gvw", "symbolica", "singular-gb")


def matrix_methods(algorithm: str) -> tuple[str, ...]:
    return tuple(f"{algorithm}-{engine}" for engine in ENGINES)


def build() -> None:
    result = subprocess.run(  # noqa: S603
        ["cargo", "build", "--release", "--bin", "branching", "--bin", "variable_zeroing"],
        cwd=SYMBOLICA,
    )
    if result.returncode:
        raise SystemExit(result.returncode)


def run_branching(
    path: Path, branching_args: list[str], timeout: int, engine: str
) -> dict[str, object]:
    method = f"branching-{engine}"
    stdout, stderr, wall, killed, returncode = run(
        [
            str(BINARIES["branching"]),
            str(path),
            *solver_args_with_engine(branching_args, engine),
            "--timeout",
            str(timeout * 1000),
            "--summary",
        ],
        timeout + 2,
    )
    if killed:
        return {
            "method": method,
            "status": "timeout",
            "solve_secs": float(timeout),
            "wall_secs": wall,
        }

    match = SUMMARY_RE.search(stdout)
    if match is None:
        return {
            "method": method,
            "status": "error",
            "solve_secs": wall,
            "wall_secs": wall,
            "error": (stderr or stdout)[-600:],
        }

    (
        total,
        satisfiable,
        unsatisfiable,
        cover_timeouts,
        complete,
        largest_dimension,
        first_solution,
        solve,
    ) = match.groups()
    return {
        "method": method,
        "status": (
            "ok"
            if returncode == 0 and complete == "true" and int(cover_timeouts) == 0
            else "partial"
        ),
        "solve_secs": float(solve),
        "wall_secs": wall,
        "total": int(total),
        "satisfiable": int(satisfiable),
        "unsatisfiable": int(unsatisfiable),
        "cover_timeouts": int(cover_timeouts),
        "complete": complete == "true",
        "first_solution_secs": (
            None if first_solution == "unknown" else float(first_solution)
        ),
        "largest_dimension": (
            None if largest_dimension == "unknown" else int(largest_dimension)
        ),
    }


def run_zeroing(
    path: Path, solver_args: list[str], timeout: int, engine: str
) -> dict[str, object]:
    method = f"zeroing-{engine}"
    stdout, stderr, wall, killed, returncode = run(
        [
            str(BINARIES["zeroing"]),
            str(path),
            *solver_args_with_engine(solver_args, engine),
            "--timeout",
            str(ZEROING_BRANCH_TIMEOUT_MS),
            "--max-runtime",
            str(timeout * 1000),
            "--summary",
        ],
        timeout + 2,
    )
    if killed:
        return {
            "method": method,
            "status": "timeout",
            "solve_secs": float(timeout),
            "wall_secs": wall,
        }

    match = ZEROING_SUMMARY_RE.search(stdout)
    if match is None:
        return {
            "method": method,
            "status": "error",
            "solve_secs": wall,
            "wall_secs": wall,
            "error": (stderr or stdout)[-600:],
        }

    (
        nodes,
        evaluations,
        components,
        branch_timeouts,
        splits,
        unresolved,
        complete,
        first_solution,
        solve,
    ) = match.groups()
    return {
        "method": method,
        "status": "ok" if returncode == 0 and complete == "true" else "partial",
        "solve_secs": float(solve),
        "wall_secs": wall,
        "nodes": int(nodes),
        "evaluations": int(evaluations),
        "components": int(components),
        "branch_timeouts": int(branch_timeouts),
        "splits": int(splits),
        "unresolved": int(unresolved),
        "first_solution_secs": (
            None if first_solution == "unknown" else float(first_solution)
        ),
    }


def initial_report(algorithm: str, instances: list[Instance]) -> list[str]:
    methods = " | ".join(f"`{method}`" for method in matrix_methods(algorithm))
    separator = " | ".join("---:" for _ in matrix_methods(algorithm))
    header = {
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
            f"{METHOD_TIMEOUT_SECS} s budget. Each cell records solver-reported total time /",
            "wall time / first-solution time in seconds, then `N` recursion nodes,",
            "`E` hitting-set evaluations, `O/S` per-branch timeouts / zeroing splits,",
            "`C/U` components / unresolved branches. `P` marks a partial run,",
            "`TO` an outer timeout and `ERR` a failed run.",
        ],
    }[algorithm]
    rows = [
        *header,
        "",
        f"| system | {methods} |",
        f"|---|{separator}|",
    ]
    rows.extend(
        f"| `{instance.name}` | " + " | ".join("NR" for _ in matrix_methods(algorithm)) + " |"
        for instance in instances
    )
    rows.extend(
        [
            "",
            "## Reproduction",
            "",
            "```sh",
            f"python3 scripts/benchmark_branching.py --algorithm {algorithm} --no-build --jobs 1",
            "```",
            "",
            "The script generates the benchmark systems in a temporary directory.",
            "It does not read the physics systems in `systems/`.",
        ]
    )
    return rows


class MatrixReport:
    """Resumable Markdown matrix keyed by system and solver backend."""

    def __init__(self, path: Path, algorithm: str, instances: list[Instance]) -> None:
        self.path = path
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                "\n".join(initial_report(algorithm, instances)) + "\n", encoding="utf-8"
            )
        self.lines = path.read_text(encoding="utf-8").splitlines()
        self.cells: dict[tuple[str, str], tuple[int, int]] = {}
        methods: list[str] = []
        for line_number, line in enumerate(self.lines):
            cells = self._split(line)
            if cells and cells[0] == "system":
                methods = [cell.strip("`") for cell in cells[1:]]
                continue
            if not cells or not methods or cells[0].startswith("---"):
                continue
            instance = cells[0].strip("`")
            for column, method in enumerate(methods, start=1):
                if method not in matrix_methods(algorithm):
                    continue
                key = (instance, method)
                if key in self.cells:
                    raise ValueError(f"duplicate matrix result cell: {key}")
                self.cells[key] = (line_number, column)

    @staticmethod
    def _split(line: str) -> list[str]:
        if not line.startswith("|"):
            return []
        return [cell.strip() for cell in line.strip().strip("|").split("|")]

    def tasks(self, rerun_all: bool) -> list[tuple[str, str]]:
        tasks = []
        for key, (line_number, column) in self.cells.items():
            value = self._split(self.lines[line_number])[column]
            if rerun_all or value == "NR":
                tasks.append(key)
        return tasks

    def update(self, instance: str, method: str, value: str) -> None:
        line_number, column = self.cells[(instance, method)]
        cells = self._split(self.lines[line_number])
        cells[column] = value
        self.lines[line_number] = "| " + " | ".join(cells) + " |"
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text("\n".join(self.lines) + "\n", encoding="utf-8")
        temporary.replace(self.path)


def format_result(instance: str, row: dict[str, object]) -> str:
    status = str(row["status"])
    solve = float(row.get("solve_secs", 0))
    wall = float(row.get("wall_secs", 0))
    if status == "timeout":
        return f"TO {solve:.3f}/{wall:.3f}"
    if status == "error":
        return "ERR"

    value = f"{solve:.3f}/{wall:.3f}"
    required = MATRIX_REQUIRED_COVERS[instance]
    if status == "partial":
        completed = int(row.get("satisfiable", 0)) + int(
            row.get("unsatisfiable", 0)
        )
        prefix = "P" if row.get("cover_timeouts") else "CAP"
        value = f"{prefix} {completed}/{required}; {value}"
    dimension = row.get("largest_dimension")
    first_solution = row.get("first_solution_secs")
    if first_solution is not None:
        value += f"; first={float(first_solution):.3f}"
    if dimension is not None:
        value += f"; d={dimension}"
    return value


def format_zeroing_result(row: dict[str, object]) -> str:
    status = str(row["status"])
    solve = float(row.get("solve_secs", 0))
    wall = float(row.get("wall_secs", 0))
    if status == "timeout":
        return f"TO {solve:.3f}/{wall:.3f}"
    if status == "error":
        return "ERR"
    first_solution = row.get("first_solution_secs")
    first = "--" if first_solution is None else f"{float(first_solution):.3f}"
    prefix = "P " if status == "partial" else ""
    return (
        f"{prefix}{solve:.3f}/{wall:.3f}/{first}; N={row['nodes']}; E={row['evaluations']}; "
        f"O/S={row['branch_timeouts']}/{row['splits']}; "
        f"C/U={row['components']}/{row['unresolved']}"
    )


def benchmark_matrix(
    algorithm: str,
    instances: list[Instance],
    output: Path,
    report_path: Path,
    jobs: int,
    rerun_all: bool,
    only: str | None,
) -> None:
    report = MatrixReport(report_path, algorithm, instances)
    tasks = report.tasks(rerun_all)
    if only:
        selected = {item.strip() for item in only.split(",") if item.strip()}
        unknown = selected - set(matrix_methods(algorithm))
        if unknown:
            raise ValueError(f"unknown matrix method(s): {', '.join(sorted(unknown))}")
        tasks = [task for task in tasks if task[1] in selected]

    by_name = {instance.name: instance for instance in instances}
    missing_systems = {instance for instance, _ in tasks} - by_name.keys()
    if missing_systems:
        raise ValueError(
            f"matrix report has unknown system(s): {', '.join(sorted(missing_systems))}"
        )

    def execute(task: tuple[str, str]) -> tuple[str, str, dict[str, object]]:
        instance_name, method = task
        instance = by_name[instance_name]
        runner = run_branching if algorithm == "branching" else run_zeroing
        row = runner(
            output / instance.path,
            instance.solver_args,
            METHOD_TIMEOUT_SECS,
            method.removeprefix(f"{algorithm}-"),
        )
        return instance_name, method, row

    print(f"running {len(tasks)} {algorithm} checks with {jobs} workers")
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {executor.submit(execute, task): task for task in tasks}
        for completed, future in enumerate(
            concurrent.futures.as_completed(futures), start=1
        ):
            instance_name, method = futures[future]
            try:
                _, _, row = future.result()
                value = (
                    format_result(instance_name, row)
                    if algorithm == "branching"
                    else format_zeroing_result(row)
                )
            except Exception as error:  # Preserve progress and expose failures.
                value = "ERR"
                print(f"[{completed}/{len(tasks)}] {instance_name} {method}: {error}")
            else:
                print(f"[{completed}/{len(tasks)}] {instance_name} {method}: {value}")
            report.update(instance_name, method, value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=sorted(BINARIES), default="branching")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument(
        "--report",
        type=Path,
        help="resumable Markdown result matrix (default: per-algorithm file in results/)",
    )
    parser.add_argument(
        "--rerun-all",
        action="store_true",
        help="rerun populated matrix cells too; by default only NR cells run",
    )
    parser.add_argument(
        "--only",
        help=(
            "comma-separated exact method labels, such as "
            "branching-symbolica,zeroing-singular-gb"
        ),
    )
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be at least 1")
    limit = subprocess_limit()
    if args.jobs > limit:
        print(
            f"capping --jobs from {args.jobs} to {limit} (available CPUs minus two)",
            file=sys.stderr,
        )
        args.jobs = limit
    if args.generate_only and args.output_dir is None:
        parser.error("--generate-only requires --output-dir so generated systems persist")
    if args.report is None:
        args.report = MATRIX_REPORTS[args.algorithm]
    return args


def main() -> int:
    args = parse_args()
    if not args.generate_only and not args.no_build:
        build()

    temporary: tempfile.TemporaryDirectory[str] | None = None
    if args.output_dir is None:
        temporary = tempfile.TemporaryDirectory(prefix="branching_benchmarks_")
        root = Path(temporary.name)
    else:
        root = args.output_dir

    try:
        instances = generate_matrix(root)
        print(f"generated {len(instances)} non-physics benchmark instances in {root}")
        if not args.generate_only:
            benchmark_matrix(
                args.algorithm,
                instances,
                root,
                args.report,
                args.jobs,
                args.rerun_all,
                args.only,
            )
            print(f"updated {args.report}")
    finally:
        if temporary is not None:
            temporary.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
