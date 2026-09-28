#!/usr/bin/env python3
"""Generate the 20-system benchmark matrix and run the baseline solvers.

The generated systems are shared with ``benchmark_branching.py``. Baseline
results are checkpointed atomically in one Markdown matrix.

Examples:

    python3 scripts/benchmark_idos.py --timeout 60 --jobs 1
    python3 scripts/benchmark_idos.py --only singular-gb-lex,m2-decompose
    python3 scripts/benchmark_idos.py --rerun-all
    python3 scripts/benchmark_idos.py --generate-only --output-dir systems/benchmarks
"""

from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import json
import os
import random
import re
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SYMBOLICA = REPO / "symbolica"
RESULTS = REPO / "results"
BASELINES = Path(__file__).with_name("baselines")
SIG = SYMBOLICA / "target" / "release" / "sig_groebner"
MATRIX_REPORT = RESULTS / "idos_engine_comparison.md"


def subprocess_limit() -> int:
    """Maximum concurrent child processes, leaving two CPUs responsive."""
    affinity = getattr(os, "sched_getaffinity", None)
    cores = len(affinity(0)) if affinity is not None else (os.cpu_count() or 1)
    return max(1, cores - 2)


FINISHED_RE = re.compile(
    r"^\s*(\w+) finished in ([0-9.]+)(ns|µs|ms|s)", re.MULTILINE
)


@dataclass(frozen=True)
class Instance:
    suite: str
    name: str
    path: str
    variables: int
    equations: int
    solver_args: list[str]
    description: str
    tier: str | None = None
    raw_factor_choices: int | None = None
    expected_largest_component: str | None = None


def write_system(path: Path, polynomials: list[str]) -> None:
    path.write_text(
        "{\n  "
        + ",\n  ".join(f"0 -> ({polynomial})" for polynomial in polynomials)
        + "\n}\n",
        encoding="utf-8",
    )


def cyclic_polynomials(order: int) -> list[str]:
    if order < 1:
        raise ValueError("cyclic order must be at least 1")
    variables = [f"c{index}" for index in range(order)]
    polynomials = []
    for degree in range(1, order):
        terms = [
            "*".join(
                variables[(start + offset) % order] for offset in range(degree)
            )
            for start in range(order)
        ]
        polynomials.append(" + ".join(terms))
    polynomials.append("*".join(variables) + " - 1")
    return polynomials


def gated_cyclic(output: Path, order: int) -> Instance:
    path = output / f"gated_cyclic_{order}.txt"
    write_system(path, [f"gate*({polynomial})" for polynomial in cyclic_polynomials(order)])
    return Instance(
        suite="adversarial",
        name=path.stem,
        tier="exhaustive_covers",
        path=path.name,
        variables=order + 1,
        equations=order,
        solver_args=["--gb", "m4gb"],
        description=(
            "I = gate*Cyclic-n. Two minimal hitting sets: the hyperplane "
            "{gate} and the full residual Cyclic-n ideal."
        ),
    )


def katsura_polynomials(order: int) -> list[str]:
    if order < 1:
        raise ValueError("Katsura order must be at least 1")
    variables = [f"q{index}" for index in range(order + 1)]
    normalization = (
        f"{variables[0]} + "
        + " + ".join(f"2*{variable}" for variable in variables[1:])
        + " - 1"
    )
    polynomials = [normalization]
    for target in range(order):
        terms = []
        for index in range(-order, order + 1):
            other = target - index
            if abs(other) <= order:
                terms.append(
                    f"{variables[abs(index)]}*{variables[abs(other)]}"
                )
        polynomials.append(" + ".join(terms) + f" - {variables[target]}")
    return polynomials


def gated_katsura(output: Path, order: int) -> Instance:
    path = output / f"gated_katsura_{order}.txt"
    write_system(path, [f"k_gate*({polynomial})" for polynomial in katsura_polynomials(order)])
    return Instance(
        suite="adversarial",
        name=path.stem,
        tier="exhaustive_covers",
        path=path.name,
        variables=order + 2,
        equations=order + 1,
        solver_args=["--gb", "m4gb"],
        description=(
            "I = k_gate*Katsura-n. Two minimal hitting sets: the common "
            "hyperplane {k_gate} and the full residual Katsura-n ideal."
        ),
    )


def random_dense_quadratics(
    variables: int, seed: int, *, count: int | None = None
) -> list[str]:
    if variables < 2:
        raise ValueError("dense quadratic systems require at least two variables")
    generator = random.Random(seed)
    names = [f"r{index}" for index in range(variables)]
    polynomials = []
    for _ in range(variables if count is None else count):
        terms = []
        for index, left in enumerate(names):
            for right in names[index:]:
                terms.append(
                    f"{generator.choice((-3, -2, -1, 1, 2, 3))}*{left}*{right}"
                )
        for variable in names:
            terms.append(
                f"{generator.choice((-3, -2, -1, 1, 2, 3))}*{variable}"
            )
        terms.append(str(generator.choice((-3, -2, -1, 1, 2, 3))))
        polynomials.append(" + ".join(terms))
    return polynomials


def gated_random_dense_quadratic(
    output: Path, variables: int, seed: int
) -> Instance:
    path = output / f"gated_random_dense_quadratic_n{variables}_seed{seed}.txt"
    write_system(
        path,
        [
            f"q_gate*({polynomial})"
            for polynomial in random_dense_quadratics(variables, seed)
        ],
    )
    return Instance(
        suite="adversarial",
        name=path.stem,
        tier="exhaustive_covers",
        path=path.name,
        variables=variables + 1,
        equations=variables,
        solver_args=["--gb", "m4gb"],
        description=(
            "A selector gates a seeded generic dense quadratic complete intersection. "
            "Hitting sets: the common hyperplane and the full residual system."
        ),
    )


def overlapping_random_multiples(
    output: Path,
    variables: int,
    factor_pool: int,
    factors_per_equation: int,
    seed: int,
) -> Instance:
    if (
        variables < 3
        or factor_pool <= factors_per_equation
        or factors_per_equation < 2
    ):
        raise ValueError(
            "require at least three variables and 2 <= factors per equation < pool"
        )
    generator = random.Random(seed)
    for _ in range(100):
        groups = [
            tuple(sorted(generator.sample(range(factor_pool), factors_per_equation)))
            for _ in range(variables)
        ]
        used = set(itertools.chain.from_iterable(groups))
        common = set(groups[0]).intersection(*groups[1:])
        if len(set(groups)) == variables and len(used) == factor_pool and not common:
            break
    else:
        raise ValueError("could not generate a covering non-universal factor pattern")

    covers: list[tuple[int, ...]] = []
    for size in range(1, factor_pool + 1):
        for candidate in itertools.combinations(range(factor_pool), size):
            chosen = set(candidate)
            if any(set(cover).issubset(chosen) for cover in covers):
                continue
            if all(chosen.intersection(group) for group in groups):
                covers.append(candidate)

    factors = random_dense_quadratics(variables, seed + 1, count=factor_pool)
    polynomials = [
        "*".join(f"({factors[index]})" for index in group) for group in groups
    ]
    path = output / (
        f"overlapping_random_multiples_n{variables}_p{factor_pool}"
        f"_k{factors_per_equation}_seed{seed}.txt"
    )
    write_system(path, polynomials)
    return Instance(
        suite="positive-dimensional",
        name=path.stem,
        path=path.name,
        variables=variables,
        equations=variables,
        solver_args=["--no-interreduce", "--gb", "symbolica"],
        description=(
            f"Each equation is the product of {factors_per_equation} factors "
            f"selected from a pool of {factor_pool} seeded dense quadratics; "
            "factors overlap between equations but none divides every equation."
        ),
        expected_largest_component=(
            f"The {len(covers)} minimal covers contain between "
            f"{min(map(len, covers))} and {max(map(len, covers))} random quadratics."
        ),
    )


def affine_factor_hypergraph(
    output: Path,
    variables: int,
    degree: int,
    seed: int,
    *,
    equations: int | None = None,
    suite: str = "adversarial",
) -> Instance:
    equation_count = variables if equations is None else equations
    if (
        variables < 4
        or not 1 <= equation_count <= variables
        or not 2 <= degree <= variables
    ):
        raise ValueError(
            "require affine variables >= 4, 1 <= equations <= variables, and "
            "2 <= degree <= variables"
        )
    generator = random.Random(seed)
    forms = []
    for index in range(2 * variables):
        left = index % variables
        middle = (index + 1) % variables
        right = (index + 3) % variables
        alpha, beta = (index % 5) + 1, (index % 3) + 1
        constant = generator.randint(1, 11)
        forms.append(
            f"(u{left} + {alpha}*u{middle} - {beta}*u{right} - {constant})"
        )
    polynomials = []
    for index in range(equation_count):
        chosen = generator.sample(range(2 * variables), degree)
        if index not in chosen:
            chosen[0] = index
        polynomials.append("*".join(forms[choice] for choice in chosen))
    dimension_label = (
        ""
        if equation_count == variables
        else f"_m{equation_count}"
    )
    path = output / (
        f"affine_factor_hypergraph_n{variables}{dimension_label}_k{degree}_seed{seed}.txt"
    )
    write_system(path, polynomials)
    return Instance(
        suite=suite,
        name=path.stem,
        tier="exhaustive_covers" if suite == "adversarial" else None,
        path=path.name,
        variables=variables,
        equations=equation_count,
        solver_args=[
            *(["--no-interreduce"] if suite == "positive-dimensional" else []),
            "--gb",
            "symbolica",
        ],
        description=(
            "Overlapping sparse affine factors; "
            "each candidate is linear while the expanded full ideal is dense."
        ),
        expected_largest_component=(
            f"Every cover has at most {equation_count} affine-linear generators in "
            f"{variables} variables, so every component has dimension at least "
            f"{variables - equation_count}."
            if suite == "positive-dimensional"
            else None
        ),
    )


def signed_distance_cycle(output: Path, vertices: int) -> Instance:
    if vertices < 4 or vertices % 2:
        raise ValueError("use an even signed-distance cycle of size at least four")
    polynomials = []
    for vertex in range(vertices):
        following = (vertex + 1) % vertices
        difference = f"z{following} - z{vertex}"
        polynomials.append(f"({difference} - 1)*({difference} + 1)")
    path = output / f"signed_distance_cycle_n{vertices}.txt"
    write_system(path, polynomials)
    return Instance(
        suite="positive-dimensional",
        name=path.stem,
        path=path.name,
        variables=vertices,
        equations=vertices,
        raw_factor_choices=1 << vertices,
        solver_args=["--gb", "m4gb"],
        description=(
            "Every edge of a closed one-dimensional linkage has signed length +1 or "
            "-1. Balanced choices give translation lines."
        ),
        expected_largest_component=(
            "A dimension-one subvariety; an interreduced cover may group balanced lines."
        ),
    )


def generate_matrix(output: Path) -> list[Instance]:
    """Generate each system in the consolidated comparison matrix exactly once."""
    output.mkdir(parents=True, exist_ok=True)
    instances = [
        *(signed_distance_cycle(output, size) for size in (8, 10, 12)),
        overlapping_random_multiples(output, 4, 6, 2, 31415),
        overlapping_random_multiples(output, 6, 8, 3, 31415),
        overlapping_random_multiples(output, 8, 10, 3, 31415),
        affine_factor_hypergraph(
            output,
            8,
            4,
            42,
            equations=6,
            suite="positive-dimensional",
        ),
        affine_factor_hypergraph(
            output,
            10,
            5,
            42,
            equations=8,
            suite="positive-dimensional",
        ),
        affine_factor_hypergraph(
            output,
            12,
            6,
            42,
            equations=10,
            suite="positive-dimensional",
        ),
        gated_cyclic(output, 5),
        gated_cyclic(output, 6),
        gated_cyclic(output, 7),
        gated_katsura(output, 6),
        gated_katsura(output, 7),
        gated_katsura(output, 8),
        gated_random_dense_quadratic(output, 5, 137),
        gated_random_dense_quadratic(output, 6, 137),
        gated_random_dense_quadratic(output, 7, 137),
        affine_factor_hypergraph(output, 8, 4, 42),
        affine_factor_hypergraph(output, 10, 5, 42),
    ]
    manifest = {"schema": 1, "instances": [asdict(instance) for instance in instances]}
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return instances


def run(
    argv: list[str], timeout: float, cwd: Path | None = None
) -> tuple[str, str, float, bool, int]:
    start = time.perf_counter()
    process = subprocess.Popen(  # noqa: S603 -- commands are constructed locally.
        argv,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        return stdout, stderr, time.perf_counter() - start, False, process.returncode
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
        return stdout, stderr, timeout, True, -signal.SIGKILL


def build() -> None:
    result = subprocess.run(  # noqa: S603
        ["cargo", "build", "--release", "--bin", "sig_groebner"],
        cwd=SYMBOLICA,
    )
    if result.returncode:
        raise SystemExit(result.returncode)


def parse_baseline(
    stdout: str, fallback: str, wall: float, timed_out: bool
) -> dict[str, object]:
    if timed_out:
        return {
            "method": fallback,
            "status": "timeout",
            "solve_secs": wall,
            "wall_secs": wall,
        }
    for line in reversed(stdout.splitlines()):
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and "status" in parsed:
            parsed["method"] = parsed.pop("baseline", fallback)
            parsed["solve_secs"] = parsed.pop("elapsed_secs", wall)
            parsed["wall_secs"] = wall
            return parsed
    return {
        "method": fallback,
        "status": "error",
        "solve_secs": wall,
        "wall_secs": wall,
        "error": "no JSON result",
    }


def solver_args_with_engine(solver_args: list[str], engine: str) -> list[str]:
    args = solver_args.copy()
    try:
        gb_index = args.index("--gb")
    except ValueError:
        args.extend(["--gb", engine])
    else:
        if gb_index + 1 == len(args):
            raise ValueError("--gb in instance arguments has no engine")
        args[gb_index + 1] = engine
    return args


def run_direct(path: Path, algorithm: str, timeout: int) -> dict[str, object]:
    label = f"{algorithm}-grevlex"
    stdout, stderr, wall, killed, returncode = run(
        [
            str(SIG),
            str(path),
            "--algo",
            algorithm,
            "--order",
            "grevlex",
            "--timeout-secs",
            str(timeout),
            "--summary",
        ],
        timeout + 2,
    )
    if killed or "TIMEOUT" in stdout:
        return {
            "method": label,
            "status": "timeout",
            "solve_secs": float(timeout),
            "wall_secs": wall,
        }
    match = FINISHED_RE.search(stdout)
    if returncode == 0 and match:
        value, unit = match.group(2), match.group(3)
        scale = {"ns": 1e-9, "µs": 1e-6, "ms": 1e-3, "s": 1.0}[unit]
        return {
            "method": label,
            "status": "ok",
            "solve_secs": float(value) * scale,
            "wall_secs": wall,
        }
    return {
        "method": label,
        "status": "error",
        "solve_secs": wall,
        "wall_secs": wall,
        "error": (stderr or stdout)[-600:],
    }


EXTERNAL_METHODS = (
    ("singular-gb-lex", "singular", "system_to_singular.py", ["--task", "gb"]),
    (
        "singular-facstd",
        "singular",
        "system_to_singular.py",
        ["--task", "facstd"],
    ),
    (
        "singular-primdecGTZ",
        "singular",
        "system_to_singular.py",
        ["--task", "decompose"],
    ),
    (
        "m2-gb-lex",
        "macaulay2",
        "system_to_macaulay2.py",
        ["--task", "gb"],
    ),
    (
        "m2-decompose",
        "macaulay2",
        "system_to_macaulay2.py",
        ["--task", "decompose"],
    ),
    (
        "homotopy-jl",
        "homotopy",
        "system_to_homotopy_jl.py",
        [],
    ),
)


def run_external(
    path: Path, timeout: int, selected: set[str]
) -> list[dict[str, object]]:
    run_all = "external" in selected
    rows = []
    for label, group, script, extra in EXTERNAL_METHODS:
        if not run_all and group not in selected and label not in selected:
            continue
        stdout, _, wall, killed, _ = run(
            [
                sys.executable,
                str(BASELINES / script),
                str(path),
                *extra,
                "--timeout-secs",
                str(timeout),
            ],
            timeout + 2,
        )
        row = parse_baseline(stdout, label, wall, killed)
        if killed:
            row["solve_secs"] = float(timeout)
        rows.append(row)
    return rows


MATRIX_METHODS = {
    "m4gb-grevlex",
    "f5-grevlex",
    "gvw-grevlex",
    "builtin-grevlex",
    *(label for label, _, _, _ in EXTERNAL_METHODS),
}

DECOMPOSITION_METHODS = {
    "singular-facstd",
    "singular-primdecGTZ",
    "m2-decompose",
    "homotopy-jl",
}

MATRIX_REQUIRED_COVERS = {
    "signed_distance_cycle_n8": 8,
    "signed_distance_cycle_n10": 8,
    "signed_distance_cycle_n12": 32,
    "overlapping_random_multiples_n4_p6_k2_seed31415": 4,
    "overlapping_random_multiples_n6_p8_k3_seed31415": 16,
    "overlapping_random_multiples_n8_p10_k3_seed31415": 28,
    "affine_factor_hypergraph_n8_m6_k4_seed42": 63,
    "affine_factor_hypergraph_n10_m8_k5_seed42": 239,
    "affine_factor_hypergraph_n12_m10_k6_seed42": 833,
    "gated_cyclic_5": 2,
    "gated_cyclic_6": 2,
    "gated_cyclic_7": 2,
    "gated_katsura_6": 2,
    "gated_katsura_7": 2,
    "gated_katsura_8": 2,
    "gated_random_dense_quadratic_n5_seed137": 2,
    "gated_random_dense_quadratic_n6_seed137": 2,
    "gated_random_dense_quadratic_n7_seed137": 2,
    "affine_factor_hypergraph_n8_k4_seed42": 31,
    "affine_factor_hypergraph_n10_k5_seed42": 304,
}


class MatrixReport:
    """A Markdown matrix that is also the resumable, deduplicated result store."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.lines = path.read_text(encoding="utf-8").splitlines()
        self.cells: dict[tuple[str, str], tuple[int, int]] = {}
        methods: list[str] = []
        for line_number, line in enumerate(self.lines):
            cells = self._split(line)
            if cells and cells[0] == "system":
                methods = [
                    cell.strip("`") if cell.strip("`") in MATRIX_METHODS else ""
                    for cell in cells[1:]
                ]
                continue
            if not cells or not methods or cells[0].startswith("---"):
                continue
            instance = cells[0].strip("`")
            for column, method in enumerate(methods, start=1):
                if not method:
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


def run_matrix_method(
    instance: Instance, path: Path, method: str, timeout: int
) -> dict[str, object]:
    if method.endswith("-grevlex"):
        return run_direct(path, method.removesuffix("-grevlex"), timeout)
    rows = run_external(path, timeout, {method})
    if len(rows) != 1:
        raise RuntimeError(f"{method} selected {len(rows)} external methods")
    return rows[0]


def format_matrix_result(
    instance: str, method: str, row: dict[str, object]
) -> str:
    status = str(row["status"])
    solve = float(row.get("solve_secs", 0))
    wall = float(row.get("wall_secs", 0))
    if status == "timeout":
        return f"TO {solve:.3f}/{wall:.3f}"
    if status in {"error", "tool_unavailable"}:
        return "ERR" if status == "error" else "UNAVAILABLE"

    value = f"{solve:.3f}/{wall:.3f}"
    dimension = row.get("largest_dimension", row.get("max_dim"))
    if method in DECOMPOSITION_METHODS and dimension not in (None, ""):
        value += f"; d={dimension}"
    return value


def benchmark_matrix(
    instances: list[Instance],
    output: Path,
    report_path: Path,
    timeout: int,
    jobs: int,
    rerun_all: bool,
    only: str | None,
) -> None:
    report = MatrixReport(report_path)
    tasks = report.tasks(rerun_all)
    if only:
        selected = {item.strip() for item in only.split(",") if item.strip()}
        unknown = selected - MATRIX_METHODS
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
        row = run_matrix_method(instance, output / instance.path, method, timeout)
        return instance_name, method, row

    print(f"running {len(tasks)} matrix checks with {jobs} workers")
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {executor.submit(execute, task): task for task in tasks}
        for completed, future in enumerate(
            concurrent.futures.as_completed(futures), start=1
        ):
            instance_name, method = futures[future]
            try:
                _, _, row = future.result()
                value = format_matrix_result(instance_name, method, row)
            except Exception as error:  # Preserve progress and make failures explicit.
                value = "ERR"
                print(f"[{completed}/{len(tasks)}] {instance_name} {method}: {error}")
            else:
                print(f"[{completed}/{len(tasks)}] {instance_name} {method}: {value}")
            report.update(instance_name, method, value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=int, default=10)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument(
        "--report",
        type=Path,
        default=MATRIX_REPORT,
        help="resumable Markdown result matrix",
    )
    parser.add_argument(
        "--rerun-all",
        action="store_true",
        help="rerun populated matrix cells too; by default only NR cells run",
    )
    parser.add_argument(
        "--only",
        help=(
            "comma-separated exact matrix method labels, such as "
            "singular-gb-lex,m2-decompose"
        ),
    )
    args = parser.parse_args()
    if args.timeout < 1:
        parser.error("--timeout must be at least 1")
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
    return args


def main() -> int:
    args = parse_args()
    if not args.generate_only and not args.no_build:
        build()

    temporary: tempfile.TemporaryDirectory[str] | None = None
    if args.output_dir is None:
        temporary = tempfile.TemporaryDirectory(prefix="baseline_benchmarks_")
        root = Path(temporary.name)
    else:
        root = args.output_dir

    try:
        instances = generate_matrix(root)
        print(f"generated {len(instances)} benchmark instances in {root}")
        if not args.generate_only:
            benchmark_matrix(
                instances,
                root,
                args.report,
                args.timeout,
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
