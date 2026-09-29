#!/usr/bin/env python3
"""Generate and benchmark application systems with complementarity structure.

Every equation has the form (selector factors) * (residual), so the minimal
hitting sets enumerate the supports of the solutions:

* totally mixed and partially mixed Nash equilibria of n-player binary games
  (generic payoffs and degenerate payoffs in {-1, 0, 1});
* critical points of a box-constrained quadratic program on every face;
* equilibria of a Lotka--Volterra community on every species support.

Results are appended to a resumable JSON-lines file, and ``--report`` renders
the best completed method per class as a Markdown table.

Examples:

    python3 scripts/benchmark_applications.py --jobs 4
    python3 scripts/benchmark_applications.py --only branching-f5,homotopy-jl
    python3 scripts/benchmark_applications.py --report-only
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import itertools
import json
import random
import subprocess
import tempfile
import threading
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

from benchmark_branching import run_branching, run_zeroing
from benchmark_idos import (
    EXTERNAL_METHODS,
    RESULTS,
    SIG,
    SYMBOLICA,
    Instance,
    run,
    run_direct,
    run_external,
    subprocess_limit,
    write_system,
)

RESULTS_FILE = RESULTS / "applications_results.jsonl"
REPORT_FILE = RESULTS / "applications_comparison.md"
TIMEOUT_SECS = 60
ENGINES: tuple[str, ...] = (
    "m4gb",
    "f5",
    "gvw",
    "symbolica",
    "singular-gb",
    "singular-facstd",
)
DIRECT_ALGORITHMS: tuple[str, ...] = ("m4gb", "f5", "gvw", "builtin")
EXTERNAL_LABELS: tuple[str, ...] = tuple(label for label, _, _, _ in EXTERNAL_METHODS)
METHODS: tuple[str, ...] = (
    *(f"branching-{engine}" for engine in ENGINES),
    *(f"zeroing-{engine}" for engine in ENGINES),
    *(f"{algorithm}-grevlex" for algorithm in DIRECT_ALGORITHMS),
    *EXTERNAL_LABELS,
)
METHOD_CLASSES: dict[str, tuple[str, ...]] = {
    "factor branching": tuple(f"branching-{engine}" for engine in ENGINES),
    "variable zeroing": tuple(f"zeroing-{engine}" for engine in ENGINES),
    "direct GB": (
        *(f"{algorithm}-grevlex" for algorithm in DIRECT_ALGORITHMS),
        "singular-gb-lex",
        "m2-gb-lex",
    ),
    "decomposition": ("singular-facstd", "singular-primdecGTZ", "m2-decompose"),
    "numerical": ("homotopy-jl",),
}


def mixed_profile_product(players: list[int], profile: tuple[int, ...]) -> str:
    """Probability that the players in ``players`` play the pure ``profile``."""
    return "*".join(
        f"p{player}" if action else f"(1-p{player})"
        for player, action in zip(players, profile, strict=True)
    ) or "1"


def nash_game(output: Path, players: int, seed: int, payoff_bound: int) -> Instance:
    """Nash equilibria of a binary game with integer payoff gaps in [-bound, bound].

    Player i mixes with probability p_i and is either pure (p_i in {0, 1}) or
    indifferent, i.e. its payoff gap, multilinear in the other players'
    probabilities, vanishes.
    """
    rng = random.Random(seed)
    polynomials = []
    for player in range(players):
        others = [other for other in range(players) if other != player]
        gap = " + ".join(
            f"({rng.randint(-payoff_bound, payoff_bound)})*"
            f"{mixed_profile_product(others, profile)}"
            for profile in itertools.product((0, 1), repeat=players - 1)
        )
        polynomials.append(f"p{player}*(p{player}-1)*({gap})")
    kind = "generic" if payoff_bound > 1 else "degenerate"
    path = output / f"nash_{kind}_n{players}.txt"
    write_system(path, polynomials)
    return Instance(
        suite="applications",
        name=path.stem,
        path=path.name,
        variables=players,
        equations=players,
        raw_factor_choices=3**players,
        solver_args=[],
        description=f"Nash equilibria of a {players}-player binary game, {kind} payoffs.",
    )


def box_qp(output: Path, variables: int, seed: int, density: float) -> Instance:
    """Critical points of x^T Q x / 2 + c^T x on every face of the unit box."""
    rng = random.Random(seed)
    upper = {
        (row, column): rng.randint(-50, 50)
        for row in range(variables)
        for column in range(row, variables)
        if row == column or rng.random() < density
    }
    hessian = {**upper, **{(column, row): value for (row, column), value in upper.items()}}
    polynomials = []
    for row in range(variables):
        gradient = " + ".join(
            f"({hessian[row, column]})*x{column}"
            for column in range(variables)
            if hessian.get((row, column), 0)
        )
        polynomials.append(f"x{row}*(x{row}-1)*({gradient} + ({rng.randint(-50, 50)}))")
    path = output / f"box_qp_n{variables}.txt"
    write_system(path, polynomials)
    return Instance(
        suite="applications",
        name=path.stem,
        path=path.name,
        variables=variables,
        equations=variables,
        raw_factor_choices=3**variables,
        solver_args=[],
        description=f"Box-QP critical points on all faces, n={variables}.",
    )


def lotka_volterra(output: Path, species: int, seed: int, density: float) -> Instance:
    """Equilibria x_i (r_i + sum_j a_ij x_j) = 0 of a random Lotka--Volterra community."""
    rng = random.Random(seed)
    polynomials = []
    for row in range(species):
        interactions = [f"({-rng.randint(1, 5)})*x{row}"] + [
            f"({rng.randint(-5, 5)})*x{column}"
            for column in range(species)
            if column != row and rng.random() < density
        ]
        polynomials.append(f"x{row}*({rng.randint(1, 5)} + {' + '.join(interactions)})")
    path = output / f"lotka_volterra_n{species}.txt"
    write_system(path, polynomials)
    return Instance(
        suite="applications",
        name=path.stem,
        path=path.name,
        variables=species,
        equations=species,
        raw_factor_choices=2**species,
        solver_args=[],
        description=f"Lotka--Volterra equilibria on all supports, {species} species.",
    )


def generate_applications(output: Path) -> list[Instance]:
    output.mkdir(parents=True, exist_ok=True)
    instances = [
        *(nash_game(output, players, 1000 + players, 9) for players in (4, 5, 6)),
        *(nash_game(output, players, 2000 + players, 1) for players in (4, 5, 6)),
        *(box_qp(output, variables, 3000 + variables, 0.5) for variables in (6, 7, 8)),
        *(lotka_volterra(output, species, 4000 + species, 0.4) for species in (8, 10, 12)),
    ]
    manifest = {"schema": 1, "instances": [asdict(instance) for instance in instances]}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return instances


def builtin_basis(path: Path, timeout: int) -> list[str] | None:
    """Symbolica's F4 output basis, or None if it did not finish."""
    stdout, _, _, killed, returncode = run(
        [str(SIG), str(path), "--algo", "builtin", "--order", "grevlex",
         "--timeout-secs", str(timeout)],
        timeout + 2,
    )
    lines = stdout.splitlines()
    starts = [index for index, line in enumerate(lines) if "basis size" in line]
    if killed or returncode or not starts:
        return None
    return [line.strip() for line in lines[starts[0] + 1 :] if line.startswith("\t")]


def leading_ideal_dimension(basis: list[str], variables: list[str], timeout: int) -> int | None:
    """Krull dimension of the leading ideal of ``basis``, taken as a standard basis.

    A Groebner basis has the dimension of its ideal, so a mismatch with the
    exact dimension shows that the basis is not a Groebner basis.
    """
    program = (
        f"ring r=0,({','.join(variables)}),dp;\n"
        f"ideal G={','.join(basis)};\n"
        'attrib(G,"isSB",1);\n'
        'print("DIM " + string(dim(G)));\nquit;\n'
    )
    try:
        stdout = subprocess.run(  # noqa: S603
            ["Singular", "-q"], input=program, capture_output=True, text=True, timeout=timeout
        ).stdout
    except subprocess.TimeoutExpired:
        return None
    return next(
        (int(line.split()[1]) for line in stdout.splitlines() if line.startswith("DIM ")),
        None,
    )


def variable_names(instance: Instance) -> list[str]:
    prefix = "p" if instance.name.startswith("nash") else "x"
    return [f"{prefix}{index}" for index in range(instance.variables)]


def run_builtin_checked(path: Path, instance: Instance, timeout: int) -> dict[str, object]:
    """Direct F4, annotated with the dimension of its output's leading ideal."""
    row = run_direct(path, "builtin", timeout)
    if row["status"] != "ok":
        return row
    basis = builtin_basis(path, timeout)
    if basis is None:
        return row
    return {
        **row,
        "leading_dimension": leading_ideal_dimension(basis, variable_names(instance), timeout),
    }


def method_runner(
    method: str, path: Path, instance: Instance, timeout: int
) -> Callable[[], dict[str, object]]:
    engine = method.split("-", 1)[1]
    if method.startswith("branching-"):
        return lambda: run_branching(path, instance.solver_args, timeout, engine)
    if method.startswith("zeroing-"):
        return lambda: run_zeroing(path, instance.solver_args, timeout, engine)
    if method == "builtin-grevlex":
        return lambda: run_builtin_checked(path, instance, timeout)
    if method.endswith("-grevlex"):
        return lambda: run_direct(path, method.removesuffix("-grevlex"), timeout)
    return lambda: run_external(path, timeout, {method})[0]


def load_results(path: Path) -> dict[tuple[str, str], dict[str, object]]:
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return {(str(row["system"]), str(row["label"])): row for row in rows}


def benchmark(
    instances: list[Instance],
    root: Path,
    methods: tuple[str, ...],
    timeout: int,
    jobs: int,
    rerun_all: bool,
) -> None:
    done = load_results(RESULTS_FILE)
    tasks = [
        (instance, method)
        for instance in instances
        for method in methods
        if rerun_all or (instance.name, method) not in done
    ]
    lock = threading.Lock()
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

    def execute(task: tuple[Instance, str]) -> None:
        instance, method = task
        row = method_runner(method, root / instance.path, instance, timeout)()
        row.pop("error", None)
        record = {"system": instance.name, "label": method, **row}
        with lock, RESULTS_FILE.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(f"{instance.name} {method} {row['status']} {row['solve_secs']:.3f}", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        list(pool.map(execute, tasks))


def reference_dimension(rows: dict[str, dict[str, object]]) -> int | None:
    """Dimension on which the complete exact runs agree."""
    dimensions = [
        int(row["largest_dimension"])
        for label, row in rows.items()
        if label.startswith("branching-")
        and row["status"] == "ok"
        and row.get("largest_dimension") is not None
    ] + [
        int(row["max_dim"])
        for label, row in rows.items()
        if label in METHOD_CLASSES["decomposition"]
        and row["status"] == "ok"
        and row.get("max_dim") is not None
    ]
    return collections.Counter(dimensions).most_common(1)[0][0] if dimensions else None


def reported_dimension(row: dict[str, object]) -> object:
    return row.get("largest_dimension", row.get("leading_dimension", row.get("max_dim")))


def is_correct(row: dict[str, object], dimension: int | None) -> bool:
    """Completed with the reference dimension."""
    if row["status"] != "ok":
        return False
    reported = reported_dimension(row)
    return dimension is None or reported is None or int(reported) == dimension


def best_entry(
    labels: tuple[str, ...], rows: dict[str, dict[str, object]], dimension: int | None
) -> str:
    completed = [
        (float(rows[label]["solve_secs"]), label)
        for label in labels
        if label in rows and is_correct(rows[label], dimension)
    ]
    if not completed:
        return "timeout"
    seconds, label = min(completed)
    first = rows[label].get("first_solution_secs")
    suffix = f"/{first:.3f}" if isinstance(first, float) else ""
    count = rows[label].get("num_components")
    components = f" ({count})" if count is not None else ""
    return f"{label} {seconds:.3f}{suffix}{components}"


def results_by_system() -> dict[str, dict[str, dict[str, object]]]:
    grouped: dict[str, dict[str, dict[str, object]]] = collections.defaultdict(dict)
    for (system, label), row in load_results(RESULTS_FILE).items():
        grouped[system][label] = row
    return grouped


def render_report(instances: list[Instance]) -> str:
    grouped = results_by_system()
    header = "| System | dim | " + " | ".join(METHOD_CLASSES) + " |"
    table = [header, "|" + "---|" * (len(METHOD_CLASSES) + 2)]
    for instance in instances:
        rows = grouped.get(instance.name, {})
        dimension = reference_dimension(rows)
        cells = [best_entry(labels, rows, dimension) for labels in METHOD_CLASSES.values()]
        table.append(f"| {instance.name} | {dimension} | " + " | ".join(cells) + " |")
    excluded = [
        f"- {system} {label}: dimension {reported_dimension(row)}, "
        f"reference {reference_dimension(rows)}"
        for system, rows in sorted(grouped.items())
        for label, row in sorted(rows.items())
        if row["status"] == "ok" and not is_correct(row, reference_dimension(rows))
    ]
    return "\n".join(
        [
            "# Application systems",
            "",
            "Best completed method per class; factor branching and variable zeroing "
            "give T/F in seconds, HomotopyContinuation.jl and decompositions give the "
            "component count in parentheses.",
            "",
            *table,
            "",
            "Completed runs with a wrong dimension, excluded above:",
            "",
            *excluded,
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=int, default=TIMEOUT_SECS)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument("--rerun-all", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--only", help="comma-separated method labels")
    args = parser.parse_args()
    args.jobs = min(args.jobs, subprocess_limit())
    return args


def build() -> None:
    result = subprocess.run(  # noqa: S603
        ["cargo", "build", "--release", "--bin", "sig_groebner", "--bin", "branching",
         "--bin", "variable_zeroing"],
        cwd=SYMBOLICA,
    )
    if result.returncode:
        raise SystemExit(result.returncode)


def main() -> int:
    args = parse_args()
    methods = tuple(args.only.split(",")) if args.only else METHODS
    unknown = set(methods) - set(METHODS)
    if unknown:
        raise SystemExit(f"unknown methods: {', '.join(sorted(unknown))}")
    with tempfile.TemporaryDirectory(prefix="application_benchmarks_") as temporary:
        root = args.output_dir or Path(temporary)
        instances = generate_applications(root)
        if not args.report_only:
            if not args.no_build:
                build()
            benchmark(instances, root, methods, args.timeout, args.jobs, args.rerun_all)
        REPORT_FILE.write_text(render_report(instances), encoding="utf-8")
    print(f"updated {REPORT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
