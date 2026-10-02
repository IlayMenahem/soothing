#!/usr/bin/env python3
"""Benchmark factor branching and variable zeroing on systems we did not design.

Three sources, all independent of this paper:

* SymbolicData IntPS, the standard collection of polynomial-system benchmarks
  (pinned to commit ``SYMBOLICDATA_COMMIT``), downloaded on first use;
* ODEbase (Lueders, Sturm, Radulescu 2022), mass-action and other polynomial
  models from BioModels with exact parameter values: the steady-state system
  (numerators of the right-hand sides) and, when the network has conservation
  laws, the steady states in the model's stoichiometric compatibility class;
* classical decomposition benchmarks from combinatorial commutative algebra,
  whose minimal primes are known and lie in coordinate subspaces:
  - binomial edge ideals J_G = (x_i y_j - x_j y_i : ij in E(G))
    (Herzog, Hibi, Hreinsdottir, Kahle, Rauh 2010; Ohtani 2011);
  - ideals of adjacent 2x2 minors of an m x n matrix
    (Diaconis, Eisenbud, Sturmfels 1998; Hosten, Sullivant 2004);
  - ideals of 2x2 permanents of an m x n matrix
    (Laubenbacher, Swanson 2000; Kirkup 2008).

Every method sees the same input. Branch backends are compared with the same
tool run standalone: ``singular-gb`` with ``singular-gb-dp``,
``singular-facstd`` and ``singular-primdecGTZ`` with themselves.

Results are appended to a resumable JSON-lines file.

Examples:

    python3 scripts/benchmark_natural.py --jobs 12 --timeout 30 --only screen
    python3 scripts/benchmark_natural.py --systems Gerdt-85,Roczen --jobs 2
    python3 scripts/benchmark_natural.py --report-only
"""

from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import json
import random
import re
import sys
import threading
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Callable
from pathlib import Path

import sympy as sp

from benchmark_applications import build
from benchmark_branching import run_branching, run_zeroing
from benchmark_idos import (
    BASELINES,
    RESULTS,
    run,
    run_direct,
    parse_baseline,
    subprocess_limit,
    write_system,
)

RESULTS_FILE = RESULTS / "natural_results.jsonl"
SYMBOLICDATA_COMMIT = "a5cdf5366150bffba22a2c91a86a012dd23831af"
SYMBOLICDATA_LIST = (
    "https://api.github.com/repos/symbolicdata/data/contents/XMLResources/IntPS"
    f"?ref={SYMBOLICDATA_COMMIT}"
)
SYMBOLICDATA_RAW = (
    f"https://raw.githubusercontent.com/symbolicdata/data/{SYMBOLICDATA_COMMIT}"
    "/XMLResources/IntPS/"
)
# 3-SAT encodings with 100-200 distinct linear factors; the hitting-set
# enumeration packs distinct factors into a 63-bit mask.
SYMBOLICDATA_EXCLUDED_PREFIXES: tuple[str, ...] = ("satlib.",)
ENGINES: tuple[str, ...] = (
    "m4gb",
    "f5",
    "gvw",
    "symbolica",
    "singular-gb",
    "singular-facstd",
    "singular-primdecGTZ",
)
EXTERNAL: dict[str, tuple[str, list[str]]] = {
    "singular-gb-dp": ("system_to_singular.py", ["--task", "gb-dp"]),
    "singular-gb-lex": ("system_to_singular.py", ["--task", "gb"]),
    "singular-facstd": ("system_to_singular.py", ["--task", "facstd"]),
    "singular-primdecGTZ": ("system_to_singular.py", ["--task", "decompose"]),
    "m2-gb-lex": ("system_to_macaulay2.py", ["--task", "gb"]),
    "m2-decompose": ("system_to_macaulay2.py", ["--task", "decompose"]),
}
DIRECT_ALGORITHMS: tuple[str, ...] = ("m4gb", "f5", "gvw", "builtin")
METHODS: tuple[str, ...] = (
    *(f"branching-{engine}" for engine in ENGINES),
    *(f"zeroing-{engine}" for engine in ENGINES),
    *(f"{algorithm}-grevlex" for algorithm in DIRECT_ALGORITHMS),
    *EXTERNAL,
)
# The Singular tools alone and as branch backends.
SCREEN: tuple[str, ...] = (
    "singular-gb-dp",
    "singular-facstd",
    "singular-primdecGTZ",
    *(
        f"{algorithm}-{engine}"
        for algorithm in ("branching", "zeroing")
        for engine in ("singular-gb", "singular-facstd", "singular-primdecGTZ")
    ),
)
ODEBASE = "https://odebase.org"
TOKEN = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")
ODEBASE_SPECIES = re.compile(r"x\[(\d+)\]")
ODEBASE_PARAMETER = re.compile(r"k\[(\d+)\]")


def system_record(
    output: Path, family: str, name: str, polynomials: list[str], variables: int, source: str
) -> dict[str, object]:
    path = output / f"{name}.txt"
    write_system(path, polynomials)
    return {
        "family": family,
        "name": name,
        "path": str(path),
        "variables": variables,
        "equations": len(polynomials),
        "source": source,
    }


def symbolicdata_system(output: Path, file_name: str) -> dict[str, object] | None:
    """One IntPS system with its variables renamed to x0, x1, ... in <vars> order."""
    xml_text = urllib.request.urlopen(SYMBOLICDATA_RAW + file_name, timeout=60).read()
    root = ET.fromstring(xml_text)
    names = [name.strip() for name in (root.findtext("vars") or "").split(",") if name.strip()]
    renaming = {name: f"x{index}" for index, name in enumerate(names)}
    polynomials = [
        TOKEN.sub(lambda match: renaming[match.group(0)], " ".join((poly.text or "").split()))
        for poly in root.iterfind("basis/poly")
    ]
    if not polynomials:
        return None
    return system_record(
        output,
        "symbolicdata",
        Path(file_name).stem,
        polynomials,
        len(names),
        f"SymbolicData IntPS/{file_name}@{SYMBOLICDATA_COMMIT[:7]}",
    )


def symbolicdata_systems(output: Path) -> list[dict[str, object]]:
    listing = json.loads(urllib.request.urlopen(SYMBOLICDATA_LIST, timeout=60).read())
    names = sorted(
        entry["name"]
        for entry in listing
        if not entry["name"].startswith(SYMBOLICDATA_EXCLUDED_PREFIXES)
    )
    with concurrent.futures.ThreadPoolExecutor(16) as pool:
        records = list(pool.map(lambda name: symbolicdata_system(output, name), names))
    return [record for record in records if record is not None]


def odebase_text(kind: str, model: int) -> str:
    url = f"{ODEBASE}/detail/{kind}/{model}/maple"
    return urllib.request.urlopen(url, timeout=60).read().decode("utf-8")


def odebase_polynomial_models() -> list[tuple[int, str]]:
    """(detail id, BioModels id) of every model that ODEbase marks polynomial."""
    table = urllib.request.urlopen(f"{ODEBASE}/table/", timeout=60).read().decode("utf-8")
    models = []
    for row in re.findall(r"<tr.*?</tr>", table, flags=re.S):
        cells = [re.sub(r"<[^>]+>", "", cell).strip() for cell in re.findall(r"<td.*?</td>", row, flags=re.S)]
        link = re.search(r'href="/detail/(\d+)"', row)
        if link and len(cells) == 10 and cells[8] == "Yes":
            models.append((int(link.group(1)), cells[1]))
    return models


def maple_equations(text: str) -> list[tuple[str, str]]:
    """The (left, right) sides of a Maple list of equations, one per line."""
    body = text.strip().removeprefix("[").removesuffix("]")
    lines = (line.strip().removesuffix(",") for line in body.splitlines())
    return [tuple(line.split(" = ", 1)) for line in lines if " = " in line]


def odebase_polynomial(expression: str, parameters: dict[str, str]) -> str | None:
    """Primitive integer numerator of a Maple expression, x[i] renamed to x{i-1}."""
    valued = ODEBASE_PARAMETER.sub(lambda match: f"({parameters[match.group(1)]})", expression)
    renamed = ODEBASE_SPECIES.sub(lambda match: f"x{int(match.group(1)) - 1}", valued)
    numerator = sp.expand(sp.fraction(sp.together(sp.sympify(renamed)))[0])
    if numerator == 0:
        return None
    if not numerator.free_symbols:
        return "1"
    primitive = sp.Poly(numerator, *sorted(numerator.free_symbols, key=str)).primitive()[1]
    return sp.sstr(primitive.as_expr()).replace("**", "^")


def odebase_model(output: Path, model: int, biomodel: str) -> list[dict[str, object]]:
    """The steady-state system of one model and, with conservation laws, its
    restriction to the model's stoichiometric compatibility class."""
    parameters = dict(re.findall(r"k\[(\d+)\]\s*=\s*(\S+)", odebase_text("parameters", model)))
    odes = maple_equations(odebase_text("odes", model))
    laws = maple_equations(odebase_text("constraints", model))
    try:
        steady = [
            polynomial
            for _, right in odes
            if (polynomial := odebase_polynomial(right, parameters)) is not None
        ]
        conserved = [
            polynomial
            for left, right in laws
            if (polynomial := odebase_polynomial(f"({left}) - ({right})", parameters)) is not None
        ]
    except (KeyError, sp.SympifyError, sp.PolynomialError) as error:
        print(f"skip ODEbase {biomodel}: {error!r}", file=sys.stderr)
        return []
    source = f"ODEbase {biomodel} (odebase.org/detail/{model})"
    systems = [(f"{biomodel}_ss", "odebase-ss", steady)] if steady else []
    if steady and conserved:
        systems.append((f"{biomodel}_ssc", "odebase-ssc", steady + conserved))
    return [
        system_record(output, family, name, polynomials, len(odes), source)
        for name, family, polynomials in systems
    ]


def odebase_systems(output: Path) -> list[dict[str, object]]:
    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        nested = list(
            pool.map(
                lambda model: odebase_model(output, *model),
                odebase_polynomial_models(),
            )
        )
    return [system for systems in nested for system in systems]


def matrix_variables(rows: int, columns: int) -> dict[tuple[int, int], str]:
    return {
        (row, column): f"x{row * columns + column}"
        for row in range(rows)
        for column in range(columns)
    }


def adjacent_minors(output: Path, rows: int, columns: int) -> dict[str, object]:
    entry = matrix_variables(rows, columns)
    polynomials = [
        f"{entry[row, column]}*{entry[row + 1, column + 1]}"
        f" - {entry[row, column + 1]}*{entry[row + 1, column]}"
        for row in range(rows - 1)
        for column in range(columns - 1)
    ]
    return system_record(
        output,
        "adjacent-minors",
        f"adjacent_minors_{rows}x{columns}",
        polynomials,
        rows * columns,
        "Diaconis-Eisenbud-Sturmfels 1998; Hosten-Sullivant 2004",
    )


def permanents(output: Path, rows: int, columns: int) -> dict[str, object]:
    entry = matrix_variables(rows, columns)
    polynomials = [
        f"{entry[top, left]}*{entry[bottom, right]} + {entry[top, right]}*{entry[bottom, left]}"
        for top, bottom in itertools.combinations(range(rows), 2)
        for left, right in itertools.combinations(range(columns), 2)
    ]
    return system_record(
        output,
        "permanents",
        f"permanents_{rows}x{columns}",
        polynomials,
        rows * columns,
        "Laubenbacher-Swanson 2000; Kirkup 2008",
    )


def binomial_edge_ideal(
    output: Path, name: str, vertices: int, edges: list[tuple[int, int]]
) -> dict[str, object]:
    """x_i = x{i}, y_i = x{vertices + i}."""
    polynomials = [
        f"x{i}*x{vertices + j} - x{j}*x{vertices + i}" for i, j in sorted(edges)
    ]
    return system_record(
        output,
        "binomial-edge",
        f"binomial_edge_{name}",
        polynomials,
        2 * vertices,
        "Herzog-Hibi-Hreinsdottir-Kahle-Rauh 2010; Ohtani 2011",
    )


def cycle(vertices: int) -> list[tuple[int, int]]:
    return [(index, (index + 1) % vertices) for index in range(vertices)]


def complete_bipartite(left: int, right: int) -> list[tuple[int, int]]:
    return [(i, left + j) for i in range(left) for j in range(right)]


def petersen() -> list[tuple[int, int]]:
    outer = [(index, (index + 1) % 5) for index in range(5)]
    spokes = [(index, index + 5) for index in range(5)]
    inner = [(5 + index, 5 + (index + 2) % 5) for index in range(5)]
    return outer + spokes + inner


def random_graph(vertices: int, edges: int, seed: int) -> list[tuple[int, int]]:
    pairs = list(itertools.combinations(range(vertices), 2))
    return random.Random(seed).sample(pairs, edges)


def combinatorial_systems(output: Path) -> list[dict[str, object]]:
    graphs = [
        *((f"C{n}", n, cycle(n)) for n in (6, 8, 10, 12, 14)),
        *((f"K{m}_{n}", m + n, complete_bipartite(m, n)) for m, n in ((3, 3), (3, 4), (4, 4), (4, 5))),
        ("petersen", 10, petersen()),
        *((f"G{n}_{e}", n, random_graph(n, e, 5000 + n)) for n, e in ((8, 12), (10, 15), (12, 18))),
    ]
    return [
        *(adjacent_minors(output, 2, n) for n in (6, 9, 12, 15)),
        *(adjacent_minors(output, 3, n) for n in (3, 4, 5, 6)),
        *(adjacent_minors(output, 4, n) for n in (4, 5)),
        *(permanents(output, 2, n) for n in (4, 5, 6, 7, 8)),
        *(permanents(output, 3, n) for n in (3, 4, 5)),
        permanents(output, 4, 4),
        *(binomial_edge_ideal(output, name, vertices, edges) for name, vertices, edges in graphs),
    ]


def run_standalone(label: str, path: Path, timeout: int) -> dict[str, object]:
    script, extra = EXTERNAL[label]
    stdout, _, wall, killed, _ = run(
        [sys.executable, str(BASELINES / script), str(path), *extra, "--timeout-secs", str(timeout)],
        timeout + 2,
    )
    row = parse_baseline(stdout, label, wall, killed)
    if killed:
        row["solve_secs"] = float(timeout)
    return row


def method_runner(method: str, path: Path, timeout: int) -> Callable[[], dict[str, object]]:
    if method in EXTERNAL:
        return lambda: run_standalone(method, path, timeout)
    if method.endswith("-grevlex"):
        return lambda: run_direct(path, method.removesuffix("-grevlex"), timeout)
    algorithm, engine = method.split("-", 1)
    if algorithm == "branching":
        return lambda: run_branching(path, [], timeout, engine)
    return lambda: run_zeroing(path, [], timeout, engine)


def load_rows() -> list[dict[str, object]]:
    if not RESULTS_FILE.exists():
        return []
    return [json.loads(line) for line in RESULTS_FILE.read_text(encoding="utf-8").splitlines() if line]


def benchmark(
    systems: list[dict[str, object]], methods: tuple[str, ...], timeout: int, jobs: int
) -> None:
    done = {(row["system"], row["label"], row["timeout"]) for row in load_rows()}
    tasks = [
        (system, method)
        for system in systems
        for method in methods
        if (system["name"], method, timeout) not in done
    ]
    lock = threading.Lock()

    def execute(task: tuple[dict[str, object], str]) -> None:
        system, method = task
        row = method_runner(method, Path(str(system["path"])), timeout)()
        row.pop("error", None)
        record = {
            "system": system["name"],
            "family": system["family"],
            "label": method,
            "timeout": timeout,
            **row,
        }
        with lock, RESULTS_FILE.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(f"{system['name']} {method} {row['status']} {row['solve_secs']:.3f}", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        list(pool.map(execute, tasks))


# Each Singular tool alone, and the branch backend that runs the same tool.
PAIRS: dict[str, tuple[str, str]] = {
    "std (dp)": ("singular-gb-dp", "singular-gb"),
    "facstd": ("singular-facstd", "singular-facstd"),
    "primdecGTZ": ("singular-primdecGTZ", "singular-primdecGTZ"),
}
# A speedup or slowdown counts only above this factor and this many seconds.
SIGNIFICANT_FACTOR = 2.0
SIGNIFICANT_SECS = 1.0


def seconds(row: dict[str, object] | None) -> float | None:
    """Time of a complete run, or None. The Macaulay2 driver reports CPU time
    summed over threads, which can exceed the wall time of the run."""
    if row is None or row["status"] != "ok":
        return None
    return min(float(row["solve_secs"]), float(row.get("wall_secs", row["solve_secs"])))


def gave_up(row: dict[str, object] | None) -> bool:
    """The run hit the time limit, or ended incomplete when that limit expired."""
    return row is not None and row["status"] in {"timeout", "partial"}


def ours_seconds(rows: dict[str, dict[str, object]], engine: str) -> float | None:
    times = [
        time
        for algorithm in ("branching", "zeroing")
        if (time := seconds(rows.get(f"{algorithm}-{engine}"))) is not None
    ]
    return min(times, default=None)


def verdict(rows: dict[str, dict[str, object]], pair: str) -> str | None:
    """'rescue', 'faster', 'slower' or 'lost' for one pair, or None if neutral."""
    standalone, engine = PAIRS[pair]
    alone, ours = seconds(rows.get(standalone)), ours_seconds(rows, engine)
    if alone is None and ours is not None and gave_up(rows.get(standalone)):
        return "rescue"
    if alone is not None and ours is None and all(
        gave_up(rows.get(f"{algorithm}-{engine}")) for algorithm in ("branching", "zeroing")
    ):
        return "lost"
    if alone is None or ours is None or abs(alone - ours) < SIGNIFICANT_SECS:
        return None
    if ours * SIGNIFICANT_FACTOR <= alone:
        return "faster"
    if alone * SIGNIFICANT_FACTOR <= ours:
        return "slower"
    return None


def cell(row: dict[str, object] | None) -> str:
    if row is None:
        return "-"
    if row["status"] == "ok":
        return f"{seconds(row):.2f}"
    return "TO" if gave_up(row) else "err"


def fastest(texts: list[str]) -> str:
    numbers = [text for text in texts if text.replace(".", "", 1).isdigit()]
    return min(numbers, key=float, default="")


def pair_cells(rows: dict[str, dict[str, object]], pair: str) -> list[str]:
    """Tool alone, FB and VZ with its split count; the faster of FB and VZ is
    bold when it wins."""
    standalone, engine = PAIRS[pair]
    alone = cell(rows.get(standalone))
    ours = [cell(rows.get(f"{algorithm}-{engine}")) for algorithm in ("branching", "zeroing")]
    best = fastest(ours) if verdict(rows, pair) in {"rescue", "faster"} else None
    marked = [f"**{text}**" if text == best else text for text in ours]
    splits = (rows.get(f"zeroing-{engine}") or {}).get("splits")
    return [alone, marked[0], f"{marked[1]} [{splits}]" if splits else marked[1]]


def branch_count(rows: dict[str, dict[str, object]]) -> str:
    """Terminal branches of Algorithm Refine, which no backend changes."""
    totals = {
        int(row["total"])
        for label, row in rows.items()
        if label.startswith("branching-") and row.get("total") is not None
    }
    return "/".join(str(total) for total in sorted(totals)) or "-"


def beats_every_tool(rows: dict[str, dict[str, object]]) -> bool:
    """FB or VZ completes significantly faster than every tool alone, or
    completes while every tool alone gives up."""
    alone = [seconds(rows.get(standalone)) for standalone, _ in PAIRS.values()]
    finished = [time for time in alone if time is not None]
    ours = [time for _, engine in PAIRS.values() if (time := ours_seconds(rows, engine)) is not None]
    if not ours:
        return False
    if not finished:
        return all(gave_up(rows.get(standalone)) for standalone, _ in PAIRS.values())
    best_alone, best_ours = min(finished), min(ours)
    return best_ours * SIGNIFICANT_FACTOR <= best_alone and best_alone - best_ours >= SIGNIFICANT_SECS


def best_in_class(rows: dict[str, dict[str, object]], labels: tuple[str, ...]) -> str:
    """Fastest complete run among ``labels`` as 'label seconds', or 'TO'."""
    completed = [(time, label) for label in labels if (time := seconds(rows.get(label))) is not None]
    if not completed:
        return "TO" if any(label in rows for label in labels) else "-"
    time, label = min(completed)
    return f"{label} {time:.2f}"


METHOD_CLASSES: dict[str, tuple[str, ...]] = {
    "factor branching": tuple(f"branching-{engine}" for engine in ENGINES),
    "variable zeroing": tuple(f"zeroing-{engine}" for engine in ENGINES),
    "direct GB": (
        *(f"{algorithm}-grevlex" for algorithm in DIRECT_ALGORITHMS),
        "singular-gb-dp",
        "singular-gb-lex",
        "m2-gb-lex",
    ),
    "decomposition": ("singular-facstd", "singular-primdecGTZ", "m2-decompose"),
}


def class_table(
    measured: list[dict[str, object]], grouped: dict[str, dict[str, dict[str, object]]]
) -> list[str]:
    """Fastest complete method per class, for systems run with every method."""
    complete = [
        system for system in measured if set(METHODS) <= grouped[str(system["name"])].keys()
    ]
    if not complete:
        return []
    return [
        "## Fastest method per class (systems run with every method)",
        "",
        "| System | FB branches | " + " | ".join(METHOD_CLASSES) + " |",
        "|---|---|" + "---|" * len(METHOD_CLASSES),
        *(
            f"| {system['name']} | {branch_count(grouped[str(system['name'])])} | "
            + " | ".join(
                best_in_class(grouped[str(system["name"])], labels)
                for labels in METHOD_CLASSES.values()
            )
            + " |"
            for system in complete
        ),
        "",
    ]


def render_report(systems: list[dict[str, object]], timeout: int) -> str:
    grouped: dict[str, dict[str, dict[str, object]]] = {}
    for row in load_rows():
        if row["timeout"] == timeout:
            grouped.setdefault(str(row["system"]), {})[str(row["label"])] = row
    measured = [system for system in systems if system["name"] in grouped]
    families = sorted({str(system["family"]) for system in measured})
    verdicts = {
        (system["name"], pair): verdict(grouped[system["name"]], pair)
        for system in measured
        for pair in PAIRS
    }
    summary = [
        "| Family | systems | " + " | ".join(
            f"{pair}: rescued / faster / slower / lost" for pair in PAIRS
        ) + " |",
        "|---|---|" + "---|" * len(PAIRS),
    ]
    for family in families:
        names = [system["name"] for system in measured if system["family"] == family]
        counts = [
            " / ".join(
                str(sum(verdicts[name, pair] == kind for name in names))
                for kind in ("rescue", "faster", "slower", "lost")
            )
            for pair in PAIRS
        ]
        summary.append(f"| {family} | {len(names)} | " + " | ".join(counts) + " |")
    header = "| System | family | vars/eqs | FB branches | " + " | ".join(
        f"{pair} alone | FB | VZ [splits]" for pair in PAIRS
    ) + " |"
    divider = "|---|---|---|---|" + "---|---|---|" * len(PAIRS)

    def table(selected: Callable[[str], bool]) -> list[str]:
        return [header, divider] + [
            f"| {system['name']} | {system['family']} | "
            f"{system['variables']}/{system['equations']} | {branch_count(grouped[system['name']])} | "
            + " | ".join(text for pair in PAIRS for text in pair_cells(grouped[system['name']], pair))
            + " |"
            for system in measured
            if selected(str(system["name"]))
        ]

    def with_verdict(kinds: set[str]) -> Callable[[str], bool]:
        return lambda name: any(verdicts[name, pair] in kinds for pair in PAIRS)

    return "\n".join(
        [
            f"# Systems not designed for factor branching ({timeout} s limit)",
            "",
            "Each Singular tool is compared with factor branching (FB) and variable",
            "zeroing (VZ) that use the same tool as their branch solver. *Rescued*: the",
            "tool alone times out and FB or VZ completes; *faster*/*slower*: the faster of",
            f"FB and VZ differs by at least {SIGNIFICANT_FACTOR:g}x and {SIGNIFICANT_SECS:g} s;",
            "*lost*: the tool alone completes and both FB and VZ time out. Standalone",
            "times are Singular's internal timer; FB and VZ times are end-to-end and",
            "include factorization and one Singular start per branch. TO = time limit.",
            "*FB branches* counts the terminal branches of Algorithm Refine; with one",
            "branch, FB differs from the tool alone only by the lex interreduction it",
            "applies first. *[splits]* counts the zero/nonzero splits of VZ.",
            "",
            *summary,
            "",
            *class_table(measured, grouped),
            "## Systems where FB or VZ beats every tool alone",
            "",
            *table(lambda name: beats_every_tool(grouped[name])),
            "",
            "## Systems where FB or VZ rescues or speeds up a tool",
            "",
            *table(with_verdict({"rescue", "faster"})),
            "",
            "## Systems where FB and VZ are slower or lose a completed run",
            "",
            *table(with_verdict({"slower", "lost"})),
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument("--only", help="comma-separated method labels, or 'screen'")
    parser.add_argument("--systems", help="comma-separated system names")
    parser.add_argument("--families", help="comma-separated families")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    args.jobs = min(args.jobs, subprocess_limit())
    return args


def load_systems(output_dir: Path) -> list[dict[str, object]]:
    """Every system, generating or downloading the collections not yet in the
    output directory's manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "manifest.json"
    known = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {}
    collections = {
        name: known[name] if name in known else generate(output_dir)
        for name, generate in COLLECTIONS.items()
    }
    if collections.keys() != known.keys():
        manifest.write_text(json.dumps(collections, indent=1) + "\n", encoding="utf-8")
    return [system for systems in collections.values() for system in systems]


COLLECTIONS: dict[str, Callable[[Path], list[dict[str, object]]]] = {
    "combinatorial": combinatorial_systems,
    "symbolicdata": symbolicdata_systems,
    "odebase": odebase_systems,
}


def main() -> int:
    args = parse_args()
    methods = SCREEN if args.only == "screen" else tuple(args.only.split(",")) if args.only else METHODS
    unknown = set(methods) - set(METHODS)
    if unknown:
        raise SystemExit(f"unknown methods: {', '.join(sorted(unknown))}")
    systems = load_systems(args.output_dir)
    if args.systems:
        systems = [system for system in systems if system["name"] in args.systems.split(",")]
    if args.families:
        systems = [system for system in systems if system["family"] in args.families.split(",")]
    if not args.report_only:
        if not args.no_build:
            build()
        benchmark(systems, methods, args.timeout, args.jobs)
    report = RESULTS / f"natural_comparison_{args.timeout}s.md"
    report.write_text(render_report(systems, args.timeout), encoding="utf-8")
    print(f"updated {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
