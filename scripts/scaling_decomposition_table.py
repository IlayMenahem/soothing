#!/usr/bin/env python3
"""Table ``tab:scaling-decomposition`` and the Alg. IdealRefine paragraph.

Runs the factor-branching variants of paper/sections/method_*.tex. Every method of
``factor_variants.py`` answers one cell of (object, decomposition): the object is the
ideal I or its radical sqrt I, and the decomposition is monolithic, a covering, or
irreducible. Alg. Branching is the standard algorithm, and one Singular command per
cell is the direct baseline. All methods run in the same harness: one Python process
driving one Singular process, timed by wall clock; the conversion to Singular syntax
is not timed. Runs proceed concurrently through ``_harness.run``, each holding a solver
slot shared with every other experiment script; waiting for a slot is not timed either.

``--systems scaling`` (default) runs every method on the families of
``scaling_table.py``, the families concurrently. Within a family the sizes run in
order and the methods of one size concurrently; a method that times out or fails at
one size is not run at larger sizes, except Alg. PrimaryBranching when its isolated components
finished before the embedded-component fallback timed out. The facstd columns of the
table come from ``scaling_table.py``.

``--systems baseline`` runs Alg. IdealRefine against std on the 20 baseline systems,
the systems concurrently.
Both return a description of the ideal I itself that decides ideal membership: std a
Gröbner basis of I, Alg. IdealRefine ideals with Gröbner bases whose intersection is
I (Corollary ideal-membership). Where both finish, an untimed run checks that the
leaves intersect to I.

Examples:

    python3 scripts/scaling_decomposition_table.py --timeout 60 --output-dir /tmp/axes
    python3 scripts/scaling_decomposition_table.py --families box_qp --output-dir /tmp/axes
    python3 scripts/scaling_decomposition_table.py --systems baseline --timeout 60 --output-dir /tmp/ideal_refine
    python3 scripts/scaling_decomposition_table.py --report-only
"""

from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from pathlib import Path

from _harness import RESULTS, append_row, concurrently, load_rows, run
from _systems import generate_matrix, load_system, poly_to_singular
from scaling_table import FACTOR_BRANCHING, FAMILIES
from scaling_table import RESULTS_FILE as SCALING_RESULTS

SCRIPT = Path(__file__).resolve().parent / "factor_variants.py"
RESULTS_FILE = RESULTS / "decomposition.jsonl"
REPORT_FILE = RESULTS / "decomposition.md"
# (cell, factor-branching method, direct Singular method or "" when Singular has none)
CELLS: tuple[tuple[str, str, str], ...] = (
    ("sqrt I, covering", "branching", "direct-facstd"),
    ("I, monolithic", "ideal-reduce", "direct-std"),
    ("I, covering", "ideal-refine", ""),
    ("sqrt I, irreducible", "prime", "direct-minAssGTZ"),
    ("I, irreducible", "primary", "direct-primdecGTZ"),
)
METHODS: tuple[str, ...] = tuple(method for _, ours, direct in CELLS for method in (ours, direct) if method)
TITLES: dict[str, str] = {
    "branching": "Alg. Branching",
    "direct-facstd": "facstd",
    "ideal-reduce": "IdealReduce + std",
    "direct-std": "std",
    "ideal-refine": "IdealRefine",
    "prime": "PrimeBranching",
    "direct-minAssGTZ": "minAssGTZ",
    "primary": "PrimaryBranching",
    "direct-primdecGTZ": "primdecGTZ",
}


def convert(path: Path) -> Path:
    target = path.with_suffix(".singular.json")
    if not target.exists():
        polynomials, symbols = load_system(path)
        target.write_text(
            json.dumps(
                {
                    "variables": [str(symbol) for symbol in symbols],
                    "polynomials": [poly_to_singular(polynomial) for polynomial in polynomials],
                }
            ),
            encoding="utf-8",
        )
    return target


def run_method(method: str, system: Path, timeout: int) -> dict[str, object]:
    stdout, stderr, wall, killed, _ = run([sys.executable, str(SCRIPT), method, str(system)], timeout)
    if killed:
        partial = [json.loads(line.removeprefix("PARTIAL ")) for line in stdout.splitlines() if line.startswith("PARTIAL ")]
        return {"status": "timeout", "wall_secs": float(timeout), **({"partial": partial[-1]} if partial else {})}
    results = [line for line in stdout.splitlines() if line.startswith("{")]
    if not results:
        return {"status": "error", "wall_secs": wall, "error": stderr[-500:]}
    return {**json.loads(results[-1]), "wall_secs": wall}


def finished(row: dict[str, object]) -> bool:
    return row["status"] == "ok"


def gave_up(row: dict[str, object]) -> bool:
    """A failed run stops the method at larger sizes, unless Alg. PrimaryBranching still returned its isolated components."""
    return not finished(row) and not (row["label"] == "primary" and "partial" in row)


def measure(family: str, size: int, name: str, system: Path, timeout: int, method: str) -> dict[str, object]:
    """Run one method on one system of a family and store its row."""
    row = run_method(method, system, timeout)
    record = {"family": family, "size": size, "system": name, "label": method, "timeout": timeout, **row}
    append_row(RESULTS_FILE, record)
    print(f"{name} {method} {row['status']} {float(str(row['wall_secs'])):.3f}", flush=True)
    return record


def benchmark_family(family: str, output: Path, timeout: int) -> None:
    make, sizes, _ = FAMILIES[family]
    done = {(row["size"], row["label"]): row for row in load_rows(RESULTS_FILE) if row["family"] == family}
    failed_at: dict[str, int] = {}
    for (size, label), row in sorted(done.items()):
        if gave_up(row):
            failed_at.setdefault(str(label), int(str(size)))
    for size in sizes:
        pending = [m for m in METHODS if failed_at.get(m, size) >= size and (size, m) not in done]
        if not pending:
            continue
        path = make(output, size)
        system = convert(path)
        for record in concurrently(partial(measure, family, size, path.stem, system, timeout), pending):
            if gave_up(record):
                failed_at[str(record["label"])] = min(failed_at.get(str(record["label"]), size), size)


def outcome(row: dict[str, object]) -> str:
    """What a completed run found, in the terms of its cell."""
    method = str(row["label"])
    if method in ("branching", "direct-facstd"):
        return f"{row['branches']} br"
    if method == "ideal-reduce":
        return f"{row['input_generators']}→{row['reduced_generators']} gens"
    if method == "direct-std":
        return f"{row['basis_size']} gens"
    if method == "ideal-refine":
        return f"{row['leaves']} leaves"
    if method in ("prime", "direct-minAssGTZ"):
        return f"{row['primes']} primes"
    embedded = int(str(row["embedded"]))
    return f"{row['components']} comp" + (f", {embedded} emb" if embedded else "")


def cell(row: dict[str, object] | None) -> str:
    if row is None:
        return "—"
    if row["status"] == "timeout":
        partial = row.get("partial")
        if isinstance(partial, dict) and row["label"] == "primary":
            return f"TO (isolated {float(partial['isolated_secs']):.2f}: {partial['isolated']} comp, has emb)"
        if isinstance(partial, dict) and row["label"] == "ideal-reduce":
            return (
                f"TO (reduced {float(partial['reduce_secs']):.2f}: "
                f"{partial['input_generators']}→{partial['reduced_generators']} gens)"
            )
        return "TO"
    if row["status"] != "ok":
        return "err"
    return f"{float(str(row['wall_secs'])):.2f} ({outcome(row)})"


def rust_branching(family: str, size: int) -> str:
    """Fastest recorded Rust Alg. Branching run from the size-graded benchmark."""
    times = [
        float(str(row["wall_secs"]))
        for row in load_rows(SCALING_RESULTS)
        if row["family"] == family
        and row["size"] == size
        and row["label"] in FACTOR_BRANCHING
        and row["status"] == "ok"
        and row.get("complete", True) is not False
    ]
    return f"{min(times):.2f}" if times else "TO"


def agreement(rows: dict[str, dict[str, object]]) -> list[str]:
    """Checks that a variant and its direct baseline found the same object, where both finished."""
    checks: list[tuple[str, str, tuple[str, ...]]] = [
        ("branching", "direct-facstd", ("max_dim",)),
        ("ideal-reduce", "direct-std", ("basis_hash",)),
        ("prime", "direct-minAssGTZ", ("prime_hashes",)),
        ("primary", "direct-primdecGTZ", ("components", "isolated_hashes")),
    ]
    verdicts: list[str] = []
    for ours, theirs, keys in checks:
        a, b = rows.get(ours), rows.get(theirs)
        if a is None or b is None or not finished(a) or not finished(b):
            continue
        same = all(a.get(key) == b.get(key) for key in keys)
        verdicts.append(f"{TITLES[ours]} {'=' if same else '≠'} {TITLES[theirs]}")
    return verdicts


def render_report() -> str:
    rows = load_rows(RESULTS_FILE)
    timeout = max((int(str(row["timeout"])) for row in rows), default=60)
    header = ["size", "Rust Alg. Branching", *(TITLES[m] for m in METHODS)]
    lines = [
        "# Factor-branching variants on the size-graded families",
        "",
        f"Wall time in seconds with a {timeout}-second limit, and what each completed run found.",
        "Every method runs in the same harness (`scripts/factor_variants.py`): one Python process",
        "driving one Singular 4.4.1 process. *Rust Alg. Branching* is the fastest recorded backend",
        "from `scaling.jsonl`. Cells, as (object, decomposition):",
        "",
        *(f"- ({name}): {TITLES[ours]}" + (f" vs {TITLES[direct]}" if direct else "") for name, ours, direct in CELLS),
        "",
        "br = branches or factorizing-GB components, gens = generators, comp = primary components,",
        "emb = embedded components. TO = timeout, — = not run (an earlier size failed).",
        "",
    ]
    mismatches: list[str] = []
    for family, (_, sizes, _) in FAMILIES.items():
        family_rows = [row for row in rows if row["family"] == family]
        if not family_rows:
            continue
        lines += [f"## {family}", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        for size in sizes:
            by_method = {str(row["label"]): row for row in family_rows if row["size"] == size}
            if not by_method:
                continue
            cells = [str(size), rust_branching(family, size), *(cell(by_method.get(m)) for m in METHODS)]
            lines.append("| " + " | ".join(cells) + " |")
            mismatches += [f"{family} {size}: {v}" for v in agreement(by_method) if "≠" in v]
        lines.append("")
    checked = sum(
        len(agreement({str(r["label"]): r for r in rows if r["family"] == f and r["size"] == s}))
        for f, s in {(row["family"], row["size"]) for row in rows}
    )
    lines += [
        "## Agreement",
        "",
        f"{checked} variant/baseline pairs finished on the same system; {len(mismatches)} disagree.",
        "Compared: largest dimension (coverings), reduced Gröbner basis (I), minimal primes (sqrt I),",
        "and the number of components with the isolated components (primary decompositions).",
        *(f"- {m}" for m in mismatches),
        "",
    ]
    return "\n".join(lines)




BASELINE_RESULTS_FILE = RESULTS / "ideal_refine.jsonl"
BASELINE_REPORT_FILE = RESULTS / "ideal_refine.md"
BASELINE_METHODS: tuple[str, ...] = ("ideal-refine", "direct-std")


def benchmark_baseline_system(
    output: Path, timeout: int, done: set[tuple[str, str]], instance_path: str
) -> None:
    """Both methods on one baseline system, then, where both finished, the untimed exactness check."""
    path = output / Path(instance_path).name
    system = convert(path)
    for method in BASELINE_METHODS:
        if (path.stem, method) in done:
            continue
        row = run_method(method, system, timeout)
        record = {"system": path.stem, "label": method, "timeout": timeout, **row}
        append_row(BASELINE_RESULTS_FILE, record)
        print(f"{path.stem} {method} {row['status']} {float(str(row['wall_secs'])):.3f}", flush=True)
    both = [r for r in load_rows(BASELINE_RESULTS_FILE) if r["system"] == path.stem and r["status"] == "ok"]
    if len({r["label"] for r in both}) == len(BASELINE_METHODS) and (path.stem, "verify-ideal-refine") not in done:
        row = run_method("verify-ideal-refine", system, 10 * timeout)
        append_row(BASELINE_RESULTS_FILE, {"system": path.stem, "label": "verify-ideal-refine", **row})
        print(f"{path.stem} verify {row.get('exact')}", flush=True)


def benchmark_baseline(output: Path, timeout: int) -> None:
    """Every baseline system concurrently."""
    done = {(str(row["system"]), str(row["label"])) for row in load_rows(BASELINE_RESULTS_FILE)}
    instances = [instance.path for instance in generate_matrix(output)]
    concurrently(partial(benchmark_baseline_system, output, timeout, done), instances)


def baseline_cell(row: dict[str, object] | None) -> str:
    if row is None:
        return "—"
    if row["status"] != "ok":
        return "TO" if row["status"] == "timeout" else "err"
    found = f"{row['leaves']} leaves" if row["label"] == "ideal-refine" else f"{row['basis_size']} gens"
    return f"{float(str(row['wall_secs'])):.2f} ({found})"


def render_baseline_report() -> str:
    rows = load_rows(BASELINE_RESULTS_FILE)
    systems = list(dict.fromkeys(str(row["system"]) for row in rows))
    lines = [
        "# Alg. IdealRefine vs std on the baseline systems",
        "",
        "Wall time in seconds and what each completed run found, in the harness of",
        "`scripts/factor_variants.py`. TO = timeout. *exact*: the leaves intersect to I (untimed check).",
        "",
        "| system | IdealRefine | std | exact |",
        "|---|---|---|---|",
    ]
    for system in systems:
        by_label = {str(row["label"]): row for row in rows if row["system"] == system}
        verify = by_label.get("verify-ideal-refine")
        exact = "—" if verify is None or verify["status"] != "ok" else str(verify["exact"])
        lines.append(f"| {system} | {baseline_cell(by_label.get('ideal-refine'))} | {baseline_cell(by_label.get('direct-std'))} | {exact} |")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--systems", choices=("scaling", "baseline"), default="scaling")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/scaling_decomposition"))
    parser.add_argument("--families", default="", help="comma-separated scaling families")
    parser.add_argument("--report-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    families = args.families.split(",") if args.families else list(FAMILIES)
    unknown = set(families) - FAMILIES.keys()
    if unknown:
        raise SystemExit(f"unknown families: {', '.join(sorted(unknown))}")
    if not args.report_only:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        if args.systems == "baseline":
            benchmark_baseline(args.output_dir, args.timeout)
        else:
            concurrently(lambda family: benchmark_family(family, args.output_dir, args.timeout), families)
    REPORT_FILE.write_text(render_report() + "\n", encoding="utf-8")
    BASELINE_REPORT_FILE.write_text(render_baseline_report(), encoding="utf-8")
    print(f"updated {REPORT_FILE} and {BASELINE_REPORT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
