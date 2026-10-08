#!/usr/bin/env python3
"""Run the factor-branching variants of paper/sections/method_*.tex on the size-graded families.

Every method of ``factor_variants.py`` answers one cell of (object, decomposition):
the object is the ideal I or its radical sqrt I, and the decomposition is monolithic,
a covering, or irreducible. Alg. Branching is the standard algorithm, and one Singular
command per cell is the direct baseline. All methods run in the same harness: one
Python process driving one Singular process, timed by wall clock. The systems come
from ``benchmark_scaling.py``'s families; their conversion to Singular syntax is not
timed. Within a family, a method that times out or fails at one size is not run at
larger sizes, except Alg. PrimaryBranching when its isolated components finished
before the embedded-component fallback timed out.

Examples:

    python3 scripts/benchmark_axes.py --timeout 60 --output-dir /tmp/axes
    python3 scripts/benchmark_axes.py --families box_qp --output-dir /tmp/axes
    python3 scripts/benchmark_axes.py --report-only --output-dir /tmp/axes
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "baselines"))
from _common import load_system  # noqa: E402
from benchmark_idos import RESULTS  # noqa: E402
from benchmark_scaling import FACTOR_BRANCHING, FAMILIES  # noqa: E402
from benchmark_scaling import RESULTS_FILE as SCALING_RESULTS  # noqa: E402
from system_to_singular import _poly_to_singular  # noqa: E402

SCRIPT = Path(__file__).resolve().parent / "factor_variants.py"
RESULTS_FILE = RESULTS / "axes_results.jsonl"
REPORT_FILE = RESULTS / "axes_comparison.md"
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
                    "polynomials": [_poly_to_singular(polynomial) for polynomial in polynomials],
                }
            ),
            encoding="utf-8",
        )
    return target


def run_method(method: str, system: Path, timeout: int) -> dict[str, object]:
    started = time.monotonic()
    process = subprocess.Popen(
        [sys.executable, str(SCRIPT), method, str(system)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        stdout, _ = process.communicate()
        partial = [json.loads(line.removeprefix("PARTIAL ")) for line in stdout.splitlines() if line.startswith("PARTIAL ")]
        return {"status": "timeout", "wall_secs": float(timeout), **({"partial": partial[-1]} if partial else {})}
    wall = time.monotonic() - started
    results = [line for line in stdout.splitlines() if line.startswith("{")]
    if not results:
        return {"status": "error", "wall_secs": wall, "error": stderr[-500:]}
    return {**json.loads(results[-1]), "wall_secs": wall}


def load_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def finished(row: dict[str, object]) -> bool:
    return row["status"] == "ok"


def gave_up(row: dict[str, object]) -> bool:
    """A failed run stops the method at larger sizes, unless Alg. PrimaryBranching still returned its isolated components."""
    return not finished(row) and not (row["label"] == "primary" and "partial" in row)


def benchmark_family(family: str, output: Path, timeout: int) -> None:
    make, sizes, _ = FAMILIES[family]
    done = {(row["size"], row["label"]): row for row in load_rows(RESULTS_FILE) if row["family"] == family}
    failed_at: dict[str, int] = {}
    for (size, label), row in sorted(done.items()):
        if gave_up(row):
            failed_at.setdefault(str(label), int(size))
    for size in sizes:
        pending = [m for m in METHODS if failed_at.get(m, size) >= size and (size, m) not in done]
        if not pending:
            continue
        path = make(output, size)
        system = convert(path)
        for method in pending:
            row = run_method(method, system, timeout)
            record = {"family": family, "size": size, "system": path.stem, "label": method, "timeout": timeout, **row}
            with RESULTS_FILE.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            print(f"{path.stem} {method} {row['status']} {float(row['wall_secs']):.3f}", flush=True)
            if gave_up(record):
                failed_at[method] = min(failed_at.get(method, size), size)


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
        "from `scaling_results.jsonl`. Cells, as (object, decomposition):",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/axes"))
    parser.add_argument("--families", default="")
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
        for family in families:
            benchmark_family(family, args.output_dir, args.timeout)
    REPORT_FILE.write_text(render_report() + "\n", encoding="utf-8")
    print(f"updated {REPORT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
