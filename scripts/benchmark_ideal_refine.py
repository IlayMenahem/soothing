#!/usr/bin/env python3
"""Run Alg. IdealRefine against Singular's std on the 20 baseline systems.

Both return a description of the ideal I itself that decides ideal membership:
std a Gröbner basis of I, Alg. IdealRefine ideals with Gröbner bases whose
intersection is I (Corollary ideal-membership). Both run in the harness of
``factor_variants.py``, one Python process driving one Singular process, timed by
wall clock; the conversion to Singular syntax is not timed. Where both finish, an
untimed run checks that the leaves intersect to I.

Examples:

    python3 scripts/benchmark_ideal_refine.py --timeout 60 --output-dir /tmp/ideal_refine
    python3 scripts/benchmark_ideal_refine.py --report-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_axes import convert, load_rows, run_method  # noqa: E402
from benchmark_idos import RESULTS, generate_matrix  # noqa: E402

RESULTS_FILE = RESULTS / "ideal_refine_baseline_results.jsonl"
REPORT_FILE = RESULTS / "ideal_refine_baseline_comparison.md"
METHODS: tuple[str, ...] = ("ideal-refine", "direct-std")


def benchmark(output: Path, timeout: int) -> None:
    done = {(row["system"], row["label"]) for row in load_rows(RESULTS_FILE)}
    for instance in generate_matrix(output):
        path = output / Path(instance.path).name
        system = convert(path)
        for method in METHODS:
            if (path.stem, method) in done:
                continue
            row = run_method(method, system, timeout)
            record = {"system": path.stem, "label": method, "timeout": timeout, **row}
            with RESULTS_FILE.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            print(f"{path.stem} {method} {row['status']} {float(row['wall_secs']):.3f}", flush=True)
        both = [r for r in load_rows(RESULTS_FILE) if r["system"] == path.stem and r["status"] == "ok"]
        if len({r["label"] for r in both}) == len(METHODS) and (path.stem, "verify-ideal-refine") not in done:
            row = run_method("verify-ideal-refine", system, 10 * timeout)
            with RESULTS_FILE.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"system": path.stem, "label": "verify-ideal-refine", **row}) + "\n")
            print(f"{path.stem} verify {row.get('exact')}", flush=True)


def cell(row: dict[str, object] | None) -> str:
    if row is None:
        return "—"
    if row["status"] != "ok":
        return "TO" if row["status"] == "timeout" else "err"
    found = f"{row['leaves']} leaves" if row["label"] == "ideal-refine" else f"{row['basis_size']} gens"
    return f"{float(str(row['wall_secs'])):.2f} ({found})"


def render_report() -> str:
    rows = load_rows(RESULTS_FILE)
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
        lines.append(f"| {system} | {cell(by_label.get('ideal-refine'))} | {cell(by_label.get('direct-std'))} | {exact} |")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/ideal_refine"))
    parser.add_argument("--report-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.report_only:
        benchmark(args.output_dir, args.timeout)
    REPORT_FILE.write_text(render_report(), encoding="utf-8")
    print(f"updated {REPORT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
