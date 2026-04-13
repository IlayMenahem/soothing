#!/usr/bin/env python3
"""
Generate every benchmark system supported by ``generate_benchmark_systems.py`` for all
sizes up to ``--max-size``.

For each size-parameterized family, this writes one ``.txt`` file per integer in the
valid range (e.g. cyclic for n = 2, 3, …, max-size). Size-independent families (Nash)
emit a single file; ``adjacent_minors`` emits one file per pair ``(rows, cols)`` with
``2 <= rows <= max-size`` and ``2 <= cols <= max-size``.

Output files are Mathematica-style rule lists compatible with ``soothing-linear``'s
``load_mathematica_system_from_str``.

Requires SymPy (same as ``generate_benchmark_systems.py``).
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


def _load_generator():
    here = Path(__file__).resolve().parent
    path = here / "generate_benchmark_systems.py"
    spec = importlib.util.spec_from_file_location("generate_benchmark_systems", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    p = argparse.ArgumentParser(
        description="Write all benchmark systems up to a maximum size into a directory.",
    )
    p.add_argument(
        "--max-size",
        type=int,
        required=True,
        metavar="N",
        help="Largest primary size n (cyclic, katsura, noonburg, eco, sparse_toric). "
        "Also caps rows/cols for adjacent_minors.",
    )
    p.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write ``*.txt`` files into (created if missing).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print paths that would be written without writing files.",
    )
    args = p.parse_args()

    nmax = args.max_size
    if nmax < 1:
        print("--max-size must be >= 1", file=sys.stderr)
        return 2

    g = _load_generator()
    rules_from_equations = g.rules_from_equations
    out: Path = args.output_dir

    if not args.dry_run:
        out.mkdir(parents=True, exist_ok=True)

    def write(name: str, text: str) -> None:
        path = out / name
        if args.dry_run:
            print(path)
        else:
            path.write_text(text + "\n", encoding="utf-8")

    count = 0

    # cyclic: n >= 2
    for n in range(2, nmax + 1):
        text = rules_from_equations(g.cyclic(n))
        write(f"cyclic_n{n}.txt", text)
        count += 1

    # katsura: n >= 1
    for n in range(1, nmax + 1):
        text = rules_from_equations(g.katsura(n))
        write(f"katsura_n{n}.txt", text)
        count += 1

    # noonburg: k >= 1
    for k in range(1, nmax + 1):
        text = rules_from_equations(g.noonburg(k))
        write(f"noonburg_k{k}.txt", text)
        count += 1

    # eco: n >= 2
    for n in range(2, nmax + 1):
        text = rules_from_equations(g.eco(n))
        write(f"eco_n{n}.txt", text)
        count += 1

    # sparse_toric: n >= 2
    for n in range(2, nmax + 1):
        text = rules_from_equations(g.sparse_toric(n))
        write(f"sparse_toric_n{n}.txt", text)
        count += 1

    # nash: fixed 2×2 instance
    text = rules_from_equations(g.nash_2x2())
    write("nash_2x2.txt", text)
    count += 1

    # adjacent_minors: all r,c in [2, nmax]
    if nmax >= 2:
        for n in range(2, nmax + 1):
            text = rules_from_equations(g.adjacent_minors(n, n))
            write(f"adjacent_minors_r{n}_c{n}.txt", text)
            count += 1

    if not args.dry_run:
        print(f"Wrote {count} files under {out.resolve()}", file=sys.stderr)
    else:
        print(f"{count} files (dry-run)", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
