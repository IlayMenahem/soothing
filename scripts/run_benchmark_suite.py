#!/usr/bin/env python3
"""
End-to-end benchmark pipeline: generate polynomial systems, then compare
Gröbner-basis vs MILP-cover timings and write the results to a CSV file.

Steps
-----
1. Generate benchmark systems into ``--output-dir`` using
   ``systems/generate_all_benchmark_systems.py``.
2. Build the ``bench`` Rust binary (``cargo build --release --bin bench``).
3. Run ``bench`` on the generated directory, forwarding ``--csv`` and
   ``--timeout``; the binary writes the CSV and prints a Markdown table.

The CSV is written to ``results/`` at the repo root by default.

Usage
-----
    python scripts/run_benchmark_suite.py --max-size 5
    python scripts/run_benchmark_suite.py --max-size 8 --milp-bench --timeout 60
    python scripts/run_benchmark_suite.py --max-size 3 --csv /tmp/out.csv --no-build
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GENERATOR = REPO_ROOT / "systems" / "generate_all_benchmark_systems.py"
SYMBOLICA_DIR = REPO_ROOT / "symbolica"
BENCH_BIN = REPO_ROOT / "target" / "release" / "bench"
RESULTS_DIR = REPO_ROOT / "results"


def generate_systems(output_dir: Path, max_size: int, milp_bench: bool) -> None:
    cmd: list[str] = [
        sys.executable,
        str(GENERATOR),
        "--max-size", str(max_size),
        "--output-dir", str(output_dir),
    ]
    if milp_bench:
        cmd.append("--milp-bench")
    print(f"[1/3] Generating systems into {output_dir} ...", flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(result.returncode)


def build_bench() -> None:
    print("[2/3] Building bench binary ...", flush=True)
    result = subprocess.run(
        ["cargo", "build", "--release", "--bin", "bench"],
        cwd=str(SYMBOLICA_DIR),
    )
    if result.returncode != 0:
        sys.exit(result.returncode)


def run_bench(
    systems_dir: Path,
    csv_path: Path,
    timeout: int,
    only_baselines: list[str] | None,
) -> None:
    print(f"[3/3] Running bench (timeout={timeout}s) ...", flush=True)
    argv = [
        str(BENCH_BIN),
        str(systems_dir),
        "--csv", str(csv_path),
        "--timeout", str(timeout),
    ]
    if only_baselines:
        argv += ["--only", ",".join(only_baselines)]
    result = subprocess.run(argv)
    if result.returncode != 0:
        sys.exit(result.returncode)
    print(f"\nResults written to {csv_path}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate benchmark systems and compare Gröbner vs MILP solvers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--max-size", type=int, required=True, metavar="N",
                   help="Largest primary size parameter for the system families.")
    p.add_argument("--output-dir", type=Path, default=None, metavar="DIR",
                   help="Where to write .txt system files (default: auto temp dir).")
    p.add_argument("--csv", type=Path,
                   default=RESULTS_DIR / "bench_results.csv",
                   help="CSV output path.")
    p.add_argument("--timeout", type=int, default=120, metavar="SECS",
                   help="Per-system timeout forwarded to the bench binary.")
    p.add_argument("--milp-bench", action="store_true",
                   help="Also generate MILP-friendly families (chemical, distances, gauge, chain).")
    p.add_argument("--no-build", action="store_true",
                   help="Skip cargo build (assume bench binary already exists).")
    p.add_argument("--only", type=str, default=None,
                   help="Comma-separated baseline labels to run (default: all available).")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.output_dir is not None:
        systems_dir = args.output_dir
        systems_dir.mkdir(parents=True, exist_ok=True)
        tmp = None
    else:
        tmp = tempfile.TemporaryDirectory(prefix="bench_systems_")
        systems_dir = Path(tmp.name)

    try:
        generate_systems(systems_dir, args.max_size, args.milp_bench)

        if not args.no_build:
            build_bench()
        elif not BENCH_BIN.exists():
            print(f"error: bench binary not found at {BENCH_BIN}; run without --no-build",
                  file=sys.stderr)
            return 1

        args.csv.parent.mkdir(parents=True, exist_ok=True)
        only_baselines = (
            [s.strip() for s in args.only.split(",") if s.strip()]
            if args.only is not None
            else None
        )
        run_bench(systems_dir, args.csv, args.timeout, only_baselines)
    finally:
        if tmp is not None:
            tmp.cleanup()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
