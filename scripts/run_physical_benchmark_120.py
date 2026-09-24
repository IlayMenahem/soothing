#!/usr/bin/env python3
"""Run the physical system benchmark for all backends with 120s limit and output Table 3."""

from __future__ import annotations

import concurrent.futures
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"

BACKENDS = ["m4gb", "f5", "gvw", "symbolica", "singular", "facstd"]


def run_branching(backend: str) -> dict:
    cmd = [sys.executable, str(SCRIPTS / "_run_parallel_branching_benchmark.py"), backend]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    summary_line = ""
    harness = {}
    for line in res.stdout.splitlines():
        if line.startswith("BRANCHING_SUMMARY "):
            summary_line = line
        elif line.startswith("HARNESS_SUMMARY "):
            harness = json.loads(line[len("HARNESS_SUMMARY "):])
    return {
        "type": "branching",
        "backend": backend,
        "summary": summary_line,
        "harness": harness,
        "stdout": res.stdout,
    }


def run_zeroing(backend: str) -> dict:
    cmd = [sys.executable, str(SCRIPTS / "_run_zeroing_benchmark.py"), backend]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    summary_line = ""
    harness = {}
    for line in res.stdout.splitlines():
        if line.startswith("ZEROING_SUMMARY "):
            summary_line = line
        elif line.startswith("HARNESS_SUMMARY "):
            harness = json.loads(line[len("HARNESS_SUMMARY "):])
    return {
        "type": "zeroing",
        "backend": backend,
        "summary": summary_line,
        "harness": harness,
        "stdout": res.stdout,
    }


def main():
    out_file = REPO / "results" / "physical_system_120s_results.json"
    branching_results = {}
    if out_file.exists():
        with open(out_file) as f:
            data = json.load(f)
            branching_results = data.get("branching", {})

    if not branching_results:
        print("=== Running Factor Branching (120s limit) in parallel ===")
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            futures = {pool.submit(run_branching, b): b for b in BACKENDS}
            for fut in concurrent.futures.as_completed(futures):
                b = futures[fut]
                res = fut.result()
                branching_results[b] = res
                print(f"[Branching] {b} done: {res['summary']}")

    print("\n=== Running Variable Zeroing (5s inner timeout, 120s total limit) in parallel ===")
    zeroing_results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(run_zeroing, b): b for b in BACKENDS}
        for fut in concurrent.futures.as_completed(futures):
            b = futures[fut]
            res = fut.result()
            zeroing_results[b] = res
            print(f"[Zeroing] {b} done: {res['summary']}")

    results = {
        "branching": branching_results,
        "zeroing": zeroing_results,
    }

    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved raw results to {out_file}")


if __name__ == "__main__":
    main()
