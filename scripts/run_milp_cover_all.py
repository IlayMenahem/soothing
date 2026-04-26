#!/usr/bin/env python3
"""Run the milp_cover Cargo binary for every text system in a directory."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence


OBJECTIVES = ("count", "degree", "count-degree")


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    output: str


RunCommand = Callable[[list[str], Path], CommandResult]


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run symbolica-elim's milp_cover binary on every *.txt file in systems/.",
    )
    parser.add_argument(
        "--systems-dir",
        type=Path,
        default=Path("systems"),
        help="Directory containing system *.txt files (default: systems).",
    )
    parser.add_argument(
        "--objective",
        choices=OBJECTIVES,
        default="count",
        help="MILP objective passed to milp_cover (default: count).",
    )
    build_mode = parser.add_mutually_exclusive_group()
    build_mode.add_argument(
        "--release",
        action="store_true",
        dest="release",
        default=True,
        help="Run the release binary (default).",
    )
    build_mode.add_argument(
        "--debug",
        action="store_false",
        dest="release",
        help="Run the debug binary.",
    )
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Stop after the first system whose milp_cover run exits nonzero.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        help="Optional directory for one combined stdout/stderr log per system.",
    )
    return parser.parse_args(argv)


def discover_systems(systems_dir: Path) -> list[Path]:
    return sorted(path for path in systems_dir.glob("*.txt") if path.is_file())


def build_command(system_path: Path, objective: str, release: bool) -> list[str]:
    command = ["cargo", "run"]
    if release:
        command.append("--release")
    command.extend(
        [
            "-p",
            "symbolica-elim",
            "--bin",
            "milp_cover",
            "--",
            str(system_path),
            "--objective",
            objective,
        ]
    )
    return command


def run_subprocess(command: list[str], cwd: Path) -> CommandResult:
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    output_parts: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        output_parts.append(line)
    returncode = process.wait()
    return CommandResult(returncode=returncode, output="".join(output_parts))


def write_log(log_dir: Path, system_path: Path, output: str) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"{system_path.stem}.log").write_text(output, encoding="utf-8")


def main(
    argv: Sequence[str] | None = None,
    *,
    run_command: RunCommand = run_subprocess,
    cwd: Path | None = None,
) -> int:
    options = parse_args(argv)
    workdir = Path.cwd() if cwd is None else cwd
    systems = discover_systems(options.systems_dir)
    if not systems:
        print(f"no *.txt systems found in {options.systems_dir}", file=sys.stderr)
        return 2

    failures: list[tuple[Path, int]] = []
    passed = 0

    for index, system_path in enumerate(systems, start=1):
        print(f"\n== [{index}/{len(systems)}] {system_path} ==")
        command = build_command(system_path, options.objective, options.release)
        result = run_command(command, workdir)
        if options.log_dir is not None:
            write_log(options.log_dir, system_path, result.output)

        if result.returncode == 0:
            passed += 1
        else:
            failures.append((system_path, result.returncode))
            print(f"{system_path}: exit {result.returncode}")
            if options.stop_on_failure:
                break

    print("\nSummary")
    print(f"passed: {passed}")
    print(f"failed: {len(failures)}")
    for system_path, returncode in failures:
        print(f"{system_path.name}: exit {returncode}")

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
