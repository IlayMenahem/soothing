#!/usr/bin/env python3
"""Running the solvers of the experiments and storing their results.

* ``run`` runs one command with a wall-clock limit and kills its whole process
  group on timeout, Singular children included.
* ``run_branching``, ``run_zeroing`` and ``run_direct`` run the Rust binaries
  ``branching``, ``variable_zeroing`` and ``sig_groebner`` and parse their
  summary lines.
* ``run_external`` runs one Singular or Macaulay2 baseline. Each baseline is a
  separate Python process, this file run as a script, that converts the system,
  runs the tool and prints one JSON line, so its wall time includes the Python
  start-up and the conversion, as for the Rust binaries.
* ``load_rows`` and ``append_row`` keep the resumable JSON-lines result files.

Baseline usage:

    python3 scripts/_harness.py singular SYSTEM.txt --task gb|gb-dp|facstd|decompose --timeout-secs N
    python3 scripts/_harness.py macaulay2 SYSTEM.txt --task gb|decompose --timeout-secs N
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import sympy as sp

from _systems import load_system, poly_to_singular

REPO = Path(__file__).resolve().parent.parent
SYMBOLICA = REPO / "symbolica"
RESULTS = REPO / "results"
PHYSICAL_SYSTEM = REPO / "systems" / "system.txt"
BINARIES: dict[str, Path] = {
    name: SYMBOLICA / "target" / "release" / name
    for name in ("sig_groebner", "branching", "variable_zeroing")
}
ZEROING_BRANCH_TIMEOUT_MS = 3000
# Standalone baseline label -> (tool, task arguments of the baseline CLI below).
EXTERNAL: dict[str, tuple[str, list[str]]] = {
    "singular-gb-dp": ("singular", ["--task", "gb-dp"]),
    "singular-gb-lex": ("singular", ["--task", "gb"]),
    "singular-facstd": ("singular", ["--task", "facstd"]),
    "singular-primdecGTZ": ("singular", ["--task", "decompose"]),
    "m2-gb-lex": ("macaulay2", ["--task", "gb"]),
    "m2-decompose": ("macaulay2", ["--task", "decompose"]),
}


def subprocess_limit() -> int:
    """Maximum concurrent child processes, leaving two CPUs responsive."""
    affinity = getattr(os, "sched_getaffinity", None)
    cores = len(affinity(0)) if affinity is not None else (os.cpu_count() or 1)
    return max(1, cores - 2)


FINISHED_RE = re.compile(
    r"^\s*(\w+) finished in ([0-9.]+)(ns|µs|ms|s)", re.MULTILINE
)


SUMMARY_RE = re.compile(
    r"BRANCHING_SUMMARY total=(\d+) satisfiable=(\d+) "
    r"unsatisfiable=(\d+) timeouts=(\d+) complete=(true|false) "
    r"largest_dimension=(\d+|unknown) first_solution_secs=([0-9.]+|unknown) "
    r"elapsed_secs=([0-9.]+)"
)

ZEROING_SUMMARY_RE = re.compile(
    r"ZEROING_SUMMARY nodes=(\d+) covers=\d+ evaluations=(\d+) components=(\d+) "
    r"inconsistent=\d+ timeouts=(\d+) splits=(\d+) factorization_timeouts=\d+ "
    r"factorization_errors=\d+ unresolved=(\d+) complete=(true|false) "
    r"largest_depth=\d+ first_solution_secs=([0-9.]+|unknown) elapsed_secs=([0-9.]+)"
)


def kill_group(group: int) -> None:
    """SIGKILL every process in ``group``. Once the group leader is an
    unreaped zombie, macOS refuses ``killpg`` with EPERM and signals no one, so
    fall back to the live members, e.g. Singular children of an exited solver."""
    try:
        os.killpg(group, signal.SIGKILL)
    except ProcessLookupError:
        return
    except PermissionError:
        members = subprocess.run(
            ["pgrep", "-g", str(group)], capture_output=True, text=True, check=False
        ).stdout.split()
        for member in members:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(int(member), signal.SIGKILL)


def run(
    argv: list[str], timeout: float, cwd: Path | None = None
) -> tuple[str, str, float, bool, int]:
    start = time.perf_counter()
    process = subprocess.Popen(
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
        kill_group(process.pid)
        stdout, stderr = process.communicate()
        return stdout, stderr, timeout, True, -signal.SIGKILL



def build(*binaries: str) -> None:
    result = subprocess.run(
        ["cargo", "build", "--release", *(arg for binary in binaries for arg in ("--bin", binary))],
        cwd=SYMBOLICA,
        check=False,
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
            str(BINARIES["sig_groebner"]),
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


def run_branching(
    path: Path, branching_args: list[str], timeout: int, engine: str
) -> dict[str, object]:
    method = f"branching-{engine}"
    stdout, stderr, wall, killed, returncode = run(
        [
            str(BINARIES["branching"]),
            str(path),
            *solver_args_with_engine(branching_args, engine),
            "--timeout",
            str(timeout * 1000),
            "--summary",
        ],
        timeout + 2,
    )
    if killed:
        return {
            "method": method,
            "status": "timeout",
            "solve_secs": float(timeout),
            "wall_secs": wall,
        }

    match = SUMMARY_RE.search(stdout)
    if match is None:
        return {
            "method": method,
            "status": "error",
            "solve_secs": wall,
            "wall_secs": wall,
            "error": (stderr or stdout)[-600:],
        }

    (
        total,
        satisfiable,
        unsatisfiable,
        cover_timeouts,
        complete,
        largest_dimension,
        first_solution,
        solve,
    ) = match.groups()
    return {
        "method": method,
        "status": (
            "ok"
            if returncode == 0 and complete == "true" and int(cover_timeouts) == 0
            else "partial"
        ),
        "solve_secs": float(solve),
        "wall_secs": wall,
        "total": int(total),
        "satisfiable": int(satisfiable),
        "unsatisfiable": int(unsatisfiable),
        "cover_timeouts": int(cover_timeouts),
        "complete": complete == "true",
        "first_solution_secs": (
            None if first_solution == "unknown" else float(first_solution)
        ),
        "largest_dimension": (
            None if largest_dimension == "unknown" else int(largest_dimension)
        ),
    }


def run_zeroing(
    path: Path,
    solver_args: list[str],
    timeout: int,
    engine: str,
    branch_timeout_ms: int = ZEROING_BRANCH_TIMEOUT_MS,
) -> dict[str, object]:
    method = f"zeroing-{engine}"
    stdout, stderr, wall, killed, returncode = run(
        [
            str(BINARIES["variable_zeroing"]),
            str(path),
            *solver_args_with_engine(solver_args, engine),
            "--timeout",
            str(branch_timeout_ms),
            "--max-runtime",
            str(timeout * 1000),
            "--ideal-saturation",
            "--summary",
        ],
        timeout + 2,
    )
    if killed:
        return {
            "method": method,
            "status": "timeout",
            "solve_secs": float(timeout),
            "wall_secs": wall,
        }

    match = ZEROING_SUMMARY_RE.search(stdout)
    if match is None:
        return {
            "method": method,
            "status": "error",
            "solve_secs": wall,
            "wall_secs": wall,
            "error": (stderr or stdout)[-600:],
        }

    (
        nodes,
        evaluations,
        components,
        branch_timeouts,
        splits,
        unresolved,
        complete,
        first_solution,
        solve,
    ) = match.groups()
    return {
        "method": method,
        "status": "ok" if returncode == 0 and complete == "true" else "partial",
        "solve_secs": float(solve),
        "wall_secs": wall,
        "nodes": int(nodes),
        "evaluations": int(evaluations),
        "components": int(components),
        "branch_timeouts": int(branch_timeouts),
        "splits": int(splits),
        "unresolved": int(unresolved),
        "first_solution_secs": (
            None if first_solution == "unknown" else float(first_solution)
        ),
    }


def run_external(label: str, path: Path, timeout: int) -> dict[str, object]:
    tool, extra = EXTERNAL[label]
    stdout, _, wall, killed, _ = run(
        [sys.executable, str(Path(__file__).resolve()), tool, str(path), *extra, "--timeout-secs", str(timeout)],
        timeout + 2,
    )
    row = parse_baseline(stdout, label, wall, killed)
    if killed:
        row["solve_secs"] = float(timeout)
    return row


def load_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def append_row(path: Path, record: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


@dataclass
class BaselineResult:
    """Unified result shape emitted by every baseline driver."""

    baseline: str
    status: str  # "ok" | "timeout" | "tool_missing" | "tool_unavailable" | "error" | "parse_error"
    elapsed_secs: float = 0.0
    # Algorithm-specific metrics — only the relevant fields are populated.
    basis_len: int | None = None
    num_components: int | None = None
    max_dim: int | None = None
    error: str | None = None
    tool_stdout_tail: str | None = None
    extra: dict[str, object] = field(default_factory=dict)

    def emit(self) -> None:
        """Print the JSON form on a single line (final line of stdout)."""
        d = asdict(self)
        # Drop None fields to keep the JSON compact and make missing-vs-zero unambiguous.
        d = {k: v for k, v in d.items() if v is not None and v != {}}
        sys.stdout.write(json.dumps(d) + "\n")
        sys.stdout.flush()


def require_tool(name: str, baseline_label: str) -> str | None:
    """Return path to ``name`` on PATH, or print a ``tool_missing`` result and return None."""
    found = shutil.which(name)
    if found is None:
        BaselineResult(baseline=baseline_label, status="tool_missing", error=f"{name!r} not on PATH").emit()
    return found


def run_subprocess(
    argv: list[str],
    *,
    input_text: str | None = None,
    timeout: float | None = None,
) -> tuple[int, str, str, float]:
    """Run a subprocess and return ``(returncode, stdout, stderr, walltime_secs)``.

    A wall-clock timeout raises ``subprocess.TimeoutExpired``; the caller is
    expected to translate that into a ``"timeout"`` BaselineResult.
    """
    start = time.perf_counter()
    completed = subprocess.run(
        argv,
        input=input_text,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return completed.returncode, completed.stdout, completed.stderr, time.perf_counter() - start


def tail(text: str, n_chars: int = 600) -> str:
    if len(text) <= n_chars:
        return text
    return "...\n" + text[-n_chars:]


SINGULAR_LABELS: dict[str, str] = {
    "gb": "singular-gb-lex",
    "gb-dp": "singular-gb-dp",
    "facstd": "singular-facstd",
    "decompose": "singular-primdecGTZ",
}


def _singular_program(polys: list[sp.Basic], symbols: list[sp.Symbol], task: str) -> str:
    # The Rust loader recognizes Mathematica's \[Pi] as the
    # coefficient ``pi`` and excludes it from the polynomial variable order.
    # Match that interpretation for facstd with the transcendental coefficient
    # field Q(pi).  The older baseline modes retain their historical behavior.
    coefficient_symbols = [s for s in symbols if task == "facstd" and str(s) == "pi"]
    polynomial_symbols = [s for s in symbols if s not in coefficient_symbols]
    coefficients = (
        f"(0,{','.join(str(s) for s in coefficient_symbols)})"
        if coefficient_symbols
        else "0"
    )
    ordering = "dp" if task in {"facstd", "gb-dp"} else "lp"
    ring_vars = ",".join(str(s) for s in polynomial_symbols)
    ideal_body = ",\n    ".join(poly_to_singular(p) for p in polys)
    # Singular's `printf` only accepts a literal format with no substitution;
    # we use `print("LABEL " + string(value))` instead, which auto-newlines.
    if task in {"gb", "gb-dp"}:
        body = (
            'option(redSB);\n'
            'int t0 = rtimer;\n'
            'ideal G = std(I);\n'
            'int dt = rtimer - t0;\n'
            'print("SOLVE_MS " + string(dt));\n'
            'print("BASIS_LEN " + string(size(G)));\n'
        )
    elif task == "facstd":
        body = (
            'int t0 = rtimer;\n'
            'ideal FI = interred(I);\n'
            'list F = facstd(FI);\n'
            'int dt = rtimer - t0;\n'
            'print("SOLVE_MS " + string(dt));\n'
            'print("NUM_COMPONENTS " + string(size(F)));\n'
            'int i;\n'
            'int maxd = -1;\n'
            'ideal J;\n'
            'for (i = 1; i <= size(F); i = i + 1) {\n'
            '  J = F[i];\n'
            '  int d = dim(J);\n'
            '  if (d > maxd) { maxd = d; }\n'
            '}\n'
            'print("MAX_DIM " + string(maxd));\n'
        )
    elif task == "decompose":
        body = (
            'LIB "primdec.lib";\n'
            'int t0 = rtimer;\n'
            'list pd = primdecGTZ(I);\n'
            'int dt = rtimer - t0;\n'
            'print("SOLVE_MS " + string(dt));\n'
            'print("NUM_COMPONENTS " + string(size(pd)));\n'
            'int i;\n'
            'int maxd = -1;\n'
            'for (i = 1; i <= size(pd); i = i + 1) {\n'
            '  int d = dim(std(pd[i][2]));\n'
            '  if (d > maxd) { maxd = d; }\n'
            '}\n'
            'print("MAX_DIM " + string(maxd));\n'
        )
    else:
        raise ValueError(f"unknown task: {task}")
    return (
        # rtimer ticks per second; default is 1 (whole seconds) — bump to 1000
        # so SOLVE_MS is in milliseconds, matching the other baselines.
        'system("--ticks-per-sec", 1000);\n'
        f"ring R = {coefficients}, ({ring_vars}), {ordering};\n"
        f"ideal I = {ideal_body};\n"
        f"{body}"
        f'print("DONE");\n'
        f"exit;\n"
    )


def parse_tool_output(stdout: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for line in stdout.splitlines():
        m = re.match(r"^(SOLVE_MS|BASIS_LEN|NUM_COMPONENTS|MAX_DIM)\s+(-?\d+)\s*$", line.strip())
        if m:
            out[m.group(1)] = int(m.group(2))
    return out


MACAULAY2_LABELS: dict[str, str] = {
    "gb": "m2-gb-lex",
    "decompose": "m2-decompose",
}


def _poly_to_m2(poly: sp.Basic, rename: dict[str, str]) -> str:
    text = sp.sstr(sp.expand(poly))
    text = text.replace("**", "^")
    if rename:
        pattern = re.compile(r"\b(" + "|".join(re.escape(k) for k in rename) + r")\b")
        text = pattern.sub(lambda m: rename[m.group(1)], text)
    return text


def _m2_program(polys: list[sp.Basic], symbols: list[sp.Symbol], task: str) -> str:
    # M2 predefines several lower-case identifiers as mathematical constants
    # (e, pi, ii), and "_" is its indexed-variable operator. Use positional
    # alphanumeric names so both collisions and accidental indexing are impossible.
    rename = {str(symbol): f"v{index}" for index, symbol in enumerate(symbols)}
    ring_vars = ",".join(rename[str(s)] for s in symbols)
    ideal_body = ",\n  ".join(_poly_to_m2(p, rename) for p in polys)
    # cpuTime() returns floating-point seconds; ``currentTime()`` is integer-second
    # Unix epoch and rounds to 0 for fast operations.
    if task == "gb":
        body = (
            'startTime = cpuTime();\n'
            'G = gens gb I;\n'
            'dt = cpuTime() - startTime;\n'
            '<< "SOLVE_MS " << floor(dt * 1000) << "\\n";\n'
            '<< "BASIS_LEN " << numColumns G << "\\n";\n'
        )
    elif task == "decompose":
        body = (
            'needsPackage "PrimaryDecomposition";\n'
            'startTime = cpuTime();\n'
            'comps = decompose I;\n'
            'dt = cpuTime() - startTime;\n'
            '<< "SOLVE_MS " << floor(dt * 1000) << "\\n";\n'
            '<< "NUM_COMPONENTS " << #comps << "\\n";\n'
            'maxd = -1;\n'
            'scan(comps, c -> (d = dim c; if d > maxd then maxd = d));\n'
            '<< "MAX_DIM " << maxd << "\\n";\n'
        )
    else:
        raise ValueError(f"unknown task: {task}")
    return (
        f'R = QQ[{ring_vars}, MonomialOrder => Lex];\n'
        f'I = ideal(\n  {ideal_body}\n);\n'
        f'{body}'
        f'<< "DONE\\n";\n'
        f'exit 0;\n'
    )


def singular_main(args: argparse.Namespace) -> int:
    label = SINGULAR_LABELS[args.task]

    tool = require_tool("Singular", label)
    if tool is None:
        return 0  # tool_missing already emitted

    try:
        polys, symbols = load_system(args.system_path)
    except Exception as exc:  # noqa: BLE001 — any parse failure becomes an emitted result.
        BaselineResult(baseline=label, status="parse_error", error=str(exc)).emit()
        return 0

    if not polys:
        BaselineResult(baseline=label, status="ok", elapsed_secs=0.0, basis_len=0).emit()
        return 0

    program = _singular_program(polys, symbols, args.task)

    try:
        rc, out, err, wall = run_subprocess(
            [tool, "-q", "--no-tty", "-c", program],
            timeout=args.timeout_secs,
        )
    except subprocess.TimeoutExpired:
        BaselineResult(baseline=label, status="timeout", elapsed_secs=args.timeout_secs).emit()
        return 0

    parsed = parse_tool_output(out)
    if "SOLVE_MS" not in parsed or "DONE" not in out:
        BaselineResult(
            baseline=label,
            status="error",
            elapsed_secs=wall,
            error=f"Singular rc={rc}",
            tool_stdout_tail=tail(out + "\n--- stderr ---\n" + err),
        ).emit()
        return 0

    result = BaselineResult(
        baseline=label,
        status="ok",
        elapsed_secs=parsed["SOLVE_MS"] / 1000.0,
    )
    if args.task in {"gb", "gb-dp"}:
        result.basis_len = parsed.get("BASIS_LEN")
    elif args.task in {"facstd", "decompose"}:
        result.num_components = parsed.get("NUM_COMPONENTS")
        result.max_dim = parsed.get("MAX_DIM")
    result.emit()
    return 0


def macaulay2_main(args: argparse.Namespace) -> int:
    label = MACAULAY2_LABELS[args.task]

    tool = require_tool("M2", label)
    if tool is None:
        return 0

    try:
        polys, symbols = load_system(args.system_path)
    except Exception as exc:  # noqa: BLE001
        BaselineResult(baseline=label, status="parse_error", error=str(exc)).emit()
        return 0

    if not polys:
        BaselineResult(baseline=label, status="ok", elapsed_secs=0.0, basis_len=0).emit()
        return 0

    program = _m2_program(polys, symbols, args.task)

    try:
        rc, out, err, wall = run_subprocess(
            [tool, "--no-readline", "--silent", "--stop", "-q", "-e", program],
            timeout=args.timeout_secs,
        )
    except subprocess.TimeoutExpired:
        BaselineResult(baseline=label, status="timeout", elapsed_secs=args.timeout_secs).emit()
        return 0

    parsed = parse_tool_output(out)
    if "SOLVE_MS" not in parsed or "DONE" not in out:
        if rc != 0 and "Library not loaded:" in err:
            match = re.search(r"Library not loaded:\s*(\S+)", err)
            missing = match.group(1) if match else "an unresolved dynamic-library dependency"
            BaselineResult(
                baseline=label,
                status="tool_unavailable",
                elapsed_secs=wall,
                error=f"installed M2 cannot start; missing dynamic library {missing}",
            ).emit()
            return 0
        BaselineResult(
            baseline=label,
            status="error",
            elapsed_secs=wall,
            error=f"M2 rc={rc}",
            tool_stdout_tail=tail(out + "\n--- stderr ---\n" + err),
        ).emit()
        return 0

    result = BaselineResult(
        baseline=label,
        status="ok",
        elapsed_secs=parsed["SOLVE_MS"] / 1000.0,
    )
    if args.task == "gb":
        result.basis_len = parsed.get("BASIS_LEN")
    elif args.task == "decompose":
        result.num_components = parsed.get("NUM_COMPONENTS")
        result.max_dim = parsed.get("MAX_DIM")
    result.emit()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one Singular or Macaulay2 baseline and print one JSON line.")
    parser.add_argument("tool", choices=("singular", "macaulay2"))
    parser.add_argument("system_path", type=Path)
    parser.add_argument("--task", required=True)
    parser.add_argument("--timeout-secs", type=float, default=120.0)
    args = parser.parse_args()
    tasks = SINGULAR_LABELS if args.tool == "singular" else MACAULAY2_LABELS
    if args.task not in tasks:
        parser.error(f"--task for {args.tool} is one of {', '.join(tasks)}")
    return args


def main() -> int:
    args = parse_args()
    return singular_main(args) if args.tool == "singular" else macaulay2_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
