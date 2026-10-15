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
* ``solver_slot`` lets experiment scripts run concurrently within a resource budget:
  every solver run, in any script, holds one of ``solver_slots()`` slots
  (``RESOURCE_SHARE`` of the cores) and starts only while the running experiments
  together use less than ``RESOURCE_SHARE`` of the physical memory. Their use is the
  physical footprint of every run and its descendants, which counts swapped and
  compressed memory too. When they exceed that share, ``run`` stops the most recently
  started of several runs and starts it again once there is room, so only completed
  attempts are timed. ``concurrently`` starts independent runs at once and leaves the
  pacing to the slots.
* ``physical_system`` writes the physical system: ``systems/system.txt``
  without its last equation ``kappah*rRabKappah = 1``, the Rabinowitsch
  encoding of ``kappah != 0``. The experiments impose ``kappah != 0`` by
  saturation instead, passing ``PHYSICAL_NONZERO`` to every solver.

Baseline usage (``--nonzero X`` saturates by ``X`` before the task, inside its time):

    python3 scripts/_harness.py singular SYSTEM.txt --task gb|gb-dp|facstd|decompose --timeout-secs N [--nonzero X]
    python3 scripts/_harness.py macaulay2 SYSTEM.txt --task gb|decompose --timeout-secs N [--nonzero X]
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import ctypes.util
import fcntl
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TextIO

import sympy as sp

from _systems import load_system, mathematica_rules, min_fill_variables, poly_to_singular

REPO = Path(__file__).resolve().parent.parent
SYMBOLICA = REPO / "symbolica"
RESULTS = REPO / "results"
PHYSICAL_SOURCE = REPO / "systems" / "system.txt"
PHYSICAL_SYSTEM = RESULTS / "physical_system.txt"
PHYSICAL_RABINOWITSCH_RULE = "0 -> rRabKappah*kappah - 1"
PHYSICAL_NONZERO: tuple[str, ...] = ("kappah",)
# The Singular settings of every physical experiment (the fields of factor_variants.Settings), chosen on
# the physical systems: Singular's sat, primdecSY, and the separators of Alg. PrimaryBranching. Saturations
# and decompositions run in lex with the min-fill variable order on the physical system (Rlp) and in
# degrevlex with the variables in name order on the graviton-fermion systems (Rdp). Coefficients lie in
# Q(pi) wherever Singular and Macaulay2 allow it.
PHYSICAL_SETTINGS: dict[str, object] = {"saturation": "sat", "primdec": "primdecSY", "separators": True, "ring": "Rlp"}
GRAVFERM_SETTINGS: dict[str, object] = {**PHYSICAL_SETTINGS, "ring": "Rdp"}
BINARIES: dict[str, Path] = {
    name: SYMBOLICA / "target" / "release" / name
    for name in ("sig_groebner", "branching", "variable_zeroing")
}
ZEROING_BRANCH_TIMEOUT_MS = 3000
RESOURCE_SHARE = 0.33
SLOT_DIRECTORY = Path(tempfile.gettempdir()) / "soothing_solver_slots"
APPEND_LOCK = threading.Lock()
# Standalone baseline label -> (tool, task arguments of the baseline CLI below).
EXTERNAL: dict[str, tuple[str, list[str]]] = {
    "singular-gb-dp": ("singular", ["--task", "gb-dp"]),
    "singular-gb-lex": ("singular", ["--task", "gb"]),
    "singular-facstd": ("singular", ["--task", "facstd"]),
    "singular-primdecGTZ": ("singular", ["--task", "decompose"]),
    "singular-primdecSY": ("singular", ["--task", "decompose", "--primdec", "primdecSY"]),
    "m2-gb-lex": ("macaulay2", ["--task", "gb"]),
    "m2-decompose": ("macaulay2", ["--task", "decompose"]),
}


def physical_system() -> Path:
    """Write ``PHYSICAL_SYSTEM``, the source system without its last equation, and return its path.

    The dropped equation must be ``PHYSICAL_RABINOWITSCH_RULE``, whose condition ``kappah != 0``
    the experiments impose by saturation."""
    rules = mathematica_rules(PHYSICAL_SOURCE.read_text(encoding="utf-8"))
    dropped = " ".join(rules[-1].split())
    if dropped != PHYSICAL_RABINOWITSCH_RULE:
        raise ValueError(f"the last equation of {PHYSICAL_SOURCE} is {dropped!r}, not {PHYSICAL_RABINOWITSCH_RULE!r}")
    PHYSICAL_SYSTEM.write_text(",\n".join(rules[:-1]) + "\n", encoding="utf-8")
    return PHYSICAL_SYSTEM


def nonzero_arguments(nonzero: tuple[str, ...]) -> list[str]:
    """The ``--nonzero`` option of the Rust binaries and the baseline CLI."""
    return ["--nonzero", ",".join(nonzero)] if nonzero else []


def solver_slots() -> int:
    """The number of solver runs that may run at once across all experiment processes."""
    return max(1, int((os.cpu_count() or 1) * RESOURCE_SHARE))


def concurrently[Item, Result](function: Callable[[Item], Result], items: Iterable[Item]) -> list[Result]:
    """``function`` on every item at once, in threads, with the results in item order; the solver
    slots bound how many runs proceed. An exception of any call is raised once all have ended."""
    pending = list(items)
    if len(pending) <= 1:
        return [function(item) for item in pending]
    with ThreadPoolExecutor(max_workers=len(pending)) as executor:
        return list(executor.map(function, pending))


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


class RusageInfo(ctypes.Structure):
    """``struct rusage_info_v0`` of macOS ``proc_pid_rusage``."""

    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [
        (name, ctypes.c_uint64)
        for name in (
            "user_time", "system_time", "pkg_idle_wkups", "interrupt_wkups", "pageins",
            "wired_size", "resident_size", "phys_footprint", "proc_start_abstime", "proc_exit_abstime",
        )
    ]


LIBSYSTEM = ctypes.CDLL(ctypes.util.find_library("System"))


def footprint(pid: int) -> int:
    """The physical footprint of a process in bytes: resident, compressed and swapped memory it is
    charged for, as Activity Monitor reports. A process that has exited counts zero."""
    info = RusageInfo()
    return info.phys_footprint if LIBSYSTEM.proc_pid_rusage(pid, 0, ctypes.byref(info)) == 0 else 0


def descendants(roots: list[int]) -> set[int]:
    """The processes in the trees rooted at ``roots``, the roots included."""
    table = subprocess.run(["ps", "-A", "-o", "pid=,ppid="], capture_output=True, text=True, check=True).stdout
    children: dict[int, list[int]] = {}
    for pid, parent in (map(int, line.split()) for line in table.splitlines() if line.strip()):
        children.setdefault(parent, []).append(pid)
    found: set[int] = set()
    pending = list(roots)
    while pending:
        pid = pending.pop()
        if pid not in found:
            found.add(pid)
            pending.extend(children.get(pid, []))
    return found


@dataclass(frozen=True)
class SlotRecord:
    """The run in a held slot: its root process and its start, in seconds since the epoch."""

    root: int
    started: float


def recorded_run(path: Path) -> SlotRecord | None:
    """The run recorded in a slot file while its slot is held."""
    with path.open("r", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            fields = handle.read().split()
            return SlotRecord(int(fields[0]), float(fields[1])) if len(fields) == 2 else None
        fcntl.flock(handle, fcntl.LOCK_UN)
        return None


def running_records() -> list[SlotRecord]:
    return [record for record in map(recorded_run, SLOT_DIRECTORY.glob("slot*")) if record is not None]


def experiments_footprint(records: list[SlotRecord]) -> int:
    """The bytes the recorded runs and all their descendants use."""
    return sum(map(footprint, descendants([record.root for record in records])))


def memory_budget() -> float:
    return RESOURCE_SHARE * os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")


def has_headroom(needed: int) -> bool:
    """Whether a run that needs ``needed`` bytes fits beside the running experiments; with none
    running, any run fits."""
    records = running_records()
    return not records or experiments_footprint(records) + needed < memory_budget()


def evicted(root: int) -> bool:
    """Whether the run rooted at ``root`` must make room: the experiments exceed the memory budget
    and it is the most recently started of several runs."""
    records = running_records()
    if len(records) < 2 or experiments_footprint(records) <= memory_budget():
        return False
    return max(records, key=lambda record: record.started).root == root


@dataclass(frozen=True)
class SolverSlot:
    """A held solver slot. ``track`` records the root process of its run, which makes the run's
    memory count toward the budget."""

    handle: TextIO

    def track(self, pid: int) -> None:
        self.handle.truncate(0)
        self.handle.write(f"{pid} {time.time()}")
        self.handle.flush()


@contextlib.contextmanager
def solver_slot(needed: int = 0) -> Iterator[SolverSlot]:
    """Hold one of the solver slots shared by all experiment processes of this user, taken once
    the running experiments leave ``needed`` bytes of the memory budget. A slot is a file lock,
    so a crashed process frees it; the file is opened without truncation, so waiting never
    erases a holder's record."""
    SLOT_DIRECTORY.mkdir(exist_ok=True)
    while True:
        if has_headroom(needed):
            for index in range(solver_slots()):
                handle = (SLOT_DIRECTORY / f"slot{index}").open("a", encoding="utf-8")
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    handle.close()
                    continue
                try:
                    handle.truncate(0)
                    yield SolverSlot(handle)
                finally:
                    handle.truncate(0)
                    fcntl.flock(handle, fcntl.LOCK_UN)
                    handle.close()
                return
        time.sleep(1.0)


RunOutcome = tuple[str, str, float, bool, int]


def run_attempt(argv: list[str], timeout: float, cwd: Path | None, needed: int) -> RunOutcome | int:
    """One attempt in a solver slot that leaves ``needed`` bytes, checked every second: the
    outcome, or the bytes the run used when it was stopped to bring the experiments back within
    the memory budget."""
    with solver_slot(needed) as slot:
        start = time.perf_counter()
        process = subprocess.Popen(
            argv,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        slot.track(process.pid)
        while True:
            remaining = start + timeout - time.perf_counter()
            try:
                stdout, stderr = process.communicate(timeout=min(1.0, max(remaining, 0.0)))
                return stdout, stderr, time.perf_counter() - start, False, process.returncode
            except subprocess.TimeoutExpired:
                if remaining <= 0:
                    kill_group(process.pid)
                    stdout, stderr = process.communicate()
                    return stdout, stderr, timeout, True, -signal.SIGKILL
                if evicted(process.pid):
                    used = experiments_footprint([SlotRecord(process.pid, start)])
                    kill_group(process.pid)
                    process.communicate()
                    return used


def run(argv: list[str], timeout: float, cwd: Path | None = None) -> RunOutcome:
    """Run ``argv`` in a solver slot, as (stdout, stderr, seconds, timed out, return code). A run
    stopped for memory starts again once the budget has room for what it used; only the
    completed attempt is timed, and waiting for a slot is not."""
    needed = 0
    while isinstance(outcome := run_attempt(argv, timeout, cwd, needed), int):
        needed = max(needed, outcome)
        print(
            f"memory budget exceeded; restarting {Path(argv[0]).name} once {needed / 2**30:.1f} GB fit",
            file=sys.stderr,
            flush=True,
        )
    return outcome



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


def run_direct(path: Path, algorithm: str, timeout: int, nonzero: tuple[str, ...] = ()) -> dict[str, object]:
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
            *nonzero_arguments(nonzero),
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


def run_external(
    label: str, path: Path, timeout: int, nonzero: tuple[str, ...] = (), lex_min_fill: bool = False
) -> dict[str, object]:
    """One Singular or Macaulay2 baseline; with `lex_min_fill`, in lex with the min-fill variable order."""
    tool, extra = EXTERNAL[label]
    stdout, _, wall, killed, _ = run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            tool,
            str(path),
            *extra,
            "--timeout-secs",
            str(timeout),
            *nonzero_arguments(nonzero),
            *(["--lex-min-fill"] if lex_min_fill else []),
        ],
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
    """Append one JSON line; threads of one process never interleave their lines."""
    with APPEND_LOCK, path.open("a", encoding="utf-8") as handle:
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


def _singular_program(
    polys: list[sp.Basic],
    symbols: list[sp.Symbol],
    task: str,
    nonzero: list[str],
    lex_min_fill: bool = False,
    primdec: str = "primdecGTZ",
) -> str:
    # pi is transcendental, so it is a coefficient: the field is Q(pi).
    coefficient_symbols = [s for s in symbols if str(s) == "pi"]
    polynomial_symbols = [s for s in symbols if s not in coefficient_symbols]
    coefficients = (
        f"(0,{','.join(str(s) for s in coefficient_symbols)})"
        if coefficient_symbols
        else "0"
    )
    ordering = "dp" if task in {"facstd", "gb-dp"} and not lex_min_fill else "lp"
    names = [str(s) for s in polynomial_symbols]
    ring_vars = ",".join(min_fill_variables(polys, names) if lex_min_fill else names)
    ideal_body = ",\n    ".join(poly_to_singular(p) for p in polys)
    # Singular's `printf` only accepts a literal format with no substitution;
    # we use `print("LABEL " + string(value))` instead, which auto-newlines.
    if task in {"gb", "gb-dp"}:
        body = (
            'option(redSB);\n'
            'ideal G = std(I);\n'
            'int dt = rtimer - t0;\n'
            'print("SOLVE_MS " + string(dt));\n'
            'print("BASIS_LEN " + string(size(G)));\n'
        )
    elif task == "facstd":
        body = (
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
            f'list pd = {primdec}(I);\n'
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
    libraries = ['LIB "primdec.lib";\n'] if task == "decompose" else []
    libraries += ['LIB "elim.lib";\n'] if nonzero else []
    # Singular 4.4 returns the saturation itself, older versions a list (ideal, exponent).
    saturation = (
        f"def S = sat(I, ideal({'*'.join(nonzero)}));\n"
        'if (typeof(S) == "list") { I = S[1]; } else { I = S; }\n'
        if nonzero
        else ""
    )
    return (
        # rtimer ticks per second; default is 1 (whole seconds) — bump to 1000
        # so SOLVE_MS is in milliseconds, matching the other baselines.
        'system("--ticks-per-sec", 1000);\n'
        + "".join(libraries)
        + f"ring R = {coefficients}, ({ring_vars}), {ordering};\n"
        f"ideal I = {ideal_body};\n"
        "int t0 = rtimer;\n"
        f"{saturation}"
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


def _m2_program(
    polys: list[sp.Basic], symbols: list[sp.Symbol], task: str, nonzero: list[str], lex_min_fill: bool = False
) -> str:
    # M2 predefines several lower-case identifiers as mathematical constants
    # (e, pi, ii), and "_" is its indexed-variable operator. Use positional
    # alphanumeric names so both collisions and accidental indexing are impossible.
    rename = {str(symbol): f"v{index}" for index, symbol in enumerate(symbols)}
    # pi is transcendental, so it is a coefficient: the field is Q(pi). M2's decompose has no strategy
    # over a fraction field, so there pi stays a variable.
    coefficients = [rename[str(s)] for s in symbols if str(s) == "pi" and task == "gb"]
    field = f"frac(QQ[{','.join(coefficients)}])" if coefficients else "QQ"
    names = [str(s) for s in symbols if rename[str(s)] not in coefficients]
    ring_vars = ",".join(rename[name] for name in (min_fill_variables(polys, names) if lex_min_fill else names))
    ideal_body = ",\n  ".join(_poly_to_m2(p, rename) for p in polys)
    saturation = f"I = saturate(I, {'*'.join(rename[name] for name in nonzero)});\n" if nonzero else ""
    # cpuTime() returns floating-point seconds; ``currentTime()`` is integer-second
    # Unix epoch and rounds to 0 for fast operations.
    if task == "gb":
        body = (
            'startTime = cpuTime();\n'
            f'{saturation}'
            'G = gens gb I;\n'
            'dt = cpuTime() - startTime;\n'
            '<< "SOLVE_MS " << floor(dt * 1000) << "\\n";\n'
            '<< "BASIS_LEN " << numColumns G << "\\n";\n'
        )
    elif task == "decompose":
        body = (
            'needsPackage "PrimaryDecomposition";\n'
            'startTime = cpuTime();\n'
            f'{saturation}'
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
        f'R = {field}[{ring_vars}, MonomialOrder => Lex];\n'
        f'I = ideal(\n  {ideal_body}\n);\n'
        f'{body}'
        f'<< "DONE\\n";\n'
        f'exit 0;\n'
    )


def singular_main(args: argparse.Namespace) -> int:
    label = "singular-primdecSY" if args.task == "decompose" and args.primdec == "primdecSY" else SINGULAR_LABELS[args.task]

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

    missing = sorted(set(args.nonzero) - {str(s) for s in symbols})
    if missing:
        BaselineResult(baseline=label, status="error", error=f"--nonzero: no variable {', '.join(missing)}").emit()
        return 0

    program = _singular_program(polys, symbols, args.task, args.nonzero, args.lex_min_fill, args.primdec)

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

    missing = sorted(set(args.nonzero) - {str(s) for s in symbols})
    if missing:
        BaselineResult(baseline=label, status="error", error=f"--nonzero: no variable {', '.join(missing)}").emit()
        return 0

    program = _m2_program(polys, symbols, args.task, args.nonzero, args.lex_min_fill)

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
    parser.add_argument(
        "--nonzero",
        type=lambda names: names.split(","),
        default=[],
        help="comma-separated variables assumed nonzero; the tool saturates by their product first",
    )
    parser.add_argument(
        "--primdec",
        choices=("primdecGTZ", "primdecSY"),
        default="primdecGTZ",
        help="the primary decomposition of the Singular decompose task",
    )
    parser.add_argument(
        "--lex-min-fill",
        action="store_true",
        help="lex order with the variables in the min-fill order of the supports, for every task",
    )
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
