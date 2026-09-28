"""Shared helpers for baseline driver scripts.

Each baseline driver (Singular / Macaulay2 / HomotopyContinuation.jl) loads a
Mathematica-style polynomial system from a `.txt` file, translates it into the
syntax of its target tool, runs the tool as a subprocess with a wall-clock
timeout, and prints a single JSON line in the unified ``BaselineResult`` shape
below. ``scripts/benchmark_idos.py`` consumes those JSON lines as the
external computer-algebra baselines.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Ensure ``parse_mathematica`` (sibling module under ``scripts/``) is importable
# regardless of how this file is invoked.
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import sympy as sp  # noqa: E402

from parse_mathematica import mathematica_to_sympy  # noqa: E402


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
    num_solutions: int | None = None
    dim_estimate: int | None = None
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


_CONSTANT_REPLACEMENTS: tuple[tuple[sp.Basic, sp.Symbol], ...] = (
    # ``mathematica_to_sympy`` maps Mathematica's Pi/E/I to SymPy's *constants*
    # (sp.pi, sp.E, sp.I). Those aren't returned by ``free_symbols``, so external
    # CAS drivers (Singular, M2, Julia) silently get a ring that's missing them
    # while the polynomial bodies still contain ``pi^2`` etc.
    # Convert them to plain symbols so they're explicitly declared as ring vars.
    (sp.pi, sp.Symbol("pi")),
    (sp.E, sp.Symbol("e_const")),
    (sp.I, sp.Symbol("i_unit")),
)


def load_system(path: Path) -> tuple[list[sp.Basic], list[sp.Symbol]]:
    """Parse a Mathematica-style system file to (polynomials, sorted-variables).

    Any occurrence of a SymPy constant that *acts* like a variable for the
    purposes of the polynomial ring (``pi``, ``E``, ``I``) is rewritten as a
    plain symbol so the downstream CAS sees a complete variable list.
    """
    text = path.read_text(encoding="utf-8")
    polys, symbols = mathematica_to_sympy(text)
    substituted: list[sp.Basic] = []
    for p in polys:
        q = p
        for const, sym in _CONSTANT_REPLACEMENTS:
            if q.has(const):
                q = q.subs(const, sym)
        substituted.append(q)
    # Collect symbols *after* substitution.
    all_symbols: set[sp.Symbol] = set()
    for q in substituted:
        all_symbols.update(q.free_symbols)
    # Keep any pre-existing symbols too (defensive).
    all_symbols.update(symbols)
    symbols_in_order = sorted(all_symbols, key=str)
    return substituted, symbols_in_order


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
    import time as _time

    start = _time.perf_counter()
    completed = subprocess.run(  # noqa: S603 — argv is trusted (built by us).
        argv,
        input=input_text,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return completed.returncode, completed.stdout, completed.stderr, _time.perf_counter() - start


def tail(text: str, n_chars: int = 600) -> str:
    if len(text) <= n_chars:
        return text
    return "...\n" + text[-n_chars:]
