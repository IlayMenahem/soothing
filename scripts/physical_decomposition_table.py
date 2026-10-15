#!/usr/bin/env python3
"""Table ``tab:physical-decomposition``: the pairs of Table ``tab:scaling-decomposition`` on the physical systems.

Systems, written to results/physical_systems/:

    physical           results/physical_system.txt (_harness.physical_system), kappah != 0
    graviton_alpha_A   systems/system_gravfermscal_EH_d5_alpha_A.txt with d1 = d2 = d3 = 0
    fermion_alpha_A    the same with every variable whose name begins with Dp or p set to 0

for A in 1, 1_2, 1_4, 2, 4 (alpha = 1, 1/2, 1/4, 2, 4). The graviton-fermion systems drop their last
equation, the Rabinowitsch equation KappaRefFinal*RabKappaRefFinal - 1, and keep KappaRefFinal != 0 by
saturation, as the physical system keeps kappah != 0.

Every run has a 300-second limit, runs concurrently with the others, and reports
its internal solver time. The pairs:

    facstd      Alg. Branching with the singular-facstd backend, as in physical_table.py (Rust)
                / Singular facstd after saturation (the singular-facstd baseline of _harness.py)
    primdec     Alg. PrimaryBranching / primdecSY of the saturation         (factor_variants.py)
    minAssGTZ   Alg. Branching with minAssGTZ / minAssGTZ of the saturation  (factor_variants.py)

The factor_variants.py runs factor and decide emptiness in lex with the min-fill variable order, over
Q(pi), with the settings of _harness: Singular's sat, primdecSY, and the separators; saturations and
decompositions run in lex with the min-fill order on the physical system (PHYSICAL_SETTINGS) and in
degrevlex with the variables in name order on the graviton-fermion systems (GRAVFERM_SETTINGS). facstd
alone uses the same ring.

Usage: physical_decomposition_table.py [--no-build] [--report-only]
Results: results/physical_decomposition.jsonl.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import sympy as sp
from _harness import (
    BINARIES,
    GRAVFERM_SETTINGS,
    PHYSICAL_NONZERO,
    PHYSICAL_SETTINGS,
    REPO,
    RESULTS,
    append_row,
    build,
    concurrently,
    load_rows,
    nonzero_arguments,
    physical_system,
    run,
    run_external,
)
from _systems import load_system
from physical_primary_decomposition import load as load_singular_system
from physical_table import summary_fields
from scaling_decomposition_table import run_method as run_factor_variant

SYSTEMS_DIR = RESULTS / "physical_systems"
RESULTS_FILE = RESULTS / "physical_decomposition.jsonl"
TIMEOUT_SECS = 300
SATURATION_MS = 5_000
ALPHAS: dict[str, str] = {"1": "1", "1_2": "1/2", "1_4": "1/4", "2": "2", "4": "4"}
GRAVFERM_NONZERO: tuple[str, ...] = ("KappaRefFinal",)
GRAVFERM_RABINOWITSCH = sp.sympify("KappaRefFinal*RabKappaRefFinal - 1")
SPECIALIZATIONS: dict[str, tuple[str, Callable[[str], bool]]] = {
    "graviton": ("Graviton only", lambda name: re.fullmatch(r"d[0-9]", name) is not None),
    "fermion": ("Fermion only", lambda name: name.startswith(("Dp", "p")) and name != "pi"),
}
# (column group, our method, the Singular command alone)
PAIRS: tuple[tuple[str, str, str], ...] = (
    ("\\texttt{facstd}", "fb-facstd", "facstd"),
    ("primdec", "primary", "direct-primdec"),
    ("\\texttt{minAssGTZ}", "prime", "direct-minAssGTZ"),
)
METHODS: tuple[str, ...] = tuple(method for _, ours, alone in PAIRS for method in (ours, alone))


@dataclass(frozen=True)
class System:
    name: str
    label: str
    group: str
    source: Path
    nonzero: tuple[str, ...]
    settings: dict[str, object]

    @property
    def singular(self) -> Path:
        return SYSTEMS_DIR / f"{self.name}.singular.json"


def specialized(source: Path, zeroed: Callable[[str], bool]) -> list[sp.Expr]:
    """The equations of a graviton-fermion system without its Rabinowitsch equation, with the `zeroed`
    variables set to 0; equations that vanish are dropped."""
    polynomials, symbols = load_system(source)
    if sp.expand(polynomials[-1] - GRAVFERM_RABINOWITSCH) != 0:
        raise ValueError(f"the last equation of {source} is not {GRAVFERM_RABINOWITSCH}")
    zero = {symbol: 0 for symbol in symbols if zeroed(str(symbol))}
    return [p for p in (sp.expand(q.subs(zero)) for q in polynomials[:-1]) if p != 0]


def write_mathematica(path: Path, polynomials: list[sp.Expr]) -> Path:
    path.write_text("{" + ",\n".join(f"0 -> {sp.mathematica_code(p)}" for p in polynomials) + "}\n", encoding="utf-8")
    return path


def write_singular(system: System) -> None:
    """The system in the JSON format of factor_variants.py, with the min-fill lex order and the nonzero variables."""
    loaded = load_singular_system(system.source)
    payload = {
        "variables": loaded.variables,
        "parameters": loaded.parameters,
        "polynomials": loaded.polynomials,
        "factor_variables": loaded.min_fill,
        "nonzero": list(system.nonzero),
        "settings": system.settings,
    }
    system.singular.write_text(json.dumps(payload), encoding="utf-8")


def systems() -> list[System]:
    """Write every system (Mathematica for the Rust binary, JSON for factor_variants.py) and return them."""
    SYSTEMS_DIR.mkdir(exist_ok=True)
    result = [System("physical", "Physical", "physical", physical_system(), PHYSICAL_NONZERO, PHYSICAL_SETTINGS)]
    for group, (title, zeroed) in SPECIALIZATIONS.items():
        for suffix, alpha in ALPHAS.items():
            name = f"{group}_alpha_{suffix}"
            source = REPO / "systems" / f"system_gravfermscal_EH_d5_alpha_{suffix}.txt"
            path = write_mathematica(SYSTEMS_DIR / f"{name}.txt", specialized(source, zeroed))
            label = f"{title}, $\\alpha={alpha}$"
            result.append(System(name, label, group, path, GRAVFERM_NONZERO, GRAVFERM_SETTINGS))
    for system in result:
        write_singular(system)
    return result


def run_factor_branching(system: System) -> dict[str, object]:
    """Alg. Branching with the facstd backend; its time T includes the saturation, as in Table tab:physical-system."""
    argv = [
        str(BINARIES["branching"]),
        str(system.source),
        "--gb",
        "singular-facstd",
        "--timeout",
        str(TIMEOUT_SECS * 1000),
        "--saturation-timeout",
        str(SATURATION_MS),
        *nonzero_arguments(system.nonzero),
        "--summary",
    ]
    stdout, _, wall, killed, _ = run(argv, TIMEOUT_SECS + 5)
    summary = next((line for line in stdout.splitlines() if line.startswith("BRANCHING_SUMMARY ")), "")
    fields = summary_fields(summary)
    if killed or fields.get("complete") != "true":
        return {"status": "timeout", "wall_secs": wall, "solver_secs": None, "summary": summary}
    secs = float(fields["elapsed_secs"]) + float(fields.get("saturation_secs", "0"))
    return {"status": "ok", "wall_secs": wall, "solver_secs": secs, "summary": summary}


def measure(system: System, method: str) -> dict[str, object]:
    if method == "fb-facstd":
        row = run_factor_branching(system)
    elif method == "facstd":
        lex_min_fill = system.settings["ring"] == "Rlp"
        row = run_external("singular-facstd", system.source, TIMEOUT_SECS, system.nonzero, lex_min_fill)
        row["solver_secs"] = row.get("solve_secs") if row["status"] == "ok" else None
    else:
        row = run_factor_variant(method, system.singular, TIMEOUT_SECS)
        row["solver_secs"] = row.get("internal_secs") if row["status"] == "ok" else None
    record = {"system": system.name, "label": method, "timeout": TIMEOUT_SECS, **row}
    append_row(RESULTS_FILE, record)
    print(f"{system.name} {method} {row['status']} {row['solver_secs']}", flush=True)
    return record


def benchmark(all_systems: list[System]) -> None:
    done = {(str(row["system"]), str(row["label"])) for row in load_rows(RESULTS_FILE)}
    pending = [(s, m) for s in all_systems for m in METHODS if (s.name, m) not in done]
    concurrently(lambda job: measure(*job), pending)


def seconds(row: dict[str, object] | None) -> float | None:
    """The internal solver time of a completed run."""
    if row is None or row["status"] != "ok":
        return None
    return float(str(row["solver_secs"]))


def time_cell(value: float | None, other: float | None) -> str:
    if value is None:
        return "timeout"
    # Singular's own timer counts whole milliseconds.
    shown = "<0.001" if value < 0.0005 else f"{value:.3f}" if value < 10 else f"{value:.4g}"
    return f"$\\mathbf{{{shown}}}$" if other is None or value < other else f"${shown}$"


def latex_rows(all_systems: list[System]) -> list[str]:
    rows = {(str(r["system"]), str(r["label"])): r for r in load_rows(RESULTS_FILE)}
    lines: list[str] = []
    for index, system in enumerate(all_systems):
        if index and system.group != all_systems[index - 1].group:
            lines.append("\\hline")
        cells: list[str] = []
        for _, ours, alone in PAIRS:
            ours_secs, alone_secs = seconds(rows.get((system.name, ours))), seconds(rows.get((system.name, alone)))
            cells += [time_cell(ours_secs, alone_secs), time_cell(alone_secs, ours_secs)]
        lines.append(f"{system.label} & " + " & ".join(cells) + " \\\\")
    return lines


def agreement(all_systems: list[System]) -> list[str]:
    """Where both members of a pair finish, whether they return the same primes or isolated components."""
    rows = {(str(r["system"]), str(r["label"])): r for r in load_rows(RESULTS_FILE)}
    keys = {"primary": "isolated_hashes", "prime": "prime_hashes"}
    lines: list[str] = []
    for system in all_systems:
        for ours, alone in (("primary", "direct-primdec"), ("prime", "direct-minAssGTZ")):
            a, b = rows.get((system.name, ours)), rows.get((system.name, alone))
            if a and b and seconds(a) is not None and seconds(b) is not None:
                same = a.get(keys[ours]) == b.get(keys[ours])
                lines.append(f"{system.name} {ours}: {'same' if same else 'DIFFERENT'} as {alone}")
    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-build", action="store_true", help="do not build the Rust branching binary first")
    parser.add_argument("--report-only", action="store_true", help="render the table rows from the stored results")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    all_systems = systems()
    if not args.report_only:
        if not args.no_build:
            build("branching")
        benchmark(all_systems)
    print("\n".join(latex_rows(all_systems)))
    print("\n".join(agreement(all_systems)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
