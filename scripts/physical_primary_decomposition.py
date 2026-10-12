#!/usr/bin/env python3
"""Primary decomposition of the physical system by Alg. PrimaryBranching (paper/sections/method_primary.tex).

The system P is systems/system.txt without its last equation, the Rabinowitsch equation of kappah != 0
(_harness.physical_system). The condition kappah != 0 is imposed by saturation, so the decomposed ideal
is J = <P> : kappah^oo.

    1. Q <- IdealReduce(P) with kappah nonzero: its branch solver decides by saturation whether a hitting
       set has a zero with kappah != 0, and every round keeps <Q> : kappah^oo = J
    2. J = sat(Q, kappah)
    3. the minimal primes of J: those of <Q>, by Alg. Branching with minAssGTZ, that omit kappah
    4. primdecSY of J
    5. a component is isolated iff its radical contains a minimal prime of its own dimension

Coefficients lie in Q(pi). Interreduction, factorization, the branch solver of step 1, the saturation and
primdecSY run in lex with the min-fill variable order of symbolica/factor_prep.rs. Components are written
as reduced Gröbner bases in degrevlex with the variables in name order.

The output is checked: P lies in J, kappah is a nonzerodivisor modulo J, the components intersect to J,
the isolated radicals are exactly the minimal primes, and no component is redundant. The exit status is
1 if a check fails.

The Singular session holds one solver slot of ``_harness.solver_slot`` while it runs.

Usage: physical_primary_decomposition.py [REPORT.md]
Default report: results/physical_primary.md.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import factor_variants as fv
from _harness import PHYSICAL_NONZERO, SolverSlot, physical_system, solver_slot
from _systems import load_system, poly_to_singular

ROOT = Path(__file__).resolve().parent.parent

REPORT = ROOT / "results" / "physical_primary.md"
# The two families of minimal primes, each named by a polynomial that lies in all of its primes.
FAMILIES: dict[str, str] = {"R": "kappah^2-192*pi^2", "D": "mh"}


@dataclass(frozen=True)
class PhysicalSystem:
    variables: list[str]
    parameters: list[str]
    polynomials: list[str]
    min_fill: list[str]


@dataclass(frozen=True)
class Component:
    dim: int
    degree_primary: int
    degree_radical: int
    is_prime: bool
    contained_primes: tuple[int, ...]
    primary: tuple[str, ...]
    radical: tuple[str, ...]


def hyperedge_fill(variable: int, supports: list[list[int]]) -> int:
    containing = [i for i, s in enumerate(supports) if s and variable in s]
    if len(containing) <= 1:
        return 0
    return min(
        sum(sum(1 for v in supports[d] if v != variable and v not in supports[e]) for e in containing if e != d)
        for d in containing
    )


def min_fill_order(count: int, supports: list[list[int]]) -> list[int]:
    """The elimination order of `min_fill_order` in symbolica/factor_prep.rs."""
    active_supports = [sorted(set(s)) for s in supports]
    active = [True] * count
    order: list[int] = []
    while len(order) < count:
        selected = min(
            (v for v in range(count) if active[v]),
            key=lambda v: (hyperedge_fill(v, active_supports), sum(1 for s in active_supports if s and v in s), v),
        )
        containing = [i for i, s in enumerate(active_supports) if s and selected in s]
        if len(containing) > 1:
            definer = min(
                containing,
                key=lambda d: sum(
                    sum(1 for v in active_supports[d] if v != selected and v not in active_supports[e])
                    for e in containing
                    if e != d
                ),
            )
            definer_vars = [v for v in active_supports[definer] if v != selected]
            for e in containing:
                if e != definer:
                    active_supports[e] = sorted({v for v in active_supports[e] if v != selected} | set(definer_vars))
            active_supports[definer] = []
        elif len(containing) == 1:
            active_supports[containing[0]] = []
        active[selected] = False
        order.append(selected)
    return order


def load(path: Path) -> PhysicalSystem:
    """Variables in name order, pi as the coefficient parameter, as the Rust loader reads the system."""
    polynomials, symbols = load_system(path)
    variables = sorted(str(s) for s in symbols if str(s) != "pi")
    supports = [sorted(variables.index(str(s)) for s in p.free_symbols if str(s) in variables) for p in polynomials]
    return PhysicalSystem(
        variables=variables,
        parameters=[str(s) for s in symbols if str(s) == "pi"],
        polynomials=[poly_to_singular(p) for p in polynomials],
        min_fill=[variables[i] for i in min_fill_order(len(variables), supports)],
    )


def timed_run(session: fv.Session, code: str) -> tuple[list[str], float]:
    started = time.monotonic()
    lines = session.run(code)
    return lines, time.monotonic() - started


def flag(lines: list[str], tag: str) -> bool:
    return fv.tagged(lines, tag) == ["1"]


def omitting(session: fv.Session, primes: list[fv.Prime], nonzero: tuple[str, ...]) -> list[fv.Prime]:
    """The primes that contain none of the `nonzero` polynomials: the minimal primes of the saturation."""
    tests = "".join(
        f'print("HAS {k} " + string(size(reduce(ideal({",".join(nonzero)}), std(ideal({prime.basis})))) < {len(nonzero)}));\n'
        for k, prime in enumerate(primes)
    )
    lines = session.run(f"setring Rdp;\n{tests}")
    return [prime for k, prime in enumerate(primes) if f"{k} 0" in fv.tagged(lines, "HAS")]


def prime_labels(session: fv.Session, primes: list[fv.Prime]) -> list[str]:
    """The family of each minimal prime, by the polynomial of FAMILIES it contains, and its dimension."""
    tests = "".join(
        f'print("IN {k} {family} " + string(size(reduce({polynomial}, Ps[{k + 1}])) == 0));\n'
        for k in range(len(primes))
        for family, polynomial in FAMILIES.items()
    )
    lines = session.run(f"setring Rdp;\n{tests}")
    families = {k: [f for f in FAMILIES if f"{k} {f} 1" in fv.tagged(lines, "IN")] for k in range(len(primes))}
    if any(len(found) != 1 for found in families.values()):
        raise RuntimeError(f"a minimal prime is in no family or in both: {families}")
    return [f"{families[k][0]}{prime.dim}" for k, prime in enumerate(primes)]


def decompose(session: fv.Session, saturated: list[str]) -> tuple[list[Component], float]:
    """Steps 4 and 5: primdecSY of J in lex, mapped to degrevlex, each radical tested against the primes `Ps`."""
    _, secs = timed_run(session, f"setring Rlp; list DL = primdecSY({fv.ideal_literal(saturated)});")
    lines = session.run(
        "setring Rdp; list D = imap(Rlp, DL); list Comps; int i; int j; int k; ideal Q; ideal P; string contains;\n"
        "for (i = 1; i <= size(D); i++) {\n"
        "  Q = std(D[i][1]); P = std(D[i][2]); Comps = insert(Comps, Q, size(Comps)); contains = \"\";\n"
        '  for (k = 1; k <= size(Ps); k++) { if (size(reduce(Ps[k], P)) == 0) { contains = contains + string(k - 1) + ","; } }\n'
        '  print("COMP " + string(i) + " dim=" + string(dim(P)) + " degQ=" + string(mult(Q)) + " degP=" + string(mult(P))\n'
        '        + " prime=" + string(size(reduce(P, Q)) == 0) + " contains=" + contains);\n'
        '  for (j = 1; j <= size(Q); j++) { print("Q " + string(i) + " " + string(Q[j])); }\n'
        '  for (j = 1; j <= size(P); j++) { print("P " + string(i) + " " + string(P[j])); }\n'
        "}\n"
    )
    heads = {
        int(index): dict(item.split("=", 1) for item in rest.split())
        for index, rest in (line.split(" ", 1) for line in fv.tagged(lines, "COMP"))
    }

    def polynomials(tag: str, index: int) -> tuple[str, ...]:
        return tuple(poly for i, poly in (line.split(" ", 1) for line in fv.tagged(lines, tag)) if int(i) == index)

    components = [
        Component(
            dim=int(head["dim"]),
            degree_primary=int(head["degQ"]),
            degree_radical=int(head["degP"]),
            is_prime=head["prime"] == "1",
            contained_primes=tuple(int(k) for k in head["contains"].split(",") if k),
            primary=polynomials("Q", index),
            radical=polynomials("P", index),
        )
        for index, head in sorted(heads.items())
    ]
    return components, secs


def is_isolated(component: Component, primes: list[fv.Prime]) -> bool:
    return any(primes[k].dim == component.dim for k in component.contained_primes)


def checks(session: fv.Session, components: list[Component], primes: list[fv.Prime]) -> dict[str, bool]:
    """Against `Gin`, a Gröbner basis of J: the components intersect to it and none is redundant;
    the isolated radicals are the minimal primes, each once."""
    count = len(components)
    whole = session.run(
        "setring Rdp; ideal X = Comps[1]; int c;\n"
        "for (c = 2; c <= size(Comps); c++) { X = intersect(X, Comps[c]); }\n"
        'X = std(X); print("EQUAL " + string(size(reduce(X, Gin)) == 0 && size(reduce(Gin, X)) == 0));'
    )
    redundant = [
        index
        for index in range(1, count + 1)
        if flag(
            session.run(
                "setring Rdp; ideal X = ideal(1);\n"
                + "".join(f"X = intersect(X, Comps[{k}]);\n" for k in range(1, count + 1) if k != index)
                + 'X = std(X); print("SAME " + string(size(reduce(X, Gin)) == 0));'
            ),
            "SAME",
        )
    ]
    matched = sorted(
        k for c in components if is_isolated(c, primes) for k in c.contained_primes if primes[k].dim == c.dim
    )
    return {
        "the components intersect to J": flag(whole, "EQUAL"),
        "the isolated radicals are exactly the minimal primes": matched == list(range(len(primes))),
        "no component is redundant": not redundant,
    }


def describe(component: Component, primes: list[fv.Prime], labels: list[str]) -> tuple[str, str]:
    if is_isolated(component, primes):
        own = next(k for k in component.contained_primes if primes[k].dim == component.dim)
        return "isolated", f"minimal prime {labels[own]}"
    return "embedded", "radical contains minimal primes " + ", ".join(labels[k] for k in component.contained_primes)


def report(
    system: PhysicalSystem,
    reduced: list[str],
    saturated: list[str],
    rounds: int,
    primes: list[fv.Prime],
    labels: list[str],
    components: list[Component],
    secs: dict[str, float],
    verdicts: dict[str, bool],
) -> str:
    order = sorted(components, key=lambda c: (not is_isolated(c, primes), -c.dim))
    nonzero = ", ".join(PHYSICAL_NONZERO)
    lines = [
        "# Primary decomposition of the physical system",
        "",
        (
            f"System: `systems/system.txt` without its last equation, the Rabinowitsch equation of {nonzero} != 0 "
            f"({len(system.polynomials)} equations, {len(system.variables)} variables, coefficients in Q(pi)); "
            f"{nonzero} != 0 is imposed by saturation, so the decomposed ideal is J = <P> : {nonzero}^oo."
        ),
        (
            "Computed with Alg. PrimaryBranching by `scripts/physical_primary_decomposition.py`: "
            f"Alg. IdealReduce with {nonzero} nonzero turns the {len(system.polynomials)} equations into "
            f"{len(reduced)} generators in {rounds} rounds ({secs['reduce']:.1f} s), their saturation by {nonzero} "
            f"has {len(saturated)} generators ({secs['saturate']:.1f} s), Alg. Branching with `minAssGTZ` finds the "
            f"{len(primes)} minimal primes ({secs['primes']:.1f} s), and Singular `primdecSY` decomposes the "
            f"saturation ({secs['primdec']:.0f} s)."
        ),
        (
            "Interreduction, factorization, the branch solver of Alg. IdealReduce (saturation by "
            f"{nonzero}), the saturation, and `primdecSY` run in the lexicographic order with the min-fill "
            f"variable order of `symbolica/factor_prep.rs` ({' > '.join(system.min_fill)})."
        ),
        (
            f"Checks with a Gröbner basis of J ({secs['groebner']:.1f} s): "
            + "; ".join(f"{name}: {value}" for name, value in verdicts.items())
            + "."
        ),
        "",
        (
            "Labels: R = reduced family (kappah^2 = 192 pi^2), D = doubled family (mh = 0); "
            "the number is the dimension of the slice."
        ),
        "Ideals are reduced Gröbner bases in the degree reverse lexicographic order (variables in name order).",
        "",
        "| # | type | dim | degree of primary / of radical | reduced | where |",
        "|---|---|---|---|---|---|",
    ]
    for number, component in enumerate(order, 1):
        kind, where = describe(component, primes, labels)
        lines.append(
            f"| {number} | {kind} | {component.dim} | {component.degree_primary} / {component.degree_radical} | "
            f"{'yes' if component.is_prime else 'no'} | {where} |"
        )
    for number, component in enumerate(order, 1):
        kind, where = describe(component, primes, labels)
        lines += ["", f"## Component {number}: {kind}, dimension {component.dim} ({where})", ""]
        lines += ["Primary ideal:", "", "```", *component.primary, "```"]
        if not component.is_prime:
            lines += ["", "Radical:", "", "```", *component.radical, "```"]
    return "\n".join(lines) + "\n"


def decompose_and_report(out: Path, slot: SolverSlot) -> int:
    """Run Alg. PrimaryBranching, write the report to `out`, and return 1 if a check fails."""
    system = load(physical_system())
    session = fv.Session(system.variables, system.parameters, factor_variables=system.min_fill, empty_ring="Rlp")
    slot.track(session.pid)
    secs: dict[str, float] = {}

    started = time.monotonic()
    reduced, rounds = fv.ideal_reduce(session, system.polynomials, PHYSICAL_NONZERO)
    secs["reduce"] = time.monotonic() - started
    print(f"IdealReduce: {len(system.polynomials)} -> {len(reduced)} generators, {rounds} rounds, "
          f"{secs['reduce']:.1f} s", flush=True)

    started = time.monotonic()
    saturated = fv.saturation(session, reduced, PHYSICAL_NONZERO, "Rlp")
    secs["saturate"] = time.monotonic() - started
    print(f"saturation: {len(saturated)} generators, {secs['saturate']:.1f} s", flush=True)

    started = time.monotonic()
    primes = omitting(session, fv.prime_branching(session, reduced)[0], PHYSICAL_NONZERO)
    secs["primes"] = time.monotonic() - started
    session.run("setring Rdp; list Ps = " + ",".join(f"std(ideal({p.basis}))" for p in primes) + ";")
    labels = prime_labels(session, primes)
    print(f"minimal primes: {', '.join(labels)}, {secs['primes']:.1f} s", flush=True)

    components, secs["primdec"] = decompose(session, saturated)
    print(f"primdecSY: {len(components)} components, {secs['primdec']:.0f} s", flush=True)

    nonzero = "*".join(PHYSICAL_NONZERO)
    lines, secs["groebner"] = timed_run(
        session,
        f"setring Rdp; ideal Gin = std({fv.ideal_literal(saturated)});\n"
        f'print("P_IN_J " + string(size(reduce({fv.ideal_literal(system.polynomials)}, Gin)) == 0));\n'
        f'print("SATURATED " + string(size(reduce(std(quotient(Gin, ideal({nonzero}))), Gin)) == 0));',
    )
    verdicts = {
        "the equations lie in J": flag(lines, "P_IN_J"),
        f"{nonzero} is a nonzerodivisor modulo J": flag(lines, "SATURATED"),
        **checks(session, components, primes),
    }
    session.close()
    print(f"checks: {verdicts}", flush=True)

    out.write_text(
        report(system, reduced, saturated, rounds, primes, labels, components, secs, verdicts), encoding="utf-8"
    )
    print(f"wrote {out}")
    return 0 if all(verdicts.values()) else 1


def main() -> int:
    """The Singular session is one long solver run, so it holds a solver slot throughout."""
    with solver_slot() as slot:
        return decompose_and_report(Path(sys.argv[1]) if len(sys.argv) > 1 else REPORT, slot)


if __name__ == "__main__":
    raise SystemExit(main())
