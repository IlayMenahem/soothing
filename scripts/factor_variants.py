#!/usr/bin/env python3
"""Factor-branching variants of paper/sections/method_*.tex on one system.

A run keeps one Singular process for the whole algorithm, so branch work pays no
process start-up. Each method reaches one cell of (object, decomposition):

    branching           Alg. Branching, branch solver std       (sqrt I, covering)
    direct-facstd       facstd                                   (sqrt I, covering)
    ideal-reduce        Alg. IdealReduce, then std of its output (I, monolithic)
    direct-std          std                                      (I, monolithic)
    ideal-refine        Alg. IdealRefine                         (I, covering)
    verify-ideal-refine Alg. IdealRefine, then intersect its leaves and compare with std
    prime               Alg. PrimeBranching                      (sqrt I, irreducible)
    direct-minAssGTZ    minAssGTZ                                (sqrt I, irreducible)
    primary             Alg. PrimaryBranching                    (I, irreducible)
    direct-primdecGTZ   primdecGTZ                               (I, irreducible)

Refinement factors and interreduces in lex (Singular lp); every Gröbner basis is
in degrevlex (Singular dp). Both rings set option(redSB) and option(redTail):
Singular's interred reduces tails only under redSB, as the paper's interreduction
does, and Singular stores redTail per ring. Ideals are compared by hashes of their
reduced, monic Gröbner bases, which are unique.

Usage: factor_variants.py METHOD SYSTEM.json
SYSTEM.json holds {"variables": [...], "polynomials": [...]} in Singular syntax.
Prints one JSON line.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

Factorization = dict[str, int]
Branch = frozenset[str]

PROCEDURES = r"""
// Singular's `continue` skips the increment of a for loop, so these procedures never use it.
proc factorLines(ideal Q, int doInterred)
{
  if (doInterred) { Q = interred(Q); }
  Q = simplify(Q, 2);
  int i; int j; list F;
  for (i = 1; i <= ncols(Q); i++) {
    if (deg(Q[i]) == 0) { print("CONST " + string(i)); }
    if (deg(Q[i]) > 0) {
      print("GEN " + string(i) + " " + string(simplify(Q[i], 1)));
      F = factorize(Q[i]);
      for (j = 1; j <= size(F[1]); j++) {
        if (deg(F[1][j]) > 0) {
          print("FAC " + string(i) + " " + string(F[2][j]) + " " + string(simplify(F[1][j], 1)));
        }
      }
    }
  }
}
proc emptyLine(ideal H)
{
  list F = facstd(H);
  int k; int empty = 1;
  for (k = 1; k <= size(F); k++) { if (dim(std(F[k])) >= 0) { empty = 0; } }
  print("EMPTY " + string(empty));
}
proc canonical(ideal J)
{
  option(redSB); option(redTail);
  return(simplify(std(J), 1));
}
proc primeLines(ideal H)
{
  list F = facstd(H);
  int k; int i; int linear; list L; ideal P;
  for (k = 1; k <= size(F); k++) {
    if (dim(std(F[k])) >= 0) {
      linear = 1;
      for (i = 1; i <= ncols(F[k]); i++) { if (deg(F[k][i]) > 1) { linear = 0; } }
      if (linear) { L = list(F[k]); } else { L = minAssGTZ(F[k]); }
      for (i = 1; i <= size(L); i++) {
        P = canonical(L[i]);
        print("PRIME " + string(dim(P)) + " " + string(P));
      }
    }
  }
}
proc saturateLines(ideal J, ideal U)
{
  ideal S = std(J); int s = 0; int k; list L;
  for (k = 1; k <= ncols(U); k++) {
    if (U[k] != 0) {
      L = sat_with_exp(S, ideal(U[k])); S = std(L[1]);
      if (L[2] > s) { s = L[2]; }
    }
  }
  S = canonical(S);
  print("DIM " + string(dim(S)));
  print("EXP " + string(s));
  return(S);
}
proc coveredLine(ideal R, list Leaves)
{
  // A zero remainder modulo the generators of R, Groebner basis or not, puts the leaf inside R.
  int k; int covered = 0;
  for (k = 1; k <= size(Leaves); k++) { if (!covered && size(reduce(Leaves[k], R)) == 0) { covered = 1; } }
  print("COVERED " + string(covered));
}
proc intersectionLine(list Leaves)
{
  ideal X = Leaves[1]; int k;
  for (k = 2; k <= size(Leaves); k++) { X = intersect(X, Leaves[k]); }
  print("BASIS " + string(canonical(X)));
}
proc isolatedLine(ideal G, list Ps, int i)
{
  // I : p^oo lies in every other minimal prime but not in p, so a basis element outside p separates p.
  ideal J = G; int k; poly s = 1; list L;
  if (size(Ps) > 1) {
    L = sat_with_exp(G, Ps[i]); ideal T = std(L[1]);
    for (k = 1; k <= ncols(T); k++) { if (s == 1 && reduce(T[k], Ps[i]) != 0) { s = T[k]; } }
    L = sat_with_exp(G, ideal(s)); J = L[1];
  }
  if (dim(Ps[i]) > 0) { J = equidimMax(J); }
  J = canonical(J);
  print("COMPONENT " + string(J));
}
proc pointComponentLine(ideal G, ideal P)
{
  // For zero-dimensional G, G + P^m is the P-primary component once vdim stops growing in m (Nakayama).
  ideal Q = std(G + P); int v = vdim(Q); int previous = -1; int m = 1;
  while (v != previous) { previous = v; m = m + 1; Q = std(G + P^m); v = vdim(Q); }
  print("COMPONENT " + string(canonical(Q)));
}
"""


class SingularError(RuntimeError):
    pass


class Session:
    """One Singular process; `run` sends code and returns the lines it prints.

    Coefficients lie in Q(parameters). `Rlp` is lex on `factor_variables` (default `variables`), `Rdp` is
    degrevlex on `variables`, and the branch solver certifies emptiness in `empty_ring`.
    """

    def __init__(
        self,
        variables: list[str],
        parameters: Sequence[str] = (),
        factor_variables: Sequence[str] | None = None,
        empty_ring: str = "Rdp",
    ) -> None:
        self._process = subprocess.Popen(
            ["Singular", "-q", "--no-rc", "--no-warn", "-t"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._calls = 0
        self.empty_ring = empty_ring
        field = f"(0,{','.join(parameters)})" if parameters else "0"
        options = "short = 0; option(redSB); option(redTail);"
        self.run(
            'LIB "primdec.lib";\n'
            f"ring Rlp = {field}, ({','.join(factor_variables or variables)}), lp; {options}\n"
            f"ring Rdp = {field}, ({','.join(variables)}), dp; {options}\n"
            f"{PROCEDURES}"
        )

    def run(self, code: str) -> list[str]:
        stdin, stdout = self._process.stdin, self._process.stdout
        assert stdin is not None and stdout is not None
        self._calls += 1
        marker = f"@@END{self._calls}"
        stdin.write(f'{code}\nprint("{marker}");\n')
        stdin.flush()
        lines: list[str] = []
        line = stdout.readline()
        while line and line.rstrip() != marker:
            lines.append(line.rstrip())
            line = stdout.readline()
        if not line:
            raise SingularError("Singular exited:\n" + "\n".join(lines[-5:]))
        errors = [text for text in lines if text.lstrip().startswith("?")]
        if errors:
            raise SingularError("\n".join(errors))
        return lines

    def close(self) -> None:
        if self._process.stdin is not None:
            self._process.stdin.write("quit;\n")
            self._process.stdin.flush()
        self._process.wait(10)


def tagged(lines: list[str], tag: str) -> list[str]:
    prefix = f"{tag} "
    return [line[len(prefix) :] for line in lines if line.startswith(prefix)]


def ideal_literal(polynomials: Iterable[str]) -> str:
    return f"ideal({','.join(polynomials) or '0'})"


def digest(basis: str) -> str:
    return hashlib.sha256(basis.encode()).hexdigest()[:16]


def power_product(factors: Factorization) -> str:
    return "*".join(f"({factor})^{exponent}" for factor, exponent in sorted(factors.items())) or "1"


@dataclass(frozen=True)
class Factored:
    generators: tuple[str, ...]
    factorizations: tuple[Factorization, ...]
    unsatisfiable: bool

    def factor_sets(self) -> list[frozenset[str]]:
        return [frozenset(factorization) for factorization in self.factorizations]


@dataclass(frozen=True)
class Prime:
    dim: int
    basis: str


def factorize(session: Session, polynomials: list[str], interreduce: bool) -> Factored:
    """Factor every nonzero generator in lex, after interreduction if asked; a constant marks UNSAT."""
    lines = session.run(f"setring Rlp; factorLines({ideal_literal(polynomials)}, {int(interreduce)});")
    generators: dict[int, str] = {}
    factorizations: dict[int, Factorization] = {}
    for line in tagged(lines, "GEN"):
        index, generator = line.split(" ", 1)
        generators[int(index)] = generator
        factorizations[int(index)] = {}
    for line in tagged(lines, "FAC"):
        index, exponent, factor = line.split(" ", 2)
        factorizations[int(index)][factor] = int(exponent)
    order = sorted(generators)
    return Factored(
        tuple(generators[i] for i in order),
        tuple(factorizations[i] for i in order),
        bool(tagged(lines, "CONST")),
    )


def minimal_hitting_sets(edges: list[frozenset[str]]) -> list[Branch]:
    """Berge's algorithm on bitmasks; a grown set can only be non-minimal through a kept set containing its new factor."""
    if any(not edge for edge in edges):
        return []
    universe = sorted({factor for edge in edges for factor in edge})
    bit = {factor: 1 << position for position, factor in enumerate(universe)}
    masks = sorted({sum(bit[factor] for factor in edge) for edge in edges}, key=lambda mask: mask.bit_count())
    family: list[int] = [0]
    for edge in masks:
        edge_bits = [b for b in bit.values() if edge & b]
        kept = [chosen for chosen in family if chosen & edge]
        containing = {b: [chosen for chosen in kept if chosen & b] for b in edge_bits}
        grown = {
            chosen | b
            for chosen in family
            if not chosen & edge
            for b in edge_bits
            if not any(other & (chosen | b) == other for other in containing[b])
        }
        family = kept + sorted(grown)
    return [frozenset(factor for factor in universe if mask & bit[factor]) for mask in family]


def prune(branches: Iterable[Branch]) -> list[Branch]:
    """Drop every branch that strictly contains another, whose zero set then contains its own; fewest generators first."""
    kept: list[Branch] = []
    by_least_factor: dict[str, list[Branch]] = {}
    for branch in sorted(branches, key=lambda b: (len(b), sorted(b))):
        candidates = (other for factor in branch for other in by_least_factor.get(factor, []))
        if any(other < branch for other in candidates):
            continue
        kept.append(branch)
        if branch:
            by_least_factor.setdefault(min(branch), []).append(branch)
    return kept


def refine(session: Session, root: Factored) -> list[Branch]:
    """Alg. Refine from the factored root: the pruned terminal branches, fewest generators first."""
    if root.unsatisfiable:
        return []
    stack = minimal_hitting_sets(root.factor_sets())
    seen: set[Branch] = set()
    terminal: set[Branch] = set()
    while stack:
        branch = stack.pop()
        if branch in seen:
            continue
        seen.add(branch)
        factored = factorize(session, sorted(branch), interreduce=True)
        children = set() if factored.unsatisfiable else set(minimal_hitting_sets(factored.factor_sets()))
        if children == {branch}:
            terminal.add(branch)
        else:
            stack.extend(children - seen)
    return prune(terminal)


def branch_dimension(session: Session, polynomials: Iterable[str]) -> int:
    lines = session.run(f'setring Rdp; print("DIM " + string(dim(std({ideal_literal(polynomials)}))));')
    return int(tagged(lines, "DIM")[0])


def certified_empty(session: Session, polynomials: Iterable[str]) -> bool:
    lines = session.run(f"setring {session.empty_ring}; emptyLine({ideal_literal(polynomials)});")
    return tagged(lines, "EMPTY") == ["1"]


def branch_primes(session: Session, branch: Branch) -> list[Prime]:
    lines = session.run(f"setring Rdp; primeLines({ideal_literal(sorted(branch))});")
    return [Prime(int(dim), basis) for dim, basis in (line.split(" ", 1) for line in tagged(lines, "PRIME"))]


def absorbed(session: Session, primes: list[Prime], branch: Branch) -> bool:
    """Some prime has every generator reducing to zero modulo the generators of the branch."""
    tests = " || ".join(f"size(reduce({ideal_literal([p.basis])}, H)) == 0" for p in primes)
    lines = session.run(f'setring Rdp; ideal H = {ideal_literal(sorted(branch))}; print("ABSORBED " + string({tests}));')
    return tagged(lines, "ABSORBED") == ["1"]


def minimal_primes(session: Session, primes: list[Prime]) -> list[Prime]:
    """Distinct primes by basis; a prime is dropped when one of larger dimension lies inside it."""
    distinct = list({prime.basis: prime for prime in primes}.values())
    minimal: list[Prime] = []
    for prime in distinct:
        larger = [other for other in distinct if other.dim > prime.dim]
        if larger:
            tests = " || ".join(f"size(reduce({ideal_literal([o.basis])}, P)) == 0" for o in larger)
            lines = session.run(f'setring Rdp; ideal P = {ideal_literal([prime.basis])}; print("INSIDE " + string({tests}));')
            if tagged(lines, "INSIDE") == ["1"]:
                continue
        minimal.append(prime)
    return sorted(minimal, key=lambda p: (-p.dim, p.basis))


def prime_branching(session: Session, polynomials: list[str]) -> tuple[list[Prime], dict[str, object]]:
    """Alg. PrimeBranching: refine the interreduced system, skip absorbed branches, keep the minimal primes."""
    branches = refine(session, factorize(session, polynomials, interreduce=True))
    found: list[Prime] = []
    skipped = 0
    for branch in branches:
        positive = [prime for prime in found if prime.dim > 0]
        if positive and absorbed(session, positive, branch):
            skipped += 1
            continue
        found.extend(branch_primes(session, branch))
    primes = minimal_primes(session, found)
    return primes, {"branches": len(branches), "skipped": skipped, "found": len(found)}


def ideal_reduce(session: Session, polynomials: list[str]) -> tuple[list[str], int]:
    """Alg. IdealReduce: add h^(a_h) for factors h in every SAT minimal hitting set, drop the generators they divide.

    The candidates start as all factors and shrink to the intersection of the SAT hitting sets seen so far;
    a hitting set that contains every candidate cannot shrink them and is not checked, and the checks stop
    once no candidate is left.
    """
    current = factorize(session, polynomials, interreduce=True)
    for rounds in range(1, 1000):
        if current.unsatisfiable:
            return ["1"], rounds
        candidates = frozenset(f for factorization in current.factorizations for f in factorization)
        for branch in minimal_hitting_sets(current.factor_sets()):
            if not candidates:
                break
            if not candidates <= branch and not certified_empty(session, sorted(branch)):
                candidates &= branch
        largest = {factor: max(f.get(factor, 0) for f in current.factorizations) for factor in candidates}
        present = {next(iter(f.items())) for f in current.factorizations if len(f) == 1}
        new = {factor: exponent for factor, exponent in largest.items() if (factor, exponent) not in present}
        if not new:
            return list(current.generators), rounds
        kept = [
            generator
            for generator, factorization in zip(current.generators, current.factorizations)
            if not any(factorization.get(factor, 0) >= exponent for factor, exponent in new.items())
        ]
        current = factorize(session, kept + [power_product({f: e}) for f, e in sorted(new.items())], interreduce=True)
    raise RuntimeError("ideal reduction did not stop")


def canonical_basis(session: Session, polynomials: list[str], name: str = "G") -> tuple[str, int]:
    lines = session.run(
        f"setring Rdp; ideal {name} = canonical({ideal_literal(polynomials)});\n"
        f'print("DIM " + string(dim({name}))); print("BASIS " + string({name}));'
    )
    return tagged(lines, "BASIS")[0], int(tagged(lines, "DIM")[0])


def divides(small: Factorization, big: Factorization) -> bool:
    return all(big.get(factor, 0) >= exponent for factor, exponent in small.items())


def minimal_seeds(seeds: list[Factorization]) -> list[Factorization]:
    unique = [dict(items) for items in {tuple(sorted(seed.items())) for seed in seeds}]
    return [seed for seed in unique if not any(other != seed and divides(other, seed) for other in unique)]


def chosen_parts(generators: list[Factorization], branch: Branch) -> tuple[list[Factorization], Factorization]:
    """The minimal H-parts of the generators, and for every factor outside H an exponent e_f with
    prod f^e_f times each kept H-part a multiple of a generator: one generator stands for each H-part,
    the one with the smallest part outside H."""
    representative: dict[tuple[tuple[str, int], ...], Factorization] = {}
    for generator in generators:
        part = tuple(sorted((f, e) for f, e in generator.items() if f in branch))
        rest = {f: e for f, e in generator.items() if f not in branch}
        if part not in representative or sum(rest.values()) < sum(representative[part].values()):
            representative[part] = rest
    parts = minimal_seeds([dict(part) for part in representative])
    kept = [representative[tuple(sorted(part.items()))] for part in parts]
    outside = {f for generator in generators for f in generator} - branch
    return parts, {f: max(rest.get(f, 0) for rest in kept) for f in outside}


def ideal_refine(session: Session, polynomials: list[str], max_rounds: int = 50) -> dict[str, object]:
    """Alg. IdealRefine on the interreduced system, residuals added one saturation at a time.

    Saturation uses only the H-parts of the generators. A minimal hitting set whose saturation is <1>, in
    particular one certified UNSAT, is skipped: then u_H^s already lies in the ideal. Since
    I = (cap of the leaves) cap residual, a residual that contains a leaf, or has no zeros, ends the run
    after any round; only a final residual that is kept gets a Gröbner basis. The leaves stay in `Leaves`.
    """
    root = factorize(session, polynomials, interreduce=True)
    session.run("setring Rdp; list Leaves;")
    if root.unsatisfiable:
        return {"leaves": 0, "rounds": 0, "final_leaf": "none", "leaf_dims": []}
    base = list(root.factorizations)
    residuals: list[Factorization] = []
    leaf_dims: list[int] = []
    for rounds in range(1, max_rounds + 1):
        split = False
        for branch in minimal_hitting_sets([frozenset(g) for g in base + residuals]):
            current = base + residuals
            parts, unchosen = chosen_parts(current, branch)
            seeds = [power_product(part) for part in parts]
            if certified_empty(session, seeds):
                continue
            lines = session.run(
                f"setring Rdp; ideal S = saturateLines({ideal_literal(seeds)}, {ideal_literal(sorted(unchosen))});\n"
                "if (dim(S) >= 0) { Leaves = insert(Leaves, S, size(Leaves)); }"
            )
            dim, exponent = int(tagged(lines, "DIM")[0]), int(tagged(lines, "EXP")[0])
            if dim < 0:
                continue
            leaf_dims.append(dim)
            split = True
            residuals.append({f: exponent + e for f, e in unchosen.items() if exponent + e > 0})
        residual = list(root.generators) + [power_product(r) for r in residuals]
        outcome: dict[str, object] = {"rounds": rounds, "leaf_dims": leaf_dims}
        if certified_empty(session, residual) or tagged(
            session.run(f"setring Rdp; coveredLine({ideal_literal(residual)}, Leaves);"), "COVERED"
        ) == ["1"]:
            return {**outcome, "leaves": len(leaf_dims), "final_leaf": "redundant"}
        if not split:
            session.run(f"setring Rdp; Leaves = insert(Leaves, canonical({ideal_literal(residual)}), size(Leaves));")
            return {**outcome, "leaves": len(leaf_dims) + 1, "final_leaf": "kept"}
    raise RuntimeError("ideal refinement hit the round limit")


def verify_ideal_refine(session: Session, polynomials: list[str]) -> dict[str, object]:
    """Alg. IdealRefine, then whether its leaves intersect to the ideal; an untimed check that needs std of the input."""
    refined = ideal_refine(session, polynomials)
    basis, _ = canonical_basis(session, polynomials)
    if not refined["leaves"]:
        return {**refined, "exact": basis == "1"}
    intersection = tagged(session.run("setring Rdp; intersectionLine(Leaves);"), "BASIS")[0]
    return {**refined, "exact": intersection == basis}


def zero_dimensional_radical(session: Session) -> bool:
    """For zero-dimensional G with minimal primes Ps: vdim G = sum of vdim of its components, each at least
    the vdim of its prime with equality only for the prime itself, so equal sums mean every component is prime."""
    lines = session.run(
        "setring Rdp; int total = 0; int k;\n"
        "for (k = 1; k <= size(Ps); k++) { total = total + vdim(Ps[k]); }\n"
        'print("RADICAL " + string(vdim(G) == total));\nkill total, k;'
    )
    return tagged(lines, "RADICAL") == ["1"]


def primary_branching(session: Session, polynomials: list[str]) -> dict[str, object]:
    """Alg. PrimaryBranching: IdealReduce, minimal primes by prime branching, and isolated components by
    separators; primdecGTZ of the reduced ideal only if embedded components exist, with each of its
    components marked isolated when its radical contains a minimal prime of its own dimension."""
    started = time.monotonic()
    reduced, rounds = ideal_reduce(session, polynomials)
    _, dim = canonical_basis(session, reduced)
    if dim < 0:
        return {"components": 0, "embedded": 0, "isolated_hashes": [], "reduce_rounds": rounds}
    primes, prime_stats = prime_branching(session, reduced)
    session.run(f"setring Rdp; list Ps = {','.join(f'ideal({p.basis})' for p in primes)};")
    if dim == 0 and zero_dimensional_radical(session):
        components = [prime.basis for prime in primes]
    else:
        procedure = "pointComponentLine(G, Ps[{0}])" if dim == 0 else "isolatedLine(G, Ps, {0})"
        components = [
            tagged(session.run(f"setring Rdp; {procedure.format(index)};"), "COMPONENT")[0]
            for index in range(1, len(primes) + 1)
        ]
    unmixed = dim == 0
    if not unmixed:
        lines = session.run(
            "setring Rdp; ideal X = intersect("
            + ",".join(f"ideal({c})" for c in components)
            + '); print("UNMIXED " + string(size(reduce(X, G)) == 0));'
        )
        unmixed = tagged(lines, "UNMIXED") == ["1"]
    isolated_secs = time.monotonic() - started
    outcome: dict[str, object] = {
        "reduce_rounds": rounds,
        "minimal_primes": len(primes),
        "isolated_hashes": sorted(digest(c) for c in components),
        "isolated_secs": isolated_secs,
        **{f"prime_{key}": value for key, value in prime_stats.items()},
    }
    if unmixed:
        return {**outcome, "components": len(components), "embedded": 0, "fallback": False}
    # The isolated components and the embedded verdict stand even if the fallback times out.
    print("PARTIAL " + json.dumps({**outcome, "isolated": len(components)}), flush=True)
    general = primary_decomposition(session, "G", primes="Ps")
    return {**outcome, **general, "fallback": True}


def primary_decomposition(session: Session, name: str, primes: str | None = None) -> dict[str, object]:
    """primdecGTZ of `name`. With `primes`, the Singular list of minimal primes, a component is isolated when
    its radical contains a minimal prime of its own dimension; without, when no radical of larger dimension
    lies inside its radical."""
    if primes is None:
        mark = (
            "  embedded = 0;\n"
            "  for (j = 1; j <= n; j++) {\n"
            "    if (j != i && dim(Rad[j]) > dim(Rad[i]) && size(reduce(Rad[j], Rad[i])) == 0) { embedded = 1; }\n"
            "  }\n"
        )
    else:
        mark = (
            "  embedded = 1;\n"
            f"  for (j = 1; j <= size({primes}); j++) {{\n"
            f"    if (dim({primes}[j]) == dim(Rad[i]) && size(reduce({primes}[j], Rad[i])) == 0) {{ embedded = 0; }}\n"
            "  }\n"
        )
    lines = session.run(
        f"setring Rdp; list D = primdecGTZ({name}); int n = size(D); int i; int j; int embedded;\n"
        "list Rad;\n"
        "for (i = 1; i <= n; i++) { Rad = insert(Rad, canonical(D[i][2]), size(Rad)); }\n"
        "for (i = 1; i <= n; i++) {\n"
        f"{mark}"
        '  print("COMPONENT " + string(embedded) + " " + string(canonical(D[i][1])));\n'
        "}\n"
        "kill D, n, i, j, embedded, Rad;"
    )
    parts = [line.split(" ", 1) for line in tagged(lines, "COMPONENT")]
    return {
        "components": len(parts),
        "embedded": sum(flag == "1" for flag, _ in parts),
        "isolated_hashes": sorted(digest(basis) for flag, basis in parts if flag == "0"),
    }


def run_method(method: str, session: Session, polynomials: list[str]) -> dict[str, object]:
    if method == "branching":
        branches = refine(session, factorize(session, polynomials, interreduce=False))
        dims = [d for d in (branch_dimension(session, b) for b in branches) if d >= 0]
        return {"branches": len(branches), "sat": len(dims), "max_dim": max(dims, default=-1)}
    if method == "direct-facstd":
        lines = session.run(
            f"setring Rdp; list F = facstd({ideal_literal(polynomials)}); int k;\n"
            'for (k = 1; k <= size(F); k++) { print("DIM " + string(dim(std(F[k])))); }\nkill F, k;'
        )
        dims = [d for d in map(int, tagged(lines, "DIM")) if d >= 0]
        return {"branches": len(dims), "sat": len(dims), "max_dim": max(dims, default=-1)}
    if method == "ideal-reduce":
        started = time.monotonic()
        reduced, rounds = ideal_reduce(session, polynomials)
        reduction = {
            "reduce_rounds": rounds,
            "input_generators": len(polynomials),
            "reduced_generators": len(reduced),
            "reduce_secs": time.monotonic() - started,
        }
        # The reduced system stands even if its Gröbner basis times out.
        print("PARTIAL " + json.dumps(reduction), flush=True)
        basis, dim = canonical_basis(session, reduced)
        return {**reduction, "basis_size": basis.count(",") + 1, "dim": dim, "basis_hash": digest(basis)}
    if method == "direct-std":
        basis, dim = canonical_basis(session, polynomials)
        return {"basis_size": basis.count(",") + 1, "dim": dim, "basis_hash": digest(basis)}
    if method == "ideal-refine":
        return ideal_refine(session, polynomials)
    if method == "verify-ideal-refine":
        return verify_ideal_refine(session, polynomials)
    if method == "prime":
        primes, stats = prime_branching(session, polynomials)
        return {**stats, **prime_summary(primes)}
    if method == "direct-minAssGTZ":
        lines = session.run(
            f"setring Rdp; list L = minAssGTZ({ideal_literal(polynomials)}); int k; ideal P;\n"
            'for (k = 1; k <= size(L); k++) { P = canonical(L[k]); print("PRIME " + string(dim(P)) + " " + string(P)); }\n'
            "kill L, k, P;"
        )
        primes = [Prime(int(d), b) for d, b in (line.split(" ", 1) for line in tagged(lines, "PRIME"))]
        return prime_summary(primes)
    if method == "primary":
        return primary_branching(session, polynomials)
    if method == "direct-primdecGTZ":
        session.run(f"setring Rdp; ideal I = {ideal_literal(polynomials)};")
        return primary_decomposition(session, "I")
    raise SystemExit(f"unknown method {method}")


def prime_summary(primes: list[Prime]) -> dict[str, object]:
    return {
        "primes": len(primes),
        "max_dim": max((p.dim for p in primes), default=-1),
        "prime_hashes": sorted(digest(p.basis) for p in primes),
    }


def main() -> int:
    method, path = sys.argv[1], Path(sys.argv[2])
    system: dict[str, list[str]] = json.loads(path.read_text(encoding="utf-8"))
    session = Session(system["variables"])
    started = time.monotonic()
    try:
        outcome: dict[str, object] = {"status": "ok", **run_method(method, session, system["polynomials"])}
    except (SingularError, RuntimeError) as error:
        outcome = {"status": "error", "error": str(error)[:500]}
    outcome["internal_secs"] = time.monotonic() - started
    session.close()
    print(json.dumps({"method": method, **outcome}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
