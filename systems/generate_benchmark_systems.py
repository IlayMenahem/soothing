#!/usr/bin/env python3
"""
Generate classical polynomial benchmark systems as Mathematica-style rule lists.

Output matches Mathematica format.

References (per family):
  * **cyclic** — same as ``PolynomialTestSystems.cyclic(n)`` (Julia), Björck–Fröberg.
  * **katsura** — same as ``PolynomialTestSystems.katsura(n)`` (PoSSo / Katsura).
  * **noonburg** — Noonburg neural-network benchmark (e.g. Chen thesis; common form below).
  * **eco** — Morgan economics models, Verschelde demos ``eco5``, ``eco6``, …
  * **nash** — 2×2 game; row/column indifference (linear polynomials in ``p``, ``q``).
  * **adjacent_minors** — vanishing adjacent 2×2 minors of an ``r×c`` matrix.
  * **sparse_toric** — sparse binomial fewnomial system ``x_i^2 * x_{i+1} - 1`` (cyclic indices).
  * **random_distances** — seeded planar kinematics after algebraization on a random connected
    link graph (``c_e``, ``s_e`` per spanning-tree edge, unit circles, chord loop closures, optional
    consistent end-position closure).
  * **chemical_steady_state_random** — seeded mass-action ``S * v(x) = 0`` plus conservation rows from
    ``ker(S^T)`` (pre-reduction).
"""

from __future__ import annotations

import argparse
import math
import random
import re
import sys
import itertools
from pathlib import Path
from typing import Callable, Sequence

from sympy import Expr, Integer, Matrix, Rational, Symbol, expand, prod, signsimp, symbols

def sympy_to_dag_string(expr: Expr) -> str:
    """Emit syntax compatible with ``soothing-linear`` rational DAG: ``^`` for powers, ``/`` for divide."""
    e = signsimp(expand(expr))
    # str() avoids sstr's ``a + -b`` formatting; still replace ``**`` with ``^`` for the DAG parser.
    raw = str(e).replace("**", "^")
    # sstr sometimes emits SymPy's ``S(n)`` wrappers; the DAG parser expects plain integers.
    return re.sub(r"S\((-?\d+)\)", r"\1", raw)


def rules_from_equations(equations: Sequence[Expr]) -> str:
    """Comma-separated ``0 -> (rhs)`` rules; rhs is the expression that must be zero."""

    parts = []

    for eq in equations:
        rhs = sympy_to_dag_string(expand(eq))
        parts.append(f"0 -> ({rhs})")
    
    return ", ".join(parts)


def cyclic(n: int) -> list[Expr]:
    """Cyclic n-roots style sums (Julia PolynomialTestSystems)."""
    if n < 2:
        raise ValueError("cyclic requires n >= 2")

    z = [Symbol(f"z{i}") for i in range(1, n + 1)]
    eqs: list[Expr] = []
    
    for m in range(0, n - 1):
        s: Expr = Integer(0)
        for j in range(1, n + 1):
            term: Expr = Integer(1)
            for jj in range(m + 1):
                k_julia = j + jj
                idx = (k_julia - 1) % n
                term *= z[idx]
            s += term
        eqs.append(s)
    eqs.append(prod(z) - 1)
    
    return eqs


def katsura(n: int) -> list[Expr]:
    """Katsura-(n) PoSSo formulation: variables x0..xn (n+1 vars), n+1 equations."""
    if n < 1:
        raise ValueError("katsura requires n >= 1")
    
    x = [Symbol(f"x{i}") for i in range(0, n + 1)]
    eqs: list[Expr] = []
    
    for m in range(0, n):
        s: Expr = Integer(0)
        for l in range(-n, n + 1):
            if abs(m - l) <= n:
                s += x[abs(l)] * x[abs(m - l)]
        eqs.append(s - x[m])
    eqs.append(x[0] + 2 * sum(x[i] for i in range(1, n + 1)) - 1)
    
    return eqs


def noonburg(k: int) -> list[Expr]:
    """Noonburg benchmark: x_i * (sum_j x_j^2 - x_i^2 - 11/10) + 1 = 0."""
    if k < 1:
        raise ValueError("noonburg requires k >= 1")
    
    xs = [Symbol(f"x{i}") for i in range(1, k + 1)]
    sum_sq = sum(t**2 for t in xs)
    return [x_i * (sum_sq - x_i**2 - Rational(11, 10)) + 1 for x_i in xs]


def eco(n: int) -> list[Expr]:
    """Morgan economics Eco-n (Verschelde demo eco5, eco6, …). Variables x1..xn."""
    if n < 2:
        raise ValueError("eco requires n >= 2")
    
    xs = [Symbol(f"x{i}") for i in range(1, n + 1)]
    eqs: list[Expr] = []
    
    for e in range(0, n - 2):
        s = xs[e]
        for j in range(n - 2 - e):
            s += xs[j] * xs[e + j + 1]
        prod_s = s * xs[-1]
        if e == 0:
            eqs.append(Integer(n) * prod_s - 1)
        else:
            eqs.append(prod_s - Integer(e + 1))
    eqs.append(xs[n - 2] * xs[n - 1] - Integer(n - 1))
    eqs.append(sum(xs[0 : n - 1]) + Integer(1))
    
    return eqs


def nash_2x2() -> list[Expr]:
    """
    2×2 bimatrix mixed-strategy indifference (interior conditions).
    Row: R00*q + R01*(1-q) = R10*q + R11*(1-q)
    Col: C00*p + C10*(1-p) = C01*p + C11*(1-p)

    Payoffs: matching pennies (non-degenerate mixed equilibrium at p=q=1/2).
    """
    p, q = symbols("p q")
    r = ((1, -1), (-1, 1))
    c = ((-1, 1), (1, -1))
    eq_row = r[0][0] * q + r[0][1] * (1 - q) - (r[1][0] * q + r[1][1] * (1 - q))
    eq_col = c[0][0] * p + c[1][0] * (1 - p) - (c[0][1] * p + c[1][1] * (1 - p))
    
    return [expand(eq_row), expand(eq_col)]


def adjacent_minors(rows: int, cols: int) -> list[Expr]:
    """Vanishing adjacent 2×2 minors; variables m_i_j for entries."""
    if rows < 2 or cols < 2:
        raise ValueError("adjacent_minors requires rows >= 2 and cols >= 2")
    
    m = [[Symbol(f"m_{i}_{j}") for j in range(1, cols + 1)] for i in range(1, rows + 1)]
    
    eqs: list[Expr] = []
    for i in range(rows - 1):
        for j in range(cols - 1):
            a, b = m[i][j], m[i][j + 1]
            c2, d = m[i + 1][j], m[i + 1][j + 1]
            eqs.append(expand(a * d - b * c2))
    
    return eqs


def sparse_toric(n: int) -> list[Expr]:
    """Sparse binomial cycle: x_i^2 * x_{i+1} - 1 (indices mod n)."""
    if n < 2:
        raise ValueError("sparse_toric requires n >= 2")
    
    xs = [Symbol(f"t{i}") for i in range(1, n + 1)]
    eqs: list[Expr] = []
    
    for i in range(n):
        eqs.append(xs[i] ** 2 * xs[(i + 1) % n] - 1)
    
    return eqs


def random_distances(joints: int, *, seed: int) -> list[Expr]:
    def random_graph(joints: int, prob: float, seed: int) -> list[tuple[int, int]]:
        edges = []

        for i, j in itertools.combinations(range(joints), 2):
            if rng.random() < prob:
                edges.append((i, j))

        return edges

    if joints < 2:
        raise ValueError("random_distances requires joints >= 2")

    rng = random.Random(seed)

    prob = math.log(joints) / joints
    edges = random_graph(joints, prob, seed)

    distances = {edge: rng.choice(range(1, 10)) for edge in edges}

    eqs: list[Expr] = []
    xs = [Symbol(f"x{i}") for i in range(joints)]
    ys = [Symbol(f"y{i}") for i in range(joints)]
    zs = [Symbol(f"z{i}") for i in range(joints)]

    eqs.extend([xs[0] - 0, ys[0] - 0, zs[0] - 0])
    
    for edge in edges:
        i, j = edge
        eqs.append((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2 + (zs[i] - zs[j])**2 - distances[edge]**2)

    return eqs

def chemical_steady_state_random(species: int, reactions: int, *, seed: int) -> list[Expr]:
    """
    Mass-action: ``d x / d t = S * v(x)`` with polynomial rates ``v_j``; steady state ``S * v(x) = 0``.

    Rates are zeroth order (constant) or first order in a single species (no bimolecular terms),
    so ``v_j`` are affine in ``x``. Conservation rows ``w^T x = T`` come from ``ker(S^T)``
    (``Matrix(S).T.nullspace()``), matching pre-reduction models that keep redundant linear laws.
    """
    if species < 1:
        raise ValueError("chemical_steady_state_random requires species >= 1")
    if reactions < 1:
        raise ValueError("chemical_steady_state_random requires reactions >= 1")

    rng = random.Random(seed)
    xs = [Symbol(f"x{i}") for i in range(1, species + 1)]
    s_entries: list[list[int]] = [
        [rng.randint(-2, 2) for _ in range(reactions)] for _ in range(species)
    ]
    ks = [Rational(rng.randint(1, 9), rng.randint(1, 9)) for _ in range(reactions)]

    rates: list[Expr] = []
    for j in range(reactions):
        typ = rng.randint(0, 1)
        if typ == 0:
            rates.append(ks[j])
        else:
            idx = rng.randrange(species)
            rates.append(ks[j] * xs[idx])

    eqs: list[Expr] = []
    for i in range(species):
        acc: Expr = Integer(0)
        for j in range(reactions):
            acc += s_entries[i][j] * rates[j]
        eqs.append(expand(acc))

    m = Matrix(s_entries)
    for v in m.T.nullspace():
        lin = Integer(0)
        for i in range(species):
            lin += v[i, 0] * xs[i]
        total = Rational(rng.randint(1, 24), rng.randint(1, 6))
        eqs.append(expand(lin - total))

    return eqs


def block_gauge(m: int, *, seed: int) -> list[Expr]:
    """
    Synthetic ``MILP-friendly'' family: each equation has the form
    ``x_i * f_i(x) + y_i * g_i(x)`` with random sparse ``f_i`` and ``g_i`` of
    degree 2-3 in a shared variable pool.

    Zeroing the gauge variables ``{x_i, y_i}`` makes the whole equation
    identically zero, so MILP coverage is easy. The lex Gröbner basis sees
    ``m`` random multivariate polynomials whose ideal has no symmetry to
    exploit.
    """
    if m < 1:
        raise ValueError("block_gauge requires m >= 1")

    rng = random.Random(seed)
    shared_pool = max(3, m // 2)
    zs = [Symbol(f"z{i}") for i in range(shared_pool)]
    xs = [Symbol(f"xg{i}") for i in range(m)]
    ys = [Symbol(f"yg{i}") for i in range(m)]

    def random_sparse(degree: int) -> Expr:
        term_count = rng.randint(2, max(3, shared_pool))
        acc: Expr = Integer(0)
        for _ in range(term_count):
            coef = Integer(rng.randint(-5, 5))
            if coef == 0:
                coef = Integer(1)
            mono: Expr = Integer(1)
            for _ in range(rng.randint(1, degree)):
                mono *= rng.choice(zs)
            acc += coef * mono
        return acc

    eqs: list[Expr] = []
    for i in range(m):
        f = random_sparse(rng.choice([2, 3]))
        g = random_sparse(rng.choice([2, 3]))
        eqs.append(expand(xs[i] * f + ys[i] * g))
    return eqs


def factor_chain(n: int, k: int, *, seed: int) -> list[Expr]:
    """
    Synthetic ``MILP-friendly'' family: each equation is a product of
    overlapping linear factors ``prod_j (x_{i_j} - a_{i_j})``. Many factors
    are shared across equations, so a small set of definers covers everything;
    lex Gröbner instead sees high-degree dense polynomials after expansion.
    """
    if n < 2:
        raise ValueError("factor_chain requires n >= 2")
    if k < 2:
        raise ValueError("factor_chain requires k >= 2")

    rng = random.Random(seed)
    xs = [Symbol(f"u{i}") for i in range(n)]
    constants = [Integer(rng.randint(1, 9)) for _ in range(n)]

    eqs: list[Expr] = []
    for i in range(n):
        indices = sorted(set(rng.sample(range(n), min(k, n))))
        if i not in indices:
            indices[0] = i
        prod_expr: Expr = Integer(1)
        for j in indices:
            prod_expr *= (xs[j] - constants[j])
        eqs.append(expand(prod_expr))
    return eqs


FAMILY_HANDLERS: dict[str, Callable[[int], list[Expr]]] = {
    "cyclic": cyclic,
    "katsura": katsura,
    "noonburg": noonburg,
    "eco": eco,
    "sparse_toric": sparse_toric,
}

ALL_FAMILIES: tuple[str, ...] = tuple(
    sorted(
        (
            *FAMILY_HANDLERS.keys(),
            "nash",
            "adjacent_minors",
            "random_distances",
            "chemical_steady_state_random",
            "block_gauge",
            "factor_chain",
        ),
    ),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Emit one benchmark polynomial system as a Mathematica rule list.",
    )
    
    p.add_argument(
        "--family",
        required=True,
        choices=ALL_FAMILIES,
        help="Benchmark family name",
    )
    
    p.add_argument(
        "--n",
        type=int,
        default=None,
        help="Primary size parameter (cyclic, katsura, noonburg, eco, sparse_toric). "
        "Ignored for nash, adjacent_minors, random_distances, chemical_steady_state_random.",
    )
    
    p.add_argument(
        "--rows",
        type=int,
        default=None,
        help="Matrix row count for adjacent_minors (>=2).",
    )
    
    p.add_argument(
        "--cols",
        type=int,
        default=None,
        help="Matrix column count for adjacent_minors (>=2).",
    )

    p.add_argument(
        "--joints",
        type=int,
        default=None,
        help="Number of revolute joints for random_distances (>=1).",
    )

    p.add_argument(
        "--species",
        type=int,
        default=None,
        help="Number of species for chemical_steady_state_random (>=1).",
    )

    p.add_argument(
        "--reactions",
        type=int,
        default=None,
        help="Number of reactions for chemical_steady_state_random (>=1).",
    )

    p.add_argument(
        "--k",
        type=int,
        default=None,
        help="Factors per equation for factor_chain (>=2).",
    )

    p.add_argument(
        "--m",
        type=int,
        default=None,
        help="Number of polynomials for block_gauge (>=1).",
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for random_distances and chemical_steady_state_random (default: 42).",
    )
    
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write to this file instead of stdout.",
    )
    
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    fam = args.family

    if fam == "nash":
        eqs = nash_2x2()
    elif fam == "adjacent_minors":
        if args.rows is None or args.cols is None:
            print("adjacent_minors requires --rows and --cols", file=sys.stderr)
            sys.exit(2)
        eqs = adjacent_minors(args.rows, args.cols)
    elif fam == "random_distances":
        if args.joints is None:
            print(
                "random_distances requires --joints",
                file=sys.stderr,
            )
            sys.exit(2)
        eqs = random_distances(
            args.joints,
            seed=args.seed,
        )
    elif fam == "chemical_steady_state_random":
        if args.species is None or args.reactions is None:
            print(
                "chemical_steady_state_random requires --species and --reactions",
                file=sys.stderr,
            )
            sys.exit(2)
        eqs = chemical_steady_state_random(
            args.species,
            args.reactions,
            seed=args.seed,
        )
    elif fam == "block_gauge":
        if args.m is None:
            print("block_gauge requires --m", file=sys.stderr)
            sys.exit(2)
        eqs = block_gauge(args.m, seed=args.seed)
    elif fam == "factor_chain":
        if args.n is None or args.k is None:
            print("factor_chain requires --n and --k", file=sys.stderr)
            sys.exit(2)
        eqs = factor_chain(args.n, args.k, seed=args.seed)
    else:
        if args.n is None:
            print(f"{fam} requires --n", file=sys.stderr)
            sys.exit(2)
        eqs = FAMILY_HANDLERS[fam](args.n)

    text = rules_from_equations(eqs)
    if args.output is not None:
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        sys.stdout.write(text + "\n")

