"""The polynomial systems of the experiments: generating, downloading and loading them.

* The 20 generated baseline systems of Tables ``tab:benchmark-families`` and
  ``tab:baseline-comparison`` (``generate_matrix``).
* The size-graded families of Table ``tab:scaling-classes``: Nash equilibria of
  binary games, box-QP faces and Lotka--Volterra supports (``nash_game``,
  ``box_qp``, ``lotka_volterra``), and the reduced cyclic n-roots systems of
  SymbolicData (``symbolicdata_system``), extended past n=19 by ``rcyclic_polynomials``.
* Loading a Mathematica-format system file into SymPy (``load_system``) and
  writing a polynomial in Singular syntax (``poly_to_singular``).

Every system file holds Mathematica rules ``0 -> (polynomial)``, one per equation.
"""

from __future__ import annotations

import itertools
import json
import random
import re
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path

import sympy as sp

SYMBOLICDATA_COMMIT = "a5cdf5366150bffba22a2c91a86a012dd23831af"
SYMBOLICDATA_RAW = (
    f"https://raw.githubusercontent.com/symbolicdata/data/{SYMBOLICDATA_COMMIT}"
    "/XMLResources/IntPS/"
)
TOKEN = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")


@dataclass(frozen=True)
class Instance:
    suite: str
    name: str
    path: str
    variables: int
    equations: int
    solver_args: list[str]
    description: str
    tier: str | None = None
    raw_factor_choices: int | None = None
    expected_largest_component: str | None = None


def write_system(path: Path, polynomials: list[str]) -> None:
    path.write_text(
        "{\n  "
        + ",\n  ".join(f"0 -> ({polynomial})" for polynomial in polynomials)
        + "\n}\n",
        encoding="utf-8",
    )


def cyclic_polynomials(order: int) -> list[str]:
    if order < 1:
        raise ValueError("cyclic order must be at least 1")
    variables = [f"c{index}" for index in range(order)]
    polynomials = []
    for degree in range(1, order):
        terms = [
            "*".join(
                variables[(start + offset) % order] for offset in range(degree)
            )
            for start in range(order)
        ]
        polynomials.append(" + ".join(terms))
    polynomials.append("*".join(variables) + " - 1")
    return polynomials


def rcyclic_polynomials(order: int) -> list[str]:
    """``Singular.rcyclic_n`` as ``symbolicdata_system`` writes it: for each degree below n,
    the sum of the three cyclic windows of that length starting at x0, x1 and x2."""
    polynomials = []
    for degree in range(1, order):
        windows = sorted(sorted((start + offset) % order for offset in range(degree)) for start in range(3))
        polynomials.append("+".join("*".join(f"x{index}" for index in window) for window in windows))
    return polynomials


def gated_cyclic(output: Path, order: int) -> Instance:
    path = output / f"gated_cyclic_{order}.txt"
    write_system(path, [f"gate*({polynomial})" for polynomial in cyclic_polynomials(order)])
    return Instance(
        suite="adversarial",
        name=path.stem,
        tier="exhaustive_covers",
        path=path.name,
        variables=order + 1,
        equations=order,
        solver_args=["--gb", "m4gb"],
        description=(
            "I = gate*Cyclic-n. Two minimal hitting sets: the hyperplane "
            "{gate} and the full residual Cyclic-n ideal."
        ),
    )


def katsura_polynomials(order: int) -> list[str]:
    if order < 1:
        raise ValueError("Katsura order must be at least 1")
    variables = [f"q{index}" for index in range(order + 1)]
    normalization = (
        f"{variables[0]} + "
        + " + ".join(f"2*{variable}" for variable in variables[1:])
        + " - 1"
    )
    polynomials = [normalization]
    for target in range(order):
        terms = []
        for index in range(-order, order + 1):
            other = target - index
            if abs(other) <= order:
                terms.append(
                    f"{variables[abs(index)]}*{variables[abs(other)]}"
                )
        polynomials.append(" + ".join(terms) + f" - {variables[target]}")
    return polynomials


def gated_katsura(output: Path, order: int) -> Instance:
    path = output / f"gated_katsura_{order}.txt"
    write_system(path, [f"k_gate*({polynomial})" for polynomial in katsura_polynomials(order)])
    return Instance(
        suite="adversarial",
        name=path.stem,
        tier="exhaustive_covers",
        path=path.name,
        variables=order + 2,
        equations=order + 1,
        solver_args=["--gb", "m4gb"],
        description=(
            "I = k_gate*Katsura-n. Two minimal hitting sets: the common "
            "hyperplane {k_gate} and the full residual Katsura-n ideal."
        ),
    )


def random_dense_quadratics(
    variables: int, seed: int, *, count: int | None = None
) -> list[str]:
    if variables < 2:
        raise ValueError("dense quadratic systems require at least two variables")
    generator = random.Random(seed)
    names = [f"r{index}" for index in range(variables)]
    polynomials = []
    for _ in range(variables if count is None else count):
        terms = []
        for index, left in enumerate(names):
            for right in names[index:]:
                terms.append(
                    f"{generator.choice((-3, -2, -1, 1, 2, 3))}*{left}*{right}"
                )
        for variable in names:
            terms.append(
                f"{generator.choice((-3, -2, -1, 1, 2, 3))}*{variable}"
            )
        terms.append(str(generator.choice((-3, -2, -1, 1, 2, 3))))
        polynomials.append(" + ".join(terms))
    return polynomials


def gated_random_dense_quadratic(
    output: Path, variables: int, seed: int
) -> Instance:
    path = output / f"gated_random_dense_quadratic_n{variables}_seed{seed}.txt"
    write_system(
        path,
        [
            f"q_gate*({polynomial})"
            for polynomial in random_dense_quadratics(variables, seed)
        ],
    )
    return Instance(
        suite="adversarial",
        name=path.stem,
        tier="exhaustive_covers",
        path=path.name,
        variables=variables + 1,
        equations=variables,
        solver_args=["--gb", "m4gb"],
        description=(
            "A selector gates a seeded generic dense quadratic complete intersection. "
            "Hitting sets: the common hyperplane and the full residual system."
        ),
    )


def overlapping_random_multiples(
    output: Path,
    variables: int,
    factor_pool: int,
    factors_per_equation: int,
    seed: int,
) -> Instance:
    if (
        variables < 3
        or factor_pool <= factors_per_equation
        or factors_per_equation < 2
    ):
        raise ValueError(
            "require at least three variables and 2 <= factors per equation < pool"
        )
    generator = random.Random(seed)
    for _ in range(100):
        groups = [
            tuple(sorted(generator.sample(range(factor_pool), factors_per_equation)))
            for _ in range(variables)
        ]
        used = set(itertools.chain.from_iterable(groups))
        common = set(groups[0]).intersection(*groups[1:])
        if len(set(groups)) == variables and len(used) == factor_pool and not common:
            break
    else:
        raise ValueError("could not generate a covering non-universal factor pattern")

    covers: list[tuple[int, ...]] = []
    for size in range(1, factor_pool + 1):
        for candidate in itertools.combinations(range(factor_pool), size):
            chosen = set(candidate)
            if any(set(cover).issubset(chosen) for cover in covers):
                continue
            if all(chosen.intersection(group) for group in groups):
                covers.append(candidate)

    factors = random_dense_quadratics(variables, seed + 1, count=factor_pool)
    polynomials = [
        "*".join(f"({factors[index]})" for index in group) for group in groups
    ]
    path = output / (
        f"overlapping_random_multiples_n{variables}_p{factor_pool}"
        f"_k{factors_per_equation}_seed{seed}.txt"
    )
    write_system(path, polynomials)
    return Instance(
        suite="positive-dimensional",
        name=path.stem,
        path=path.name,
        variables=variables,
        equations=variables,
        solver_args=["--gb", "symbolica"],
        description=(
            f"Each equation is the product of {factors_per_equation} factors "
            f"selected from a pool of {factor_pool} seeded dense quadratics; "
            "factors overlap between equations but none divides every equation."
        ),
        expected_largest_component=(
            f"The {len(covers)} minimal covers contain between "
            f"{min(map(len, covers))} and {max(map(len, covers))} random quadratics."
        ),
    )


def affine_factor_hypergraph(
    output: Path,
    variables: int,
    degree: int,
    seed: int,
    *,
    equations: int | None = None,
    suite: str = "adversarial",
) -> Instance:
    equation_count = variables if equations is None else equations
    if (
        variables < 4
        or not 1 <= equation_count <= variables
        or not 2 <= degree <= variables
    ):
        raise ValueError(
            "require affine variables >= 4, 1 <= equations <= variables, and "
            "2 <= degree <= variables"
        )
    generator = random.Random(seed)
    forms = []
    for index in range(2 * variables):
        left = index % variables
        middle = (index + 1) % variables
        right = (index + 3) % variables
        alpha, beta = (index % 5) + 1, (index % 3) + 1
        constant = generator.randint(1, 11)
        forms.append(
            f"(u{left} + {alpha}*u{middle} - {beta}*u{right} - {constant})"
        )
    polynomials = []
    for index in range(equation_count):
        chosen = generator.sample(range(2 * variables), degree)
        if index not in chosen:
            chosen[0] = index
        polynomials.append("*".join(forms[choice] for choice in chosen))
    dimension_label = (
        ""
        if equation_count == variables
        else f"_m{equation_count}"
    )
    path = output / (
        f"affine_factor_hypergraph_n{variables}{dimension_label}_k{degree}_seed{seed}.txt"
    )
    write_system(path, polynomials)
    return Instance(
        suite=suite,
        name=path.stem,
        tier="exhaustive_covers" if suite == "adversarial" else None,
        path=path.name,
        variables=variables,
        equations=equation_count,
        solver_args=["--gb", "symbolica"],
        description=(
            "Overlapping sparse affine factors; "
            "each candidate is linear while the expanded full ideal is dense."
        ),
        expected_largest_component=(
            f"Every cover has at most {equation_count} affine-linear generators in "
            f"{variables} variables, so every component has dimension at least "
            f"{variables - equation_count}."
            if suite == "positive-dimensional"
            else None
        ),
    )


def signed_distance_cycle(output: Path, vertices: int) -> Instance:
    if vertices < 4 or vertices % 2:
        raise ValueError("use an even signed-distance cycle of size at least four")
    polynomials = []
    for vertex in range(vertices):
        following = (vertex + 1) % vertices
        difference = f"z{following} - z{vertex}"
        polynomials.append(f"({difference} - 1)*({difference} + 1)")
    path = output / f"signed_distance_cycle_n{vertices}.txt"
    write_system(path, polynomials)
    return Instance(
        suite="positive-dimensional",
        name=path.stem,
        path=path.name,
        variables=vertices,
        equations=vertices,
        raw_factor_choices=1 << vertices,
        solver_args=["--gb", "m4gb"],
        description=(
            "Every edge of a closed one-dimensional linkage has signed length +1 or "
            "-1. Balanced choices give translation lines."
        ),
        expected_largest_component=(
            "A dimension-one subvariety; an interreduced cover may group balanced lines."
        ),
    )


def generate_matrix(output: Path) -> list[Instance]:
    """Generate each system in the consolidated comparison matrix exactly once."""
    output.mkdir(parents=True, exist_ok=True)
    instances = [
        *(signed_distance_cycle(output, size) for size in (12, 14, 16)),
        overlapping_random_multiples(output, 4, 6, 2, 31415),
        overlapping_random_multiples(output, 6, 8, 3, 31415),
        overlapping_random_multiples(output, 8, 10, 3, 31415),
        affine_factor_hypergraph(
            output,
            8,
            4,
            42,
            equations=6,
            suite="positive-dimensional",
        ),
        affine_factor_hypergraph(
            output,
            10,
            5,
            42,
            equations=8,
            suite="positive-dimensional",
        ),
        affine_factor_hypergraph(
            output,
            12,
            6,
            42,
            equations=10,
            suite="positive-dimensional",
        ),
        gated_cyclic(output, 5),
        gated_cyclic(output, 6),
        gated_cyclic(output, 7),
        gated_katsura(output, 6),
        gated_katsura(output, 7),
        gated_katsura(output, 8),
        gated_random_dense_quadratic(output, 5, 137),
        gated_random_dense_quadratic(output, 6, 137),
        gated_random_dense_quadratic(output, 7, 137),
        affine_factor_hypergraph(output, 8, 4, 42),
        affine_factor_hypergraph(output, 10, 5, 42),
    ]
    manifest = {"schema": 1, "instances": [asdict(instance) for instance in instances]}
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return instances


MATRIX_REQUIRED_COVERS = {
    "signed_distance_cycle_n12": 924,
    "signed_distance_cycle_n14": 3432,
    "signed_distance_cycle_n16": 12870,
    "overlapping_random_multiples_n4_p6_k2_seed31415": 4,
    "overlapping_random_multiples_n6_p8_k3_seed31415": 16,
    "overlapping_random_multiples_n8_p10_k3_seed31415": 28,
    "affine_factor_hypergraph_n8_m6_k4_seed42": 63,
    "affine_factor_hypergraph_n10_m8_k5_seed42": 239,
    "affine_factor_hypergraph_n12_m10_k6_seed42": 833,
    "gated_cyclic_5": 2,
    "gated_cyclic_6": 2,
    "gated_cyclic_7": 2,
    "gated_katsura_6": 2,
    "gated_katsura_7": 2,
    "gated_katsura_8": 2,
    "gated_random_dense_quadratic_n5_seed137": 2,
    "gated_random_dense_quadratic_n6_seed137": 2,
    "gated_random_dense_quadratic_n7_seed137": 2,
    "affine_factor_hypergraph_n8_k4_seed42": 178,
    "affine_factor_hypergraph_n10_k5_seed42": 309,
}


def mixed_profile_product(players: list[int], profile: tuple[int, ...]) -> str:
    """Probability that the players in ``players`` play the pure ``profile``."""
    return "*".join(
        f"p{player}" if action else f"(1-p{player})"
        for player, action in zip(players, profile, strict=True)
    ) or "1"


def nash_game(output: Path, players: int, seed: int, payoff_bound: int) -> Instance:
    """Nash equilibria of a binary game with integer payoff gaps in [-bound, bound].

    Player i mixes with probability p_i and is either pure (p_i in {0, 1}) or
    indifferent, i.e. its payoff gap, multilinear in the other players'
    probabilities, vanishes.
    """
    rng = random.Random(seed)
    polynomials = []
    for player in range(players):
        others = [other for other in range(players) if other != player]
        gap = " + ".join(
            f"({rng.randint(-payoff_bound, payoff_bound)})*"
            f"{mixed_profile_product(others, profile)}"
            for profile in itertools.product((0, 1), repeat=players - 1)
        )
        polynomials.append(f"p{player}*(p{player}-1)*({gap})")
    kind = "generic" if payoff_bound > 1 else "degenerate"
    path = output / f"nash_{kind}_n{players}.txt"
    write_system(path, polynomials)
    return Instance(
        suite="applications",
        name=path.stem,
        path=path.name,
        variables=players,
        equations=players,
        raw_factor_choices=3**players,
        solver_args=[],
        description=f"Nash equilibria of a {players}-player binary game, {kind} payoffs.",
    )


def box_qp(output: Path, variables: int, seed: int, density: float) -> Instance:
    """Critical points of x^T Q x / 2 + c^T x on every face of the unit box."""
    rng = random.Random(seed)
    upper = {
        (row, column): rng.randint(-50, 50)
        for row in range(variables)
        for column in range(row, variables)
        if row == column or rng.random() < density
    }
    hessian = {**upper, **{(column, row): value for (row, column), value in upper.items()}}
    polynomials = []
    for row in range(variables):
        gradient = " + ".join(
            f"({hessian[row, column]})*x{column}"
            for column in range(variables)
            if hessian.get((row, column), 0)
        )
        polynomials.append(f"x{row}*(x{row}-1)*({gradient} + ({rng.randint(-50, 50)}))")
    path = output / f"box_qp_n{variables}.txt"
    write_system(path, polynomials)
    return Instance(
        suite="applications",
        name=path.stem,
        path=path.name,
        variables=variables,
        equations=variables,
        raw_factor_choices=3**variables,
        solver_args=[],
        description=f"Box-QP critical points on all faces, n={variables}.",
    )


def lotka_volterra(output: Path, species: int, seed: int, density: float) -> Instance:
    """Equilibria x_i (r_i + sum_j a_ij x_j) = 0 of a random Lotka--Volterra community."""
    rng = random.Random(seed)
    polynomials = []
    for row in range(species):
        interactions = [f"({-rng.randint(1, 5)})*x{row}"] + [
            f"({rng.randint(-5, 5)})*x{column}"
            for column in range(species)
            if column != row and rng.random() < density
        ]
        polynomials.append(f"x{row}*({rng.randint(1, 5)} + {' + '.join(interactions)})")
    path = output / f"lotka_volterra_n{species}.txt"
    write_system(path, polynomials)
    return Instance(
        suite="applications",
        name=path.stem,
        path=path.name,
        variables=species,
        equations=species,
        raw_factor_choices=2**species,
        solver_args=[],
        description=f"Lotka--Volterra equilibria on all supports, {species} species.",
    )


def symbolicdata_system(output: Path, file_name: str) -> Path | None:
    """One IntPS system with its variables renamed to x0, x1, ... in <vars> order."""
    xml_text = urllib.request.urlopen(SYMBOLICDATA_RAW + file_name, timeout=60).read()
    root = ET.fromstring(xml_text)
    names = [name.strip() for name in (root.findtext("vars") or "").split(",") if name.strip()]
    renaming = {name: f"x{index}" for index, name in enumerate(names)}
    polynomials = [
        TOKEN.sub(lambda match: renaming[match.group(0)], " ".join((poly.text or "").split()))
        for poly in root.iterfind("basis/poly")
    ]
    if not polynomials:
        return None
    path = output / f"{Path(file_name).stem}.txt"
    write_system(path, polynomials)
    return path


def strip_outer_braces(text: str) -> str:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text[1:-1].strip()
    return text


def split_top_level(text: str) -> list[str]:
    """The comma-separated parts of ``text`` outside every bracket."""
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    for ch in text:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(depth - 1, 0)
        elif ch == "," and depth == 0:
            part = "".join(buf).strip()
            if part:
                parts.append(part)
            buf = []
            continue
        buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def mathematica_rules(text: str) -> list[str]:
    """The equations of a Mathematica system file, one rule or equation per entry, as written."""
    return split_top_level(strip_outer_braces(text))


def mathematica_to_sympy(expr: str) -> tuple[list[sp.Expr], set[sp.Symbol]]:
    """
    Convert Mathematica system of rational functions to SymPy polynomials.

    Args:
        expr: Mathematica expression string. Can be a single expression,
            a comma-separated list, or a Mathematica list {..}.

    Returns:
        List of SymPy expressions.
        Set of SymPy symbols.
    """
    from sympy import E, I, Symbol, fraction, pi, together
    from sympy.parsing.sympy_parser import (
        convert_xor,
        implicit_multiplication_application,
        parse_expr,
        standard_transformations,
    )

    sympy_greek_map = {
        r"\[Pi]": "pi",
        r"\[E]": "E",
        r"\[Alpha]": "alpha",
        r"\[Beta]": "beta",
        r"\[Gamma]": "gamma",
        r"\[Delta]": "delta",
        r"\[Epsilon]": "epsilon",
        r"\[Lambda]": "lambda_",
        r"\[Mu]": "mu",
        r"\[Sigma]": "sigma",
        r"\[Tau]": "tau",
        r"\[Phi]": "phi",
        r"\[Omega]": "omega",
    }

    def split_top_level_once(text: str, op: str) -> tuple[str, str] | None:
        depth = 0
        i = 0
        while i <= len(text) - len(op):
            ch = text[i]
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth = max(depth - 1, 0)
            if depth == 0 and text[i : i + len(op)] == op:
                return text[:i].strip(), text[i + len(op) :].strip()
            i += 1
        return None

    cleaned = strip_outer_braces(expr)
    if not cleaned:
        return [], set()

    raw_parts = split_top_level(cleaned)
    normalized_parts: list[str] = []

    for part in raw_parts:
        part = part.strip().rstrip(",")
        if not part:
            continue

        eq_split = split_top_level_once(part, "==")
        if eq_split is not None:
            lhs, rhs = eq_split
            part = f"({lhs}) - ({rhs})"
        else:
            rule_split = split_top_level_once(part, "->")
            if rule_split is None:
                rule_split = split_top_level_once(part, ":>")
            if rule_split is not None:
                lhs, rhs = rule_split
                part = f"({lhs}) - ({rhs})"

        for math_symbol, py_symbol in sympy_greek_map.items():
            part = part.replace(math_symbol, py_symbol)

        part = re.sub(
            r"(\d+(?:\.\d*)?|\.\d+)\s*\*\^\s*([+-]?\d+)",
            r"\1*10**\2",
            part,
        )

        part = re.sub(r"\)\s*\(", r")*(", part)
        part = re.sub(r"\s+", " ", part).strip()

        normalized_parts.append(part)

    if not normalized_parts:
        return [], set()

    token_re = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b")
    reserved = {"pi", "E", "I"}
    symbol_names: set[str] = set()
    for part in normalized_parts:
        for name in token_re.findall(part):
            if name not in reserved:
                symbol_names.add(name)

    local_symbols = {name: Symbol(name) for name in symbol_names}
    local_dict = {**local_symbols, "pi": pi, "E": E, "I": I}

    transformations = standard_transformations + (
        convert_xor,
        implicit_multiplication_application,
    )

    sympy_exprs = []
    for part in normalized_parts:
        sympy_exprs.append(
            parse_expr(
                part,
                local_dict=local_dict,
                transformations=transformations,
            )
        )

    cleared_exprs = []
    for sym_expr in sympy_exprs:
        num, den = fraction(together(sym_expr))
        if den != 1:
            # Clear rational functions by multiplying by the divisor.
            sym_expr = num
        cleared_exprs.append(sym_expr)
    sympy_exprs = cleared_exprs

    all_symbols: set[sp.Symbol] = set()
    for sym_expr in sympy_exprs:
        all_symbols.update(sym_expr.free_symbols)

    return sympy_exprs, all_symbols


_CONSTANT_REPLACEMENTS: tuple[tuple[sp.Basic, sp.Symbol], ...] = (
    # ``mathematica_to_sympy`` maps Mathematica's Pi/E/I to SymPy's *constants*
    # (sp.pi, sp.E, sp.I). Those aren't returned by ``free_symbols``, so external
    # CAS drivers (Singular, M2) silently get a ring that's missing them
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


def poly_to_singular(poly: sp.Basic) -> str:
    """Convert a SymPy polynomial expression to Singular syntax."""
    text = sp.sstr(sp.expand(poly))
    return text.replace("**", "^")


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


def min_fill_variables(polynomials: list[sp.Basic], variables: list[str]) -> list[str]:
    """`variables` in the min-fill order of the supports of `polynomials`."""
    supports = [sorted(variables.index(str(s)) for s in p.free_symbols if str(s) in variables) for p in polynomials]
    return [variables[i] for i in min_fill_order(len(variables), supports)]
