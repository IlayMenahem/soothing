import re

from sympy import Expr, Symbol

_GREEK_MAP = {
    r"\[Pi]": "jnp.pi",
    r"\[E]": "jnp.e",
    r"E": "jnp.e",
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


def parse_mathematica_system(mathematica_str: str) -> dict[str, str]:
    """
    Parse Mathematica system of equations into Python-ready expressions.

    Args:
        mathematica_str: String containing Mathematica rule definitions

    Returns:
        Dictionary mapping variable names to their expressions
    """
    # Remove outer braces and whitespace
    cleaned = mathematica_str.strip().strip("{}").strip()

    # Split by outer braces to get individual equations
    equations = cleaned.split(",")

    # Parse each equation
    system = {}
    for eq in equations:
        if "->" in eq:
            lhs, rhs = eq.split("->", 1)
            var_name = lhs.strip()
            system[var_name] = rhs.strip()

    return system


def extract_variables(expression: str) -> set[str]:
    """Extract all variable names from a Mathematica expression."""
    # Remove mathematical operators, numbers, and special symbols
    # Keep only identifiers
    pattern = r"\b([a-zA-Z][a-zA-Z0-9]*)\b"
    matches = re.findall(pattern, expression)

    # Filter out mathematical constants and keywords
    reserved = {"Pi", "E"}
    variables = {m for m in matches if m not in reserved}

    # Also extract greek variables
    for math_symbol, py_symbol in _GREEK_MAP.items():
        if "jnp" not in py_symbol:
            if math_symbol in expression:
                variables.add(py_symbol)

    return variables


def mathematica_to_sympy(expr: str) -> tuple[list[Expr], set[Symbol]]:
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

    def strip_outer_braces(text: str) -> str:
        text = text.strip()
        if text.startswith("{") and text.endswith("}"):
            return text[1:-1].strip()
        return text

    def split_top_level(text: str) -> list[str]:
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

    all_symbols: set[Symbol] = set()
    for sym_expr in sympy_exprs:
        all_symbols.update(sym_expr.free_symbols)

    return sympy_exprs, all_symbols


def generate_variable_mapping(variables: set[str]) -> tuple[dict[str, int], list[str]]:
    """Generate mapping from variable names to array indices."""
    sorted_vars = sorted(variables)
    var_to_index = {var: i for i, var in enumerate(sorted_vars)}
    return var_to_index, sorted_vars


def generate_system_evaluator(func_names: list[str], sorted_vars: list[str]) -> str:
    """Generate function to evaluate entire system."""
    # Generate unpacking statement
    unpacking = ", ".join(sorted_vars)

    # Generate return statement with individual function calls
    returns = ",\n        ".join(f"{name}(x)" for name in func_names)

    lines = [
        "def evaluate_system(x: Array) -> Array:",
        f"    {unpacking} = x",
        "    return jnp.array([",
        f"        {returns}",
        "    ])",
    ]

    return "\n".join(lines)
