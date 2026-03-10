import re
from sympy import Poly, Symbol

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


def mathematica_to_jax(expr: str, var_to_index: dict[str, int]) -> str:
    """
    Convert Mathematica expression to Python/JAX expression.

    Args:
        expr: Mathematica expression string
        var_to_index: Mapping from variable names to array indices

    Handles:
    - Power notation (^) -> **
    - Greek symbols (\\[Symbol]) -> symbol
    - Pi constant
    - Implicit multiplication (e.g., 2x -> 2*x, 3(x+y) -> 3*(x+y))
    """
    # Replace power operator
    result = expr.replace("^", "**")

    # Replace Greek symbols and special characters
    for mathematica_symbol, python_symbol in _GREEK_MAP.items():
        result = result.replace(mathematica_symbol, python_symbol)

    # Add explicit multiplication for implicit multiplication cases:
    # 1. Space between identifiers/numbers (e.g., "a b" -> "a*b", "2 x" -> "2*x")
    # 2. Number directly followed by letter (e.g., "2x" -> "2*x")
    # 3. Number/variable followed by opening parenthesis
    # 4. Closing parenthesis followed by number/variable

    # Space between closing paren and identifier
    result = re.sub(r"\)\s+([a-zA-Z])", r")*\1", result)

    # Space between closing paren and number
    result = re.sub(r"\)\s+(\d)", r")*\1", result)

    # Space between identifier and opening paren
    result = re.sub(r"([a-zA-Z_]\w*)\s+\(", r"\1*(", result)

    # Space between number and opening paren
    result = re.sub(r"(\d)\s+\(", r"\1*(", result)

    # Space between two identifiers (e.g., "DkDkh1 kkh" -> "DkDkh1*kkh")
    result = re.sub(r"([a-zA-Z_]\w*)\s+([a-zA-Z])", r"\1*\2", result)

    # Space between number and identifier (e.g., "45 DkDkh1" -> "45*DkDkh1")
    result = re.sub(r"(\d)\s+([a-zA-Z])", r"\1*\2", result)

    # Space between identifier and number (e.g., "x 2" -> "x*2")
    result = re.sub(r"([a-zA-Z_]\w*)\s+(\d)", r"\1*\2", result)

    # Number directly followed by letter without space (e.g., "2x" -> "2*x")
    result = re.sub(r"(\d)([a-zA-Z])", r"\1*\2", result)

    # Variable directly followed by opening parenthesis (no space already handled above with space)
    result = re.sub(r"([a-zA-Z_]\w*)\(", r"\1*(", result)

    # Number directly followed by opening parenthesis (no space)
    result = re.sub(r"(\d)\(", r"\1*(", result)

    # Closing parenthesis directly followed by letter
    result = re.sub(r"\)([a-zA-Z])", r")*\1", result)

    # Closing parenthesis directly followed by number
    result = re.sub(r"\)(\d)", r")*\1", result)

    # Closing parenthesis directly followed by space and open parenthesis
    result = re.sub(r"\) \(", r")*(", result)

    # Closing parenthesis directly followed by space and open parenthesis
    result = re.sub(r"\)\(", r")*(", result)

    # Replace variables with array indexing
    # Match identifiers that aren't part of jnp.* patterns
    def replace_with_index(match):
        word = match.group(0)
        # Check if this word is preceded by "jnp." or followed by "."
        start_pos = match.start()
        end_pos = match.end()

        # Check if preceded by "jnp."
        if start_pos >= 4 and result[start_pos - 4 : start_pos] == "jnp.":
            return word

        # Check if this is "jnp" itself
        if word == "jnp" and end_pos < len(result) and result[end_pos] == ".":
            return word

        # Replace with array index if this is a variable
        if word in var_to_index:
            return f"x[{var_to_index[word]}]"

        return word

    # Find all identifiers (including those that might be multi-part like DkDkh1)
    result = re.sub(r"\b([a-zA-Z][a-zA-Z0-9]*)\b", replace_with_index, result)

    return result


def mathematica_to_sympy(expr: str) -> tuple[list[Poly], set[Symbol]]:
    """
    Convert Mathematica system of rational functions to SymPy polynomials.

    Args:
        expr: Mathematica expression string. Can be a single expression,
            a comma-separated list, or a Mathematica list {..}.

    Returns:
        List of SymPy polynomials.
        Set of SymPy symbols.
    """
    from sympy import E, I, Symbol, fraction, pi, together
    from sympy.parsing.sympy_parser import (
        parse_expr,
        standard_transformations,
        implicit_multiplication_application,
        convert_xor,
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

    gens = tuple(sorted(all_symbols, key=str))
    if gens:
        return [Poly(sym_expr, *gens) for sym_expr in sympy_exprs], all_symbols
    return [Poly(sym_expr) for sym_expr in sympy_exprs], all_symbols

def generate_variable_mapping(variables: set[str]) -> tuple[dict[str, int], list[str]]:
    """Generate mapping from variable names to array indices."""
    sorted_vars = sorted(variables)
    var_to_index = {var: i for i, var in enumerate(sorted_vars)}
    return var_to_index, sorted_vars


def generate_function(
    func_name: str, expression: str, var_to_index: dict[str, int]
) -> str:
    """Generate a pure function for computing an equation."""
    python_expr = mathematica_to_jax(expression, var_to_index)

    lines = [
        f"def {func_name}(x: Array) -> Array:",
        f"    return {python_expr}",
    ]

    return "\n".join(lines)


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


def convert_mathematica_system(
    mathematica_str: str,
    output_file: str | None = None,
) -> str:
    """
    Convert complete Mathematica system to Python/JAX code.

    Args:
        mathematica_str: String containing Mathematica system of equations
        output_file: Optional path to write generated code to file

    Returns:
        Complete Python code as a string
    """
    # Parse the system
    system = parse_mathematica_system(mathematica_str)

    # Extract all variables from all equations
    all_variables = set()
    for expr in system.values():
        all_variables.update(extract_variables(expr))

    # Generate code components
    imports = [
        "import jax.numpy as jnp",
        "from jax import Array",
        "",
        "",
    ]

    # Generate variable mapping
    var_to_index, sorted_vars = generate_variable_mapping(all_variables)

    functions = []
    func_names = []
    for var_name, expression in system.items():
        # Convert CamelCase to snake_case for function names
        func_name = re.sub(r"(?<!^)(?=[A-Z])", "_", var_name).lower()
        func_names.append(func_name)
        functions.append(generate_function(func_name, expression, var_to_index))

    system_evaluator = generate_system_evaluator(func_names, sorted_vars)

    # Combine all parts
    code_parts = imports + [f + "\n\n" for f in functions] + [system_evaluator]

    generated_code = "\n".join(code_parts)

    # Write to file if output path is provided
    if output_file is not None:
        with open(output_file, "w") as f:
            f.write(generated_code)

    return generated_code


# Example usage
