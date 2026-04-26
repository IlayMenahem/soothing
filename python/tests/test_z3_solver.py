from __future__ import annotations

from pathlib import Path

import pytest
import sympy as sp
import z3

from soothing.z3_solver import solve_system_text, sympy_to_z3


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _fraction_string(value: z3.ExprRef) -> str:
    assert isinstance(value, z3.RatNumRef)
    return f"{value.numerator_as_long()}/{value.denominator_as_long()}"


def test_noonburg_k1_has_expected_real_solution() -> None:
    result = solve_system_text(
        (PROJECT_ROOT / "systems" / "noonburg_k1.txt").read_text(encoding="utf-8"),
        max_solutions=2,
    )

    assert result.status == z3.unsat
    assert len(result.solutions) == 1
    assert _fraction_string(result.solutions[0]["x1"]) == "10/11"
    assert result.reached_limit is False


def test_nash_2x2_solves_expected_rationals() -> None:
    result = solve_system_text(
        (PROJECT_ROOT / "systems" / "nash_2x2.txt").read_text(encoding="utf-8"),
        max_solutions=2,
    )

    assert result.status == z3.unsat
    assert len(result.solutions) == 1
    solution = result.solutions[0]
    assert _fraction_string(solution["p"]) == "1/2"
    assert _fraction_string(solution["q"]) == "1/2"


def test_enumerates_distinct_models_until_limit() -> None:
    result = solve_system_text("0 -> (x^2 - 1)", max_solutions=2)

    assert result.status == z3.sat
    assert result.reached_limit is True
    assert {_fraction_string(solution["x"]) for solution in result.solutions} == {
        "-1/1",
        "1/1",
    }


def test_sympy_to_z3_rejects_negative_power() -> None:
    x = sp.Symbol("x")

    with pytest.raises(ValueError, match="non-negative integer powers"):
        sympy_to_z3(x**-1, {"x": z3.Real("x")})


def test_solves_rule_with_pi_denominator() -> None:
    result = solve_system_text(r"k -> x / (2 \[Pi]^2), 0 -> (x)", max_solutions=2)

    assert result.status == z3.unsat
    assert len(result.solutions) == 1
    solution = result.solutions[0]
    assert _fraction_string(solution["k"]) == "0/1"
    assert _fraction_string(solution["x"]) == "0/1"
