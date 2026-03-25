"""Tests for scripts/profile_python_script.py."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from scripts.profile_python_script import (
    _is_project_file,
    build_cprofile_command,
    format_display_label,
    format_report,
    load_profile_rows,
    main,
    rank_optimization_candidates,
)


def test_format_display_label_short_and_full() -> None:
    assert (
        format_display_label("/a/b/linear_substitution_order.py", 393, "best_linear_substitutions_order")
        == "best_linear_substitutions_order @ linear_substitution_order.py:393"
    )
    long = "/very/long/path/pkg/mod.py"
    assert format_display_label(long, 1, "f") == "f @ mod.py:1"
    assert format_display_label(long, 1, "f", full_path=True) == f"f @ {long}:1"


def test_is_project_file_excludes_venv_third_party() -> None:
    hints = ("soothing",)
    assert _is_project_file(
        "/Users/me/soothing/.venv/lib/python3.13/site-packages/sympy/core.py", hints
    ) is False
    assert _is_project_file(
        "/Users/me/soothing/.venv/lib/python3.13/site-packages/soothing/__init__.py", hints
    ) is True
    assert _is_project_file("/Users/me/soothing/linear_substitution_order.py", hints) is True


def test_build_cprofile_command_basic() -> None:
    cmd = build_cprofile_command(
        python_executable=sys.executable,
        output_prof=Path("/tmp/out.prof"),
        target_script=Path("/proj/foo.py"),
        script_args=["--bar", "baz"],
    )
    assert cmd[0] == sys.executable
    assert "-m" in cmd
    assert "cProfile" in cmd
    assert "-o" in cmd
    oi = cmd.index("-o")
    assert cmd[oi + 1] == "/tmp/out.prof"
    assert cmd[-3:] == [str(Path("/proj/foo.py")), "--bar", "baz"]


def test_load_profile_rows_from_real_run(tmp_path: Path) -> None:
    tiny = tmp_path / "tiny.py"
    tiny.write_text(
        "def hot():\n"
        "    s = 0\n"
        "    for i in range(5000):\n"
        "        s += i * i\n"
        "    return s\n"
        "hot()\n",
        encoding="utf-8",
    )
    prof = tmp_path / "run.prof"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "cProfile",
            "-o",
            str(prof),
            str(tiny),
        ],
        check=True,
        cwd=str(tmp_path),
    )
    rows = load_profile_rows(prof)
    assert rows
    labels = {r.label for r in rows}
    assert any("hot" in lab for lab in labels)


def test_rank_optimization_candidates_prefers_hot_and_frequent() -> None:
    from scripts.profile_python_script import ProfileRow

    rows = [
        ProfileRow(
            ncalls=1,
            prim_calls=1,
            tottime=0.01,
            cumtime=100.0,
            filename="wrapper.py",
            lineno=1,
            funcname="thin",
        ),
        ProfileRow(
            ncalls=5000,
            prim_calls=5000,
            tottime=80.0,
            cumtime=80.0,
            filename="leaf.py",
            lineno=2,
            funcname="heavy",
        ),
        ProfileRow(
            ncalls=2,
            prim_calls=2,
            tottime=0.001,
            cumtime=0.001,
            filename="noise.py",
            lineno=3,
            funcname="tiny",
        ),
    ]
    ranked = rank_optimization_candidates(rows, project_hints=("wrapper", "leaf"))
    assert ranked[0].row.funcname in ("heavy", "thin")
    assert any(c.row.funcname == "heavy" for c in ranked[:3])


def test_format_report_contains_sections() -> None:
    from scripts.profile_python_script import ProfileRow

    rows = [
        ProfileRow(10, 10, 1.0, 1.0, "a.py", 1, "f"),
        ProfileRow(1, 1, 0.5, 2.0, "b.py", 2, "g"),
    ]
    text = format_report(rows, project_hints=("a",), top_n=5)
    assert "Cumulative time" in text
    assert "Self time" in text
    assert "Call volume" in text
    assert "Optimization candidates" in text
    assert "Project-local" in text
    assert "f @ a.py:1" in text


def test_main_exits_nonzero_on_target_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bad = tmp_path / "bad.py"
    # cProfile treats SystemExit as a normal script completion (exit 0); use an error instead.
    bad.write_text("1 / 0\n", encoding="utf-8")
    out = tmp_path / "out.prof"
    argv = [
        "profile_python_script.py",
        "--output",
        str(out),
        str(bad),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code != 0


def test_main_success_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tiny = tmp_path / "ok.py"
    tiny.write_text("print('ok')\n", encoding="utf-8")
    out = tmp_path / "p.prof"
    argv = [
        "profile_python_script.py",
        "--output",
        str(out),
        "--top",
        "15",
        str(tiny),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    main()
    assert out.is_file()
