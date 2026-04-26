from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "run_milp_cover_all.py"


def load_runner_module():
    spec = importlib.util.spec_from_file_location("run_milp_cover_all", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_main_runs_txt_systems_in_sorted_order_and_writes_logs(tmp_path, capsys) -> None:
    runner = load_runner_module()
    systems_dir = tmp_path / "systems"
    systems_dir.mkdir()
    (systems_dir / "b.txt").write_text("0 -> (b)\n", encoding="utf-8")
    (systems_dir / "a.txt").write_text("0 -> (a)\n", encoding="utf-8")
    (systems_dir / "generate.py").write_text("print('skip')\n", encoding="utf-8")
    (systems_dir / "notes.nb").write_text("skip\n", encoding="utf-8")
    log_dir = tmp_path / "logs"

    calls: list[list[str]] = []

    def fake_run(command: list[str], cwd: Path):
        calls.append(command)
        system_name = Path(command[-3]).name
        return runner.CommandResult(0, f"ok {system_name}\n")

    exit_code = runner.main(
        [
            "--systems-dir",
            str(systems_dir),
            "--objective",
            "degree",
            "--debug",
            "--log-dir",
            str(log_dir),
        ],
        run_command=fake_run,
        cwd=tmp_path,
    )

    assert exit_code == 0
    assert [Path(command[-3]).name for command in calls] == ["a.txt", "b.txt"]
    assert calls[0] == [
        "cargo",
        "run",
        "-p",
        "symbolica-elim",
        "--bin",
        "milp_cover",
        "--",
        str(systems_dir / "a.txt"),
        "--objective",
        "degree",
    ]
    assert (log_dir / "a.log").read_text(encoding="utf-8") == "ok a.txt\n"
    assert (log_dir / "b.log").read_text(encoding="utf-8") == "ok b.txt\n"
    captured = capsys.readouterr()
    assert "passed: 2" in captured.out
    assert "failed: 0" in captured.out


def test_main_continues_after_failure_and_returns_nonzero(tmp_path, capsys) -> None:
    runner = load_runner_module()
    systems_dir = tmp_path / "systems"
    systems_dir.mkdir()
    (systems_dir / "bad.txt").write_text("0 -> (bad)\n", encoding="utf-8")
    (systems_dir / "good.txt").write_text("0 -> (good)\n", encoding="utf-8")

    def fake_run(command: list[str], cwd: Path):
        system_name = Path(command[-3]).name
        if system_name == "bad.txt":
            return runner.CommandResult(3, "infeasible\n")
        return runner.CommandResult(0, "ok\n")

    exit_code = runner.main(
        ["--systems-dir", str(systems_dir)],
        run_command=fake_run,
        cwd=tmp_path,
    )

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "passed: 1" in captured.out
    assert "failed: 1" in captured.out
    assert "bad.txt: exit 3" in captured.out
