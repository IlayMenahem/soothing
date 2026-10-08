#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from benchmark_idos import REPO, SYMBOLICA, run


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("backend")
    args = parser.parse_args()
    argv = [
        str(SYMBOLICA / "target" / "release" / "variable_zeroing"),
        str(REPO / "systems" / "system.txt"),
        "--gb",
        args.backend,
        "--timeout",
        "5000",
        "--max-runtime",
        "120000",
        "--ideal-saturation",
        "--summary",
    ]
    stdout, stderr, wall, killed, returncode = run(argv, 125.0)
    for line in stdout.splitlines():
        if line.startswith("ZEROING_SUMMARY "):
            print(line)
    print(
        "HARNESS_SUMMARY "
        + json.dumps(
            {
                "backend": args.backend,
                "argv": argv,
                "wall_secs": wall,
                "killed": killed,
                "returncode": returncode,
                "stderr_lines": len(stderr.splitlines()),
                "stderr_tail": stderr[-600:],
            },
            sort_keys=True,
        )
    )
    return 0 if returncode == 0 or killed else returncode


if __name__ == "__main__":
    raise SystemExit(main())
