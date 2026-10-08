#!/usr/bin/env python3
"""Print the rows of Table ``tab:appendix-zeroing`` from the zeroing matrix.

For each system of ``results/zeroing_engine_comparison.md``, the row is the
fastest completed variable-zeroing configuration or, if none completes, the
partial run with the fewest unresolved branches (ties broken by time).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MATRIX = REPO / "results" / "zeroing_engine_comparison.md"
BACKENDS: dict[str, str] = {
    "zeroing-m4gb": "M4GB",
    "zeroing-f5": "F5",
    "zeroing-gvw": "GVW",
    "zeroing-symbolica": "F4",
    "zeroing-singular-gb": "Singular",
}
LABELS: tuple[str, ...] = (
    "Signed distance, $n=12$",
    "Signed distance, $n=14$",
    "Signed distance, $n=16$",
    "Overlapping $(4,6,2)$",
    "Overlapping $(6,8,3)$",
    "Overlapping $(8,10,3)$",
    "Positive affine $(8,6,4)$",
    "Positive affine $(10,8,5)$",
    "Positive affine $(12,10,6)$",
    "Gated Cyclic-5",
    "Gated Cyclic-6",
    "Gated Cyclic-7",
    "Gated Katsura-6",
    "Gated Katsura-7",
    "Gated Katsura-8",
    "Gated dense, $n=5$",
    "Gated dense, $n=6$",
    "Gated dense, $n=7$",
    "Square affine $(8,4)$",
    "Square affine $(10,5)$",
)
CELL = re.compile(
    r"(?P<partial>P )?(?P<total>[0-9.]+)/[0-9.]+/(?P<first>[0-9.]+|--); N=(?P<nodes>\d+); "
    r"E=(?P<evaluations>\d+); O/S=(?P<timeouts>\d+)/(?P<splits>\d+); "
    r"C/U=(?P<components>\d+)/(?P<unresolved>\d+)"
)


@dataclass(frozen=True)
class Run:
    backend: str
    complete: bool
    total: float
    first: str
    nodes: int
    evaluations: int
    timeouts: int
    splits: int
    components: int
    unresolved: int

    def key(self) -> tuple[int, int, float]:
        return (0 if self.complete else 1, 0 if self.complete else self.unresolved, self.total)

    def row(self, label: str) -> str:
        time = f"{self.total:.3f}" if self.complete else f"{self.total:.3f}"
        first = f"{float(self.first):.3f}" if self.first != "--" else "--"
        timing = f"${time}/{first}$" if first != "--" else f"${time}$/--"
        result = "complete" if self.complete else "partial"
        return (
            f"{label} & {self.backend} & {timing} & {self.nodes} & {self.evaluations} & "
            f"${self.timeouts}/{self.splits}$ & ${self.components}/{self.unresolved}$ & {result} \\\\"
        )


def parse_matrix(text: str) -> list[list[Run]]:
    lines = [line for line in text.splitlines() if line.startswith("| `")]
    header = next(line for line in text.splitlines() if line.startswith("| system"))
    methods = [cell.strip().strip("`") for cell in header.strip("|").split("|")[1:]]
    systems = []
    for line in lines:
        cells = [cell.strip() for cell in line.strip("|").split("|")[1:]]
        runs = []
        for method, cell in zip(methods, cells, strict=True):
            match = CELL.search(cell)
            if match is None:
                continue
            runs.append(
                Run(
                    backend=BACKENDS[method],
                    complete=match["partial"] is None,
                    total=float(match["total"]),
                    first=match["first"],
                    nodes=int(match["nodes"]),
                    evaluations=int(match["evaluations"]),
                    timeouts=int(match["timeouts"]),
                    splits=int(match["splits"]),
                    components=int(match["components"]),
                    unresolved=int(match["unresolved"]),
                )
            )
        systems.append(runs)
    return systems


def main() -> int:
    systems = parse_matrix(MATRIX.read_text(encoding="utf-8"))
    if len(systems) != len(LABELS):
        raise SystemExit(f"expected {len(LABELS)} systems, found {len(systems)}")
    for label, runs in zip(LABELS, systems, strict=True):
        print(min(runs, key=Run.key).row(label) if runs else f"{label} & -- & -- \\\\")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
