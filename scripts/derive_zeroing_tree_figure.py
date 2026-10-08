"""Derive paper/figs/variable_zeroing_tree.tex from the raw TikZ of the figure run.

Applies the plot-only changes of the current figure: a source header, "no hitting
sets" in place of "no covers", the SAT/UNSAT legend, and no unused macros.
"""
import json
import re
import sys
from pathlib import Path

repo = Path(sys.argv[1])
run = json.loads((repo / "results" / "physical_system_m4gb_5s_figure_run.json").read_text())
old_figure = (repo / "paper" / "figs" / "variable_zeroing_tree.tex").read_text()
raw = run["raw_tikz"]

legend = next(line for line in old_figure.splitlines() if "Branch outcome:" in line)
raw_legend = next(line for line in raw.splitlines() if "Cover outcome:" in line)
position = re.search(r"at \(([^)]*)\)", raw_legend).group(1)
legend = re.sub(r"at \([^)]*\)", f"at ({position})", legend, count=1)

lines = raw.splitlines()
first_line, rest = lines[0], lines[1:]
figure_lines = [
    first_line,
    "% Source: results/physical_system_m4gb_5s_figure_run.json",
    "% Plot-only changes: omit unused macros; label outcomes SAT/UNSAT.",
]
for line in rest:
    if line.startswith(r"\providecommand{\vzPruned}"):
        continue
    if "Cover outcome:" in line:
        figure_lines.append(legend)
        continue
    figure_lines.append(line.replace(r"\vzNoCovers\ no covers}", r"\vzNoCovers\ no hitting sets}"))
(repo / "paper" / "figs" / "variable_zeroing_tree.tex").write_text("\n".join(figure_lines) + "\n")
print(run["summary"])
