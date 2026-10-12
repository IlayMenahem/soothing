# results/

Stored results of the experiments in `paper/sections/experiments.tex`, one
store per table or physical-system experiment. `.json`/`.jsonl` files are the raw
results (the scripts resume from them); `.md` files are the rendered reports.
All commands run from the repo root; each script has `--report-only` to print
the table without rerunning.

| Files | Script | Paper |
|---|---|---|
| `baselines.md` | `baseline_table.py --part baselines` | `tab:benchmark-families`, `tab:baseline-comparison` |
| `branching.md` | `baseline_table.py --part branching` | `tab:baseline-comparison`, `tab:appendix-branching` |
| `zeroing.md` | `baseline_table.py --part zeroing` | `tab:baseline-comparison`, `tab:appendix-zeroing` |
| `zeroing_cutoff_sweep.json` | `baseline_table.py --part cutoff-sweep` | zeroing cutoff sweep (appendix) |
| `scaling.jsonl`, `scaling.md` | `scaling_table.py --output-dir DIR` | `tab:scaling-classes` |
| `decomposition.jsonl`, `decomposition.md` | `scaling_decomposition_table.py` | `tab:scaling-decomposition` |
| `ideal_refine.jsonl`, `ideal_refine.md` | `scaling_decomposition_table.py --systems baseline` | IdealRefine paragraph |
| `physical_system.txt` | written by every `physical_*.py` script | the physical system: `systems/system.txt` without its last equation, the Rabinowitsch equation of `kappah != 0`, which the runs impose by saturation (`--nonzero kappah`) |
| `physical.json` | `physical_table.py` | `tab:physical-system` and the nine standalone baselines |
| `physical_primary.md` | `physical_primary_decomposition.py` | Alg. PrimaryBranching on the physical system |
| `physical_zeroing_figure.json` | `physical_zeroing_figure.py` | variable-zeroing tree figure (`paper/figs/variable_zeroing_tree.tex`) |

Results of experiments no longer in the paper (natural benchmarks, applications,
split values, zeroing variants on the physical system) were removed; they remain
in git history.
