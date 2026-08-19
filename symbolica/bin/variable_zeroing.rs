//! CLI for recursive factor branching with variable zeroing.
//!
//! Usage: `variable_zeroing [path] [--unsat-core-search]
//! [--saturate-each-variable]` (default path: `systems/system.txt`; default
//! Gröbner backend: Singular).

use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use symbolica::LicenseManager;
use symbolica::domains::rational::Q;
use symbolica::poly::PolyVariable;

use polynomial_cdcl::gb_search::GbBackend;
use polynomial_cdcl::{
    VariableZeroingCoverOutcome, VariableZeroingNodeResult, VariableZeroingOptions,
    ZeroingVariable, default_systems_dir_from_manifest,
    load_mathematica_system_from_path_with_var_names, polynomials_from_loaded, solve_by_zeroing,
};

#[derive(Debug)]
struct CliOptions {
    path: PathBuf,
    interreduce: bool,
    timeout_ms: u64,
    max_runtime_ms: Option<u64>,
    gb_backend: GbBackend,
    unsat_core_search: bool,
    saturate_each_variable: bool,
    max_nodes: usize,
    max_evaluations: usize,
    max_depth: usize,
    variable_names: Vec<String>,
    summary: bool,
    plot: Option<PathBuf>,
}

fn parse_usize(args: &mut impl Iterator<Item = String>, option: &str) -> Result<usize, String> {
    args.next()
        .ok_or_else(|| format!("{option} requires a value"))?
        .parse()
        .map_err(|error| format!("invalid {option}: {error}"))
}

fn parse_args(args: impl IntoIterator<Item = String>) -> Result<CliOptions, String> {
    let mut path = None;
    let mut interreduce = true;
    let mut timeout_ms = 5_000;
    let mut max_runtime_ms = None;
    let mut gb_backend = GbBackend::SingularGb;
    let mut unsat_core_search = false;
    let mut saturate_each_variable = false;
    let mut max_nodes = usize::MAX;
    let mut max_evaluations = usize::MAX;
    let mut max_depth = usize::MAX;
    let mut variable_names = Vec::new();
    let mut summary = false;
    let mut plot = None;
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--no-interreduce" => interreduce = false,
            "--timeout" => {
                timeout_ms = parse_usize(&mut args, "--timeout")? as u64;
            }
            "--max-runtime" => {
                max_runtime_ms = Some(parse_usize(&mut args, "--max-runtime")? as u64);
            }
            "--gb" => {
                let value = args.next().ok_or_else(|| {
                    "--gb requires m4gb, f5, gvw, symbolica, singular-gb, or singular-facstd"
                        .to_string()
                })?;
                gb_backend = GbBackend::parse(&value).ok_or_else(|| {
                    format!(
                        "invalid --gb {value:?}: expected m4gb, f5, gvw, symbolica, singular-gb, or singular-facstd"
                    )
                })?;
            }
            "--unsat-core-search" => unsat_core_search = true,
            "--no-unsat-core-search" => unsat_core_search = false,
            "--saturate-each-variable" => saturate_each_variable = true,
            "--max-nodes" => max_nodes = parse_usize(&mut args, "--max-nodes")?,
            "--max-covers" | "--max-evaluations" => {
                max_evaluations = parse_usize(&mut args, "--max-evaluations")?;
            }
            "--max-depth" => max_depth = parse_usize(&mut args, "--max-depth")?,
            "--variable" => variable_names.push(
                args.next()
                    .ok_or_else(|| "--variable requires a name".to_string())?,
            ),
            "--summary" => summary = true,
            "--plot" => {
                plot =
                    Some(PathBuf::from(args.next().ok_or_else(|| {
                        "--plot requires an output .tex path".to_string()
                    })?));
            }
            _ if arg.starts_with("--") => return Err(format!("unknown option {arg:?}")),
            _ if path.is_some() => return Err(format!("unexpected positional argument {arg:?}")),
            _ => path = Some(PathBuf::from(arg)),
        }
    }
    Ok(CliOptions {
        path: path.unwrap_or_else(|| default_systems_dir_from_manifest().join("system.txt")),
        interreduce,
        timeout_ms,
        max_runtime_ms,
        gb_backend,
        unsat_core_search,
        saturate_each_variable,
        max_nodes,
        max_evaluations,
        max_depth,
        variable_names,
        summary,
        plot,
    })
}

fn die(message: impl std::fmt::Display, code: i32) -> ! {
    eprintln!("{message}");
    std::process::exit(code)
}

fn main() {
    LicenseManager::set_license_key("5568d092#6bb8b4fb#ff936916-65ba-552f-b3c1-bdc64597cd21")
        .unwrap();
    polynomial_cdcl::exit_if_solver_worker();
    let cli = parse_args(std::env::args().skip(1)).unwrap_or_else(|error| die(error, 2));
    println!("Loading {}", cli.path.display());
    let load_started = Instant::now();
    let loaded = load_mathematica_system_from_path_with_var_names(&cli.path)
        .unwrap_or_else(|error| die(format!("load error: {error}"), 1));
    let polynomials = polynomials_from_loaded(&loaded, &Q)
        .unwrap_or_else(|error| die(format!("polynomial conversion: {error}"), 1));

    let selected_names: Vec<String> = if cli.variable_names.is_empty() {
        loaded.var_names.clone()
    } else {
        cli.variable_names.clone()
    };
    let variables: Vec<ZeroingVariable> = selected_names
        .iter()
        .map(|name| {
            let index = loaded
                .var_names
                .iter()
                .position(|candidate| candidate == name)
                .unwrap_or_else(|| die(format!("unknown --variable {name:?}"), 2));
            ZeroingVariable {
                name: name.clone(),
                variable: PolyVariable::from(loaded.id_to_sym[index]),
            }
        })
        .collect();
    println!(
        "Parsed {} equation(s), {} variable(s); {} zeroing candidate(s), elapsed {:?}",
        loaded.system.equations.len(),
        loaded.var_names.len(),
        variables.len(),
        load_started.elapsed()
    );
    println!(
        "Recursive branching: {}ms/cover, {} backend, unsat core search {}, saturation equations {}, max runtime {}, max nodes {}, max evaluations {}, max depth {}",
        cli.timeout_ms,
        cli.gb_backend.label(),
        if cli.unsat_core_search {
            "enabled"
        } else {
            "disabled"
        },
        if cli.saturate_each_variable {
            "per variable"
        } else {
            "shared product"
        },
        cli.max_runtime_ms
            .map_or_else(|| "unlimited".to_string(), |value| format!("{value}ms")),
        display_limit(cli.max_nodes),
        display_limit(cli.max_evaluations),
        display_limit(cli.max_depth)
    );

    let options = VariableZeroingOptions {
        interreduce: cli.interreduce,
        timeout: Duration::from_millis(cli.timeout_ms),
        max_runtime: cli.max_runtime_ms.map(Duration::from_millis),
        gb_backend: cli.gb_backend,
        unsat_core_search: cli.unsat_core_search,
        saturate_each_variable: cli.saturate_each_variable,
        max_nodes: cli.max_nodes,
        max_evaluations: cli.max_evaluations,
        max_depth: cli.max_depth,
        verbose: !cli.summary,
    };
    let result = solve_by_zeroing(polynomials, variables, &options)
        .unwrap_or_else(|error| die(format!("variable-zeroing error: {error}"), 1));

    if let Some(path) = &cli.plot {
        write_tikz_plot(path, &cli.path, &result.nodes)
            .unwrap_or_else(|error| die(format!("plot error: {error}"), 1));
        println!("Wrote recursion-tree plot to {}", path.display());
    }

    if !cli.summary {
        for (component_index, component) in result.components.iter().enumerate() {
            let decisions = format_decisions(&component.decisions);
            let projected: Vec<_> = component.original_basis_elements().collect();
            println!(
                "\ncomponent #{component_index} (node #{}, dimension {}, decisions: {decisions})",
                component.node_id,
                component
                    .dimension
                    .map_or_else(|| "unknown".to_string(), |value| value.to_string())
            );
            if component.leading_monomials_only() {
                println!("  original-variable leading monomials:");
            } else {
                println!("  original-variable basis elements:");
            }
            for (index, polynomial) in projected.iter().enumerate() {
                println!("    g{index} = {polynomial}");
            }
            let hidden = component.basis_len() - projected.len();
            if hidden != 0 {
                println!("  {hidden} Rabinowitsch-variable basis element(s) omitted");
            }
        }
        for (index, branch) in result.unresolved.iter().enumerate() {
            println!(
                "\nunresolved #{index}: {:?}, {} generator(s), {} variable(s) left, decisions: {}",
                branch.reason,
                branch.polynomials.len(),
                branch.remaining_variables.len(),
                format_decisions(&branch.decisions)
            );
        }
    }

    let first_solution = result.first_solution.map_or_else(
        || "unknown".to_string(),
        |duration| format!("{:.6}", duration.as_secs_f64()),
    );
    if cli.summary {
        for node in &result.nodes {
            println!("{}", format_node_summary(node));
        }
    }
    println!(
        "ZEROING_SUMMARY nodes={} covers={} evaluations={} components={} inconsistent={} core_pruned={} timeouts={} splits={} unresolved={} complete={} largest_depth={} first_solution_secs={} elapsed_secs={:.6}",
        result.stats.nodes,
        result.stats.covers,
        result.stats.evaluations,
        result.components.len(),
        result.stats.inconsistent,
        result.stats.core_pruned,
        result.stats.timeouts,
        result.stats.zeroing_splits,
        result.unresolved.len(),
        result.complete(),
        result.stats.largest_depth,
        first_solution,
        result.elapsed.as_secs_f64()
    );
}

fn format_node_summary(node: &VariableZeroingNodeResult) -> String {
    let zeroed = format_variable_list(
        node.decisions
            .iter()
            .filter(|decision| decision.is_zero)
            .map(|decision| decision.variable.as_str()),
    );
    let nonzero = format_variable_list(
        node.decisions
            .iter()
            .filter(|decision| !decision.is_zero)
            .map(|decision| decision.variable.as_str()),
    );
    format!(
        "ZEROING_NODE node={} zeroed={} nonzero={} covers={} evaluations={} consistent={} inconsistent={} core_pruned={} timeouts={} splits={} elapsed_secs={:.6}",
        node.node_id,
        zeroed,
        nonzero,
        node.covers,
        node.evaluations,
        node.consistent,
        node.inconsistent,
        node.core_pruned,
        node.timeouts,
        node.zeroing_splits,
        node.elapsed.as_secs_f64()
    )
}

/// Write a TikZ recursion tree in the same cover-centric style as the paper.
/// Every hitting-set node contains the time spent checking that set.
fn write_tikz_plot(
    output_path: &Path,
    input_path: &Path,
    nodes: &[VariableZeroingNodeResult],
) -> Result<(), String> {
    let input_label = input_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("P");
    let tikz = render_tikz_plot(input_label, nodes)?;
    std::fs::write(output_path, tikz)
        .map_err(|error| format!("cannot write {}: {error}", output_path.display()))
}

fn render_tikz_plot(
    input_label: &str,
    nodes: &[VariableZeroingNodeResult],
) -> Result<String, String> {
    if nodes.is_empty() {
        return Err("the run did not create a recursion node".to_string());
    }

    let nodes_by_id: HashMap<usize, &VariableZeroingNodeResult> =
        nodes.iter().map(|node| (node.node_id, node)).collect();
    let mut rows: BTreeMap<usize, Vec<(usize, Option<usize>)>> = BTreeMap::new();
    for node in nodes {
        let row = rows.entry(node.decisions.len()).or_default();
        row.extend(plot_members(node));
    }

    let mut positions = HashMap::new();
    for row in rows.values() {
        let mut raw_positions = Vec::with_capacity(row.len());
        let mut x = 0.0;
        let mut previous_node = None;
        for &(node_id, cover_index) in row {
            if previous_node.is_some_and(|previous| previous != node_id) {
                x += 0.8;
            }
            raw_positions.push((node_id, cover_index, x));
            x += 2.2;
            previous_node = Some(node_id);
        }
        let center = raw_positions
            .first()
            .zip(raw_positions.last())
            .map_or(0.0, |(first, last)| (first.2 + last.2) / 2.0);
        for (node_id, cover_index, raw_x) in raw_positions {
            positions.insert((node_id, cover_index), raw_x - center);
        }
    }

    let mut output = String::new();
    writeln!(output, "% Input: {}", input_label.replace('\n', " ")).unwrap();
    output.push_str(
        r#"\providecommand{\vzInconsistent}{\ensuremath{\textcolor{red!75!black}{\boldsymbol{\times}}}}
\providecommand{\vzPruned}{\ensuremath{\textcolor{black}{\boldsymbol{\times}}}}
\providecommand{\vzConsistent}{\ensuremath{\textcolor{green!45!black}{\boldsymbol{\checkmark}}}}
\providecommand{\vzNoCovers}{\ensuremath{\textcolor{gray!75!black}{\boldsymbol{\varnothing}}}}
\providecommand{\vzTimeout}{\tikz[baseline=-0.55ex,scale=0.105]{\filldraw[fill=yellow!70,draw=orange!85!black,line width=1.2pt] (0,0) circle (1);\draw[orange!85!black,line width=1.4pt,line cap=round] (0,0) -- (0,0.58) (0,0) -- (0.48,-0.28);}}
\providecommand{\vzContracted}{\ensuremath{\textcolor{violet!80!black}{\boldsymbol{\leadsto}}}}

\begin{tikzpicture}[
  font=\small,
  cover/.style={draw=blue!65!black,fill=blue!3,rounded corners=2pt,line width=0.65pt,minimum width=18mm,minimum height=10mm,inner sep=2pt,align=center},
  timeout cover/.style={cover,draw=orange!85!black,fill=yellow!12},
  contracted cover/.style={cover,draw=violet!75!black,fill=violet!5},
  empty branch/.style={draw=gray!65,dashed,fill=gray!5,rounded corners=2pt,minimum width=18mm,minimum height=10mm,inner sep=2pt,align=center},
  input/.style={draw=blue!70!black,fill=blue!8,circle,line width=0.7pt,minimum size=8mm,align=center},
  branch/.style={-{Stealth[length=1.8mm]},line width=0.65pt},
  stem/.style={line width=0.65pt},
  edge label/.style={fill=white,inner sep=1pt,font=\scriptsize}
]
"#,
    );

    for node in nodes {
        let depth = node.decisions.len();
        let y = -1.45 - depth as f64 * 2.15;
        let members = plot_members(node);
        let center = members
            .iter()
            .filter_map(|key| positions.get(key))
            .sum::<f64>()
            / members.len() as f64;
        writeln!(
            output,
            "\\coordinate (group-{}) at ({center:.3},{:.3});",
            node.node_id,
            y + 0.72
        )
        .unwrap();

        for check in &node.checks {
            let x = positions[&(node.node_id, Some(check.cover_index))];
            let name = cover_node_name(node.node_id, check.cover_index);
            let style = match check.outcome {
                VariableZeroingCoverOutcome::Timeout => "timeout cover",
                VariableZeroingCoverOutcome::Contracted => "contracted cover",
                _ => "cover",
            };
            let outcome = match check.outcome {
                VariableZeroingCoverOutcome::Consistent => "\\vzConsistent",
                VariableZeroingCoverOutcome::Inconsistent => "\\vzInconsistent",
                VariableZeroingCoverOutcome::CorePruned => "\\vzPruned",
                VariableZeroingCoverOutcome::Timeout => "\\vzTimeout",
                VariableZeroingCoverOutcome::Contracted => "\\vzContracted",
            };
            let split = check
                .split_variable
                .as_ref()
                .map_or_else(String::new, |variable| {
                    format!(
                        "\\\\[-1pt]{{\\scriptsize split: \\texttt{{{}}}}}",
                        tex_escape(variable)
                    )
                });
            writeln!(
                output,
                "\\node[{style}] ({name}) at ({x:.3},{y:.3}) {{{outcome}{split}\\\\[-1pt]{{\\scriptsize hit \\#{}}}\\\\[-1pt]{{\\scriptsize {}}}}};",
                check.cover_index,
                format_check_duration(check.elapsed)
            )
            .unwrap();
        }
        if node.checks.len() < node.covers || node.covers == 0 {
            let x = positions[&(node.node_id, None)];
            if node.covers == 0 {
                writeln!(
                    output,
                    "\\node[empty branch] (pending-{}) at ({x:.3},{y:.3}) {{\\vzNoCovers\\\\[-1pt]{{\\scriptsize no covers}}}};",
                    node.node_id
                )
                .unwrap();
            } else {
                let unchecked = node.covers - node.checks.len();
                writeln!(
                    output,
                    "\\node[empty branch] (pending-{}) at ({x:.3},{y:.3}) {{$\\boldsymbol{{\\cdots}}$\\\\[-1pt]{{\\scriptsize {unchecked} unchecked}}}};",
                    node.node_id
                )
                .unwrap();
            }
        }
    }

    let root = nodes_by_id
        .get(&0)
        .ok_or_else(|| "recursion trace has no root node #0".to_string())?;
    let root_center = group_center(root, &positions);
    writeln!(
        output,
        "\\node[input] (input) at ({root_center:.3},0) {{$P$}};"
    )
    .unwrap();
    output.push_str("\\draw[stem] (input) -- (group-0);\n");
    draw_group_fan(&mut output, root);

    for node in nodes {
        for check in &node.checks {
            let parent_name = cover_node_name(node.node_id, check.cover_index);
            for (child_id, relation) in [(check.zero_child, "=0"), (check.nonzero_child, "\\ne 0")]
            {
                let Some(child_id) = child_id else { continue };
                let child = nodes_by_id.get(&child_id).ok_or_else(|| {
                    format!("cover {parent_name} refers to missing child node {child_id}")
                })?;
                writeln!(
                    output,
                    "\\draw[stem] ({parent_name}.south) -- node[edge label,sloped,above] {{$ {relation} $}} (group-{child_id});"
                )
                .unwrap();
                draw_group_fan(&mut output, child);
            }
        }
    }

    let max_depth = nodes
        .iter()
        .map(|node| node.decisions.len())
        .max()
        .unwrap_or(0);
    let legend_y = -2.25 - max_depth as f64 * 2.15;
    writeln!(
        output,
        "\\node[draw=gray!55,fill=white,rounded corners=2pt,inner sep=3pt,anchor=north] at (0,{legend_y:.3}) {{\\scriptsize \\vzInconsistent\\ inconsistent\\quad \\vzPruned\\ core-pruned\\quad \\vzConsistent\\ consistent\\quad \\vzTimeout\\ timed out\\quad \\vzContracted\\ contracted\\quad \\vzNoCovers\\ no covers}};"
    )
    .unwrap();
    output.push_str("\\end{tikzpicture}\n");
    Ok(output)
}

fn group_center(
    node: &VariableZeroingNodeResult,
    positions: &HashMap<(usize, Option<usize>), f64>,
) -> f64 {
    let members = plot_members(node);
    members.iter().map(|key| positions[key]).sum::<f64>() / members.len() as f64
}

fn draw_group_fan(output: &mut String, node: &VariableZeroingNodeResult) {
    for check in &node.checks {
        writeln!(
            output,
            "\\draw[branch] (group-{}) -- ({}.north);",
            node.node_id,
            cover_node_name(node.node_id, check.cover_index)
        )
        .unwrap();
    }
    if node.checks.len() < node.covers || node.covers == 0 {
        writeln!(
            output,
            "\\draw[branch] (group-{}) -- (pending-{}.north);",
            node.node_id, node.node_id
        )
        .unwrap();
    }
}

fn plot_members(node: &VariableZeroingNodeResult) -> Vec<(usize, Option<usize>)> {
    let mut members = node
        .checks
        .iter()
        .map(|check| (node.node_id, Some(check.cover_index)))
        .collect::<Vec<_>>();
    if node.checks.len() < node.covers || node.covers == 0 {
        members.push((node.node_id, None));
    }
    members
}

fn cover_node_name(node_id: usize, cover_index: usize) -> String {
    format!("cover-{node_id}-{cover_index}")
}

fn format_check_duration(duration: Duration) -> String {
    if duration.as_secs() != 0 {
        format!("{:.3} s", duration.as_secs_f64())
    } else if duration.as_millis() != 0 {
        format!("{:.3} ms", duration.as_secs_f64() * 1_000.0)
    } else {
        format!("{:.3} $\\mu$s", duration.as_secs_f64() * 1_000_000.0)
    }
}

fn tex_escape(value: &str) -> String {
    value
        .chars()
        .map(|character| match character {
            '\\' => "\\textbackslash{}".to_string(),
            '{' => "\\{".to_string(),
            '}' => "\\}".to_string(),
            '#' => "\\#".to_string(),
            '$' => "\\$".to_string(),
            '%' => "\\%".to_string(),
            '&' => "\\&".to_string(),
            '_' => "\\_".to_string(),
            '^' => "\\^{}".to_string(),
            '~' => "\\~{}".to_string(),
            other => other.to_string(),
        })
        .collect()
}

fn format_variable_list<'a>(variables: impl Iterator<Item = &'a str>) -> String {
    let variables = variables.collect::<Vec<_>>();
    if variables.is_empty() {
        "none".to_string()
    } else {
        variables.join(",")
    }
}

fn display_limit(value: usize) -> String {
    if value == usize::MAX {
        "unlimited".to_string()
    } else {
        value.to_string()
    }
}

fn format_decisions(decisions: &[polynomial_cdcl::SplitDecision]) -> String {
    if decisions.is_empty() {
        return "none".to_string();
    }
    decisions
        .iter()
        .map(|decision| {
            format!(
                "{} {} 0",
                decision.variable,
                if decision.is_zero { "=" } else { "!=" }
            )
        })
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_search_limits_and_candidate_variables() {
        let options = parse_args(
            [
                "input.txt",
                "--timeout",
                "75",
                "--max-runtime",
                "900000",
                "--max-nodes",
                "12",
                "--max-covers",
                "30",
                "--max-depth",
                "4",
                "--variable",
                "x",
                "--gb",
                "gvw",
                "--no-unsat-core-search",
                "--saturate-each-variable",
                "--plot",
                "tree.tex",
                "--summary",
            ]
            .into_iter()
            .map(String::from),
        )
        .expect("arguments");
        assert_eq!(options.path, PathBuf::from("input.txt"));
        assert_eq!(options.timeout_ms, 75);
        assert_eq!(options.max_runtime_ms, Some(900_000));
        assert_eq!(options.max_nodes, 12);
        assert_eq!(options.max_evaluations, 30);
        assert_eq!(options.max_depth, 4);
        assert_eq!(options.variable_names, vec!["x"]);
        assert_eq!(options.gb_backend, GbBackend::Gvw);
        assert!(!options.unsat_core_search);
        assert!(options.saturate_each_variable);
        assert_eq!(options.plot, Some(PathBuf::from("tree.tex")));
        assert!(options.summary);
    }

    #[test]
    fn parse_args_uses_fast_defaults() {
        let options = parse_args(std::iter::empty()).expect("arguments");

        assert_eq!(options.gb_backend, GbBackend::SingularGb);
        assert!(!options.unsat_core_search);
        assert!(!options.saturate_each_variable);
    }

    #[test]
    fn parse_args_can_enable_unsat_core_search() {
        let options =
            parse_args(["--unsat-core-search"].into_iter().map(String::from)).expect("arguments");

        assert!(options.unsat_core_search);
    }

    #[test]
    fn node_summary_lists_zero_and_nonzero_decisions() {
        let node = VariableZeroingNodeResult {
            node_id: 3,
            decisions: vec![
                polynomial_cdcl::SplitDecision {
                    variable: "x".to_string(),
                    is_zero: true,
                },
                polynomial_cdcl::SplitDecision {
                    variable: "y".to_string(),
                    is_zero: false,
                },
            ],
            covers: 4,
            evaluations: 3,
            consistent: 1,
            inconsistent: 1,
            core_pruned: 1,
            timeouts: 1,
            zeroing_splits: 1,
            elapsed: Duration::from_millis(125),
            checks: Vec::new(),
        };

        assert_eq!(
            format_node_summary(&node),
            "ZEROING_NODE node=3 zeroed=x nonzero=y covers=4 evaluations=3 consistent=1 inconsistent=1 core_pruned=1 timeouts=1 splits=1 elapsed_secs=0.125000"
        );
    }

    #[test]
    fn tikz_plot_contains_each_check_time_and_split_edges() {
        let nodes = vec![
            VariableZeroingNodeResult {
                node_id: 0,
                decisions: Vec::new(),
                covers: 1,
                evaluations: 1,
                consistent: 0,
                inconsistent: 0,
                core_pruned: 0,
                timeouts: 1,
                zeroing_splits: 1,
                elapsed: Duration::from_secs(5),
                checks: vec![polynomial_cdcl::VariableZeroingCoverResult {
                    cover_index: 0,
                    outcome: VariableZeroingCoverOutcome::Timeout,
                    elapsed: Duration::from_millis(5_012),
                    split_variable: Some("x_1".to_string()),
                    zero_child: Some(1),
                    nonzero_child: Some(2),
                }],
            },
            VariableZeroingNodeResult {
                node_id: 1,
                decisions: vec![polynomial_cdcl::SplitDecision {
                    variable: "x_1".to_string(),
                    is_zero: true,
                }],
                covers: 0,
                evaluations: 0,
                consistent: 0,
                inconsistent: 0,
                core_pruned: 0,
                timeouts: 0,
                zeroing_splits: 0,
                elapsed: Duration::ZERO,
                checks: Vec::new(),
            },
            VariableZeroingNodeResult {
                node_id: 2,
                decisions: vec![polynomial_cdcl::SplitDecision {
                    variable: "x_1".to_string(),
                    is_zero: false,
                }],
                covers: 1,
                evaluations: 1,
                consistent: 0,
                inconsistent: 1,
                core_pruned: 0,
                timeouts: 0,
                zeroing_splits: 0,
                elapsed: Duration::from_millis(20),
                checks: vec![polynomial_cdcl::VariableZeroingCoverResult {
                    cover_index: 0,
                    outcome: VariableZeroingCoverOutcome::Inconsistent,
                    elapsed: Duration::from_millis(20),
                    split_variable: None,
                    zero_child: None,
                    nonzero_child: None,
                }],
            },
        ];

        let plot = render_tikz_plot("system_test.txt", &nodes).expect("plot");
        assert!(plot.contains("5.012 s"));
        assert!(plot.contains("split: \\texttt{x\\_1}"));
        assert!(plot.contains("$ =0 $"));
        assert!(plot.contains("$ \\ne 0 $"));
        assert!(plot.contains("20.000 ms"));
        assert!(plot.contains("no covers"));
    }
}
