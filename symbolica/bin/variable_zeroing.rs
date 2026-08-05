//! CLI for recursive factor branching with variable zeroing.
//!
//! Usage: `variable_zeroing [path]` (default: `systems/system.txt`).

use std::path::PathBuf;
use std::time::{Duration, Instant};

use symbolica::LicenseManager;
use symbolica::domains::rational::Q;
use symbolica::poly::PolyVariable;

use polynomial_cdcl::gb_search::GbBackend;
use polynomial_cdcl::{
    VariableZeroingNodeResult, VariableZeroingOptions, ZeroingVariable,
    default_systems_dir_from_manifest, load_mathematica_system_from_path_with_var_names,
    polynomials_from_loaded, solve_by_zeroing,
};

#[derive(Debug)]
struct CliOptions {
    path: PathBuf,
    interreduce: bool,
    timeout_ms: u64,
    gb_backend: GbBackend,
    max_nodes: usize,
    max_evaluations: usize,
    max_depth: usize,
    variable_names: Vec<String>,
    summary: bool,
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
    let mut gb_backend = GbBackend::M4gb;
    let mut max_nodes = usize::MAX;
    let mut max_evaluations = usize::MAX;
    let mut max_depth = usize::MAX;
    let mut variable_names = Vec::new();
    let mut summary = false;
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--no-interreduce" => interreduce = false,
            "--timeout" => {
                timeout_ms = parse_usize(&mut args, "--timeout")? as u64;
            }
            "--gb" => {
                let value = args.next().ok_or_else(|| {
                    "--gb requires m4gb, f5, gvw, symbolica, or singular-gb".to_string()
                })?;
                gb_backend = GbBackend::parse(&value).ok_or_else(|| {
                    format!(
                        "invalid --gb {value:?}: expected m4gb, f5, gvw, symbolica, or singular-gb"
                    )
                })?;
            }
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
            _ if arg.starts_with("--") => return Err(format!("unknown option {arg:?}")),
            _ if path.is_some() => return Err(format!("unexpected positional argument {arg:?}")),
            _ => path = Some(PathBuf::from(arg)),
        }
    }
    Ok(CliOptions {
        path: path.unwrap_or_else(|| default_systems_dir_from_manifest().join("system.txt")),
        interreduce,
        timeout_ms,
        gb_backend,
        max_nodes,
        max_evaluations,
        max_depth,
        variable_names,
        summary,
    })
}

fn die(message: impl std::fmt::Display, code: i32) -> ! {
    eprintln!("{message}");
    std::process::exit(code)
}

fn main() {
    LicenseManager::set_license_key("5568d092#6bb8b4fb#ff936916-65ba-552f-b3c1-bdc64597cd21")
        .unwrap();
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
        "Recursive branching: {}ms/cover, {} backend, max nodes {}, max evaluations {}, max depth {}",
        cli.timeout_ms,
        cli.gb_backend.label(),
        display_limit(cli.max_nodes),
        display_limit(cli.max_evaluations),
        display_limit(cli.max_depth)
    );

    let options = VariableZeroingOptions {
        interreduce: cli.interreduce,
        timeout: Duration::from_millis(cli.timeout_ms),
        gb_backend: cli.gb_backend,
        max_nodes: cli.max_nodes,
        max_evaluations: cli.max_evaluations,
        max_depth: cli.max_depth,
        verbose: !cli.summary,
    };
    let result = solve_by_zeroing(polynomials, variables, &options)
        .unwrap_or_else(|error| die(format!("variable-zeroing error: {error}"), 1));

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
        "{} zeroed={} nonzero={} covers={} consistent={} inconsistent={} core_pruned={} timeouts={} elapsed_secs={:.6}",
        node.node_id,
        zeroed,
        nonzero,
        node.covers,
        node.consistent,
        node.inconsistent,
        node.core_pruned,
        node.timeouts,
        node.elapsed.as_secs_f64()
    )
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
                "--summary",
            ]
            .into_iter()
            .map(String::from),
        )
        .expect("arguments");
        assert_eq!(options.path, PathBuf::from("input.txt"));
        assert_eq!(options.timeout_ms, 75);
        assert_eq!(options.max_nodes, 12);
        assert_eq!(options.max_evaluations, 30);
        assert_eq!(options.max_depth, 4);
        assert_eq!(options.variable_names, vec!["x"]);
        assert_eq!(options.gb_backend, GbBackend::Gvw);
        assert!(options.summary);
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
        };

        assert_eq!(
            format_node_summary(&node),
            "ZEROING_NODE node=3 zeroed=x nonzero=y covers=4 evaluations=3 consistent=1 inconsistent=1 core_pruned=1 timeouts=1 splits=1 elapsed_secs=0.125000"
        );
    }
}
