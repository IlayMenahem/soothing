//! Benchmark MILP-cover against a panel of symbolic and numerical baselines
//! on a directory of Mathematica-style polynomial systems.
//!
//! Baselines (only those whose external tools are present on PATH are run):
//!   - symbolica-gb-lex     : Symbolica's Buchberger over ℚ with lex order
//!   - symbolica-f5-lex     : Symbolica's F5 signature algorithm
//!   - symbolica-gvw-lex    : Symbolica's GVW signature algorithm
//!   - singular-gb-lex      : Singular's `std(I)`
//!   - singular-primdecGTZ  : Singular's `primdec.lib` primary decomposition
//!   - m2-gb-lex            : Macaulay2's `gb(I)`
//!   - m2-decompose         : Macaulay2's `decompose I`
//!   - homotopy-jl          : HomotopyContinuation.jl numerical irreducible decomposition
//!   - milp-cover           : our method
//!
//! Each (system, baseline) pair runs in its own subprocess so a runaway
//! computation can be SIGKILL'd on timeout. Workers emit a single JSON line
//! in the unified `BaselineResult` shape.
//!
//! Outputs (under `<repo>/results/` by default):
//!   - bench_primary.csv    : one row per (system, baseline) — long format
//!   - bench_dimension.csv  : one row per (system, source) for the dimension probe
//!   - bench_summary.md     : pivot table for the paper

use std::collections::BTreeMap;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

use symbolica::LicenseManager;
use symbolica::domains::algebraic_number::AlgebraicExtension;
use symbolica::domains::rational::Q;
use symbolica::poly::LexOrder;

use polynomial_milp::milp_cover::{build_problem, solve_milp};
use polynomial_milp::{
    groebner_basis_from_system_lex_q, groebner_f5_from_system_with,
    groebner_gvw_from_system_with, load_mathematica_system_from_path_with_var_names, preprocess,
    variable_order,
};

const LICENSE_KEY: &str = "5568d092#6bb8b4fb#ff936916-65ba-552f-b3c1-bdc64597cd21";

// =========================================================================
// CLI / Mode
// =========================================================================

#[derive(Debug)]
enum Mode {
    Orchestrator {
        dir: PathBuf,
        timeout: Duration,
        no_cache: bool,
        csv: Option<PathBuf>,
        only_baselines: Option<Vec<String>>,
    },
    WorkerSymbolicaGb { path: PathBuf },
    WorkerSymbolicaF5 { path: PathBuf },
    WorkerSymbolicaGvw { path: PathBuf },
    WorkerMilpCover { path: PathBuf },
}

fn parse_args(args: impl IntoIterator<Item = String>) -> Result<Mode, String> {
    let mut dir: Option<PathBuf> = None;
    let mut timeout_secs: u64 = 120;
    let mut no_cache = false;
    let mut csv: Option<PathBuf> = None;
    let mut only_baselines: Option<Vec<String>> = None;
    let mut worker: Option<Mode> = None;

    let mut it = args.into_iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--timeout" => {
                timeout_secs = it
                    .next()
                    .ok_or("--timeout requires a value")?
                    .parse()
                    .map_err(|e| format!("--timeout: {e}"))?;
            }
            "--no-cache" => no_cache = true,
            "--csv" => csv = Some(PathBuf::from(it.next().ok_or("--csv requires a path")?)),
            "--only" => {
                let csv_arg = it.next().ok_or("--only requires a comma-separated list")?;
                only_baselines = Some(csv_arg.split(',').map(|s| s.trim().to_string()).collect());
            }
            "--worker-symbolica-gb" => {
                worker = Some(Mode::WorkerSymbolicaGb {
                    path: PathBuf::from(it.next().ok_or("--worker-symbolica-gb needs path")?),
                });
            }
            "--worker-symbolica-f5" => {
                worker = Some(Mode::WorkerSymbolicaF5 {
                    path: PathBuf::from(it.next().ok_or("--worker-symbolica-f5 needs path")?),
                });
            }
            "--worker-symbolica-gvw" => {
                worker = Some(Mode::WorkerSymbolicaGvw {
                    path: PathBuf::from(it.next().ok_or("--worker-symbolica-gvw needs path")?),
                });
            }
            "--worker-milp-cover" => {
                worker = Some(Mode::WorkerMilpCover {
                    path: PathBuf::from(it.next().ok_or("--worker-milp-cover needs path")?),
                });
            }
            s if s.starts_with("--") => return Err(format!("unknown option {s:?}")),
            _ => dir = Some(PathBuf::from(arg)),
        }
    }

    if let Some(m) = worker {
        return Ok(m);
    }
    Ok(Mode::Orchestrator {
        dir: dir.unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../systems/bench")),
        timeout: Duration::from_secs(timeout_secs),
        no_cache,
        csv,
        only_baselines,
    })
}

// =========================================================================
// Unified worker result shape
// =========================================================================

#[derive(Serialize, Deserialize, Default, Clone, Debug)]
struct BaselineResult {
    baseline: String,
    status: String, // "ok" | "timeout" | "tool_missing" | "error" | "parse_error" | "infeasible" | "empty"
    #[serde(default)]
    elapsed_secs: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    basis_len: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    num_components: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    max_dim: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    num_solutions: Option<usize>,
    // MILP-specific
    #[serde(default, skip_serializing_if = "Option::is_none")]
    preprocess_secs: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    coverage: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    polys: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    free_params: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    selected: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    zeroed: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

impl BaselineResult {
    fn emit(self) -> ! {
        println!("{}", serde_json::to_string(&self).unwrap());
        std::process::exit(0);
    }
}

// =========================================================================
// In-process workers (Symbolica)
// =========================================================================

fn worker_symbolica_gb(path: &Path) -> BaselineResult {
    let label = "symbolica-gb-lex";
    let loaded = match load_mathematica_system_from_path_with_var_names(path) {
        Ok(s) => s,
        Err(e) => {
            return BaselineResult {
                baseline: label.into(),
                status: "parse_error".into(),
                error: Some(format!("{e}")),
                ..Default::default()
            };
        }
    };
    let var_order = variable_order(&loaded.id_to_sym);
    let t = Instant::now();
    match groebner_basis_from_system_lex_q(&loaded.system, var_order, false) {
        Ok(gb) => BaselineResult {
            baseline: label.into(),
            status: "ok".into(),
            elapsed_secs: t.elapsed().as_secs_f64(),
            basis_len: Some(gb.system.len()),
            ..Default::default()
        },
        Err(e) => BaselineResult {
            baseline: label.into(),
            status: "error".into(),
            elapsed_secs: t.elapsed().as_secs_f64(),
            error: Some(format!("{e}")),
            ..Default::default()
        },
    }
}

fn worker_symbolica_f5(path: &Path) -> BaselineResult {
    let label = "symbolica-f5-lex";
    let loaded = match load_mathematica_system_from_path_with_var_names(path) {
        Ok(s) => s,
        Err(e) => {
            return BaselineResult {
                baseline: label.into(),
                status: "parse_error".into(),
                error: Some(format!("{e}")),
                ..Default::default()
            };
        }
    };
    let var_order: Arc<_> = variable_order(&loaded.id_to_sym);
    let t = Instant::now();
    match groebner_f5_from_system_with::<Q, LexOrder>(&loaded.system, &Q, var_order) {
        Ok(gb) => BaselineResult {
            baseline: label.into(),
            status: "ok".into(),
            elapsed_secs: t.elapsed().as_secs_f64(),
            basis_len: Some(gb.system.len()),
            ..Default::default()
        },
        Err(e) => BaselineResult {
            baseline: label.into(),
            status: "error".into(),
            elapsed_secs: t.elapsed().as_secs_f64(),
            error: Some(format!("{e}")),
            ..Default::default()
        },
    }
}

fn worker_symbolica_gvw(path: &Path) -> BaselineResult {
    let label = "symbolica-gvw-lex";
    let loaded = match load_mathematica_system_from_path_with_var_names(path) {
        Ok(s) => s,
        Err(e) => {
            return BaselineResult {
                baseline: label.into(),
                status: "parse_error".into(),
                error: Some(format!("{e}")),
                ..Default::default()
            };
        }
    };
    let var_order: Arc<_> = variable_order(&loaded.id_to_sym);
    let t = Instant::now();
    match groebner_gvw_from_system_with::<Q, LexOrder>(&loaded.system, &Q, var_order) {
        Ok(gb) => BaselineResult {
            baseline: label.into(),
            status: "ok".into(),
            elapsed_secs: t.elapsed().as_secs_f64(),
            basis_len: Some(gb.system.len()),
            ..Default::default()
        },
        Err(e) => BaselineResult {
            baseline: label.into(),
            status: "error".into(),
            elapsed_secs: t.elapsed().as_secs_f64(),
            error: Some(format!("{e}")),
            ..Default::default()
        },
    }
}

fn worker_milp_cover(path: &Path) -> BaselineResult {
    let label = "milp-cover";
    let loaded = match load_mathematica_system_from_path_with_var_names(path) {
        Ok(s) => s,
        Err(e) => {
            return BaselineResult {
                baseline: label.into(),
                status: "parse_error".into(),
                error: Some(format!("{e}")),
                ..Default::default()
            };
        }
    };
    let field = AlgebraicExtension::new_complex(Q);
    let t_pre = Instant::now();
    let pre = match preprocess(&loaded, &field) {
        Ok(p) => p,
        Err(e) => {
            return BaselineResult {
                baseline: label.into(),
                status: "error".into(),
                preprocess_secs: Some(t_pre.elapsed().as_secs_f64()),
                error: Some(format!("preprocess: {e}")),
                ..Default::default()
            };
        }
    };
    let pre_secs = t_pre.elapsed().as_secs_f64();
    let polys = pre.len();
    if polys == 0 {
        return BaselineResult {
            baseline: label.into(),
            status: "empty".into(),
            preprocess_secs: Some(pre_secs),
            polys: Some(0),
            ..Default::default()
        };
    }
    let data = build_problem(pre, &loaded);
    let n_vars_in_factors = data.v_ids.len();
    let t_solve = Instant::now();
    match solve_milp(&data) {
        Ok(cover) => {
            let selected = cover.selected.len();
            let zeroed = cover.zeroed_locals.len();
            BaselineResult {
                baseline: label.into(),
                status: "ok".into(),
                elapsed_secs: t_solve.elapsed().as_secs_f64(),
                preprocess_secs: Some(pre_secs),
                polys: Some(polys),
                coverage: Some(cover.zeroed_polynomials.len()),
                free_params: Some(n_vars_in_factors.saturating_sub(selected + zeroed)),
                selected: Some(selected),
                zeroed: Some(zeroed),
                ..Default::default()
            }
        }
        Err(good_lp::ResolutionError::Infeasible) => BaselineResult {
            baseline: label.into(),
            status: "infeasible".into(),
            elapsed_secs: t_solve.elapsed().as_secs_f64(),
            preprocess_secs: Some(pre_secs),
            polys: Some(polys),
            ..Default::default()
        },
        Err(e) => BaselineResult {
            baseline: label.into(),
            status: "error".into(),
            elapsed_secs: t_solve.elapsed().as_secs_f64(),
            preprocess_secs: Some(pre_secs),
            polys: Some(polys),
            error: Some(format!("{e:?}")),
            ..Default::default()
        },
    }
}

// =========================================================================
// Subprocess infrastructure
// =========================================================================

enum SubprocessOutcome {
    Stdout(String),
    Timeout,
    Crashed { stderr: String },
}

fn run_subprocess(cmd: Command, timeout: Duration) -> SubprocessOutcome {
    let mut cmd = cmd;
    let mut child = match cmd.stdout(Stdio::piped()).stderr(Stdio::piped()).spawn() {
        Ok(c) => c,
        Err(e) => {
            return SubprocessOutcome::Crashed {
                stderr: format!("spawn: {e}"),
            };
        }
    };
    let id = child.id();
    let (tx, rx) = mpsc::channel();
    let watchdog = thread::spawn(move || {
        if rx.recv_timeout(timeout).is_err() {
            let _ = Command::new("kill").arg("-9").arg(id.to_string()).status();
        }
    });

    let wait_result = child.wait();
    let _ = tx.send(());
    let _ = watchdog.join();

    let mut stdout = String::new();
    if let Some(mut s) = child.stdout.take() {
        let _ = s.read_to_string(&mut stdout);
    }
    let mut stderr = String::new();
    if let Some(mut s) = child.stderr.take() {
        let _ = s.read_to_string(&mut stderr);
    }
    match wait_result {
        Ok(status) if status.success() => SubprocessOutcome::Stdout(stdout),
        Ok(status) if status.code().is_none() => SubprocessOutcome::Timeout,
        Ok(_) => SubprocessOutcome::Crashed { stderr },
        Err(e) => SubprocessOutcome::Crashed {
            stderr: format!("wait: {e}"),
        },
    }
}

fn parse_last_json_line(stdout: &str, baseline: &str) -> BaselineResult {
    let line = stdout
        .lines()
        .rev()
        .find(|l| l.trim_start().starts_with('{'));
    match line {
        Some(l) => match serde_json::from_str::<BaselineResult>(l) {
            Ok(r) => r,
            Err(e) => BaselineResult {
                baseline: baseline.into(),
                status: "error".into(),
                error: Some(format!("json parse: {e}")),
                ..Default::default()
            },
        },
        None => BaselineResult {
            baseline: baseline.into(),
            status: "error".into(),
            error: Some("no JSON line on stdout".into()),
            ..Default::default()
        },
    }
}

// =========================================================================
// BaselineSpec — how to launch each baseline
// =========================================================================

enum Invocation {
    /// Re-exec this binary with `--worker-X`.
    SelfWorker { flag: &'static str },
    /// Run a Python baseline driver under `scripts/baselines/`.
    PythonScript {
        script: &'static str,
        task: Option<&'static str>,
        /// Optional external tool name to probe with `which`; baseline is
        /// skipped if missing.
        required_tool: Option<&'static str>,
    },
}

struct BaselineSpec {
    label: &'static str,
    invocation: Invocation,
    /// Per-baseline timeout override (else the global timeout).
    timeout_override: Option<Duration>,
}

fn all_baselines() -> Vec<BaselineSpec> {
    vec![
        BaselineSpec {
            label: "symbolica-gb-lex",
            invocation: Invocation::SelfWorker {
                flag: "--worker-symbolica-gb",
            },
            timeout_override: None,
        },
        BaselineSpec {
            label: "symbolica-f5-lex",
            invocation: Invocation::SelfWorker {
                flag: "--worker-symbolica-f5",
            },
            timeout_override: None,
        },
        BaselineSpec {
            label: "symbolica-gvw-lex",
            invocation: Invocation::SelfWorker {
                flag: "--worker-symbolica-gvw",
            },
            timeout_override: None,
        },
        BaselineSpec {
            label: "singular-gb-lex",
            invocation: Invocation::PythonScript {
                script: "system_to_singular.py",
                task: Some("gb"),
                required_tool: Some("Singular"),
            },
            timeout_override: None,
        },
        BaselineSpec {
            label: "singular-primdecGTZ",
            invocation: Invocation::PythonScript {
                script: "system_to_singular.py",
                task: Some("decompose"),
                required_tool: Some("Singular"),
            },
            timeout_override: None,
        },
        BaselineSpec {
            label: "m2-gb-lex",
            invocation: Invocation::PythonScript {
                script: "system_to_macaulay2.py",
                task: Some("gb"),
                required_tool: Some("M2"),
            },
            timeout_override: None,
        },
        BaselineSpec {
            label: "m2-decompose",
            invocation: Invocation::PythonScript {
                script: "system_to_macaulay2.py",
                task: Some("decompose"),
                required_tool: Some("M2"),
            },
            timeout_override: None,
        },
        BaselineSpec {
            label: "homotopy-jl",
            invocation: Invocation::PythonScript {
                script: "system_to_homotopy_jl.py",
                task: None,
                required_tool: Some("julia"),
            },
            // Julia startup alone is ~2s; give it more wall budget.
            timeout_override: Some(Duration::from_secs(600)),
        },
        BaselineSpec {
            label: "milp-cover",
            invocation: Invocation::SelfWorker {
                flag: "--worker-milp-cover",
            },
            timeout_override: None,
        },
    ]
}

fn tool_available(name: &str) -> bool {
    Command::new("which")
        .arg(name)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn scripts_baselines_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap_or(Path::new("."))
        .join("scripts/baselines")
}

fn results_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap_or(Path::new("."))
        .join("results")
}

// =========================================================================
// Cache (only for in-process Symbolica baselines, keyed by hash+label)
// =========================================================================

fn cache_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../target/bench-cache")
}

fn cache_key(path: &Path, baseline: &str) -> Option<String> {
    let content = fs::read(path).ok()?;
    Some(format!(
        "{}-{}",
        baseline,
        blake3::hash(&content).to_hex()
    ))
}

fn cache_load(key: &str) -> Option<BaselineResult> {
    let path = cache_dir().join(format!("{key}.json"));
    serde_json::from_slice(&fs::read(&path).ok()?).ok()
}

fn cache_store(key: &str, value: &BaselineResult) {
    let dir = cache_dir();
    if fs::create_dir_all(&dir).is_err() {
        return;
    }
    if let Ok(bytes) = serde_json::to_vec(value) {
        let _ = fs::write(dir.join(format!("{key}.json")), bytes);
    }
}

// =========================================================================
// Orchestrator: run one (system, baseline) pair
// =========================================================================

fn invoke_baseline(
    self_exe: &Path,
    spec: &BaselineSpec,
    system_path: &Path,
    base_timeout: Duration,
) -> BaselineResult {
    let timeout = spec.timeout_override.unwrap_or(base_timeout);
    let wall = timeout + Duration::from_secs(5);
    let path_str = system_path.to_string_lossy().to_string();

    let cmd = match &spec.invocation {
        Invocation::SelfWorker { flag } => {
            let mut c = Command::new(self_exe);
            c.arg(flag).arg(&path_str);
            c
        }
        Invocation::PythonScript {
            script,
            task,
            required_tool,
        } => {
            if let Some(tool) = required_tool {
                if !tool_available(tool) {
                    return BaselineResult {
                        baseline: spec.label.into(),
                        status: "tool_missing".into(),
                        error: Some(format!("{tool} not on PATH")),
                        ..Default::default()
                    };
                }
            }
            let mut c = Command::new("python3");
            c.arg(scripts_baselines_dir().join(script))
                .arg(&path_str)
                .arg("--timeout-secs")
                .arg(timeout.as_secs().to_string());
            if let Some(t) = task {
                c.arg("--task").arg(*t);
            }
            c
        }
    };

    match run_subprocess(cmd, wall) {
        SubprocessOutcome::Stdout(s) => parse_last_json_line(&s, spec.label),
        SubprocessOutcome::Timeout => BaselineResult {
            baseline: spec.label.into(),
            status: "timeout".into(),
            elapsed_secs: timeout.as_secs_f64(),
            ..Default::default()
        },
        SubprocessOutcome::Crashed { stderr } => BaselineResult {
            baseline: spec.label.into(),
            status: "error".into(),
            error: Some(format!(
                "subprocess crashed; stderr tail: {}",
                stderr.chars().rev().take(400).collect::<String>()
            )),
            ..Default::default()
        },
    }
}

// =========================================================================
// Dimension probe (Z3-based reference)
// =========================================================================

fn invoke_dimension_probe(system_path: &Path, base_timeout: Duration) -> Option<DimensionEntry> {
    if !tool_available("python3") {
        return None;
    }
    let script = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap_or(Path::new("."))
        .join("scripts/dimension_via_jacobian.py");
    if !script.exists() {
        return None;
    }
    let timeout = base_timeout;
    let mut cmd = Command::new("python3");
    cmd.arg(&script)
        .arg(system_path)
        .arg("--output-json")
        .arg("--timeout-ms")
        .arg(timeout.as_millis().to_string());
    let wall = timeout + Duration::from_secs(10);
    match run_subprocess(cmd, wall) {
        SubprocessOutcome::Stdout(s) => {
            let line = s.lines().rev().find(|l| l.trim_start().starts_with('{'))?;
            let v: JsonValue = serde_json::from_str(line).ok()?;
            Some(DimensionEntry {
                source: v.get("source").and_then(|x| x.as_str()).unwrap_or("z3").to_string(),
                status: v.get("status").and_then(|x| x.as_str()).unwrap_or("error").to_string(),
                found_dim: v.get("found_dim").and_then(|x| x.as_i64()),
                rank: v.get("rank").and_then(|x| x.as_i64()),
                elapsed_secs: v
                    .get("elapsed_secs")
                    .and_then(|x| x.as_f64())
                    .unwrap_or(0.0),
            })
        }
        SubprocessOutcome::Timeout => Some(DimensionEntry {
            source: "z3".into(),
            status: "timeout".into(),
            found_dim: None,
            rank: None,
            elapsed_secs: timeout.as_secs_f64(),
        }),
        SubprocessOutcome::Crashed { .. } => None,
    }
}

#[derive(Clone, Debug)]
struct DimensionEntry {
    source: String,
    status: String,
    found_dim: Option<i64>,
    rank: Option<i64>,
    elapsed_secs: f64,
}

// =========================================================================
// Per-system processing
// =========================================================================

struct SystemRecord {
    name: String,
    n_eqs: usize,
    n_vars: usize,
    results: Vec<BaselineResult>,
    dimension_entry: Option<DimensionEntry>,
}

fn process_file(
    self_exe: &Path,
    path: &Path,
    base_timeout: Duration,
    specs: &[BaselineSpec],
    no_cache: bool,
) -> Result<SystemRecord, String> {
    let name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("?")
        .to_string();

    let loaded = load_mathematica_system_from_path_with_var_names(path)
        .map_err(|e| format!("load: {e}"))?;
    let n_eqs = loaded.system.equations.len();
    let n_vars = loaded.var_names.len();
    drop(loaded);

    let mut results: Vec<BaselineResult> = Vec::with_capacity(specs.len());
    for spec in specs {
        eprintln!("    [{}] {}", spec.label, path.file_name().unwrap().to_string_lossy());
        // Cache only deterministic in-process Symbolica baselines.
        let cacheable = matches!(spec.invocation, Invocation::SelfWorker { .. });
        let cached = if cacheable && !no_cache {
            cache_key(path, spec.label).and_then(|k| cache_load(&k))
        } else {
            None
        };
        let res = if let Some(r) = cached {
            r
        } else {
            let r = invoke_baseline(self_exe, spec, path, base_timeout);
            if cacheable {
                if let Some(k) = cache_key(path, spec.label) {
                    cache_store(&k, &r);
                }
            }
            r
        };
        eprintln!(
            "       -> {} ({:.3}s)",
            res.status,
            res.elapsed_secs,
        );
        results.push(res);
    }

    let dimension_entry = invoke_dimension_probe(path, base_timeout);

    Ok(SystemRecord {
        name,
        n_eqs,
        n_vars,
        results,
        dimension_entry,
    })
}

// =========================================================================
// CSV / Markdown output
// =========================================================================

fn collect_inputs(dir: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = match fs::read_dir(dir) {
        Ok(it) => it
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("txt"))
            .collect(),
        Err(_) => Vec::new(),
    };
    files.sort();
    files
}

fn opt<T: std::fmt::Display>(v: &Option<T>) -> String {
    v.as_ref().map(|x| x.to_string()).unwrap_or_default()
}

fn write_primary_csv(path: &Path, records: &[SystemRecord]) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut body = String::from(
        "system,n_eqs,n_vars,baseline,status,elapsed_secs,basis_len,num_components,max_dim,num_solutions,polys,coverage,free_params,selected,zeroed,error\n",
    );
    for rec in records {
        for r in &rec.results {
            let err = r.error.as_deref().unwrap_or("").replace(',', ";");
            body.push_str(&format!(
                "{},{},{},{},{},{:.6},{},{},{},{},{},{},{},{},{},{}\n",
                rec.name,
                rec.n_eqs,
                rec.n_vars,
                r.baseline,
                r.status,
                r.elapsed_secs,
                opt(&r.basis_len),
                opt(&r.num_components),
                opt(&r.max_dim),
                opt(&r.num_solutions),
                opt(&r.polys),
                opt(&r.coverage),
                opt(&r.free_params),
                opt(&r.selected),
                opt(&r.zeroed),
                err,
            ));
        }
    }
    fs::write(path, body)
}

fn write_dimension_csv(path: &Path, records: &[SystemRecord]) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut body = String::from(
        "system,n_vars,reference_source,reference_status,reference_dim,reference_rank,reference_secs,method,found_dim,codim_shortfall\n",
    );
    for rec in records {
        let (ref_source, ref_status, ref_dim, ref_rank, ref_secs) =
            match &rec.dimension_entry {
                Some(d) => (
                    d.source.clone(),
                    d.status.clone(),
                    d.found_dim,
                    d.rank,
                    d.elapsed_secs,
                ),
                None => ("none".into(), "skipped".into(), None, None, 0.0),
            };
        // For each baseline that *reports* a dimension, emit a row.
        for r in &rec.results {
            if r.status != "ok" {
                continue;
            }
            let found = match r.baseline.as_str() {
                // Methods that produce a parametrization or witness — free_params
                // is our method's claim; max_dim is the decomposition baselines' claim.
                "milp-cover" => r.free_params.map(|v| v as i64),
                "singular-primdecGTZ" | "m2-decompose" | "homotopy-jl" => r.max_dim,
                _ => continue,
            };
            let Some(found_dim) = found else { continue };
            let shortfall = ref_dim.map(|rd| rd - found_dim);
            body.push_str(&format!(
                "{},{},{},{},{},{},{:.6},{},{},{}\n",
                rec.name,
                rec.n_vars,
                ref_source,
                ref_status,
                opt(&ref_dim),
                opt(&ref_rank),
                ref_secs,
                r.baseline,
                found_dim,
                opt(&shortfall),
            ));
        }
    }
    fs::write(path, body)
}

fn write_markdown_summary(path: &Path, records: &[SystemRecord]) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    // Build a pivot: rows=systems, cols=baselines, cell="status (time)".
    let mut all_baselines: BTreeMap<String, ()> = BTreeMap::new();
    for rec in records {
        for r in &rec.results {
            all_baselines.insert(r.baseline.clone(), ());
        }
    }
    let baselines: Vec<String> = all_baselines.into_keys().collect();

    let mut out = String::new();
    out.push_str("# Benchmark summary\n\n");
    out.push_str("## Primary: per-baseline status and time (seconds)\n\n");
    out.push_str("| system | n_eqs | n_vars |");
    for b in &baselines {
        out.push_str(" ");
        out.push_str(b);
        out.push_str(" |");
    }
    out.push('\n');
    out.push_str("|---|---:|---:|");
    for _ in &baselines {
        out.push_str("---|");
    }
    out.push('\n');
    for rec in records {
        out.push_str(&format!("| {} | {} | {} |", rec.name, rec.n_eqs, rec.n_vars));
        for b in &baselines {
            let cell = rec
                .results
                .iter()
                .find(|r| &r.baseline == b)
                .map(|r| match r.status.as_str() {
                    "ok" => format!("ok ({:.2}s)", r.elapsed_secs),
                    "tool_missing" => "skip".to_string(),
                    other => other.to_string(),
                })
                .unwrap_or_else(|| "-".to_string());
            out.push_str(&format!(" {} |", cell));
        }
        out.push('\n');
    }

    out.push_str("\n## Secondary: dimension findings\n\n");
    out.push_str("| system | ref_dim (Z3+Jac) | method | found_dim | shortfall |\n");
    out.push_str("|---|---:|---|---:|---:|\n");
    for rec in records {
        let ref_dim_str = rec
            .dimension_entry
            .as_ref()
            .and_then(|d| d.found_dim)
            .map(|d| d.to_string())
            .unwrap_or_else(|| "-".into());
        for r in &rec.results {
            if r.status != "ok" {
                continue;
            }
            let found = match r.baseline.as_str() {
                "milp-cover" => r.free_params.map(|v| v as i64),
                "singular-primdecGTZ" | "m2-decompose" | "homotopy-jl" => r.max_dim,
                _ => continue,
            };
            let Some(found_dim) = found else { continue };
            let short = rec
                .dimension_entry
                .as_ref()
                .and_then(|d| d.found_dim)
                .map(|rd| (rd - found_dim).to_string())
                .unwrap_or_else(|| "-".into());
            out.push_str(&format!(
                "| {} | {} | {} | {} | {} |\n",
                rec.name, ref_dim_str, r.baseline, found_dim, short
            ));
        }
    }

    fs::write(path, out)
}

// =========================================================================
// main
// =========================================================================

fn run_worker(label: &str, f: impl FnOnce() -> BaselineResult) -> ! {
    LicenseManager::set_license_key(LICENSE_KEY).ok();
    let r = f();
    // The label is supplied by the worker itself in `r.baseline`; sanity-check it.
    debug_assert_eq!(r.baseline, label);
    r.emit()
}

fn main() {
    let mode = match parse_args(std::env::args().skip(1)) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    };

    match mode {
        Mode::WorkerSymbolicaGb { path } => run_worker("symbolica-gb-lex", || worker_symbolica_gb(&path)),
        Mode::WorkerSymbolicaF5 { path } => run_worker("symbolica-f5-lex", || worker_symbolica_f5(&path)),
        Mode::WorkerSymbolicaGvw { path } => run_worker("symbolica-gvw-lex", || worker_symbolica_gvw(&path)),
        Mode::WorkerMilpCover { path } => run_worker("milp-cover", || worker_milp_cover(&path)),
        Mode::Orchestrator {
            dir,
            timeout,
            no_cache,
            csv,
            only_baselines,
        } => {
            LicenseManager::set_license_key(LICENSE_KEY).ok();
            let self_exe = std::env::current_exe().expect("current_exe");
            eprintln!(
                "bench: self={} dir={} timeout={:?}",
                self_exe.display(),
                dir.display(),
                timeout
            );

            let mut specs = all_baselines();
            if let Some(only) = &only_baselines {
                specs.retain(|s| only.iter().any(|n| n == s.label));
            }
            eprintln!("baselines:");
            for s in &specs {
                eprintln!("  - {}", s.label);
            }

            let files = collect_inputs(&dir);
            if files.is_empty() {
                eprintln!("no *.txt files in {}", dir.display());
                std::process::exit(1);
            }

            let mut records = Vec::with_capacity(files.len());
            for path in &files {
                eprintln!("  > {}", path.file_name().unwrap().to_string_lossy());
                match process_file(&self_exe, path, timeout, &specs, no_cache) {
                    Ok(rec) => records.push(rec),
                    Err(e) => eprintln!("    SKIP: {e}"),
                }
            }

            let primary = csv.clone().unwrap_or_else(|| results_dir().join("bench_primary.csv"));
            let dimension = results_dir().join("bench_dimension.csv");
            let markdown = results_dir().join("bench_summary.md");

            match write_primary_csv(&primary, &records) {
                Ok(()) => eprintln!("\nWrote {}", primary.display()),
                Err(e) => eprintln!("primary CSV write failed: {e}"),
            }
            match write_dimension_csv(&dimension, &records) {
                Ok(()) => eprintln!("Wrote {}", dimension.display()),
                Err(e) => eprintln!("dimension CSV write failed: {e}"),
            }
            match write_markdown_summary(&markdown, &records) {
                Ok(()) => eprintln!("Wrote {}", markdown.display()),
                Err(e) => eprintln!("markdown write failed: {e}"),
            }
        }
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_args_uses_default_dir() {
        let dir = match parse_args(std::iter::empty::<String>()).expect("parse args") {
            Mode::Orchestrator { dir, .. } => dir,
            _ => panic!("expected orchestrator"),
        };
        assert!(dir.ends_with("systems/bench"));
    }

    #[test]
    fn parse_args_accepts_a_path() {
        let dir = match parse_args(["systems/my_bench".to_string()]).expect("parse args") {
            Mode::Orchestrator { dir, .. } => dir,
            _ => panic!("expected orchestrator"),
        };
        assert_eq!(dir, PathBuf::from("systems/my_bench"));
    }

    #[test]
    fn parse_args_rejects_unknown_flag() {
        let err = parse_args(["--objective".to_string()]).expect_err("error");
        assert!(err.contains("unknown option"));
    }

    #[test]
    fn parse_args_only_filters_baselines() {
        let mode = parse_args(
            ["--only".to_string(), "symbolica-gb-lex,milp-cover".to_string()],
        )
        .expect("parse");
        match mode {
            Mode::Orchestrator { only_baselines, .. } => {
                let v = only_baselines.expect("only set");
                assert_eq!(v, vec!["symbolica-gb-lex", "milp-cover"]);
            }
            _ => panic!("expected orchestrator"),
        }
    }

    #[test]
    fn baseline_result_emits_compact_json() {
        let r = BaselineResult {
            baseline: "test".into(),
            status: "ok".into(),
            elapsed_secs: 1.5,
            basis_len: Some(7),
            ..Default::default()
        };
        let s = serde_json::to_string(&r).unwrap();
        // Optional None fields must be skipped, not emitted as nulls.
        assert!(!s.contains("null"));
        assert!(s.contains("\"basis_len\":7"));
    }
}
