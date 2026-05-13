//! Time MILP-cover vs lex Gröbner basis on a directory of Mathematica-style systems.
//!
//! Usage:
//!   `bench <dir> [--timeout SECS] [--no-cache] [--csv PATH]`     (orchestrator)
//!   `bench --worker-groebner <path>`                              (one-shot)
//!   `bench --worker-milp <path>`                                  (one-shot)
//!
//! The orchestrator launches a fresh **subprocess** per algorithm per system,
//! so a runaway Buchberger run (which Symbolica cannot interrupt) is reliably
//! killed by the OS after the timeout. Each worker prints a single JSON line
//! with its result.
//!
//! Gröbner timings are memoized on disk under `target/bench-cache/` keyed by a
//! blake3 hash of the file contents.

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use symbolica::LicenseManager;
use symbolica::domains::algebraic_number::AlgebraicExtension;
use symbolica::domains::rational::Q;

use polynomial_milp::milp_cover::{build_problem, solve_milp};
use polynomial_milp::{
    groebner_basis_from_system_lex_q, load_mathematica_system_from_path_with_var_names, preprocess,
    variable_order,
};

const LICENSE_KEY: &str = "5568d092#6bb8b4fb#ff936916-65ba-552f-b3c1-bdc64597cd21";

/// Default CSV output path: `<repo_root>/results/bench_results.csv`.
fn default_csv_path() -> PathBuf {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap_or(Path::new("."));
    repo_root.join("results").join("bench_results.csv")
}

#[derive(Debug)]
enum Mode {
    Orchestrator {
        dir: PathBuf,
        timeout: Duration,
        no_cache: bool,
        csv: Option<PathBuf>,
    },
    WorkerGroebner {
        path: PathBuf,
    },
    WorkerMilp {
        path: PathBuf,
    },
}

fn parse_args(args: impl IntoIterator<Item = String>) -> Result<Mode, String> {
    let mut dir: Option<PathBuf> = None;
    let mut worker_groebner: Option<PathBuf> = None;
    let mut worker_milp: Option<PathBuf> = None;
    let mut timeout_secs: u64 = 120;
    let mut no_cache = false;
    let mut csv: Option<PathBuf> = None;

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
            "--worker-groebner" => {
                worker_groebner = Some(PathBuf::from(
                    it.next().ok_or("--worker-groebner needs path")?,
                ));
            }
            "--worker-milp" => {
                worker_milp = Some(PathBuf::from(it.next().ok_or("--worker-milp needs path")?));
            }
            s if s.starts_with("--") => return Err(format!("unknown option {s:?}")),
            _ => dir = Some(PathBuf::from(arg)),
        }
    }

    if let Some(path) = worker_groebner {
        return Ok(Mode::WorkerGroebner { path });
    }
    if let Some(path) = worker_milp {
        return Ok(Mode::WorkerMilp { path });
    }
    Ok(Mode::Orchestrator {
        dir: dir.unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../systems/bench")),
        timeout: Duration::from_secs(timeout_secs),
        no_cache,
        csv,
    })
}

#[derive(Clone, Serialize, Deserialize)]
struct CachedGroebner {
    elapsed_secs: f64,
    status: String,
    basis_len: usize,
}

fn cache_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../target/bench-cache")
}

fn cache_key(path: &Path) -> Option<String> {
    let content = fs::read(path).ok()?;
    Some(blake3::hash(&content).to_hex().to_string())
}

fn cache_load(key: &str) -> Option<CachedGroebner> {
    let path = cache_dir().join(format!("{key}.json"));
    serde_json::from_slice(&fs::read(&path).ok()?).ok()
}

fn cache_store(key: &str, value: &CachedGroebner) {
    let dir = cache_dir();
    if fs::create_dir_all(&dir).is_err() {
        return;
    }
    if let Ok(bytes) = serde_json::to_vec(value) {
        let _ = fs::write(dir.join(format!("{key}.json")), bytes);
    }
}

#[derive(Serialize, Deserialize)]
struct WorkerGroebnerResult {
    status: String, // "ok" | "error"
    elapsed_secs: f64,
    basis_len: usize,
    error: Option<String>,
}

#[derive(Serialize, Deserialize)]
struct WorkerMilpResult {
    status: String, // "ok" | "infeasible" | "empty" | "error"
    preprocess_secs: f64,
    solve_secs: f64,
    polys: usize,
    coverage: usize,
    free_params: usize,
    selected: usize,
    zeroed: usize,
    error: Option<String>,
}

fn worker_groebner_result(path: &Path) -> Result<WorkerGroebnerResult, String> {
    let loaded =
        load_mathematica_system_from_path_with_var_names(path).map_err(|e| format!("load: {e}"))?;
    let var_order = variable_order(&loaded.id_to_sym);
    let t = Instant::now();
    let gb = groebner_basis_from_system_lex_q(&loaded.system, var_order, false)
        .map_err(|e| format!("groebner: {e}"))?;
    Ok(WorkerGroebnerResult {
        status: "ok".into(),
        elapsed_secs: t.elapsed().as_secs_f64(),
        basis_len: gb.system.len(),
        error: None,
    })
}

fn run_worker_groebner(path: &Path) -> ! {
    LicenseManager::set_license_key(LICENSE_KEY).ok();
    let payload = worker_groebner_result(path).unwrap_or_else(|e| WorkerGroebnerResult {
        status: "error".into(),
        elapsed_secs: 0.0,
        basis_len: 0,
        error: Some(e),
    });
    println!("{}", serde_json::to_string(&payload).unwrap());
    std::process::exit(0);
}

fn worker_milp_result(path: &Path) -> Result<WorkerMilpResult, String> {
    let loaded =
        load_mathematica_system_from_path_with_var_names(path).map_err(|e| format!("load: {e}"))?;
    let field = AlgebraicExtension::new_complex(Q);
    let t_pre = Instant::now();
    let pre = preprocess(&loaded, &field).map_err(|e| format!("preprocess: {e}"))?;
    let pre_secs = t_pre.elapsed().as_secs_f64();
    let polys = pre.len();
    if polys == 0 {
        return Ok(WorkerMilpResult {
            status: "empty".into(),
            preprocess_secs: pre_secs,
            solve_secs: 0.0,
            polys: 0,
            coverage: 0,
            free_params: 0,
            selected: 0,
            zeroed: 0,
            error: None,
        });
    }
    let data = build_problem(pre, &loaded);
    let n_vars_in_factors = data.v_ids.len();
    let t_solve = Instant::now();
    match solve_milp(&data) {
        Ok(cover) => {
            let selected = cover.selected.len();
            let zeroed = cover.zeroed_locals.len();
            Ok(WorkerMilpResult {
                status: "ok".into(),
                preprocess_secs: pre_secs,
                solve_secs: t_solve.elapsed().as_secs_f64(),
                polys,
                coverage: cover.zeroed_polynomials.len(),
                free_params: n_vars_in_factors.saturating_sub(selected + zeroed),
                selected,
                zeroed,
                error: None,
            })
        }
        Err(good_lp::ResolutionError::Infeasible) => Ok(WorkerMilpResult {
            status: "infeasible".into(),
            preprocess_secs: pre_secs,
            solve_secs: t_solve.elapsed().as_secs_f64(),
            polys,
            coverage: 0,
            free_params: 0,
            selected: 0,
            zeroed: 0,
            error: None,
        }),
        Err(e) => Ok(WorkerMilpResult {
            status: "error".into(),
            preprocess_secs: pre_secs,
            solve_secs: t_solve.elapsed().as_secs_f64(),
            polys,
            coverage: 0,
            free_params: 0,
            selected: 0,
            zeroed: 0,
            error: Some(format!("{e:?}")),
        }),
    }
}

fn run_worker_milp(path: &Path) -> ! {
    LicenseManager::set_license_key(LICENSE_KEY).ok();
    let payload = worker_milp_result(path).unwrap_or_else(|e| WorkerMilpResult {
        status: "error".into(),
        preprocess_secs: 0.0,
        solve_secs: 0.0,
        polys: 0,
        coverage: 0,
        free_params: 0,
        selected: 0,
        zeroed: 0,
        error: Some(e),
    });
    println!("{}", serde_json::to_string(&payload).unwrap());
    std::process::exit(0);
}

enum SubprocessResult {
    Stdout(String),
    Timeout,
    Crashed,
}

/// Run `self_exe` with `args`, killing it after `timeout` (best-effort via SIGKILL).
///
/// The timeout is wall-clock. Callers pass `algorithm_timeout + grace_period` so that
/// a subprocess that completes near the deadline still has time to flush its JSON output
/// before the watchdog fires.
fn run_subprocess(self_exe: &Path, args: &[&str], timeout: Duration) -> SubprocessResult {
    let mut child = match Command::new(self_exe)
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(_) => return SubprocessResult::Crashed,
    };

    let id = child.id();
    let (tx, rx) = mpsc::channel();
    let watchdog = thread::spawn(move || {
        if rx.recv_timeout(timeout).is_err() {
            // best-effort: child may have already exited
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
    match wait_result {
        Ok(status) if status.success() => SubprocessResult::Stdout(stdout),
        Ok(status) if status.code().is_none() => SubprocessResult::Timeout,
        Ok(_) => SubprocessResult::Crashed,
        Err(_) => SubprocessResult::Crashed,
    }
}

enum Timing {
    Ok { elapsed: Duration, basis_len: usize },
    Timeout,
    Error,
}

enum MilpOutcome {
    Ok {
        preprocess: Duration,
        solve: Duration,
        polys: usize,
        coverage: usize,
        free_params: usize,
        selected: usize,
        zeroed: usize,
    },
    Infeasible {
        preprocess: Duration,
        solve: Duration,
        polys: usize,
    },
    Empty,
    Timeout,
    Error,
}

struct Row {
    name: String,
    n_eqs: usize,
    n_vars: usize,
    groebner: Timing,
    milp: MilpOutcome,
}

fn groebner_display(t: &Timing) -> String {
    match t {
        Timing::Ok { elapsed, .. } => format!("{:.3}", elapsed.as_secs_f64()),
        Timing::Timeout => "TIMEOUT".to_string(),
        Timing::Error => "ERR".to_string(),
    }
}

fn milp_status(o: &MilpOutcome) -> &'static str {
    match o {
        MilpOutcome::Ok { .. } => "ok",
        MilpOutcome::Infeasible { .. } => "infeasible",
        MilpOutcome::Empty => "empty",
        MilpOutcome::Timeout => "timeout",
        MilpOutcome::Error => "error",
    }
}

fn milp_solve_secs(o: &MilpOutcome) -> f64 {
    match o {
        MilpOutcome::Ok { solve, .. } | MilpOutcome::Infeasible { solve, .. } => {
            solve.as_secs_f64()
        }
        _ => 0.0,
    }
}

fn milp_pre_secs(o: &MilpOutcome) -> f64 {
    match o {
        MilpOutcome::Ok { preprocess, .. } | MilpOutcome::Infeasible { preprocess, .. } => {
            preprocess.as_secs_f64()
        }
        _ => 0.0,
    }
}

fn milp_faster_than(milp_secs: f64, gb_secs: f64) -> bool {
    milp_secs * 2.0 < gb_secs
}

fn winner(row: &Row) -> &'static str {
    let gb_secs = match &row.groebner {
        Timing::Ok { elapsed, .. } => elapsed.as_secs_f64(),
        Timing::Timeout => {
            return match &row.milp {
                MilpOutcome::Ok { .. } => "MILP (gb TIMEOUT)",
                _ => "n/a",
            };
        }
        Timing::Error => return "n/a",
    };
    match &row.milp {
        MilpOutcome::Ok { solve, .. } => {
            let m = solve.as_secs_f64();
            if milp_faster_than(m, gb_secs) {
                "MILP"
            } else if milp_faster_than(gb_secs, m) {
                "groebner"
            } else {
                "tie"
            }
        }
        _ => "groebner",
    }
}

fn print_markdown(rows: &[Row]) {
    println!(
        "| system | n_eqs | n_vars | t_groebner (s) | t_milp_solve (s) | t_milp_pre (s) | coverage | free | sel | zero | milp_status | winner |"
    );
    println!("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|");
    for row in rows {
        let cov = match &row.milp {
            MilpOutcome::Ok {
                polys, coverage, ..
            } => format!("{}/{}", coverage, polys),
            MilpOutcome::Infeasible { polys, .. } => format!("0/{}", polys),
            _ => "-".to_string(),
        };
        let (free, sel, zero) = match &row.milp {
            MilpOutcome::Ok {
                free_params,
                selected,
                zeroed,
                ..
            } => (*free_params, *selected, *zeroed),
            _ => (0, 0, 0),
        };
        println!(
            "| {} | {} | {} | {} | {:.3} | {:.3} | {} | {} | {} | {} | {} | {} |",
            row.name,
            row.n_eqs,
            row.n_vars,
            groebner_display(&row.groebner),
            milp_solve_secs(&row.milp),
            milp_pre_secs(&row.milp),
            cov,
            free,
            sel,
            zero,
            milp_status(&row.milp),
            winner(row),
        );
    }
}

fn row_to_csv_line(row: &Row) -> String {
    let (gb_secs, gb_status) = match &row.groebner {
        Timing::Ok { elapsed, .. } => (elapsed.as_secs_f64().to_string(), "ok"),
        Timing::Timeout => (String::new(), "timeout"),
        Timing::Error => (String::new(), "error"),
    };
    let (polys, coverage, free, sel, zero) = match &row.milp {
        MilpOutcome::Ok {
            polys,
            coverage,
            free_params,
            selected,
            zeroed,
            ..
        } => (*polys, *coverage, *free_params, *selected, *zeroed),
        MilpOutcome::Infeasible { polys, .. } => (*polys, 0, 0, 0, 0),
        _ => (0, 0, 0, 0, 0),
    };
    format!(
        "{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
        row.name,
        row.n_eqs,
        row.n_vars,
        gb_secs,
        gb_status,
        milp_solve_secs(&row.milp),
        milp_pre_secs(&row.milp),
        milp_status(&row.milp),
        coverage,
        polys,
        free,
        sel,
        zero,
        winner(row),
    )
}

fn write_csv(path: &Path, rows: &[Row]) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let header = "system,n_eqs,n_vars,t_groebner_s,gb_status,t_milp_solve_s,t_milp_pre_s,milp_status,coverage,polys,free_params,selected,zeroed,winner\n";
    let body: String = rows.iter().map(|r| row_to_csv_line(r) + "\n").collect();
    fs::write(path, format!("{header}{body}"))
}

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

fn parse_groebner_stdout(stdout: &str) -> Result<WorkerGroebnerResult, String> {
    stdout
        .lines()
        .rfind(|l| l.trim_start().starts_with('{'))
        .ok_or_else(|| "no JSON line on stdout".to_string())
        .and_then(|line| serde_json::from_str(line).map_err(|e| format!("parse: {e}")))
}

fn parse_milp_stdout(stdout: &str) -> Result<WorkerMilpResult, String> {
    stdout
        .lines()
        .rfind(|l| l.trim_start().starts_with('{'))
        .ok_or_else(|| "no JSON line on stdout".to_string())
        .and_then(|line| serde_json::from_str(line).map_err(|e| format!("parse: {e}")))
}

fn run_groebner_via_subprocess(self_exe: &Path, path: &Path, timeout: Duration) -> Timing {
    let timeout_str = timeout.as_secs().to_string();
    let path_str = path.to_string_lossy().to_string();
    let args = ["--worker-groebner", &path_str, "--timeout", &timeout_str];
    // 5-second grace period so the worker can flush its JSON output before SIGKILL fires
    let wall = timeout + Duration::from_secs(5);
    match run_subprocess(self_exe, &args, wall) {
        SubprocessResult::Stdout(s) => match parse_groebner_stdout(&s) {
            Ok(r) if r.status == "ok" => Timing::Ok {
                elapsed: Duration::from_secs_f64(r.elapsed_secs),
                basis_len: r.basis_len,
            },
            Ok(_) => Timing::Error,
            Err(_) => Timing::Error,
        },
        SubprocessResult::Timeout => Timing::Timeout,
        SubprocessResult::Crashed => Timing::Error,
    }
}

fn run_milp_via_subprocess(self_exe: &Path, path: &Path, timeout: Duration) -> MilpOutcome {
    let timeout_str = timeout.as_secs().to_string();
    let path_str = path.to_string_lossy().to_string();
    let args = ["--worker-milp", &path_str, "--timeout", &timeout_str];
    let wall = timeout + Duration::from_secs(5);
    match run_subprocess(self_exe, &args, wall) {
        SubprocessResult::Stdout(s) => match parse_milp_stdout(&s) {
            Ok(r) => match r.status.as_str() {
                "ok" => MilpOutcome::Ok {
                    preprocess: Duration::from_secs_f64(r.preprocess_secs),
                    solve: Duration::from_secs_f64(r.solve_secs),
                    polys: r.polys,
                    coverage: r.coverage,
                    free_params: r.free_params,
                    selected: r.selected,
                    zeroed: r.zeroed,
                },
                "infeasible" => MilpOutcome::Infeasible {
                    preprocess: Duration::from_secs_f64(r.preprocess_secs),
                    solve: Duration::from_secs_f64(r.solve_secs),
                    polys: r.polys,
                },
                "empty" => MilpOutcome::Empty,
                _ => MilpOutcome::Error,
            },
            Err(_) => MilpOutcome::Error,
        },
        SubprocessResult::Timeout => MilpOutcome::Timeout,
        SubprocessResult::Crashed => MilpOutcome::Error,
    }
}

fn process_file(
    self_exe: &Path,
    path: &Path,
    timeout: Duration,
    no_cache: bool,
) -> Result<Row, String> {
    let name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("?")
        .to_string();

    let loaded =
        load_mathematica_system_from_path_with_var_names(path).map_err(|e| format!("load: {e}"))?;
    let n_eqs = loaded.system.equations.len();
    let n_vars = loaded.var_names.len();
    drop(loaded);

    let groebner = match (!no_cache).then(|| cache_key(path)).flatten() {
        Some(k) => match cache_load(&k) {
            Some(c) => match c.status.as_str() {
                "ok" => Timing::Ok {
                    elapsed: Duration::from_secs_f64(c.elapsed_secs),
                    basis_len: c.basis_len,
                },
                "timeout" => Timing::Timeout,
                _ => Timing::Error,
            },
            None => {
                let t = run_groebner_via_subprocess(self_exe, path, timeout);
                let cached = match &t {
                    Timing::Ok { elapsed, basis_len } => Some(CachedGroebner {
                        elapsed_secs: elapsed.as_secs_f64(),
                        status: "ok".into(),
                        basis_len: *basis_len,
                    }),
                    Timing::Timeout => Some(CachedGroebner {
                        elapsed_secs: timeout.as_secs_f64(),
                        status: "timeout".into(),
                        basis_len: 0,
                    }),
                    Timing::Error => None,
                };
                if let Some(c) = cached {
                    cache_store(&k, &c);
                }
                t
            }
        },
        None => run_groebner_via_subprocess(self_exe, path, timeout),
    };

    let milp = run_milp_via_subprocess(self_exe, path, timeout);

    Ok(Row {
        name,
        n_eqs,
        n_vars,
        groebner,
        milp,
    })
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
        Mode::WorkerGroebner { path } => run_worker_groebner(&path),
        Mode::WorkerMilp { path } => run_worker_milp(&path),
        Mode::Orchestrator {
            dir,
            timeout,
            no_cache,
            csv,
        } => {
            LicenseManager::set_license_key(LICENSE_KEY).ok();
            let self_exe = std::env::current_exe().expect("current_exe");
            eprintln!(
                "bench: self={} dir={} timeout={:?}",
                self_exe.display(),
                dir.display(),
                timeout
            );
            let files = collect_inputs(&dir);
            if files.is_empty() {
                eprintln!("no *.txt files in {}", dir.display());
                std::process::exit(1);
            }
            let mut rows = Vec::with_capacity(files.len());
            for path in &files {
                eprintln!("  > {}", path.file_name().unwrap().to_string_lossy());
                match process_file(&self_exe, path, timeout, no_cache) {
                    Ok(row) => {
                        eprintln!(
                            "    gb={} milp={:.3} ({}) -> {}",
                            groebner_display(&row.groebner),
                            milp_solve_secs(&row.milp),
                            milp_status(&row.milp),
                            winner(&row),
                        );
                        rows.push(row);
                    }
                    Err(e) => eprintln!("    SKIP: {e}"),
                }
            }
            println!();
            print_markdown(&rows);
            let csv_path = csv.unwrap_or_else(default_csv_path);
            match write_csv(&csv_path, &rows) {
                Ok(()) => eprintln!("\nWrote {}", csv_path.display()),
                Err(e) => eprintln!("CSV write failed: {e}"),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_args_uses_default_dir() {
        let options = match parse_args(std::iter::empty::<String>()).expect("parse args") {
            Mode::Orchestrator { dir, .. } => dir,
            _ => panic!("expected orchestrator"),
        };
        assert!(options.ends_with("systems/bench"));
    }

    #[test]
    fn parse_args_accepts_a_path() {
        let options = match parse_args(["systems/my_bench".to_string()]).expect("parse args") {
            Mode::Orchestrator { dir, .. } => dir,
            _ => panic!("expected orchestrator"),
        };
        assert_eq!(options, PathBuf::from("systems/my_bench"));
    }

    #[test]
    fn parse_args_rejects_unknown_flag() {
        let err = parse_args(["--objective".to_string()]).expect_err("error");
        assert!(err.contains("unknown option"));
    }

    #[test]
    fn milp_faster_than_checks_2x_threshold() {
        assert!(milp_faster_than(1.0, 3.0));
        assert!(!milp_faster_than(3.0, 3.0));
        assert!(!milp_faster_than(3.0, 1.0));
    }

    #[test]
    fn default_csv_path_is_in_results_dir() {
        let p = default_csv_path();
        assert!(p.to_str().unwrap().contains("results"));
        assert_eq!(p.file_name().unwrap(), "bench_results.csv");
    }
}
