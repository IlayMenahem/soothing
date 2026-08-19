//! Process-isolated execution for solver computations that cannot be cancelled
//! safely in-process.

use std::fs::{File, OpenOptions, remove_file};
use std::io::{BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::thread;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use symbolica::domains::integer::Integer;
use symbolica::domains::rational::{Q, Rational};
use symbolica::poly::polynomial::MultivariatePolynomial;
use symbolica::poly::{GrevLexOrder, LexOrder, MonomialOrder, PolyVariable};

use crate::factor_prep::PolyExtLex;
use crate::gb_search::{ComputedBasis, GbBackend, Outcome, PolyExtGrevLex};

const WORKER_REQUEST_ENV: &str = "POLYNOMIAL_CDCL_SOLVER_REQUEST";
const WORKER_RESPONSE_ENV: &str = "POLYNOMIAL_CDCL_SOLVER_RESPONSE";
const WORKER_PARENT_PID_ENV: &str = "POLYNOMIAL_CDCL_SOLVER_PARENT_PID";
static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);
static PROCESS_LIMITER: OnceLock<ProcessLimiter> = OnceLock::new();

fn solver_process_limit_for(cores: usize) -> usize {
    cores.saturating_sub(2).max(1)
}

fn solver_process_limit() -> usize {
    solver_process_limit_for(
        thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1),
    )
}

/// Cargo compiles a library dependency without `cfg(test)` for binary unit
/// tests. Re-executing that binary would start the test harness, not its source
/// `main`, so those calls must stay in-process.
pub(crate) fn running_under_test_harness() -> bool {
    std::env::current_exe()
        .ok()
        .and_then(|path| {
            path.parent()
                .and_then(|parent| parent.file_name())
                .map(|name| name == "deps")
        })
        .unwrap_or(false)
}

struct ProcessLimiter {
    active: Mutex<usize>,
    available: Condvar,
    limit: usize,
}

impl ProcessLimiter {
    fn new(limit: usize) -> Self {
        Self {
            active: Mutex::new(0),
            available: Condvar::new(),
            limit,
        }
    }

    fn acquire(&'static self, deadline: Instant) -> Option<ProcessPermit> {
        let mut active = self
            .active
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        loop {
            if *active < self.limit {
                *active += 1;
                return Some(ProcessPermit { limiter: self });
            }
            let remaining = deadline.checked_duration_since(Instant::now())?;
            if remaining.is_zero() {
                return None;
            }
            let (next_active, wait) = self
                .available
                .wait_timeout(active, remaining)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            active = next_active;
            if wait.timed_out() && *active >= self.limit {
                return None;
            }
        }
    }
}

pub(crate) struct ProcessPermit {
    limiter: &'static ProcessLimiter,
}

pub(crate) fn acquire_solver_process(deadline: Instant) -> Option<ProcessPermit> {
    PROCESS_LIMITER
        .get_or_init(|| ProcessLimiter::new(solver_process_limit()))
        .acquire(deadline)
}

impl Drop for ProcessPermit {
    fn drop(&mut self) {
        let mut active = self
            .limiter
            .active
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *active -= 1;
        self.limiter.available.notify_one();
    }
}

/// Ensures an early return or panic cannot leave a solver process running.
struct ChildGuard {
    child: Child,
    finished: bool,
}

impl ChildGuard {
    fn new(child: Child) -> Self {
        Self {
            child,
            finished: false,
        }
    }

    fn try_wait(&mut self) -> std::io::Result<Option<ExitStatus>> {
        let status = self.child.try_wait()?;
        if status.is_some() {
            self.finished = true;
        }
        Ok(status)
    }

    fn terminate(&mut self) {
        if self.finished {
            return;
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
        self.finished = true;
    }
}

impl Drop for ChildGuard {
    fn drop(&mut self) {
        self.terminate();
    }
}

#[derive(Serialize, Deserialize)]
struct WireRational {
    numerator: String,
    denominator: String,
}

#[derive(Serialize, Deserialize)]
struct WirePolynomial {
    nvars: usize,
    coefficients: Vec<WireRational>,
    exponents: Vec<u16>,
}

#[derive(Serialize, Deserialize)]
enum WorkerRequest {
    Cover {
        backend: GbBackend,
        polynomials: Vec<WirePolynomial>,
    },
    GbCore {
        polynomials: Vec<WirePolynomial>,
    },
}

#[derive(Serialize, Deserialize)]
struct WireGbCoreRun {
    ranked_core: Vec<WirePolynomial>,
    core: Vec<WirePolynomial>,
}

#[derive(Serialize, Deserialize)]
enum WorkerResponse {
    Consistent {
        basis: Vec<WirePolynomial>,
        leading_monomials_only: bool,
    },
    Inconsistent,
    GbCore(Option<WireGbCoreRun>),
    Error(String),
}

pub(crate) enum ProcessRun<T> {
    Completed(T),
    Timeout,
}

struct TempFiles {
    request: PathBuf,
    response: PathBuf,
}

impl Drop for TempFiles {
    fn drop(&mut self) {
        let _ = remove_file(&self.request);
        let _ = remove_file(&self.response);
    }
}

fn create_temp_file(kind: &str) -> Result<(PathBuf, File), String> {
    for _ in 0..100 {
        let sequence = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "polynomial-cdcl-{kind}-{}-{sequence}.json",
            std::process::id()
        ));
        match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(file) => return Ok((path, file)),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(format!("could not create solver IPC file: {error}"));
            }
        }
    }
    Err("could not allocate a unique solver IPC file".to_string())
}

fn encode_polynomial<O: MonomialOrder>(
    polynomial: &MultivariatePolynomial<Q, u16, O>,
) -> WirePolynomial {
    WirePolynomial {
        nvars: polynomial.nvars(),
        coefficients: polynomial
            .coefficients
            .iter()
            .map(|coefficient| WireRational {
                numerator: coefficient.numerator_ref().to_string(),
                denominator: coefficient.denominator_ref().to_string(),
            })
            .collect(),
        exponents: polynomial.exponents.clone(),
    }
}

fn encode_polynomials<O: MonomialOrder>(
    polynomials: &[MultivariatePolynomial<Q, u16, O>],
) -> Vec<WirePolynomial> {
    polynomials.iter().map(encode_polynomial).collect()
}

fn decode_rational(rational: WireRational) -> Result<Rational, String> {
    let numerator: Integer = rational
        .numerator
        .parse()
        .map_err(|error| format!("invalid solver numerator: {error}"))?;
    let denominator: Integer = rational
        .denominator
        .parse()
        .map_err(|error| format!("invalid solver denominator: {error}"))?;
    if denominator == 0 {
        return Err("solver response contains a zero denominator".to_string());
    }
    Ok(Q.to_element(numerator, denominator, true))
}

fn decode_polynomial<O: MonomialOrder>(
    polynomial: WirePolynomial,
    variables: Arc<Vec<PolyVariable>>,
) -> Result<MultivariatePolynomial<Q, u16, O>, String> {
    if polynomial.nvars != variables.len() {
        return Err(format!(
            "solver polynomial has {} variables, expected {}",
            polynomial.nvars,
            variables.len()
        ));
    }
    if polynomial.exponents.len() != polynomial.coefficients.len() * polynomial.nvars {
        return Err("solver polynomial has an invalid exponent array".to_string());
    }
    let mut result =
        MultivariatePolynomial::new(&Q, Some(polynomial.coefficients.len()), variables);
    for (index, coefficient) in polynomial.coefficients.into_iter().enumerate() {
        let start = index * polynomial.nvars;
        result.append_monomial(
            decode_rational(coefficient)?,
            &polynomial.exponents[start..start + polynomial.nvars],
        );
    }
    Ok(result)
}

fn temporary_variables(polynomials: &[WirePolynomial]) -> Result<Arc<Vec<PolyVariable>>, String> {
    let nvars = polynomials.first().map_or(0, |polynomial| polynomial.nvars);
    if polynomials
        .iter()
        .any(|polynomial| polynomial.nvars != nvars)
    {
        return Err("solver request mixes polynomial variable counts".to_string());
    }
    Ok(Arc::new((0..nvars).map(PolyVariable::Temporary).collect()))
}

fn decode_lex_polynomials(
    polynomials: Vec<WirePolynomial>,
    variables: Arc<Vec<PolyVariable>>,
) -> Result<Vec<PolyExtLex>, String> {
    polynomials
        .into_iter()
        .map(|polynomial| decode_polynomial::<LexOrder>(polynomial, variables.clone()))
        .collect()
}

fn execute_worker(request: WorkerRequest) -> Result<WorkerResponse, String> {
    match request {
        WorkerRequest::Cover {
            backend,
            polynomials,
        } => {
            let variables = temporary_variables(&polynomials)?;
            let cover = decode_lex_polynomials(polynomials, variables)?;
            match crate::gb_search::evaluate_cover_in_process(cover, backend) {
                Outcome::Consistent(basis) => Ok(WorkerResponse::Consistent {
                    basis: encode_polynomials(&basis.elements),
                    leading_monomials_only: basis.leading_monomials_only,
                }),
                Outcome::ConsistentComponents(_) => Err(
                    "factorizing components cannot run through the in-process worker entry point"
                        .to_string(),
                ),
                Outcome::Inconsistent => Ok(WorkerResponse::Inconsistent),
                Outcome::Timeout => Err("nested solver worker timed out".to_string()),
                Outcome::BackendError(error) => Ok(WorkerResponse::Error(error)),
            }
        }
        WorkerRequest::GbCore { polynomials } => {
            let variables = temporary_variables(&polynomials)?;
            let candidate = decode_lex_polynomials(polynomials, variables)?;
            let run =
                crate::gb_core::gb_core_once_for_worker(candidate).map(|(ranked_core, core)| {
                    WireGbCoreRun {
                        ranked_core: encode_polynomials(&ranked_core),
                        core: encode_polynomials(&core),
                    }
                });
            Ok(WorkerResponse::GbCore(run))
        }
    }
}

#[cfg(unix)]
fn start_parent_watchdog() {
    let Some(expected_parent) = std::env::var(WORKER_PARENT_PID_ENV)
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
    else {
        return;
    };
    thread::spawn(move || {
        unsafe extern "C" {
            fn getppid() -> i32;
        }
        loop {
            thread::sleep(Duration::from_millis(100));
            // SAFETY: getppid has no preconditions and does not dereference memory.
            let current_parent = unsafe { getppid() };
            if current_parent < 0 || current_parent as u32 != expected_parent {
                std::process::exit(70);
            }
        }
    });
}

#[cfg(not(unix))]
fn start_parent_watchdog() {}

/// Exit the current executable as a hidden solver worker when the worker
/// environment is present. CLI binaries call this immediately after activating
/// the Symbolica license and before parsing user arguments.
pub fn exit_if_solver_worker() {
    let Some(request_path) = std::env::var_os(WORKER_REQUEST_ENV).map(PathBuf::from) else {
        return;
    };
    let Some(response_path) = std::env::var_os(WORKER_RESPONSE_ENV).map(PathBuf::from) else {
        eprintln!("solver worker is missing its response path");
        std::process::exit(70);
    };
    start_parent_watchdog();

    let response = File::open(&request_path)
        .map_err(|error| format!("could not open solver request: {error}"))
        .and_then(|file| {
            serde_json::from_reader(BufReader::new(file))
                .map_err(|error| format!("could not decode solver request: {error}"))
        })
        .and_then(execute_worker)
        .unwrap_or_else(WorkerResponse::Error);

    let result = OpenOptions::new()
        .write(true)
        .truncate(true)
        .open(&response_path)
        .map_err(|error| format!("could not open solver response: {error}"))
        .and_then(|file| {
            let mut writer = BufWriter::new(file);
            serde_json::to_writer(&mut writer, &response)
                .map_err(|error| format!("could not encode solver response: {error}"))?;
            writer
                .flush()
                .map_err(|error| format!("could not flush solver response: {error}"))
        });
    if let Err(error) = result {
        eprintln!("{error}");
        std::process::exit(70);
    }
    std::process::exit(0);
}

fn run_worker(
    request: &WorkerRequest,
    timeout: Duration,
) -> Result<ProcessRun<WorkerResponse>, String> {
    // A downstream binary that forgot to call `exit_if_solver_worker` must not
    // recursively re-execute itself until the machine runs out of processes.
    if std::env::var_os(WORKER_REQUEST_ENV).is_some() {
        return Err(
            "refusing to launch a nested solver worker; call exit_if_solver_worker() at the start of main"
                .to_string(),
        );
    }
    let deadline = Instant::now() + timeout;
    let Some(_permit) = acquire_solver_process(deadline) else {
        return Ok(ProcessRun::Timeout);
    };

    let (request_path, request_file) = create_temp_file("request")?;
    let (response_path, response_file) = create_temp_file("response")?;
    let files = TempFiles {
        request: request_path,
        response: response_path,
    };
    let mut request_writer = BufWriter::new(request_file);
    serde_json::to_writer(&mut request_writer, request)
        .map_err(|error| format!("could not encode solver request: {error}"))?;
    request_writer
        .flush()
        .map_err(|error| format!("could not flush solver request: {error}"))?;
    drop(response_file);

    let executable = std::env::current_exe()
        .map_err(|error| format!("could not locate the current executable: {error}"))?;
    let child = Command::new(executable)
        .env(WORKER_REQUEST_ENV, &files.request)
        .env(WORKER_RESPONSE_ENV, &files.response)
        .env(WORKER_PARENT_PID_ENV, std::process::id().to_string())
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .spawn()
        .map_err(|error| format!("could not start solver worker: {error}"))?;
    let mut child = ChildGuard::new(child);

    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Some(status),
            Ok(None) if Instant::now() < deadline => thread::sleep(Duration::from_millis(2)),
            Ok(None) => {
                child.terminate();
                break None;
            }
            Err(error) => {
                child.terminate();
                return Err(format!("failed while waiting for solver worker: {error}"));
            }
        }
    };
    let Some(status) = status else {
        return Ok(ProcessRun::Timeout);
    };
    if !status.success() {
        return Err(format!("solver worker exited with {status}"));
    }
    let response_file = File::open(&files.response)
        .map_err(|error| format!("could not open solver response: {error}"))?;
    let response = serde_json::from_reader(BufReader::new(response_file))
        .map_err(|error| format!("could not decode solver response: {error}"))?;
    Ok(ProcessRun::Completed(response))
}

pub(crate) fn evaluate_cover_isolated(
    cover: Vec<PolyExtLex>,
    timeout: Duration,
    backend: GbBackend,
) -> Outcome {
    let variables = cover[0].reorder::<GrevLexOrder>().variables;
    let request = WorkerRequest::Cover {
        backend,
        polynomials: encode_polynomials(&cover),
    };
    match run_worker(&request, timeout) {
        Ok(ProcessRun::Timeout) => Outcome::Timeout,
        Err(error) => Outcome::BackendError(error),
        Ok(ProcessRun::Completed(WorkerResponse::Consistent {
            basis,
            leading_monomials_only,
        })) => {
            let elements: Result<Vec<PolyExtGrevLex>, _> = basis
                .into_iter()
                .map(|polynomial| decode_polynomial(polynomial, variables.clone()))
                .collect();
            match elements {
                Ok(elements) => Outcome::Consistent(ComputedBasis {
                    elements,
                    leading_monomials_only,
                }),
                Err(error) => Outcome::BackendError(error),
            }
        }
        Ok(ProcessRun::Completed(WorkerResponse::Inconsistent)) => Outcome::Inconsistent,
        Ok(ProcessRun::Completed(WorkerResponse::Error(error))) => Outcome::BackendError(error),
        Ok(ProcessRun::Completed(WorkerResponse::GbCore(_))) => Outcome::BackendError(
            "solver worker returned a GB-core response for a cover".to_string(),
        ),
    }
}

pub(crate) fn gb_core_once_isolated(
    candidate: Vec<PolyExtLex>,
    timeout: Duration,
) -> Result<ProcessRun<Option<(Vec<PolyExtLex>, Vec<PolyExtLex>)>>, String> {
    let variables = candidate.first().map_or_else(
        || Arc::new(Vec::new()),
        |polynomial| polynomial.variables.clone(),
    );
    let request = WorkerRequest::GbCore {
        polynomials: encode_polynomials(&candidate),
    };
    match run_worker(&request, timeout)? {
        ProcessRun::Timeout => Ok(ProcessRun::Timeout),
        ProcessRun::Completed(WorkerResponse::GbCore(run)) => {
            let run = run
                .map(
                    |run| -> Result<(Vec<PolyExtLex>, Vec<PolyExtLex>), String> {
                        let ranked_core = run
                            .ranked_core
                            .into_iter()
                            .map(|polynomial| decode_polynomial(polynomial, variables.clone()))
                            .collect::<Result<_, _>>()?;
                        let core = run
                            .core
                            .into_iter()
                            .map(|polynomial| decode_polynomial(polynomial, variables.clone()))
                            .collect::<Result<_, _>>()?;
                        Ok((ranked_core, core))
                    },
                )
                .transpose()?;
            Ok(ProcessRun::Completed(run))
        }
        ProcessRun::Completed(WorkerResponse::Error(error)) => Err(error),
        ProcessRun::Completed(_) => {
            Err("solver worker returned a cover response for GB-core".to_string())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solver_process_limit_leaves_two_cores_free() {
        assert_eq!(solver_process_limit_for(1), 1);
        assert_eq!(solver_process_limit_for(2), 1);
        assert_eq!(solver_process_limit_for(3), 1);
        assert_eq!(solver_process_limit_for(8), 6);
    }

    #[test]
    fn process_limiter_releases_permits() {
        let limiter = Box::leak(Box::new(ProcessLimiter::new(1)));
        let first = limiter
            .acquire(Instant::now() + Duration::from_secs(1))
            .expect("first permit");
        assert!(limiter.acquire(Instant::now()).is_none());
        drop(first);
        assert!(limiter.acquire(Instant::now()).is_some());
    }

    #[test]
    fn polynomial_wire_round_trip_preserves_terms() {
        let variables = Arc::new(vec![PolyVariable::Temporary(4), PolyVariable::Temporary(9)]);
        let mut polynomial = PolyExtLex::new(&Q, Some(2), variables.clone());
        polynomial.append_monomial(Q.to_element(3.into(), 2.into(), true), &[2, 0]);
        polynomial.append_monomial(Q.to_element((-5).into(), 7.into(), true), &[0, 1]);

        let decoded = decode_polynomial::<LexOrder>(encode_polynomial(&polynomial), variables)
            .expect("wire round trip");

        assert_eq!(decoded, polynomial);
    }
}
