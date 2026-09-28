//! Run F5 and GVW on a Mathematica-style polynomial system.
//!
//! Usage:
//!   `sig_groebner [path] [--algo m4gb|f5|gvw|both] [--order lex|grevlex] [--timeout-secs N]`
//!
//! The default path is `<repo>/systems/system_s.txt`. Lex is the default
//! monomial order; pass `--order grevlex` to compute a degrevlex basis.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use symbolica::LicenseManager;
use symbolica::domains::rational::Q;
use symbolica::poly::groebner::GroebnerBasis;
use symbolica::poly::polynomial::MultivariatePolynomial;
use symbolica::poly::{GrevLexOrder, LexOrder, MonomialOrder, PolyVariable};

use polynomial_cdcl::{
    PolynomialSystem, default_systems_dir_from_manifest, groebner_f5_from_system_with,
    groebner_gvw_from_system_with, groebner_m4gb_from_system_with, ideal_in_order,
    load_mathematica_system_from_path_with_var_names, variable_order,
};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Algo {
    M4gb,
    F5,
    Gvw,
    Both,
    Builtin,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Order {
    Lex,
    GrevLex,
}

impl Order {
    fn label(self) -> &'static str {
        match self {
            Order::Lex => "lex",
            Order::GrevLex => "grevlex",
        }
    }
}

struct Args {
    path: PathBuf,
    algo: Algo,
    order: Order,
    timeout: Option<Duration>,
    summary: bool,
}

fn parse_args(it: impl IntoIterator<Item = String>) -> Result<Args, String> {
    let mut path: Option<PathBuf> = None;
    let mut algo = Algo::Both;
    let mut order = Order::Lex;
    let mut timeout: Option<Duration> = None;
    let mut summary = false;
    let mut it = it.into_iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--algo" => {
                algo = match it.next().as_deref() {
                    Some("m4gb") => Algo::M4gb,
                    Some("f5") => Algo::F5,
                    Some("gvw") => Algo::Gvw,
                    Some("both") => Algo::Both,
                    Some("builtin") => Algo::Builtin,
                    other => return Err(format!("--algo: unknown {other:?}")),
                };
            }
            "--order" => {
                order = match it.next().as_deref() {
                    Some("lex") => Order::Lex,
                    Some("grevlex") | Some("degrevlex") => Order::GrevLex,
                    other => return Err(format!("--order: unknown {other:?}")),
                };
            }
            "--timeout-secs" => {
                let v = it.next().ok_or("--timeout-secs requires a value")?;
                let secs: u64 = v.parse().map_err(|e| format!("--timeout-secs: {e}"))?;
                timeout = Some(Duration::from_secs(secs));
            }
            "--summary" => summary = true,
            s if s.starts_with("--") => return Err(format!("unknown option {s:?}")),
            _ => path = Some(PathBuf::from(arg)),
        }
    }
    let path = path.unwrap_or_else(|| default_systems_dir_from_manifest().join("system_s.txt"));
    Ok(Args {
        path,
        algo,
        order,
        timeout,
        summary,
    })
}

fn run_with_optional_timeout<F, R>(label: &str, timeout: Option<Duration>, f: F) -> Option<R>
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    let Some(deadline) = timeout else {
        let t0 = Instant::now();
        let r = f();
        println!("  {label} finished in {:?}", t0.elapsed());
        return Some(r);
    };

    let (tx, rx) = std::sync::mpsc::channel();
    let handle = std::thread::spawn(move || {
        let t0 = Instant::now();
        let r = f();
        let _ = tx.send((r, t0.elapsed()));
    });

    match rx.recv_timeout(deadline) {
        Ok((r, dur)) => {
            let _ = handle.join();
            println!("  {label} finished in {dur:?}");
            Some(r)
        }
        Err(_) => {
            println!("  {label} TIMEOUT after {deadline:?}");
            None
        }
    }
}

fn report<O: MonomialOrder>(gb: &GroebnerBasis<Q, u16, O>, summary: bool)
where
    MultivariatePolynomial<Q, u16, O>: std::fmt::Display,
{
    println!("  basis size: {}", gb.system.len());
    if summary {
        return;
    }
    for g in &gb.system {
        println!("\t{g}");
    }
}

fn run_algo<O: MonomialOrder + Send + 'static>(
    label: &str,
    algo: Algo,
    system: &PolynomialSystem,
    var_order: Arc<Vec<PolyVariable>>,
    timeout: Option<Duration>,
    summary: bool,
) where
    MultivariatePolynomial<Q, u16, O>: std::fmt::Display,
{
    if matches!(algo, Algo::M4gb | Algo::Both) {
        println!("\nM4GB ({label} over ℚ)");
        let sys = system.clone();
        let vo = var_order.clone();
        let r = run_with_optional_timeout("M4GB", timeout, move || {
            groebner_m4gb_from_system_with::<Q, O>(&sys, &Q, vo)
        });
        match r {
            Some(Ok(gb)) => report(&gb, summary),
            Some(Err(e)) => eprintln!("  error: {e}"),
            None => {}
        }
    }
    if matches!(algo, Algo::F5 | Algo::Both) {
        println!("\nF5 ({label} over ℚ)");
        let sys = system.clone();
        let vo = var_order.clone();
        let r = run_with_optional_timeout("F5", timeout, move || {
            groebner_f5_from_system_with::<Q, O>(&sys, &Q, vo)
        });
        match r {
            Some(Ok(gb)) => report(&gb, summary),
            Some(Err(e)) => eprintln!("  error: {e}"),
            None => {}
        }
    }
    if matches!(algo, Algo::Gvw | Algo::Both) {
        println!("\nGVW ({label} over ℚ)");
        let sys = system.clone();
        let vo = var_order.clone();
        let r = run_with_optional_timeout("GVW", timeout, move || {
            groebner_gvw_from_system_with::<Q, O>(&sys, &Q, vo)
        });
        match r {
            Some(Ok(gb)) => report(&gb, summary),
            Some(Err(e)) => eprintln!("  error: {e}"),
            None => {}
        }
    }
    if matches!(algo, Algo::Builtin) {
        println!("\nSymbolica GroebnerBasis::new ({label} over ℚ)");
        let sys = system.clone();
        let vo = var_order.clone();
        let r = run_with_optional_timeout(
            "builtin",
            timeout,
            move || -> Result<GroebnerBasis<Q, u16, O>, String> {
                let ideal = ideal_in_order::<Q, O>(&sys, &Q, vo)?;
                Ok(GroebnerBasis::new(&ideal, false))
            },
        );
        match r {
            Some(Ok(gb)) => report(&gb, summary),
            Some(Err(e)) => eprintln!("  error: {e}"),
            None => {}
        }
    }
}

fn main() {
    let args = match parse_args(std::env::args().skip(1)) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    };

    LicenseManager::set_license_key(polynomial_cdcl::SYMBOLICA_LICENSE_KEY).ok();
    polynomial_cdcl::exit_if_solver_worker();

    println!("Loading {}", args.path.display());
    let loaded = match load_mathematica_system_from_path_with_var_names(&args.path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("load error: {e}");
            std::process::exit(1);
        }
    };
    println!(
        "Parsed: {} equations, {} variables, order = {}",
        loaded.system.equations.len(),
        loaded.var_names.len(),
        args.order.label(),
    );

    let var_order = variable_order(&loaded.id_to_sym);

    match args.order {
        Order::Lex => run_algo::<LexOrder>(
            "lex",
            args.algo,
            &loaded.system,
            var_order,
            args.timeout,
            args.summary,
        ),
        Order::GrevLex => run_algo::<GrevLexOrder>(
            "grevlex",
            args.algo,
            &loaded.system,
            var_order,
            args.timeout,
            args.summary,
        ),
    }
}
