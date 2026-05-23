//! MILP-based radical cover as described in `problem statement.md`.
//!
//! Thin CLI driver around [`polynomial_milp::milp_cover`].
//!
//! Usage: `milp_cover [path]` — default path: `systems/system.txt`.

use std::path::PathBuf;
use std::time::Instant;

use good_lp::ResolutionError;

use symbolica::LicenseManager;
use symbolica::domains::algebraic_number::AlgebraicExtension;
use symbolica::domains::rational::Q;

use polynomial_milp::milp_cover::{
    ParameterizationError, build_problem, certify_cover_groebner, check_cover_preprocessed,
    print_certification, print_cover, print_variable_sets, solve_milp,
};
use polynomial_milp::{
    default_systems_dir_from_manifest, load_mathematica_system_from_path_with_var_names, preprocess,
};

#[derive(Debug)]
struct CliOptions {
    path: PathBuf,
}

fn parse_args(args: impl IntoIterator<Item = String>) -> Result<CliOptions, String> {
    let mut path: Option<PathBuf> = None;
    for arg in args {
        if arg.starts_with("--") {
            return Err(format!("unknown option {arg:?}"));
        }
        path = Some(PathBuf::from(arg));
    }
    Ok(CliOptions {
        path: path.unwrap_or_else(|| default_systems_dir_from_manifest().join("system.txt")),
    })
}

fn main() {
    let key = "5568d092#6bb8b4fb#ff936916-65ba-552f-b3c1-bdc64597cd21";
    LicenseManager::set_license_key(key).unwrap();

    let options = match parse_args(std::env::args().skip(1)) {
        Ok(options) => options,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    };

    println!("polynomial-milp — MILP radical cover (good_lp / HiGHS)\n");
    println!("Loading {}", options.path.display());

    let loaded = match load_mathematica_system_from_path_with_var_names(&options.path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("load error: {e}");
            std::process::exit(1);
        }
    };
    println!(
        "Parsed: {} equations, {} variables (loader order)",
        loaded.system.equations.len(),
        loaded.var_names.len()
    );

    let field = AlgebraicExtension::new_complex(Q);

    println!("\nPreprocessing (interreduce + factor)");
    let t_pre = Instant::now();
    let pre = match preprocess(&loaded, &field) {
        Ok(pre) => pre,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };
    println!(
        "  {} non-trivial polynomial(s), elapsed {:?}",
        pre.len(),
        t_pre.elapsed()
    );
    if pre.is_empty() {
        println!("Nothing to cover — all polynomials reduced to zero.");
        return;
    }

    let data = build_problem(pre, &loaded);
    println!(
        "  |F|={}, |V|={}, |A|={}",
        data.factors.len(),
        data.v_ids.len(),
        data.admissible.len()
    );

    println!("\nSolving MILP (maximize zeroed polynomials) with HiGHS ...");
    let t_solve = Instant::now();
    let cover = match solve_milp(&data) {
        Ok(c) => c,
        Err(ResolutionError::Infeasible) => {
            eprintln!("infeasible: no valid cover exists for this system.");
            std::process::exit(3);
        }
        Err(e) => {
            eprintln!("solver error: {e}");
            std::process::exit(4);
        }
    };
    println!("  solved in {:?}", t_solve.elapsed());

    print_cover(&loaded, &data, &cover);
    print_variable_sets(&loaded, &data, &cover);

    match check_cover_preprocessed(&loaded, &data, &cover) {
        Ok(report) if report.uncovered_count == 0 => println!(
            "Preprocessed cover check: all {} factor group(s) covered.",
            report.checked_factor_groups
        ),
        Ok(report) => {
            eprintln!(
                "\nPreprocessed cover check: {}/{} factor group(s) not covered.",
                report.uncovered_count, report.checked_factor_groups
            );
            std::process::exit(6);
        }
        Err(ParameterizationError::AlgebraicSubstitutionNotImplemented { variable, degree }) => {
            println!(
                "Note: preprocessed cover check skipped — degree-{degree} algebraic solve for \
                 `{variable}` is not yet implemented."
            );
        }
        Err(error) => {
            eprintln!("\nPreprocessed cover check failed: {error}");
            std::process::exit(6);
        }
    }

    let t_cert = Instant::now();
    match certify_cover_groebner(&loaded, &data, &cover) {
        Ok(report) => {
            print_certification(&report);
            println!("  (certification elapsed {:?})", t_cert.elapsed());
        }
        Err(e) => {
            eprintln!("\nSymbolic certification failed: {e}");
        }
    }

    println!("\nDone.");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_args_uses_system_txt_by_default() {
        let options = parse_args(std::iter::empty::<String>()).expect("parse args");
        assert!(options.path.ends_with("systems/system.txt"));
    }

    #[test]
    fn parse_args_accepts_a_path() {
        let options = parse_args(["systems/s_system.txt".to_string()]).expect("parse args");
        assert_eq!(options.path, PathBuf::from("systems/s_system.txt"));
    }

    #[test]
    fn parse_args_rejects_unknown_flag() {
        let err = parse_args(["--objective".to_string()]).expect_err("error");
        assert!(err.contains("unknown option"));
    }
}
