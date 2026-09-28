//! Evaluates a candidate hitting set against a Gröbner-basis backend.

use std::io::{Read, Write};
use std::process::{Command, ExitStatus, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use symbolica::domains::Ring;
use symbolica::domains::rational::Q;
use symbolica::poly::polynomial::MultivariatePolynomial;
use symbolica::poly::{GrevLexOrder, MonomialOrder, PolyVariable};

use crate::cover::minimal_covers;
use crate::factor_prep::PolyExtLex;
use crate::{groebner_f5, groebner_gvw, groebner_m4gb};

pub type PolyExtGrevLex = MultivariatePolynomial<Q, u16, GrevLexOrder>;

pub fn solution_dimension(basis: &[PolyExtGrevLex], nvars: usize) -> Option<usize> {
    let supports: Vec<Vec<usize>> = basis
        .iter()
        .filter(|p| !p.is_zero())
        .map(|p| {
            p.max_exp()
                .iter()
                .enumerate()
                .filter(|&(_, &e)| e > 0)
                .map(|(i, _)| i)
                .collect()
        })
        .collect();
    minimal_covers(&supports, nvars)?
        .into_iter()
        .map(|transversal| nvars - transversal.count_ones() as usize)
        .max()
}

/// A reduced degrevlex Gröbner basis equal to `{1}` (a non-zero constant)
/// certifies that the cover's ideal is the whole ring, i.e. the cover is
/// inconsistent.
fn is_unit_ideal<O: MonomialOrder>(system: &[MultivariatePolynomial<Q, u16, O>]) -> bool {
    system.iter().any(|p| !p.is_zero() && p.is_constant())
}

/// Outcome of computing the Gröbner basis of a single cover.
pub struct ComputedBasis {
    /// Full basis elements for in-process engines, or leading monomials for
    /// an external engine that reports only the initial ideal.
    pub elements: Vec<PolyExtGrevLex>,
    /// Whether `elements` contains only enough initial-ideal information for
    /// basis length and dimension reporting.
    pub leading_monomials_only: bool,
}

pub enum Outcome {
    /// Non-unit reduced basis: the cover is consistent.
    Consistent(ComputedBasis),
    /// A factorizing backend represented the cover as a union of components.
    ConsistentComponents(Vec<ComputedBasis>),
    /// Reduced basis is `{1}`: the cover's ideal is the whole ring.
    Inconsistent,
    /// Computation exceeded the per-cover time budget.
    Timeout,
    /// The selected external backend could not compute or report a basis.
    BackendError(String),
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum GbBackend {
    M4gb,
    F5,
    Gvw,
    Symbolica,
    SingularGb,
    SingularFacstd,
    /// `SingularGb` after replacing each Rabinowitsch generator by an ideal
    /// saturation.
    SingularGbSat,
    /// `SingularFacstd` after replacing each Rabinowitsch generator by an
    /// ideal saturation.
    SingularFacstdSat,
}

impl GbBackend {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "m4gb" => Some(Self::M4gb),
            "f5" => Some(Self::F5),
            "gvw" => Some(Self::Gvw),
            "symbolica" => Some(Self::Symbolica),
            "singular-gb" | "singular" => Some(Self::SingularGb),
            "singular-facstd" | "facstd" => Some(Self::SingularFacstd),
            "singular-gb-sat" | "singular-sat" => Some(Self::SingularGbSat),
            "singular-facstd-sat" | "facstd-sat" => Some(Self::SingularFacstdSat),
            _ => None,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::M4gb => "m4gb",
            Self::F5 => "f5",
            Self::Gvw => "gvw",
            Self::Symbolica => "symbolica",
            Self::SingularGb => "singular-gb",
            Self::SingularFacstd => "singular-facstd",
            Self::SingularGbSat => "singular-gb-sat",
            Self::SingularFacstdSat => "singular-facstd-sat",
        }
    }
}

fn singular_polynomial(poly: &PolyExtLex) -> String {
    let mut terms = Vec::with_capacity(poly.nterms());
    for (coefficient, exponents) in poly.coefficients.iter().zip(poly.exponents_iter()) {
        let mut factors = vec![format!("({coefficient})")];
        for (index, &exponent) in exponents.iter().enumerate() {
            match exponent {
                0 => {}
                1 => factors.push(format!("x{index}")),
                _ => factors.push(format!("x{index}^{exponent}")),
            }
        }
        terms.push(factors.join("*"));
    }
    terms.join("+")
}

/// Split a cover into ordinary generators and Rabinowitsch generators. A
/// Rabinowitsch generator is the unique generator containing a temporary
/// variable `s`, is linear in `s`, and contains no other temporary; it is
/// returned with the position of `s`. Any other generator stays ordinary,
/// which keeps the encoding sound.
fn rabinowitsch_generators(cover: &[PolyExtLex]) -> (Vec<&PolyExtLex>, Vec<(usize, &PolyExtLex)>) {
    let temporaries: Vec<usize> = cover[0]
        .variables
        .iter()
        .enumerate()
        .filter_map(|(index, variable)| {
            matches!(variable, PolyVariable::Temporary(_)).then_some(index)
        })
        .collect();
    let degree = |poly: &PolyExtLex, position: usize| {
        poly.exponents_iter()
            .map(|exponents| exponents[position])
            .max()
            .unwrap_or(0)
    };
    let uses_one_temporary = |poly: &PolyExtLex| {
        temporaries
            .iter()
            .filter(|&&position| degree(poly, position) > 0)
            .count()
            == 1
    };
    let rabinowitsch: Vec<(usize, &PolyExtLex)> = temporaries
        .iter()
        .filter_map(|&position| {
            let users: Vec<&PolyExtLex> = cover
                .iter()
                .filter(|poly| degree(poly, position) > 0)
                .collect();
            match users.as_slice() {
                [poly] if degree(poly, position) == 1 && uses_one_temporary(poly) => {
                    Some((position, *poly))
                }
                _ => None,
            }
        })
        .collect();
    let ordinary = cover
        .iter()
        .filter(|poly| {
            !rabinowitsch
                .iter()
                .any(|(_, generator)| std::ptr::eq(*generator, *poly))
        })
        .collect();
    (ordinary, rabinowitsch)
}

/// Declare the ring and the cover's ideal `I`. With `saturate`, each
/// Rabinowitsch generator `c*(1 - s*h)` is replaced by the saturation
/// `I:h^oo` (Singular `sat`, applied to `diff(g,s) = -c*h`), and `s` is then
/// set to zero so the dimension over all ring variables equals the dimension
/// over the original ones.
fn singular_ideal(cover: &[PolyExtLex], saturate: bool) -> String {
    let variables = (0..cover[0].nvars())
        .map(|index| format!("x{index}"))
        .collect::<Vec<_>>()
        .join(",");
    let (ordinary, rabinowitsch) = if saturate {
        rabinowitsch_generators(cover)
    } else {
        (cover.iter().collect(), Vec::new())
    };
    let generators = if ordinary.is_empty() {
        "0".to_string()
    } else {
        ordinary
            .into_iter()
            .map(singular_polynomial)
            .collect::<Vec<_>>()
            .join(",")
    };
    let ring = format!("ring r=0,({variables}),dp;\nideal I={generators};\n");
    if rabinowitsch.is_empty() {
        return ring;
    }
    let saturations: String = rabinowitsch
        .iter()
        .map(|(position, generator)| {
            format!(
                "S=sat(I,ideal(diff({},x{position})));\n\
                 if(typeof(S)==\"list\"){{I=S[1];}}else{{I=S;}}\n",
                singular_polynomial(generator)
            )
        })
        .collect();
    let pinned = rabinowitsch
        .iter()
        .map(|(position, _)| format!("x{position}"))
        .collect::<Vec<_>>()
        .join(",");
    format!("LIB \"elim.lib\";\n{ring}def S;\n{saturations}I=I+ideal({pinned});\n")
}

fn singular_program(cover: &[PolyExtLex], saturate: bool) -> String {
    let ideal = singular_ideal(cover, saturate);
    format!(
        "{ideal}\
         option(redSB);\n\
         ideal G=std(I);\n\
         print(\"IDOS_UNIT \"+string(reduce(1,G)==0));\n\
         int i;\n\
         for(i=1;i<=size(G);i=i+1){{print(\"IDOS_LEAD \"+string(leadexp(G[i])));}};\n\
         print(\"IDOS_DONE\");\n\
         exit;\n"
    )
}

fn singular_facstd_program(cover: &[PolyExtLex], saturate: bool) -> String {
    let ideal = singular_ideal(cover, saturate);
    format!(
        "{ideal}\
         ideal FI=interred(I);\n\
         list F=facstd(FI);\n\
         int i; ideal J; ideal G;\n\
         for(i=1;i<=size(F);i=i+1){{\n\
           print(\"IDOS_COMPONENT\");\n\
           J=F[i]; G=std(J);\n\
           print(\"IDOS_UNIT \"+string(reduce(1,G)==0));\n\
           int j;\n\
           for(j=1;j<=size(G);j=j+1){{print(\"IDOS_LEAD \"+string(leadexp(G[j])));}};\n\
           print(\"IDOS_END_COMPONENT\");\n\
         }};\n\
         print(\"IDOS_DONE\");\n\
         exit;\n"
    )
}

enum SingularRun {
    Completed {
        status: ExitStatus,
        stdout: String,
        stderr: String,
    },
    Timeout,
}

fn run_singular(program: &str, timeout: Duration) -> Result<SingularRun, String> {
    let deadline = Instant::now() + timeout;
    let Some(_permit) = crate::solver_process::acquire_solver_process(deadline) else {
        return Ok(SingularRun::Timeout);
    };
    let mut child = Command::new("Singular")
        .args(["-q", "--no-tty"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("could not start Singular: {error}"))?;
    let mut stdout = child.stdout.take().expect("piped Singular stdout");
    let mut stderr = child.stderr.take().expect("piped Singular stderr");
    let stdout_reader = thread::spawn(move || {
        let mut text = String::new();
        let _ = stdout.read_to_string(&mut text);
        text
    });
    let stderr_reader = thread::spawn(move || {
        let mut text = String::new();
        let _ = stderr.read_to_string(&mut text);
        text
    });
    if let Some(mut stdin) = child.stdin.take()
        && let Err(error) = stdin.write_all(program.as_bytes())
    {
        let _ = child.kill();
        let _ = child.wait();
        let stdout = stdout_reader.join().unwrap_or_default();
        let stderr = stderr_reader.join().unwrap_or_default();
        return Err(format!(
            "could not send input to Singular: {error}; stdout={stdout:?}; stderr={stderr:?}"
        ));
    }

    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Some(status),
            Ok(None) if Instant::now() < deadline => thread::sleep(Duration::from_millis(2)),
            Ok(None) => {
                let _ = child.kill();
                let _ = child.wait();
                break None;
            }
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                let stdout = stdout_reader.join().unwrap_or_default();
                let stderr = stderr_reader.join().unwrap_or_default();
                return Err(format!(
                    "failed while waiting for Singular: {error}; stdout={stdout:?}; stderr={stderr:?}"
                ));
            }
        }
    };
    let stdout = stdout_reader.join().unwrap_or_default();
    let stderr = stderr_reader.join().unwrap_or_default();
    Ok(match status {
        Some(status) => SingularRun::Completed {
            status,
            stdout,
            stderr,
        },
        None => SingularRun::Timeout,
    })
}

/// Run Singular in a child process so a timeout can terminate the actual
/// computation instead of leaving an abandoned worker thread behind.  IDOS
/// only needs the basis length and leading-monomial ideal, so the compact
/// `leadexp` output is reconstructed as monomials in the original ring.
fn evaluate_cover_singular(cover: Vec<PolyExtLex>, timeout: Duration, saturate: bool) -> Outcome {
    let program = singular_program(&cover, saturate);
    let (status, stdout, stderr) = match run_singular(&program, timeout) {
        Ok(SingularRun::Completed {
            status,
            stdout,
            stderr,
        }) => (status, stdout, stderr),
        Ok(SingularRun::Timeout) => return Outcome::Timeout,
        Err(error) => return Outcome::BackendError(error),
    };
    if !status.success() || !stdout.lines().any(|line| line.trim() == "IDOS_DONE") {
        return Outcome::BackendError(format!(
            "Singular exited with {status}; stdout={stdout:?}; stderr={stderr:?}"
        ));
    }
    if stdout.lines().any(|line| line.trim() == "IDOS_UNIT 1") {
        return Outcome::Inconsistent;
    }

    let template = cover[0].reorder::<GrevLexOrder>();
    let mut basis = Vec::new();
    for line in stdout.lines() {
        let Some(encoded) = line.trim().strip_prefix("IDOS_LEAD ") else {
            continue;
        };
        let exponents = match encoded
            .split(',')
            .map(str::parse::<u16>)
            .collect::<Result<Vec<_>, _>>()
        {
            Ok(exponents) if exponents.len() == template.nvars() => exponents,
            _ => {
                return Outcome::BackendError(format!(
                    "invalid Singular leading exponent vector {encoded:?}"
                ));
            }
        };
        basis.push(template.monomial(template.ring.one(), exponents));
    }
    Outcome::Consistent(ComputedBasis {
        elements: basis,
        leading_monomials_only: true,
    })
}

fn evaluate_cover_singular_facstd(
    cover: Vec<PolyExtLex>,
    timeout: Duration,
    saturate: bool,
) -> Outcome {
    let program = singular_facstd_program(&cover, saturate);
    let (status, stdout, stderr) = match run_singular(&program, timeout) {
        Ok(SingularRun::Completed {
            status,
            stdout,
            stderr,
        }) => (status, stdout, stderr),
        Ok(SingularRun::Timeout) => return Outcome::Timeout,
        Err(error) => return Outcome::BackendError(error),
    };
    if !status.success() || !stdout.lines().any(|line| line.trim() == "IDOS_DONE") {
        return Outcome::BackendError(format!(
            "Singular facstd exited with {status}; stdout={stdout:?}; stderr={stderr:?}"
        ));
    }

    let template = cover[0].reorder::<GrevLexOrder>();
    let mut components = Vec::new();
    let mut current = None;
    let mut current_is_unit = false;
    for line in stdout.lines().map(str::trim) {
        match line {
            "IDOS_COMPONENT" => {
                current = Some(Vec::new());
                current_is_unit = false;
            }
            "IDOS_UNIT 1" => current_is_unit = true,
            "IDOS_END_COMPONENT" => {
                let Some(elements) = current.take() else {
                    return Outcome::BackendError(
                        "Singular facstd ended a component before starting it".to_string(),
                    );
                };
                if !current_is_unit {
                    components.push(ComputedBasis {
                        elements,
                        leading_monomials_only: true,
                    });
                }
            }
            _ => {
                let Some(encoded) = line.strip_prefix("IDOS_LEAD ") else {
                    continue;
                };
                let exponents = match encoded
                    .split(',')
                    .map(str::parse::<u16>)
                    .collect::<Result<Vec<_>, _>>()
                {
                    Ok(exponents) if exponents.len() == template.nvars() => exponents,
                    _ => {
                        return Outcome::BackendError(format!(
                            "invalid Singular facstd leading exponent vector {encoded:?}"
                        ));
                    }
                };
                let Some(elements) = current.as_mut() else {
                    return Outcome::BackendError(
                        "Singular facstd reported a basis outside a component".to_string(),
                    );
                };
                elements.push(template.monomial(template.ring.one(), exponents));
            }
        }
    }
    if current.is_some() {
        return Outcome::BackendError("Singular facstd left a component unfinished".to_string());
    }
    if components.is_empty() {
        Outcome::Inconsistent
    } else {
        Outcome::ConsistentComponents(components)
    }
}

/// Compute the cover's Gröbner basis in a killable child process. The cover is
/// reordered to degrevlex before computation, which is typically far cheaper
/// than lex for deciding satisfiability (Section 3.1).
pub fn evaluate_cover(cover: Vec<PolyExtLex>, timeout: Duration, backend: GbBackend) -> Outcome {
    // Interreduction retains zero polynomials to preserve the input slots, but
    // they impose no constraint. In particular, Symbolica 1.5.1 attempts to
    // normalize a zero generator and divides by its zero leading coefficient.
    let cover: Vec<_> = cover.into_iter().filter(|poly| !poly.is_zero()).collect();
    if cover.is_empty() {
        return Outcome::Consistent(ComputedBasis {
            elements: Vec::new(),
            leading_monomials_only: false,
        });
    }
    // Do not send an explicit unit ideal to a Gröbner backend. Besides being
    // unnecessary, Symbolica 1.5.1 attempts to normalize the constant again
    // and can divide by zero in its algebraic-number arithmetic.
    if is_unit_ideal(&cover) {
        return Outcome::Inconsistent;
    }
    match backend {
        GbBackend::SingularGb => return evaluate_cover_singular(cover, timeout, false),
        GbBackend::SingularFacstd => return evaluate_cover_singular_facstd(cover, timeout, false),
        GbBackend::SingularGbSat => return evaluate_cover_singular(cover, timeout, true),
        GbBackend::SingularFacstdSat => {
            return evaluate_cover_singular_facstd(cover, timeout, true);
        }
        _ => {}
    }
    if cfg!(test) || crate::solver_process::running_under_test_harness() {
        evaluate_cover_in_process(cover, backend)
    } else {
        crate::solver_process::evaluate_cover_isolated(cover, timeout, backend)
    }
}

pub(crate) fn evaluate_cover_in_process(cover: Vec<PolyExtLex>, backend: GbBackend) -> Outcome {
    let grevlex: Vec<_> = cover.iter().map(|p| p.reorder::<GrevLexOrder>()).collect();
    let system = match backend {
        GbBackend::M4gb => groebner_m4gb(&grevlex).system,
        GbBackend::F5 => groebner_f5(&grevlex).system,
        GbBackend::Gvw => groebner_gvw(&grevlex).system,
        GbBackend::Symbolica => {
            symbolica::poly::groebner::GroebnerBasis::new(&grevlex, false).system
        }
        GbBackend::SingularGb
        | GbBackend::SingularFacstd
        | GbBackend::SingularGbSat
        | GbBackend::SingularFacstdSat => {
            return Outcome::BackendError(
                "Singular cannot run through the in-process worker entry point".to_string(),
            );
        }
    };
    if is_unit_ideal(&system) {
        Outcome::Inconsistent
    } else {
        Outcome::Consistent(ComputedBasis {
            elements: system,
            leading_monomials_only: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Once;

    use symbolica::LicenseManager;
    use symbolica::atom::AtomCore;
    use symbolica::domains::rational::Q;

    use crate::atom_util::parse_expression;
    use crate::factor_prep::{ideal_ext_lex, variable_order};
    use crate::mathematica::load_mathematica_system_from_str_with_var_names;

    static LICENSE: Once = Once::new();

    fn activate_license() {
        LICENSE.call_once(|| {
            let _ = LicenseManager::set_license_key(
                "5568d092#6bb8b4fb#ff936916-65ba-552f-b3c1-bdc64597cd21",
            );
        });
    }

    fn test_poly(source: &str) -> PolyExtLex {
        activate_license();
        let loaded = load_mathematica_system_from_str_with_var_names(&format!("{{{source} == 0}}"))
            .expect("load");
        let field = Q;
        let order = variable_order(&loaded.id_to_sym);
        ideal_ext_lex(&loaded.system, &field, order)
            .expect("polynomial")
            .remove(0)
    }

    fn poly_like(template: &PolyExtLex, source: &str) -> PolyExtLex {
        parse_expression(source.to_string())
            .expect("parse")
            .try_to_polynomial::<_, u16>(&template.ring, Some(template.variables.clone()))
            .expect("polynomial")
    }

    fn grevlex_basis(template: &PolyExtLex, sources: &[&str]) -> Vec<PolyExtGrevLex> {
        sources
            .iter()
            .map(|source| poly_like(template, source).reorder::<GrevLexOrder>())
            .collect()
    }

    #[test]
    fn solution_dimension_counts_independent_variables() {
        let template = test_poly("x + y + z");
        assert_eq!(
            solution_dimension(&grevlex_basis(&template, &["x"]), 3),
            Some(2)
        );
        assert_eq!(
            solution_dimension(&grevlex_basis(&template, &["x*y"]), 3),
            Some(2)
        );
        assert_eq!(
            solution_dimension(&grevlex_basis(&template, &["x*y", "z"]), 3),
            Some(1)
        );
        assert_eq!(
            solution_dimension(&grevlex_basis(&template, &["x", "y", "z"]), 3),
            Some(0)
        );
    }

    #[test]
    fn solution_dimension_of_empty_basis_is_ambient() {
        assert_eq!(solution_dimension(&[], 3), Some(3));
    }

    #[test]
    fn explicit_unit_ideal_skips_symbolica_backend() {
        let template = test_poly("x");
        let one = poly_like(&template, "1");

        assert!(matches!(
            evaluate_cover(vec![one], Duration::from_millis(10), GbBackend::Symbolica),
            Outcome::Inconsistent
        ));
    }

    #[test]
    fn zero_generators_are_removed_before_symbolica_backend() {
        let template = test_poly("x");
        let zero = poly_like(&template, "0");

        assert!(matches!(
            evaluate_cover(
                vec![zero, template],
                Duration::from_secs(5),
                GbBackend::Symbolica
            ),
            Outcome::Consistent(_)
        ));
    }

    #[test]
    fn singular_backend_reports_initial_ideal_and_unit_ideal() {
        if Command::new("Singular").arg("--version").output().is_err() {
            return;
        }

        let product = test_poly("x*y");
        match evaluate_cover(vec![product], Duration::from_secs(5), GbBackend::SingularGb) {
            Outcome::Consistent(basis) => {
                assert!(basis.leading_monomials_only);
                assert_eq!(basis.elements.len(), 1);
                assert_eq!(solution_dimension(&basis.elements, 2), Some(1));
            }
            _ => panic!("Singular should find a consistent product ideal"),
        }

        let x = test_poly("x");
        let x_minus_one = poly_like(&x, "x - 1");
        assert!(matches!(
            evaluate_cover(
                vec![x, x_minus_one],
                Duration::from_secs(5),
                GbBackend::SingularGb,
            ),
            Outcome::Inconsistent
        ));
    }

    fn rabinowitsch_cover() -> Vec<PolyExtLex> {
        let mut product = test_poly("x*y");
        product.add_variable(&PolyVariable::Temporary(0));
        let x = poly_like(&product, "x");
        let inverse = product
            .variable(&PolyVariable::Temporary(0))
            .expect("temporary");
        let sigma = product.one() - inverse * &x;
        vec![product, sigma]
    }

    #[test]
    fn saturation_backends_round_trip_through_labels() {
        for backend in [GbBackend::SingularGbSat, GbBackend::SingularFacstdSat] {
            assert_eq!(GbBackend::parse(backend.label()), Some(backend));
        }
    }

    #[test]
    fn singular_saturates_only_rabinowitsch_generators() {
        let plain = test_poly("x*y");
        assert!(!singular_program(&[plain], true).contains("sat("));
        let cover = rabinowitsch_cover();
        assert!(!singular_program(&cover, false).contains("sat("));
        let program = singular_program(&cover, true);
        assert!(program.contains("sat("));
        assert_eq!(rabinowitsch_generators(&cover).0.len(), 1);
    }

    #[test]
    fn singular_saturation_backend_projects_out_the_temporary() {
        if Command::new("Singular").arg("--version").output().is_err() {
            return;
        }

        let cover = rabinowitsch_cover();
        let nvars = cover[0].nvars();
        match evaluate_cover(cover, Duration::from_secs(5), GbBackend::SingularGbSat) {
            Outcome::Consistent(basis) => {
                assert_eq!(basis.elements.len(), 2);
                assert_eq!(solution_dimension(&basis.elements, nvars), Some(1));
            }
            _ => panic!("the saturated ideal (y, s) should be consistent"),
        }
    }

    #[test]
    fn singular_facstd_backend_reports_each_component() {
        if Command::new("Singular").arg("--version").output().is_err() {
            return;
        }

        let product = test_poly("x*y");
        match evaluate_cover(
            vec![product],
            Duration::from_secs(5),
            GbBackend::SingularFacstd,
        ) {
            Outcome::ConsistentComponents(components) => {
                assert_eq!(components.len(), 2);
                assert!(components.iter().all(|basis| basis.leading_monomials_only));
                assert!(
                    components
                        .iter()
                        .all(|basis| solution_dimension(&basis.elements, 2) == Some(1))
                );
            }
            _ => panic!("Singular facstd should split the product ideal"),
        }
    }
}
