//! Evaluates a candidate hitting set against a Gröbner-basis backend.

use std::io::{Read, Write};
use std::process::{Command, ExitStatus, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use symbolica::domains::Ring;
use symbolica::domains::integer::Integer;
use symbolica::domains::rational::{Q, Rational};
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
    minimal_covers(&supports)
        .into_iter()
        .map(|transversal| nvars - transversal.len())
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
    /// Singular's `primdecGTZ`; each primary component is reported by its
    /// associated prime.
    SingularPrimdecGtz,
    /// `SingularGb` after replacing each Rabinowitsch generator by an ideal
    /// saturation.
    SingularGbSat,
    /// `SingularFacstd` with each Rabinowitsch generator replaced by an ideal
    /// saturation: `facstd` splits the other generators, and each component
    /// is saturated afterwards.
    SingularFacstdSat,
    /// Macaulay2's `decompose`; each minimal prime is reported as a component.
    M2Decompose,
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
            "singular-primdecGTZ" | "primdecGTZ" => Some(Self::SingularPrimdecGtz),
            "singular-gb-sat" | "singular-sat" => Some(Self::SingularGbSat),
            "singular-facstd-sat" | "facstd-sat" => Some(Self::SingularFacstdSat),
            "m2-decompose" => Some(Self::M2Decompose),
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
            Self::SingularPrimdecGtz => "singular-primdecGTZ",
            Self::SingularGbSat => "singular-gb-sat",
            Self::SingularFacstdSat => "singular-facstd-sat",
            Self::M2Decompose => "m2-decompose",
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
        .variables()
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

/// The ring and the cover's ideal `I`, without its Rabinowitsch generators
/// when `saturate`, and those generators with the positions of their
/// temporaries.
fn singular_declarations(
    cover: &[PolyExtLex],
    saturate: bool,
) -> (String, Vec<(usize, &PolyExtLex)>) {
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
    let prelude = if rabinowitsch.is_empty() {
        ""
    } else {
        "LIB \"elim.lib\";\n"
    };
    (
        format!("{prelude}ring r=0,({variables}),dp;\nideal I={generators};\ndef S;\n"),
        rabinowitsch,
    )
}

/// Statements replacing the ideal `name` by its saturation `name : h^oo` for
/// each Rabinowitsch generator `c*(1 - s*h)`, one at a time (Singular `sat`,
/// applied to `diff(g,s) = -c*h`), and then setting every `s` to zero, so the
/// dimension over all ring variables equals the dimension over the original
/// ones.
fn saturation_statements(name: &str, rabinowitsch: &[(usize, &PolyExtLex)]) -> String {
    if rabinowitsch.is_empty() {
        return String::new();
    }
    let saturations: String = rabinowitsch
        .iter()
        .map(|(position, generator)| {
            format!(
                "S=sat({name},ideal(diff({},x{position})));\n\
                 if(typeof(S)==\"list\"){{{name}=S[1];}}else{{{name}=S;}}\n",
                singular_polynomial(generator)
            )
        })
        .collect();
    let pinned = rabinowitsch
        .iter()
        .map(|(position, _)| format!("x{position}"))
        .collect::<Vec<_>>()
        .join(",");
    format!("{saturations}{name}={name}+ideal({pinned});\n")
}

/// Declare the ring and the cover's ideal `I`. With `saturate`, each
/// Rabinowitsch generator is replaced by the saturation of `I` by its
/// polynomial (see [`saturation_statements`]).
fn singular_ideal(cover: &[PolyExtLex], saturate: bool) -> String {
    let (declarations, rabinowitsch) = singular_declarations(cover, saturate);
    format!(
        "{declarations}{}",
        saturation_statements("I", &rabinowitsch)
    )
}

fn singular_program(cover: &[PolyExtLex], saturate: bool) -> String {
    let ideal = singular_ideal(cover, saturate);
    format!(
        "{ideal}\
         option(redSB);\n\
         ideal G=std(I);\n\
         print(\"SINGULAR_UNIT \"+string(reduce(1,G)==0));\n\
         int i;\n\
         for(i=1;i<=size(G);i=i+1){{print(\"SINGULAR_LEAD \"+string(leadexp(G[i])));}};\n\
         print(\"SINGULAR_DONE\");\n\
         exit;\n"
    )
}

/// How a component-reporting Singular backend splits the cover's ideal.
#[derive(Clone, Copy)]
enum SingularSplitter {
    /// Gräbe's factorizing Gröbner bases after `interred`.
    Facstd,
    /// Primary decomposition; a component is represented by its associated
    /// prime, which has the same zero set.
    PrimdecGtz,
}

/// A program printing each component of the cover's ideal. With `saturate`,
/// `facstd` splits the generators other than the Rabinowitsch ones first and
/// each component is then saturated, since `V(I) \\ V(h)` is the union of the
/// `V(F) \\ V(h)`; a component whose saturation is `⟨1⟩` reports as a unit.
/// `primdecGTZ` decomposes the saturated ideal instead.
fn singular_components_program(
    cover: &[PolyExtLex],
    saturate: bool,
    splitter: SingularSplitter,
) -> String {
    let (declarations, rabinowitsch) = singular_declarations(cover, saturate);
    let (before_split, split, component, after_split) = match splitter {
        SingularSplitter::Facstd => (
            String::new(),
            "ideal FI=interred(I);\nlist F=facstd(FI);\n",
            "F[i]",
            saturation_statements("J", &rabinowitsch),
        ),
        SingularSplitter::PrimdecGtz => (
            saturation_statements("I", &rabinowitsch),
            "LIB \"primdec.lib\";\nlist F=primdecGTZ(I);\n",
            "F[i][2]",
            String::new(),
        ),
    };
    format!(
        "{declarations}{before_split}\
         {split}\
         int i; ideal J; ideal G;\n\
         for(i=1;i<=size(F);i=i+1){{\n\
           print(\"SINGULAR_COMPONENT\");\n\
           J={component};\n\
           {after_split}\
           G=std(J);\n\
           print(\"SINGULAR_UNIT \"+string(reduce(1,G)==0));\n\
           int j;\n\
           for(j=1;j<=size(G);j=j+1){{print(\"SINGULAR_LEAD \"+string(leadexp(G[j])));}};\n\
           print(\"SINGULAR_END_COMPONENT\");\n\
         }};\n\
         print(\"SINGULAR_DONE\");\n\
         exit;\n"
    )
}

/// The external computer-algebra system a child-process backend runs.
#[derive(Clone, Copy)]
enum ExternalTool {
    Singular,
    Macaulay2,
}

impl ExternalTool {
    fn name(self) -> &'static str {
        match self {
            Self::Singular => "Singular",
            Self::Macaulay2 => "Macaulay2",
        }
    }

    /// A command that reads its program from standard input. Macaulay2 echoes
    /// each input line behind an `iN :` prompt, which never matches a marker line.
    fn command(self) -> Command {
        match self {
            Self::Singular => {
                let mut command = Command::new("Singular");
                command.args(["-q", "--no-tty"]);
                command
            }
            Self::Macaulay2 => {
                let mut command = Command::new("M2");
                command.args(["--no-readline", "--silent", "--stop", "-q"]);
                command
            }
        }
    }
}

/// The cover's minimal primes from Macaulay2's `decompose`, printed with the
/// marker lines of [`singular_components_program`].
fn m2_components_program(cover: &[PolyExtLex]) -> String {
    let variables = (0..cover[0].nvars())
        .map(|index| format!("x{index}"))
        .collect::<Vec<_>>()
        .join(",");
    let generators = cover
        .iter()
        .map(singular_polynomial)
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "R = QQ[{variables}, MonomialOrder => GRevLex];\n\
         I = ideal({generators});\n\
         C = decompose I;\n\
         scan(C, P -> (print \"SINGULAR_COMPONENT\"; G = gens gb P; \
         scan(flatten entries G, g -> print(\"SINGULAR_LEAD \" | \
         demark(\",\", apply(first exponents leadMonomial g, toString)))); \
         print \"SINGULAR_END_COMPONENT\"));\n\
         print \"SINGULAR_DONE\";\n\
         exit 0;\n"
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

fn run_singular(
    tool: ExternalTool,
    program: &str,
    timeout: Duration,
) -> Result<SingularRun, String> {
    let deadline = Instant::now() + timeout;
    let Some(_permit) = crate::solver_process::acquire_solver_process(deadline) else {
        return Ok(SingularRun::Timeout);
    };
    let name = tool.name();
    let mut child = tool
        .command()
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("could not start {name}: {error}"))?;
    let mut stdout = child.stdout.take().expect("piped solver stdout");
    let mut stderr = child.stderr.take().expect("piped solver stderr");
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
            "could not send input to {name}: {error}; stdout={stdout:?}; stderr={stderr:?}"
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
                    "failed while waiting for {name}: {error}; stdout={stdout:?}; stderr={stderr:?}"
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

/// Run a program that ends by printing `SINGULAR_DONE` and return its
/// standard output, or the outcome that ends the evaluation early.
fn completed_singular_stdout(
    tool: ExternalTool,
    program: &str,
    timeout: Duration,
) -> Result<String, Outcome> {
    match run_singular(tool, program, timeout) {
        Ok(SingularRun::Completed { status, stdout, .. })
            if status.success() && stdout.lines().any(|line| line.trim() == "SINGULAR_DONE") =>
        {
            Ok(stdout)
        }
        Ok(SingularRun::Completed {
            status,
            stdout,
            stderr,
        }) => Err(Outcome::BackendError(format!(
            "{} exited with {status}; stdout={stdout:?}; stderr={stderr:?}",
            tool.name()
        ))),
        Ok(SingularRun::Timeout) => Err(Outcome::Timeout),
        Err(error) => Err(Outcome::BackendError(error)),
    }
}

/// Decode one `leadexp` exponent vector into a monomial of `template`'s ring.
fn leading_monomial(template: &PolyExtGrevLex, encoded: &str) -> Result<PolyExtGrevLex, Outcome> {
    encoded
        .split(',')
        .map(str::parse::<u16>)
        .collect::<Result<Vec<_>, _>>()
        .ok()
        .filter(|exponents| exponents.len() == template.nvars())
        .map(|exponents| template.monomial(template.ring().one(), exponents))
        .ok_or_else(|| {
            Outcome::BackendError(format!(
                "invalid Singular leading exponent vector {encoded:?}"
            ))
        })
}

/// Run Singular in a child process so a timeout can terminate the actual
/// computation. Only the basis length and leading-monomial ideal are needed,
/// so the compact `leadexp` output is reconstructed as monomials in the
/// original ring.
fn evaluate_cover_singular(cover: Vec<PolyExtLex>, timeout: Duration, saturate: bool) -> Outcome {
    let program = singular_program(&cover, saturate);
    let stdout = match completed_singular_stdout(ExternalTool::Singular, &program, timeout) {
        Ok(stdout) => stdout,
        Err(outcome) => return outcome,
    };
    if stdout.lines().any(|line| line.trim() == "SINGULAR_UNIT 1") {
        return Outcome::Inconsistent;
    }
    let template = cover[0].reorder::<GrevLexOrder>();
    stdout
        .lines()
        .filter_map(|line| line.trim().strip_prefix("SINGULAR_LEAD "))
        .map(|encoded| leading_monomial(&template, encoded))
        .collect::<Result<Vec<_>, _>>()
        .map_or_else(
            |outcome| outcome,
            |elements| {
                Outcome::Consistent(ComputedBasis {
                    elements,
                    leading_monomials_only: true,
                })
            },
        )
}

/// Like [`evaluate_cover_singular`], but runs a `program` of `tool` that splits
/// the ideal and reports the leading-monomial ideal of every nonunit component.
fn evaluate_cover_components(
    cover: Vec<PolyExtLex>,
    timeout: Duration,
    tool: ExternalTool,
    program: &str,
) -> Outcome {
    let stdout = match completed_singular_stdout(tool, program, timeout) {
        Ok(stdout) => stdout,
        Err(outcome) => return outcome,
    };
    let template = cover[0].reorder::<GrevLexOrder>();
    let mut components = Vec::new();
    let mut current = None;
    let mut current_is_unit = false;
    for line in stdout.lines().map(str::trim) {
        match line {
            "SINGULAR_COMPONENT" => {
                current = Some(Vec::new());
                current_is_unit = false;
            }
            "SINGULAR_UNIT 1" => current_is_unit = true,
            "SINGULAR_END_COMPONENT" => {
                let Some(elements) = current.take() else {
                    return Outcome::BackendError(
                        "the decomposition backend ended a component before starting it"
                            .to_string(),
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
                let Some(encoded) = line.strip_prefix("SINGULAR_LEAD ") else {
                    continue;
                };
                let monomial = match leading_monomial(&template, encoded) {
                    Ok(monomial) => monomial,
                    Err(outcome) => return outcome,
                };
                let Some(elements) = current.as_mut() else {
                    return Outcome::BackendError(
                        "the decomposition backend reported a basis outside a component"
                            .to_string(),
                    );
                };
                elements.push(monomial);
            }
        }
    }
    if current.is_some() {
        return Outcome::BackendError(
            "the decomposition backend left a component unfinished".to_string(),
        );
    }
    if components.is_empty() {
        Outcome::Inconsistent
    } else {
        Outcome::ConsistentComponents(components)
    }
}

/// Outcome of saturating an ideal by a product of polynomials.
pub enum Saturation {
    /// Reduced degrevlex Gröbner basis of the saturation, in the input ring.
    Basis(Vec<PolyExtLex>),
    /// The saturation is the whole ring.
    Unit,
    Timeout,
    Error(String),
}

/// Singular program printing every term of the reduced degrevlex basis of
/// `⟨generators⟩ : (∏ divisors)^∞`, computed by Singular's `sat` one divisor
/// at a time, since `I : (d_1 d_2)^∞ = (I : d_1^∞) : d_2^∞` and Singular
/// saturates by a single variable far faster than by a product. All
/// polynomials share one ring.
fn singular_saturation_program(generators: &[PolyExtLex], divisors: &[PolyExtLex]) -> String {
    let variables = (0..generators[0].nvars())
        .map(|index| format!("x{index}"))
        .collect::<Vec<_>>()
        .join(",");
    let ideal = generators
        .iter()
        .map(singular_polynomial)
        .collect::<Vec<_>>()
        .join(",");
    let saturations: String = divisors
        .iter()
        .map(|divisor| {
            format!(
                "S=sat(I,ideal({}));\nif(typeof(S)==\"list\"){{I=S[1];}}else{{I=S;}}\n",
                singular_polynomial(divisor)
            )
        })
        .collect();
    format!(
        "LIB \"elim.lib\";\n\
         ring r=0,({variables}),dp;\n\
         ideal I={ideal};\n\
         def S;\n\
         {saturations}\
         option(redSB);\n\
         I=std(I);\n\
         print(\"SINGULAR_UNIT \"+string(reduce(1,I)==0));\n\
         int i; int k;\n\
         for(i=1;i<=size(I);i=i+1){{\n\
           print(\"SINGULAR_POLY\");\n\
           for(k=1;k<=size(I[i]);k=k+1){{print(\"SINGULAR_TERM \"+string(leadcoef(I[i][k]))+\" \"+string(leadexp(I[i][k])));}};\n\
         }};\n\
         print(\"SINGULAR_DONE\");\n\
         exit;\n"
    )
}

/// Decode a Singular `SINGULAR_TERM` payload `c e1,...,en`, where `c` is an
/// integer or a fraction `p/q`.
fn singular_term(encoded: &str, nvars: usize) -> Result<(Rational, Vec<u16>), String> {
    let invalid = || format!("invalid Singular term {encoded:?}");
    let (coefficient, exponents) = encoded.split_once(' ').ok_or_else(invalid)?;
    let (numerator, denominator) = coefficient.split_once('/').unwrap_or((coefficient, "1"));
    let coefficient = Rational::from((
        numerator.parse::<Integer>().map_err(|_| invalid())?,
        denominator.parse::<Integer>().map_err(|_| invalid())?,
    ));
    let exponents = exponents
        .split(',')
        .map(str::parse::<u16>)
        .collect::<Result<Vec<_>, _>>()
        .ok()
        .filter(|exponents| exponents.len() == nvars)
        .ok_or_else(invalid)?;
    Ok((coefficient, exponents))
}

/// The polynomials printed by [`singular_saturation_program`], in the ring of
/// `template`.
fn singular_basis(template: &PolyExtLex, stdout: &str) -> Result<Vec<PolyExtLex>, String> {
    stdout
        .lines()
        .map(str::trim)
        .try_fold(Vec::new(), |mut basis: Vec<PolyExtLex>, line| {
            if line == "SINGULAR_POLY" {
                basis.push(template.zero());
            } else if let Some(term) = line.strip_prefix("SINGULAR_TERM ") {
                let (coefficient, exponents) = singular_term(term, template.nvars())?;
                basis
                    .last_mut()
                    .ok_or_else(|| "Singular printed a term outside a polynomial".to_string())?
                    .append_monomial(coefficient, &exponents);
            }
            Ok(basis)
        })
}

/// Compute `⟨generators⟩ : (∏ divisors)^∞` with Singular's `sat` in a killable
/// child process. The basis is returned in the ring of the first generator,
/// extended by any variable that only a divisor uses.
pub fn saturate(
    generators: &[PolyExtLex],
    divisors: &[PolyExtLex],
    timeout: Duration,
) -> Saturation {
    if generators.is_empty() || divisors.is_empty() {
        return Saturation::Error("saturation needs generators and divisors".to_string());
    }
    let mut ring: Vec<PolyExtLex> = generators.iter().chain(divisors).cloned().collect();
    MultivariatePolynomial::unify_variables_list(&mut ring);
    let (generators, divisors) = ring.split_at(generators.len());
    let program = singular_saturation_program(generators, divisors);
    let stdout = match completed_singular_stdout(ExternalTool::Singular, &program, timeout) {
        Ok(stdout) => stdout,
        Err(Outcome::Timeout) => return Saturation::Timeout,
        Err(Outcome::BackendError(error)) => return Saturation::Error(error),
        Err(_) => return Saturation::Error("unexpected Singular outcome".to_string()),
    };
    if stdout.lines().any(|line| line.trim() == "SINGULAR_UNIT 1") {
        return Saturation::Unit;
    }
    singular_basis(&generators[0], &stdout).map_or_else(Saturation::Error, Saturation::Basis)
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
    let singular_components = |cover: Vec<PolyExtLex>, saturate, splitter| {
        let program = singular_components_program(&cover, saturate, splitter);
        evaluate_cover_components(cover, timeout, ExternalTool::Singular, &program)
    };
    match backend {
        GbBackend::SingularGb => return evaluate_cover_singular(cover, timeout, false),
        GbBackend::SingularFacstd => {
            return singular_components(cover, false, SingularSplitter::Facstd);
        }
        GbBackend::SingularPrimdecGtz => {
            return singular_components(cover, false, SingularSplitter::PrimdecGtz);
        }
        GbBackend::SingularGbSat => return evaluate_cover_singular(cover, timeout, true),
        GbBackend::SingularFacstdSat => {
            return singular_components(cover, true, SingularSplitter::Facstd);
        }
        GbBackend::M2Decompose => {
            let program = m2_components_program(&cover);
            return evaluate_cover_components(cover, timeout, ExternalTool::Macaulay2, &program);
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
        | GbBackend::SingularPrimdecGtz
        | GbBackend::SingularGbSat
        | GbBackend::SingularFacstdSat
        | GbBackend::M2Decompose => {
            return Outcome::BackendError(
                "Singular and Macaulay2 cannot run through the in-process worker entry point"
                    .to_string(),
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

    use symbolica::atom::AtomCore;
    use symbolica::domains::rational::Q;
    use symbolica::license::LicenseManager;

    use crate::atom_util::parse_expression;
    use crate::factor_prep::{ideal_ext_lex, variable_order};
    use crate::mathematica::load_mathematica_system_from_str_with_var_names;

    static LICENSE: Once = Once::new();

    fn activate_license() {
        LICENSE.call_once(|| {
            let _ = LicenseManager::set_license_key(crate::SYMBOLICA_LICENSE_KEY);
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
            .try_to_polynomial::<_, u16>(template.ring(), Some(template.variables().clone()))
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

    #[test]
    fn singular_primdec_backend_reports_associated_primes() {
        if Command::new("Singular").arg("--version").output().is_err() {
            return;
        }

        match evaluate_cover(
            vec![test_poly("x*y"), test_poly("x^2")],
            Duration::from_secs(5),
            GbBackend::SingularPrimdecGtz,
        ) {
            Outcome::ConsistentComponents(components) => {
                let dimensions: Vec<Option<usize>> = components
                    .iter()
                    .map(|basis| solution_dimension(&basis.elements, 2))
                    .collect();
                assert_eq!(dimensions, vec![Some(1), Some(0)]);
            }
            _ => panic!("primdecGTZ should report the prime (x) and the embedded prime (x, y)"),
        }
        assert!(matches!(
            evaluate_cover(
                vec![test_poly("x"), test_poly("x-1")],
                Duration::from_secs(5),
                GbBackend::SingularPrimdecGtz,
            ),
            Outcome::Inconsistent
        ));
    }

    #[test]
    fn m2_decompose_backend_reports_minimal_primes() {
        if Command::new("M2").arg("--version").output().is_err() {
            return;
        }

        match evaluate_cover(
            vec![test_poly("x*y"), test_poly("x^2-x")],
            Duration::from_secs(30),
            GbBackend::M2Decompose,
        ) {
            Outcome::ConsistentComponents(components) => {
                let mut dimensions: Vec<Option<usize>> = components
                    .iter()
                    .map(|basis| solution_dimension(&basis.elements, 2))
                    .collect();
                dimensions.sort();
                assert_eq!(dimensions, vec![Some(0), Some(1)]);
            }
            _ => panic!("decompose should report the primes (x) and (x-1, y)"),
        }
        assert!(matches!(
            evaluate_cover(
                vec![test_poly("x"), test_poly("x-1")],
                Duration::from_secs(30),
                GbBackend::M2Decompose,
            ),
            Outcome::Inconsistent
        ));
    }
}
