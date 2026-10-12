//! Nonzero variables: conditions `x != 0` on the input, imposed by
//! saturation.
//!
//! A system `P` whose variables in `N` must not vanish describes
//! `V(P) \ V(∏ N)`, whose closure is the zero set of the saturation
//! `⟨P⟩ : (∏ N)^∞`. Factor branching and variable zeroing start from
//! `Saturate(P, N)` (`paper/sections/variable_zeroing.tex`): that saturation
//! when Singular computes it within the limit, and `P` otherwise. Refinement
//! treats every `x` in `N` as a nonzero side condition, and the branch solver
//! receives each terminal hitting set with one Rabinowitsch equation
//! `1 - s*x` per `x`, since the Gröbner backends cannot saturate.

use std::time::Duration;

use symbolica::atom::AtomCore;
use symbolica::domains::rational::{Q, Rational};
use symbolica::poly::{MonomialOrder, PolyVariable, polynomial::MultivariatePolynomial};

use crate::atom_util::parse_expression;
use crate::factor_prep::PolyExtLex;
use crate::gb_search::{Saturation, saturate};
use crate::mathematica::LoadedMathematicaSystem;
use crate::var_id::VarId;

/// The variables of `loaded` named in `names`, in the order of `names`.
///
/// # Errors
///
/// A name that is not a variable of the system.
///
/// # Examples
///
/// ```no_run
/// use polynomial_cdcl::load_mathematica_system_from_str_with_var_names;
/// use polynomial_cdcl::nonzero::nonzero_variables;
///
/// let loaded = load_mathematica_system_from_str_with_var_names("{x*y == 0}").unwrap();
/// let nonzero = nonzero_variables(&loaded, &["x".to_string()]).unwrap();
/// assert_eq!(nonzero.len(), 1);
/// ```
pub fn nonzero_variables(
    loaded: &LoadedMathematicaSystem,
    names: &[String],
) -> Result<Vec<PolyVariable>, String> {
    names
        .iter()
        .map(|name| {
            loaded
                .var_names
                .iter()
                .position(|candidate| candidate == name)
                .map(|index| PolyVariable::from(loaded.id_to_sym[index]))
                .ok_or_else(|| format!("{name:?} is not a variable of the system"))
        })
        .collect()
}

/// The monic polynomials `x` of the variables in `nonzero`, in the ring of
/// `template`: the side conditions that refinement keeps nonzero.
///
/// # Errors
///
/// A variable that the ring of `template` cannot represent.
///
/// # Examples
///
/// ```no_run
/// use polynomial_cdcl::nonzero::{condition_polynomials, nonzero_variables};
/// use polynomial_cdcl::{load_mathematica_system_from_str_with_var_names, polynomials_from_loaded};
/// use symbolica::domains::rational::Q;
///
/// let loaded = load_mathematica_system_from_str_with_var_names("{x*y == 0}").unwrap();
/// let polynomials = polynomials_from_loaded(&loaded, &Q).unwrap();
/// let nonzero = nonzero_variables(&loaded, &["x".to_string()]).unwrap();
/// let conditions = condition_polynomials(&polynomials[0], &nonzero).unwrap();
/// assert_eq!(conditions.len(), 1);
/// ```
pub fn condition_polynomials(
    template: &PolyExtLex,
    nonzero: &[PolyVariable],
) -> Result<Vec<PolyExtLex>, String> {
    nonzero
        .iter()
        .map(|variable| difference(template, variable, 0).map(PolyExtLex::make_monic))
        .collect()
}

/// `⟨polynomials⟩ : (∏ nonzero)^∞`, computed by Singular's `sat` within
/// `timeout`; zero polynomials are ignored.
///
/// # Examples
///
/// ```no_run
/// use std::time::Duration;
///
/// use polynomial_cdcl::gb_search::Saturation;
/// use polynomial_cdcl::nonzero::{nonzero_variables, saturate_by_variables};
/// use polynomial_cdcl::{load_mathematica_system_from_str_with_var_names, polynomials_from_loaded};
/// use symbolica::domains::rational::Q;
///
/// let loaded = load_mathematica_system_from_str_with_var_names("{x*y == 0}").unwrap();
/// let polynomials = polynomials_from_loaded(&loaded, &Q).unwrap();
/// let nonzero = nonzero_variables(&loaded, &["x".to_string()]).unwrap();
/// let saturation = saturate_by_variables(&polynomials, &nonzero, Duration::from_secs(5));
/// assert!(matches!(saturation, Saturation::Basis(basis) if basis.len() == 1));
/// ```
pub fn saturate_by_variables(
    polynomials: &[PolyExtLex],
    nonzero: &[PolyVariable],
    timeout: Duration,
) -> Saturation {
    let generators: Vec<PolyExtLex> = polynomials
        .iter()
        .filter(|polynomial| !polynomial.is_zero())
        .cloned()
        .collect();
    let Some(template) = generators.first() else {
        return Saturation::Error("saturation needs a nonzero generator".to_string());
    };
    condition_polynomials(template, nonzero).map_or_else(Saturation::Error, |divisors| {
        saturate(&generators, &divisors, timeout)
    })
}

/// The branch solver's input for `cover` with the variables in `nonzero`
/// nonzero: `cover` and one Rabinowitsch equation `1 - s_i*x_i` per
/// variable, `s_i` a temporary variable. Without nonzero variables, `cover`
/// itself.
///
/// # Errors
///
/// A variable that the ring of `cover` cannot represent.
///
/// # Examples
///
/// ```no_run
/// use polynomial_cdcl::nonzero::{nonzero_variables, with_nonzero_equations};
/// use polynomial_cdcl::{load_mathematica_system_from_str_with_var_names, polynomials_from_loaded};
/// use symbolica::domains::rational::Q;
///
/// let loaded = load_mathematica_system_from_str_with_var_names("{x*y == 0}").unwrap();
/// let polynomials = polynomials_from_loaded(&loaded, &Q).unwrap();
/// let nonzero = nonzero_variables(&loaded, &["x".to_string()]).unwrap();
/// let input = with_nonzero_equations(polynomials, &nonzero).unwrap();
/// assert_eq!(input.len(), 2);
/// ```
pub fn with_nonzero_equations(
    cover: Vec<PolyExtLex>,
    nonzero: &[PolyVariable],
) -> Result<Vec<PolyExtLex>, String> {
    if nonzero.is_empty() || cover.is_empty() {
        return Ok(cover);
    }
    let conditions: Vec<(PolyVariable, i64)> = nonzero
        .iter()
        .map(|variable| (variable.clone(), 0))
        .collect();
    with_rabinowitsch_equations(cover, &conditions, true)
}

/// `loaded` with one Rabinowitsch equation `1 - s*x` in a new variable `s`
/// for each variable `x` named in `names`: the input of a Gröbner-basis
/// computation of `⟨P⟩ : (∏ x)^∞`, whose dimension and emptiness it shares.
/// The variable of `x` is named `rabinowitsch_x` and comes last.
///
/// # Errors
///
/// A name that is not a variable of the system, or whose new variable name
/// is taken.
///
/// # Examples
///
/// ```no_run
/// use polynomial_cdcl::load_mathematica_system_from_str_with_var_names;
/// use polynomial_cdcl::nonzero::with_rabinowitsch_variables;
///
/// let loaded = load_mathematica_system_from_str_with_var_names("{x*y == 0}").unwrap();
/// let extended = with_rabinowitsch_variables(loaded, &["x".to_string()]).unwrap();
/// assert_eq!(extended.system.equations.len(), 2);
/// assert_eq!(extended.var_names.last().unwrap(), "rabinowitsch_x");
/// ```
pub fn with_rabinowitsch_variables(
    loaded: LoadedMathematicaSystem,
    names: &[String],
) -> Result<LoadedMathematicaSystem, String> {
    names.iter().try_fold(loaded, |mut loaded, name| {
        let inverse = format!("rabinowitsch_{name}");
        if !loaded.var_names.contains(name) {
            return Err(format!("{name:?} is not a variable of the system"));
        }
        if loaded.var_names.contains(&inverse) {
            return Err(format!("the system already has a variable {inverse:?}"));
        }
        let symbol = parse_expression(inverse.clone())?
            .get_symbol()
            .ok_or_else(|| format!("{inverse:?} is not a symbol"))?;
        let equation = parse_expression(format!("1-{inverse}*{name}"))?;
        loaded
            .sym_to_vid
            .insert(symbol, VarId(loaded.var_names.len() as u32));
        loaded.var_names.push(inverse);
        loaded.id_to_sym.push(symbol);
        loaded.system.equations.push(equation);
        Ok(loaded)
    })
}

/// The Rabinowitsch-free generators of `candidate` together with Rabinowitsch
/// equations for `inequality_conditions`: one per condition, or one for
/// their product.
pub(crate) fn with_rabinowitsch_equations(
    candidate: Vec<PolyExtLex>,
    inequality_conditions: &[(PolyVariable, i64)],
    saturate_each_variable: bool,
) -> Result<Vec<PolyExtLex>, String> {
    let mut unequal_branch: Vec<_> = candidate
        .into_iter()
        .filter(|polynomial| !uses_temporary_variable(polynomial))
        .collect();
    if unequal_branch.is_empty() {
        return Err("cannot rebuild saturation from an empty cover".to_string());
    }
    let saturation_variable_count = if saturate_each_variable {
        inequality_conditions.len()
    } else {
        1
    };
    for index in 0..saturation_variable_count {
        let saturation_variable = PolyVariable::Temporary(index);
        for polynomial in &mut unequal_branch {
            polynomial.add_variable(&saturation_variable);
        }
    }

    let template = unequal_branch
        .first()
        .cloned()
        .expect("the inequality branch retains the nonempty cover");
    if saturate_each_variable {
        for (index, (variable, value)) in inequality_conditions.iter().enumerate() {
            let inverse = template.variable(&PolyVariable::Temporary(index))?;
            let product = inverse * &difference(&template, variable, *value)?;
            unequal_branch.push(template.one() - product);
        }
    } else {
        let mut product = template.variable(&PolyVariable::Temporary(0))?;
        for (variable, value) in inequality_conditions {
            product = product * &difference(&template, variable, *value)?;
        }
        unequal_branch.push(template.one() - product);
    }
    Ok(unequal_branch)
}

/// The polynomial `x - a` in the ring of `template`.
pub(crate) fn difference(
    template: &PolyExtLex,
    variable: &PolyVariable,
    value: i64,
) -> Result<PolyExtLex, String> {
    Ok(template.variable(variable)? - template.constant(Rational::from(value)))
}

pub(crate) fn uses_variable<O: MonomialOrder>(
    polynomial: &MultivariatePolynomial<Q, u16, O>,
    variable: &PolyVariable,
) -> bool {
    let Some(position) = polynomial
        .variables()
        .iter()
        .position(|candidate| candidate == variable)
    else {
        return false;
    };
    polynomial
        .exponents_iter()
        .any(|exponents| exponents[position] != 0)
}

pub(crate) fn uses_temporary_variable<O: MonomialOrder>(
    polynomial: &MultivariatePolynomial<Q, u16, O>,
) -> bool {
    polynomial
        .variables()
        .iter()
        .filter(|variable| matches!(variable, PolyVariable::Temporary(_)))
        .any(|variable| uses_variable(polynomial, variable))
}

#[cfg(test)]
mod tests {
    use std::sync::Once;

    use symbolica::license::LicenseManager;

    use super::*;
    use crate::{load_mathematica_system_from_str_with_var_names, polynomials_from_loaded};

    static LICENSE: Once = Once::new();

    fn load(source: &str) -> (LoadedMathematicaSystem, Vec<PolyExtLex>) {
        LICENSE.call_once(|| {
            let _ = LicenseManager::set_license_key(crate::SYMBOLICA_LICENSE_KEY);
        });
        let loaded = load_mathematica_system_from_str_with_var_names(source).expect("load");
        let polynomials = polynomials_from_loaded(&loaded, &Q).expect("polynomials");
        (loaded, polynomials)
    }

    fn names(names: &[&str]) -> Vec<String> {
        names.iter().map(|name| (*name).to_string()).collect()
    }

    fn singular_missing() -> bool {
        std::process::Command::new("Singular")
            .arg("--version")
            .output()
            .is_err()
    }

    #[test]
    fn nonzero_variables_are_looked_up_by_name() {
        let (loaded, _) = load("{x*y == 0}");

        let nonzero = nonzero_variables(&loaded, &names(&["y"])).expect("variables");

        assert_eq!(nonzero.len(), 1);
        assert_eq!(nonzero[0], PolyVariable::from(loaded.id_to_sym[1]));
        assert!(nonzero_variables(&loaded, &names(&["w"])).is_err());
    }

    #[test]
    fn solver_input_gets_one_rabinowitsch_equation_per_nonzero_variable() {
        let (loaded, polynomials) = load("{x*y == 0}");
        let nonzero = nonzero_variables(&loaded, &names(&["x", "y"])).expect("variables");

        let input = with_nonzero_equations(polynomials.clone(), &nonzero).expect("input");

        assert_eq!(input.len(), 3);
        assert_eq!(
            input.iter().filter(|p| uses_temporary_variable(p)).count(),
            2
        );
        assert_eq!(
            with_nonzero_equations(polynomials.clone(), &[]).expect("input"),
            polynomials
        );
    }

    #[test]
    fn rabinowitsch_variables_extend_the_loaded_system() {
        let (loaded, _) = load("{x*y == 0}");

        let extended =
            with_rabinowitsch_variables(loaded.clone(), &names(&["x"])).expect("extended");

        assert_eq!(extended.system.equations.len(), 2);
        assert_eq!(extended.var_names, names(&["x", "y", "rabinowitsch_x"]));
        assert_eq!(extended.id_to_sym.len(), 3);
        assert!(with_rabinowitsch_variables(loaded, &names(&["w"])).is_err());
    }

    #[test]
    fn saturation_by_a_variable_divides_it_out() {
        if singular_missing() {
            return;
        }
        let (loaded, polynomials) = load("{x*y == 0, x^2*z == 0}");
        let x = nonzero_variables(&loaded, &names(&["x"])).expect("variables");

        let Saturation::Basis(basis) =
            saturate_by_variables(&polynomials, &x, Duration::from_secs(5))
        else {
            panic!("saturation did not return a basis");
        };

        assert_eq!(basis.len(), 2);
        assert!(
            basis
                .iter()
                .all(|polynomial| !uses_variable(polynomial, &x[0]))
        );
    }

    #[test]
    fn saturation_is_the_unit_ideal_when_the_variable_vanishes() {
        if singular_missing() {
            return;
        }
        let (loaded, polynomials) = load("{x*y == 0, y - 1 == 0}");
        let x = nonzero_variables(&loaded, &names(&["x"])).expect("variables");

        let saturation = saturate_by_variables(&polynomials, &x, Duration::from_secs(5));

        assert!(matches!(saturation, Saturation::Unit));
    }
}
