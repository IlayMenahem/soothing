//! Sparse-variable candidate resolution and polynomial structure hints used by
//! the outer-branching search.

use symbolica::atom::Symbol;
use symbolica::domains::Ring;
use symbolica::poly::PolyVariable;

use crate::factor_prep::PolyExtLex;
use crate::mathematica::LoadedMathematicaSystem;

/// Resolve the requested `--sparse` candidate variable names to their symbols.
/// An empty list means "all variables" (bare `--sparse`); a non-empty list is
/// resolved against the loader's names, with an unknown name an error.
pub fn resolve_candidates(
    loaded: &LoadedMathematicaSystem,
    names: &[String],
) -> Result<Vec<(String, Symbol)>, String> {
    if names.is_empty() {
        return Ok(loaded
            .var_names
            .iter()
            .cloned()
            .zip(loaded.id_to_sym.iter().copied())
            .collect());
    }
    names
        .iter()
        .map(|name| {
            loaded
                .var_names
                .iter()
                .position(|n| n == name)
                .map(|i| (name.clone(), loaded.id_to_sym[i]))
                .ok_or_else(|| format!("unknown variable {name:?} for --sparse"))
        })
        .collect()
}

/// Resolve sparse candidates to variable positions in `template`'s variable map.
/// A variable absent from the map imposes no substitution and is skipped.
pub fn usable_sparse_variables(
    template: &PolyExtLex,
    candidates: &[(String, Symbol)],
) -> Vec<(String, usize)> {
    candidates
        .iter()
        .filter_map(|(name, sym)| {
            let variable = PolyVariable::from(*sym);
            template
                .variables
                .iter()
                .position(|var| *var == variable)
                .map(|position| (name.clone(), position))
        })
        .collect()
}

/// The polynomial `x_position` in the same ring and variable map as `template`.
fn variable_generator(template: &PolyExtLex, position: usize) -> PolyExtLex {
    let mut exponents = vec![0u16; template.nvars()];
    exponents[position] = 1;
    let mut out = template.zero_with_capacity(1);
    out.append_monomial_back(template.ring.one(), &exponents);
    out
}

/// Convert a sparse-variable subset into its structure-hint polynomials
/// `x_i = 0`, in the same ring and variable map as `template`.
pub fn decode_hint_mask(
    template: &PolyExtLex,
    sparse_variables: &[(String, usize)],
    hint_mask: u64,
) -> Vec<PolyExtLex> {
    sparse_variables
        .iter()
        .enumerate()
        .filter(|(i, _)| hint_mask & (1u64 << i) != 0)
        .map(|(_, (_, position))| variable_generator(template, *position))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Once;

    use symbolica::LicenseManager;
    use symbolica::atom::AtomCore;
    use symbolica::domains::algebraic_number::AlgebraicExtension;
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
        let field = AlgebraicExtension::new_complex(Q);
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

    #[test]
    fn decode_hint_mask_builds_selected_variable_generators() {
        let template = test_poly("x*y");
        let sparse_variables = vec![("x".to_string(), 0), ("y".to_string(), 1)];
        let decoded = decode_hint_mask(&template, &sparse_variables, 0b11);
        let expected = [poly_like(&template, "x"), poly_like(&template, "y")];
        assert_eq!(decoded.len(), 2);
        assert!(expected.iter().all(|e| decoded.contains(e)));
    }
}
