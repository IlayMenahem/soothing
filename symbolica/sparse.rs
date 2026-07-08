//! Sparse-variable candidate resolution and zero-substitution used by the
//! CDCL search to try forcing variables to zero alongside a factor cover.

use symbolica::atom::Symbol;
use symbolica::domains::Ring;
use symbolica::poly::PolyVariable;

use crate::cover::{WorkItem, decode_cover};
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

fn zero_positions<'a>(
    sparse_variables: &'a [(String, usize)],
    zero_mask: u64,
) -> impl Iterator<Item = usize> + 'a {
    sparse_variables
        .iter()
        .enumerate()
        .filter(move |(i, _)| zero_mask & (1u64 << i) != 0)
        .map(|(_, (_, position))| *position)
}

fn substitute_zeroes(poly: &PolyExtLex, zero_positions: &[usize]) -> PolyExtLex {
    if zero_positions.is_empty() || poly.is_zero() {
        return poly.clone();
    }
    let mut out = poly.zero_with_capacity(poly.nterms());
    for i in 0..poly.nterms() {
        let exponents = poly.exponents(i);
        if zero_positions
            .iter()
            .all(|&position| exponents[position] == 0)
        {
            out.append_monomial_back(poly.coefficients[i].clone(), exponents);
        }
    }
    out
}

/// The polynomial `x_position` in the same ring and variable map as `template`.
fn variable_generator(template: &PolyExtLex, position: usize) -> PolyExtLex {
    let mut exponents = vec![0u16; template.nvars()];
    exponents[position] = 1;
    let mut out = template.zero_with_capacity(1);
    out.append_monomial_back(template.ring.one(), &exponents);
    out
}

/// Generators of the work item's ideal: the cover's factors with the zeroed
/// variables substituted away, plus one generator per zeroed variable — the
/// constraints `x = 0` belong to the ideal, so the basis must account for
/// them (otherwise zeroing everything yields an empty basis that looks like
/// the whole ring rather than the origin).
pub fn decode_work_item(
    factors: &[PolyExtLex],
    sparse_variables: &[(String, usize)],
    item: WorkItem,
) -> Vec<PolyExtLex> {
    let zero_positions: Vec<_> = zero_positions(sparse_variables, item.zero_mask).collect();
    let substituted = decode_cover(factors, item.cover_mask)
        .into_iter()
        .map(|factor| substitute_zeroes(&factor, &zero_positions))
        .filter(|factor| !factor.is_zero());
    let zero_generators = zero_positions
        .iter()
        .map(|&position| variable_generator(&factors[0], position));
    substituted.chain(zero_generators).collect()
}

/// Krull dimension of the solution locus from a reduced degrevlex Gröbner
/// basis, via the initial ideal: a variable set is independent iff its
/// complement hits every leading-monomial support, so the dimension is
/// `nvars` minus the size of a minimum transversal of the support hypergraph
/// (reusing `minimal_covers`). `None` when `nvars` exceeds the 63-bit mask
/// limit or the ideal is the whole ring.

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
    fn substitute_zeroes_drops_terms_containing_zeroed_variables() {
        let poly = test_poly("x*y + z");

        let substituted = substitute_zeroes(&poly, &[0]);

        assert_eq!(substituted, poly_like(&poly, "z"));
        assert!(substitute_zeroes(&test_poly("x*y"), &[0]).is_zero());
        let unaffected = poly_like(&poly, "z + 1");
        assert_eq!(substitute_zeroes(&unaffected, &[0]), unaffected);
    }

    #[test]
    fn decode_work_item_keeps_zeroed_variables_as_generators() {
        let template = test_poly("x*y");
        let sparse_variables = vec![("x".to_string(), 0), ("y".to_string(), 1)];
        let item = WorkItem {
            cover_mask: 0b1,
            zero_mask: 0b11,
        };
        let decoded = decode_work_item(&[template.clone()], &sparse_variables, item);
        let expected = [poly_like(&template, "x"), poly_like(&template, "y")];
        assert_eq!(decoded.len(), 2);
        assert!(expected.iter().all(|e| decoded.contains(e)));
    }
}
