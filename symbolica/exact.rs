//! Exact polynomial predicates over `Q` via Symbolica [`Atom::try_to_polynomial`](symbolica::atom::AtomCore).

use std::collections::HashMap;
use std::sync::Arc;

use symbolica::atom::{Atom, AtomCore, Symbol};
use symbolica::domains::integer::Z;
use symbolica::domains::rational::Q;
use symbolica::poly::PolyVariable;
use thiserror::Error;

use crate::atom_util::vars_in_atom;
use crate::var_id::VarId;

#[derive(Debug, Error)]
pub enum ExactPredicateError {
    #[error("{0}")]
    Conversion(String),
}

fn poly_var_map(
    expr: &Atom,
    sym_to_vid: &HashMap<Symbol, VarId>,
    id_to_sym: &[Symbol],
) -> Option<Arc<Vec<PolyVariable>>> {
    let mut vids: Vec<VarId> = vars_in_atom(expr, sym_to_vid).into_iter().collect();
    if vids.is_empty() {
        return None;
    }
    vids.sort_by_key(|v| v.0);
    Some(Arc::new(
        vids.into_iter()
            .map(|v| PolyVariable::from(id_to_sym[v.0 as usize]))
            .collect(),
    ))
}

/// True iff `expr` expands to the zero polynomial in `Q` on the variables occurring in `expr`.
pub fn is_zero_exact(
    expr: &Atom,
    sym_to_vid: &HashMap<Symbol, VarId>,
    id_to_sym: &[Symbol],
) -> Result<bool, ExactPredicateError> {
    let expanded = expr.clone().expand();
    let poly = expanded
        .try_to_polynomial::<_, u16>(&Q, poly_var_map(&expanded, sym_to_vid, id_to_sym))
        .map_err(ExactPredicateError::Conversion)?;
    Ok(poly.is_zero())
}

/// `true` iff `expr`, viewed as a rational function, has numerator equal to the zero polynomial.
///
/// Unlike [`is_zero_exact`], this tolerates rational expressions (e.g. `1/π^2 * p`):
/// only the numerator polynomial of `expr` matters for vanishing.
pub fn is_numerator_zero_exact(
    expr: &Atom,
    sym_to_vid: &HashMap<Symbol, VarId>,
    id_to_sym: &[Symbol],
) -> Result<bool, ExactPredicateError> {
    let expanded = expr.clone().expand();
    let rp = expanded
        .try_to_rational_polynomial::<_, _, u16>(
            &Q,
            &Z,
            poly_var_map(&expanded, sym_to_vid, id_to_sym),
        )
        .map_err(ExactPredicateError::Conversion)?;
    Ok(rp.numerator.is_zero())
}

/// `true` if `expr` represents a rational function whose denominator is not the zero polynomial.
pub fn rhs_denominator_nonzero_exact(
    expr: &Atom,
    sym_to_vid: &HashMap<Symbol, VarId>,
    id_to_sym: &[Symbol],
) -> Result<bool, ExactPredicateError> {
    let expanded = expr.clone().expand();
    let rp = expanded
        .try_to_rational_polynomial::<_, _, u16>(
            &Q,
            &Z,
            poly_var_map(&expanded, sym_to_vid, id_to_sym),
        )
        .map_err(ExactPredicateError::Conversion)?;
    Ok(!rp.denominator.is_zero())
}
