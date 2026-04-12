//! Exact polynomial predicates over `Q` via Symbolica [`Atom::try_to_polynomial`](symbolica::atom::AtomCore).

use std::collections::HashMap;
use std::ops::{Mul, Sub};
use std::sync::Arc;

use symbolica::atom::{Atom, AtomCore, Symbol};
use symbolica::domains::integer::Z;
use symbolica::domains::rational::Q;
use symbolica::poly::PolyVariable;
use thiserror::Error;

use crate::atom_util::{parse_identifier, substitute_symbol, vars_in_atom};
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

/// Paper's `q` test for linearity in `x`, decided by exact zero polynomial on `Q`.
pub fn is_linear_in_exact(
    expr: &Atom,
    x: VarId,
    sym_to_vid: &HashMap<Symbol, VarId>,
    id_to_sym: &[Symbol],
) -> Result<bool, ExactPredicateError> {
    let x_sym = id_to_sym[x.0 as usize];
    let base = id_to_sym.len();
    let u_name = format!("___lin_u_{base}");
    let v_name = format!("___lin_v_{base}");
    let w_name = format!("___lin_w_{base}");

    let u_atom = parse_identifier(&u_name).map_err(ExactPredicateError::Conversion)?;
    let v_atom = parse_identifier(&v_name).map_err(ExactPredicateError::Conversion)?;
    let w_atom = parse_identifier(&w_name).map_err(ExactPredicateError::Conversion)?;

    let p_u = substitute_symbol(expr, x_sym, u_atom.clone());
    let p_v = substitute_symbol(expr, x_sym, v_atom.clone());
    let p_w = substitute_symbol(expr, x_sym, w_atom.clone());

    let p_wu = p_w.sub(&p_u);
    let p_vu = p_v.sub(&p_u);
    let v_minus_u = v_atom.clone().sub(&u_atom.clone());
    let w_minus_u = w_atom.clone().sub(&u_atom);
    let term1 = p_wu.mul(&v_minus_u);
    let term2 = p_vu.mul(&w_minus_u);
    let q = term1.sub(&term2).expand();

    let u_sym = u_atom
        .get_symbol()
        .ok_or(ExactPredicateError::Conversion(u_name))?;
    let v_sym = v_atom
        .get_symbol()
        .ok_or(ExactPredicateError::Conversion(v_name))?;
    let w_sym = w_atom
        .get_symbol()
        .ok_or(ExactPredicateError::Conversion(w_name))?;

    let n = id_to_sym.len();
    let mut extended_syms: Vec<Symbol> = id_to_sym.to_vec();
    extended_syms.push(u_sym);
    extended_syms.push(v_sym);
    extended_syms.push(w_sym);

    let mut extended_map: HashMap<Symbol, VarId> = sym_to_vid.clone();
    let b =
        u32::try_from(n).map_err(|_| ExactPredicateError::Conversion("var id overflow".into()))?;
    extended_map.insert(u_sym, VarId(b));
    extended_map.insert(v_sym, VarId(b + 1));
    extended_map.insert(w_sym, VarId(b + 2));

    is_zero_exact(&q, &extended_map, &extended_syms)
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
