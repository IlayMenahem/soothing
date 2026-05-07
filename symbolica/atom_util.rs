//! Symbolica helpers: parse in the crate namespace, substitute, collect variables.

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};

use symbolica::atom::{Atom, AtomCore, DefaultNamespace, Symbol};
use symbolica::id::Replacement;
use symbolica::parser::ParseSettings;

use crate::var_id::VarId;

fn crate_namespace() -> &'static str {
    env!("CARGO_CRATE_NAME")
}

/// Parse a full expression string (same namespace as equation loading).
pub fn parse_expression(data: String) -> Result<Atom, String> {
    Atom::parse(
        DefaultNamespace {
            data,
            namespace: Cow::Borrowed(crate_namespace()),
            file: Cow::Borrowed(""),
            line: 0,
        },
        ParseSettings::default(),
    )
}

/// Parse a single identifier as an atom (one variable).
pub fn parse_identifier(name: &str) -> Result<Atom, String> {
    parse_expression(name.to_string())
}

/// `x` ↦ `rhs` (pattern replace).
pub fn substitute_symbol(expr: &Atom, x: Symbol, rhs: Atom) -> Atom {
    let pat = Atom::var(x).to_pattern();
    expr.replace_multiple(&[Replacement::new(pat, rhs)])
        .expand()
}

/// Variables from `expr` that appear in `sym_to_vid`.
pub fn vars_in_atom(expr: &Atom, sym_to_vid: &HashMap<Symbol, VarId>) -> HashSet<VarId> {
    expr.get_all_symbols(false)
        .into_iter()
        .filter_map(|s| sym_to_vid.get(&s).copied())
        .collect()
}

