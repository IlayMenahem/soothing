//! Symbolica helpers: parse in the crate namespace.

use symbolica::atom::Atom;
use symbolica::parser::ParseSettings;

fn crate_namespace() -> &'static str {
    env!("CARGO_CRATE_NAME")
}

/// Parse a full expression string (same namespace as equation loading).
pub fn parse_expression(data: String) -> Result<Atom, String> {
    Atom::parse(data, crate_namespace(), ParseSettings::default())
}

/// Parse a single identifier as an atom (one variable).
pub(crate) fn parse_identifier(name: &str) -> Result<Atom, String> {
    parse_expression(name.to_string())
}
