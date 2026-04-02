//! Expression parsing for [`Dag::from_rational_string`](crate::dag::Dag::from_rational_string).
//! Parsing is delegated to [`arithmetic_parser`]; this module only defines a rational literal
//! grammar and lowers the AST into a [`DagBuilder`].

use crate::dag::{Dag, DagBuilder, DagParseError, NodeId, VarId};
use crate::field::Q;
use arithmetic_parser::{
    grammars::{Features, Parse, ParseLiteral, Untyped},
    BinaryOp, Error as ApError, ErrorKind as ApErrorKind, Expr, InputSpan, NomResult, Spanned,
    SpannedExpr, UnaryOp,
};
use nom::{
    character::complete::{char as tag_char, digit1},
    combinator::{map_res, opt, recognize},
    sequence::tuple,
    Err as NomErr, Slice,
};
use num_bigint::BigInt;
use num_traits::Zero;
use std::collections::HashMap;

/// Grammar: [`Q`] literals (decimal integers or `p/q`), variables, and operators handled by `arithmetic-parser`
/// with [`Features::empty()`] (no tuples, blocks, function defs, methods, or boolean comparisons).
#[derive(Debug)]
struct RationalGrammar;

impl ParseLiteral for RationalGrammar {
    type Lit = Q;

    fn parse_literal(input: InputSpan<'_>) -> NomResult<'_, Self::Lit> {
        let mut parse_bigint = map_res(
            recognize(tuple((opt(tag_char('-')), digit1))),
            |s: InputSpan<'_>| s.fragment().parse::<BigInt>().map_err(ApErrorKind::literal),
        );
        let (rest, num) = parse_bigint(input)?;
        if rest.fragment().starts_with('/') {
            let after_slash = rest.slice(1..);
            let (rest2, den) = parse_bigint(after_slash)?;
            if den.is_zero() {
                let frag = after_slash.fragment();
                let sp = Spanned::from_str(frag, 0..frag.len());
                return Err(NomErr::Failure(
                    ApErrorKind::literal(anyhow::anyhow!("zero denominator")).with_span(&sp),
                ));
            }
            Ok((rest2, Q::new(num, den)))
        } else {
            Ok((rest, Q::new(num, 1.into())))
        }
    }
}

impl Parse for RationalGrammar {
    type Base = Untyped<Self>;
    const FEATURES: Features = Features::empty();
}

pub(crate) fn parse_rational_expression(s: &str) -> Result<(Dag, NodeId), DagParseError> {
    let block = RationalGrammar::parse_statements(s).map_err(|e| map_ap_error(s, e))?;
    if !block.statements.is_empty() {
        return Err(DagParseError::Unsupported {
            reason: "only one expression is allowed (no semicolons or assignments)",
            pos: 0,
        });
    }
    let Some(ret) = block.return_value else {
        return Err(DagParseError::UnexpectedEof);
    };
    let mut builder = DagBuilder::new();
    let mut name_to_var = HashMap::new();
    let mut next_free = 0u32;
    let root = lower_expr_allocating(&ret, &mut builder, &mut name_to_var, &mut next_free)?;
    Ok((builder.build(), root))
}

/// Parse using a fixed variable name → [`VarId`] map (no fresh ids). Unknown identifiers error.
pub fn parse_rational_expression_with_var_map(
    s: &str,
    var_map: &HashMap<String, VarId>,
) -> Result<(Dag, NodeId), DagParseError> {
    let block = RationalGrammar::parse_statements(s).map_err(|e| map_ap_error(s, e))?;
    if !block.statements.is_empty() {
        return Err(DagParseError::Unsupported {
            reason: "only one expression is allowed (no semicolons or assignments)",
            pos: 0,
        });
    }
    let Some(ret) = block.return_value else {
        return Err(DagParseError::UnexpectedEof);
    };
    let mut builder = DagBuilder::new();
    let root = lower_expr_fixed_map(&ret, &mut builder, var_map)?;
    Ok((builder.build(), root))
}

fn map_ap_error(s: &str, e: ApError) -> DagParseError {
    let pos = e.location().location_offset();
    let char_at_pos = || s[pos..].chars().next().unwrap_or('\u{0}');
    match e.kind() {
        ApErrorKind::Leftovers => DagParseError::TrailingInput { pos },
        ApErrorKind::UnexpectedChar { .. } | ApErrorKind::UnexpectedTerm { .. } => {
            DagParseError::UnexpectedChar {
                ch: char_at_pos(),
                pos,
            }
        }
        ApErrorKind::Literal(_) => DagParseError::InvalidNumber {
            pos,
            message: e.kind().to_string(),
        },
        ApErrorKind::UnsupportedOp(_) => DagParseError::Unsupported {
            reason: "operator not available in this grammar",
            pos,
        },
        ApErrorKind::Other { .. } => DagParseError::UnexpectedChar {
            ch: char_at_pos(),
            pos,
        },
        _ => DagParseError::InvalidNumber {
            pos,
            message: e.kind().to_string(),
        },
    }
}

fn var_id_from_v_digits(name: &str) -> Option<VarId> {
    let mut chars = name.chars();
    let first = chars.next()?;
    if first != 'v' && first != 'V' {
        return None;
    }
    let rest: String = chars.collect();
    if rest.is_empty() || !rest.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    rest.parse::<u32>().ok().map(VarId)
}

fn var_for_ident(
    name: &str,
    name_to_var: &mut HashMap<String, VarId>,
    next_free: &mut u32,
) -> VarId {
    if let Some(id) = var_id_from_v_digits(name) {
        return id;
    }
    if let Some(&id) = name_to_var.get(name) {
        return id;
    }
    let id = VarId(*next_free);
    *next_free = next_free.saturating_add(1);
    name_to_var.insert(name.to_string(), id);
    id
}

fn lower_expr_allocating(
    sp: &SpannedExpr<'_, Untyped<RationalGrammar>>,
    b: &mut DagBuilder,
    name_to_var: &mut HashMap<String, VarId>,
    next_free: &mut u32,
) -> Result<NodeId, DagParseError> {
    let pos = sp.location_offset();
    match &sp.extra {
        Expr::Literal(q) => Ok(b.constant(q.clone())),
        Expr::Variable => {
            let name: &str = sp.fragment();
            if name == "pi" {
                return Ok(b.const_pi());
            }
            let v = var_for_ident(name, name_to_var, next_free);
            Ok(b.variable(v))
        }
        Expr::Unary { op, inner } => match op.extra {
            UnaryOp::Neg => {
                let x = lower_expr_allocating(inner, b, name_to_var, next_free)?;
                Ok(b.neg(x))
            }
            UnaryOp::Not => Err(DagParseError::Unsupported {
                reason: "logical negation is not part of the polynomial fragment",
                pos,
            }),
            _ => Err(DagParseError::Unsupported {
                reason: "unsupported unary operator",
                pos,
            }),
        },
        Expr::Binary { lhs, op, rhs } => {
            let l = lower_expr_allocating(lhs, b, name_to_var, next_free)?;
            let r = lower_expr_allocating(rhs, b, name_to_var, next_free)?;
            match op.extra {
                BinaryOp::Add => Ok(b.add(vec![l, r])),
                BinaryOp::Sub => Ok(b.sub(l, r)),
                BinaryOp::Mul => Ok(b.mul2(l, r)),
                BinaryOp::Div => Ok(b.div(l, r)),
                BinaryOp::Power => Ok(b.pow(l, r)),
                _ => Err(DagParseError::Unsupported {
                    reason: "only arithmetic operators are supported",
                    pos: op.location_offset(),
                }),
            }
        }
        Expr::Function { .. }
        | Expr::Method { .. }
        | Expr::FieldAccess { .. }
        | Expr::Tuple(_)
        | Expr::Block(_)
        | Expr::Object(_)
        | Expr::FnDefinition(_)
        | Expr::TypeCast { .. } => Err(DagParseError::Unsupported {
            reason: "only literals, variables, and arithmetic operations are supported",
            pos,
        }),
        _ => Err(DagParseError::Unsupported {
            reason: "unsupported expression form",
            pos,
        }),
    }
}

fn lower_expr_fixed_map(
    sp: &SpannedExpr<'_, Untyped<RationalGrammar>>,
    b: &mut DagBuilder,
    var_map: &HashMap<String, VarId>,
) -> Result<NodeId, DagParseError> {
    let pos = sp.location_offset();
    match &sp.extra {
        Expr::Literal(q) => Ok(b.constant(q.clone())),
        Expr::Variable => {
            let name: &str = sp.fragment();
            if name == "pi" {
                return Ok(b.const_pi());
            }
            let v = var_map
                .get(name)
                .copied()
                .ok_or_else(|| DagParseError::UnknownVariable {
                    name: name.to_string(),
                    pos,
                })?;
            Ok(b.variable(v))
        }
        Expr::Unary { op, inner } => match op.extra {
            UnaryOp::Neg => {
                let x = lower_expr_fixed_map(inner, b, var_map)?;
                Ok(b.neg(x))
            }
            UnaryOp::Not => Err(DagParseError::Unsupported {
                reason: "logical negation is not part of the polynomial fragment",
                pos,
            }),
            _ => Err(DagParseError::Unsupported {
                reason: "unsupported unary operator",
                pos,
            }),
        },
        Expr::Binary { lhs, op, rhs } => {
            let l = lower_expr_fixed_map(lhs, b, var_map)?;
            let r = lower_expr_fixed_map(rhs, b, var_map)?;
            match op.extra {
                BinaryOp::Add => Ok(b.add(vec![l, r])),
                BinaryOp::Sub => Ok(b.sub(l, r)),
                BinaryOp::Mul => Ok(b.mul2(l, r)),
                BinaryOp::Div => Ok(b.div(l, r)),
                BinaryOp::Power => Ok(b.pow(l, r)),
                _ => Err(DagParseError::Unsupported {
                    reason: "only arithmetic operators are supported",
                    pos: op.location_offset(),
                }),
            }
        }
        Expr::Function { .. }
        | Expr::Method { .. }
        | Expr::FieldAccess { .. }
        | Expr::Tuple(_)
        | Expr::Block(_)
        | Expr::Object(_)
        | Expr::FnDefinition(_)
        | Expr::TypeCast { .. } => Err(DagParseError::Unsupported {
            reason: "only literals, variables, and arithmetic operations are supported",
            pos,
        }),
        _ => Err(DagParseError::Unsupported {
            reason: "unsupported expression form",
            pos,
        }),
    }
}
