//! Evaluation of a [`egg::RecExpr`] extracted from an e-graph root.

use crate::expr_lang::{RationalLang, VarId};
use crate::field::{pi_approx_q, q_pow_integer, Q};
use egg::{AstSize, EGraph, Extractor, Id, RecExpr};
use num_traits::Zero;
use std::collections::HashMap;
use std::hash::BuildHasher;

/// Evaluation error.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EvalError {
    MissingVariable(VarId),
    DivisionByZero,
    NonIntegerExponent,
}

/// Evaluate the expression at `root` using `assignment` for variables.
pub fn evaluate<S: BuildHasher>(
    egraph: &EGraph<RationalLang, ()>,
    root: Id,
    assignment: &HashMap<VarId, Q, S>,
) -> Result<Q, EvalError> {
    let extractor = Extractor::new(egraph, AstSize);
    let (_, recexpr) = extractor.find_best(root);
    let mut memo = HashMap::new();
    eval_recexpr_memo(&recexpr, recexpr.root(), assignment, &mut memo)
}

fn eval_recexpr_memo<S: BuildHasher>(
    recexpr: &RecExpr<RationalLang>,
    id: Id,
    assignment: &HashMap<VarId, Q, S>,
    memo: &mut HashMap<Id, Q>,
) -> Result<Q, EvalError> {
    if let Some(v) = memo.get(&id) {
        return Ok(v.clone());
    }
    let v = match recexpr[id].clone() {
        RationalLang::Const(c) => c,
        RationalLang::Pi => pi_approx_q(),
        RationalLang::Var(x) => assignment
            .get(&x)
            .cloned()
            .ok_or(EvalError::MissingVariable(x))?,
        RationalLang::Add([a, b]) => {
            eval_recexpr_memo(recexpr, a, assignment, memo)?
                + eval_recexpr_memo(recexpr, b, assignment, memo)?
        }
        RationalLang::Mul([a, b]) => {
            eval_recexpr_memo(recexpr, a, assignment, memo)?
                * eval_recexpr_memo(recexpr, b, assignment, memo)?
        }
        RationalLang::Neg(x) => -eval_recexpr_memo(recexpr, x, assignment, memo)?,
        RationalLang::Div([n, d]) => {
            let num = eval_recexpr_memo(recexpr, n, assignment, memo)?;
            let den = eval_recexpr_memo(recexpr, d, assignment, memo)?;
            if den.is_zero() {
                return Err(EvalError::DivisionByZero);
            }
            num / den
        }
        RationalLang::Pow([b, e]) => {
            let base = eval_recexpr_memo(recexpr, b, assignment, memo)?;
            let exp = eval_recexpr_memo(recexpr, e, assignment, memo)?;
            if !exp.is_integer() {
                return Err(EvalError::NonIntegerExponent);
            }
            q_pow_integer(base, exp.numer()).ok_or(EvalError::DivisionByZero)?
        }
    };
    memo.insert(id, v.clone());
    Ok(v)
}
