//! Depth-first linear elimination using exact polynomial predicates ([`crate::exact`]).
//!
//! Same search shape as [`crate::elimination::dfs_eliminate`], but linearity and
//! zero-testing use exact rational checks instead of randomized sampling.

use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::io::Write;
use std::ops::{Div, Sub};
use std::time::Instant;

use symbolica::atom::{Atom, AtomCore, Symbol};
use thiserror::Error;

use crate::atom_util::{substitute_symbol, vars_in_atom};
use crate::elimination::EliminationResult;
use crate::exact::{
    ExactPredicateError, is_linear_in_exact, is_zero_exact, rhs_denominator_nonzero_exact,
};
use crate::substitution::{SubstitutionError, SubstitutionMap};
use crate::system::PolynomialSystem;
use crate::var_id::VarId;

#[derive(Debug, Error)]
pub enum EliminationExactError {
    #[error("illegitimate elimination: a=0 and b≠0 for variable {var:?}")]
    IllegitimateElimination { var: VarId },
    #[error(
        "degenerate substitution: rational denominator is zero or conversion failed (variable {var:?})"
    )]
    DegenerateSubstitution { var: VarId },
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Predicate(#[from] ExactPredicateError),
}

#[derive(Default)]
struct EliminationTraceState {
    last_tick: Option<Instant>,
    dfs_visits: u64,
}

fn expanded_rhs_map(
    result: &EliminationResult,
    sym_to_vid: &HashMap<Symbol, VarId>,
    id_to_sym: &[Symbol],
) -> Result<HashMap<VarId, Atom>, SubstitutionError> {
    let order = result.substitution.replay_order(sym_to_vid)?;
    let mut expanded: HashMap<VarId, Atom> = HashMap::default();
    for &v in &order {
        let mut node = result
            .substitution
            .get(v)
            .expect("expanded_rhs: missing")
            .clone();
        for &u in &order {
            if u == v {
                break;
            }
            if let Some(ur) = expanded.get(&u) {
                let usym = id_to_sym[u.0 as usize];
                node = substitute_symbol(&node, usym, ur.clone());
            }
        }
        expanded.insert(v, node.expand());
    }
    Ok(expanded)
}

/// Like [`crate::elimination::is_solution_nonzero_kappah`], but uses exact zero
/// tests on the expanded RHS for `kappah`.
#[must_use]
pub fn is_solution_nonzero_kappah_exact(
    r: &EliminationResult,
    kappah: VarId,
    sym_to_vid: &HashMap<Symbol, VarId>,
    id_to_sym: &[Symbol],
) -> bool {
    if !crate::elimination::default_is_solution(r) {
        return false;
    }
    if !r.substitution.contains(kappah) {
        return true;
    }
    let expanded_zero = match expanded_rhs_map(r, sym_to_vid, id_to_sym) {
        Ok(expanded_rhs) => {
            let Some(expanded_atom) = expanded_rhs.get(&kappah) else {
                return false;
            };
            is_zero_exact(expanded_atom, sym_to_vid, id_to_sym).unwrap_or(true)
        }
        Err(SubstitutionError::Cycle) => return false,
    };
    !expanded_zero
}

fn validate_elimination_result(
    original: &PolynomialSystem,
    result: &EliminationResult,
    sym_to_vid: &HashMap<Symbol, VarId>,
    id_to_sym: &[Symbol],
) {
    if result.substitution.is_empty() || !result.reduced_system.equations.is_empty() {
        return;
    }
    let expanded_rhs = expanded_rhs_map(result, sym_to_vid, id_to_sym)
        .expect("elimination validation: cyclic substitution map");

    let order = result
        .substitution
        .replay_order(sym_to_vid)
        .expect("elimination validation: replay");

    for (i, eq) in original.equations.iter().enumerate() {
        let mut r = eq.clone();
        for &v in &order {
            if let Some(repl) = expanded_rhs.get(&v) {
                let sym = id_to_sym[v.0 as usize];
                r = substitute_symbol(&r, sym, repl.clone());
            }
        }
        let ok = is_zero_exact(&r, sym_to_vid, id_to_sym).unwrap_or(false);
        assert!(
            ok,
            "elimination validation failed: original equation index {i} does not vanish"
        );
    }
}

/// Exact-predicate variant of [`crate::elimination::dfs_eliminate`].
pub fn dfs_eliminate_exact<F>(
    system: &PolynomialSystem,
    trace_writer: Option<&mut dyn Write>,
    is_solution: F,
    sym_to_vid: &HashMap<Symbol, VarId>,
    id_to_sym: &[Symbol],
) -> Result<EliminationResult, EliminationExactError>
where
    F: Fn(&EliminationResult) -> bool,
{
    let original = system.clone();
    let eqs: Vec<Atom> = system
        .equations
        .iter()
        .map(|e| e.clone().expand())
        .collect();
    let subst = SubstitutionMap::new();
    let result = match trace_writer {
        None => {
            dfs_eliminate_exact_inner(eqs, subst, None, None, &is_solution, sym_to_vid, id_to_sym)?
        }
        Some(w) => {
            let trace_w = RefCell::new(w);
            let trace_st = RefCell::new(EliminationTraceState::default());
            dfs_eliminate_exact_inner(
                eqs,
                subst,
                Some(&trace_w),
                Some(&trace_st),
                &is_solution,
                sym_to_vid,
                id_to_sym,
            )?
        }
    };
    validate_elimination_result(&original, &result, sym_to_vid, id_to_sym);
    Ok(result)
}

/// Recursive step for [`dfs_eliminate_exact`].
#[allow(clippy::too_many_arguments)]
fn dfs_eliminate_exact_inner<F>(
    eq_roots: Vec<Atom>,
    subst: SubstitutionMap,
    trace_w: Option<&RefCell<&mut dyn Write>>,
    trace_st: Option<&RefCell<EliminationTraceState>>,
    is_solution: &F,
    sym_to_vid: &HashMap<Symbol, VarId>,
    id_to_sym: &[Symbol],
) -> Result<EliminationResult, EliminationExactError>
where
    F: Fn(&EliminationResult) -> bool,
{
    if let (Some(cw), Some(cs)) = (trace_w, trace_st) {
        let mut st = cs.borrow_mut();
        let mut out = cw.borrow_mut();
        st.dfs_visits += 1;
        let elapsed = st.last_tick.map(|t| t.elapsed()).unwrap_or_default();
        st.last_tick = Some(Instant::now());
        let remaining_eq = eq_roots.len();
        writeln!(
            &mut *out,
            "{} {} elapsed: {:.6}s",
            st.dfs_visits,
            remaining_eq,
            elapsed.as_secs_f64()
        )?;
        out.flush()?;
    }

    if eq_roots.is_empty() {
        return Ok(EliminationResult {
            reduced_system: PolynomialSystem::new(eq_roots),
            substitution: subst,
        });
    }

    let remaining: BTreeSet<VarId> = eq_roots
        .iter()
        .flat_map(|e| vars_in_atom(e, sym_to_vid))
        .collect();

    let pairs = collect_linear_pairs(&eq_roots, &remaining, id_to_sym, sym_to_vid);
    if pairs.is_empty() {
        return Ok(EliminationResult {
            reduced_system: PolynomialSystem::new(eq_roots),
            substitution: subst,
        });
    }

    let mut best = EliminationResult {
        reduced_system: PolynomialSystem::new(eq_roots.clone()),
        substitution: subst.clone(),
    };

    let mut branch_skips: usize = 0;
    for (x, i) in pairs {
        let mut eq_branch = eq_roots.clone();
        let mut subst_branch = subst.clone();
        let remaining_branch: BTreeSet<VarId> = eq_branch
            .iter()
            .flat_map(|e| vars_in_atom(e, sym_to_vid))
            .collect();
        match apply_one_elimination(
            &mut eq_branch,
            &mut subst_branch,
            x,
            i,
            &remaining_branch,
            id_to_sym,
            sym_to_vid,
        ) {
            Ok(()) => {}
            Err(
                EliminationExactError::IllegitimateElimination { .. }
                | EliminationExactError::DegenerateSubstitution { .. },
            ) => {
                branch_skips += 1;
                continue;
            }
            Err(e) => return Err(e),
        }

        if let Some(cw) = trace_w {
            let mut out = cw.borrow_mut();
            writeln!(
                &mut *out,
                "branch var={x:?} eq_index={i} skips_before_success={branch_skips}"
            )?;
            out.flush()?;
        }

        let r = dfs_eliminate_exact_inner(
            eq_branch,
            subst_branch,
            trace_w,
            trace_st,
            is_solution,
            sym_to_vid,
            id_to_sym,
        )?;
        if is_solution(&r) {
            return Ok(r);
        }
        if is_better_elimination(&r, &best, sym_to_vid) {
            best = r;
        }
    }

    Ok(best)
}

fn collect_linear_pairs(
    eq_roots: &[Atom],
    remaining: &BTreeSet<VarId>,
    id_to_sym: &[Symbol],
    sym_to_vid: &HashMap<Symbol, VarId>,
) -> Vec<(VarId, usize)> {
    let candidates: Vec<(usize, VarId)> = (0..eq_roots.len())
        .flat_map(|i| {
            let mut vars_in_eq: Vec<VarId> = vars_in_atom(&eq_roots[i], sym_to_vid)
                .into_iter()
                .filter(|v| remaining.contains(v))
                .collect();
            vars_in_eq.sort_by_key(|v| v.0);
            vars_in_eq.into_iter().map(move |x| (i, x))
        })
        .collect();

    let mut out = Vec::new();
    for &(i, x) in &candidates {
        if matches!(
            is_linear_in_exact(&eq_roots[i], x, sym_to_vid, id_to_sym),
            Ok(true)
        ) {
            out.push((x, i));
        }
    }
    out
}

fn is_better_elimination(
    a: &EliminationResult,
    b: &EliminationResult,
    sym_to_vid: &HashMap<Symbol, VarId>,
) -> bool {
    let na = a.reduced_system.equations.len();
    let nb = b.reduced_system.equations.len();
    if na != nb {
        return na < nb;
    }
    let free_a: HashSet<VarId> = a
        .reduced_system
        .equations
        .iter()
        .flat_map(|e| vars_in_atom(e, sym_to_vid))
        .collect();
    let free_b: HashSet<VarId> = b
        .reduced_system
        .equations
        .iter()
        .flat_map(|e| vars_in_atom(e, sym_to_vid))
        .collect();
    free_a.len() > free_b.len()
}

fn simplify_equations(eqs: &mut [Atom]) {
    for e in eqs.iter_mut() {
        *e = e.clone().expand();
    }
}

fn retain_nonzero_equations(
    eq_roots: &mut Vec<Atom>,
    id_to_sym: &[Symbol],
    sym_to_vid: &HashMap<Symbol, VarId>,
) {
    eq_roots.retain(|eq| !matches!(is_zero_exact(eq, sym_to_vid, id_to_sym), Ok(true)));
}

#[allow(clippy::too_many_arguments)]
fn apply_one_elimination(
    eq_roots: &mut Vec<Atom>,
    subst: &mut SubstitutionMap,
    x: VarId,
    i: usize,
    _remaining: &BTreeSet<VarId>,
    id_to_sym: &[Symbol],
    sym_to_vid: &HashMap<Symbol, VarId>,
) -> Result<(), EliminationExactError> {
    let x_sym = id_to_sym[x.0 as usize];
    let eq = &eq_roots[i];
    let zero = Atom::num(0i64);
    let one = Atom::num(1i64);
    let b_coeff = substitute_symbol(eq, x_sym, zero);
    let p_at_one = substitute_symbol(eq, x_sym, one);
    let a_coeff = p_at_one.sub(&b_coeff).expand();

    let a_is_zero = is_zero_exact(&a_coeff, sym_to_vid, id_to_sym)?;
    let b_is_zero = is_zero_exact(&b_coeff, sym_to_vid, id_to_sym)?;

    if a_is_zero && !b_is_zero {
        return Err(EliminationExactError::IllegitimateElimination { var: x });
    }
    if a_is_zero && b_is_zero {
        let eq_slice: &[Atom] = eq_roots.as_slice();
        *eq_roots = (0..eq_slice.len())
            .filter(|&j| j != i)
            .map(|j| eq_slice[j].clone())
            .collect();
        retain_nonzero_equations(eq_roots, id_to_sym, sym_to_vid);
        simplify_equations(eq_roots);
        return Ok(());
    }

    let neg_a = Atom::num(0i64).sub(&a_coeff).expand();
    let x_sol = b_coeff.div(&neg_a).expand();

    if !rhs_denominator_nonzero_exact(&x_sol, sym_to_vid, id_to_sym)? {
        return Err(EliminationExactError::DegenerateSubstitution { var: x });
    }

    subst.insert(x, x_sol.clone());

    let x_sol_cloned = x_sol.clone();
    let eq_slice: &[Atom] = eq_roots.as_slice();
    *eq_roots = (0..eq_slice.len())
        .filter(|&j| j != i)
        .map(|j| substitute_symbol(&eq_slice[j], x_sym, x_sol_cloned.clone()).expand())
        .collect();

    retain_nonzero_equations(eq_roots, id_to_sym, sym_to_vid);
    simplify_equations(eq_roots);
    Ok(())
}
