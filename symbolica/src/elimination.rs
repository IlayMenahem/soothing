//! DFS linear elimination by substitution.

use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::io::Write;
use std::ops::{Div, Sub};
use std::time::Instant;

use rayon::prelude::*;
use symbolica::atom::{Atom, AtomCore, Symbol};
use thiserror::Error;

use crate::atom_util::{substitute_symbol, vars_in_atom};
use crate::predicates::{
    PredicateError, SampleConfig, expr_finite_at_some_random_sample, is_linear_in, is_zero,
};
use crate::substitution::{SubstitutionError, SubstitutionMap};
use crate::system::PolynomialSystem;
use crate::var_id::VarId;

#[derive(Debug, Error)]
pub enum EliminationError {
    #[error("illegitimate elimination: a=0 and b≠0 for variable {var:?}")]
    IllegitimateElimination { var: VarId },
    #[error(
        "degenerate substitution: RHS never evaluates successfully at random exact samples (variable {var:?})"
    )]
    DegenerateSubstitution { var: VarId },
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Predicate(#[from] PredicateError),
}

#[derive(Default)]
struct EliminationTraceState {
    last_tick: Option<Instant>,
    dfs_visits: u64,
}

pub struct EliminationResult {
    pub reduced_system: PolynomialSystem,
    pub substitution: SubstitutionMap,
}

#[must_use]
pub fn default_is_solution(r: &EliminationResult) -> bool {
    r.reduced_system.equations.is_empty()
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

/// True when the elimination fully solves the system and the expanded RHS for
/// `kappah` is not identically zero (random exact sampling).
#[must_use]
pub fn is_solution_nonzero_kappah(
    r: &EliminationResult,
    kappah: VarId,
    cfg: &SampleConfig,
    sym_to_vid: &HashMap<Symbol, VarId>,
    id_to_sym: &[Symbol],
) -> bool {
    if !default_is_solution(r) {
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
            let vs: Vec<VarId> = vars_in_atom(expanded_atom, sym_to_vid)
                .into_iter()
                .collect();
            is_zero(expanded_atom, &vs, cfg, id_to_sym).unwrap_or(true)
        }
        Err(SubstitutionError::Cycle) => return false,
    };
    !expanded_zero
}

fn validate_elimination_result(
    original: &PolynomialSystem,
    result: &EliminationResult,
    cfg: &SampleConfig,
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
        let free: Vec<VarId> = vars_in_atom(&r, sym_to_vid).into_iter().collect();
        let ok = is_zero(&r, &free, cfg, id_to_sym).unwrap_or(false);
        assert!(
            ok,
            "elimination validation failed: original equation index {i} does not vanish"
        );
    }
}

/// DFS over affine elimination steps. `is_solution` decides when to return a
/// branch immediately; otherwise the search keeps the best partial result
/// (fewest remaining equations, then fewest free variables among residuals).
pub fn dfs_eliminate<F>(
    system: &PolynomialSystem,
    cfg: &SampleConfig,
    trace_writer: Option<&mut dyn Write>,
    is_solution: F,
    sym_to_vid: &HashMap<Symbol, VarId>,
    id_to_sym: &[Symbol],
) -> Result<EliminationResult, EliminationError>
where
    F: Fn(&EliminationResult) -> bool,
{
    let original = system.clone();
    let eqs: Vec<Atom> = system
        .equations
        .par_iter()
        .map(|e| e.clone().expand())
        .collect();
    let subst = SubstitutionMap::new();
    let result = match trace_writer {
        None => dfs_eliminate_inner(
            eqs,
            subst,
            cfg,
            None,
            None,
            &is_solution,
            sym_to_vid,
            id_to_sym,
        )?,
        Some(w) => {
            let trace_w = RefCell::new(w);
            let trace_st = RefCell::new(EliminationTraceState::default());
            dfs_eliminate_inner(
                eqs,
                subst,
                cfg,
                Some(&trace_w),
                Some(&trace_st),
                &is_solution,
                sym_to_vid,
                id_to_sym,
            )?
        }
    };
    validate_elimination_result(&original, &result, cfg, sym_to_vid, id_to_sym);
    Ok(result)
}

/// Greedy elimination: at each step, among all applicable linear eliminations,
/// choose one that minimizes the number of equations immediately after
/// [`apply_one_elimination`] (including zero-equation removal). Ties break by
/// fewer distinct variables in the residual, then smaller [`VarId`], then
/// smaller equation index. Recurses only on that single branch.
pub fn greedy_eliminate<F>(
    system: &PolynomialSystem,
    cfg: &SampleConfig,
    trace_writer: Option<&mut dyn Write>,
    is_solution: F,
    sym_to_vid: &HashMap<Symbol, VarId>,
    id_to_sym: &[Symbol],
) -> Result<EliminationResult, EliminationError>
where
    F: Fn(&EliminationResult) -> bool,
{
    let original = system.clone();
    let eqs: Vec<Atom> = system
        .equations
        .par_iter()
        .map(|e| e.clone().expand())
        .collect();
    let subst = SubstitutionMap::new();
    let result = match trace_writer {
        None => greedy_eliminate_inner(
            eqs,
            subst,
            cfg,
            None,
            None,
            &is_solution,
            sym_to_vid,
            id_to_sym,
        )?,
        Some(w) => {
            let trace_w = RefCell::new(w);
            let trace_st = RefCell::new(EliminationTraceState::default());
            greedy_eliminate_inner(
                eqs,
                subst,
                cfg,
                Some(&trace_w),
                Some(&trace_st),
                &is_solution,
                sym_to_vid,
                id_to_sym,
            )?
        }
    };
    validate_elimination_result(&original, &result, cfg, sym_to_vid, id_to_sym);
    Ok(result)
}

/// Recursive step for [`dfs_eliminate`]: try each linear pair, recurse, accept
/// when `is_solution` holds or keep the best partial elimination.
#[allow(clippy::too_many_arguments)]
fn dfs_eliminate_inner<F>(
    eq_roots: Vec<Atom>,
    subst: SubstitutionMap,
    cfg: &SampleConfig,
    trace_w: Option<&RefCell<&mut dyn Write>>,
    trace_st: Option<&RefCell<EliminationTraceState>>,
    is_solution: &F,
    sym_to_vid: &HashMap<Symbol, VarId>,
    id_to_sym: &[Symbol],
) -> Result<EliminationResult, EliminationError>
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

    let pairs = collect_linear_pairs(&eq_roots, &remaining, cfg, id_to_sym, sym_to_vid);
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
            cfg,
            id_to_sym,
            sym_to_vid,
        ) {
            Ok(()) => {}
            Err(
                EliminationError::IllegitimateElimination { .. }
                | EliminationError::DegenerateSubstitution { .. },
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

        let r = dfs_eliminate_inner(
            eq_branch,
            subst_branch,
            cfg,
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

#[allow(clippy::too_many_arguments)]
fn greedy_eliminate_inner<F>(
    eq_roots: Vec<Atom>,
    subst: SubstitutionMap,
    cfg: &SampleConfig,
    trace_w: Option<&RefCell<&mut dyn Write>>,
    trace_st: Option<&RefCell<EliminationTraceState>>,
    is_solution: &F,
    sym_to_vid: &HashMap<Symbol, VarId>,
    id_to_sym: &[Symbol],
) -> Result<EliminationResult, EliminationError>
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

    let partial = EliminationResult {
        reduced_system: PolynomialSystem::new(eq_roots.clone()),
        substitution: subst.clone(),
    };
    if is_solution(&partial) {
        return Ok(partial);
    }

    let remaining: BTreeSet<VarId> = eq_roots
        .iter()
        .flat_map(|e| vars_in_atom(e, sym_to_vid))
        .collect();

    let pairs = collect_linear_pairs(&eq_roots, &remaining, cfg, id_to_sym, sym_to_vid);
    if pairs.is_empty() {
        return Ok(EliminationResult {
            reduced_system: PolynomialSystem::new(eq_roots),
            substitution: subst,
        });
    }

    let mut branch_skips: usize = 0;
    let mut best_tuple: Option<(usize, usize, u32, usize)> = None;
    let mut best_eq: Option<Vec<Atom>> = None;
    let mut best_subst: Option<SubstitutionMap> = None;
    let mut best_x: Option<VarId> = None;
    let mut best_i: Option<usize> = None;

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
            cfg,
            id_to_sym,
            sym_to_vid,
        ) {
            Ok(()) => {}
            Err(
                EliminationError::IllegitimateElimination { .. }
                | EliminationError::DegenerateSubstitution { .. },
            ) => {
                branch_skips += 1;
                continue;
            }
            Err(e) => return Err(e),
        }

        let score_eq = eq_branch.len();
        let score_free = eq_branch
            .iter()
            .flat_map(|e| vars_in_atom(e, sym_to_vid))
            .collect::<HashSet<VarId>>()
            .len();
        let cand = (score_eq, score_free, x.0, i);
        let replace = match best_tuple {
            None => true,
            Some(t) => cand < t,
        };
        if replace {
            best_tuple = Some(cand);
            best_eq = Some(eq_branch);
            best_subst = Some(subst_branch);
            best_x = Some(x);
            best_i = Some(i);
        }
    }

    let (Some(eq_branch), Some(subst_branch), Some(x), Some(i)) =
        (best_eq, best_subst, best_x, best_i)
    else {
        return Ok(EliminationResult {
            reduced_system: PolynomialSystem::new(eq_roots),
            substitution: subst,
        });
    };

    if let Some(cw) = trace_w {
        let mut out = cw.borrow_mut();
        writeln!(
            &mut *out,
            "branch var={x:?} eq_index={i} skips_before_success={branch_skips}"
        )?;
        out.flush()?;
    }

    greedy_eliminate_inner(
        eq_branch,
        subst_branch,
        cfg,
        trace_w,
        trace_st,
        is_solution,
        sym_to_vid,
        id_to_sym,
    )
}

fn collect_linear_pairs(
    eq_roots: &[Atom],
    remaining: &BTreeSet<VarId>,
    cfg: &SampleConfig,
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
    let linear_ok: Vec<bool> = candidates
        .par_iter()
        .map(|&(i, x)| {
            let others: Vec<VarId> = remaining.iter().filter(|&&v| v != x).copied().collect();
            matches!(
                is_linear_in(&eq_roots[i], x, &others, cfg, id_to_sym),
                Ok(true)
            )
        })
        .collect();
    candidates
        .iter()
        .zip(&linear_ok)
        .filter_map(|(&(i, x), &ok)| ok.then_some((x, i)))
        .collect()
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
    eqs.par_iter_mut().for_each(|e| {
        *e = e.clone().expand();
    });
}

fn retain_nonzero_equations(
    eq_roots: &mut Vec<Atom>,
    cfg: &SampleConfig,
    id_to_sym: &[Symbol],
    sym_to_vid: &HashMap<Symbol, VarId>,
) {
    let n = eq_roots.len();
    let eq_slice: &[Atom] = eq_roots.as_slice();
    let keep: Vec<bool> = (0..n)
        .into_par_iter()
        .map(|idx| {
            let eq = &eq_slice[idx];
            let vs: Vec<VarId> = vars_in_atom(eq, sym_to_vid).into_iter().collect();
            match is_zero(eq, &vs, cfg, id_to_sym) {
                Ok(true) => false,
                Ok(false) | Err(_) => true,
            }
        })
        .collect();
    *eq_roots = eq_roots
        .iter()
        .zip(&keep)
        .filter_map(|(e, &k)| k.then_some(e.clone()))
        .collect();
}

#[allow(clippy::too_many_arguments)]
fn apply_one_elimination(
    eq_roots: &mut Vec<Atom>,
    subst: &mut SubstitutionMap,
    x: VarId,
    i: usize,
    remaining: &BTreeSet<VarId>,
    cfg: &SampleConfig,
    id_to_sym: &[Symbol],
    sym_to_vid: &HashMap<Symbol, VarId>,
) -> Result<(), EliminationError> {
    let x_sym = id_to_sym[x.0 as usize];
    let eq = &eq_roots[i];
    let zero = Atom::num(0i64);
    let one = Atom::num(1i64);
    let b_coeff = substitute_symbol(eq, x_sym, zero);
    let p_at_one = substitute_symbol(eq, x_sym, one);
    let a_coeff = p_at_one.sub(&b_coeff).expand();

    let free: Vec<VarId> = remaining.iter().filter(|&&v| v != x).copied().collect();
    let a_is_zero = is_zero(&a_coeff, &free, cfg, id_to_sym)?;
    let b_is_zero = is_zero(&b_coeff, &free, cfg, id_to_sym)?;

    if a_is_zero && !b_is_zero {
        return Err(EliminationError::IllegitimateElimination { var: x });
    }
    if a_is_zero && b_is_zero {
        let eq_slice: &[Atom] = eq_roots.as_slice();
        *eq_roots = (0..eq_slice.len())
            .into_par_iter()
            .filter(|&j| j != i)
            .map(|j| eq_slice[j].clone())
            .collect();
        retain_nonzero_equations(eq_roots, cfg, id_to_sym, sym_to_vid);
        simplify_equations(eq_roots);
        return Ok(());
    }

    let neg_a = Atom::num(0i64).sub(&a_coeff).expand();
    let x_sol = b_coeff.div(&neg_a).expand();

    let x_sol_vars: Vec<VarId> = vars_in_atom(&x_sol, sym_to_vid).into_iter().collect();
    if !expr_finite_at_some_random_sample(&x_sol, &x_sol_vars, cfg, id_to_sym) {
        return Err(EliminationError::DegenerateSubstitution { var: x });
    }

    subst.insert(x, x_sol.clone());

    let x_sol_cloned = x_sol.clone();
    let eq_slice: &[Atom] = eq_roots.as_slice();
    *eq_roots = (0..eq_slice.len())
        .into_par_iter()
        .filter(|&j| j != i)
        .map(|j| substitute_symbol(&eq_slice[j], x_sym, x_sol_cloned.clone()).expand())
        .collect();

    retain_nonzero_equations(eq_roots, cfg, id_to_sym, sym_to_vid);
    simplify_equations(eq_roots);
    Ok(())
}
