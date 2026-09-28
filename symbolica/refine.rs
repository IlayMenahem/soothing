//! Recursive refinement of minimal hitting sets (Algorithm `Refine`).
//!
//! Each hitting set is interreduced and factored again; its own minimal
//! hitting sets replace it until the set no longer changes. The resulting
//! terminal hitting sets are shared by factor branching and variable zeroing.
//! Interreduction and factorization are supplied by the caller through
//! [`RefineOps`] so that they can run in a worker process with a deadline.

use crate::cover::{build_factor_index, decode_cover, minimal_covers, ordered_cover_masks};
use crate::factor_prep::{PolyExtLex, interreduce_polys, try_factor_groups_from_polys};

/// Counters describing one refinement run.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RefinementStats {
    pub passes: usize,
    pub subdivisions: usize,
    pub max_depth: usize,
}

fn normalize_candidate(polynomials: Vec<PolyExtLex>) -> Vec<PolyExtLex> {
    polynomials
        .into_iter()
        .filter(|polynomial| !polynomial.is_zero())
        .map(PolyExtLex::make_monic)
        .fold(Vec::new(), |mut normalized, polynomial| {
            if !normalized.contains(&polynomial) {
                normalized.push(polynomial);
            }
            normalized
        })
}

fn same_polynomial_set(left: &[PolyExtLex], right: &[PolyExtLex]) -> bool {
    left.len() == right.len()
        && left.iter().all(|polynomial| right.contains(polynomial))
        && right.iter().all(|polynomial| left.contains(polynomial))
}

/// Factor groups of a polynomial set: one group of irreducible factors per
/// nonzero polynomial.
pub type FactorGroups = Vec<Vec<PolyExtLex>>;

/// The polynomial operations refinement performs. Implementors decide where,
/// and under which limits, they run.
pub trait RefineOps {
    type Error: From<String>;

    fn interreduce(&mut self, polynomials: Vec<PolyExtLex>)
    -> Result<Vec<PolyExtLex>, Self::Error>;

    fn factor(&mut self, polynomials: Vec<PolyExtLex>) -> Result<FactorGroups, Self::Error>;
}

/// Runs both operations in the calling process, without a deadline.
struct InProcess;

impl RefineOps for InProcess {
    type Error = String;

    fn interreduce(&mut self, polynomials: Vec<PolyExtLex>) -> Result<Vec<PolyExtLex>, String> {
        Ok(interreduce_polys(polynomials))
    }

    fn factor(&mut self, polynomials: Vec<PolyExtLex>) -> Result<FactorGroups, String> {
        try_factor_groups_from_polys(polynomials)
    }
}

fn factor_cover_candidates<O: RefineOps>(
    polynomials: Vec<PolyExtLex>,
    ops: &mut O,
) -> Result<Vec<Vec<PolyExtLex>>, O::Error> {
    let groups = ops.factor(polynomials)?;
    let (factors, index_groups) = build_factor_index(&groups);
    let covers = minimal_covers(&index_groups, factors.len()).ok_or_else(|| {
        O::Error::from(format!(
            "too many distinct factors ({}); the cover enumerator supports at most 63",
            factors.len()
        ))
    })?;
    Ok(ordered_cover_masks(&covers)
        .into_iter()
        .map(|cover| decode_cover(&factors, cover))
        .collect())
}

fn push_unique(terminal: &mut Vec<Vec<PolyExtLex>>, candidate: Vec<PolyExtLex>) {
    if !terminal
        .iter()
        .any(|existing| same_polynomial_set(existing, &candidate))
    {
        terminal.push(candidate);
    }
}

fn refine_cover_with<O: RefineOps>(
    candidate: Vec<PolyExtLex>,
    interreduce: bool,
    depth: usize,
    ancestors: &mut Vec<Vec<PolyExtLex>>,
    terminal: &mut Vec<Vec<PolyExtLex>>,
    stats: &mut RefinementStats,
    ops: &mut O,
) -> Result<(), O::Error> {
    stats.passes += 1;
    stats.max_depth = stats.max_depth.max(depth);

    let candidate = normalize_candidate(candidate);
    let reduced = if interreduce {
        ops.interreduce(candidate.clone())?
    } else {
        candidate.clone()
    };
    let children = factor_cover_candidates(reduced.clone(), ops)?;

    if children.is_empty() {
        return Ok(());
    }

    let reduced = normalize_candidate(reduced);
    if children.len() == 1 && same_polynomial_set(&children[0], &reduced) {
        push_unique(terminal, children.into_iter().next().expect("one child"));
        return Ok(());
    }

    if children.len() > 1 {
        stats.subdivisions += children.len() - 1;
    }
    ancestors.push(candidate);
    for child in children {
        if ancestors
            .iter()
            .any(|ancestor| same_polynomial_set(ancestor, &child))
        {
            push_unique(terminal, child);
        } else {
            refine_cover_with(
                child,
                interreduce,
                depth + 1,
                ancestors,
                terminal,
                stats,
                ops,
            )?;
        }
    }
    ancestors.pop();
    Ok(())
}

/// Refine every minimal hitting set in `covers` into terminal hitting sets,
/// ordered by increasing number of generators. Factorization runs in-process.
pub fn refine_covers(
    factors: &[PolyExtLex],
    covers: &[u64],
    interreduce: bool,
) -> Result<(Vec<Vec<PolyExtLex>>, RefinementStats), String> {
    refine_covers_with(factors, covers, interreduce, &mut InProcess)
}

/// [`refine_covers`] with caller-supplied operations, e.g. ones that run in
/// an isolated worker under a deadline.
pub fn refine_covers_with<O: RefineOps>(
    factors: &[PolyExtLex],
    covers: &[u64],
    interreduce: bool,
    ops: &mut O,
) -> Result<(Vec<Vec<PolyExtLex>>, RefinementStats), O::Error> {
    let mut terminal = Vec::new();
    let mut stats = RefinementStats::default();
    for &cover in covers {
        refine_cover_with(
            decode_cover(factors, cover),
            interreduce,
            1,
            &mut Vec::new(),
            &mut terminal,
            &mut stats,
            ops,
        )?;
    }
    terminal.sort_by_key(Vec::len);
    Ok((terminal, stats))
}
