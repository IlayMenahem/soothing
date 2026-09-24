//! Recursive refinement of minimal hitting sets (Algorithm `Refine`).
//!
//! Each hitting set is interreduced and factored again; its own minimal
//! hitting sets replace it until the set no longer changes. The resulting
//! terminal hitting sets are shared by factor branching and variable zeroing.

use crate::cover::{build_factor_index, decode_cover, minimal_covers, ordered_cover_masks};
use crate::factor_prep::{PolyExtLex, interreduce_polys, try_factor_groups_from_polys};
use crate::gb_search::polynomial_set_contains;

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
        && polynomial_set_contains(left, right)
        && polynomial_set_contains(right, left)
}

fn factor_cover_candidates(polynomials: Vec<PolyExtLex>) -> Result<Vec<Vec<PolyExtLex>>, String> {
    let groups = try_factor_groups_from_polys(polynomials)?;
    let (factors, index_groups) = build_factor_index(&groups);
    let covers = minimal_covers(&index_groups, factors.len()).ok_or_else(|| {
        format!(
            "too many distinct factors ({}); the cover enumerator supports at most 63",
            factors.len()
        )
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

/// Refine one hitting set, appending its terminal hitting sets to `terminal`.
pub fn refine_cover(
    candidate: Vec<PolyExtLex>,
    interreduce: bool,
    depth: usize,
    ancestors: &mut Vec<Vec<PolyExtLex>>,
    terminal: &mut Vec<Vec<PolyExtLex>>,
    stats: &mut RefinementStats,
) -> Result<(), String> {
    stats.passes += 1;
    stats.max_depth = stats.max_depth.max(depth);

    let candidate = normalize_candidate(candidate);
    let reduced = if interreduce {
        interreduce_polys(candidate.clone())
    } else {
        candidate.clone()
    };
    let children = factor_cover_candidates(reduced.clone())?;

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
            refine_cover(child, interreduce, depth + 1, ancestors, terminal, stats)?;
        }
    }
    ancestors.pop();
    Ok(())
}

/// Refine every minimal hitting set in `covers` into terminal hitting sets,
/// ordered by increasing number of generators.
pub fn refine_covers(
    factors: &[PolyExtLex],
    covers: &[u64],
    interreduce: bool,
) -> Result<(Vec<Vec<PolyExtLex>>, RefinementStats), String> {
    let mut terminal = Vec::new();
    let mut stats = RefinementStats::default();
    for &cover in covers {
        refine_cover(
            decode_cover(factors, cover),
            interreduce,
            1,
            &mut Vec::new(),
            &mut terminal,
            &mut stats,
        )?;
    }
    terminal.sort_by_key(Vec::len);
    Ok((terminal, stats))
}
