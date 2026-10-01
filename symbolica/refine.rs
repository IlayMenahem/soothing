//! Recursive refinement of minimal hitting sets (Algorithm `Refinement`,
//! `alg:Refine` in `paper/sections/method.tex`).
//!
//! Each hitting set is interreduced and factored again; its own minimal
//! hitting sets replace it until the set no longer changes. The resulting
//! terminal hitting sets, less those whose zero set is certified to lie in
//! that of another (every generator of the other reduces to zero modulo
//! them), are shared by factor branching and variable zeroing.
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
    /// Terminal hitting sets dropped because their zero set lies in that of
    /// another terminal hitting set.
    pub pruned: usize,
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
    depth: usize,
    ancestors: &mut Vec<Vec<PolyExtLex>>,
    terminal: &mut Vec<Vec<PolyExtLex>>,
    stats: &mut RefinementStats,
    ops: &mut O,
) -> Result<(), O::Error> {
    stats.passes += 1;
    stats.max_depth = stats.max_depth.max(depth);

    let candidate = normalize_candidate(candidate);
    let reduced = ops.interreduce(candidate.clone())?;
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
            refine_cover_with(child, depth + 1, ancestors, terminal, stats, ops)?;
        }
    }
    ancestors.pop();
    Ok(())
}

/// Whether every generator of `other` reduces to zero modulo `branch`, which
/// certifies `V(branch) ⊆ V(other)`.
fn certified_contained(branch: &[PolyExtLex], other: &[PolyExtLex]) -> bool {
    other
        .iter()
        .all(|polynomial| branch.contains(polynomial) || polynomial.reduce(branch).is_zero())
}

/// Drop every branch whose zero set is certified to lie in that of a kept
/// branch, preserving the union of the zero sets. Branches are visited in
/// order; a newly kept branch evicts the kept branches it certifiably
/// contains, so every dropped branch stays covered by a kept one.
fn prune_contained(branches: Vec<Vec<PolyExtLex>>) -> Vec<Vec<PolyExtLex>> {
    branches.into_iter().fold(Vec::new(), |kept, branch| {
        if kept
            .iter()
            .any(|existing: &Vec<PolyExtLex>| certified_contained(&branch, existing))
        {
            return kept;
        }
        let mut kept: Vec<Vec<PolyExtLex>> = kept
            .into_iter()
            .filter(|existing| !certified_contained(existing, &branch))
            .collect();
        kept.push(branch);
        kept
    })
}

/// Refine every minimal hitting set in `covers` into terminal hitting sets,
/// drop those whose zero set lies in that of another, and order the rest by
/// increasing number of generators. Factorization runs in-process.
pub fn refine_covers(
    factors: &[PolyExtLex],
    covers: &[u64],
) -> Result<(Vec<Vec<PolyExtLex>>, RefinementStats), String> {
    refine_covers_with(factors, covers, &mut InProcess)
}

/// [`refine_covers`] with caller-supplied operations, e.g. ones that run in
/// an isolated worker under a deadline.
pub fn refine_covers_with<O: RefineOps>(
    factors: &[PolyExtLex],
    covers: &[u64],
    ops: &mut O,
) -> Result<(Vec<Vec<PolyExtLex>>, RefinementStats), O::Error> {
    let mut terminal = Vec::new();
    let mut stats = RefinementStats::default();
    for &cover in covers {
        refine_cover_with(
            decode_cover(factors, cover),
            1,
            &mut Vec::new(),
            &mut terminal,
            &mut stats,
            ops,
        )?;
    }
    terminal.sort_by_key(Vec::len);
    let count = terminal.len();
    let mut terminal = prune_contained(terminal);
    stats.pruned = count - terminal.len();
    terminal.sort_by_key(Vec::len);
    Ok((terminal, stats))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Once;

    use symbolica::domains::rational::Q;
    use symbolica::license::LicenseManager;

    use crate::factor_prep::polynomials_from_loaded;
    use crate::mathematica::load_mathematica_system_from_str_with_var_names;

    static LICENSE: Once = Once::new();

    /// The generators of each system, all over the ring of their union.
    fn systems(sources: &[&str]) -> Vec<Vec<PolyExtLex>> {
        LICENSE.call_once(|| {
            let _ = LicenseManager::set_license_key(crate::SYMBOLICA_LICENSE_KEY);
        });
        let joined = format!("{{{}}}", sources.join(", "));
        let loaded = load_mathematica_system_from_str_with_var_names(&joined).expect("load");
        let mut all = polynomials_from_loaded(&loaded, &Q)
            .expect("polynomials")
            .into_iter();
        sources
            .iter()
            .map(|source| {
                all.by_ref()
                    .take(source.matches("==").count())
                    .map(PolyExtLex::make_monic)
                    .collect()
            })
            .collect()
    }

    #[test]
    fn prune_drops_a_branch_inside_another() {
        let branches = systems(&["x == 0, y == 0", "x == 0", "y - 1 == 0"]);

        let kept = prune_contained(branches.clone());

        assert_eq!(kept, vec![branches[1].clone(), branches[2].clone()]);
    }

    #[test]
    fn prune_uses_reduction_not_only_membership() {
        // x*y lies in ⟨x⟩, so V(x) ⊆ V(x*y) although x*y is not a generator.
        let branches = systems(&["x == 0", "x*y == 0"]);

        let kept = prune_contained(branches.clone());

        assert_eq!(kept, vec![branches[1].clone()]);
    }

    #[test]
    fn prune_keeps_one_of_equal_zero_sets() {
        let branches = systems(&["x + y == 0, y == 0", "x == 0, y == 0"]);

        assert_eq!(prune_contained(branches).len(), 1);
    }

    #[test]
    fn prune_keeps_incomparable_branches() {
        let branches = systems(&["x == 0", "y == 0"]);

        assert_eq!(prune_contained(branches.clone()), branches);
    }
}
