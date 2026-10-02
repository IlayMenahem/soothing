//! Recursive refinement of minimal hitting sets (Algorithm `Refinement`,
//! `alg:Refine` in `paper/sections/method.tex`).
//!
//! Each hitting set is interreduced and factored again; its own minimal
//! hitting sets replace it until the set no longer changes. The resulting
//! terminal hitting sets, less those whose zero set is certified to lie in
//! that of another (every generator of the other reduces to zero modulo
//! them), are shared by factor branching and variable zeroing. Siblings in
//! the refinement tree are not compared, which keeps pruning cheap when
//! refinement returns many root-level hitting sets.
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

/// A terminal hitting set and the refinement node whose child it is.
struct TerminalBranch {
    generators: Vec<PolyExtLex>,
    parent: usize,
}

/// What a refinement run accumulates: the terminal branches, the counters,
/// and the number of refinement-tree nodes named so far (the root is node 0).
#[derive(Default)]
struct RefinementState {
    terminal: Vec<TerminalBranch>,
    stats: RefinementStats,
    nodes: usize,
}

impl RefinementState {
    fn push_unique(&mut self, generators: Vec<PolyExtLex>, parent: usize) {
        if !self
            .terminal
            .iter()
            .any(|existing| same_polynomial_set(&existing.generators, &generators))
        {
            self.terminal.push(TerminalBranch { generators, parent });
        }
    }
}

fn refine_cover_with<O: RefineOps>(
    candidate: Vec<PolyExtLex>,
    depth: usize,
    parent: usize,
    ancestors: &mut Vec<Vec<PolyExtLex>>,
    state: &mut RefinementState,
    ops: &mut O,
) -> Result<(), O::Error> {
    state.stats.passes += 1;
    state.stats.max_depth = state.stats.max_depth.max(depth);

    let candidate = normalize_candidate(candidate);
    let reduced = ops.interreduce(candidate.clone())?;
    let children = factor_cover_candidates(reduced.clone(), ops)?;

    if children.is_empty() {
        return Ok(());
    }

    let reduced = normalize_candidate(reduced);
    if children.len() == 1 && same_polynomial_set(&children[0], &reduced) {
        state.push_unique(children.into_iter().next().expect("one child"), parent);
        return Ok(());
    }

    if children.len() > 1 {
        state.stats.subdivisions += children.len() - 1;
    }
    state.nodes += 1;
    let node = state.nodes;
    ancestors.push(candidate);
    for child in children {
        if ancestors
            .iter()
            .any(|ancestor| same_polynomial_set(ancestor, &child))
        {
            state.push_unique(child, node);
        } else {
            refine_cover_with(child, depth + 1, node, ancestors, state, ops)?;
        }
    }
    ancestors.pop();
    Ok(())
}

/// The variables of each generator's leading monomial, as a bitmask; `None`
/// when the ring has more than 64 variables.
fn leading_supports(branch: &[PolyExtLex]) -> Option<Vec<u64>> {
    branch
        .iter()
        .map(|polynomial| {
            (polynomial.nvars() <= 64).then(|| {
                polynomial
                    .max_exp()
                    .iter()
                    .enumerate()
                    .filter(|&(_, &exponent)| exponent > 0)
                    .fold(0u64, |mask, (variable, _)| mask | (1u64 << variable))
            })
        })
        .collect()
}

/// A terminal branch with the leading-monomial supports of its generators.
struct PruneCandidate {
    branch: TerminalBranch,
    leading_supports: Option<Vec<u64>>,
}

/// A necessary condition for every generator of `other` to reduce to zero
/// modulo `branch`: each leading monomial of `other` must be divisible by one
/// of `branch`, so in particular contain all its variables.
fn may_be_contained(branch: &PruneCandidate, other: &PruneCandidate) -> bool {
    match (&branch.leading_supports, &other.leading_supports) {
        (Some(divisors), Some(monomials)) => monomials
            .iter()
            .all(|&monomial| divisors.iter().any(|&divisor| divisor & !monomial == 0)),
        _ => true,
    }
}

/// Whether every generator of `other` reduces to zero modulo `branch`, which
/// certifies `V(branch) ⊆ V(other)`. Siblings, distinct minimal hitting sets
/// of the same factor sets, are never compared; the cheap necessary test on
/// leading monomials runs before any reduction.
fn certified_contained(branch: &PruneCandidate, other: &PruneCandidate) -> bool {
    branch.branch.parent != other.branch.parent
        && may_be_contained(branch, other)
        && other.branch.generators.iter().all(|polynomial| {
            branch.branch.generators.contains(polynomial)
                || polynomial.reduce(&branch.branch.generators).is_zero()
        })
}

/// Drop every branch whose zero set is certified to lie in that of a kept
/// branch, preserving the union of the zero sets. Branches are visited in
/// order; a newly kept branch evicts the kept branches it certifiably
/// contains, so every dropped branch stays covered by a kept one.
fn prune_contained(branches: Vec<TerminalBranch>) -> Vec<Vec<PolyExtLex>> {
    branches
        .into_iter()
        .map(|branch| PruneCandidate {
            leading_supports: leading_supports(&branch.generators),
            branch,
        })
        .fold(Vec::new(), |kept: Vec<PruneCandidate>, branch| {
            if kept
                .iter()
                .any(|existing| certified_contained(&branch, existing))
            {
                return kept;
            }
            let mut kept: Vec<PruneCandidate> = kept
                .into_iter()
                .filter(|existing| !certified_contained(existing, &branch))
                .collect();
            kept.push(branch);
            kept
        })
        .into_iter()
        .map(|candidate| candidate.branch.generators)
        .collect()
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
    let mut state = RefinementState::default();
    for &cover in covers {
        refine_cover_with(
            decode_cover(factors, cover),
            1,
            0,
            &mut Vec::new(),
            &mut state,
            ops,
        )?;
    }
    let RefinementState {
        mut terminal,
        mut stats,
        ..
    } = state;
    terminal.sort_by_key(|branch| branch.generators.len());
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

        let kept = prune_contained(cousins(&branches));

        assert_eq!(kept, vec![branches[1].clone(), branches[2].clone()]);
    }

    #[test]
    fn prune_uses_reduction_not_only_membership() {
        // x*y lies in ⟨x⟩, so V(x) ⊆ V(x*y) although x*y is not a generator.
        let branches = systems(&["x == 0", "x*y == 0"]);

        let kept = prune_contained(cousins(&branches));

        assert_eq!(kept, vec![branches[1].clone()]);
    }

    #[test]
    fn prune_keeps_one_of_equal_zero_sets() {
        let branches = systems(&["x + y == 0, y == 0", "x == 0, y == 0"]);

        assert_eq!(prune_contained(cousins(&branches)).len(), 1);
    }

    #[test]
    fn prune_keeps_incomparable_branches() {
        let branches = systems(&["x == 0", "y == 0"]);

        assert_eq!(prune_contained(cousins(&branches)), branches);
    }

    #[test]
    fn prune_does_not_compare_siblings() {
        let branches = systems(&["x == 0", "x*y == 0"]);
        let siblings = branches
            .iter()
            .map(|generators| TerminalBranch {
                generators: generators.clone(),
                parent: 1,
            })
            .collect();

        assert_eq!(prune_contained(siblings), branches);
    }

    /// Each branch as the child of its own refinement node.
    fn cousins(branches: &[Vec<PolyExtLex>]) -> Vec<TerminalBranch> {
        branches
            .iter()
            .enumerate()
            .map(|(index, generators)| TerminalBranch {
                generators: generators.clone(),
                parent: index + 1,
            })
            .collect()
    }
}
