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
//!
//! Siblings are refined in order, and a branch carries nonzero side
//! conditions: a factor `f` with `H_k \ H_j = {f}` for an earlier sibling
//! `H_k` may be assumed nonzero on `H_j`, since a common zero of `H_j` and `f`
//! is a zero of `H_k`. Descendants inherit the conditions. A branch is dropped
//! once a condition reduces to zero modulo it, and the conditions are removed
//! from every factor set before hitting sets are enumerated. The conditions
//! only discard points that an earlier branch keeps, so the terminal zero sets
//! still cover `V(P)`; the solver never sees them.

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
    /// Branches dropped because a nonzero side condition vanishes on them.
    pub condition_kills: usize,
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

/// For each cover, in order, the factors `f` such that some earlier cover
/// exceeds it by exactly `f`, as a bitmask over the same factor universe.
fn sibling_conditions(covers: &[u64]) -> Vec<u64> {
    covers
        .iter()
        .enumerate()
        .map(|(index, &cover)| {
            covers[..index]
                .iter()
                .map(|&earlier| earlier & !cover)
                .filter(|extra| extra.count_ones() == 1)
                .fold(0u64, |conditions, extra| conditions | extra)
        })
        .collect()
}

/// A hitting set to refine and the nonzero side conditions it carries.
struct Branch {
    generators: Vec<PolyExtLex>,
    conditions: Vec<PolyExtLex>,
}

/// The ordered covers of `factors` as branches, each carrying the parent's
/// conditions together with its own sibling conditions.
fn branches_of_covers(
    factors: &[PolyExtLex],
    covers: &[u64],
    parent_conditions: &[PolyExtLex],
) -> Vec<Branch> {
    covers
        .iter()
        .zip(sibling_conditions(covers))
        .map(|(&cover, conditions)| Branch {
            generators: decode_cover(factors, cover),
            conditions: union(parent_conditions, decode_cover(factors, conditions)),
        })
        .collect()
}

fn union(left: &[PolyExtLex], right: Vec<PolyExtLex>) -> Vec<PolyExtLex> {
    right
        .into_iter()
        .fold(left.to_vec(), |mut all, polynomial| {
            if !all.contains(&polynomial) {
                all.push(polynomial);
            }
            all
        })
}

/// Each factor group less the factors assumed nonzero.
fn without_conditions(groups: FactorGroups, conditions: &[PolyExtLex]) -> FactorGroups {
    groups
        .into_iter()
        .map(|group| {
            group
                .into_iter()
                .filter(|factor| !conditions.contains(&factor.clone().make_monic()))
                .collect()
        })
        .collect()
}

fn factor_cover_candidates<O: RefineOps>(
    polynomials: Vec<PolyExtLex>,
    conditions: &[PolyExtLex],
    ops: &mut O,
) -> Result<Vec<Branch>, O::Error> {
    let groups = without_conditions(ops.factor(polynomials)?, conditions);
    let (factors, index_groups) = build_factor_index(&groups);
    let covers = minimal_covers(&index_groups, factors.len()).ok_or_else(|| {
        O::Error::from(format!(
            "too many distinct factors ({}); the cover enumerator supports at most 63",
            factors.len()
        ))
    })?;
    Ok(branches_of_covers(
        &factors,
        &ordered_cover_masks(&covers),
        conditions,
    ))
}

/// Whether some condition lies in the ideal of `generators`, so that no
/// point of the branch has all its conditions nonzero.
fn violates_condition(generators: &[PolyExtLex], conditions: &[PolyExtLex]) -> bool {
    conditions
        .iter()
        .any(|condition| condition.reduce(generators).is_zero())
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
    branch: Branch,
    depth: usize,
    parent: usize,
    ancestors: &mut Vec<Vec<PolyExtLex>>,
    state: &mut RefinementState,
    ops: &mut O,
) -> Result<(), O::Error> {
    state.stats.passes += 1;
    state.stats.max_depth = state.stats.max_depth.max(depth);

    let Branch {
        generators,
        conditions,
    } = branch;
    let candidate = normalize_candidate(generators);
    let reduced = ops.interreduce(candidate.clone())?;
    if violates_condition(&reduced, &conditions) {
        state.stats.condition_kills += 1;
        return Ok(());
    }
    let children = factor_cover_candidates(reduced.clone(), &conditions, ops)?;

    if children.is_empty() {
        return Ok(());
    }

    let reduced = normalize_candidate(reduced);
    if children.len() == 1 && same_polynomial_set(&children[0].generators, &reduced) {
        let child = children.into_iter().next().expect("one child");
        state.push_unique(child.generators, parent);
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
            .any(|ancestor| same_polynomial_set(ancestor, &child.generators))
        {
            state.push_unique(child.generators, node);
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
    for branch in branches_of_covers(factors, covers, &[]) {
        refine_cover_with(branch, 1, 0, &mut Vec::new(), &mut state, ops)?;
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
    fn sibling_conditions_take_single_extra_factors_of_earlier_covers() {
        // {0}, {1, 2}, {0, 3}: {0} \ {1, 2} = {0} gives the second cover the
        // condition 0; neither earlier cover exceeds {0, 3} by one factor.
        assert_eq!(
            sibling_conditions(&[0b0001, 0b0110, 0b1001]),
            vec![0, 0b0001, 0]
        );
    }

    #[test]
    fn side_condition_drops_a_branch_whose_points_an_earlier_sibling_keeps() {
        // x*(y - x) = x*(y + x) = 0 has the hitting sets {x} and
        // {y - x, y + x}; the second assumes x ≠ 0 but interreduces to {x, y}.
        let polynomials = systems(&["x*(y - x) == 0, x*(y + x) == 0"]).remove(0);
        let x = systems(&["x == 0, y == 0"]).remove(0).remove(0);
        let groups = try_factor_groups_from_polys(polynomials).expect("factor");
        let (factors, index_groups) = build_factor_index(&groups);
        let covers = minimal_covers(&index_groups, factors.len()).expect("covers");

        let (terminal, stats) =
            refine_covers(&factors, &ordered_cover_masks(&covers)).expect("refine");

        assert_eq!(terminal, vec![vec![x]]);
        assert_eq!(stats.condition_kills, 1);
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
