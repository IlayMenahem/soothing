//! Recursive factor branching with variable zeroing.
//!
//! This implements Algorithm `ZeroingBranching` from
//! `paper/sections/variable_zeroing.tex`. A factor cover that exceeds its
//! Gröbner-basis time budget is split into the constructible cases `x = 0`
//! and `x != 0`. The latter is encoded by the Rabinowitsch equation
//! `1 - x*z = 0`, where `z` is fresh. If one side of a split has no factor
//! cover and the other has a unique cover, the unique cover is split again
//! without a redundant Gröbner-basis call. Only original-variable basis
//! elements and the zero/nonzero decisions are exposed by the projected
//! result.

use std::time::{Duration, Instant};

use symbolica::poly::PolyVariable;

use crate::cover::{build_factor_index, decode_cover, minimal_covers, ordered_cover_masks};
use crate::factor_prep::{PolyExtLex, interreduce_polys, try_factor_groups_from_polys};
use crate::gb_core::iteratively_refined_unsat_core;
use crate::gb_search::{
    ComputedBasis, Core, GbBackend, Outcome, PolyExtGrevLex, evaluate_cover,
    polynomial_set_contains, solution_dimension,
};

/// An original variable that may be selected for zeroing.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ZeroingVariable {
    pub name: String,
    pub variable: PolyVariable,
}

/// A constructible-set condition accumulated on the route to a component.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SplitDecision {
    pub variable: String,
    pub is_zero: bool,
}

/// Search controls for recursive variable zeroing.
#[derive(Clone, Debug)]
pub struct VariableZeroingOptions {
    /// Apply the optional interreduction in every invocation of Branching.
    pub interreduce: bool,
    /// Per-cover Gröbner-basis time budget.
    pub timeout: Duration,
    /// Optional wall-clock budget for the entire recursive search.
    pub max_runtime: Option<Duration>,
    pub gb_backend: GbBackend,
    /// Learn unsatisfiable cores and use them to prune later covers.
    pub unsat_core_search: bool,
    /// Global recursion-node budget. Exhaustion is reported, never treated as
    /// unsatisfiability.
    pub max_nodes: usize,
    /// Global cover-evaluation budget.
    pub max_evaluations: usize,
    /// Maximum number of zeroing decisions on one route.
    pub max_depth: usize,
    /// Emit one compact progress line per evaluated cover.
    pub verbose: bool,
}

impl Default for VariableZeroingOptions {
    fn default() -> Self {
        Self {
            interreduce: true,
            timeout: Duration::from_secs(5),
            max_runtime: None,
            gb_backend: GbBackend::M4gb,
            unsat_core_search: true,
            max_nodes: usize::MAX,
            max_evaluations: usize::MAX,
            max_depth: usize::MAX,
            verbose: false,
        }
    }
}

/// A solved locally closed component, projected onto the original variables.
pub struct ProjectedComponent {
    pub node_id: usize,
    pub decisions: Vec<SplitDecision>,
    pub dimension: Option<usize>,
    basis: ComputedBasis,
}

impl ProjectedComponent {
    #[must_use]
    pub fn basis_len(&self) -> usize {
        self.basis.elements.len()
    }

    #[must_use]
    pub fn leading_monomials_only(&self) -> bool {
        self.basis.leading_monomials_only
    }

    /// Basis entries not involving a fresh Rabinowitsch variable.
    ///
    /// Together with `decisions`, these form the printable projection view.
    /// The nonzero decisions are essential: an open condition such as `x != 0`
    /// cannot in general be represented by polynomial equalities alone.
    pub fn original_basis_elements(&self) -> impl Iterator<Item = &PolyExtGrevLex> {
        self.basis
            .elements
            .iter()
            .filter(|polynomial| !uses_temporary_variable(polynomial))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UnresolvedReason {
    TimeoutAfterVariablesExhausted,
    NoUsefulVariable,
    NodeBudget,
    EvaluationBudget,
    DepthBudget,
    RuntimeBudget,
}

/// A branch whose variety is still part of the answer but was not solved.
pub struct UnresolvedBranch {
    pub decisions: Vec<SplitDecision>,
    pub polynomials: Vec<PolyExtLex>,
    pub remaining_variables: Vec<ZeroingVariable>,
    pub reason: UnresolvedReason,
}

#[derive(Clone, Debug, Default)]
pub struct VariableZeroingStats {
    pub nodes: usize,
    pub covers: usize,
    pub evaluations: usize,
    pub consistent: usize,
    pub inconsistent: usize,
    pub core_pruned: usize,
    /// All timed-out evaluations, including those subsequently resolved by a
    /// zeroing split.
    pub timeouts: usize,
    pub zeroing_splits: usize,
    pub largest_depth: usize,
}

/// Outcome recorded for one hitting-set check in the recursion tree.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VariableZeroingCoverOutcome {
    Consistent,
    Inconsistent,
    CorePruned,
    Timeout,
    /// The solver call was bypassed because the sibling branch had no cover
    /// and this branch had a unique cover.
    Contracted,
}

/// Trace information for one hitting set considered at a recursion node.
#[derive(Clone, Debug)]
pub struct VariableZeroingCoverResult {
    pub cover_index: usize,
    pub outcome: VariableZeroingCoverOutcome,
    /// Time spent deciding this hitting set, excluding recursive child work.
    pub elapsed: Duration,
    /// Variable selected when a timed-out check was recursively split.
    pub split_variable: Option<String>,
    /// Recursion node reached by the `split_variable = 0` branch, if created.
    pub zero_child: Option<usize>,
    /// Recursion node reached by the `split_variable != 0` branch, if created.
    pub nonzero_child: Option<usize>,
}

/// Work performed while checking one recursion node.
///
/// `elapsed` excludes time spent recursively checking child nodes, so summing
/// this field does not count nested work more than once.
#[derive(Clone, Debug)]
pub struct VariableZeroingNodeResult {
    pub node_id: usize,
    pub decisions: Vec<SplitDecision>,
    pub covers: usize,
    pub evaluations: usize,
    pub consistent: usize,
    pub inconsistent: usize,
    pub core_pruned: usize,
    pub timeouts: usize,
    pub zeroing_splits: usize,
    pub elapsed: Duration,
    /// Hitting sets in their evaluation order. This may be shorter than
    /// `covers` when a global budget stops the node early.
    pub checks: Vec<VariableZeroingCoverResult>,
}

pub struct VariableZeroingResult {
    pub components: Vec<ProjectedComponent>,
    pub unresolved: Vec<UnresolvedBranch>,
    /// Unsatisfiable cores learned anywhere in the recursive search.
    pub cores: Vec<Core>,
    pub nodes: Vec<VariableZeroingNodeResult>,
    pub stats: VariableZeroingStats,
    pub first_solution: Option<Duration>,
    pub elapsed: Duration,
}

impl VariableZeroingResult {
    #[must_use]
    pub fn complete(&self) -> bool {
        self.unresolved.is_empty()
    }
}

type Evaluator<'a> = dyn Fn(Vec<PolyExtLex>, Duration, GbBackend) -> Outcome + Send + Sync + 'a;

struct Solver<'a> {
    options: &'a VariableZeroingOptions,
    evaluator: &'a Evaluator<'a>,
    started: Instant,
    fresh_variable: usize,
    components: Vec<ProjectedComponent>,
    unresolved: Vec<UnresolvedBranch>,
    /// Shared by every recursion node so a core learned in one branch can
    /// prune covers in all subsequently visited branches.
    cores: Vec<Core>,
    nodes: Vec<VariableZeroingNodeResult>,
    stats: VariableZeroingStats,
    first_solution: Option<Duration>,
}

impl Solver<'_> {
    fn runtime_exhausted(&self) -> bool {
        self.options
            .max_runtime
            .is_some_and(|limit| self.started.elapsed() >= limit)
    }

    fn evaluation_timeout(&self) -> Duration {
        self.options
            .max_runtime
            .map_or(self.options.timeout, |limit| {
                self.options
                    .timeout
                    .min(limit.saturating_sub(self.started.elapsed()))
            })
    }

    fn retain_unresolved(
        &mut self,
        polynomials: Vec<PolyExtLex>,
        variables: &[ZeroingVariable],
        decisions: &[SplitDecision],
        reason: UnresolvedReason,
    ) {
        self.unresolved.push(UnresolvedBranch {
            decisions: decisions.to_vec(),
            polynomials,
            remaining_variables: variables.to_vec(),
            reason,
        });
    }

    fn solve_node(
        &mut self,
        polynomials: Vec<PolyExtLex>,
        variables: Vec<ZeroingVariable>,
        decisions: Vec<SplitDecision>,
    ) -> Result<Option<usize>, String> {
        self.solve_node_with_contraction(polynomials, variables, decisions, false)
    }

    fn solve_node_with_contraction(
        &mut self,
        mut polynomials: Vec<PolyExtLex>,
        variables: Vec<ZeroingVariable>,
        decisions: Vec<SplitDecision>,
        contract_unique_cover: bool,
    ) -> Result<Option<usize>, String> {
        let depth = decisions.len();
        self.stats.largest_depth = self.stats.largest_depth.max(depth);
        if self.runtime_exhausted() {
            self.retain_unresolved(
                polynomials,
                &variables,
                &decisions,
                UnresolvedReason::RuntimeBudget,
            );
            return Ok(None);
        }
        if self.stats.nodes >= self.options.max_nodes {
            self.retain_unresolved(
                polynomials,
                &variables,
                &decisions,
                UnresolvedReason::NodeBudget,
            );
            return Ok(None);
        }

        let node_id = self.stats.nodes;
        self.stats.nodes += 1;
        let node_started = Instant::now();
        let mut child_elapsed = Duration::ZERO;
        let mut node_result = VariableZeroingNodeResult {
            node_id,
            decisions: decisions.clone(),
            covers: 0,
            evaluations: 0,
            consistent: 0,
            inconsistent: 0,
            core_pruned: 0,
            timeouts: 0,
            zeroing_splits: 0,
            elapsed: Duration::ZERO,
            checks: Vec::new(),
        };
        if self.options.interreduce {
            polynomials = interreduce_polys(polynomials);
        }
        let node_nvars = polynomials.first().map_or(0, PolyExtLex::nvars);
        let groups = try_factor_groups_from_polys(polynomials)
            .map_err(|error| format!("node {node_id} factorization: {error}"))?;
        let (factors, index_groups) = build_factor_index(&groups);
        let covers = minimal_covers(&index_groups, factors.len()).ok_or_else(|| {
            format!(
                "node {node_id} has {} distinct factors; the cover enumerator supports at most 63",
                factors.len()
            )
        })?;
        let covers = ordered_cover_masks(&covers);
        self.stats.covers += covers.len();
        node_result.covers = covers.len();

        if self.options.verbose {
            println!(
                "node #{node_id} depth {depth}: {} factor group(s), {} cover(s), {} variable(s) left",
                groups.len(),
                covers.len(),
                variables.len()
            );
        }

        for (cover_index, cover_mask) in covers.iter().copied().enumerate() {
            let cover_started = Instant::now();
            let candidate = decode_cover(&factors, cover_mask);
            if self.runtime_exhausted() {
                self.retain_unresolved(
                    candidate,
                    &variables,
                    &decisions,
                    UnresolvedReason::RuntimeBudget,
                );
                for &pending_mask in &covers[cover_index + 1..] {
                    self.retain_unresolved(
                        decode_cover(&factors, pending_mask),
                        &variables,
                        &decisions,
                        UnresolvedReason::RuntimeBudget,
                    );
                }
                break;
            }
            if self.options.unsat_core_search
                && self
                    .cores
                    .iter()
                    .any(|core| polynomial_set_contains(&candidate, core))
            {
                self.stats.core_pruned += 1;
                node_result.core_pruned += 1;
                node_result.checks.push(VariableZeroingCoverResult {
                    cover_index,
                    outcome: VariableZeroingCoverOutcome::CorePruned,
                    elapsed: cover_started.elapsed(),
                    split_variable: None,
                    zero_child: None,
                    nonzero_child: None,
                });
                if self.options.verbose {
                    println!("  cover #{cover_index}: core-pruned");
                }
                continue;
            }

            // The caller has already established that this is the unique
            // cover on one side of a zero/nonzero split and that the sibling
            // has no cover. Hence this cover represents the entire parent
            // branch, and another zeroing split is sound without first asking
            // the Gröbner solver to rediscover that the branch is difficult.
            if contract_unique_cover
                && covers.len() == 1
                && depth < self.options.max_depth
                && let Some(selected_index) = choose_variable(&candidate, &variables)
            {
                let selected = variables[selected_index].clone();
                if self.options.verbose {
                    println!(
                        "  cover #{cover_index}: branch-contracted; splitting {} = 0 / {} != 0",
                        selected.name, selected.name
                    );
                }
                let (zero_child, nonzero_child, recursive_elapsed) =
                    self.split_and_solve(candidate, &variables, &decisions, selected_index)?;
                self.stats.zeroing_splits += 1;
                node_result.zeroing_splits += 1;
                child_elapsed += recursive_elapsed;
                node_result.checks.push(VariableZeroingCoverResult {
                    cover_index,
                    outcome: VariableZeroingCoverOutcome::Contracted,
                    elapsed: cover_started.elapsed().saturating_sub(recursive_elapsed),
                    split_variable: Some(selected.name),
                    zero_child,
                    nonzero_child,
                });
                continue;
            }

            if self.stats.evaluations >= self.options.max_evaluations {
                self.retain_unresolved(
                    candidate,
                    &variables,
                    &decisions,
                    UnresolvedReason::EvaluationBudget,
                );
                for &pending_mask in &covers[cover_index + 1..] {
                    self.retain_unresolved(
                        decode_cover(&factors, pending_mask),
                        &variables,
                        &decisions,
                        UnresolvedReason::EvaluationBudget,
                    );
                }
                break;
            }

            self.stats.evaluations += 1;
            node_result.evaluations += 1;
            let outcome = (self.evaluator)(
                candidate.clone(),
                self.evaluation_timeout(),
                self.options.gb_backend,
            );
            let cover_elapsed = cover_started.elapsed();
            match outcome {
                Outcome::Consistent(basis) => {
                    self.stats.consistent += 1;
                    node_result.consistent += 1;
                    self.first_solution
                        .get_or_insert_with(|| self.started.elapsed());
                    let dimension = solution_dimension(&basis.elements, node_nvars);
                    if self.options.verbose {
                        println!(
                            "  cover #{cover_index}: solved ({} basis element(s), {:?})",
                            basis.elements.len(),
                            cover_elapsed
                        );
                    }
                    node_result.checks.push(VariableZeroingCoverResult {
                        cover_index,
                        outcome: VariableZeroingCoverOutcome::Consistent,
                        elapsed: cover_elapsed,
                        split_variable: None,
                        zero_child: None,
                        nonzero_child: None,
                    });
                    self.components.push(ProjectedComponent {
                        node_id,
                        decisions: decisions.clone(),
                        dimension,
                        basis,
                    });
                }
                Outcome::Inconsistent => {
                    self.stats.inconsistent += 1;
                    node_result.inconsistent += 1;
                    if self.options.verbose {
                        println!("  cover #{cover_index}: inconsistent ({:?})", cover_elapsed);
                    }
                    node_result.checks.push(VariableZeroingCoverResult {
                        cover_index,
                        outcome: VariableZeroingCoverOutcome::Inconsistent,
                        elapsed: cover_elapsed,
                        split_variable: None,
                        zero_child: None,
                        nonzero_child: None,
                    });
                    if self.options.unsat_core_search {
                        if self.options.max_runtime.is_some() {
                            // The full candidate is already a certified core;
                            // avoid an unbounded sequence of refinements when
                            // the caller requested a wall-clock budget.
                            self.cores.push(candidate.clone());
                        } else {
                            self.cores.push(iteratively_refined_unsat_core(
                                &candidate,
                                self.options.timeout,
                            ));
                        }
                    }
                }
                Outcome::Timeout => {
                    self.stats.timeouts += 1;
                    node_result.timeouts += 1;
                    if self.runtime_exhausted() {
                        self.retain_unresolved(
                            candidate,
                            &variables,
                            &decisions,
                            UnresolvedReason::RuntimeBudget,
                        );
                        for &pending_mask in &covers[cover_index + 1..] {
                            self.retain_unresolved(
                                decode_cover(&factors, pending_mask),
                                &variables,
                                &decisions,
                                UnresolvedReason::RuntimeBudget,
                            );
                        }
                        node_result.checks.push(VariableZeroingCoverResult {
                            cover_index,
                            outcome: VariableZeroingCoverOutcome::Timeout,
                            elapsed: cover_elapsed,
                            split_variable: None,
                            zero_child: None,
                            nonzero_child: None,
                        });
                        break;
                    }
                    let Some(selected_index) = choose_variable(&candidate, &variables) else {
                        let reason = if variables.is_empty() {
                            UnresolvedReason::TimeoutAfterVariablesExhausted
                        } else {
                            UnresolvedReason::NoUsefulVariable
                        };
                        self.retain_unresolved(candidate, &variables, &decisions, reason);
                        node_result.checks.push(VariableZeroingCoverResult {
                            cover_index,
                            outcome: VariableZeroingCoverOutcome::Timeout,
                            elapsed: cover_elapsed,
                            split_variable: None,
                            zero_child: None,
                            nonzero_child: None,
                        });
                        continue;
                    };
                    if depth >= self.options.max_depth {
                        self.retain_unresolved(
                            candidate,
                            &variables,
                            &decisions,
                            UnresolvedReason::DepthBudget,
                        );
                        node_result.checks.push(VariableZeroingCoverResult {
                            cover_index,
                            outcome: VariableZeroingCoverOutcome::Timeout,
                            elapsed: cover_elapsed,
                            split_variable: None,
                            zero_child: None,
                            nonzero_child: None,
                        });
                        continue;
                    }

                    let selected = variables[selected_index].clone();
                    if self.options.verbose {
                        println!(
                            "  cover #{cover_index}: timed out ({:?}); splitting {} = 0 / {} != 0",
                            cover_elapsed, selected.name, selected.name
                        );
                    }
                    let (zero_child, nonzero_child, recursive_elapsed) =
                        self.split_and_solve(candidate, &variables, &decisions, selected_index)?;
                    self.stats.zeroing_splits += 1;
                    node_result.zeroing_splits += 1;
                    child_elapsed += recursive_elapsed;
                    node_result.checks.push(VariableZeroingCoverResult {
                        cover_index,
                        outcome: VariableZeroingCoverOutcome::Timeout,
                        elapsed: cover_elapsed,
                        split_variable: Some(selected.name.clone()),
                        zero_child,
                        nonzero_child,
                    });
                }
                Outcome::BackendError(error) => {
                    return Err(format!(
                        "node {node_id}, cover {cover_index}, {} backend: {error}",
                        self.options.gb_backend.label()
                    ));
                }
            }
        }
        node_result.elapsed = node_started.elapsed().saturating_sub(child_elapsed);
        self.nodes.push(node_result);
        Ok(Some(node_id))
    }

    fn split_and_solve(
        &mut self,
        candidate: Vec<PolyExtLex>,
        variables: &[ZeroingVariable],
        decisions: &[SplitDecision],
        selected_index: usize,
    ) -> Result<(Option<usize>, Option<usize>, Duration), String> {
        let selected = variables[selected_index].clone();
        let mut remaining = variables.to_vec();
        remaining.remove(selected_index);
        let fresh = self.fresh_variable;
        self.fresh_variable += 1;
        let (zero_branch, nonzero_branch) =
            split_on_variable(candidate, &selected.variable, fresh)?;

        let zero_cover_count = factor_cover_count(&zero_branch)?;
        let nonzero_cover_count = factor_cover_count(&nonzero_branch)?;
        let contract_zero = zero_cover_count == 1 && nonzero_cover_count == 0;
        let contract_nonzero = nonzero_cover_count == 1 && zero_cover_count == 0;

        let mut zero_decisions = decisions.to_vec();
        zero_decisions.push(SplitDecision {
            variable: selected.name.clone(),
            is_zero: true,
        });
        let children_started = Instant::now();
        let zero_child = self.solve_node_with_contraction(
            zero_branch,
            remaining.clone(),
            zero_decisions,
            contract_zero,
        )?;

        let mut nonzero_decisions = decisions.to_vec();
        nonzero_decisions.push(SplitDecision {
            variable: selected.name,
            is_zero: false,
        });
        let nonzero_child = self.solve_node_with_contraction(
            nonzero_branch,
            remaining,
            nonzero_decisions,
            contract_nonzero,
        )?;
        Ok((zero_child, nonzero_child, children_started.elapsed()))
    }
}

/// Solve a polynomial system by factor branching, recursively zeroing an
/// original variable whenever a cover times out.
pub fn solve_by_zeroing(
    polynomials: Vec<PolyExtLex>,
    variables: Vec<ZeroingVariable>,
    options: &VariableZeroingOptions,
) -> Result<VariableZeroingResult, String> {
    solve_by_zeroing_with(polynomials, variables, options, &evaluate_cover)
}

fn solve_by_zeroing_with(
    polynomials: Vec<PolyExtLex>,
    variables: Vec<ZeroingVariable>,
    options: &VariableZeroingOptions,
    evaluator: &Evaluator<'_>,
) -> Result<VariableZeroingResult, String> {
    let started = Instant::now();
    let mut solver = Solver {
        options,
        evaluator,
        started,
        fresh_variable: 0,
        components: Vec::new(),
        unresolved: Vec::new(),
        cores: Vec::new(),
        nodes: Vec::new(),
        stats: VariableZeroingStats::default(),
        first_solution: None,
    };
    solver.solve_node(polynomials, variables, Vec::new())?;
    Ok(VariableZeroingResult {
        components: solver.components,
        unresolved: solver.unresolved,
        cores: solver.cores,
        nodes: {
            solver.nodes.sort_unstable_by_key(|node| node.node_id);
            solver.nodes
        },
        stats: solver.stats,
        first_solution: solver.first_solution,
        elapsed: started.elapsed(),
    })
}

/// Choose the variable occurring in the most generators, breaking ties by
/// term incidence, total exponent, and finally the caller's stable order.
fn choose_variable(polynomials: &[PolyExtLex], variables: &[ZeroingVariable]) -> Option<usize> {
    let mut best: Option<(usize, (usize, usize, usize))> = None;
    for (candidate_index, candidate) in variables.iter().enumerate() {
        let mut polynomial_count = 0;
        let mut term_count = 0;
        let mut exponent_sum = 0;
        for polynomial in polynomials {
            let Some(position) = polynomial
                .variables
                .iter()
                .position(|variable| variable == &candidate.variable)
            else {
                continue;
            };
            let mut appears = false;
            for exponents in polynomial.exponents_iter() {
                let exponent = exponents[position] as usize;
                if exponent != 0 {
                    appears = true;
                    term_count += 1;
                    exponent_sum += exponent;
                }
            }
            polynomial_count += usize::from(appears);
        }
        let score = (polynomial_count, term_count, exponent_sum);
        if score.0 != 0
            && best
                .as_ref()
                .is_none_or(|(_, best_score)| score > *best_score)
        {
            best = Some((candidate_index, score));
        }
    }
    best.map(|(index, _)| index)
}

/// Construct and interreduce `Q + {x}` and `Q + {1 - x*z}`.
fn split_on_variable(
    mut candidate: Vec<PolyExtLex>,
    variable: &PolyVariable,
    fresh_id: usize,
) -> Result<(Vec<PolyExtLex>, Vec<PolyExtLex>), String> {
    let template = candidate
        .first()
        .ok_or_else(|| "cannot zero a variable in an empty cover".to_string())?;
    let zero_constraint = template.variable(variable)?;
    let mut zero_branch = candidate.clone();
    zero_branch.push(zero_constraint);
    let zero_branch = interreduce_polys(zero_branch);

    let fresh = PolyVariable::Temporary(fresh_id);
    for polynomial in &mut candidate {
        polynomial.add_variable(&fresh);
    }
    let template = candidate
        .first()
        .expect("the nonzero branch retains the nonempty cover");
    let x = template.variable(variable)?;
    let z = template.variable(&fresh)?;
    let xz = x * &z;
    let nonzero_constraint = template.one() - xz;
    candidate.push(nonzero_constraint);
    let nonzero_branch = interreduce_polys(candidate);
    Ok((zero_branch, nonzero_branch))
}

/// Count the minimal factor covers of a split child without invoking a
/// Gröbner-basis backend.
fn factor_cover_count(polynomials: &[PolyExtLex]) -> Result<usize, String> {
    let groups = try_factor_groups_from_polys(polynomials.to_vec())
        .map_err(|error| format!("factorization while checking branch contraction: {error}"))?;
    let (factors, index_groups) = build_factor_index(&groups);
    minimal_covers(&index_groups, factors.len())
        .map(|covers| covers.len())
        .ok_or_else(|| {
            format!(
                "split child has {} distinct factors; the cover enumerator supports at most 63",
                factors.len()
            )
        })
}

fn uses_temporary_variable(polynomial: &PolyExtGrevLex) -> bool {
    let temporary_positions: Vec<usize> = polynomial
        .variables
        .iter()
        .enumerate()
        .filter_map(|(index, variable)| {
            matches!(variable, PolyVariable::Temporary(_)).then_some(index)
        })
        .collect();
    polynomial.exponents_iter().any(|exponents| {
        temporary_positions
            .iter()
            .any(|&position| exponents[position] != 0)
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Once;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use symbolica::LicenseManager;
    use symbolica::domains::rational::Q;
    use symbolica::poly::GrevLexOrder;

    use super::*;
    use crate::{
        LoadedMathematicaSystem, load_mathematica_system_from_str_with_var_names,
        polynomials_from_loaded,
    };

    static LICENSE: Once = Once::new();

    fn load(source: &str) -> (LoadedMathematicaSystem, Vec<PolyExtLex>) {
        LICENSE.call_once(|| {
            let _ = LicenseManager::set_license_key(
                "5568d092#6bb8b4fb#ff936916-65ba-552f-b3c1-bdc64597cd21",
            );
        });
        let loaded = load_mathematica_system_from_str_with_var_names(source).expect("load");
        let polynomials = polynomials_from_loaded(&loaded, &Q).expect("polynomials");
        (loaded, polynomials)
    }

    fn candidates(loaded: &LoadedMathematicaSystem) -> Vec<ZeroingVariable> {
        loaded
            .var_names
            .iter()
            .cloned()
            .zip(loaded.id_to_sym.iter().copied())
            .map(|(name, symbol)| ZeroingVariable {
                name,
                variable: PolyVariable::from(symbol),
            })
            .collect()
    }

    #[test]
    fn choice_prefers_the_variable_in_more_generators() {
        let (loaded, polynomials) = load("{x + y == 0, x*z + 1 == 0}");
        let variables = candidates(&loaded);
        let selected = choose_variable(&polynomials, &variables).expect("useful variable");
        assert_eq!(variables[selected].name, "x");
    }

    #[test]
    fn split_adds_zero_and_rabinowitsch_constraints() {
        let (loaded, polynomials) = load("{x*y == 0}");
        let x = PolyVariable::from(
            loaded.id_to_sym[loaded
                .var_names
                .iter()
                .position(|name| name == "x")
                .unwrap()],
        );
        let (zero, nonzero) = split_on_variable(polynomials, &x, 7).expect("split");

        assert!(zero.iter().any(|polynomial| {
            polynomial.nterms() == 1
                && polynomial
                    .variable(&x)
                    .is_ok_and(|generator| polynomial.clone().make_monic() == generator)
        }));
        assert!(nonzero.iter().any(|polynomial| {
            polynomial.variables.contains(&PolyVariable::Temporary(7))
                && polynomial.to_string().contains("_TMP_7")
                && polynomial.nterms() == 2
        }));
    }

    #[test]
    fn timeout_is_recursively_partitioned_and_budget_is_retained() {
        let (loaded, polynomials) = load("{x + y == 0}");
        let options = VariableZeroingOptions {
            max_nodes: 1,
            ..VariableZeroingOptions::default()
        };
        let always_timeout = |_: Vec<PolyExtLex>, _: Duration, _: GbBackend| Outcome::Timeout;

        let result =
            solve_by_zeroing_with(polynomials, candidates(&loaded), &options, &always_timeout)
                .expect("search");

        assert_eq!(result.stats.zeroing_splits, 1);
        assert_eq!(result.nodes.len(), 1);
        assert_eq!(result.nodes[0].node_id, 0);
        assert!(result.nodes[0].decisions.is_empty());
        assert_eq!(result.nodes[0].covers, 1);
        assert_eq!(result.nodes[0].evaluations, 1);
        assert_eq!(result.nodes[0].timeouts, 1);
        assert_eq!(result.nodes[0].zeroing_splits, 1);
        assert_eq!(result.nodes[0].checks.len(), 1);
        assert_eq!(
            result.nodes[0].checks[0].outcome,
            VariableZeroingCoverOutcome::Timeout
        );
        assert!(result.nodes[0].checks[0].split_variable.is_some());
        assert_eq!(result.unresolved.len(), 2);
        assert!(
            result
                .unresolved
                .iter()
                .all(|branch| branch.reason == UnresolvedReason::NodeBudget)
        );
        assert_eq!(
            result
                .unresolved
                .iter()
                .map(|branch| branch.decisions[0].is_zero)
                .collect::<Vec<_>>(),
            vec![true, false]
        );
    }

    #[test]
    fn exhausted_runtime_retains_the_root_without_evaluating_it() {
        let (loaded, polynomials) = load("{x + y == 0}");
        let options = VariableZeroingOptions {
            max_runtime: Some(Duration::ZERO),
            ..VariableZeroingOptions::default()
        };
        let evaluations = AtomicUsize::new(0);
        let evaluator = |_: Vec<PolyExtLex>, _: Duration, _: GbBackend| {
            evaluations.fetch_add(1, Ordering::Relaxed);
            Outcome::Timeout
        };

        let result = solve_by_zeroing_with(polynomials, candidates(&loaded), &options, &evaluator)
            .expect("search");

        assert_eq!(evaluations.load(Ordering::Relaxed), 0);
        assert!(result.nodes.is_empty());
        assert_eq!(result.unresolved.len(), 1);
        assert_eq!(result.unresolved[0].reason, UnresolvedReason::RuntimeBudget);
    }

    #[test]
    fn node_results_include_each_zero_and_nonzero_branch() {
        let (loaded, polynomials) = load("{x + y == 0}");
        let options = VariableZeroingOptions {
            max_depth: 1,
            ..VariableZeroingOptions::default()
        };
        let always_timeout = |_: Vec<PolyExtLex>, _: Duration, _: GbBackend| Outcome::Timeout;

        let result =
            solve_by_zeroing_with(polynomials, candidates(&loaded), &options, &always_timeout)
                .expect("search");

        assert_eq!(result.nodes.len(), 3);
        assert_eq!(
            result
                .nodes
                .iter()
                .map(|node| node.node_id)
                .collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        assert!(result.nodes[0].decisions.is_empty());
        assert!(result.nodes[1].decisions[0].is_zero);
        assert!(!result.nodes[2].decisions[0].is_zero);
        assert_eq!(
            result.nodes[1].decisions[0].variable,
            result.nodes[2].decisions[0].variable
        );
    }

    #[test]
    fn singleton_branch_is_split_again_without_an_evaluation() {
        let (loaded, polynomials) = load("{x == 0, y == 0}");
        let options = VariableZeroingOptions::default();
        let evaluations = AtomicUsize::new(0);
        let always_timeout = |_: Vec<PolyExtLex>, _: Duration, _: GbBackend| {
            evaluations.fetch_add(1, Ordering::Relaxed);
            Outcome::Timeout
        };

        let result =
            solve_by_zeroing_with(polynomials, candidates(&loaded), &options, &always_timeout)
                .expect("search");

        // The root is evaluated and split on x. Its x = 0 child has one
        // cover, while its x != 0 sibling has none, so the unique cover is
        // split directly on y. Only the final y = 0 leaf is evaluated.
        assert_eq!(evaluations.load(Ordering::Relaxed), 2);
        assert_eq!(result.stats.evaluations, 2);
        assert_eq!(result.stats.zeroing_splits, 2);
        let contracted = result
            .nodes
            .iter()
            .flat_map(|node| &node.checks)
            .filter(|check| check.outcome == VariableZeroingCoverOutcome::Contracted)
            .collect::<Vec<_>>();
        assert_eq!(contracted.len(), 1);
        assert!(contracted[0].split_variable.is_some());
    }

    #[test]
    fn core_learned_in_one_node_prunes_a_later_node() {
        let (loaded, polynomials) = load("{x == 0, x - 1 == 0}");
        let options = VariableZeroingOptions {
            interreduce: false,
            ..VariableZeroingOptions::default()
        };
        let evaluations = AtomicUsize::new(0);
        let inconsistent = |_: Vec<PolyExtLex>, _: Duration, _: GbBackend| {
            evaluations.fetch_add(1, Ordering::Relaxed);
            Outcome::Inconsistent
        };
        let mut solver = Solver {
            options: &options,
            evaluator: &inconsistent,
            started: Instant::now(),
            fresh_variable: 0,
            components: Vec::new(),
            unresolved: Vec::new(),
            cores: Vec::new(),
            nodes: Vec::new(),
            stats: VariableZeroingStats::default(),
            first_solution: None,
        };
        let decisions = vec![SplitDecision {
            variable: loaded.var_names[0].clone(),
            is_zero: true,
        }];

        solver
            .solve_node(polynomials.clone(), Vec::new(), decisions.clone())
            .expect("first node");
        let mut sibling_decisions = decisions;
        sibling_decisions[0].is_zero = false;
        solver
            .solve_node(polynomials, Vec::new(), sibling_decisions)
            .expect("later node");

        assert_eq!(evaluations.load(Ordering::Relaxed), 1);
        assert_eq!(solver.cores.len(), 1);
        assert_eq!(solver.stats.core_pruned, 1);
        assert_eq!(
            solver
                .nodes
                .iter()
                .map(|node| node.core_pruned)
                .sum::<usize>(),
            1
        );
    }

    #[test]
    fn disabled_core_search_checks_later_nodes() {
        let (loaded, polynomials) = load("{x == 0, x - 1 == 0}");
        let options = VariableZeroingOptions {
            interreduce: false,
            unsat_core_search: false,
            ..VariableZeroingOptions::default()
        };
        let evaluations = AtomicUsize::new(0);
        let inconsistent = |_: Vec<PolyExtLex>, _: Duration, _: GbBackend| {
            evaluations.fetch_add(1, Ordering::Relaxed);
            Outcome::Inconsistent
        };
        let mut solver = Solver {
            options: &options,
            evaluator: &inconsistent,
            started: Instant::now(),
            fresh_variable: 0,
            components: Vec::new(),
            unresolved: Vec::new(),
            cores: Vec::new(),
            nodes: Vec::new(),
            stats: VariableZeroingStats::default(),
            first_solution: None,
        };
        let decisions = vec![SplitDecision {
            variable: loaded.var_names[0].clone(),
            is_zero: true,
        }];

        solver
            .solve_node(polynomials.clone(), Vec::new(), decisions.clone())
            .expect("first node");
        let mut sibling_decisions = decisions;
        sibling_decisions[0].is_zero = false;
        solver
            .solve_node(polynomials, Vec::new(), sibling_decisions)
            .expect("later node");

        assert_eq!(evaluations.load(Ordering::Relaxed), 2);
        assert!(solver.cores.is_empty());
        assert_eq!(solver.stats.core_pruned, 0);
    }

    #[test]
    fn fresh_variables_are_absent_from_projected_basis_view() {
        let (_, polynomials) = load("{x == 0}");
        let mut polynomial = polynomials[0].clone();
        polynomial.add_variable(&PolyVariable::Temporary(3));
        let temp = polynomial
            .variable(&PolyVariable::Temporary(3))
            .expect("temporary variable");
        let original = polynomial.reorder::<GrevLexOrder>();
        let with_temp = temp.reorder::<GrevLexOrder>();
        let component = ProjectedComponent {
            node_id: 0,
            decisions: Vec::new(),
            dimension: None,
            basis: ComputedBasis {
                elements: vec![original, with_temp],
                leading_monomials_only: false,
            },
        };

        let projected = component.original_basis_elements().collect::<Vec<_>>();
        assert_eq!(projected.len(), 1);
        assert!(!projected[0].to_string().contains("_TMP_"));
    }
}
