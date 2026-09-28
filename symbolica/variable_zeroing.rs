//! Recursive factor branching with variable zeroing.
//!
//! This implements Algorithm `Factor branching with variable zeroing`
//! (`alg:ZeroingBranching` in `paper/sections/variable_zeroing.tex`). A
//! factor cover that exceeds its Gröbner-basis time budget is split into the
//! constructible cases `x = a` and `x != a`. By default `a = 0`; fixed-one
//! and fresh-random-value modes are also available. In the equality branch,
//! `x` is replaced by `a`. Each inequality decision gets its own equation
//! `1 - s_i*(x_i-a_i) = 0` by default. An option instead makes all
//! inequality decisions on a branch share the saturation equation
//! `1 - s*(x_1-a_1)*...*(x_n-a_n) = 0`. If one side of a split has no factor
//! cover and the other has a unique cover, the unique cover is split again
//! without a redundant Gröbner-basis call. Only original-variable basis
//! elements and the zero/nonzero decisions are exposed by the projected
//! result.

use std::time::{Duration, Instant};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use symbolica::domains::rational::{Q, Rational};
use symbolica::poly::{MonomialOrder, PolyVariable, polynomial::MultivariatePolynomial};

use crate::cover::{build_factor_index, minimal_covers, ordered_cover_masks};
use crate::factor_prep::{
    PolyExtLex, interreduce_polys, reorder_by_min_fill, try_factor_groups_from_polys,
};
use crate::gb_search::{
    ComputedBasis, GbBackend, Outcome, PolyExtGrevLex, evaluate_cover, solution_dimension,
};
use crate::refine::{FactorGroups, RefineOps, refine_covers_with};
use crate::solver_process::{
    ProcessRun, factor_groups_isolated, interreduce_isolated, running_under_test_harness,
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
    pub value: i64,
    pub is_equal: bool,
}

fn format_decision_path(decisions: &[SplitDecision]) -> String {
    if decisions.is_empty() {
        return "no decisions".to_string();
    }
    decisions
        .iter()
        .map(|decision| {
            format!(
                "{} {} {}",
                decision.variable,
                if decision.is_equal { "=" } else { "!=" },
                decision.value
            )
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// Values used in the covering split `x = a` / `x != a`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SplitValueStrategy {
    /// Use the same integer `a` at every split. `Fixed(0)` reproduces
    /// variable zeroing.
    Fixed(i64),
    /// Draw a fresh integer uniformly from `[-10,10]` at every split.
    Random,
}

impl Default for SplitValueStrategy {
    fn default() -> Self {
        Self::Fixed(0)
    }
}

/// Search controls for recursive variable zeroing.
#[derive(Clone, Debug)]
pub struct VariableZeroingOptions {
    /// Apply the optional interreduction in every invocation of Branching.
    pub interreduce: bool,
    /// Per-cover Gröbner-basis and per-node factorization time budget.
    pub timeout: Duration,
    /// Optional wall-clock budget for the entire recursive search.
    pub max_runtime: Option<Duration>,
    /// Gröbner-basis implementation used to evaluate covers. Defaults to
    /// Singular.
    pub gb_backend: GbBackend,
    /// Use one Rabinowitsch variable and equation per nonzero variable instead
    /// of one equation for their product.
    pub saturate_each_variable: bool,
    /// Select the value `a` in each `x = a` / `x != a` split.
    pub split_value_strategy: SplitValueStrategy,
    /// Seed used by `SplitValueStrategy::Random`.
    pub random_seed: u64,
    /// Emit one compact progress line per evaluated cover.
    pub verbose: bool,
}

impl Default for VariableZeroingOptions {
    fn default() -> Self {
        Self {
            interreduce: true,
            timeout: Duration::from_millis(1_000),
            max_runtime: None,
            gb_backend: GbBackend::SingularGb,
            saturate_each_variable: true,
            split_value_strategy: SplitValueStrategy::Fixed(0),
            random_seed: 0,
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

    /// Basis entries not involving a saturation variable.
    ///
    /// Together with `decisions`, these form the printable projection view.
    /// The inequality decisions are essential: an open condition such as
    /// `x != a` cannot in general be represented by polynomial equalities alone.
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
    RuntimeBudget,
    FactorizationTimeout,
    FactorizationError,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PreprocessingFailure {
    Timeout,
    Error,
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
    /// All timed-out evaluations, including those subsequently resolved by a
    /// zeroing split.
    pub timeouts: usize,
    pub zeroing_splits: usize,
    pub factorization_timeouts: usize,
    pub factorization_errors: usize,
    pub largest_depth: usize,
}

/// Outcome recorded for one hitting-set check in the recursion tree.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VariableZeroingCoverOutcome {
    Consistent,
    Inconsistent,
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
    /// Recursion node reached by the equality branch, if created.
    pub zero_child: Option<usize>,
    /// Recursion node reached by the inequality branch, if created.
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
    pub timeouts: usize,
    pub zeroing_splits: usize,
    /// Set when factorization did not produce covers for this node.
    pub preprocessing_failure: Option<PreprocessingFailure>,
    pub elapsed: Duration,
    /// Hitting sets in their evaluation order. This may be shorter than
    /// `covers` when a global budget stops the node early.
    pub checks: Vec<VariableZeroingCoverResult>,
}

pub struct VariableZeroingResult {
    pub components: Vec<ProjectedComponent>,
    pub unresolved: Vec<UnresolvedBranch>,
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
    components: Vec<ProjectedComponent>,
    unresolved: Vec<UnresolvedBranch>,
    nodes: Vec<VariableZeroingNodeResult>,
    stats: VariableZeroingStats,
    first_solution: Option<Duration>,
    random: StdRng,
}

impl Solver<'_> {
    fn next_split_value(&mut self) -> i64 {
        next_split_value(self.options.split_value_strategy, &mut self.random)
    }

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
        self.solve_node_with_contraction(polynomials, variables, decisions, Vec::new(), false)
    }

    fn solve_node_with_contraction(
        &mut self,
        polynomials: Vec<PolyExtLex>,
        variables: Vec<ZeroingVariable>,
        decisions: Vec<SplitDecision>,
        inequality_conditions: Vec<(PolyVariable, i64)>,
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
            timeouts: 0,
            zeroing_splits: 0,
            preprocessing_failure: None,
            elapsed: Duration::ZERO,
            checks: Vec::new(),
        };
        let polynomials =
            reorder_by_min_fill(polynomials).map_err(|error| format!("node {node_id}: {error}"))?;
        let node_nvars = polynomials.first().map_or(0, PolyExtLex::nvars);
        let (groups, covers) = match terminal_hitting_sets_with_timeout(
            polynomials.clone(),
            self.options.interreduce,
            self.evaluation_timeout(),
        ) {
            Ok(preprocessed) => preprocessed,
            Err(HittingSetFailure::Fatal(error)) => return Err(format!("node {node_id}: {error}")),
            Err(HittingSetFailure::Timeout) => {
                self.stats.factorization_timeouts += 1;
                node_result.preprocessing_failure = Some(PreprocessingFailure::Timeout);
                self.retain_unresolved(
                    polynomials,
                    &variables,
                    &decisions,
                    UnresolvedReason::FactorizationTimeout,
                );
                node_result.elapsed = node_started.elapsed();
                if self.options.verbose {
                    println!(
                        "node #{node_id} depth {depth}: factorization timed out ({:?})",
                        node_result.elapsed
                    );
                }
                self.nodes.push(node_result);
                return Ok(Some(node_id));
            }
            Err(HittingSetFailure::Factorization(error)) => {
                self.stats.factorization_errors += 1;
                node_result.preprocessing_failure = Some(PreprocessingFailure::Error);
                self.retain_unresolved(
                    polynomials,
                    &variables,
                    &decisions,
                    UnresolvedReason::FactorizationError,
                );
                node_result.elapsed = node_started.elapsed();
                if self.options.verbose {
                    println!(
                        "node #{node_id} depth {depth}: factorization failed ({:?}): {error}",
                        node_result.elapsed
                    );
                }
                self.nodes.push(node_result);
                return Ok(Some(node_id));
            }
        };
        self.stats.covers += covers.len();
        node_result.covers = covers.len();

        if self.options.verbose {
            println!(
                "node #{node_id} depth {depth}: {groups} factor group(s), {} cover(s), {} variable(s) left",
                covers.len(),
                variables.len()
            );
        }

        for (cover_index, hitting_set) in covers.iter().enumerate() {
            let cover_started = Instant::now();
            let candidate = hitting_set.clone();
            if self.runtime_exhausted() {
                self.retain_unresolved(
                    candidate,
                    &variables,
                    &decisions,
                    UnresolvedReason::RuntimeBudget,
                );
                for pending in &covers[cover_index + 1..] {
                    self.retain_unresolved(
                        pending.clone(),
                        &variables,
                        &decisions,
                        UnresolvedReason::RuntimeBudget,
                    );
                }
                break;
            }
            // The caller has already established that this is the unique
            // cover on one side of an equality/inequality split and that the sibling
            // has no cover. Hence this cover represents the entire parent
            // branch, and another zeroing split is sound without first asking
            // the Gröbner solver to rediscover that the branch is difficult.
            if contract_unique_cover
                && covers.len() == 1
                && let Some(selected_index) = choose_variable(&candidate, &variables)
            {
                let selected = variables[selected_index].clone();
                if self.options.verbose {
                    println!(
                        "  cover #{cover_index}: branch-contracted; splitting {}",
                        selected.name
                    );
                }
                let (zero_child, nonzero_child, recursive_elapsed) = self.split_and_solve(
                    candidate,
                    &variables,
                    &decisions,
                    &inequality_conditions,
                    selected_index,
                )?;
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
                    self.record_consistent_components(
                        vec![basis],
                        node_id,
                        node_nvars,
                        &decisions,
                        cover_index,
                        cover_elapsed,
                        &mut node_result,
                    );
                }
                Outcome::ConsistentComponents(bases) => {
                    self.record_consistent_components(
                        bases,
                        node_id,
                        node_nvars,
                        &decisions,
                        cover_index,
                        cover_elapsed,
                        &mut node_result,
                    );
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
                }
                Outcome::Timeout => {
                    self.stats.timeouts += 1;
                    node_result.timeouts += 1;
                    if self.options.verbose {
                        println!(
                            "  cover #{cover_index}: timed-out hitting set ({} generators) under {}",
                            candidate.len(),
                            format_decision_path(&decisions)
                        );
                        for (index, polynomial) in candidate.iter().enumerate() {
                            println!("    t{index} = {polynomial}");
                        }
                    }
                    if self.runtime_exhausted() {
                        self.retain_unresolved(
                            candidate,
                            &variables,
                            &decisions,
                            UnresolvedReason::RuntimeBudget,
                        );
                        for pending in &covers[cover_index + 1..] {
                            self.retain_unresolved(
                                pending.clone(),
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

                    let selected = variables[selected_index].clone();
                    if self.options.verbose {
                        println!(
                            "  cover #{cover_index}: timed out ({:?}); splitting {}",
                            cover_elapsed, selected.name
                        );
                    }
                    let (zero_child, nonzero_child, recursive_elapsed) = self.split_and_solve(
                        candidate,
                        &variables,
                        &decisions,
                        &inequality_conditions,
                        selected_index,
                    )?;
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

    #[allow(clippy::too_many_arguments)]
    fn record_consistent_components(
        &mut self,
        bases: Vec<ComputedBasis>,
        node_id: usize,
        node_nvars: usize,
        decisions: &[SplitDecision],
        cover_index: usize,
        elapsed: Duration,
        node_result: &mut VariableZeroingNodeResult,
    ) {
        self.stats.consistent += 1;
        node_result.consistent += 1;
        self.first_solution
            .get_or_insert_with(|| self.started.elapsed());
        if self.options.verbose {
            let basis_elements = bases
                .iter()
                .map(|basis| basis.elements.len())
                .sum::<usize>();
            println!(
                "  cover #{cover_index}: solved ({} component(s), {basis_elements} basis element(s), {:?})",
                bases.len(),
                elapsed
            );
        }
        node_result.checks.push(VariableZeroingCoverResult {
            cover_index,
            outcome: VariableZeroingCoverOutcome::Consistent,
            elapsed,
            split_variable: None,
            zero_child: None,
            nonzero_child: None,
        });
        self.components.extend(bases.into_iter().map(|basis| {
            let dimension = solution_dimension(&basis.elements, node_nvars);
            ProjectedComponent {
                node_id,
                decisions: decisions.to_vec(),
                dimension,
                basis,
            }
        }));
    }

    fn split_and_solve(
        &mut self,
        candidate: Vec<PolyExtLex>,
        variables: &[ZeroingVariable],
        decisions: &[SplitDecision],
        inequality_conditions: &[(PolyVariable, i64)],
        selected_index: usize,
    ) -> Result<(Option<usize>, Option<usize>, Duration), String> {
        let selected = variables[selected_index].clone();
        let mut remaining = variables.to_vec();
        remaining.remove(selected_index);
        let split_value = self.next_split_value();
        let mut updated_inequality_conditions = inequality_conditions.to_vec();
        updated_inequality_conditions.push((selected.variable.clone(), split_value));
        let (equal_branch, unequal_branch) = split_on_variable(
            candidate,
            &selected.variable,
            split_value,
            &updated_inequality_conditions,
            self.options.saturate_each_variable,
        )?;

        let equal_cover_count = factor_cover_count(
            &equal_branch,
            self.options.interreduce,
            self.evaluation_timeout(),
        )?;
        let unequal_cover_count = if equal_cover_count.is_some() {
            factor_cover_count(
                &unequal_branch,
                self.options.interreduce,
                self.evaluation_timeout(),
            )?
        } else {
            None
        };
        let (contract_equal, contract_unequal) = equal_cover_count.zip(unequal_cover_count).map_or(
            (false, false),
            |(equal_count, unequal_count)| {
                (
                    equal_count == 1 && unequal_count == 0,
                    unequal_count == 1 && equal_count == 0,
                )
            },
        );

        let mut equal_decisions = decisions.to_vec();
        equal_decisions.push(SplitDecision {
            variable: selected.name.clone(),
            value: split_value,
            is_equal: true,
        });
        let children_started = Instant::now();
        let equal_child = self.solve_node_with_contraction(
            equal_branch,
            remaining.clone(),
            equal_decisions,
            inequality_conditions.to_vec(),
            contract_equal,
        )?;

        let mut unequal_decisions = decisions.to_vec();
        unequal_decisions.push(SplitDecision {
            variable: selected.name,
            value: split_value,
            is_equal: false,
        });
        let unequal_child = self.solve_node_with_contraction(
            unequal_branch,
            remaining,
            unequal_decisions,
            updated_inequality_conditions,
            contract_unequal,
        )?;
        Ok((equal_child, unequal_child, children_started.elapsed()))
    }
}

fn next_split_value(strategy: SplitValueStrategy, random: &mut StdRng) -> i64 {
    match strategy {
        SplitValueStrategy::Fixed(value) => value,
        SplitValueStrategy::Random => random.random_range(-10..=10),
    }
}

/// Solve a polynomial system by factor branching, recursively splitting on an
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
        components: Vec::new(),
        unresolved: Vec::new(),
        nodes: Vec::new(),
        stats: VariableZeroingStats::default(),
        first_solution: None,
        random: StdRng::seed_from_u64(options.random_seed),
    };
    solver.solve_node(polynomials, variables, Vec::new())?;
    Ok(VariableZeroingResult {
        components: solver.components,
        unresolved: solver.unresolved,
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

/// Substitute `x = a` in the equality branch and rebuild the saturation
/// equation(s) for all inequalities in the other branch.
fn split_on_variable(
    candidate: Vec<PolyExtLex>,
    variable: &PolyVariable,
    value: i64,
    inequality_conditions: &[(PolyVariable, i64)],
    saturate_each_variable: bool,
) -> Result<(Vec<PolyExtLex>, Vec<PolyExtLex>), String> {
    if candidate.is_empty() {
        return Err("cannot split a variable in an empty cover".to_string());
    }
    let split_value = Rational::from(value);
    let equal_branch = candidate
        .iter()
        .map(|polynomial| {
            polynomial
                .variables
                .iter()
                .position(|candidate| candidate == variable)
                .map_or_else(
                    || polynomial.clone(),
                    |position| polynomial.replace(position, &split_value),
                )
        })
        .collect();
    let mut unequal_branch: Vec<_> = candidate
        .into_iter()
        .filter(|polynomial| !uses_temporary_variable(polynomial))
        .collect();
    if unequal_branch.is_empty() {
        return Err("cannot rebuild saturation from an empty cover".to_string());
    }
    let saturation_variable_count = if saturate_each_variable {
        inequality_conditions.len()
    } else {
        1
    };
    for index in 0..saturation_variable_count {
        let saturation_variable = PolyVariable::Temporary(index);
        for polynomial in &mut unequal_branch {
            polynomial.add_variable(&saturation_variable);
        }
    }

    let template = unequal_branch
        .first()
        .cloned()
        .expect("the inequality branch retains the nonempty cover");
    if saturate_each_variable {
        for (index, (variable, value)) in inequality_conditions.iter().enumerate() {
            let inverse = template.variable(&PolyVariable::Temporary(index))?;
            let difference =
                template.variable(variable)? - template.constant(Rational::from(*value));
            let product = inverse * &difference;
            unequal_branch.push(template.one() - product);
        }
    } else {
        let mut product = template.variable(&PolyVariable::Temporary(0))?;
        for (variable, value) in inequality_conditions {
            let difference =
                template.variable(variable)? - template.constant(Rational::from(*value));
            product = product * &difference;
        }
        unequal_branch.push(template.one() - product);
    }
    Ok((equal_branch, unequal_branch))
}

fn uses_variable<O: MonomialOrder>(
    polynomial: &MultivariatePolynomial<Q, u16, O>,
    variable: &PolyVariable,
) -> bool {
    let Some(position) = polynomial
        .variables
        .iter()
        .position(|candidate| candidate == variable)
    else {
        return false;
    };
    polynomial
        .exponents_iter()
        .any(|exponents| exponents[position] != 0)
}

fn uses_temporary_variable<O: MonomialOrder>(
    polynomial: &MultivariatePolynomial<Q, u16, O>,
) -> bool {
    polynomial
        .variables
        .iter()
        .filter(|variable| matches!(variable, PolyVariable::Temporary(_)))
        .any(|variable| uses_variable(polynomial, variable))
}

fn factor_groups_with_timeout(
    polynomials: Vec<PolyExtLex>,
    interreduce: bool,
    timeout: Duration,
) -> Result<ProcessRun<Vec<Vec<PolyExtLex>>>, String> {
    if cfg!(test) || running_under_test_harness() {
        let polynomials = if interreduce {
            interreduce_polys(polynomials)
        } else {
            polynomials
        };
        try_factor_groups_from_polys(polynomials).map(ProcessRun::Completed)
    } else {
        factor_groups_isolated(polynomials, interreduce, timeout)
    }
}

/// Why a node's terminal hitting sets could not be computed.
enum HittingSetFailure {
    Timeout,
    Factorization(String),
    /// A limitation of the search itself; aborts the whole run.
    Fatal(String),
}

impl From<String> for HittingSetFailure {
    fn from(error: String) -> Self {
        Self::Fatal(error)
    }
}

fn remaining_until(deadline: Instant) -> Result<Duration, HittingSetFailure> {
    deadline
        .checked_duration_since(Instant::now())
        .filter(|remaining| !remaining.is_zero())
        .ok_or(HittingSetFailure::Timeout)
}

fn completed<T>(run: Result<ProcessRun<T>, String>) -> Result<T, HittingSetFailure> {
    match run {
        Ok(ProcessRun::Completed(value)) => Ok(value),
        Ok(ProcessRun::Timeout) => Err(HittingSetFailure::Timeout),
        Err(error) => Err(HittingSetFailure::Factorization(error)),
    }
}

/// Refinement operations run in worker processes that share one deadline.
struct IsolatedBeforeDeadline {
    deadline: Instant,
}

impl RefineOps for IsolatedBeforeDeadline {
    type Error = HittingSetFailure;

    fn interreduce(
        &mut self,
        polynomials: Vec<PolyExtLex>,
    ) -> Result<Vec<PolyExtLex>, HittingSetFailure> {
        let remaining = remaining_until(self.deadline)?;
        if cfg!(test) || running_under_test_harness() {
            Ok(interreduce_polys(polynomials))
        } else {
            completed(interreduce_isolated(polynomials, remaining))
        }
    }

    fn factor(&mut self, polynomials: Vec<PolyExtLex>) -> Result<FactorGroups, HittingSetFailure> {
        factor_before_deadline(polynomials, false, self.deadline)
    }
}

fn factor_before_deadline(
    polynomials: Vec<PolyExtLex>,
    interreduce: bool,
    deadline: Instant,
) -> Result<FactorGroups, HittingSetFailure> {
    let remaining = remaining_until(deadline)?;
    completed(factor_groups_with_timeout(
        polynomials,
        interreduce,
        remaining,
    ))
}

/// Factor a node's equations and refine its minimal hitting sets into
/// terminal hitting sets (Algorithm `Refine`), all within `timeout`. Returns
/// the number of factor groups and the terminal hitting sets.
fn terminal_hitting_sets_with_timeout(
    polynomials: Vec<PolyExtLex>,
    interreduce: bool,
    timeout: Duration,
) -> Result<(usize, Vec<Vec<PolyExtLex>>), HittingSetFailure> {
    let deadline = Instant::now() + timeout;
    let groups = factor_before_deadline(polynomials, interreduce, deadline)?;
    let (factors, index_groups) = build_factor_index(&groups);
    let covers = minimal_covers(&index_groups, factors.len()).ok_or_else(|| {
        HittingSetFailure::Fatal(format!(
            "{} distinct factors; the cover enumerator supports at most 63",
            factors.len()
        ))
    })?;
    let (terminal, _) = refine_covers_with(
        &factors,
        &ordered_cover_masks(&covers),
        interreduce,
        &mut IsolatedBeforeDeadline { deadline },
    )?;
    Ok((groups.len(), terminal))
}

/// Count the minimal factor covers of a split child without invoking a
/// Gröbner-basis backend. A preprocessing timeout disables branch contraction;
/// the child node will retain the branch as unresolved if factorization times
/// out again.
fn factor_cover_count(
    polynomials: &[PolyExtLex],
    interreduce: bool,
    timeout: Duration,
) -> Result<Option<usize>, String> {
    let groups = match factor_groups_with_timeout(polynomials.to_vec(), interreduce, timeout) {
        Ok(ProcessRun::Completed(groups)) => groups,
        Ok(ProcessRun::Timeout) | Err(_) => return Ok(None),
    };
    let (factors, index_groups) = build_factor_index(&groups);
    minimal_covers(&index_groups, factors.len())
        .map(|covers| Some(covers.len()))
        .ok_or_else(|| {
            format!(
                "split child has {} distinct factors; the cover enumerator supports at most 63",
                factors.len()
            )
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
            let _ = LicenseManager::set_license_key(crate::SYMBOLICA_LICENSE_KEY);
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
    fn options_default_to_singular() {
        let options = VariableZeroingOptions::default();

        assert_eq!(options.timeout, Duration::from_millis(1_000));
        assert_eq!(options.gb_backend, GbBackend::SingularGb);
        assert!(options.saturate_each_variable);
    }

    #[test]
    fn choice_prefers_the_variable_in_more_generators() {
        let (loaded, polynomials) = load("{x + y == 0, x*z + 1 == 0}");
        let variables = candidates(&loaded);
        let selected = choose_variable(&polynomials, &variables).expect("useful variable");
        assert_eq!(variables[selected].name, "x");
    }

    #[test]
    fn split_substitutes_zero_and_keeps_one_saturation_equation() {
        let (loaded, polynomials) = load("{x*y + y == 0}");
        let x = PolyVariable::from(
            loaded.id_to_sym[loaded
                .var_names
                .iter()
                .position(|name| name == "x")
                .unwrap()],
        );
        let y = PolyVariable::from(
            loaded.id_to_sym[loaded
                .var_names
                .iter()
                .position(|name| name == "y")
                .unwrap()],
        );
        let (zero, nonzero) =
            split_on_variable(polynomials, &x, 0, &[(x.clone(), 0)], false).expect("split");

        assert!(zero.iter().all(|polynomial| !uses_variable(polynomial, &x)));
        assert_eq!(
            nonzero
                .iter()
                .filter(|polynomial| uses_variable(polynomial, &PolyVariable::Temporary(0)))
                .count(),
            1
        );

        let (_, twice_nonzero) =
            split_on_variable(nonzero, &y, 0, &[(x, 0), (y.clone(), 0)], false)
                .expect("second split");
        assert_eq!(
            twice_nonzero
                .iter()
                .filter(|polynomial| uses_variable(polynomial, &PolyVariable::Temporary(0)))
                .count(),
            1
        );
    }

    #[test]
    fn split_substitutes_one_and_saturates_x_minus_one() {
        let (loaded, polynomials) = load("{x*y + y == 0}");
        let x = PolyVariable::from(
            loaded.id_to_sym[loaded
                .var_names
                .iter()
                .position(|name| name == "x")
                .unwrap()],
        );
        let (equal, unequal) =
            split_on_variable(polynomials, &x, 1, &[(x.clone(), 1)], true).expect("split");

        assert!(
            equal
                .iter()
                .all(|polynomial| !uses_variable(polynomial, &x))
        );
        let saturation = unequal
            .iter()
            .find(|polynomial| uses_temporary_variable(polynomial))
            .expect("saturation equation");
        let x_position = saturation
            .variables
            .iter()
            .position(|variable| variable == &x)
            .expect("x in saturation equation");
        assert!(saturation.replace(x_position, &Rational::from(1)).is_one());
    }

    #[test]
    fn random_split_values_are_fresh_and_seeded() {
        let draw = || {
            let mut random = StdRng::seed_from_u64(0);
            (0..8)
                .map(|_| next_split_value(SplitValueStrategy::Random, &mut random))
                .collect::<Vec<_>>()
        };
        let first = draw();
        let second = draw();

        assert_eq!(first, second);
        assert!(first.iter().all(|value| (-10..=10).contains(value)));
        assert!(first.windows(2).any(|pair| pair[0] != pair[1]));
        let mut random = StdRng::seed_from_u64(99);
        assert_eq!(
            next_split_value(SplitValueStrategy::Fixed(1), &mut random),
            1
        );
    }

    #[test]
    fn split_can_saturate_each_nonzero_variable_with_a_separate_equation() {
        let (loaded, polynomials) = load("{x*y + y == 0}");
        let variable = |name: &str| {
            PolyVariable::from(
                loaded.id_to_sym[loaded
                    .var_names
                    .iter()
                    .position(|candidate| candidate == name)
                    .unwrap()],
            )
        };
        let x = variable("x");
        let y = variable("y");

        let (_, x_nonzero) =
            split_on_variable(polynomials, &x, 0, &[(x.clone(), 0)], true).expect("split x");
        assert_eq!(
            x_nonzero
                .iter()
                .filter(|polynomial| uses_temporary_variable(polynomial))
                .count(),
            1
        );

        let (_, x_y_nonzero) =
            split_on_variable(x_nonzero, &y, 0, &[(x.clone(), 0), (y.clone(), 0)], true)
                .expect("split y");
        let saturation_equations = x_y_nonzero
            .iter()
            .filter(|polynomial| uses_temporary_variable(polynomial))
            .collect::<Vec<_>>();
        assert_eq!(saturation_equations.len(), 2);
        assert!(saturation_equations.iter().any(|polynomial| {
            uses_variable(polynomial, &PolyVariable::Temporary(0))
                && uses_variable(polynomial, &x)
                && !uses_variable(polynomial, &y)
        }));
        assert!(saturation_equations.iter().any(|polynomial| {
            uses_variable(polynomial, &PolyVariable::Temporary(1))
                && uses_variable(polynomial, &y)
                && !uses_variable(polynomial, &x)
        }));
    }

    #[test]
    fn timeout_is_recursively_partitioned() {
        let (loaded, polynomials) = load("{x + y == 0}");
        let options = VariableZeroingOptions::default();
        let always_timeout = |_: Vec<PolyExtLex>, _: Duration, _: GbBackend| Outcome::Timeout;

        let result =
            solve_by_zeroing_with(polynomials, candidates(&loaded), &options, &always_timeout)
                .expect("search");

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
        assert!(!result.unresolved.is_empty());
        assert!(result.unresolved.iter().all(|branch| {
            branch.reason == UnresolvedReason::TimeoutAfterVariablesExhausted
                && !branch.decisions.is_empty()
        }));
    }

    #[test]
    fn root_covers_are_the_refined_terminal_hitting_sets() {
        let (loaded, polynomials) = load("{(x + y*z)*w == 0, x*v == 0}");
        let options = VariableZeroingOptions::default();
        let always_timeout = |_: Vec<PolyExtLex>, _: Duration, _: GbBackend| Outcome::Timeout;

        let result =
            solve_by_zeroing_with(polynomials, candidates(&loaded), &options, &always_timeout)
                .expect("search");

        assert_eq!(result.nodes[0].covers, 5);
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
        let options = VariableZeroingOptions::default();
        let always_timeout = |_: Vec<PolyExtLex>, _: Duration, _: GbBackend| Outcome::Timeout;

        let result =
            solve_by_zeroing_with(polynomials, candidates(&loaded), &options, &always_timeout)
                .expect("search");

        let root_children = result
            .nodes
            .iter()
            .filter(|node| node.decisions.len() == 1)
            .collect::<Vec<_>>();
        assert!(result.nodes[0].decisions.is_empty());
        assert_eq!(root_children.len(), 2);
        assert!(root_children[0].decisions[0].is_equal);
        assert!(!root_children[1].decisions[0].is_equal);
        assert_eq!(
            root_children[0].decisions[0].variable,
            root_children[1].decisions[0].variable
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
    fn saturation_variable_is_absent_from_projected_basis_view() {
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
