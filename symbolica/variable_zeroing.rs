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
//! `1 - s*(x_1-a_1)*...*(x_n-a_n) = 0`. A third option replaces the
//! inequality branch by the ideal saturation `⟨H⟩ : (∏ (x_i-a_i))^∞` in the
//! original variables; refinement then treats the `x_i - a_i` as nonzero side
//! conditions. If one side of a split has no factor
//! cover and the other has a unique cover, the unique cover is split again
//! without a redundant Gröbner-basis call. Optionally, a cover whose zero set
//! lies in that of a solved component is skipped, and solved components that
//! lie in another are dropped. Only original-variable basis
//! elements and the zero/nonzero decisions are exposed by the projected
//! result.

use std::sync::{Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rayon::prelude::*;
use symbolica::domains::rational::{Q, Rational};
use symbolica::poly::{
    GrevLexOrder, LexOrder, MonomialOrder, PolyVariable, polynomial::MultivariatePolynomial,
};

use crate::cover::{build_factor_index, minimal_covers, ordered_cover_masks};
use crate::factor_prep::{
    PolyExtLex, interreduce_polys, reorder_by_min_fill, try_factor_groups_from_polys,
};
use crate::gb_search::{
    ComputedBasis, GbBackend, Outcome, PolyExtGrevLex, Saturation, evaluate_cover, saturate,
    solution_dimension,
};
use crate::refine::{FactorGroups, RefineOps, refine_covers_under_conditions, without_conditions};
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

/// How an inequality branch encodes the conditions `x_i != a_i` of its path.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum NonzeroEncoding {
    /// One Rabinowitsch equation `1 - s_i*(x_i-a_i) = 0` per condition.
    #[default]
    Separate,
    /// One equation `1 - s*(x_1-a_1)*...*(x_n-a_n) = 0` for all conditions.
    SharedProduct,
    /// The saturation `⟨H⟩ : (∏ (x_i-a_i))^∞`, computed by Singular within the
    /// per-cover timeout; on timeout or error the branch keeps the
    /// unsaturated generators. Branches have no auxiliary variable: the
    /// decisions `x_i != a_i` are refinement side conditions, and only the
    /// branch solver's input carries them as separate Rabinowitsch equations.
    Saturation,
}

impl NonzeroEncoding {
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Separate => "per variable",
            Self::SharedProduct => "shared product",
            Self::Saturation => "ideal saturation",
        }
    }
}

/// The `x = a` and `x != a` decisions of a branch.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct BranchConditions {
    equal: Vec<(PolyVariable, i64)>,
    unequal: Vec<(PolyVariable, i64)>,
}

impl BranchConditions {
    /// Whether every decision of `self` is also a decision of `other`, so that
    /// the points satisfying `other` satisfy `self`.
    fn implied_by(&self, other: &Self) -> bool {
        self.equal
            .iter()
            .all(|decision| other.equal.contains(decision))
            && self
                .unequal
                .iter()
                .all(|decision| other.unequal.contains(decision))
    }
}

/// Search controls for recursive variable zeroing.
#[derive(Clone, Debug)]
pub struct VariableZeroingOptions {
    /// Per-cover Gröbner-basis and per-node factorization time budget.
    pub timeout: Duration,
    /// Time budget of a leaf: a cover with no variable left to split on.
    /// `None` gives leaves the per-cover `timeout`.
    pub leaf_timeout: Option<Duration>,
    /// Worker threads; the covers of a node and the two children of a split
    /// are solved concurrently. Keep it at most the solver-process limit
    /// (cores - 2), or covers wait for a process slot within their timeout.
    pub jobs: usize,
    /// Optional wall-clock budget for the entire recursive search.
    pub max_runtime: Option<Duration>,
    /// Gröbner-basis implementation used to evaluate covers. Defaults to
    /// Singular.
    pub gb_backend: GbBackend,
    /// Encoding of the inequality branch of a split.
    pub nonzero_encoding: NonzeroEncoding,
    /// Skip a cover whose zero set, under the branch's decisions, is
    /// certified to lie in that of a solved component, and drop solved
    /// components that lie in another.
    pub prune_by_solutions: bool,
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
            timeout: Duration::from_millis(1_000),
            leaf_timeout: None,
            jobs: 1,
            max_runtime: None,
            gb_backend: GbBackend::SingularGb,
            nonzero_encoding: NonzeroEncoding::Separate,
            prune_by_solutions: false,
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
    conditions: BranchConditions,
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

    /// Whether the full basis lies in the original variables, so that it
    /// describes the component's zero set without a projection.
    fn has_original_basis(&self) -> bool {
        !self.basis.leading_monomials_only
            && !self.basis.elements.iter().any(uses_temporary_variable)
    }

    /// Whether the points of `branch` that satisfy `conditions` certifiably lie
    /// in this component: its decisions are among `conditions`, and each basis
    /// element, with the equality decisions of `conditions` substituted,
    /// reduces to zero modulo the generators of `branch`.
    fn covers_branch(&self, branch: &[PolyExtLex], conditions: &BranchConditions) -> bool {
        self.has_original_basis()
            && self.conditions.implied_by(conditions)
            && self.basis.elements.iter().all(|element| {
                reduces_to_zero(
                    &substitute_values(element.reorder::<LexOrder>(), &conditions.equal),
                    branch,
                )
            })
    }

    /// Whether this component certifiably lies in `other`: the decisions of
    /// `other` are among its own, and each basis element of `other` reduces to
    /// zero modulo this component's Gröbner basis, an exact membership test.
    fn lies_in(&self, other: &Self) -> bool {
        self.has_original_basis()
            && other.has_original_basis()
            && other.conditions.implied_by(&self.conditions)
            && other.basis.elements.iter().all(|element| {
                let element =
                    substitute_values(element.reorder::<LexOrder>(), &self.conditions.equal);
                reduces_to_zero(&element.reorder::<GrevLexOrder>(), &self.basis.elements)
            })
    }
}

/// `polynomial` with each variable of `values` that occurs in its ring
/// replaced by the paired value.
fn substitute_values(polynomial: PolyExtLex, values: &[(PolyVariable, i64)]) -> PolyExtLex {
    values
        .iter()
        .fold(polynomial, |polynomial, (variable, value)| {
            polynomial
                .variables()
                .iter()
                .position(|candidate| candidate == variable)
                .map_or(polynomial.clone(), |position| {
                    polynomial.replace(position, &Rational::from(*value))
                })
        })
}

/// Whether `polynomial` has remainder zero on division by `divisors`, after
/// both are expressed in the ring of the first divisor extended by the
/// variables of `polynomial`. Appending variables keeps a Gröbner basis one.
fn reduces_to_zero<O: MonomialOrder>(
    polynomial: &MultivariatePolynomial<Q, u16, O>,
    divisors: &[MultivariatePolynomial<Q, u16, O>],
) -> bool {
    let mut ring: Vec<_> = divisors
        .iter()
        .filter(|divisor| !divisor.is_zero())
        .cloned()
        .chain(std::iter::once(polynomial.clone()))
        .collect();
    MultivariatePolynomial::unify_variables_list(&mut ring);
    let (polynomial, divisors) = ring.split_last().expect("the polynomial itself");
    polynomial.is_zero() || (!divisors.is_empty() && polynomial.reduce(divisors).is_zero())
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
    /// Inequality branches for which a saturation was attempted.
    pub saturations: usize,
    /// Saturations equal to the unit ideal.
    pub saturation_units: usize,
    /// Saturations that timed out or failed, so that the inequality branch
    /// keeps the unsaturated generators.
    pub saturation_fallbacks: usize,
    /// Covers skipped because a solved component contains their zero set.
    pub pruned_covers: usize,
    /// Solved components dropped because another contains them.
    pub redundant_components: usize,
    /// Covers evaluated with the leaf timeout.
    pub leaf_evaluations: usize,
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
    /// The solver call was bypassed because a solved component contains the
    /// cover's zero set.
    Pruned,
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

/// What the search accumulates, shared by the worker threads.
#[derive(Default)]
struct SearchState {
    components: Vec<ProjectedComponent>,
    unresolved: Vec<UnresolvedBranch>,
    nodes: Vec<VariableZeroingNodeResult>,
    stats: VariableZeroingStats,
    first_solution: Option<Duration>,
}

/// What the cover checks of one recursion node share.
struct NodeContext<'n> {
    node_id: usize,
    node_nvars: usize,
    variables: &'n [ZeroingVariable],
    decisions: &'n [SplitDecision],
    conditions: &'n BranchConditions,
    /// Whether the node's unique cover may be split without a solver call.
    contract: bool,
}

/// One cover check and the counters it adds to its node.
struct CoverCheck {
    result: VariableZeroingCoverResult,
    evaluations: usize,
    timeouts: usize,
    splits: usize,
}

impl CoverCheck {
    fn evaluated(
        cover_index: usize,
        outcome: VariableZeroingCoverOutcome,
        elapsed: Duration,
    ) -> Self {
        Self {
            result: VariableZeroingCoverResult {
                cover_index,
                outcome,
                elapsed,
                split_variable: None,
                zero_child: None,
                nonzero_child: None,
            },
            evaluations: 1,
            timeouts: usize::from(outcome == VariableZeroingCoverOutcome::Timeout),
            splits: 0,
        }
    }
}

struct Solver<'a> {
    options: &'a VariableZeroingOptions,
    evaluator: &'a Evaluator<'a>,
    started: Instant,
    state: Mutex<SearchState>,
    random: Mutex<StdRng>,
}

impl<'a> Solver<'a> {
    fn new(options: &'a VariableZeroingOptions, evaluator: &'a Evaluator<'a>) -> Self {
        Self {
            options,
            evaluator,
            started: Instant::now(),
            state: Mutex::default(),
            random: Mutex::new(StdRng::seed_from_u64(options.random_seed)),
        }
    }

    /// The shared state. It is never held across a solver call or a recursive
    /// search, so worker threads only wait for bookkeeping.
    fn state(&self) -> MutexGuard<'_, SearchState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn next_split_value(&self) -> i64 {
        next_split_value(
            self.options.split_value_strategy,
            &mut self.random.lock().unwrap_or_else(PoisonError::into_inner),
        )
    }

    fn runtime_exhausted(&self) -> bool {
        self.options
            .max_runtime
            .is_some_and(|limit| self.started.elapsed() >= limit)
    }

    /// `limit`, cut to what remains of the global budget.
    fn within_budget(&self, limit: Duration) -> Duration {
        self.options.max_runtime.map_or(limit, |budget| {
            limit.min(budget.saturating_sub(self.started.elapsed()))
        })
    }

    fn evaluation_timeout(&self) -> Duration {
        self.within_budget(self.options.timeout)
    }

    fn retain_unresolved(
        &self,
        polynomials: Vec<PolyExtLex>,
        variables: &[ZeroingVariable],
        decisions: &[SplitDecision],
        reason: UnresolvedReason,
    ) {
        self.state().unresolved.push(UnresolvedBranch {
            decisions: decisions.to_vec(),
            polynomials,
            remaining_variables: variables.to_vec(),
            reason,
        });
    }

    fn solve_node(
        &self,
        polynomials: Vec<PolyExtLex>,
        variables: Vec<ZeroingVariable>,
        decisions: Vec<SplitDecision>,
    ) -> Result<Option<usize>, String> {
        self.solve_node_with_contraction(
            polynomials,
            variables,
            decisions,
            BranchConditions::default(),
            false,
        )
    }

    /// The monic polynomials `x - a` of the inequality decisions, in the ring
    /// of `polynomials`, for refinement to treat as nonzero. Only saturated
    /// branches need them; the Rabinowitsch encodings carry the decisions in
    /// their equations.
    fn refinement_conditions(
        &self,
        polynomials: &[PolyExtLex],
        unequal: &[(PolyVariable, i64)],
    ) -> Vec<PolyExtLex> {
        match (self.options.nonzero_encoding, polynomials.first()) {
            (NonzeroEncoding::Saturation, Some(template)) => unequal
                .iter()
                .filter_map(|(variable, value)| difference(template, variable, *value).ok())
                .map(PolyExtLex::make_monic)
                .collect(),
            _ => Vec::new(),
        }
    }

    /// The system the branch solver receives for `candidate`. A saturated
    /// branch keeps its inequality decisions outside its generators, so the
    /// solver receives them as separate Rabinowitsch equations; the
    /// Rabinowitsch encodings already contain them.
    fn solver_input(
        &self,
        candidate: &[PolyExtLex],
        conditions: &BranchConditions,
    ) -> Result<Vec<PolyExtLex>, String> {
        match self.options.nonzero_encoding {
            NonzeroEncoding::Saturation
                if !conditions.unequal.is_empty() && !candidate.is_empty() =>
            {
                with_rabinowitsch_equations(candidate.to_vec(), &conditions.unequal, true)
            }
            _ => Ok(candidate.to_vec()),
        }
    }

    /// Record a node whose factorization failed and retain its branch.
    fn preprocessing_failed(
        &self,
        mut node_result: VariableZeroingNodeResult,
        polynomials: Vec<PolyExtLex>,
        variables: &[ZeroingVariable],
        failure: PreprocessingFailure,
        detail: &str,
    ) -> usize {
        let node_id = node_result.node_id;
        let reason = match failure {
            PreprocessingFailure::Timeout => UnresolvedReason::FactorizationTimeout,
            PreprocessingFailure::Error => UnresolvedReason::FactorizationError,
        };
        self.retain_unresolved(polynomials, variables, &node_result.decisions, reason);
        node_result.preprocessing_failure = Some(failure);
        if self.options.verbose {
            println!(
                "node #{node_id} depth {}: factorization {detail} ({:?})",
                node_result.decisions.len(),
                node_result.elapsed
            );
        }
        let mut state = self.state();
        match failure {
            PreprocessingFailure::Timeout => state.stats.factorization_timeouts += 1,
            PreprocessingFailure::Error => state.stats.factorization_errors += 1,
        }
        state.nodes.push(node_result);
        node_id
    }

    fn solve_node_with_contraction(
        &self,
        polynomials: Vec<PolyExtLex>,
        variables: Vec<ZeroingVariable>,
        decisions: Vec<SplitDecision>,
        conditions: BranchConditions,
        contract_unique_cover: bool,
    ) -> Result<Option<usize>, String> {
        let depth = decisions.len();
        {
            let mut state = self.state();
            state.stats.largest_depth = state.stats.largest_depth.max(depth);
        }
        if self.runtime_exhausted() {
            self.retain_unresolved(
                polynomials,
                &variables,
                &decisions,
                UnresolvedReason::RuntimeBudget,
            );
            return Ok(None);
        }

        let node_id = {
            let mut state = self.state();
            state.stats.nodes += 1;
            state.stats.nodes - 1
        };
        let node_started = Instant::now();
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
        let refinement_conditions = self.refinement_conditions(&polynomials, &conditions.unequal);
        let (groups, covers) = match terminal_hitting_sets_with_timeout(
            polynomials.clone(),
            self.evaluation_timeout(),
            &refinement_conditions,
        ) {
            Ok(preprocessed) => preprocessed,
            Err(HittingSetFailure::Fatal(error)) => return Err(format!("node {node_id}: {error}")),
            Err(HittingSetFailure::Timeout) => {
                node_result.elapsed = node_started.elapsed();
                return Ok(Some(self.preprocessing_failed(
                    node_result,
                    polynomials,
                    &variables,
                    PreprocessingFailure::Timeout,
                    "timed out",
                )));
            }
            Err(HittingSetFailure::Factorization(error)) => {
                node_result.elapsed = node_started.elapsed();
                return Ok(Some(self.preprocessing_failed(
                    node_result,
                    polynomials,
                    &variables,
                    PreprocessingFailure::Error,
                    &format!("failed: {error}"),
                )));
            }
        };
        node_result.covers = covers.len();
        self.state().stats.covers += covers.len();
        if self.options.verbose {
            println!(
                "node #{node_id} depth {depth}: {groups} factor group(s), {} cover(s), {} variable(s) left",
                covers.len(),
                variables.len()
            );
        }

        let preprocessing = node_started.elapsed();
        let context = NodeContext {
            node_id,
            node_nvars,
            variables: &variables,
            decisions: &decisions,
            conditions: &conditions,
            // The caller has already established that this is the unique
            // cover on one side of an equality/inequality split and that the
            // sibling has no cover. Hence this cover represents the entire
            // parent branch, and another zeroing split is sound without first
            // asking the Gröbner solver to rediscover that the branch is
            // difficult.
            contract: contract_unique_cover && covers.len() == 1,
        };
        let checks = covers
            .into_par_iter()
            .enumerate()
            .map(|(cover_index, candidate)| self.check_cover(&context, cover_index, candidate))
            .collect::<Result<Vec<Option<CoverCheck>>, String>>()?;
        for check in checks.into_iter().flatten() {
            node_result.evaluations += check.evaluations;
            node_result.timeouts += check.timeouts;
            node_result.zeroing_splits += check.splits;
            match check.result.outcome {
                VariableZeroingCoverOutcome::Consistent => node_result.consistent += 1,
                VariableZeroingCoverOutcome::Inconsistent => node_result.inconsistent += 1,
                _ => {}
            }
            node_result.elapsed += check.result.elapsed;
            node_result.checks.push(check.result);
        }
        node_result.elapsed += preprocessing;
        let mut state = self.state();
        state.stats.evaluations += node_result.evaluations;
        state.stats.timeouts += node_result.timeouts;
        state.stats.zeroing_splits += node_result.zeroing_splits;
        state.stats.consistent += node_result.consistent;
        state.stats.inconsistent += node_result.inconsistent;
        state.nodes.push(node_result);
        Ok(Some(node_id))
    }

    /// Check one cover of a node: prune it, contract it, or evaluate it and
    /// split it on a timeout. A cover with no variable left to split on is a
    /// leaf and receives the leaf timeout. Returns `None` when the global
    /// budget ran out before the check started.
    fn check_cover(
        &self,
        node: &NodeContext<'_>,
        cover_index: usize,
        candidate: Vec<PolyExtLex>,
    ) -> Result<Option<CoverCheck>, String> {
        let cover_started = Instant::now();
        if self.runtime_exhausted() {
            self.retain_unresolved(
                candidate,
                node.variables,
                node.decisions,
                UnresolvedReason::RuntimeBudget,
            );
            return Ok(None);
        }
        if self.options.prune_by_solutions
            && self
                .state()
                .components
                .iter()
                .any(|component| component.covers_branch(&candidate, node.conditions))
        {
            self.state().stats.pruned_covers += 1;
            if self.options.verbose {
                println!("  cover #{cover_index}: pruned, inside a solved component");
            }
            let mut check = CoverCheck::evaluated(
                cover_index,
                VariableZeroingCoverOutcome::Pruned,
                cover_started.elapsed(),
            );
            check.evaluations = 0;
            return Ok(Some(check));
        }
        let split_variable = choose_variable(&candidate, node.variables);
        if node.contract
            && let Some(selected_index) = split_variable
        {
            let selected = node.variables[selected_index].name.clone();
            if self.options.verbose {
                println!("  cover #{cover_index}: branch-contracted; splitting {selected}");
            }
            let (zero_child, nonzero_child, recursive_elapsed) = self.split_and_solve(
                candidate,
                node.variables,
                node.decisions,
                node.conditions,
                selected_index,
            )?;
            return Ok(Some(CoverCheck {
                result: VariableZeroingCoverResult {
                    cover_index,
                    outcome: VariableZeroingCoverOutcome::Contracted,
                    elapsed: cover_started.elapsed().saturating_sub(recursive_elapsed),
                    split_variable: Some(selected),
                    zero_child,
                    nonzero_child,
                },
                evaluations: 0,
                timeouts: 0,
                splits: 1,
            }));
        }

        let limit = match (split_variable, self.options.leaf_timeout) {
            (None, Some(leaf_timeout)) => {
                self.state().stats.leaf_evaluations += 1;
                leaf_timeout
            }
            _ => self.options.timeout,
        };
        let solver_input = self.solver_input(&candidate, node.conditions)?;
        let nvars = solver_input
            .first()
            .map_or(node.node_nvars, PolyExtLex::nvars);
        let outcome = (self.evaluator)(
            solver_input,
            self.within_budget(limit),
            self.options.gb_backend,
        );
        let cover_elapsed = cover_started.elapsed();
        let evaluated = |outcome| {
            Ok(Some(CoverCheck::evaluated(
                cover_index,
                outcome,
                cover_elapsed,
            )))
        };
        match outcome {
            Outcome::Consistent(basis) => {
                self.record_consistent_components(
                    vec![basis],
                    node,
                    nvars,
                    cover_index,
                    cover_elapsed,
                );
                evaluated(VariableZeroingCoverOutcome::Consistent)
            }
            Outcome::ConsistentComponents(bases) => {
                self.record_consistent_components(bases, node, nvars, cover_index, cover_elapsed);
                evaluated(VariableZeroingCoverOutcome::Consistent)
            }
            Outcome::Inconsistent => {
                if self.options.verbose {
                    println!("  cover #{cover_index}: inconsistent ({cover_elapsed:?})");
                }
                evaluated(VariableZeroingCoverOutcome::Inconsistent)
            }
            Outcome::Timeout => {
                if self.options.verbose {
                    println!(
                        "  cover #{cover_index}: timed-out hitting set ({} generators) under {}",
                        candidate.len(),
                        format_decision_path(node.decisions)
                    );
                    for (index, polynomial) in candidate.iter().enumerate() {
                        println!("    t{index} = {polynomial}");
                    }
                }
                let selected_index = match (split_variable, self.runtime_exhausted()) {
                    (Some(selected_index), false) => selected_index,
                    (_, exhausted) => {
                        let reason = if exhausted {
                            UnresolvedReason::RuntimeBudget
                        } else if node.variables.is_empty() {
                            UnresolvedReason::TimeoutAfterVariablesExhausted
                        } else {
                            UnresolvedReason::NoUsefulVariable
                        };
                        self.retain_unresolved(candidate, node.variables, node.decisions, reason);
                        return evaluated(VariableZeroingCoverOutcome::Timeout);
                    }
                };
                let selected = node.variables[selected_index].name.clone();
                if self.options.verbose {
                    println!(
                        "  cover #{cover_index}: timed out ({cover_elapsed:?}); splitting {selected}"
                    );
                }
                let (zero_child, nonzero_child, _) = self.split_and_solve(
                    candidate,
                    node.variables,
                    node.decisions,
                    node.conditions,
                    selected_index,
                )?;
                let mut check = CoverCheck::evaluated(
                    cover_index,
                    VariableZeroingCoverOutcome::Timeout,
                    cover_elapsed,
                );
                check.result.split_variable = Some(selected);
                check.result.zero_child = zero_child;
                check.result.nonzero_child = nonzero_child;
                check.splits = 1;
                Ok(Some(check))
            }
            Outcome::BackendError(error) => Err(format!(
                "node {}, cover {cover_index}, {} backend: {error}",
                node.node_id,
                self.options.gb_backend.label()
            )),
        }
    }

    /// Record solved components; `nvars` is the size of the solver input's
    /// ring, auxiliary variables included.
    fn record_consistent_components(
        &self,
        bases: Vec<ComputedBasis>,
        node: &NodeContext<'_>,
        nvars: usize,
        cover_index: usize,
        elapsed: Duration,
    ) {
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
        let components = bases.into_iter().map(|basis| ProjectedComponent {
            node_id: node.node_id,
            decisions: node.decisions.to_vec(),
            dimension: solution_dimension(&basis.elements, nvars),
            basis,
            conditions: node.conditions.clone(),
        });
        let mut state = self.state();
        state
            .first_solution
            .get_or_insert_with(|| self.started.elapsed());
        state.components.extend(components);
    }

    /// Split `candidate` on the selected variable and solve both children,
    /// concurrently. Returns the children's node ids and the wall time spent
    /// on them.
    fn split_and_solve(
        &self,
        candidate: Vec<PolyExtLex>,
        variables: &[ZeroingVariable],
        decisions: &[SplitDecision],
        conditions: &BranchConditions,
        selected_index: usize,
    ) -> Result<(Option<usize>, Option<usize>, Duration), String> {
        let selected = variables[selected_index].clone();
        let mut remaining = variables.to_vec();
        remaining.remove(selected_index);
        let split_value = self.next_split_value();
        let decision = (selected.variable.clone(), split_value);
        let equal_conditions = BranchConditions {
            equal: [conditions.equal.as_slice(), std::slice::from_ref(&decision)].concat(),
            unequal: conditions.unequal.clone(),
        };
        let unequal_conditions = BranchConditions {
            equal: conditions.equal.clone(),
            unequal: [conditions.unequal.as_slice(), &[decision]].concat(),
        };
        let (equal_branch, rabinowitsch_branch) = split_on_variable(
            candidate.clone(),
            &selected.variable,
            split_value,
            &unequal_conditions.unequal,
            self.options.nonzero_encoding != NonzeroEncoding::SharedProduct,
        )?;
        let unequal_branch = match self.options.nonzero_encoding {
            NonzeroEncoding::Saturation => self
                .saturated_branch(&candidate, &unequal_conditions.unequal)
                .unwrap_or_else(|| candidate.clone()),
            NonzeroEncoding::Separate | NonzeroEncoding::SharedProduct => rabinowitsch_branch,
        };

        let equal_cover_count = terminal_cover_count(
            &equal_branch,
            self.evaluation_timeout(),
            &self.refinement_conditions(&equal_branch, &equal_conditions.unequal),
        )?;
        let unequal_cover_count = if equal_cover_count.is_some() {
            terminal_cover_count(
                &unequal_branch,
                self.evaluation_timeout(),
                &self.refinement_conditions(&unequal_branch, &unequal_conditions.unequal),
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

        let child_decisions = |is_equal| {
            [
                decisions,
                &[SplitDecision {
                    variable: selected.name.clone(),
                    value: split_value,
                    is_equal,
                }],
            ]
            .concat()
        };
        let equal_decisions = child_decisions(true);
        let unequal_decisions = child_decisions(false);
        let equal_remaining = remaining.clone();
        let children_started = Instant::now();
        let (equal_child, unequal_child) = rayon::join(
            move || {
                self.solve_node_with_contraction(
                    equal_branch,
                    equal_remaining,
                    equal_decisions,
                    equal_conditions,
                    contract_equal,
                )
            },
            move || {
                self.solve_node_with_contraction(
                    unequal_branch,
                    remaining,
                    unequal_decisions,
                    unequal_conditions,
                    contract_unequal,
                )
            },
        );
        Ok((equal_child?, unequal_child?, children_started.elapsed()))
    }

    /// The saturation of the Rabinowitsch-free generators of `candidate` by
    /// the product of the `x - a` in `unequal`, or `None` when it did not
    /// finish within the per-cover timeout. The unit ideal is returned as `{1}`.
    fn saturated_branch(
        &self,
        candidate: &[PolyExtLex],
        unequal: &[(PolyVariable, i64)],
    ) -> Option<Vec<PolyExtLex>> {
        let ordinary: Vec<PolyExtLex> = candidate
            .iter()
            .filter(|polynomial| !uses_temporary_variable(polynomial))
            .cloned()
            .collect();
        let template = ordinary.first()?.clone();
        let divisors = unequal
            .iter()
            .map(|(variable, value)| difference(&template, variable, *value))
            .collect::<Result<Vec<_>, _>>()
            .ok()?;
        self.state().stats.saturations += 1;
        let started = Instant::now();
        let (branch, description) = match saturate(&ordinary, &divisors, self.evaluation_timeout())
        {
            Saturation::Basis(basis) => {
                let description = format!("{} generator(s)", basis.len());
                (Some(basis), description)
            }
            Saturation::Unit => {
                self.state().stats.saturation_units += 1;
                (Some(vec![template.one()]), "the unit ideal".to_string())
            }
            Saturation::Timeout => {
                self.state().stats.saturation_fallbacks += 1;
                (
                    None,
                    "timed out; keeping the unsaturated generators".to_string(),
                )
            }
            Saturation::Error(error) => {
                self.state().stats.saturation_fallbacks += 1;
                (
                    None,
                    format!("failed ({error}); keeping the unsaturated generators"),
                )
            }
        };
        if self.options.verbose {
            println!(
                "  saturation by {} condition(s): {description} ({:?})",
                unequal.len(),
                started.elapsed()
            );
        }
        branch
    }
}

/// Drop every component that certifiably lies in a kept one, visiting them in
/// order; a newly kept component evicts the kept components it contains, so
/// the union of the components' sets is preserved.
fn drop_redundant_components(
    components: Vec<ProjectedComponent>,
) -> (Vec<ProjectedComponent>, usize) {
    let count = components.len();
    let kept =
        components
            .into_iter()
            .fold(Vec::new(), |kept: Vec<ProjectedComponent>, component| {
                if kept.iter().any(|existing| component.lies_in(existing)) {
                    return kept;
                }
                let mut kept: Vec<ProjectedComponent> = kept
                    .into_iter()
                    .filter(|existing| !existing.lies_in(&component))
                    .collect();
                kept.push(component);
                kept
            });
    let dropped = count - kept.len();
    (kept, dropped)
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
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(options.jobs.max(1))
        .build()
        .map_err(|error| format!("cannot start {} worker thread(s): {error}", options.jobs))?;
    let solver = Solver::new(options, evaluator);
    pool.install(|| solver.solve_node(polynomials, variables, Vec::new()))?;
    let elapsed = solver.started.elapsed();
    let SearchState {
        components,
        unresolved,
        mut nodes,
        mut stats,
        first_solution,
    } = solver
        .state
        .into_inner()
        .unwrap_or_else(PoisonError::into_inner);
    let components = if options.prune_by_solutions {
        let (kept, dropped) = drop_redundant_components(components);
        stats.redundant_components = dropped;
        kept
    } else {
        components
    };
    nodes.sort_unstable_by_key(|node| node.node_id);
    Ok(VariableZeroingResult {
        components,
        unresolved,
        nodes,
        stats,
        first_solution,
        elapsed,
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
                .variables()
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
                .variables()
                .iter()
                .position(|candidate| candidate == variable)
                .map_or_else(
                    || polynomial.clone(),
                    |position| polynomial.replace(position, &split_value),
                )
        })
        .collect();
    Ok((
        equal_branch,
        with_rabinowitsch_equations(candidate, inequality_conditions, saturate_each_variable)?,
    ))
}

/// The Rabinowitsch-free generators of `candidate` together with Rabinowitsch
/// equations for `inequality_conditions`: one per condition, or one for
/// their product.
fn with_rabinowitsch_equations(
    candidate: Vec<PolyExtLex>,
    inequality_conditions: &[(PolyVariable, i64)],
    saturate_each_variable: bool,
) -> Result<Vec<PolyExtLex>, String> {
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
            let product = inverse * &difference(&template, variable, *value)?;
            unequal_branch.push(template.one() - product);
        }
    } else {
        let mut product = template.variable(&PolyVariable::Temporary(0))?;
        for (variable, value) in inequality_conditions {
            product = product * &difference(&template, variable, *value)?;
        }
        unequal_branch.push(template.one() - product);
    }
    Ok(unequal_branch)
}

/// The polynomial `x - a` in the ring of `template`.
fn difference(
    template: &PolyExtLex,
    variable: &PolyVariable,
    value: i64,
) -> Result<PolyExtLex, String> {
    Ok(template.variable(variable)? - template.constant(Rational::from(value)))
}

fn uses_variable<O: MonomialOrder>(
    polynomial: &MultivariatePolynomial<Q, u16, O>,
    variable: &PolyVariable,
) -> bool {
    let Some(position) = polynomial
        .variables()
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
        .variables()
        .iter()
        .filter(|variable| matches!(variable, PolyVariable::Temporary(_)))
        .any(|variable| uses_variable(polynomial, variable))
}

fn factor_groups_with_timeout(
    polynomials: Vec<PolyExtLex>,
    timeout: Duration,
) -> Result<ProcessRun<Vec<Vec<PolyExtLex>>>, String> {
    if cfg!(test) || running_under_test_harness() {
        try_factor_groups_from_polys(polynomials).map(ProcessRun::Completed)
    } else {
        factor_groups_isolated(polynomials, timeout)
    }
}

/// Why a node's terminal hitting sets could not be computed.
#[derive(Debug)]
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
        factor_before_deadline(polynomials, self.deadline)
    }
}

fn factor_before_deadline(
    polynomials: Vec<PolyExtLex>,
    deadline: Instant,
) -> Result<FactorGroups, HittingSetFailure> {
    let remaining = remaining_until(deadline)?;
    completed(factor_groups_with_timeout(polynomials, remaining))
}

/// Factor a node's equations without interreduction and refine its minimal hitting sets into
/// terminal hitting sets (Algorithm `Refine`), all within `timeout`. Returns
/// the number of factor groups and the terminal hitting sets.
fn terminal_hitting_sets_with_timeout(
    polynomials: Vec<PolyExtLex>,
    timeout: Duration,
    conditions: &[PolyExtLex],
) -> Result<(usize, Vec<Vec<PolyExtLex>>), HittingSetFailure> {
    let deadline = Instant::now() + timeout;
    let groups = without_conditions(factor_before_deadline(polynomials, deadline)?, conditions);
    let (factors, index_groups) = build_factor_index(&groups);
    let covers = minimal_covers(&index_groups, factors.len()).ok_or_else(|| {
        HittingSetFailure::Fatal(format!(
            "{} distinct factors; the cover enumerator supports at most 63",
            factors.len()
        ))
    })?;
    let (terminal, _) = refine_covers_under_conditions(
        &factors,
        &ordered_cover_masks(&covers),
        conditions,
        &mut IsolatedBeforeDeadline { deadline },
    )?;
    Ok((groups.len(), terminal))
}

/// Count the terminal hitting sets of a split child without invoking a
/// Gröbner-basis backend. A preprocessing timeout disables branch contraction;
/// the child node will retain the branch as unresolved if factorization times
/// out again.
fn terminal_cover_count(
    polynomials: &[PolyExtLex],
    timeout: Duration,
    conditions: &[PolyExtLex],
) -> Result<Option<usize>, String> {
    match terminal_hitting_sets_with_timeout(polynomials.to_vec(), timeout, conditions) {
        Ok((_, terminal)) => Ok(Some(terminal.len())),
        Err(HittingSetFailure::Timeout | HittingSetFailure::Factorization(_)) => Ok(None),
        Err(HittingSetFailure::Fatal(error)) => Err(format!("split child: {error}")),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Once;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use symbolica::domains::rational::Q;
    use symbolica::license::LicenseManager;
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
        assert_eq!(options.nonzero_encoding, NonzeroEncoding::Separate);
        assert!(!options.prune_by_solutions);
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
            .variables()
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
    fn parallel_search_builds_the_same_tree() {
        let (loaded, polynomials) = load("{(x + y*z)*w == 0, x*v == 0}");
        let always_timeout = |_: Vec<PolyExtLex>, _: Duration, _: GbBackend| Outcome::Timeout;
        let search = |jobs| {
            let options = VariableZeroingOptions {
                jobs,
                ..VariableZeroingOptions::default()
            };
            solve_by_zeroing_with(
                polynomials.clone(),
                candidates(&loaded),
                &options,
                &always_timeout,
            )
            .expect("search")
        };

        let sequential = search(1);
        let parallel = search(4);

        assert_eq!(parallel.stats.nodes, sequential.stats.nodes);
        assert_eq!(
            parallel.stats.zeroing_splits,
            sequential.stats.zeroing_splits
        );
        assert_eq!(parallel.stats.evaluations, sequential.stats.evaluations);
        assert_eq!(parallel.unresolved.len(), sequential.unresolved.len());
        let decision_sets = |result: &VariableZeroingResult| {
            let mut sets = result
                .nodes
                .iter()
                .map(|node| format_decision_path(&node.decisions))
                .collect::<Vec<_>>();
            sets.sort();
            sets
        };
        assert_eq!(decision_sets(&parallel), decision_sets(&sequential));
    }

    #[test]
    fn a_cover_without_a_split_variable_gets_the_leaf_timeout() {
        let (_, polynomials) = load("{x + y == 0}");
        let options = VariableZeroingOptions {
            timeout: Duration::from_millis(5),
            leaf_timeout: Some(Duration::from_secs(600)),
            ..VariableZeroingOptions::default()
        };
        let longest_limit = std::sync::Mutex::new(Duration::ZERO);
        let evaluator = |_: Vec<PolyExtLex>, limit: Duration, _: GbBackend| {
            let mut longest = longest_limit.lock().unwrap();
            *longest = (*longest).max(limit);
            Outcome::Timeout
        };

        let result =
            solve_by_zeroing_with(polynomials, Vec::new(), &options, &evaluator).expect("search");

        assert_eq!(*longest_limit.lock().unwrap(), Duration::from_secs(600));
        assert_eq!(result.stats.leaf_evaluations, 1);
        assert_eq!(
            result.unresolved[0].reason,
            UnresolvedReason::TimeoutAfterVariablesExhausted
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
            conditions: BranchConditions::default(),
        };

        let projected = component.original_basis_elements().collect::<Vec<_>>();
        assert_eq!(projected.len(), 1);
        assert!(!projected[0].to_string().contains("_TMP_"));
    }

    fn variable(loaded: &LoadedMathematicaSystem, name: &str) -> PolyVariable {
        PolyVariable::from(
            loaded.id_to_sym[loaded
                .var_names
                .iter()
                .position(|candidate| candidate == name)
                .unwrap()],
        )
    }

    /// A solved component with the full basis `sources`, in the ring of `ring`.
    fn component(
        ring: &[PolyExtLex],
        sources: &[&str],
        conditions: BranchConditions,
    ) -> ProjectedComponent {
        let (_, polynomials) = load(&format!(
            "{{{}}}",
            sources
                .iter()
                .map(|source| format!("{source} == 0"))
                .collect::<Vec<_>>()
                .join(", ")
        ));
        let mut elements: Vec<PolyExtLex> = ring.iter().cloned().chain(polynomials).collect();
        MultivariatePolynomial::unify_variables_list(&mut elements);
        ProjectedComponent {
            node_id: 0,
            decisions: Vec::new(),
            dimension: None,
            basis: ComputedBasis {
                elements: elements[ring.len()..]
                    .iter()
                    .map(|polynomial| polynomial.reorder::<GrevLexOrder>())
                    .collect(),
                leading_monomials_only: false,
            },
            conditions,
        }
    }

    #[test]
    fn solved_component_covers_a_branch_inside_it_under_more_decisions() {
        let (loaded, polynomials) = load("{x*z == 0, y == 0, z - 1 == 0}");
        let x = variable(&loaded, "x");
        let solved = component(&polynomials, &["x + y"], BranchConditions::default());
        let branch = vec![polynomials[1].clone()];
        let x_zero = BranchConditions {
            equal: vec![(x.clone(), 0)],
            unequal: Vec::new(),
        };

        assert!(solved.covers_branch(&branch, &x_zero));
        assert!(!solved.covers_branch(&branch, &BranchConditions::default()));
        let x_nonzero_only = component(
            &polynomials,
            &["x + y"],
            BranchConditions {
                equal: Vec::new(),
                unequal: vec![(x, 0)],
            },
        );
        assert!(!x_nonzero_only.covers_branch(&branch, &x_zero));
    }

    #[test]
    fn redundant_components_are_dropped_by_exact_containment() {
        let (_, polynomials) = load("{x*y == 0}");
        let plane = component(&polynomials, &["x"], BranchConditions::default());
        let line = component(&polynomials, &["x", "y"], BranchConditions::default());
        let other_plane = component(&polynomials, &["y"], BranchConditions::default());

        let (kept, dropped) = drop_redundant_components(vec![line, plane, other_plane]);

        assert_eq!(dropped, 1);
        assert_eq!(kept.len(), 2);
        assert!(kept.iter().all(|component| component.basis_len() == 1));
    }

    #[test]
    fn saturated_refinement_drops_the_factor_assumed_nonzero() {
        let (loaded, polynomials) = load("{x*y == 0, x*(z - 1) == 0}");
        let x = variable(&loaded, "x");
        let options = VariableZeroingOptions {
            nonzero_encoding: NonzeroEncoding::Saturation,
            ..VariableZeroingOptions::default()
        };
        let always_timeout = |_: Vec<PolyExtLex>, _: Duration, _: GbBackend| Outcome::Timeout;
        let solver = Solver::new(&options, &always_timeout);
        let conditions = solver.refinement_conditions(&polynomials, &[(x, 0)]);

        let (_, unconditioned) =
            terminal_hitting_sets_with_timeout(polynomials.clone(), Duration::from_secs(5), &[])
                .expect("covers");
        let (_, conditioned) =
            terminal_hitting_sets_with_timeout(polynomials, Duration::from_secs(5), &conditions)
                .expect("covers");

        assert_eq!(unconditioned.len(), 2);
        assert_eq!(conditioned.len(), 1);
        assert_eq!(conditioned[0].len(), 2);
    }

    #[test]
    fn saturated_branches_give_the_solver_rabinowitsch_equations() {
        let (loaded, polynomials) = load("{x*y - x == 0, x*z == 0}");
        let x = variable(&loaded, "x");
        let options = VariableZeroingOptions {
            nonzero_encoding: NonzeroEncoding::Saturation,
            ..VariableZeroingOptions::default()
        };
        let always_timeout = |_: Vec<PolyExtLex>, _: Duration, _: GbBackend| Outcome::Timeout;
        let solver = Solver::new(&options, &always_timeout);
        let x_nonzero = BranchConditions {
            equal: Vec::new(),
            unequal: vec![(x, 0)],
        };

        let conditioned = solver
            .solver_input(&polynomials, &x_nonzero)
            .expect("solver input");
        let unconditioned = solver
            .solver_input(&polynomials, &BranchConditions::default())
            .expect("solver input");

        assert_eq!(conditioned.len(), polynomials.len() + 1);
        assert_eq!(
            conditioned
                .iter()
                .filter(|polynomial| uses_temporary_variable(polynomial))
                .count(),
            1
        );
        assert_eq!(unconditioned, polynomials);
    }

    #[test]
    fn ideal_saturation_replaces_the_rabinowitsch_equation() {
        if std::process::Command::new("Singular")
            .arg("--version")
            .output()
            .is_err()
        {
            return;
        }
        let (loaded, polynomials) = load("{x*y - x == 0, x*z == 0}");
        let x = variable(&loaded, "x");
        let options = VariableZeroingOptions {
            nonzero_encoding: NonzeroEncoding::Saturation,
            ..VariableZeroingOptions::default()
        };
        let always_timeout = |_: Vec<PolyExtLex>, _: Duration, _: GbBackend| Outcome::Timeout;
        let solver = Solver::new(&options, &always_timeout);

        let saturated = solver
            .saturated_branch(&polynomials, &[(x.clone(), 0)])
            .expect("saturation");

        assert_eq!(solver.state().stats.saturations, 1);
        assert_eq!(saturated.len(), 2);
        assert!(
            saturated
                .iter()
                .all(|polynomial| !uses_temporary_variable(polynomial))
        );
        assert!(
            saturated
                .iter()
                .all(|polynomial| !uses_variable(polynomial, &x))
        );
    }

    #[test]
    fn unit_saturation_gives_a_branch_without_covers() {
        if std::process::Command::new("Singular")
            .arg("--version")
            .output()
            .is_err()
        {
            return;
        }
        let (loaded, polynomials) = load("{x*y == 0, y - 1 == 0}");
        let x = variable(&loaded, "x");
        let options = VariableZeroingOptions {
            nonzero_encoding: NonzeroEncoding::Saturation,
            ..VariableZeroingOptions::default()
        };
        let always_timeout = |_: Vec<PolyExtLex>, _: Duration, _: GbBackend| Outcome::Timeout;
        let solver = Solver::new(&options, &always_timeout);

        let saturated = solver
            .saturated_branch(&polynomials, &[(x, 0)])
            .expect("saturation");

        assert_eq!(solver.state().stats.saturation_units, 1);
        let (_, covers) =
            terminal_hitting_sets_with_timeout(saturated, Duration::from_secs(5), &[])
                .expect("covers");
        assert!(covers.is_empty());
    }
}
