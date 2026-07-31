//! Proof-carrying unsatisfiable-core extraction for polynomial systems.
//!
//! This is Algorithm 2 (GB-Core) and the iterative refutation-tree refinement
//! from Section 5 of Sun et al., CP 2016.  The branching binaries use it after
//! a Gröbner backend has certified that a factor cover generates the unit ideal.

use std::collections::VecDeque;
use std::sync::mpsc;
use std::time::Duration;

use symbolica::domains::rational::Rational;
use symbolica::domains::{Field, Ring};

use crate::factor_prep::PolyExtLex;
use crate::gb_search::{Core, PolyExtGrevLex};

#[derive(Clone)]
struct ProofStats {
    distance: Vec<usize>,
    frequency: Vec<usize>,
}

#[derive(Clone)]
struct TrackedPolynomial {
    polynomial: PolyExtGrevLex,
    coefficients: Vec<PolyExtGrevLex>,
    stats: ProofStats,
}

struct GbCoreRun {
    core: Core,
    ranked_core: Core,
}

fn monomial_divides(divisor: &[u16], dividend: &[u16]) -> bool {
    divisor
        .iter()
        .zip(dividend)
        .all(|(left, right)| left <= right)
}

fn monomial_quotient(dividend: &[u16], divisor: &[u16]) -> Vec<u16> {
    dividend
        .iter()
        .zip(divisor)
        .map(|(left, right)| left - right)
        .collect()
}

fn monomial_lcm(left: &[u16], right: &[u16]) -> Vec<u16> {
    left.iter()
        .zip(right)
        .map(|(left, right)| (*left).max(*right))
        .collect()
}

fn scale_polynomial(
    polynomial: &PolyExtGrevLex,
    exponents: &[u16],
    coefficient: &Rational,
) -> PolyExtGrevLex {
    polynomial
        .clone()
        .mul_exp(exponents)
        .mul_coeff(coefficient.clone())
}

fn scale_tracked(
    tracked: &TrackedPolynomial,
    exponents: &[u16],
    coefficient: &Rational,
) -> TrackedPolynomial {
    TrackedPolynomial {
        polynomial: scale_polynomial(&tracked.polynomial, exponents, coefficient),
        coefficients: tracked
            .coefficients
            .iter()
            .map(|polynomial| scale_polynomial(polynomial, exponents, coefficient))
            .collect(),
        stats: tracked.stats.clone(),
    }
}

fn subtract_tracked(left: TrackedPolynomial, right: TrackedPolynomial) -> TrackedPolynomial {
    TrackedPolynomial {
        polynomial: left.polynomial - right.polynomial,
        coefficients: left
            .coefficients
            .into_iter()
            .zip(right.coefficients)
            .map(|(left, right)| left - right)
            .collect(),
        stats: left.stats,
    }
}

fn dependency_stats(
    basis: &[TrackedPolynomial],
    parents: &[usize],
    generator_count: usize,
) -> ProofStats {
    let mut distance = vec![usize::MAX; generator_count];
    let mut frequency = vec![0usize; generator_count];
    for &parent in parents {
        for generator in 0..generator_count {
            let parent_frequency = basis[parent].stats.frequency[generator];
            if parent_frequency == 0 {
                continue;
            }
            frequency[generator] = frequency[generator].saturating_add(parent_frequency);
            distance[generator] =
                distance[generator].min(basis[parent].stats.distance[generator].saturating_add(1));
        }
    }
    ProofStats {
        distance,
        frequency,
    }
}

fn s_polynomial(basis: &[TrackedPolynomial], left: usize, right: usize) -> TrackedPolynomial {
    let left_lead = basis[left].polynomial.max_exp();
    let right_lead = basis[right].polynomial.max_exp();
    let lcm = monomial_lcm(left_lead, right_lead);
    let left_exponents = monomial_quotient(&lcm, left_lead);
    let right_exponents = monomial_quotient(&lcm, right_lead);
    let ring = &basis[left].polynomial.ring;
    let left_coefficient = ring.div(&ring.one(), basis[left].polynomial.max_coeff());
    let right_coefficient = ring.div(&ring.one(), basis[right].polynomial.max_coeff());
    subtract_tracked(
        scale_tracked(&basis[left], &left_exponents, &left_coefficient),
        scale_tracked(&basis[right], &right_exponents, &right_coefficient),
    )
}

fn reduce_tracked(
    mut tracked: TrackedPolynomial,
    basis: &[TrackedPolynomial],
) -> (TrackedPolynomial, Vec<usize>) {
    let mut reducers = Vec::new();
    loop {
        let reducible = (0..tracked.polynomial.nterms()).find_map(|term| {
            basis.iter().enumerate().find_map(|(index, divisor)| {
                monomial_divides(
                    divisor.polynomial.max_exp(),
                    tracked.polynomial.exponents(term),
                )
                .then_some((term, index))
            })
        });
        let Some((term, divisor_index)) = reducible else {
            return (tracked, reducers);
        };
        let divisor = &basis[divisor_index];
        let exponents = monomial_quotient(
            tracked.polynomial.exponents(term),
            divisor.polynomial.max_exp(),
        );
        let coefficient = tracked.polynomial.ring.div(
            &tracked.polynomial.coefficients[term],
            divisor.polynomial.max_coeff(),
        );
        tracked = subtract_tracked(tracked, scale_tracked(divisor, &exponents, &coefficient));
        reducers.push(divisor_index);
    }
}

fn make_tracked_monic(mut tracked: TrackedPolynomial) -> TrackedPolynomial {
    let coefficient = tracked.polynomial.ring.div(
        &tracked.polynomial.ring.one(),
        tracked.polynomial.max_coeff(),
    );
    tracked.polynomial = tracked.polynomial.mul_coeff(coefficient.clone());
    tracked.coefficients = tracked
        .coefficients
        .into_iter()
        .map(|polynomial| polynomial.mul_coeff(coefficient.clone()))
        .collect();
    tracked
}

fn gb_core_once(candidate: Vec<PolyExtLex>) -> Option<GbCoreRun> {
    let generators: Core = candidate
        .into_iter()
        .filter(|polynomial| !polynomial.is_zero())
        .collect();
    if generators.is_empty() {
        return None;
    }
    let generator_count = generators.len();
    let grevlex: Vec<PolyExtGrevLex> = generators
        .iter()
        .map(|polynomial| polynomial.reorder())
        .collect();
    let zero = grevlex[0].zero();
    let one = grevlex[0].monomial(grevlex[0].ring.one(), vec![0; grevlex[0].nvars()]);
    let mut basis = Vec::with_capacity(generator_count);
    for (index, polynomial) in grevlex.into_iter().enumerate() {
        let mut coefficients = vec![zero.clone(); generator_count];
        coefficients[index] = one.clone();
        let mut distance = vec![usize::MAX; generator_count];
        let mut frequency = vec![0; generator_count];
        distance[index] = 0;
        frequency[index] = 1;
        basis.push(make_tracked_monic(TrackedPolynomial {
            polynomial,
            coefficients,
            stats: ProofStats {
                distance,
                frequency,
            },
        }));
    }

    let finish = |unit: &TrackedPolynomial| {
        let mut included: Vec<usize> = unit
            .coefficients
            .iter()
            .enumerate()
            .filter_map(|(index, coefficient)| (!coefficient.is_zero()).then_some(index))
            .collect();
        let core = included
            .iter()
            .map(|&index| generators[index].clone())
            .collect();
        included.sort_by_key(|&index| {
            (
                unit.stats.distance[index],
                std::cmp::Reverse(unit.stats.frequency[index]),
                index,
            )
        });
        let ranked_core = included
            .iter()
            .map(|&index| generators[index].clone())
            .collect();
        GbCoreRun { core, ranked_core }
    };

    if let Some(unit) = basis
        .iter()
        .find(|tracked| tracked.polynomial.is_constant())
    {
        return Some(finish(unit));
    }

    let mut pairs = VecDeque::new();
    for right in 1..basis.len() {
        for left in 0..right {
            pairs.push_back((left, right));
        }
    }
    while let Some((left, right)) = pairs.pop_front() {
        let spoly = s_polynomial(&basis, left, right);
        let (mut remainder, reducers) = reduce_tracked(spoly, &basis);
        if remainder.polynomial.is_zero() {
            continue;
        }
        let mut parents = Vec::with_capacity(reducers.len() + 2);
        parents.push(left);
        parents.push(right);
        parents.extend(reducers);
        remainder.stats = dependency_stats(&basis, &parents, generator_count);
        remainder = make_tracked_monic(remainder);
        if remainder.polynomial.is_constant() {
            return Some(finish(&remainder));
        }
        let new_index = basis.len();
        for old_index in 0..new_index {
            pairs.push_back((old_index, new_index));
        }
        basis.push(remainder);
    }
    None
}

/// Run proof-carrying GB-Core repeatedly using refutation distance and
/// frequency to reorder generators. Returns `None` if a run times out or does
/// not obtain a unit certificate.
pub fn paper_gb_core(candidate: &[PolyExtLex], timeout: Duration) -> Option<Core> {
    let mut current = candidate.to_vec();
    let mut first_run = true;
    loop {
        let input_len = current.len();
        let (sender, receiver) = mpsc::channel();
        let input = current.clone();
        std::thread::spawn(move || {
            let _ = sender.send(gb_core_once(input));
        });
        let run = receiver.recv_timeout(timeout).ok()??;
        let shrunk = run.ranked_core.len() < input_len;
        let reordered = run.ranked_core != current;
        current = run.ranked_core;
        if shrunk || (first_run && reordered) {
            first_run = false;
            continue;
        }
        return Some(run.core);
    }
}

/// Extract an iteratively refined unsatisfiable core. If refinement cannot
/// finish within the time budget, the certified-unsatisfiable input is kept as
/// a sound, though potentially larger, core.
pub fn iteratively_refined_unsat_core(candidate: &[PolyExtLex], timeout: Duration) -> Core {
    paper_gb_core(candidate, timeout).unwrap_or_else(|| candidate.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Once;

    use symbolica::domains::rational::Q;
    use symbolica::LicenseManager;

    use crate::gb_search::{evaluate_cover, GbBackend, Outcome};
    use crate::{load_mathematica_system_from_str_with_var_names, polynomials_from_loaded};

    static LICENSE: Once = Once::new();

    fn activate_license() {
        LICENSE.call_once(|| {
            let _ = LicenseManager::set_license_key(
                "5568d092#6bb8b4fb#ff936916-65ba-552f-b3c1-bdc64597cd21",
            );
        });
    }

    #[test]
    fn extracts_a_sound_proper_core() {
        activate_license();
        let loaded =
            load_mathematica_system_from_str_with_var_names("{x == 0, x - 1 == 0, y == 0}")
                .expect("load");
        let candidate = polynomials_from_loaded(&loaded, &Q).expect("polynomials");

        let core = iteratively_refined_unsat_core(&candidate, Duration::from_secs(5));

        assert_eq!(core.len(), 2);
        assert!(core.iter().all(|polynomial| candidate.contains(polynomial)));
        assert!(matches!(
            evaluate_cover(core, Duration::from_secs(5), GbBackend::Symbolica),
            Outcome::Inconsistent
        ));
    }
}
