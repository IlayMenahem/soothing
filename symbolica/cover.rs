//! Minimal factor covers and deterministic subset enumeration over bitmasks.

use std::collections::HashMap;

use crate::factor_prep::PolyExtLex;

pub fn build_factor_index(groups: &[Vec<PolyExtLex>]) -> (Vec<PolyExtLex>, Vec<Vec<usize>>) {
    let mut factor_index: HashMap<PolyExtLex, usize> = HashMap::new();
    let mut factors: Vec<PolyExtLex> = Vec::new();
    let index_groups = groups
        .iter()
        .map(|group| {
            group
                .iter()
                .map(|f| {
                    let monic = f.clone().make_monic();
                    *factor_index.entry(monic.clone()).or_insert_with(|| {
                        factors.push(monic);
                        factors.len() - 1
                    })
                })
                .collect()
        })
        .collect();
    (factors, index_groups)
}

/// Each factor group as a bitmask over the factor universe, deduplicated.
fn group_masks(index_groups: &[Vec<usize>]) -> Vec<u64> {
    let mut masks: Vec<u64> = index_groups
        .iter()
        .map(|ids| ids.iter().fold(0u64, |mask, &i| mask | (1u64 << i)))
        .collect();
    masks.sort_unstable();
    masks.dedup();
    masks
}

/// Keep only the inclusion-minimal masks (an antichain), discarding any mask
/// that is a superset of another; duplicates collapse since a mask is a
/// (non-strict) superset of its equal.
fn keep_minimal(mut masks: Vec<u64>) -> Vec<u64> {
    masks.sort_by_key(|m| m.count_ones());
    masks.into_iter().fold(Vec::new(), |mut minimal, cand| {
        if minimal.iter().all(|&m| m & cand != m) {
            minimal.push(cand);
        }
        minimal
    })
}

/// Minimal factor covers as bitmasks over the factor universe: the sets of
/// factors that hit every polynomial and from which no proper subset still hits
/// every polynomial (the minimal transversals of the group hypergraph). Computed
/// by Berge multiplication --- fold the groups in, extending each partial
/// transversal that misses a group by every factor of that group, then prune to
/// the minimal elements. An empty group yields no cover.
///
/// Returns `None` when the distinct-factor count exceeds the 63-bit mask limit.
pub fn minimal_covers(index_groups: &[Vec<usize>], n_factors: usize) -> Option<Vec<u64>> {
    if n_factors >= 64 {
        return None;
    }
    let covers = group_masks(index_groups)
        .into_iter()
        .fold(vec![0u64], |transversals, group| {
            let expanded = transversals
                .into_iter()
                .flat_map(|t| {
                    if t & group != 0 {
                        vec![t]
                    } else {
                        (0..n_factors)
                            .filter(|&v| group & (1u64 << v) != 0)
                            .map(|v| t | (1u64 << v))
                            .collect()
                    }
                })
                .collect();
            keep_minimal(expanded)
        });
    Some(covers)
}

/// Decode a cover bitmask back into its factors.
pub fn decode_cover(factors: &[PolyExtLex], mask: u64) -> Vec<PolyExtLex> {
    (0..factors.len())
        .filter(|&i| mask & (1u64 << i) != 0)
        .map(|i| factors[i].clone())
        .collect()
}

/// Branch-local covers in easiest-first order: fewer selected factors first,
/// with the numeric mask as a deterministic tie-breaker.
pub fn ordered_cover_masks(covers: &[u64]) -> Vec<u64> {
    let mut ordered = covers.to_vec();
    ordered.sort_unstable_by_key(|&mask| (mask.count_ones(), mask));
    ordered
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Collect a cover list into a sorted vec of sorted factor-index vectors.
    fn as_sets(masks: Vec<u64>) -> Vec<Vec<usize>> {
        let mut sets: Vec<Vec<usize>> = masks
            .into_iter()
            .map(|m| (0..64).filter(|&i| m & (1u64 << i) != 0).collect())
            .collect();
        sets.sort();
        sets
    }

    #[test]
    fn minimal_covers_disjoint_groups() {
        // Groups {0,1} and {2}: every cover must take 2 and one of {0,1}.
        let covers = minimal_covers(&[vec![0, 1], vec![2]], 3).unwrap();
        assert_eq!(as_sets(covers), vec![vec![0, 2], vec![1, 2]]);
    }

    #[test]
    fn minimal_covers_shared_factor() {
        // Groups {0,1} and {1,2} share factor 1; minimal transversals are
        // {1} (hits both) and {0,2}; {0,1} is dropped as a superset of {1}.
        let covers = minimal_covers(&[vec![0, 1], vec![1, 2]], 3).unwrap();
        assert_eq!(as_sets(covers), vec![vec![0, 2], vec![1]]);
    }

    #[test]
    fn minimal_covers_empty_group_has_none() {
        // A group with no factors can never be hit ⇒ no cover exists.
        assert_eq!(
            minimal_covers(&[vec![0], vec![]], 1).unwrap(),
            Vec::<u64>::new()
        );
    }

    #[test]
    fn keep_minimal_drops_supersets_and_dups() {
        // {0} subsumes {0,1}; the duplicate {2} collapses.
        let kept = keep_minimal(vec![0b001, 0b011, 0b100, 0b100]);
        assert_eq!(kept, vec![0b001, 0b100]);
    }

    #[test]
    fn cover_masks_check_fewest_factors_first() {
        assert_eq!(
            ordered_cover_masks(&[0b110, 0b001, 0b010, 0b101]),
            vec![0b001, 0b010, 0b101, 0b110]
        );
    }
}
