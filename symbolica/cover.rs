//! Minimal factor covers: hitting-set enumeration over bitmasks, and the
//! `WorkItem`/`Core` types the CDCL search prunes and learns against.

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

/// Single-bit masks for every set bit of `mask`, least significant first.
pub fn set_bits(mask: u64) -> impl Iterator<Item = u64> {
    (0..64).map(|i| 1u64 << i).filter(move |b| mask & b != 0)
}

/// Decode a cover bitmask back into its factors.
pub fn decode_cover(factors: &[PolyExtLex], mask: u64) -> Vec<PolyExtLex> {
    (0..factors.len())
        .filter(|&i| mask & (1u64 << i) != 0)
        .map(|i| factors[i].clone())
        .collect()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WorkItem {
    pub cover_mask: u64,
    pub zero_mask: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Core {
    pub cover_mask: u64,
    pub zero_mask: u64,
}

/// Masks over the `n` low bits with exactly `k` set bits, ascending (Gosper's
/// hack). `k == 0` yields the single empty mask; `k > n` yields nothing.
fn masks_with_popcount(n: usize, k: usize) -> impl Iterator<Item = u64> {
    let limit = 1u64 << n;
    let mut next = (k <= n).then(|| (1u64 << k) - 1);
    std::iter::from_fn(move || {
        let cur = next?;
        next = (cur != 0)
            .then(|| {
                let c = cur & cur.wrapping_neg();
                let r = cur + c;
                (((r ^ cur) >> 2) / c) | r
            })
            .filter(|&m| m < limit);
        Some(cur)
    })
}

/// Work items streamed lazily in difficulty order: items with more zeroed
/// variables first (sparser is easier), then covers with fewer factors first
/// (more factors is harder); ties broken by cover mask then zero mask. Never
/// materializes the 2^sparse_count expansion. For `sparse_count == 0` this
/// yields each cover once with an empty zero mask.
pub fn ordered_work_items(
    base_covers: &[u64],
    sparse_count: usize,
) -> impl Iterator<Item = WorkItem> {
    let mut covers = base_covers.to_vec();
    covers.sort_unstable_by_key(|&m| (m.count_ones(), m));
    (0..=sparse_count).rev().flat_map(move |k| {
        covers.clone().into_iter().flat_map(move |cover_mask| {
            masks_with_popcount(sparse_count, k).map(move |zero_mask| WorkItem {
                cover_mask,
                zero_mask,
            })
        })
    })
}

pub fn core_subsumes(core: Core, item: WorkItem) -> bool {
    core.cover_mask & item.cover_mask == core.cover_mask
        && core.zero_mask & item.zero_mask == core.zero_mask
}

pub fn satisfiable_cover_contains(solution: WorkItem, item: WorkItem) -> bool {
    solution.cover_mask & item.cover_mask == item.cover_mask
        && solution.zero_mask & item.zero_mask == item.zero_mask
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
    fn masks_with_popcount_enumerates_combinations() {
        assert_eq!(
            masks_with_popcount(3, 2).collect::<Vec<_>>(),
            vec![0b011, 0b101, 0b110]
        );
        assert_eq!(masks_with_popcount(2, 0).collect::<Vec<_>>(), vec![0]);
        assert_eq!(masks_with_popcount(2, 3).count(), 0);
    }

    #[test]
    fn sparse_expansion_no_mask_keeps_base_covers() {
        assert_eq!(
            ordered_work_items(&[0b001, 0b010], 0).collect::<Vec<_>>(),
            vec![
                WorkItem {
                    cover_mask: 0b001,
                    zero_mask: 0
                },
                WorkItem {
                    cover_mask: 0b010,
                    zero_mask: 0
                }
            ]
        );
    }

    #[test]
    fn work_item_order_checks_more_zeroes_first_for_same_cover() {
        assert_eq!(
            ordered_work_items(&[0b001], 2).collect::<Vec<_>>(),
            vec![
                WorkItem {
                    cover_mask: 0b001,
                    zero_mask: 0b011
                },
                WorkItem {
                    cover_mask: 0b001,
                    zero_mask: 0b001
                },
                WorkItem {
                    cover_mask: 0b001,
                    zero_mask: 0b010
                },
                WorkItem {
                    cover_mask: 0b001,
                    zero_mask: 0
                }
            ]
        );
    }

    #[test]
    fn work_item_order_checks_sparsest_first_then_fewest_factors() {
        assert_eq!(
            ordered_work_items(&[0b011, 0b100], 1).collect::<Vec<_>>(),
            vec![
                WorkItem {
                    cover_mask: 0b100,
                    zero_mask: 0b1
                },
                WorkItem {
                    cover_mask: 0b011,
                    zero_mask: 0b1
                },
                WorkItem {
                    cover_mask: 0b100,
                    zero_mask: 0
                },
                WorkItem {
                    cover_mask: 0b011,
                    zero_mask: 0
                }
            ]
        );
    }

    #[test]
    fn core_subsumes_zero_supersets() {
        let core = Core {
            cover_mask: 0b001,
            zero_mask: 0b001,
        };

        assert!(core_subsumes(
            core,
            WorkItem {
                cover_mask: 0b101,
                zero_mask: 0b011
            }
        ));
    }

    #[test]
    fn core_subsumes_requires_cover_and_zero_bits() {
        let core = Core {
            cover_mask: 0b101,
            zero_mask: 0b011,
        };

        assert!(!core_subsumes(
            core,
            WorkItem {
                cover_mask: 0b001,
                zero_mask: 0b011
            }
        ));
        assert!(!core_subsumes(
            core,
            WorkItem {
                cover_mask: 0b101,
                zero_mask: 0b001
            }
        ));
    }

    #[test]
    fn satisfiable_cover_contains_factor_and_zero_subsets() {
        let solution = WorkItem {
            cover_mask: 0b101,
            zero_mask: 0b011,
        };

        assert!(satisfiable_cover_contains(
            solution,
            WorkItem {
                cover_mask: 0b001,
                zero_mask: 0b001
            }
        ));
    }

    #[test]
    fn satisfiable_cover_contains_requires_item_to_be_subset() {
        let solution = WorkItem {
            cover_mask: 0b001,
            zero_mask: 0b001,
        };

        assert!(!satisfiable_cover_contains(
            solution,
            WorkItem {
                cover_mask: 0b101,
                zero_mask: 0b001
            }
        ));
        assert!(!satisfiable_cover_contains(
            solution,
            WorkItem {
                cover_mask: 0b001,
                zero_mask: 0b011
            }
        ));
    }
}
