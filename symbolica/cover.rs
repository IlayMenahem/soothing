//! Minimal factor covers and deterministic subset enumeration over bitsets,
//! with no limit on the number of factors.

use std::cmp::Ordering;
use std::collections::HashMap;

use crate::factor_prep::PolyExtLex;

/// A finite set of indices, such as factor or variable positions, stored as
/// a growable bitmask. Sets are ordered as their bitmasks read as binary
/// numbers, so sets of indices below 64 order as their `u64` masks.
///
/// # Examples
///
/// ```
/// use polynomial_cdcl::cover::BitSet;
///
/// let set = BitSet::from_indices([0, 70]);
/// assert!(set.contains(70));
/// assert_eq!(set.len(), 2);
/// assert_eq!(set.iter().collect::<Vec<_>>(), vec![0, 70]);
/// ```
#[derive(Clone, Debug, Default, Eq, Hash, PartialEq)]
pub struct BitSet {
    /// Little-endian 64-bit words, without trailing zero words.
    words: Vec<u64>,
}

impl BitSet {
    fn from_words(mut words: Vec<u64>) -> Self {
        while words.last() == Some(&0) {
            words.pop();
        }
        Self { words }
    }

    /// The set of `indices`.
    ///
    /// # Examples
    ///
    /// ```
    /// use polynomial_cdcl::cover::BitSet;
    ///
    /// assert_eq!(BitSet::from_indices([3, 3, 1]), BitSet::from_indices([1, 3]));
    /// ```
    pub fn from_indices(indices: impl IntoIterator<Item = usize>) -> Self {
        Self::from_words(indices.into_iter().fold(Vec::new(), |mut words, index| {
            let (word, bit) = (index / 64, index % 64);
            if words.len() <= word {
                words.resize(word + 1, 0);
            }
            words[word] |= 1u64 << bit;
            words
        }))
    }

    /// Whether `index` is in the set.
    ///
    /// # Examples
    ///
    /// ```
    /// use polynomial_cdcl::cover::BitSet;
    ///
    /// assert!(!BitSet::from_indices([1]).contains(65));
    /// ```
    #[must_use]
    pub fn contains(&self, index: usize) -> bool {
        self.words
            .get(index / 64)
            .is_some_and(|word| word & (1u64 << (index % 64)) != 0)
    }

    /// The number of indices in the set.
    ///
    /// # Examples
    ///
    /// ```
    /// use polynomial_cdcl::cover::BitSet;
    ///
    /// assert_eq!(BitSet::from_indices([2, 200]).len(), 2);
    /// ```
    #[must_use]
    pub fn len(&self) -> usize {
        self.words
            .iter()
            .map(|word| word.count_ones() as usize)
            .sum()
    }

    /// Whether the set has no index.
    ///
    /// # Examples
    ///
    /// ```
    /// use polynomial_cdcl::cover::BitSet;
    ///
    /// assert!(BitSet::default().is_empty());
    /// ```
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.words.is_empty()
    }

    /// Whether the two sets share an index.
    ///
    /// # Examples
    ///
    /// ```
    /// use polynomial_cdcl::cover::BitSet;
    ///
    /// assert!(BitSet::from_indices([1, 90]).intersects(&BitSet::from_indices([90])));
    /// ```
    #[must_use]
    pub fn intersects(&self, other: &Self) -> bool {
        self.words
            .iter()
            .zip(&other.words)
            .any(|(left, right)| left & right != 0)
    }

    /// Whether every index of `self` is in `other`.
    ///
    /// # Examples
    ///
    /// ```
    /// use polynomial_cdcl::cover::BitSet;
    ///
    /// assert!(BitSet::from_indices([90]).is_subset(&BitSet::from_indices([1, 90])));
    /// ```
    #[must_use]
    pub fn is_subset(&self, other: &Self) -> bool {
        self.words.len() <= other.words.len()
            && self
                .words
                .iter()
                .zip(&other.words)
                .all(|(left, right)| left & !right == 0)
    }

    /// The indices in either set.
    ///
    /// # Examples
    ///
    /// ```
    /// use polynomial_cdcl::cover::BitSet;
    ///
    /// let union = BitSet::from_indices([1]).union(&BitSet::from_indices([100]));
    /// assert_eq!(union, BitSet::from_indices([1, 100]));
    /// ```
    #[must_use]
    pub fn union(&self, other: &Self) -> Self {
        let (longer, shorter) = if self.words.len() >= other.words.len() {
            (self, other)
        } else {
            (other, self)
        };
        Self::from_words(
            longer
                .words
                .iter()
                .enumerate()
                .map(|(index, word)| word | shorter.words.get(index).copied().unwrap_or(0))
                .collect(),
        )
    }

    /// The indices of `self` that are not in `other`.
    ///
    /// # Examples
    ///
    /// ```
    /// use polynomial_cdcl::cover::BitSet;
    ///
    /// let difference = BitSet::from_indices([1, 100]).difference(&BitSet::from_indices([1]));
    /// assert_eq!(difference, BitSet::from_indices([100]));
    /// ```
    #[must_use]
    pub fn difference(&self, other: &Self) -> Self {
        Self::from_words(
            self.words
                .iter()
                .enumerate()
                .map(|(index, word)| word & !other.words.get(index).copied().unwrap_or(0))
                .collect(),
        )
    }

    /// The indices in increasing order.
    ///
    /// # Examples
    ///
    /// ```
    /// use polynomial_cdcl::cover::BitSet;
    ///
    /// assert_eq!(BitSet::from_indices([64, 0]).iter().collect::<Vec<_>>(), vec![0, 64]);
    /// ```
    pub fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.words
            .iter()
            .enumerate()
            .flat_map(|(word_index, &word)| {
                (0..64)
                    .filter(move |bit| word & (1u64 << bit) != 0)
                    .map(move |bit| word_index * 64 + bit)
            })
    }
}

impl Ord for BitSet {
    fn cmp(&self, other: &Self) -> Ordering {
        self.words
            .len()
            .cmp(&other.words.len())
            .then_with(|| self.words.iter().rev().cmp(other.words.iter().rev()))
    }
}

impl PartialOrd for BitSet {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

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

/// Each factor group as a bitset over the factor universe, deduplicated.
fn group_masks(index_groups: &[Vec<usize>]) -> Vec<BitSet> {
    let mut masks: Vec<BitSet> = index_groups
        .iter()
        .map(|ids| BitSet::from_indices(ids.iter().copied()))
        .collect();
    masks.sort_unstable();
    masks.dedup();
    masks
}

/// Keep only the inclusion-minimal sets (an antichain), discarding any set
/// that is a superset of another; duplicates collapse since a set is a
/// (non-strict) superset of its equal.
fn keep_minimal(mut masks: Vec<BitSet>) -> Vec<BitSet> {
    masks.sort_by_key(BitSet::len);
    masks
        .into_iter()
        .fold(Vec::new(), |mut minimal, candidate| {
            if minimal.iter().all(|kept| !kept.is_subset(&candidate)) {
                minimal.push(candidate);
            }
            minimal
        })
}

/// Minimal factor covers as bitsets over the factor universe: the sets of
/// factors that hit every polynomial and from which no proper subset still hits
/// every polynomial (the minimal transversals of the group hypergraph). Computed
/// by Berge multiplication --- fold the groups in, extending each partial
/// transversal that misses a group by every factor of that group, then prune to
/// the minimal elements. An empty group yields no cover.
///
/// # Examples
///
/// ```
/// use polynomial_cdcl::cover::{BitSet, minimal_covers};
///
/// let covers = minimal_covers(&[vec![0, 1], vec![1, 70]]);
/// assert_eq!(covers, vec![BitSet::from_indices([1]), BitSet::from_indices([0, 70])]);
/// ```
pub fn minimal_covers(index_groups: &[Vec<usize>]) -> Vec<BitSet> {
    group_masks(index_groups)
        .into_iter()
        .fold(vec![BitSet::default()], |transversals, group| {
            let expanded = transversals
                .into_iter()
                .flat_map(|transversal| {
                    if transversal.intersects(&group) {
                        vec![transversal]
                    } else {
                        group
                            .iter()
                            .map(|factor| transversal.union(&BitSet::from_indices([factor])))
                            .collect()
                    }
                })
                .collect();
            keep_minimal(expanded)
        })
}

/// Decode a cover bitset back into its factors.
///
/// # Examples
///
/// ```no_run
/// use polynomial_cdcl::PolyExtLex;
/// use polynomial_cdcl::cover::{BitSet, decode_cover};
///
/// # fn run(factors: &[PolyExtLex]) {
/// let chosen = decode_cover(factors, &BitSet::from_indices([0]));
/// assert_eq!(chosen.len(), 1);
/// # }
/// ```
pub fn decode_cover(factors: &[PolyExtLex], mask: &BitSet) -> Vec<PolyExtLex> {
    mask.iter().map(|index| factors[index].clone()).collect()
}

/// Branch-local covers in easiest-first order: fewer selected factors first,
/// with the bitset order as a deterministic tie-breaker.
///
/// # Examples
///
/// ```
/// use polynomial_cdcl::cover::{BitSet, ordered_cover_masks};
///
/// let ordered = ordered_cover_masks(&[BitSet::from_indices([0, 1]), BitSet::from_indices([2])]);
/// assert_eq!(ordered[0], BitSet::from_indices([2]));
/// ```
pub fn ordered_cover_masks(covers: &[BitSet]) -> Vec<BitSet> {
    let mut ordered = covers.to_vec();
    ordered
        .sort_unstable_by(|left, right| left.len().cmp(&right.len()).then_with(|| left.cmp(right)));
    ordered
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Collect a cover list into a sorted vec of sorted factor-index vectors.
    fn as_sets(masks: Vec<BitSet>) -> Vec<Vec<usize>> {
        let mut sets: Vec<Vec<usize>> = masks.iter().map(|m| m.iter().collect()).collect();
        sets.sort();
        sets
    }

    fn set(indices: &[usize]) -> BitSet {
        BitSet::from_indices(indices.iter().copied())
    }

    #[test]
    fn minimal_covers_disjoint_groups() {
        // Groups {0,1} and {2}: every cover must take 2 and one of {0,1}.
        let covers = minimal_covers(&[vec![0, 1], vec![2]]);
        assert_eq!(as_sets(covers), vec![vec![0, 2], vec![1, 2]]);
    }

    #[test]
    fn minimal_covers_shared_factor() {
        // Groups {0,1} and {1,2} share factor 1; minimal transversals are
        // {1} (hits both) and {0,2}; {0,1} is dropped as a superset of {1}.
        let covers = minimal_covers(&[vec![0, 1], vec![1, 2]]);
        assert_eq!(as_sets(covers), vec![vec![0, 2], vec![1]]);
    }

    #[test]
    fn minimal_covers_empty_group_has_none() {
        // A group with no factors can never be hit ⇒ no cover exists.
        assert_eq!(minimal_covers(&[vec![0], vec![]]), Vec::<BitSet>::new());
    }

    #[test]
    fn minimal_covers_use_more_than_64_factors() {
        // 70 singleton groups force one cover with every factor; the last
        // group shares factor 69 with a group of factor 130.
        let groups: Vec<Vec<usize>> = (0..70).map(|i| vec![i]).chain([vec![69, 130]]).collect();
        let covers = minimal_covers(&groups);
        assert_eq!(as_sets(covers), vec![(0..70).collect::<Vec<_>>()]);
        let covers = minimal_covers(&[vec![0, 130], vec![65, 130]]);
        assert_eq!(as_sets(covers), vec![vec![0, 65], vec![130]]);
    }

    #[test]
    fn keep_minimal_drops_supersets_and_dups() {
        // {0} subsumes {0,1}; the duplicate {2} collapses.
        let kept = keep_minimal(vec![set(&[0]), set(&[0, 1]), set(&[2]), set(&[2])]);
        assert_eq!(kept, vec![set(&[0]), set(&[2])]);
    }

    #[test]
    fn cover_masks_check_fewest_factors_first() {
        assert_eq!(
            ordered_cover_masks(&[set(&[1, 2]), set(&[0]), set(&[1]), set(&[0, 2])]),
            vec![set(&[0]), set(&[1]), set(&[0, 2]), set(&[1, 2])]
        );
    }

    #[test]
    fn bitsets_order_as_binary_numbers() {
        let masks: [u64; 5] = [0b0, 0b1, 0b10, 0b11, 0b110];
        let sets: Vec<BitSet> = masks
            .iter()
            .map(|&mask| BitSet::from_indices((0..64).filter(|bit| mask & (1u64 << bit) != 0)))
            .collect();
        assert!(sets.windows(2).all(|pair| pair[0] < pair[1]));
        assert!(set(&[63]) < set(&[64]));
        assert!(set(&[0, 64]) > set(&[64]));
    }
}
