//! Stable variable index aligned with sorted identifier order from the Mathematica loader.

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Ord, PartialOrd)]
pub struct VarId(pub u32);
