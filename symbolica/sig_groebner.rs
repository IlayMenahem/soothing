//! Signature-based Gröbner basis algorithms: F5 (Faugère 2002) and GVW
//! (Gao–Volny–Wang 2010), generic over the coefficient field and monomial
//! order.
//!
//! Each working polynomial carries a `Signature` `(i, m)` where `i` is the
//! index of the input generator it originated from and `m` is a monomial.
//! Signatures are ordered position-over-term (POT): higher `i` first, then
//! larger `m` in the underlying monomial order `O`. The algorithms process
//! pairs in ascending signature, skip pairs ruled out by signature criteria,
//! and finish with a Symbolica `reduce_basis()` interreduction so the
//! returned list is a reduced Gröbner basis in `O`.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::marker::PhantomData;
use std::sync::Arc;

use symbolica::atom::AtomCore;
use symbolica::coefficient::ConvertToRing;
use symbolica::domains::{EuclideanDomain, Field};
use symbolica::poly::groebner::{Echelonize, GroebnerBasis};
use symbolica::poly::polynomial::MultivariatePolynomial;
use symbolica::poly::{MonomialOrder, PolyVariable};

use crate::monom::{monom_divides, monom_lcm, monom_quot};
use crate::system::PolynomialSystem;

/// Position-over-term (POT) signature `(index, monomial)` carrying the
/// monomial order `O` only at the type level (no runtime cost).
pub struct Signature<O: MonomialOrder> {
    pub index: usize,
    pub monomial: Vec<u16>,
    _phantom: PhantomData<fn() -> O>,
}

impl<O: MonomialOrder> Signature<O> {
    fn new(index: usize, monomial: Vec<u16>) -> Self {
        Self {
            index,
            monomial,
            _phantom: PhantomData,
        }
    }

    fn unit(index: usize, nvars: usize) -> Self {
        Self::new(index, vec![0; nvars])
    }

    fn shifted(&self, by: &[u16]) -> Self {
        Self::new(
            self.index,
            self.monomial.iter().zip(by).map(|(a, b)| a + b).collect(),
        )
    }
}

impl<O: MonomialOrder> Clone for Signature<O> {
    fn clone(&self) -> Self {
        Self::new(self.index, self.monomial.clone())
    }
}

impl<O: MonomialOrder> std::fmt::Debug for Signature<O> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Signature(index={}, monomial={:?})",
            self.index, self.monomial
        )
    }
}

impl<O: MonomialOrder> PartialEq for Signature<O> {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index && self.monomial == other.monomial
    }
}

impl<O: MonomialOrder> Eq for Signature<O> {}

impl<O: MonomialOrder> Ord for Signature<O> {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.index.cmp(&other.index) {
            Ordering::Equal => O::cmp::<u16>(&self.monomial, &other.monomial),
            ord => ord,
        }
    }
}

impl<O: MonomialOrder> PartialOrd for Signature<O> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

struct LabeledPoly<F: Field, O: MonomialOrder> {
    signature: Signature<O>,
    poly: MultivariatePolynomial<F, u16, O>,
}

impl<F: Field, O: MonomialOrder> Clone for LabeledPoly<F, O> {
    fn clone(&self) -> Self {
        Self {
            signature: self.signature.clone(),
            poly: self.poly.clone(),
        }
    }
}

struct Pair<F: Field, O: MonomialOrder> {
    sig: Signature<O>,
    poly: MultivariatePolynomial<F, u16, O>,
}

impl<F: Field, O: MonomialOrder> PartialEq for Pair<F, O> {
    fn eq(&self, other: &Self) -> bool {
        self.sig == other.sig
    }
}
impl<F: Field, O: MonomialOrder> Eq for Pair<F, O> {}
impl<F: Field, O: MonomialOrder> Ord for Pair<F, O> {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap: smaller signature has higher priority.
        other.sig.cmp(&self.sig)
    }
}
impl<F: Field, O: MonomialOrder> PartialOrd for Pair<F, O> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn initial_basis<F: Field, O: MonomialOrder>(
    polys: Vec<MultivariatePolynomial<F, u16, O>>,
) -> (Vec<LabeledPoly<F, O>>, usize) {
    let mut nonzero: Vec<MultivariatePolynomial<F, u16, O>> =
        polys.into_iter().filter(|p| !p.is_zero()).collect();
    MultivariatePolynomial::unify_variables_list(&mut nonzero);
    let nvars = nonzero.first().map(|p| p.nvars()).unwrap_or(0);
    let labeled = nonzero
        .into_iter()
        .enumerate()
        .map(|(i, poly)| LabeledPoly {
            signature: Signature::unit(i, nvars),
            poly,
        })
        .collect();
    (labeled, nvars)
}

/// The polynomial to subtract from `p` to eliminate its leading term, using one
/// basis element `g` whose cofactor monomial `u = lm(p) / lm(g)` produces a
/// signature `u · sig(g)` strictly less than `sig`. `None` when no
/// signature-safe reducer applies.
fn sig_top_reduce_once<F: Field, O: MonomialOrder>(
    p: &MultivariatePolynomial<F, u16, O>,
    sig: &Signature<O>,
    basis: &[LabeledPoly<F, O>],
) -> Option<MultivariatePolynomial<F, u16, O>> {
    if p.is_zero() {
        return None;
    }
    let lm_p = p.max_exp().to_vec();
    basis.iter().find_map(|g| {
        let lm_g = g.poly.max_exp();
        if !monom_divides(lm_g, &lm_p) {
            return None;
        }
        let u = monom_quot(&lm_p, lm_g);
        let new_sig = g.signature.shifted(&u);
        (new_sig.cmp(sig) == Ordering::Less).then(|| {
            let ratio = p.ring().div(p.max_coeff(), g.poly.max_coeff());
            g.poly.clone().mul_exp(&u).mul_coeff(ratio)
        })
    })
}

fn sig_top_reduce<F: Field, O: MonomialOrder>(
    mut p: MultivariatePolynomial<F, u16, O>,
    sig: &Signature<O>,
    basis: &[LabeledPoly<F, O>],
) -> MultivariatePolynomial<F, u16, O> {
    while let Some(to_sub) = sig_top_reduce_once(&p, sig, basis) {
        p = p - to_sub;
        if p.is_zero() {
            return p;
        }
    }
    p
}

/// Build the J-pair between a new element `(sig_new, p_new)` and an existing
/// basis entry `g`. Returns `None` if both candidate signatures collide.
fn jpair<F: Field, O: MonomialOrder>(
    sig_new: &Signature<O>,
    p_new: &MultivariatePolynomial<F, u16, O>,
    g: &LabeledPoly<F, O>,
) -> Option<Pair<F, O>> {
    let lcm = monom_lcm(p_new.max_exp(), g.poly.max_exp());
    let u_new = monom_quot(&lcm, p_new.max_exp());
    let u_g = monom_quot(&lcm, g.poly.max_exp());
    let sig_from_new = sig_new.shifted(&u_new);
    let sig_from_g = g.signature.shifted(&u_g);
    match sig_from_new.cmp(&sig_from_g) {
        Ordering::Greater => Some(Pair {
            sig: sig_from_new,
            poly: p_new.clone().mul_exp(&u_new),
        }),
        Ordering::Less => Some(Pair {
            sig: sig_from_g,
            poly: g.poly.clone().mul_exp(&u_g),
        }),
        Ordering::Equal => None,
    }
}

fn finalize<F: Field + Echelonize, O: MonomialOrder>(
    basis: Vec<LabeledPoly<F, O>>,
) -> GroebnerBasis<F, u16, O> {
    let system: Vec<MultivariatePolynomial<F, u16, O>> = basis
        .into_iter()
        .map(|lp| lp.poly)
        .filter(|p| !p.is_zero())
        .collect();
    GroebnerBasis {
        system,
        print_stats: false,
    }
    .reduce_basis()
}

/// Rewrite criterion: skip a pair whose signature exactly matches that of an
/// existing basis element (a duplicate signature carries no new information).
///
/// This is intentionally *exact*, not divisibility-based. The stronger
/// divisibility ("rewritable") criterion of GVW/F5 is only sound alongside full
/// regular reduction with a latest-rewriter discipline; with the plain
/// signature-safe top reduction used here it over-prunes and drops J-pairs that
/// are still needed, yielding a non-Gröbner basis.
fn is_rewritten<F: Field, O: MonomialOrder>(
    sig: &Signature<O>,
    basis: &[LabeledPoly<F, O>],
) -> bool {
    basis.iter().any(|g| g.signature == *sig)
}

/// Top-reduce `poly` (carrying signature `sig`) against `basis`. If it reduces
/// to zero, returns its signature (a syzygy). Otherwise spawns the J-pairs
/// against every basis element onto `queue` and appends the reduced polynomial
/// to `basis`.
fn reduce_and_extend<F: Field, O: MonomialOrder>(
    sig: Signature<O>,
    poly: MultivariatePolynomial<F, u16, O>,
    basis: &mut Vec<LabeledPoly<F, O>>,
    queue: &mut BinaryHeap<Pair<F, O>>,
) -> Option<Signature<O>> {
    let reduced = sig_top_reduce(poly, &sig, basis);
    if reduced.is_zero() {
        return Some(sig);
    }
    for g in basis.iter() {
        if let Some(p) = jpair(&sig, &reduced, g) {
            queue.push(p);
        }
    }
    basis.push(LabeledPoly {
        signature: sig,
        poly: reduced,
    });
    None
}

/// Gao–Volny–Wang signature-based Gröbner basis algorithm.
///
/// Generic over the coefficient field `F` (must support echelonization for
/// the final interreduction) and the monomial order `O`. Maintains a basis
/// of labeled polynomials and a set of known syzygy signatures.
pub fn groebner_gvw<F, O>(ideal: &[MultivariatePolynomial<F, u16, O>]) -> GroebnerBasis<F, u16, O>
where
    F: Field + Echelonize,
    O: MonomialOrder,
{
    let (initial, _nvars) = initial_basis(ideal.to_vec());
    if initial.is_empty() {
        return GroebnerBasis {
            system: Vec::new(),
            print_stats: false,
        };
    }

    let mut basis: Vec<LabeledPoly<F, O>> = Vec::new();
    let mut syzygy_sigs: Vec<Signature<O>> = Vec::new();
    let mut queue: BinaryHeap<Pair<F, O>> = initial
        .into_iter()
        .map(|lp| Pair {
            sig: lp.signature,
            poly: lp.poly,
        })
        .collect();

    while let Some(Pair { sig, poly }) = queue.pop() {
        let is_syzygy = syzygy_sigs
            .iter()
            .any(|h| h.index == sig.index && monom_divides(&h.monomial, &sig.monomial));
        if is_syzygy || is_rewritten(&sig, &basis) {
            continue;
        }
        if let Some(syz) = reduce_and_extend(sig, poly, &mut basis, &mut queue) {
            syzygy_sigs.push(syz);
        }
    }

    finalize(basis)
}

/// Incremental Faugère F5 (POT).
///
/// Processes generators `f_0, f_1, …` one at a time. While extending the
/// basis with `f_i`, applies the F5 criterion: a pair signature `(i, m)`
/// with `m` divisible by the leading monomial of any element whose
/// signature has index `< i` is a principal syzygy and is dropped.
pub fn groebner_f5<F, O>(ideal: &[MultivariatePolynomial<F, u16, O>]) -> GroebnerBasis<F, u16, O>
where
    F: Field + Echelonize,
    O: MonomialOrder,
{
    let (initial, _nvars) = initial_basis(ideal.to_vec());
    if initial.is_empty() {
        return GroebnerBasis {
            system: Vec::new(),
            print_stats: false,
        };
    }

    let mut basis: Vec<LabeledPoly<F, O>> = Vec::new();

    for entry in initial {
        let cutoff = entry.signature.index;
        let lower_lms: Vec<Vec<u16>> = basis
            .iter()
            .filter(|g| g.signature.index < cutoff)
            .map(|g| g.poly.max_exp().to_vec())
            .collect();

        let mut queue: BinaryHeap<Pair<F, O>> = BinaryHeap::new();
        queue.push(Pair {
            sig: entry.signature,
            poly: entry.poly,
        });

        while let Some(Pair { sig, poly }) = queue.pop() {
            let is_principal_syzygy = lower_lms.iter().any(|lm| monom_divides(lm, &sig.monomial));
            if is_principal_syzygy || is_rewritten(&sig, &basis) {
                continue;
            }
            reduce_and_extend(sig, poly, &mut basis, &mut queue);
        }
    }

    finalize(basis)
}

/// Parse each equation in `system` into a polynomial over `field` and reorder
/// to monomial order `O`. The `try_to_polynomial` parser always returns
/// `LexOrder`, so the result is `reorder::<O>()`-ed before being collected.
pub fn ideal_in_order<R, O>(
    system: &PolynomialSystem,
    field: &R,
    var_order: Arc<Vec<PolyVariable>>,
) -> Result<Vec<MultivariatePolynomial<R, u16, O>>, String>
where
    R: EuclideanDomain + ConvertToRing,
    O: MonomialOrder,
{
    system
        .equations
        .iter()
        .map(|eq| {
            eq.expand()
                .try_to_polynomial::<_, u16>(field, Some(var_order.clone()))
                .map_err(|error| error.to_string())
                .map(|p| p.reorder::<O>())
        })
        .collect()
}

/// Convenience: build the ideal from `system` in order `O` over `field` and
/// run GVW.
pub fn groebner_gvw_from_system_with<R, O>(
    system: &PolynomialSystem,
    field: &R,
    var_order: Arc<Vec<PolyVariable>>,
) -> Result<GroebnerBasis<R, u16, O>, String>
where
    R: Field + Echelonize + ConvertToRing,
    O: MonomialOrder,
{
    let ideal = ideal_in_order::<R, O>(system, field, var_order)?;
    Ok(groebner_gvw(&ideal))
}

/// Convenience: build the ideal from `system` in order `O` over `field` and
/// run F5.
pub fn groebner_f5_from_system_with<R, O>(
    system: &PolynomialSystem,
    field: &R,
    var_order: Arc<Vec<PolyVariable>>,
) -> Result<GroebnerBasis<R, u16, O>, String>
where
    R: Field + Echelonize + ConvertToRing,
    O: MonomialOrder,
{
    let ideal = ideal_in_order::<R, O>(system, field, var_order)?;
    Ok(groebner_f5(&ideal))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mathematica::load_mathematica_system_from_str_with_var_names;
    use crate::variable_order;
    use std::sync::Once;
    use symbolica::domains::rational::Q;
    use symbolica::license::LicenseManager;
    use symbolica::poly::GrevLexOrder;
    use symbolica::poly::LexOrder;
    use symbolica::poly::groebner::GroebnerBasis;

    type QPolys = Vec<MultivariatePolynomial<Q, u16, LexOrder>>;
    type QGb = GroebnerBasis<Q, u16, LexOrder>;

    static LICENSE: Once = Once::new();
    fn activate_license() {
        LICENSE.call_once(|| {
            let _ = LicenseManager::set_license_key(crate::SYMBOLICA_LICENSE_KEY);
        });
    }

    fn basis_strings<O: MonomialOrder>(gb: &GroebnerBasis<Q, u16, O>) -> Vec<String> {
        let mut v: Vec<String> = gb.system.iter().map(|p| p.to_string()).collect();
        v.sort();
        v
    }

    fn run_both(src: &str) -> (QGb, QGb, QPolys) {
        activate_license();
        let loaded = load_mathematica_system_from_str_with_var_names(src).expect("load");
        let var_order = variable_order(&loaded.id_to_sym);
        let gens =
            ideal_in_order::<Q, LexOrder>(&loaded.system, &Q, var_order.clone()).expect("ideal");
        let f5 = groebner_f5_from_system_with::<Q, LexOrder>(&loaded.system, &Q, var_order.clone())
            .expect("f5");
        let gvw = groebner_gvw_from_system_with::<Q, LexOrder>(&loaded.system, &Q, var_order)
            .expect("gvw");
        (f5, gvw, gens)
    }

    /// Self-contained correctness check: `basis` is a Gröbner basis of the ideal
    /// generated by `gens`. It must be a Gröbner basis and generate the same
    /// ideal — every generator reduces to zero modulo it (the reverse inclusion
    /// holds by construction, since signature algorithms only emit ideal members).
    fn is_gb_of(
        basis: &[MultivariatePolynomial<Q, u16, LexOrder>],
        gens: &[MultivariatePolynomial<Q, u16, LexOrder>],
    ) -> bool {
        GroebnerBasis::is_groebner_basis(basis) && gens.iter().all(|g| g.reduce(basis).is_zero())
    }

    #[test]
    fn sig_pot_orders_by_index_then_monomial() {
        let a: Signature<LexOrder> = Signature::new(0, vec![2, 0]);
        let b: Signature<LexOrder> = Signature::new(1, vec![0, 0]);
        let c: Signature<LexOrder> = Signature::new(1, vec![1, 0]);
        assert!(a < b);
        assert!(b < c);
    }

    #[test]
    fn sig_with_grevlex_uses_total_degree() {
        // In grevlex with two variables, total-degree wins over lex.
        // monomial [2,0] (degree 2) > [0,1] (degree 1), regardless of lex.
        let a: Signature<GrevLexOrder> = Signature::new(0, vec![2, 0]);
        let b: Signature<GrevLexOrder> = Signature::new(0, vec![0, 1]);
        assert!(a > b);
    }

    #[test]
    fn empty_ideal_returns_empty_basis() {
        let gb: GroebnerBasis<Q, u16, LexOrder> = groebner_gvw::<Q, LexOrder>(&[]);
        assert!(gb.system.is_empty());
        let gb: GroebnerBasis<Q, u16, LexOrder> = groebner_f5::<Q, LexOrder>(&[]);
        assert!(gb.system.is_empty());
    }

    #[test]
    fn gvw_on_linear_system_is_a_groebner_basis() {
        let (_, gvw, gens) = run_both("{x + y - 1 == 0, x - y == 0}");
        assert!(is_gb_of(&gvw.system, &gens));
    }

    #[test]
    fn f5_on_linear_system_is_a_groebner_basis() {
        let (f5, _, gens) = run_both("{x + y - 1 == 0, x - y == 0}");
        assert!(is_gb_of(&f5.system, &gens));
    }

    #[test]
    fn gvw_on_quadratic_system_is_a_groebner_basis() {
        let (_, gvw, gens) = run_both("{x^2 + y^2 - 1 == 0, x + y == 0}");
        assert!(is_gb_of(&gvw.system, &gens));
    }

    #[test]
    fn f5_on_quadratic_system_is_a_groebner_basis() {
        let (f5, _, gens) = run_both("{x^2 + y^2 - 1 == 0, x + y == 0}");
        assert!(is_gb_of(&f5.system, &gens));
    }

    #[test]
    fn f5_and_gvw_agree_on_3var_system() {
        let src = "{x^2 + y + z - 1 == 0, x + y^2 + z - 1 == 0, x + y + z^2 - 1 == 0}";
        let (f5, gvw, _) = run_both(src);
        assert_eq!(basis_strings(&f5), basis_strings(&gvw));
    }

    fn computed_gb(src: &str) -> Vec<MultivariatePolynomial<Q, u16, LexOrder>> {
        activate_license();
        let loaded = load_mathematica_system_from_str_with_var_names(src).expect("load");
        let var_order = variable_order(&loaded.id_to_sym);
        groebner_gvw_from_system_with::<Q, LexOrder>(&loaded.system, &Q, var_order)
            .expect("gvw")
            .system
    }

    #[test]
    fn gvw_output_is_a_groebner_basis_on_3var_system() {
        let basis =
            computed_gb("{x^2 + y + z - 1 == 0, x + y^2 + z - 1 == 0, x + y + z^2 - 1 == 0}");
        assert!(GroebnerBasis::is_groebner_basis(&basis));
    }

    #[test]
    fn f5_output_is_a_groebner_basis_on_3var_system() {
        activate_license();
        let loaded = load_mathematica_system_from_str_with_var_names(
            "{x^2 + y + z - 1 == 0, x + y^2 + z - 1 == 0, x + y + z^2 - 1 == 0}",
        )
        .expect("load");
        let var_order = variable_order(&loaded.id_to_sym);
        let gb =
            groebner_f5_from_system_with::<Q, LexOrder>(&loaded.system, &Q, var_order).expect("f5");
        assert!(GroebnerBasis::is_groebner_basis(&gb.system));
    }

    #[test]
    fn katsura3_f5_and_gvw_agree_and_form_a_gb() {
        let src =
            "{2*x + 2*y + 2*z - 1 == 0, x^2 + 2*y^2 + 2*z^2 - x == 0, 2*x*y + 2*y*z - y == 0}";
        let (f5, gvw, gens) = run_both(src);
        assert_eq!(basis_strings(&f5), basis_strings(&gvw));
        assert!(is_gb_of(&gvw.system, &gens));
    }

    #[test]
    fn grevlex_gvw_output_is_a_groebner_basis() {
        activate_license();
        let loaded = load_mathematica_system_from_str_with_var_names(
            "{x^2 + y + z - 1 == 0, x + y^2 + z - 1 == 0, x + y + z^2 - 1 == 0}",
        )
        .expect("load");
        let var_order = variable_order(&loaded.id_to_sym);
        let gb = groebner_gvw_from_system_with::<Q, GrevLexOrder>(&loaded.system, &Q, var_order)
            .expect("gvw");
        assert!(GroebnerBasis::is_groebner_basis(&gb.system));
    }

    #[test]
    fn grevlex_f5_output_is_a_groebner_basis() {
        activate_license();
        let loaded = load_mathematica_system_from_str_with_var_names(
            "{x^2 + y + z - 1 == 0, x + y^2 + z - 1 == 0, x + y + z^2 - 1 == 0}",
        )
        .expect("load");
        let var_order = variable_order(&loaded.id_to_sym);
        let gb = groebner_f5_from_system_with::<Q, GrevLexOrder>(&loaded.system, &Q, var_order)
            .expect("f5");
        assert!(GroebnerBasis::is_groebner_basis(&gb.system));
    }

    #[test]
    fn grevlex_f5_and_gvw_agree_on_katsura3() {
        activate_license();
        let src =
            "{2*x + 2*y + 2*z - 1 == 0, x^2 + 2*y^2 + 2*z^2 - x == 0, 2*x*y + 2*y*z - y == 0}";
        let loaded = load_mathematica_system_from_str_with_var_names(src).expect("load");
        let var_order = variable_order(&loaded.id_to_sym);
        let f5 =
            groebner_f5_from_system_with::<Q, GrevLexOrder>(&loaded.system, &Q, var_order.clone())
                .expect("f5");
        let gvw = groebner_gvw_from_system_with::<Q, GrevLexOrder>(&loaded.system, &Q, var_order)
            .expect("gvw");
        let f5s = basis_strings(&f5);
        let gvws = basis_strings(&gvw);
        assert_eq!(f5s, gvws);
    }
}
