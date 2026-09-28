//! M4GB Gröbner basis algorithm (Makarim–Stevens, ISSAC 2017), generic over
//! the coefficient field and monomial order.
//!
//! M4GB is a Buchberger-style algorithm distinguished by two ideas:
//!
//! 1. **Fully tail-reduced basis** — every basis polynomial has *all* of its
//!    terms (not only the leading one) irreducible with respect to the leading
//!    monomials of the other basis elements.
//! 2. **Persistent reducer table** — a lazily built, cached map from a monomial
//!    to the fully-reduced monic polynomial that reduces it (the "M" table).
//!    Because each cached reducer is itself fully reduced, reducing a
//!    polynomial is essentially one linear pass: every reducible monomial is
//!    rewritten exactly once by its precomputed reducer.
//!
//! Critical pairs are pruned with the Gebauer–Möller criteria and selected with
//! the normal strategy (least lcm degree first). The final basis is run through
//! Symbolica's [`GroebnerBasis::reduce_basis`] so the output is the canonical
//! reduced Gröbner basis.

use std::collections::HashMap;
use std::sync::Arc;

use symbolica::coefficient::ConvertToRing;
use symbolica::domains::Field;
use symbolica::poly::groebner::{Echelonize, GroebnerBasis};
use symbolica::poly::polynomial::MultivariatePolynomial;
use symbolica::poly::{MonomialOrder, PolyVariable};

use crate::monom::{monom_divides, monom_lcm, monom_quot};
use crate::sig_groebner::ideal_in_order;
use crate::system::PolynomialSystem;

type Poly<F, O> = MultivariatePolynomial<F, u16, O>;

/// A critical pair `(i, j)` together with the lcm of the two leading monomials
/// and its total degree (the normal-selection key).
struct CritPair {
    i: usize,
    j: usize,
    lcm: Vec<u16>,
    deg: u32,
}

/// Working state of the M4GB run.
struct M4gb<F: Field, O: MonomialOrder> {
    /// Basis polynomials, kept monic and fully tail-reduced.
    basis: Vec<Poly<F, O>>,
    /// Leading exponent vectors, index-aligned with `basis`.
    leads: Vec<Vec<u16>>,
    /// The reducer table: monomial → fully-reduced monic reductor with that lead.
    reducers: HashMap<Vec<u16>, Poly<F, O>>,
    /// Pending critical pairs.
    pairs: Vec<CritPair>,
    /// Empty polynomial sharing the variables and ring of the input.
    zero: Poly<F, O>,
}

impl<F: Field, O: MonomialOrder> M4gb<F, O> {
    fn new(zero: Poly<F, O>) -> Self {
        Self {
            basis: Vec::new(),
            leads: Vec::new(),
            reducers: HashMap::new(),
            pairs: Vec::new(),
            zero,
        }
    }

    /// A single-term polynomial `coeff · x^exps`.
    fn monomial(&self, coeff: F::Element, exps: &[u16]) -> Poly<F, O> {
        let mut p = self.zero.clone();
        p.append_monomial(coeff, exps);
        p
    }

    /// `true` iff some basis leading monomial divides `m`.
    fn is_reducible(&self, m: &[u16]) -> bool {
        self.leads.iter().any(|l| monom_divides(l, m))
    }

    /// Whether `t` divides any term of `p` (used to detect stale reducers/tails).
    fn any_term_divisible(p: &Poly<F, O>, t: &[u16]) -> bool {
        (0..p.nterms()).any(|i| monom_divides(t, p.exponents(i)))
    }

    /// The fully-reduced monic reductor whose leading monomial is `m`, computed
    /// lazily and cached. `m` must be reducible by the current basis.
    fn reductor(&mut self, m: &[u16]) -> Poly<F, O> {
        if let Some(r) = self.reducers.get(m) {
            return r.clone();
        }
        let k = self
            .leads
            .iter()
            .position(|l| monom_divides(l, m))
            .expect("reductor called on an irreducible monomial");
        let u = monom_quot(m, &self.leads[k]);
        // `basis[k]` is monic, so `t` is monic with leading monomial exactly `m`.
        let t = self.basis[k].clone().mul_exp(&u);
        let lead = self.monomial(t.max_coeff().clone(), m);
        let tail = t - lead.clone();
        let reduced_tail = self.reduce(tail);
        let r = lead + reduced_tail;
        self.reducers.insert(m.to_vec(), r.clone());
        r
    }

    /// Full reduction of `p` to its normal form modulo the basis.
    ///
    /// Each reducible leading monomial is rewritten once by its reductor; since
    /// reductors are fully reduced, the terms they introduce are irreducible, so
    /// the loop performs the minimal number of rewrites.
    fn reduce(&mut self, mut p: Poly<F, O>) -> Poly<F, O> {
        let mut done = self.zero.clone();
        while !p.is_zero() {
            let m = p.max_exp().to_vec();
            let c = p.max_coeff().clone();
            if self.is_reducible(&m) {
                let r = self.reductor(&m);
                p = p - r.mul_coeff(c);
            } else {
                let lead = self.monomial(c.clone(), &m);
                done.append_monomial(c, &m);
                p = p - lead;
            }
        }
        done
    }

    /// Reduce every term of `p` below its leading term, keeping the lead fixed.
    fn reduce_tail(&mut self, p: Poly<F, O>) -> Poly<F, O> {
        let m = p.max_exp().to_vec();
        let lead = self.monomial(p.max_coeff().clone(), &m);
        let tail = p - lead.clone();
        lead + self.reduce(tail)
    }

    /// The S-polynomial of basis elements `i` and `j` (both monic).
    fn spoly(&self, i: usize, j: usize) -> Poly<F, O> {
        let lcm = monom_lcm(&self.leads[i], &self.leads[j]);
        let ui = monom_quot(&lcm, &self.leads[i]);
        let uj = monom_quot(&lcm, &self.leads[j]);
        self.basis[i].clone().mul_exp(&ui) - self.basis[j].clone().mul_exp(&uj)
    }

    /// Insert a new monic, fully-reduced polynomial `g` into the basis: update
    /// the critical pairs with the Gebauer–Möller criteria, then maintain the
    /// fully-reduced invariant and the reducer cache.
    fn add_to_basis(&mut self, g: Poly<F, O>) {
        let t = g.max_exp().to_vec();
        let k = self.basis.len();

        self.purge_pairs(&t);
        self.spawn_pairs(&t, k);

        self.basis.push(g);
        self.leads.push(t.clone());
        self.update_reducers(&t);
    }

    /// Gebauer–Möller chain + product criteria for the new pairs `(i, k)`.
    fn spawn_pairs(&mut self, t: &[u16], k: usize) {
        let lcms: Vec<Vec<u16>> = (0..k).map(|i| monom_lcm(&self.leads[i], t)).collect();
        let coprime = |i: usize| self.leads[i].iter().zip(t).all(|(a, b)| *a == 0 || *b == 0);
        for i in 0..k {
            if coprime(i) {
                continue;
            }
            // Chain criterion: drop (i, k) if another candidate's lcm divides
            // this one (properly, or equally with a smaller index representative).
            let dominated = (0..k).any(|j| {
                j != i && monom_divides(&lcms[j], &lcms[i]) && (lcms[j] != lcms[i] || j < i)
            });
            if dominated {
                continue;
            }
            let deg = lcms[i].iter().map(|&e| e as u32).sum();
            self.pairs.push(CritPair {
                i,
                j: k,
                lcm: lcms[i].clone(),
                deg,
            });
        }
    }

    /// Gebauer–Möller purge of existing pairs made redundant by new lead `t`.
    fn purge_pairs(&mut self, t: &[u16]) {
        let leads = &self.leads;
        self.pairs.retain(|p| {
            if !monom_divides(t, &p.lcm) {
                return true;
            }
            let li = monom_lcm(&leads[p.i], t);
            let lj = monom_lcm(&leads[p.j], t);
            li == p.lcm || lj == p.lcm
        });
    }

    /// Restore the fully-reduced invariant after adding lead `t`: drop reducer
    /// cache entries that `t` can now reduce, and re-reduce basis tails that
    /// contain a monomial divisible by `t`.
    fn update_reducers(&mut self, t: &[u16]) {
        self.reducers
            .retain(|key, val| !monom_divides(t, key) && !Self::any_term_divisible(val, t));

        for idx in 0..self.basis.len() {
            let lead = self.leads[idx].as_slice();
            let stale = (0..self.basis[idx].nterms()).any(|i| {
                let e = self.basis[idx].exponents(i);
                e != lead && monom_divides(t, e)
            });
            if stale {
                let p = self.basis[idx].clone();
                self.basis[idx] = self.reduce_tail(p);
            }
        }
    }
}

/// Compute a Gröbner basis of `ideal` with the M4GB algorithm.
///
/// The result is the canonical reduced Gröbner basis under the monomial order
/// `O`.
///
/// # Examples
///
/// ```no_run
/// use polynomial_cdcl::groebner_m4gb;
/// use symbolica::poly::polynomial::MultivariatePolynomial;
/// use symbolica::poly::LexOrder;
/// use symbolica::domains::rational::Q;
///
/// # fn run(ideal: &[MultivariatePolynomial<Q, u16, LexOrder>]) {
/// let gb = groebner_m4gb(ideal);
/// assert!(!gb.system.is_empty());
/// # }
/// ```
pub fn groebner_m4gb<F, O>(ideal: &[Poly<F, O>]) -> GroebnerBasis<F, u16, O>
where
    F: Field + Echelonize,
    O: MonomialOrder,
{
    let mut polys: Vec<Poly<F, O>> = ideal.iter().filter(|p| !p.is_zero()).cloned().collect();
    MultivariatePolynomial::unify_variables_list(&mut polys);

    if polys.is_empty() {
        return GroebnerBasis {
            system: Vec::new(),
            print_stats: false,
        };
    }

    let mut state = M4gb::new(polys[0].zero());
    polys.sort_by(|a, b| O::cmp::<u16>(a.max_exp(), b.max_exp()));

    for f in polys {
        reduce_and_insert(&mut state, f);
    }

    while !state.pairs.is_empty() {
        let sel = state
            .pairs
            .iter()
            .enumerate()
            .min_by_key(|(_, p)| p.deg)
            .map(|(i, _)| i)
            .expect("pairs non-empty");
        let pair = state.pairs.swap_remove(sel);
        let s = state.spoly(pair.i, pair.j);
        reduce_and_insert(&mut state, s);
    }

    GroebnerBasis {
        system: state.basis,
        print_stats: false,
    }
    .reduce_basis()
}

/// Reduce `f` and, if non-zero, add its monic normal form to the basis.
fn reduce_and_insert<F: Field, O: MonomialOrder>(state: &mut M4gb<F, O>, f: Poly<F, O>) {
    let r = state.reduce(f);
    if !r.is_zero() {
        state.add_to_basis(r.make_monic());
    }
}

/// Compute an M4GB Gröbner basis of a [`PolynomialSystem`] over the field
/// `field`, ordering variables by `var_order`.
pub fn groebner_m4gb_from_system_with<R, O>(
    system: &PolynomialSystem,
    field: &R,
    var_order: Arc<Vec<PolyVariable>>,
) -> Result<GroebnerBasis<R, u16, O>, String>
where
    R: Field + Echelonize + ConvertToRing,
    O: MonomialOrder,
{
    let ideal = ideal_in_order::<R, O>(system, field, var_order)?;
    Ok(groebner_m4gb(&ideal))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mathematica::load_mathematica_system_from_str_with_var_names;
    use crate::monom::monom_divides;
    use crate::sig_groebner::{groebner_f5_from_system_with, groebner_gvw_from_system_with};
    use crate::variable_order;
    use std::sync::Once;
    use symbolica::LicenseManager;
    use symbolica::domains::rational::Q;
    use symbolica::poly::groebner::GroebnerBasis;
    use symbolica::poly::polynomial::MultivariatePolynomial;
    use symbolica::poly::{GrevLexOrder, LexOrder};

    static LICENSE: Once = Once::new();

    fn activate_license() {
        LICENSE.call_once(|| {
            let _ = LicenseManager::set_license_key(crate::SYMBOLICA_LICENSE_KEY);
        });
    }

    /// A basis of rational polynomials in monomial order `O`.
    type Basis<O> = Vec<MultivariatePolynomial<Q, u16, O>>;

    /// Compute the reduced M4GB, F5 and GVW bases of `src` in order `O`.
    ///
    /// F5 and GVW are independent signature-based solvers already in the crate;
    /// they form the correctness oracle. (Symbolica's own `GroebnerBasis::new`
    /// is *not* used: it can return a non-complete basis, e.g. on `SYMMETRIC3`
    /// in lex order.)
    fn three_bases<O: MonomialOrder>(src: &str) -> (Basis<O>, Basis<O>, Basis<O>) {
        activate_license();
        let loaded = load_mathematica_system_from_str_with_var_names(src).expect("load");
        let vo = variable_order(&loaded.id_to_sym);
        let m4 = groebner_m4gb_from_system_with::<Q, O>(&loaded.system, &Q, vo.clone())
            .expect("m4gb")
            .system;
        let f5 = groebner_f5_from_system_with::<Q, O>(&loaded.system, &Q, vo.clone())
            .expect("f5")
            .system;
        let gvw = groebner_gvw_from_system_with::<Q, O>(&loaded.system, &Q, vo)
            .expect("gvw")
            .system;
        (m4, f5, gvw)
    }

    /// Sorted string form of a basis, for canonical comparison.
    fn strings<O: MonomialOrder>(basis: &[MultivariatePolynomial<Q, u16, O>]) -> Vec<String> {
        let mut v: Vec<String> = basis.iter().map(|p| p.to_string()).collect();
        v.sort();
        v
    }

    /// A basis is reduced iff no term of any element is divisible by the leading
    /// monomial of a *different* element.
    fn is_reduced<O: MonomialOrder>(basis: &[MultivariatePolynomial<Q, u16, O>]) -> bool {
        let leads: Vec<Vec<u16>> = basis.iter().map(|p| p.max_exp().to_vec()).collect();
        basis.iter().enumerate().all(|(a, p)| {
            (0..p.nterms()).all(|ti| {
                let e = p.exponents(ti);
                leads
                    .iter()
                    .enumerate()
                    .all(|(b, l)| a == b || !monom_divides(l, e))
            })
        })
    }

    /// Assert M4GB returns a reduced Gröbner basis matching the independent
    /// F5/GVW oracle (which must itself agree).
    fn assert_reduced_gb<O: MonomialOrder>(src: &str) {
        let (m4, f5, gvw) = three_bases::<O>(src);
        assert!(
            GroebnerBasis::is_groebner_basis(&m4),
            "m4gb output is not a Gröbner basis"
        );
        assert!(is_reduced(&m4), "m4gb output is not reduced");
        assert_eq!(
            strings(&f5),
            strings(&gvw),
            "F5 and GVW disagree (bad oracle)"
        );
        assert_eq!(
            strings(&m4),
            strings(&gvw),
            "m4gb differs from the F5/GVW reduced basis"
        );
    }

    const QUADRATIC: &str = "{x^2 + y^2 - 1 == 0, x + y == 0}";
    const SYMMETRIC3: &str = "{x^2 + y + z - 1 == 0, x + y^2 + z - 1 == 0, x + y + z^2 - 1 == 0}";
    const KATSURA3: &str =
        "{2*x + 2*y + 2*z - 1 == 0, x^2 + 2*y^2 + 2*z^2 - x == 0, 2*x*y + 2*y*z - y == 0}";
    const CYCLIC3: &str = "{x + y + z == 0, x*y + y*z + z*x == 0, x*y*z - 1 == 0}";
    const CYCLIC4: &str = "{x + y + z + w == 0, x*y + y*z + z*w + w*x == 0, \
        x*y*z + y*z*w + z*w*x + w*x*y == 0, x*y*z*w - 1 == 0}";

    #[test]
    fn m4gb_quadratic_lex() {
        assert_reduced_gb::<LexOrder>(QUADRATIC);
    }

    #[test]
    fn m4gb_quadratic_grevlex() {
        assert_reduced_gb::<GrevLexOrder>(QUADRATIC);
    }

    #[test]
    fn m4gb_symmetric3_lex() {
        assert_reduced_gb::<LexOrder>(SYMMETRIC3);
    }

    #[test]
    fn m4gb_symmetric3_grevlex() {
        assert_reduced_gb::<GrevLexOrder>(SYMMETRIC3);
    }

    #[test]
    fn m4gb_katsura3_lex() {
        assert_reduced_gb::<LexOrder>(KATSURA3);
    }

    #[test]
    fn m4gb_katsura3_grevlex() {
        assert_reduced_gb::<GrevLexOrder>(KATSURA3);
    }

    #[test]
    fn m4gb_cyclic3_lex() {
        assert_reduced_gb::<LexOrder>(CYCLIC3);
    }

    #[test]
    fn m4gb_cyclic3_grevlex() {
        assert_reduced_gb::<GrevLexOrder>(CYCLIC3);
    }

    #[test]
    fn m4gb_cyclic4_lex() {
        assert_reduced_gb::<LexOrder>(CYCLIC4);
    }

    #[test]
    fn m4gb_cyclic4_grevlex() {
        assert_reduced_gb::<GrevLexOrder>(CYCLIC4);
    }

    #[test]
    fn m4gb_empty_ideal_is_empty() {
        let ideal: Vec<MultivariatePolynomial<Q, u16, LexOrder>> = Vec::new();
        assert!(groebner_m4gb(&ideal).system.is_empty());
    }
}
