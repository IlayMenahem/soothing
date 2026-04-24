//! MILP-based radical cover as described in `problem statement.md`.
//!
//! Preprocess the input polynomial system (interreduce, square-free factor,
//! drop constants/zero polynomials), then solve an integer program with
//! [`good_lp`]/HiGHS to pick tuples `(f_i, x_i)` that cover every polynomial
//! by a radical-solvable, acyclic assignment.
//!
//! Usage: `milp_cover [path]` — default path: `systems/s_system.txt`.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use good_lp::{
    Expression, ResolutionError, Solution, SolverModel, constraint, default_solver, variable,
    variables,
};

use symbolica::LicenseManager;
use symbolica::atom::Symbol;
use symbolica::domains::algebraic_number::AlgebraicExtension;
use symbolica::domains::rational::Q;
use symbolica::poly::PolyVariable;

use symbolica_elim::{
    LoadedMathematicaSystem, PolyExtLex, default_systems_dir_from_manifest,
    load_mathematica_system_from_path_with_var_names, preprocess,
};

/// A single square-free factor together with the global-variable ids it
/// depends on and their (per-variable) degrees.
struct FactorInfo {
    poly: PolyExtLex,
    /// `(global_var_id, degree)` for every variable occurring in `poly`.
    vars: Vec<(usize, u16)>,
}

/// Problem data assembled from the preprocessed factor groups.
struct ProblemData {
    factors: Vec<FactorInfo>,
    /// For each original (non-zero, non-trivial) polynomial, the indices of
    /// its distinct square-free factors in `factors`.
    f_by_poly: Vec<Vec<usize>>,
    /// Global variable ids that appear in at least one factor.
    v_ids: Vec<usize>,
    /// Inverse of `v_ids`.
    v_idx: HashMap<usize, usize>,
    /// Admissible pairs as `(factor_idx, local_var_idx)` with `1 <= deg <= 4`.
    admissible: Vec<(usize, usize)>,
    /// `deg_x(f)` indexed the same way as `admissible`.
    deg_of_pair: Vec<u16>,
}

fn factor_vars(
    f: &PolyExtLex,
    sym_to_vid: &HashMap<Symbol, symbolica_elim::VarId>,
) -> Vec<(usize, u16)> {
    let mut out = Vec::new();
    for (xi, pv) in f.get_vars_ref().iter().enumerate() {
        if !f.contains(xi) {
            continue;
        }
        let PolyVariable::Symbol(sym) = pv else {
            continue;
        };
        let Some(&vid) = sym_to_vid.get(sym) else {
            continue;
        };
        out.push((vid.0 as usize, f.degree(xi)));
    }
    out
}

fn build_problem(
    factor_groups: Vec<Vec<PolyExtLex>>,
    loaded: &LoadedMathematicaSystem,
) -> ProblemData {
    let mut factor_index: HashMap<PolyExtLex, usize> = HashMap::new();
    let mut factors: Vec<FactorInfo> = Vec::new();
    let mut f_by_poly: Vec<Vec<usize>> = Vec::with_capacity(factor_groups.len());

    for group in factor_groups {
        let mut ids = Vec::with_capacity(group.len());
        for f in group {
            let canon = f.make_monic();
            let idx = *factor_index.entry(canon.clone()).or_insert_with(|| {
                let info = FactorInfo {
                    vars: factor_vars(&canon, &loaded.sym_to_vid),
                    poly: canon,
                };
                factors.push(info);
                factors.len() - 1
            });
            if !ids.contains(&idx) {
                ids.push(idx);
            }
        }
        f_by_poly.push(ids);
    }

    let mut v_set: Vec<usize> = factors
        .iter()
        .flat_map(|fi| fi.vars.iter().map(|(v, _)| *v))
        .collect();
    v_set.sort_unstable();
    v_set.dedup();
    let v_idx: HashMap<usize, usize> = v_set.iter().enumerate().map(|(i, v)| (*v, i)).collect();

    let mut admissible = Vec::new();
    let mut deg_of_pair = Vec::new();
    for (fi, info) in factors.iter().enumerate() {
        for (vid, deg) in &info.vars {
            if (1..=4).contains(deg) {
                admissible.push((fi, v_idx[vid]));
                deg_of_pair.push(*deg);
            }
        }
    }

    ProblemData {
        factors,
        f_by_poly,
        v_ids: v_set,
        v_idx,
        admissible,
        deg_of_pair,
    }
}

fn var_name(loaded: &LoadedMathematicaSystem, global_vid: usize) -> &str {
    &loaded.var_names[global_vid]
}

/// Result of solving the MILP: list of selected `(factor_idx, global_var_id)`
/// pairs in topological order along with the chosen degrees.
#[cfg_attr(test, derive(Debug))]
struct Cover {
    selected: Vec<(usize, usize, u16, f64)>,
    dependent_locals: Vec<usize>,
}

fn solve_milp(data: &ProblemData, minimize_degree: bool) -> Result<Cover, ResolutionError> {
    let n = data.v_ids.len();
    let big_m = (n as f64) + 1.0;

    let mut vars = variables!();
    let s_var: Vec<_> = data
        .admissible
        .iter()
        .map(|_| vars.add(variable().binary()))
        .collect();
    let z_var: Vec<_> = (0..n).map(|_| vars.add(variable().binary())).collect();
    let t_var: Vec<_> = (0..n)
        .map(|_| vars.add(variable().min(0.0).max(n as f64)))
        .collect();

    let mut pairs_by_var: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (k, &(_, vloc)) in data.admissible.iter().enumerate() {
        pairs_by_var[vloc].push(k);
    }

    let objective: Expression = if minimize_degree {
        data.admissible
            .iter()
            .enumerate()
            .map(|(k, _)| f64::from(data.deg_of_pair[k]) * s_var[k])
            .sum()
    } else {
        s_var.iter().map(|v| Expression::from(*v)).sum()
    };

    let mut model = vars.minimise(objective).using(default_solver);

    for (vloc, ks) in pairs_by_var.iter().enumerate() {
        let sum: Expression = ks.iter().map(|&k| Expression::from(s_var[k])).sum();
        model = model.with(constraint!(z_var[vloc] - sum.clone() == 0));
        model = model.with(constraint!(sum <= 1));
    }

    for (j, fs) in data.f_by_poly.iter().enumerate() {
        let sum: Expression = data
            .admissible
            .iter()
            .enumerate()
            .filter(|(_, (fi, _))| fs.contains(fi))
            .map(|(k, _)| Expression::from(s_var[k]))
            .sum();
        if data.admissible.iter().all(|(fi, _)| !fs.contains(fi)) {
            eprintln!(
                "error: polynomial {j} has no admissible factor (no square-free factor with 1 <= deg_x(f) <= 4). The instance is infeasible."
            );
            return Err(ResolutionError::Infeasible);
        }
        model = model.with(constraint!(sum >= 1));
    }

    for (k, &(fi, vloc)) in data.admissible.iter().enumerate() {
        let info = &data.factors[fi];
        for (vid_other, _deg) in &info.vars {
            let Some(&vloc_other) = data.v_idx.get(vid_other) else {
                continue;
            };
            if vloc_other == vloc {
                continue;
            }
            model = model.with(constraint!(
                t_var[vloc_other] + 1.0
                    <= t_var[vloc]
                        + big_m
                            * (2.0
                                - Expression::from(s_var[k])
                                - Expression::from(z_var[vloc_other]))
            ));
        }
    }

    let sol = model.solve()?;

    let mut selected = Vec::new();
    for (k, &(fi, vloc)) in data.admissible.iter().enumerate() {
        if sol.value(s_var[k]) > 0.5 {
            let vid = data.v_ids[vloc];
            selected.push((fi, vid, data.deg_of_pair[k], sol.value(t_var[vloc])));
        }
    }
    selected.sort_by(|a, b| a.3.partial_cmp(&b.3).unwrap_or(std::cmp::Ordering::Equal));

    let dependent_locals: Vec<usize> = (0..n).filter(|&i| sol.value(z_var[i]) > 0.5).collect();

    Ok(Cover {
        selected,
        dependent_locals,
    })
}

fn main() {
    let key = "5568d092#6bb8b4fb#ff936916-65ba-552f-b3c1-bdc64597cd21";
    LicenseManager::set_license_key(key).unwrap();

    let mut args = std::env::args().skip(1);
    let mut path: Option<PathBuf> = None;
    let mut minimize_degree = false;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--objective" => match args.next().as_deref() {
                Some("count") => minimize_degree = false,
                Some("degree") => minimize_degree = true,
                other => {
                    eprintln!("--objective expects 'count' or 'degree' (got {other:?})");
                    std::process::exit(2);
                }
            },
            _ => path = Some(PathBuf::from(a)),
        }
    }
    let path = path.unwrap_or_else(|| default_systems_dir_from_manifest().join("s_system.txt"));

    println!("symbolica-elim — MILP radical cover (good_lp / HiGHS)\n");
    println!("Loading {}", path.display());

    let loaded = match load_mathematica_system_from_path_with_var_names(&path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("load error: {e}");
            std::process::exit(1);
        }
    };
    println!(
        "Parsed: {} equations, {} variables (loader order)",
        loaded.system.equations.len(),
        loaded.var_names.len()
    );

    let field = AlgebraicExtension::new_complex(Q);

    println!("\nPreprocessing (interreduce + square-free factor)");
    let t_pre = Instant::now();
    let pre = preprocess(&loaded, &field);
    println!(
        "  {} non-trivial polynomial(s), elapsed {:?}",
        pre.factor_groups.len(),
        t_pre.elapsed()
    );
    if pre.factor_groups.is_empty() {
        println!("Nothing to cover — all polynomials reduced to zero.");
        return;
    }

    let data = build_problem(pre.factor_groups, &loaded);
    println!(
        "  |F|={}, |V|={}, |A|={}",
        data.factors.len(),
        data.v_ids.len(),
        data.admissible.len()
    );

    println!(
        "\nSolving MILP ({}) with HiGHS ...",
        if minimize_degree {
            "minimize total degree"
        } else {
            "minimize cover size"
        }
    );
    let t_solve = Instant::now();
    let cover = match solve_milp(&data, minimize_degree) {
        Ok(c) => c,
        Err(ResolutionError::Infeasible) => {
            eprintln!("infeasible: no valid cover exists for this system.");
            std::process::exit(3);
        }
        Err(e) => {
            eprintln!("solver error: {e}");
            std::process::exit(4);
        }
    };
    println!("  solved in {:?}", t_solve.elapsed());

    println!("\nCover (in topological order):");
    for (rank, (fi, vid, deg, t)) in cover.selected.iter().enumerate() {
        let info = &data.factors[*fi];
        println!(
            "  [{rank}] t={t:.2}  x = {:<16} deg_x(f) = {deg}\n        f = {}",
            var_name(&loaded, *vid),
            info.poly
        );
    }

    let dep: Vec<&str> = cover
        .dependent_locals
        .iter()
        .map(|&i| var_name(&loaded, data.v_ids[i]))
        .collect();
    let all_used: Vec<&str> = data
        .v_ids
        .iter()
        .map(|&vid| var_name(&loaded, vid))
        .collect();
    let free: Vec<&&str> = all_used.iter().filter(|n| !dep.contains(n)).collect();
    println!("\nDependent variables X = {dep:?}");
    println!("Free parameters    Y = {free:?}");
    println!("(Variables not appearing in any factor are also free parameters.)");

    println!("\nDone.");
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Once;
    use symbolica_elim::{load_mathematica_system_from_str_with_var_names, preprocess};

    static LICENSE: Once = Once::new();
    fn activate_license() {
        LICENSE.call_once(|| {
            let _ = LicenseManager::set_license_key(
                "5568d092#6bb8b4fb#ff936916-65ba-552f-b3c1-bdc64597cd21",
            );
        });
    }

    fn problem_from(source: &str) -> (LoadedMathematicaSystem, ProblemData) {
        activate_license();
        let loaded = load_mathematica_system_from_str_with_var_names(source).expect("load");
        let field = AlgebraicExtension::new_complex(Q);
        let pre = preprocess(&loaded, &field);
        let data = build_problem(pre.factor_groups, &loaded);
        (loaded, data)
    }

    #[test]
    fn build_problem_dedupes_shared_factors() {
        activate_license();
        let loaded = load_mathematica_system_from_str_with_var_names("{x + 1 == 0, y + 2 == 0}")
            .expect("load");
        let field = AlgebraicExtension::new_complex(Q);
        let mut pre = preprocess(&loaded, &field);
        assert_eq!(pre.factor_groups.len(), 2);
        let shared = pre.factor_groups[0][0].clone();
        pre.factor_groups[1].push(shared);

        let data = build_problem(pre.factor_groups, &loaded);
        assert_eq!(
            data.factors.len(),
            2,
            "shared factor must be deduplicated into a single entry in F"
        );
        let shared_idx = data.f_by_poly[0][0];
        assert!(
            data.f_by_poly[1].contains(&shared_idx),
            "dedup must reuse the same factor index across polynomials, got {:?}",
            data.f_by_poly
        );
    }

    #[test]
    fn coefficient_can_be_variable() {
        let (_loaded, data) = problem_from("{x^2 * y^6 + x * y^3 + z^2 == 0}");
        assert_eq!(data.factors.len(), 1);

        // the polynomial should be of degree 2 in x and 6 in y and 2 in z
        assert_eq!(data.factors[0].vars.len(), 3);
        assert_eq!(data.factors[0].vars[0].1, 2);
        assert_eq!(data.factors[0].vars[1].1, 6);
        assert_eq!(data.factors[0].vars[2].1, 2);

        assert!(data.admissible.len() == 2);
    }

    #[test]
    fn admissible_respects_degree_bounds() {
        // A single polynomial of degree 5 in x whose (irreducible) square-free part
        // is itself the degree-5 polynomial. No admissible pair can exist.
        let (_loaded, data) = problem_from("{x^5 + x + 3 == 0}");
        assert_eq!(data.factors.len(), 1);
        assert!(
            data.factors[0].vars.iter().any(|(_, d)| *d >= 5),
            "test precondition: factor should have a variable of degree >= 5"
        );
        assert!(
            data.admissible.is_empty(),
            "no admissible pair can exist when every variable has degree >= 5, got {:?}",
            data.admissible
        );
        for &d in &data.deg_of_pair {
            assert!((1..=4).contains(&d));
        }
    }

    fn topo_ok(data: &ProblemData, cover: &Cover) -> bool {
        let dep_vids: std::collections::HashSet<usize> = cover
            .dependent_locals
            .iter()
            .map(|&i| data.v_ids[i])
            .collect();
        let t: std::collections::HashMap<usize, f64> = cover
            .selected
            .iter()
            .map(|&(_, vid, _, t)| (vid, t))
            .collect();
        for &(fi, vid, _, ti) in &cover.selected {
            for (vother, _) in &data.factors[fi].vars {
                if *vother == vid || !dep_vids.contains(vother) {
                    continue;
                }
                let tj = t.get(vother).copied().unwrap_or(f64::INFINITY);
                if tj + 1.0 > ti + 1e-6 {
                    return false;
                }
            }
        }
        true
    }

    #[test]
    fn solve_covers_every_polynomial_linearly() {
        let (_loaded, data) = problem_from("{p -> x + 1, q -> y + 2, r -> z + 3}");
        let cover = solve_milp(&data, false).expect("feasible");
        assert_eq!(cover.selected.len(), 3);
        for j in 0..data.f_by_poly.len() {
            let covered = cover
                .selected
                .iter()
                .any(|(fi, _, _, _)| data.f_by_poly[j].contains(fi));
            assert!(covered, "polynomial {j} is not covered");
        }
        assert!(topo_ok(&data, &cover));
    }

    #[test]
    fn solve_respects_acyclicity() {
        // q factors as x*(y + x); picking (y + x, y) forces t_x < t_y.
        let (_loaded, data) = problem_from("{p -> x + 1, q -> x*y + x^2}");
        let cover = solve_milp(&data, false).expect("feasible");
        assert!(topo_ok(&data, &cover));
        for j in 0..data.f_by_poly.len() {
            assert!(
                cover
                    .selected
                    .iter()
                    .any(|(fi, _, _, _)| data.f_by_poly[j].contains(fi))
            );
        }
    }

    #[test]
    fn solve_infeasible_when_only_high_degree() {
        let (_loaded, data) = problem_from("{x^5 + x + 3 == 0}");
        let res = solve_milp(&data, false);
        assert!(
            matches!(res, Err(ResolutionError::Infeasible)),
            "expected infeasible, got {res:?}"
        );
    }

    #[test]
    fn minimize_degree_prefers_linear() {
        // p has two square-free factors (x+1) (degree 1 in x) and (x^2 + y + 1) (degree 2 in x,
        // degree 1 in y). Count-minimization picks any 1 tuple; degree-minimization should still
        // pick 1 tuple but prefer the degree-1 one.
        let (_loaded, data) = problem_from("{p -> (x + 1) * (x^2 + y + 1)}");
        let cover_deg = solve_milp(&data, true).expect("feasible");
        let total_deg: u32 = cover_deg
            .selected
            .iter()
            .map(|(_, _, d, _)| u32::from(*d))
            .sum();
        assert_eq!(
            total_deg, 1,
            "degree-minimization should pick the linear factor"
        );
    }
}
