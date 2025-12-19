from pprint import pprint
from typing import Callable

import jax
import jax.numpy as jnp
from jax import jacfwd, jit, lax, vmap
from jaxtyping import Array


@jit
def _nullspace(J: Array, rtol: float = 1e-9) -> Array:
    """
    Compute an orthonormal basis for the null space of matrix J using SVD.

    Args:
        J: An Array of shape (m, n).
        rtol: Relative tolerance for determining the rank.

    Returns:
        An Array of shape (n, k) where k is the dimension of the null space.
    """

    _, _, vh = jnp.linalg.svd(J)
    rank = jnp.linalg.matrix_rank(J, rtol)
    null_space = vh.T * jnp.arange(vh.shape[0]) >= rank

    return null_space


def newton_raphson(
    system: Callable[[Array], Array],
    initial_guess: Array,
    tol: float = 1e-6,
    max_iter: int = 100,
    jacobian: Callable[[Array], Array] | None = None,
) -> tuple[Array, Array, Array]:
    """
    Solve a system of equations using the Newton-Raphson method.

    Uses a Jacobian pseudo-inverse step, then adjusts within the Jacobian null
    space with L0-biased weights to prefer sparse directions.

    Args:
        system: A callable that takes an Array of shape (n,) and returns an
            Array of shape (n,).
        initial_guess: An Array of shape (n,) representing the initial guess
            for the solution.
        tol: Tolerance for convergence.
        max_iter: Maximum number of iterations.
        jacobian: Optional callable returning the system Jacobian; if provided
            it will be reused instead of recompiling.

    Returns:
        A tuple containing:
            - The best solution found (Array of shape (n,)).
            - A history of F-norms per iteration (shape (max_iter,)).
            - A history of step norms per iteration (shape (max_iter,)).
    """
    J_system = jacobian if jacobian is not None else jit(jacfwd(system))

    def body_fn(carry: tuple) -> tuple:
        """Execute one iteration of the Newton-Raphson loop.

        Args:
            carry: A tuple containing iteration state (i, x, best_x, best_err,
                err_arr, dx_arr, done).

        Returns:
            Updated carry tuple for the next iteration.
        """
        i, x, best_x, best_err, err_arr, dx_arr, _ = carry

        J = J_system(x)
        system_val = system(x)
        error = jnp.linalg.norm(system_val)

        dx_newton = jnp.linalg.pinv(J, rtol=1e-9) @ system_val
        x_target = x - dx_newton

        null = _nullspace(J)
        null_dim = jnp.sum(jnp.linalg.norm(null, axis=0) > 0)

        def _null_adjust(_: None) -> tuple[Array, Array]:
            """
            Adjust the Newton step within the null space to minimize L0-like error.

            Minimizes weighted error to prefer sparse solutions.

            Args:
                _: Dummy argument for lax.cond compatibility.

            Returns:
                A tuple containing:
                    - step: The adjusted step to take.
                    - x_adj: The adjusted target point.
            """
            weights = 1.0 / (jnp.abs(x_target) + 1e-6)
            weighted_null = null * weights[:, None]

            # Solve weighted least squares: min || W * (x_target + null @ coef) ||
            coef = jnp.linalg.lstsq(weighted_null, -weights * x_target, rcond=1e-9)[0]

            x_adj_candidate = x_target + null @ coef
            x_adj = jnp.where(
                jnp.all(jnp.isfinite(x_adj_candidate)),
                x_adj_candidate,
                x_target,
            )
            return x_adj - x, x_adj

        def _no_adjust(_: None) -> tuple[Array, Array]:
            """Take the standard Newton step without null space adjustment.

            Args:
                _: Dummy argument for lax.cond compatibility.

            Returns:
                A tuple containing:
                    - step: The Newton step.
                    - x_target: The target point after Newton step.
            """
            step = -dx_newton
            return step, x_target

        step, x_next = lax.cond(
            null_dim > 0,
            _null_adjust,
            _no_adjust,
            operand=None,
        )

        dx_norm = jnp.linalg.norm(step)
        err_arr = err_arr.at[i].set(error)
        dx_arr = dx_arr.at[i].set(dx_norm)

        is_better = error < best_err
        best_x = jnp.where(is_better, x, best_x)
        best_err = jnp.where(is_better, error, best_err)

        non_finite = (~jnp.isfinite(error)) | (~jnp.isfinite(dx_norm))
        terminate = (error < tol) | non_finite

        return i + 1, x_next, best_x, best_err, err_arr, dx_arr, terminate

    def cond_fn(carry: tuple) -> Array:
        i, _, _, _, _, _, done = carry
        return (i < max_iter) & (~done)

    err_init = jnp.full(max_iter, jnp.inf)
    dx_init = jnp.full(max_iter, jnp.inf)

    init_carry = (
        0,
        initial_guess,
        initial_guess,
        jnp.inf,
        err_init,
        dx_init,
        False,
    )
    _, x_final, best_x, best_err, err_arr, dx_arr, _ = lax.while_loop(
        cond_fn, body_fn, init_carry
    )

    # Choose the best iterate observed (smallest error).
    x_out = jnp.where(jnp.isfinite(best_err), best_x, x_final)

    return x_out, err_arr, dx_arr


def multiattempt_newton_raphson(
    system: Callable[[Array], Array],
    initial_guesses: Array,
    tol: float = 1e-6,
    max_iter: int = 100,
    jacobian: Callable[[Array], Array] | None = None,
) -> tuple[Array, Array, Array]:
    """
    Solve a system of equations using multiple attempts of the Newton-Raphson method.

    Args:
        system: A callable that takes an Array of shape (n,) and returns an
            Array of shape (n,).
        initial_guesses: An Array of shape (m, n) representing m initial
            guesses for the solution.
        tol: Tolerance for convergence.
        max_iter: Maximum number of iterations for each attempt.
        jacobian: Optional callable returning the system Jacobian; if provided
            it will be reused instead of recompiling.

    Returns:
        A tuple containing:
            - The best solution found across the provided initial guesses
                (Array of shape (n,)).
            - The error history associated with that attempt (shape (max_iter,)).
            - The step norm history associated with that attempt
                (shape (max_iter,)).
    """
    jacobian = jacobian if jacobian is not None else jit(jacfwd(system))

    def single_attempt(x0: Array) -> tuple[Array, Array, Array]:
        return newton_raphson(system, x0, tol, max_iter, jacobian)

    solutions, errors, dxs = vmap(single_attempt)(initial_guesses)
    final_errors = jnp.array([jnp.min(err) for err in errors])
    best_index = jnp.argmin(final_errors)

    return solutions[best_index], errors[best_index], dxs[best_index]


def homotopy_continuation(
    system: Callable[[Array], Array],
    solution: Array,
    target_system: Callable[[Array], Array],
    tol: float = 1e-15,
    max_iter: int = 100,
    steps: int = 10,
    homotopy_jacobian: Callable[[Array, Array], Array] | None = None,
) -> tuple[Array, float]:
    """
    Solve a target system of equations using homotopy continuation from a known solution of a source system.

    Args:
        system: A callable that takes an Array of shape (n,) and returns an
            Array of shape (n,).
        solution: An Array of shape (n,) representing the known solution to
            the source system.
        target_system: A callable that takes an Array of shape (n,) and
            returns an Array of shape (n,).
        tol: Tolerance for convergence.
        max_iter: Maximum number of iterations for each step.
        steps: Number of homotopy steps.
        homotopy_jacobian: Optional Jacobian of the homotopy system to reuse
            across runs.

    Returns:
        A tuple containing:
            - An Array of shape (n,) representing the solution to the target
                system.
            - A float representing the final error.
    """

    @jit
    def homotopy_system(x: Array, t: Array) -> Array:
        return (1 - t) * system(x) + t * target_system(x)

    homotopy_jacobian = (
        homotopy_jacobian
        if homotopy_jacobian is not None
        else jit(jacfwd(homotopy_system))
    )

    def euler_predictor(x: Array, t: Array, dt: Array) -> Array:
        """
        Perform Euler predictor step for homotopy continuation.

        Args:
            x: Current solution point of shape (n,).
            t: Current homotopy parameter.
            dt: Step size in parameter space.

        Returns:
            Predicted point after Euler step (shape (n,)).
        """
        J = homotopy_jacobian(x, t)
        f = homotopy_system(x, t)

        dx_dt = -jnp.linalg.pinv(J) @ f
        new_x = x + dx_dt * dt

        return new_x

    x = jnp.copy(solution)

    dt = jnp.array(1 / steps, dtype=x.dtype)

    for i in range(1, steps + 1):
        t = jnp.array(i / steps, dtype=x.dtype)

        def system_at_t(x_inner: Array) -> Array:
            return homotopy_system(x_inner, t)

        def jacobian_at_t(x_inner: Array) -> Array:
            return homotopy_jacobian(x_inner, t)

        t_prev = jnp.array((i - 1) / steps, dtype=x.dtype)
        x = euler_predictor(x, t_prev, dt)

        x, _, _ = newton_raphson(system_at_t, x, tol, max_iter, jacobian_at_t)

    error = jnp.linalg.norm(target_system(x))

    return x, error


def _best_rationalization(
    x: Array,
    mask: Array,
    system: Callable[[Array], Array],
    tol: float,
    nr_max_iter: int = 10,
    denominators: tuple[int, ...] = tuple(i for i in range(1, 12)),
    reference: Array | None = None,
) -> tuple[int, float | Array, Array]:
    """
    Find the parameter that rationalizing it has the smallest effect on system error.

    Identifies which variable to set to rationalize via by snapping to small-denominator
    fractions, then re-solving the system using homotopy continuation. if the variable is already
    very close to a simple fraction homotopy continuation won't be used on that variable.
    we also prefer to snap to zero if possible.

    Args:
        x: The current solution of shape (n,).
        mask: Boolean mask of shape (n,) indicating which entries are already
            fixed.
        system: The system callable that takes an Array of shape (n,) and
            returns an Array of shape (n,).
        tol: Tolerance for newton_raphson convergence.
        nr_max_iter: Maximum number of iterations for newton_raphson.
        denominators: Candidate denominators to use when snapping masked
            entries to small fractions.
        reference: Reference values used to choose nearby fractions for masked
            entries. Defaults to x if not provided.

    Returns:
        A tuple containing:
            - The index selected for rationalization (int), or -1 if no valid candidate is found.
            - The residual norm after snapping that variable to its chosen simple fraction (float).
            - The updated solution (Array of shape (n,)) with masked entries snapped and the selected variable set to that fraction (zero preferred but not guaranteed).
    """
    reference = x if reference is None else reference
    denominators_arr = jnp.array(denominators, dtype=x.dtype)

    def _nearest_fraction(value: Array) -> Array:
        candidates = jnp.array(
            [0.0] + [jnp.round(value * d) / d for d in denominators],
            dtype=value.dtype,
        )
        deltas = jnp.abs(candidates - value)
        best_idx = jnp.argmin(deltas)
        return candidates[best_idx]

    def _apply_mask_with_fractions(x_full: Array) -> Array:
        """
        Snap masked entries to closest simple fractions.

        Args:
            x_full: Full solution vector of shape (n,).

        Returns:
            Solution with masked entries snapped to fractions (shape (n,)).
        """
        candidate_grid = (
            jnp.round(reference * denominators_arr[:, None]) / denominators_arr[:, None]
        )
        candidate_grid = jnp.vstack(
            [jnp.zeros((1, reference.shape[0]), dtype=x.dtype), candidate_grid]
        )
        closest_idx = jnp.argmin(jnp.abs(candidate_grid - reference), axis=0)
        snapped = candidate_grid[closest_idx, jnp.arange(reference.shape[0])]
        return jnp.where(mask, snapped, x_full)

    def _solve_for_index(idx_to_zero: Array) -> tuple[Array, Array]:
        """
        Solve system while snapping the selected variable to a simple fraction.

        Evaluates small-denominator candidates (preferring zero), and if not
        already near a simple fraction, uses homotopy continuation from the
        current value toward the chosen candidate.

        Args:
            idx_to_zero: The index of the variable being rationalized.

        Returns:
            A tuple containing:
                - The error after snapping that index and re-solving (float).
                - The new solution (Array of shape (n,)).
        """

        nearest_fraction = _nearest_fraction(reference[idx_to_zero])
        x_masked_base = _apply_mask_with_fractions(x)

        def _near_branch(_: None) -> tuple[Array, Array]:
            snapped = x_masked_base.at[idx_to_zero].set(nearest_fraction)
            error = jnp.linalg.norm(system(snapped))
            return error, snapped

        def _far_branch(_: None) -> tuple[Array, Array]:
            candidates = jnp.array(
                [0.0]
                + [jnp.round(reference[idx_to_zero] * d) / d for d in denominators],
                dtype=x.dtype,
            )

            def _eval_candidate(c: Array) -> Array:
                return jnp.linalg.norm(system(x_masked_base.at[idx_to_zero].set(c)))

            candidate_errors = vmap(_eval_candidate)(candidates)
            error_bias = 1e-12
            errors_with_bias = candidate_errors - error_bias * (candidates == 0)
            best_idx = jnp.argmin(errors_with_bias)
            best_error = candidate_errors[best_idx]
            best_candidate = candidates[best_idx]

            def _good_candidate(_: None) -> tuple[Array, Array]:
                snapped = x_masked_base.at[idx_to_zero].set(best_candidate)
                return best_error, snapped

            def _homotopy_branch(_: None) -> tuple[Array, Array]:
                def initial_system(x: Array) -> Array:
                    """System with constraint that x[idx_to_zero] equals its current value.

                    Args:
                        x: Solution point of shape (n+1,) (includes constraint).

                    Returns:
                        System residual plus constraint (shape (n+1,)).
                    """
                    x_masked = _apply_mask_with_fractions(x)
                    residual = system(x_masked)
                    constraint = jnp.array([x_masked[idx_to_zero] - x[idx_to_zero]])
                    return jnp.concatenate([residual, constraint], axis=0)

                def constrained_system(x_full: Array) -> Array:
                    """System with hard constraint that x[idx_to_zero] = best_candidate.

                    Args:
                        x_full: Solution point of shape (n+1,).

                    Returns:
                        System residual plus target constraint (shape (n+1,)).
                    """
                    x_masked = _apply_mask_with_fractions(x_full)
                    residual = system(x_masked)
                    target_constraint = jnp.array(
                        [x_masked[idx_to_zero] - best_candidate]
                    )
                    return jnp.concatenate([residual, target_constraint], axis=0)

                system_jacobian = jacfwd(initial_system)

                def homotopy_jacobian(x: Array, t: Array) -> Array:
                    """Jacobian of homotopy system interpolating constraints.

                    Args:
                        x: Solution point of shape (n+1,).
                        t: Homotopy parameter in [0, 1].

                    Returns:
                        Jacobian matrix of shape (n+1, n+1).
                    """
                    constraint_jacobian = (
                        jnp.zeros((1, x.shape[0]), dtype=x.dtype)
                        .at[0, idx_to_zero]
                        .set(1.0)
                    )
                    jacobian = jnp.concatenate(
                        [system_jacobian(x)[:-1, :], constraint_jacobian], axis=0
                    )

                    return jacobian

                full_solution, error = homotopy_continuation(
                    initial_system,
                    x,
                    constrained_system,
                    tol,
                    nr_max_iter,
                    homotopy_jacobian=homotopy_jacobian,
                )

                masked_solution = (
                    _apply_mask_with_fractions(full_solution)
                    .at[idx_to_zero]
                    .set(best_candidate)
                )
                error = jnp.linalg.norm(system(masked_solution))

                return error, masked_solution

            return lax.cond(
                best_error <= tol,
                _good_candidate,
                _homotopy_branch,
                operand=None,
            )

        return lax.cond(
            jnp.abs(nearest_fraction - reference[idx_to_zero]) <= tol,
            _near_branch,
            _far_branch,
            operand=None,
        )

    non_zero_indices = jnp.where(~mask)[0]
    if len(non_zero_indices) == 0:
        return -1, jnp.inf, x

    errors, masked_solutions = vmap(_solve_for_index)(non_zero_indices)

    finite_errors = jnp.isfinite(errors)
    finite_solutions = jnp.all(jnp.isfinite(masked_solutions), axis=1)
    valid_mask = finite_errors & finite_solutions

    if not bool(jnp.any(valid_mask)):
        return -1, jnp.inf, x

    valid_errors = errors[valid_mask]
    valid_solutions = masked_solutions[valid_mask]
    valid_indices = non_zero_indices[valid_mask]

    best_error_index = jnp.argmin(valid_errors)
    best_error = valid_errors[best_error_index]
    masked_solution = valid_solutions[best_error_index]

    if best_error > tol:
        return -1, best_error, x

    index = int(valid_indices[best_error_index])

    return index, best_error, masked_solution


def _candidate_fraction_values(value: Array, denominators: tuple[int, ...]) -> Array:
    """
    Generate candidate rational approximations for a value.

    Args:
        value: A scalar value to approximate.
        denominators: Tuple of candidate denominators to consider.

    Returns:
        Sorted array of unique candidate fractions including 0.
    """
    raw = jnp.array(
        [0.0] + [jnp.round(value * d) / d for d in denominators], dtype=value.dtype
    )
    return jnp.unique(raw)


def _snap_masked_entries_to_fractions(
    x: Array,
    mask: Array,
    system: Callable[[Array], Array],
    denominators: tuple[int, ...],
    reference: Array,
) -> Array:
    """
    Snap masked entries to simple fractions that minimize system error.

    For each masked entry, tries candidate fractions and selects the one
    that minimizes the system residual norm.

    Args:
        x: Solution vector of shape (n,).
        mask: Boolean mask of shape (n,) indicating which entries are masked.
        system: System callable that takes Array of shape (n,) and returns
            Array of shape (n,).
        denominators: Tuple of candidate denominators for fractions.
        reference: Reference values used to generate fraction candidates.

    Returns:
        Solution with masked entries snapped to fractions (shape (n,)).
    """
    snapped = jnp.copy(x)
    for idx in range(x.shape[0]):
        if not bool(mask[idx]):
            continue

        candidates = _candidate_fraction_values(reference[idx], denominators)
        errors = jnp.array(
            [
                jnp.linalg.norm(system(snapped.at[idx].set(candidate)))
                for candidate in candidates
            ]
        )
        best_idx = int(jnp.argmin(errors))
        snapped = snapped.at[idx].set(candidates[best_idx])

    return snapped


def solution_sparseification(
    solution: Array,
    system: Callable[[Array], Array],
    tol: float = 1e-6,
    max_iter: int = 100,
    denominators: tuple[int, ...] = tuple(i for i in range(1, 12)),
) -> tuple[Array, float]:
    """
    Sparsify solution by zeroing variables and snapping to simple fractions.

    Iteratively identifies the parameter that has the smallest effect on the
    system error when zeroed, then snaps masked entries to small-denominator
    fractions.

    Args:
        solution: Solution vector of shape (n,) to be sparsified.
        system: Callable that takes Array of shape (n,) and returns Array of
            shape (n,).
        tol: Tolerance for considering a solution as valid.
        max_iter: Maximum number of sparsification iterations.
        denominators: Candidate denominators to use when snapping masked
            entries to simple fractions.

    Returns:
        A tuple containing:
            - The sparsified solution (Array of shape (n,)).
            - The residual norm after snapping (float).
    """
    x = jnp.copy(solution)
    reference = jnp.copy(solution)
    mask = jnp.zeros_like(x, dtype=bool)

    for _ in range(max_iter):
        idx_to_zero, best_error, new_x = _best_rationalization(
            x, mask, system, tol, denominators=denominators, reference=reference
        )

        print(best_error, new_x)

        if idx_to_zero == -1 or best_error > tol:
            break

        mask = mask.at[idx_to_zero].set(True)
        x = new_x

    x = _snap_masked_entries_to_fractions(x, mask, system, denominators, reference)
    error = jnp.linalg.norm(system(x))

    return x, error


def estimate_solution_dim(
    system: Callable[[Array], Array],
    solution: Array,
    jacobian: Callable[[Array], Array] | None = None,
) -> int:
    """
    Estimate the dimension of the solution space.

    Computes the rank of the Jacobian at the solution and uses the
    nullity formula: dim = n - rank.

    Args:
        system: Callable that takes Array of shape (n,) and returns Array of
            shape (n,).
        solution: Solution point of shape (n,).
        jacobian: Optional Jacobian of the system to reuse.

    Returns:
        An integer representing the estimated dimension of the solution space.
    """
    J_system = jacobian if jacobian is not None else jacfwd(system)

    J_at_solution = J_system(solution)
    rank = jnp.linalg.matrix_rank(J_at_solution)
    dim = solution.shape[0] - rank

    return dim


def solution_test(solution: Array, system: Callable[[Array], Array]) -> None:
    """
    Test whether the provided solution satisfies the system of equations.

    Performs a first-order check where the Jacobian should be large relative
    to the residual for a good solution.

    Args:
        solution: Solution vector of shape (n,) to be tested.
        system: Callable that takes Array of shape (n,) and returns Array of
            shape (n,).

    Raises:
        NotImplementedError: This function is a placeholder for future
            implementation.
    """
    raise NotImplementedError(
        "This function is a placeholder for future implementation."
    )


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)

    @jit
    def system_of_equations(x: Array) -> Array:
        """
        Example system of equations for testing.

        Args:
            x: Input vector of shape (5,).

        Returns:
            System residual of shape (2,).
        """
        return jnp.array(
            [
                x[0] ** 2 + x[1] + x[2] ** 3 - x[1] * x[4] - 3,
                9 * x[0] + x[1] ** 2 + x[3] - x[2] * x[3] ** 4 - 5,
            ]
        )

    solution, errors, _ = newton_raphson(system_of_equations, jnp.ones(5), 1e-16)

    print("Found solution with error:")
    pprint(solution)
    print([float(error) for error in errors], "\n")
    print(
        "Estimated solution dimension:",
        estimate_solution_dim(system_of_equations, solution),
    )

    print("\nSparsified solution:")
    sparsified, error = solution_sparseification(solution, system_of_equations, 1e-18)

    pprint(sparsified)
    pprint(error)
