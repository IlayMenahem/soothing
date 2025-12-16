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
    - J: An Array of shape (m, n).
    - rtol: Relative tolerance for determining the rank.

    Returns:
    - An Array of shape (n, k) where k is the dimension of the null space.
    """

    u, s, vh = jnp.linalg.svd(J, full_matrices=True)
    rank = jnp.linalg.matrix_rank(J, rtol)
    null_space = vh[rank:].T

    return null_space


def newton_raphson(
    system: Callable[[Array], Array],
    initial_guess: Array,
    tol: float = 1e-6,
    max_iter: int = 100,
    retry: bool = False,
    key: Array = jax.random.key(0),
    jacobian: Callable[[Array], Array] | None = None,
) -> tuple[Array, Array, Array]:
    """
    Solve a system of equations using the Newton-Raphson method.
    Uses a Jacobian pseudo-inverse step, then adjusts within the Jacobian null space with L0-biased weights to prefer sparse directions.

    Args:
    - system: A callable that takes an Array of shape (n,) and returns an Array of shape (n,).
    - initial_guess: An Array of shape (n,) representing the initial guess for the solution.
    - tol: Tolerance for convergence.
    - max_iter: Maximum number of iterations.
    - retry: Whether to retry with a different initial guess if convergence fails.
    - key: A key for generating random numbers.
    - jacobian: Optional callable returning the system Jacobian; if provided it will be reused instead of recompiling.
    """
    J_system = jacobian if jacobian is not None else jit(jacfwd(system))

    def body_fn(carry):
        i, x, best_x, best_err, err_arr, dx_arr, done, key = carry

        J = J_system(x)
        system_val = system(x)
        error = jnp.linalg.norm(system_val)

        dx_newton = jnp.linalg.pinv(J, rtol=1e-9) @ system_val
        x_target = x - dx_newton

        null = _nullspace(J)
        null_dim = jnp.sum(jnp.linalg.norm(null, axis=0) > 0)

        def _null_adjust(_) -> tuple[Array, Array]:
            """
            Adjust the Newton step within the null space to minimize the L0-like weighted error.

            Args:
            - _: Dummy argument for lax.cond compatibility.

            Returns:
            - step: The adjusted step to take.
            - x_adj: The adjusted target point.
            """
            weights = 1.0 / (jnp.abs(x_target) + 1e-6)
            weighted_null = null * weights[:, None]
            rhs = -weights * x_target
            gram = weighted_null.T @ weighted_null
            coef = jnp.linalg.pinv(gram, rtol=1e-9) @ (weighted_null.T @ rhs)
            x_adj_candidate = x_target + null @ coef
            x_adj = lax.cond(
                jnp.all(jnp.isfinite(x_adj_candidate)),
                lambda _: x_adj_candidate,
                lambda _: x_target,
                operand=None,
            )
            step = x_adj - x
            return step, x_adj

        def _no_adjust(_):
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

        def retry_branch(args):
            i, x_next, best_x, best_err, err_arr, dx_arr, _, key = args
            key, subkey = jax.random.split(key)
            new_x = jax.random.uniform(subkey, initial_guess.shape).astype(
                initial_guess.dtype
            )
            return i + 1, new_x, best_x, best_err, err_arr, dx_arr, False, key

        def continue_branch(args):
            i, x_next, best_x, best_err, err_arr, dx_arr, terminate, key = args
            return i + 1, x_next, best_x, best_err, err_arr, dx_arr, terminate, key

        return lax.cond(
            retry & non_finite,
            retry_branch,
            continue_branch,
            (i, x_next, best_x, best_err, err_arr, dx_arr, terminate, key),
        )

    def cond_fn(carry):
        i, _, _, _, _, _, done, _ = carry
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
        key,
    )
    i_final, x_final, best_x, best_err, err_arr, dx_arr, _, _ = lax.while_loop(
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
    - system: A callable that takes an Array of shape (n,) and returns an Array of shape (n,).
    - initial_guesses: An Array of shape (m, n) representing m initial guesses for the solution.
    - tol: Tolerance for convergence.
    - max_iter: Maximum number of iterations for each attempt.
    - key: A key for generating random numbers.
    - jacobian: Optional callable returning the system Jacobian; if provided it will be reused instead of recompiling.

    Returns:
    - The best solution found across all attempts, along with its error and step sizes.
    """
    jacobian = jacobian if jacobian is not None else jit(jacfwd(system))

    def single_attempt(x0: Array) -> tuple[Array, Array, Array]:
        return newton_raphson(system, x0, tol, max_iter, False, jacobian=jacobian)

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
    - system: A callable that takes an Array of shape (n,) and returns an Array of shape (n,).
    - solution: An Array of shape (n,) representing the known solution to the source system.
    - target_system: A callable that takes an Array of shape (n,) and returns an Array of shape (n,).
    - tol: Tolerance for convergence.
    - max_iter: Maximum number of iterations for each step.
    - steps: Number of homotopy steps.
    - homotopy_jacobian: Optional Jacobian of the homotopy system to reuse across runs.

    Returns:
    - An Array of shape (n,) representing the solution to the target system.
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

    @jit
    def euler_predictor(x: Array, t: Array, dt: Array) -> Array:
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

        x, _, _ = newton_raphson(
            system_at_t,
            x,
            tol,
            max_iter,
            jacobian=jacobian_at_t,
        )

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
) -> tuple[int, float, Array]:
    """
    Finds the index of the parameter that, when zeroed, has the smallest effect on the system error.

    Args:
    - x: The current solution.
    - mask: Boolean mask indicating which entries are already fixed.
    - system: The system callable.
    - tol: Tolerance for newton_raphson.
    - nr_max_iter: Maximum number of iterations for newton_raphson.
    - denominators: Candidate denominators to use when snapping masked entries to small fractions.
    - reference: Reference values used to choose nearby fractions for masked entries.

    Returns:
    - A tuple containing:
        - The index to zero.
        - The error after zeroing that index and re-solving.
        - The new solution.
    """
    reference = x if reference is None else reference
    denominators_arr = jnp.array(denominators, dtype=x.dtype)

    @jit
    def _apply_mask_with_fractions(x_full: Array) -> Array:
        candidate_grid = (
            jnp.round(reference * denominators_arr[:, None]) / denominators_arr[:, None]
        )
        candidate_grid = jnp.vstack(
            [jnp.zeros((1, reference.shape[0]), dtype=x.dtype), candidate_grid]
        )
        closest_idx = jnp.argmin(jnp.abs(candidate_grid - reference), axis=0)
        snapped = candidate_grid[closest_idx, jnp.arange(reference.shape[0])]
        return jnp.where(mask, snapped, x_full)

    def _solve_for_index(idx_to_zero: Array) -> tuple[float, Array]:
        """
        Solves the system with the goal of zeroing the variable at idx_to_zero, this will be done homotopy continuation by adding a constraint to the system of the form x[idx_to_zero] = t.

        Args:
        - idx_to_zero: The index of the variable to zero.
        - key: A key for generating random numbers.

        Returns:
        - A tuple containing:
            - The error after zeroing that index and re-solving.
            - The new solution.
        """

        @jit
        def initial_system(x: Array) -> Array:
            x_masked = _apply_mask_with_fractions(x)
            residual = system(x_masked)
            constraint = jnp.array([x_masked[idx_to_zero] - x[idx_to_zero]])
            return jnp.concatenate([residual, constraint], axis=0)

        @jit
        def constrained_system(x_full: Array) -> Array:
            x_masked = _apply_mask_with_fractions(x_full)
            residual = system(x_masked)
            zero_constraint = jnp.array([x_masked[idx_to_zero]])
            return jnp.concatenate([residual, zero_constraint], axis=0)

        system_jacobian = jit(jacfwd(initial_system))

        @jit
        def homotopy_jacobian(x: Array, t: Array) -> Array:
            constraint_jacobian = (
                jnp.zeros((1, x.shape[0]), dtype=x.dtype).at[0, idx_to_zero].set(1.0)
            )
            jacobian = jnp.concatenate(
                [system_jacobian(x)[:-1, :], constraint_jacobian], axis=0
            )

            return jacobian

        full_solution, _ = homotopy_continuation(
            initial_system,
            x,
            constrained_system,
            tol,
            nr_max_iter,
            steps=50,
            homotopy_jacobian=homotopy_jacobian,
        )

        masked_solution = (
            _apply_mask_with_fractions(full_solution).at[idx_to_zero].set(0.0)
        )
        error = jnp.linalg.norm(system(masked_solution))

        return error, masked_solution

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

    return valid_indices[best_error_index], best_error, masked_solution


def _candidate_fraction_values(value: Array, denominators: tuple[int, ...]) -> Array:
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
    Sparsify the solution by taking the parameter that zeroing it has the smallest effect on the system, then snap masked entries to small-denominator fractions.

    Args:
    - solution: An Array of shape (n,) representing the solution to be sparsified.
    - system: A callable that takes an Array of shape (n,) and returns an Array of shape (n,).
    - tol: Tolerance for considering a solution as valid.
    - max_iter: Maximum number of iterations for sparsification.
    - denominators: Candidate denominators to use when snapping masked entries to simple fractions.

    Returns:
    - An Array of shape (n,) representing the sparsified solution.
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
    Estimate the dimension of the solution space by computing the rank of the Jacobian at the solution.

    Args:
    - system: A callable that takes an Array of shape (n,) and returns an Array of shape (n,).
    - solution: An Array of shape (n,) representing the solution.
    - jacobian: Optional Jacobian of the system to reuse.

    Returns:
    - An integer representing the estimated dimension of the solution space.
    """
    J_system = jacobian if jacobian is not None else jacfwd(system)

    J_at_solution = J_system(solution)
    rank = jnp.linalg.matrix_rank(J_at_solution)
    dim = solution.shape[0] - rank

    return dim


def solution_test(solution: Array, system: Callable[[Array], Array]) -> None:
    """
    Test whether the provided solution satisfies the system of equations, the test is a first order check, meaning the jacobian should be large relative to the residual.

    Args:
    - solution: An Array of shape (n,) representing the solution to be tested.
    - system: A callable that takes an Array of shape (n,) and returns an Array of shape (n,).
    """
    raise NotImplementedError(
        "This function is a placeholder for future implementation."
    )


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)

    @jit
    def system_of_equations(x: Array) -> Array:
        return jnp.array(
            [
                x[0] ** 2 + x[1] + x[2] ** 3 - x[1] * x[4] - 1,
                9 * x[0] + x[1] ** 2 + x[3] - x[2] * x[3] ** 4 - 5,
            ]
        )

    solution, errors, _ = newton_raphson(system_of_equations, jnp.ones(5), 1e-16)

    pprint(solution)
    print([float(error) for error in errors], "\n")
    print(
        "Estimated solution dimension:",
        estimate_solution_dim(system_of_equations, solution),
    )

    sparsified, error = solution_sparseification(solution, system_of_equations, 1e-15)

    pprint(sparsified)
    pprint(error)
