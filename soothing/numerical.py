from typing import Callable
from pprint import pprint

import jax
import jax.numpy as jnp
from jax import jacfwd, jit, lax, vmap
from jaxtyping import Array


def newton_raphson(
    system: Callable[[Array], Array],
    initial_guess: Array,
    tol: float = 1e-6,
    max_iter: int = 100,
    retry: bool = False,
    key: Array = jax.random.key(0),
    jacobian: Callable[[Array], Array] | None = None,
) -> tuple[Array, list[float], list[float]]:
    """
    solve a system of equations using the Newton-Raphson method.
    we use shortest vectors in the Jacobian's null space to update our guess.

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

        system_val = system(x)
        error = jnp.linalg.norm(system_val)

        dx = jnp.linalg.pinv(J_system(x)) @ system_val
        dx_norm = jnp.linalg.norm(dx)
        x_next = x - dx

        err_arr = err_arr.at[i].set(error)
        dx_arr = dx_arr.at[i].set(dx_norm)

        is_better = error < best_err
        best_x = jnp.where(is_better, x_next, best_x)
        best_err = jnp.where(is_better, error, best_err)

        non_finite = (~jnp.isfinite(error)) | (~jnp.isfinite(dx_norm))
        terminate = (error < tol) | non_finite

        def retry_branch(args):
            i, x_next, best_x, best_err, err_arr, dx_arr, _, key = args
            key, subkey = jax.random.split(key)
            new_x = jax.random.uniform(subkey, initial_guess.shape, dtype=initial_guess.dtype)
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

    init_carry = (0, initial_guess, initial_guess, jnp.inf, err_init, dx_init, False, key)
    i_final, x_final, best_x, best_err, err_arr, dx_arr, _, _ = lax.while_loop(
        cond_fn, body_fn, init_carry
    )

    # Choose the best iterate observed (smallest error).
    x_out = jnp.where(best_err < jnp.inf, best_x, x_final)

    return x_out, err_arr, dx_arr


def _find_best_index_to_zero(
    x: Array,
    mask: Array,
    system: Callable[[Array], Array],
    tol: float,
    nr_max_iter: int = 10,
    key: Array = jax.random.key(1),
    jacobian: Callable[[Array], Array] | None = None,
) -> tuple[int, float, Array]:
    """
    Finds the index of the parameter that, when zeroed, has the smallest effect on the system error.

    Args:
    - x: The current solution.
    - system: The system callable.
    - tol: Tolerance for newton_raphson.
    - nr_max_iter: Maximum number of iterations for newton_raphson.
    - key: A key for generating random numbers.
    - jacobian: Optional Jacobian of the original system to reuse across candidate zeroings.

    Returns:
    - A tuple containing:
        - The index to zero.
        - The error after zeroing that index and re-solving.
        - The new solution.
    """

    def create_masked_system(mask: Array) -> Callable[[Array], Array]:
        """
        Creates a system function for a reduced number of variables.

        Args:
        - mask: A boolean Array indicating which variables are fixed to zero (True) and which are free (False).
        """

        def masked_system(x_full: Array) -> Array:
            constrained_x = jnp.where(mask, 0.0, x_full)
            return system(constrained_x)

        return masked_system

    base_jacobian = jacobian if jacobian is not None else jit(jacfwd(system))

    def _solve_for_index(idx_to_zero: int, key: Array) -> tuple[float, Array]:
        """
        Solves the system with the specified index zeroed out.

        Args:
        - idx_to_zero: The index of the variable to zero.
        - key: A key for generating random numbers.

        Returns:
        - A tuple containing:
            - The error after zeroing that index and re-solving.
            - The new solution.
        """
        new_mask = mask.at[idx_to_zero].set(True)
        masked_sys = create_masked_system(new_mask)

        def masked_jacobian(x_full: Array) -> Array:
            constrained_x = jnp.where(new_mask, 0.0, x_full)
            return base_jacobian(constrained_x) * (~new_mask).astype(x_full.dtype)

        full_solution, _, _ = newton_raphson(
            masked_sys, x, tol, nr_max_iter, key=key, jacobian=masked_jacobian
        )

        masked_solution = jnp.where(new_mask, 0.0, full_solution)
        error = jnp.linalg.norm(system(masked_solution))
        return error, masked_solution

    non_zero_indices = jnp.where(~mask)[0]
    if len(non_zero_indices) == 0:
        return -1, jnp.inf, x

    keys = jax.random.split(key, len(non_zero_indices))
    errors, masked_solutions = vmap(_solve_for_index)(non_zero_indices, keys)

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
        return -1, jnp.inf, x

    return valid_indices[best_error_index], best_error, masked_solution

def solution_sparseification(
    solution: Array,
    system: Callable[[Array], Array],
    tol: float = 1e-6,
    max_iter: int = 10,
    key: Array = jax.random.key(1),
    jacobian: Callable[[Array], Array] | None = None,
) -> tuple[Array, float]:
    """
    Sparsify the solution by taking the parameter that zeroing it has the smallest effect on the system.

    Args:
    - solution: An Array of shape (n,) representing the solution to be sparsified.
    - system: A callable that takes an Array of shape (n,) and returns an Array of shape (n,).
    - tol: Tolerance for considering a solution as valid.
    - max_iter: Maximum number of iterations for sparsification.
    - key: A key for generating random numbers.
    - jacobian: Optional Jacobian of the system to reuse across runs.

    Returns:
    - An Array of shape (n,) representing the sparsified solution.
    """
    x = jnp.copy(solution)
    mask = jnp.zeros_like(x, dtype=bool)

    for _ in range(max_iter):
        idx_to_zero, best_error, new_x = _find_best_index_to_zero(
            x, mask, system, tol, key=key, jacobian=jacobian
        )

        print(best_error, new_x)

        if idx_to_zero == -1 or best_error > tol:
            break

        mask = mask.at[idx_to_zero].set(True)
        x = new_x

    x = jnp.where(mask, 0.0, x)
    error = jnp.linalg.norm(system(x))

    return x, error


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)

    def system_of_equations(x: Array) -> Array:
        return jnp.array(
            [
                x[0] ** 2 + x[1] + x[2] ** 3 - x[1] * x[4] - 1,
                9 * x[0] + x[1] ** 2 + x[3] - x[2] * x[3] - 6,
            ]
        )

    solution, errors, _ = newton_raphson(system_of_equations, jnp.ones(5), 1e-16)

    pprint(solution)
    print([float(error) for error in errors],'\n')

    sparsified, error = solution_sparseification(solution, system_of_equations, 1e-16)

    pprint(sparsified)
    pprint(error)