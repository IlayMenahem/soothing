from typing import Callable
from pprint import pprint

import jax
import jax.numpy as jnp
from jax import jacfwd, jit
from jaxtyping import Array


def newton_raphson(
    system: Callable[[Array], Array],
    initial_guess: Array,
    tol: float = 1e-6,
    max_iter: int = 100,
    retry: bool = False,
    key: Array = jax.random.key(0),
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
    """
    x = initial_guess
    J_system = jit(jacfwd(system))

    x_list = []
    error_norms = []
    dx_norms = []

    for _ in range(max_iter):
        system_val = system(x)
        error = jnp.linalg.norm(system_val)

        dx = jnp.linalg.pinv(J_system(x)) @ system_val
        x -= dx

        x_list.append(x)
        error_norms.append(error)
        dx_norms.append(jnp.linalg.norm(dx))

    error_arr = jnp.stack(error_norms)
    dx_arr = jnp.stack(dx_norms)
    x_arr = jnp.stack(x_list)
    min_error_index = jnp.argmin(error_arr)
    x = x_arr[min_error_index]

    return x, error_arr, dx_arr


def _find_best_index_to_zero(
    x: Array,
    mask: Array,
    system: Callable[[Array], Array],
    tol: float,
    nr_max_iter: int = 10,
    key: Array = jax.random.key(1),
) -> tuple[int, float, Array]:
    """
    Finds the index of the parameter that, when zeroed, has the smallest effect on the system error.

    Args:
    - x: The current solution.
    - system: The system callable.
    - tol: Tolerance for newton_raphson.
    - nr_max_iter: Maximum number of iterations for newton_raphson.
    - key: A key for generating random numbers.

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

        full_solution, _, _ = newton_raphson(
            masked_sys, x, tol, nr_max_iter, key=key
        )

        masked_solution = jnp.where(new_mask, 0.0, full_solution)
        error = jnp.linalg.norm(system(masked_solution))
        return error, masked_solution

    non_zero_indices = jnp.where(~mask)[0]
    if len(non_zero_indices) == 0:
        return -1, jnp.inf, x

    keys = jax.random.split(key, len(non_zero_indices))
    errors, masked_solutions = jax.vmap(_solve_for_index)(non_zero_indices, keys)

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
) -> tuple[Array, float]:
    """
    Sparsify the solution by taking the parameter that zeroing it has the smallest effect on the system.

    Args:
    - solution: An Array of shape (n,) representing the solution to be sparsified.
    - system: A callable that takes an Array of shape (n,) and returns an Array of shape (n,).
    - tol: Tolerance for considering a solution as valid.
    - max_iter: Maximum number of iterations for sparsification.
    - key: A key for generating random numbers.

    Returns:
    - An Array of shape (n,) representing the sparsified solution.
    """
    x = jnp.copy(solution)
    mask = jnp.zeros_like(x, dtype=bool)

    for _ in range(max_iter):
        idx_to_zero, best_error, new_x = _find_best_index_to_zero(
            x, mask, system, tol, key=key
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

    solution, errors, _ = newton_raphson(system_of_equations, jnp.zeros(5), 1e-20)

    pprint(solution)
    print([float(error) for error in errors],'\n')

    sparsified, error = solution_sparseification(solution, system_of_equations, 1e-20)

    pprint(sparsified)
    pprint(error)