from pprint import pprint
from typing import Callable

import jax
import jax.numpy as jnp
from equinox import filter_jit
from jax import jacfwd, lax, vmap
from jaxtyping import Array

from .symbolic import simple_sparsification
from .utils import estimate_solution_dim, nullspace


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
    J_system = jacobian if jacobian is not None else filter_jit(jacfwd(system))

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

        null = nullspace(J)
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
    jacobian = jacobian if jacobian is not None else filter_jit(jacfwd(system))

    def single_attempt(x0: Array) -> tuple[Array, Array, Array]:
        return newton_raphson(system, x0, tol, max_iter, jacobian)

    solutions, errors, dxs = vmap(single_attempt)(initial_guesses)
    final_errors = jnp.array([jnp.min(err) for err in errors])
    best_index = jnp.argmin(final_errors)

    return solutions[best_index], errors[best_index], dxs[best_index]


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update(
        "jax_persistent_cache_enable_xla_caches",
        "xla_gpu_per_fusion_autotune_cache_dir",
    )

    @filter_jit
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
    sparsified, error = simple_sparsification(solution, system_of_equations, 1e-18)

    pprint(sparsified)
    pprint(error)
