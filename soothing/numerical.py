from jax import jit, jacfwd
import jax
import jax.numpy as jnp
from jaxtyping import Array
import numpy as np

from typing import Callable

def newton_raphson(system: Callable[[Array], Array],
                   initial_guess: Array,
                   tol: float = 1e-6,   
                   max_iter: int = 100, 
                   retry: bool = False,
                   key: Array = jax.random.key(0)) -> tuple[Array, list[float], list[float]]:
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

    for step in range(max_iter):
        system_val = system(x)
        error = jnp.linalg.norm(system_val)

        dx = jnp.linalg.pinv(J_system(x)) @ system_val
        x -= dx

        x_list.append(x)
        error_norms.append(error)
        dx_norms.append(jnp.linalg.norm(dx))

        # Check convergence
        if (
            jnp.linalg.norm(system_val, ord=jnp.inf) < tol
            or jnp.linalg.norm(dx, ord=jnp.inf) < tol
        ):
            break

        if (
            jnp.isnan(x).any()
            or jnp.isinf(x).any()
            or jnp.isnan(system_val).any()
            or jnp.isinf(system_val).any()
            ):
            print("Numerical instability encountered.")

            if not retry:
                break

            key, subkey = jax.random.split(key)
            x = jax.random.uniform(subkey, (len(x),), dtype=jnp.float64)

        print(f"Step {step}: Error = {system_val}, dx = {dx_norms[-1]}")

    min_error_index = np.argmin(error_norms)
    x = x_list[min_error_index]

    return x, error_norms, dx_norms


if __name__ == "__main__":
    def system_of_equations(x: Array)->Array:
        return jnp.array([
            x[0]**2 + x[1]**2 - 1,
            x[0]**3 - x[1]
        ])
    
    initial_guess = jnp.array([0.5, 0.5])
    solution, errors, deltas = newton_raphson(system_of_equations, initial_guess)
    print("Solution:", solution)
