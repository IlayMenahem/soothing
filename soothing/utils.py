from typing import Callable

import jax.numpy as jnp
from equinox import filter_jit
from jax import jacfwd
from jaxtyping import Array
from sympy import Poly, Symbol, lambdify


def sympy_to_jax_system(
    polys: list[Poly], 
    variables: list[Symbol]
) -> Callable[[Array], Array]:
    """Convert SymPy polynomials to a JAX-compatible system function.
    
    Uses sympy.lambdify with the 'jax' module to create JAX-compatible functions.
    
    Args:
        polys: List of SymPy Poly objects representing the system of equations.
        variables: List of Symbol objects specifying the variable ordering.
            The order determines how the input array maps to variables.
            
    Returns:
        A callable that takes a JAX array of shape (n,) where n = len(variables)
        and returns a JAX array of shape (m,) where m = len(polys).
    """
    exprs = [p.as_expr() for p in polys]
    funcs = [lambdify(variables, expr, modules=['jax']) for expr in exprs]
    
    def system(x: Array) -> Array:
        """Evaluate the polynomial system at point x.
        
        Args:
            x: JAX array of shape (n,) with variable values.
            
        Returns:
            JAX array of shape (m,) with polynomial evaluations.
        """

        args = [x[i] for i in range(len(variables))]
        results = [f(*args) for f in funcs]
        
        return jnp.array(results)
    
    return system


@filter_jit
def nullspace(J: Array, rtol: float = 1e-9) -> Array:
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
    from .numerical import newton_raphson

    @filter_jit
    def homotopy_system(x: Array, t: Array) -> Array:
        return (1 - t) * system(x) + t * target_system(x)

    homotopy_jacobian = (
        homotopy_jacobian
        if homotopy_jacobian is not None
        else filter_jit(jacfwd(homotopy_system))
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


def solution_test(
    solution: Array,
    system: Callable[[Array], Array],
    jacobian: Callable[[Array], Array] | None = None,
):
    """
    Test whether the provided solution satisfies the system of equations.

    Performs a residual check, and error analysis using condition numbers and sensitivity metrics.

    Args:
        solution: Solution vector of shape (n,) to be tested.
        system: Callable that takes Array of shape (n,) and returns Array of shape (n,).
        jacobian: Optional Jacobian of the system to reuse.
    """
    residual = system(solution)
    residual_norm = jnp.linalg.norm(residual, ord=jnp.inf)
    print(f"Residual norm: {residual_norm:.3e}")

    J_system = jacfwd(system) if jacobian is None else jacobian
    J_at_solution = J_system(solution)

    _, s, _ = jnp.linalg.svd(J_at_solution)
    print(f"Singular values of Jacobian at solution: {s}")
