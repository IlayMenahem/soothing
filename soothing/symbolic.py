import math
from typing import Callable

import jax.numpy as jnp
from jaxtyping import Array

from .utils import homotopy_continuation


def try_zero_variables(
    solution: Array,
    system: Callable[[Array], Array],
    variable_indices: list[int] | None = None,
    tol: float = 1e-6,
) -> list[tuple[int, Array, float]]:
    """Try to zero each variable and return which ones can be zeroed.
    
    For each variable index, uses homotopy continuation to deform the system
    to one where that variable is constrained to 0, then checks if a valid
    solution still exists.
    
    Args:
        solution: Current solution vector of shape (n,).
        system: Callable that takes Array of shape (n,) and returns residuals.
        variable_indices: List of indices to try zeroing. If None, tries all.
        tol: Tolerance for considering a zeroing successful.
        
    Returns:
        List of tuples (idx, new_solution, error) for each variable that can
        be successfully zeroed (error < tol). Sorted by error (best first).
    """
    if variable_indices is None:
        variable_indices = list(range(len(solution)))
    
    results = []
    
    for idx in variable_indices:
        # Skip if already zero
        if jnp.abs(solution[idx]) < 1e-12:
            results.append((idx, solution, 0.0))
            continue
            
        # Create target system where variable idx is constrained to 0
        def make_target_system(i: int):
            def target(x: Array) -> Array:
                return system(x.at[i].set(0.0))
            return target
        
        target_system = make_target_system(idx)
        
        # Use homotopy continuation to find solution with x[idx] = 0
        new_solution, error = homotopy_continuation(
            system, solution, target_system
        )
        
        # Set the zeroed variable explicitly to 0
        new_solution = new_solution.at[idx].set(0.0)
        error = float(jnp.linalg.norm(system(new_solution)))
        
        if error < tol:
            results.append((idx, new_solution, error))
    
    # Sort by error (best first)
    results.sort(key=lambda x: x[2])
    
    return results


def try_zero_single_variable(
    solution: Array,
    system: Callable[[Array], Array],
    idx: int,
    tol: float = 1e-6,
) -> tuple[Array, float] | None:
    """Try to zero a single variable at the given index.
    
    Args:
        solution: Current solution vector of shape (n,).
        system: Callable that takes Array of shape (n,) and returns residuals.
        idx: Index of the variable to zero.
        tol: Tolerance for considering the zeroing successful.
        
    Returns:
        Tuple of (new_solution, error) if zeroing succeeded, None otherwise.
    """
    # Skip if already zero
    if jnp.abs(solution[idx]) < 1e-12:
        return solution, 0.0
    
    # Create target system where variable idx is constrained to 0
    def target_system(x: Array) -> Array:
        return system(x.at[idx].set(0.0))
    
    # Use homotopy continuation
    new_solution, _ = homotopy_continuation(system, solution, target_system)
    
    # Set the zeroed variable explicitly to 0
    new_solution = new_solution.at[idx].set(0.0)
    error = float(jnp.linalg.norm(system(new_solution)))
    
    if error < tol:
        return new_solution, error
    return None


def simple_sparsification(
    solution: Array,
    system: Callable[[Array], Array],
    tol: float = 1e-6,
    max_iter: int = 100,
    jacobian: Callable[[Array], Array] | None = None,
    verbose: bool = True
) -> tuple[Array, float]:
    """
    do a simple sparsification of the solution by zeroing entries and making newton corrections.

    Args:
        solution: Solution vector of shape (n,) to be sparsified.
        system: Callable that takes Array of shape (n,) and returns Array of
            shape (n,).
        tol: Tolerance for considering a solution as valid.
        max_iter: Maximum number of sparsification iterations.
        jacobian: Optional Jacobian of the system to reuse.
        verbose: Whether to print verbose output.

    Returns:
        A tuple containing:
            - The sparsified solution (Array of shape (n,)).
            - The residual norm after sparsification (float).
    """
    x = jnp.copy(solution)
    mask = jnp.zeros_like(x, dtype=bool)

    def masked_system(active_mask: Array) -> Callable[[Array], Array]:
        def _system(x_inner: Array) -> Array:
            return system(x_inner.at[active_mask].set(0.0))

        return _system

    for _ in range(min(len(x), max_iter)):
        idx_candidates = jnp.where(~mask)[0]
        if idx_candidates.size == 0:
            break

        current_system = masked_system(mask)
        x = x.at[mask].set(0.0)

        closest_idx = int(idx_candidates[jnp.argmin(jnp.abs(x[idx_candidates]))])
        candidate_mask = mask.at[closest_idx].set(True)
        target_system = masked_system(candidate_mask)
        x_homotopy, _ = homotopy_continuation(current_system, x, target_system)
        x_candidate = x_homotopy.at[candidate_mask].set(0.0)
        error = float(jnp.linalg.norm(system(x_candidate)))

        if not math.isfinite(error) or error > tol:
            break

        mask = mask.at[closest_idx].set(True)
        x = x_candidate

        if verbose:
            print(error, x)

    x = x.at[mask].set(0.0)
    error = jnp.linalg.norm(system(x))
    return x, error
