from typing import Callable

import jax.numpy as jnp
from equinox import filter_jit
from jax import jacfwd, lax, vmap
from jaxtyping import Array

from .utils import homotopy_continuation


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


def solution_rationalization(
    solution: Array,
    system: Callable[[Array], Array],
    tol: float = 1e-6,
    max_iter: int = 100,
    denominators: tuple[int, ...] = tuple(i for i in range(1, 12)),
) -> tuple[Array, float]:
    """
    rationalize solution by rationalizing variables and snapping to simple fractions.

    Iteratively identifies the parameter that has the smallest effect on the
    system error when zeroed, then snaps masked entries to small-denominator
    fractions.

    Args:
        solution: Solution vector of shape (n,) to be rationalized.
        system: Callable that takes Array of shape (n,) and returns Array of
            shape (n,).
        tol: Tolerance for considering a solution as valid.
        max_iter: Maximum number of rationalization iterations.
        denominators: Candidate denominators to use when snapping masked
            entries to simple fractions.

    Returns:
        A tuple containing:
            - The rationalized solution (Array of shape (n,)).
            - The residual norm after snapping (float).
    """

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

        def _candidate_fraction_values(
            value: Array, denominators: tuple[int, ...]
        ) -> Array:
            """
            Generate candidate rational approximations for a value.

            Args:
                value: A scalar value to approximate.
                denominators: Tuple of candidate denominators to consider.

            Returns:
                Sorted array of unique candidate fractions including 0.
            """
            raw = jnp.array(
                [0.0] + [jnp.round(value * d) / d for d in denominators],
                dtype=value.dtype,
            )
            return jnp.unique(raw)

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


def simple_sparsification(
    solution: Array,
    system: Callable[[Array], Array],
    tol: float = 1e-6,
    max_iter: int = 100,
    jacobian: Callable[[Array], Array] | None = None,
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

    Returns:
        A tuple containing:
            - The sparsified solution (Array of shape (n,)).
            - The residual norm after sparsification (float).
    """
    x = jnp.copy(solution)
    mask = jnp.zeros_like(x, dtype=bool)
    jacobian = jacobian if jacobian is not None else filter_jit(jacfwd(system))

    for _ in range(min(len(x), max_iter)):

        def _zero_index(idx: Array) -> tuple[Array, Array]:
            """
            Attempt to also zero the entry at idx and correct via Newton-Raphson.

            Args:
                idx: Index to zero.

            Returns:
                - Tuple containing:
                    - The error after attempting to zero that index (float).
                    - The new solution (Array of shape (n,)).
            """
            x_candidate = x.at[idx].set(0.0)
            error = jnp.linalg.norm(system(x_candidate))

            return error, x_candidate

        idx_candidates = jnp.where(~mask)[0]

        errors, candidate_solutions = vmap(_zero_index)(idx_candidates)
        best_idx = jnp.argmin(errors)
        best_error = errors[best_idx]
        new_x = candidate_solutions[best_idx]

        print(best_error, new_x)

        if best_error > tol:
            break

        mask = mask.at[idx_candidates[best_idx]].set(True)
        x = new_x

    error = jnp.linalg.norm(system(x))
    return x, error
