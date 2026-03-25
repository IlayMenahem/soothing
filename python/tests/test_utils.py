import jax.numpy as jnp

from soothing.utils import homotopy_continuation


def test_homotopy_continuation_linear_shift() -> None:
    start = jnp.array([2.0, -1.0])
    target = jnp.array([0.5, 3.0])

    def system(x: jnp.ndarray) -> jnp.ndarray:
        return x - start

    def target_system(x: jnp.ndarray) -> jnp.ndarray:
        return x - target

    solution, error = homotopy_continuation(
        system,
        start,
        target_system,
        tol=1e-10
    )

    assert jnp.allclose(solution, target, atol=1e-6)
    assert float(error) < 1e-6


def test_homotopy_continuation_with_zeroed_variable() -> None:
    solution = jnp.array([0.0, 2.0])

    def system(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.array([x[0], x[1] - 2.0])

    def target_system(x: jnp.ndarray) -> jnp.ndarray:
        return system(x.at[0].set(0.0))

    new_solution, error = homotopy_continuation(
        system,
        solution,
        target_system,
        tol=1e-10,
    )

    assert jnp.allclose(new_solution, solution, atol=1e-6)
    assert float(error) < 1e-6


def test_homotopy_continuation_nonlinear_rotation() -> None:
    start = jnp.array([1.0, 0.0])
    target = jnp.array([0.0, 1.0])

    def system(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.array([x[0] ** 2 + x[1] ** 2 - 1.0, x[0] - 1.0])

    def target_system(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.array([x[0] ** 2 + x[1] ** 2 - 1.0, x[1] - 1.0])

    solution, error = homotopy_continuation(
        system,
        start,
        target_system,
        tol=1e-10,
        max_iter=25,
        steps=20,
    )

    assert jnp.allclose(solution, target, atol=1e-4)
    assert float(error) < 1e-5
