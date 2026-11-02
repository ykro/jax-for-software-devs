"""Experiment 03: jax.grad for Software Devs (Autodiff as Code Transformation).

Demonstrates:
1. Automatic differentiation without Deep Learning: `jax.grad(f)` transforms a pure
   Python mathematical function into a new function that computes exact derivatives.
2. Why `jax.value_and_grad` beats central finite differences `(f(x+eps)-f(x-eps))/(2*eps)`
   in both precision and computational complexity (`O(1)` passes vs `O(2*D)` evaluations
   for a `D`-dimensional parameter vector).
3. Solving a 32-parameter thermal/sensor calibration problem in 20 optimization steps.

Two speedups are reported. The headline one compares interpreted NumPy finite
differences against compiled `value_and_grad`, so it mixes compilation with autodiff.
The fair one compiles the finite differences too (`jit(vmap)` over the 64 perturbed
evaluations), which isolates what autodiff itself saves.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


def sensor_calibration_loss(
    params: jax.Array,
    features: jax.Array,
    targets: jax.Array,
) -> jax.Array:
    """Non-linear sensor response calibration objective (pure scalar function)."""
    linear_response = features @ params
    non_linear_correction = 0.15 * jnp.sin(linear_response) + 0.05 * (linear_response**2)
    predictions = linear_response + non_linear_correction
    residuals = predictions - targets
    # Pseudo-Huber smooth loss + L2 regularization
    delta = 0.5
    huber = (delta**2) * (jnp.sqrt(1.0 + (residuals / delta) ** 2) - 1.0)
    return jnp.mean(huber) + 1e-5 * jnp.sum(params**2)


def finite_difference_grad(
    params: np.ndarray,
    features: np.ndarray,
    targets: np.ndarray,
    eps: float = 1e-3,
) -> np.ndarray:
    """Compute gradient in NumPy via central finite differences (2*D evaluations)."""

    def loss_np(p: np.ndarray) -> float:
        lin = features @ p
        nl = 0.15 * np.sin(lin) + 0.05 * (lin**2)
        res = (lin + nl) - targets
        delta = 0.5
        huber = (delta**2) * (np.sqrt(1.0 + (res / delta) ** 2) - 1.0)
        return float(np.mean(huber) + 1e-5 * np.sum(p**2))

    grad = np.zeros_like(params)
    for i in range(params.shape[0]):
        p_plus = params.copy()
        p_minus = params.copy()
        p_plus[i] += eps
        p_minus[i] -= eps
        grad[i] = (loss_np(p_plus) - loss_np(p_minus)) / (2.0 * eps)
    return grad


loss_and_grad_fn = jax.jit(jax.value_and_grad(sensor_calibration_loss))


@jax.jit
def finite_difference_grad_jit(
    params: jax.Array,
    features: jax.Array,
    targets: jax.Array,
    eps: float = 1e-3,
) -> jax.Array:
    """Fairer baseline: the same 2*D central differences, compiled with jit(vmap)."""
    steps = jnp.eye(params.shape[0], dtype=params.dtype) * eps

    def loss_at(delta: jax.Array) -> jax.Array:
        return sensor_calibration_loss(params + delta, features, targets)

    plus = jax.vmap(loss_at)(steps)
    minus = jax.vmap(loss_at)(-steps)
    return (plus - minus) / (2.0 * eps)


def _median_ms(fn: Callable[[], Any], n_iters: int) -> float:
    """Median wall time of `fn()` in ms (robust to noisy neighbours on a shared machine)."""
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times)


@jax.jit
def calibration_step(
    params: jax.Array,
    features: jax.Array,
    targets: jax.Array,
    lr: float = 0.25,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Perform one exact gradient descent step."""
    loss_val, grads = loss_and_grad_fn(params, features, targets)
    new_params = params - lr * grads
    return new_params, loss_val, grads


def run_benchmark(
    n_samples: int = 2048,
    n_params: int = 32,
    n_steps: int = 20,
    n_iters: int = 80,
) -> dict[str, Any]:
    """Compare finite-difference gradients vs compiled `jax.value_and_grad` and calibrate."""
    rng = np.random.default_rng(123)
    features_np = (rng.standard_normal((n_samples, n_params)) / np.sqrt(n_params)).astype(
        np.float32
    )
    true_params_np = rng.uniform(-1.2, 1.2, size=(n_params,)).astype(np.float32)
    lin_true = features_np @ true_params_np
    targets_np = (lin_true + 0.15 * np.sin(lin_true) + 0.05 * (lin_true**2)).astype(np.float32)

    init_params_np = np.zeros((n_params,), dtype=np.float32)
    features_jax = jnp.asarray(features_np)
    targets_jax = jnp.asarray(targets_np)
    init_params_jax = jnp.asarray(init_params_np)

    # Warmup JAX autodiff
    init_loss, exact_grad = loss_and_grad_fn(init_params_jax, features_jax, targets_jax)
    exact_grad.block_until_ready()

    # Compute central finite differences in NumPy
    fd_grad = finite_difference_grad(init_params_np, features_np, targets_np, eps=1e-3)
    grad_max_diff = float(np.max(np.abs(fd_grad - np.asarray(exact_grad))))
    fd_jit_grad = finite_difference_grad_jit(init_params_jax, features_jax, targets_jax)
    fd_jit_grad.block_until_ready()
    fd_jit_max_diff = float(np.max(np.abs(np.asarray(fd_jit_grad) - np.asarray(exact_grad))))

    # Benchmark finite differences vs jax.jit(jax.value_and_grad) (median per call)
    fd_ms = _median_ms(
        lambda: finite_difference_grad(init_params_np, features_np, targets_np, eps=1e-3), n_iters
    )
    fd_jit_ms = _median_ms(
        lambda: finite_difference_grad_jit(
            init_params_jax, features_jax, targets_jax
        ).block_until_ready(),
        n_iters,
    )
    autodiff_ms = _median_ms(
        lambda: loss_and_grad_fn(init_params_jax, features_jax, targets_jax)[1].block_until_ready(),
        n_iters,
    )

    speedup = fd_ms / autodiff_ms
    speedup_vs_compiled_fd = fd_jit_ms / autodiff_ms

    # Run 20-step sensor calibration optimization
    params = init_params_jax
    initial_loss_val = float(init_loss)
    final_loss_val = initial_loss_val
    for _ in range(n_steps):
        params, step_loss, _ = calibration_step(params, features_jax, targets_jax, lr=6.0)
        final_loss_val = float(step_loss)

    loss_reduction_ratio = initial_loss_val / max(final_loss_val, 1e-9)

    return {
        "experiment": "03_grad_as_code_transformation",
        "backend": jax.default_backend(),
        "n_samples": n_samples,
        "n_params": n_params,
        "n_steps": n_steps,
        "initial_loss": round(initial_loss_val, 6),
        "final_loss": round(final_loss_val, 6),
        "loss_reduction_ratio": round(loss_reduction_ratio, 2),
        "grad_max_diff_vs_finite_diff": grad_max_diff,
        "finite_diff_jit_max_diff_vs_exact": fd_jit_max_diff,
        "finite_diff_ms": round(fd_ms, 4),
        "finite_diff_jit_ms": round(fd_jit_ms, 4),
        "jax_value_and_grad_ms": round(autodiff_ms, 4),
        "speedup": round(speedup, 2),
        "speedup_vs_compiled_fd": round(speedup_vs_compiled_fd, 2),
    }


def main() -> None:
    res = run_benchmark()
    print("=" * 72)
    print("Experiment 03: jax.grad as a Code Transformation (Sensor Calibration)")
    print("=" * 72)
    print(f"Backend Device                   : {res['backend']}")
    print(f"Calibration Problem              : {res['n_samples']} samples, {res['n_params']} params")
    print(f"Initial Calibration Loss         : {res['initial_loss']:.6f}")
    print(f"Loss After {res['n_steps']} Steps              : {res['final_loss']:.6f} ({res['loss_reduction_ratio']:.2f}x lower)")
    print("-" * 72)
    print(f"Central Finite Diff (64 evals)   : {res['finite_diff_ms']:.4f} ms")
    print(f"Compiled Finite Diff jit(vmap)   : {res['finite_diff_jit_ms']:.4f} ms")
    print(f"JAX @jit(value_and_grad) (1 pass): {res['jax_value_and_grad_ms']:.4f} ms")
    print(f"Speedup vs Finite Differences    : {res['speedup']:.2f}x")
    print(f"Speedup vs Compiled Finite Diff  : {res['speedup_vs_compiled_fd']:.2f}x (autodiff alone)")
    print(f"Max |Finite Diff - Exact Grad|   : {res['grad_max_diff_vs_finite_diff']:.2e}")


if __name__ == "__main__":
    main()
