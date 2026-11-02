"""Experiment 01: From NumPy to jax.numpy and Functional Immutability.

Demonstrates:
1. API parity between `numpy` (`np`) and `jax.numpy` (`jnp`).
2. Why JAX arrays are immutable (preventing shared-buffer aliasing so OpenXLA
   can fuse and parallelize operations without race conditions).
3. Functional out-of-place updates via `.at[idx].set(val)` (which XLA optimizes
   in-place inside compiled `@jax.jit` kernels when safe).
4. A vectorized Monte Carlo Black-Scholes / portfolio simulation comparing
   NumPy vs JAX Eager vs `@jax.jit` compiled execution.

Where the speedup comes from: `.at[].set()` by itself does not make code faster
(JAX eager with the same `.at[].set()` is only slightly faster than NumPy). The gain
comes from `@jax.jit` handing the whole function to XLA, which fuses it into a few
kernels and computes `tanh(shocks)` and `exp(...)` once instead of once per loop step
(common-subexpression elimination). To show how much is left once that redundant work
is removed by hand, the script also times a NumPy version with the loop-invariant
expressions hoisted out of the loop (`numpy_hoisted_portfolio_simulation`).
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


def numpy_portfolio_simulation(
    prices: np.ndarray,
    weights: np.ndarray,
    shocks: np.ndarray,
) -> np.ndarray:
    """Simulate multi-step compounded portfolio returns in NumPy."""
    h = prices.copy()
    # Apply shock cap in-place on a copy
    h[:, 0] = np.clip(h[:, 0], 80.0, 120.0)
    for _ in range(6):
        drifted = h * (1.0 + 0.015 * np.tanh(shocks))
        volatility_drag = 0.5 * (shocks**2) * 0.02
        h = drifted * np.exp(-volatility_drag) + 0.05 * weights
    return np.sum(h * weights, axis=-1)


def numpy_hoisted_portfolio_simulation(
    prices: np.ndarray,
    weights: np.ndarray,
    shocks: np.ndarray,
) -> np.ndarray:
    """Fairer NumPy baseline: same math with the loop-invariant work hoisted out of the loop."""
    h = prices.copy()
    h[:, 0] = np.clip(h[:, 0], 80.0, 120.0)
    growth = (1.0 + 0.015 * np.tanh(shocks)) * np.exp(-(0.5 * (shocks**2) * 0.02))
    offset = 0.05 * weights
    for _ in range(6):
        h *= growth
        h += offset
    return np.sum(h * weights, axis=-1)


def jax_portfolio_simulation(
    prices: jax.Array,
    weights: jax.Array,
    shocks: jax.Array,
) -> jax.Array:
    """Simulate multi-step compounded portfolio returns in JAX using pure ops."""
    # Functional out-of-place update via .at[].set()
    h = prices.at[:, 0].set(jnp.clip(prices[:, 0], 80.0, 120.0))
    for _ in range(6):
        drifted = h * (1.0 + 0.015 * jnp.tanh(shocks))
        volatility_drag = 0.5 * (shocks**2) * 0.02
        h = drifted * jnp.exp(-volatility_drag) + 0.05 * weights
    return jnp.sum(h * weights, axis=-1)


jax_jit_portfolio_simulation = jax.jit(jax_portfolio_simulation)


def _median_ms(fn: Callable[[], Any], n_iters: int) -> float:
    """Median wall time of `fn()` in ms (robust to noisy neighbours on a shared machine)."""
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times)


def run_benchmark(
    n_paths: int = 8192,
    n_assets: int = 32,
    n_iters: int = 120,
) -> dict[str, Any]:
    """Run deterministic immutability check and NumPy vs JAX simulation benchmark."""
    # 1. Verify immutability vs functional .at[].set()
    base_arr = jnp.array([10.0, 20.0, 30.0, 40.0], dtype=jnp.float32)
    immutability_error_raised = False
    try:
        base_arr[1] = 99.0  # type: ignore[index]
    except TypeError:
        immutability_error_raised = True

    updated_arr = base_arr.at[1].set(99.0)
    original_unmodified = bool(float(base_arr[1]) == 20.0 and float(updated_arr[1]) == 99.0)

    # 2. Prepare deterministic inputs for portfolio simulation
    rng = np.random.default_rng(42)
    prices_np = rng.uniform(90.0, 110.0, size=(n_paths, n_assets)).astype(np.float32)
    weights_np = (rng.dirichlet(np.ones(n_assets), size=(n_paths,))).astype(np.float32)
    shocks_np = rng.standard_normal(size=(n_paths, n_assets)).astype(np.float32)

    prices_jax = jnp.asarray(prices_np)
    weights_jax = jnp.asarray(weights_np)
    shocks_jax = jnp.asarray(shocks_np)

    # Warmup
    out_np = numpy_portfolio_simulation(prices_np, weights_np, shocks_np)
    out_hoisted = numpy_hoisted_portfolio_simulation(prices_np, weights_np, shocks_np)
    out_eager = jax_portfolio_simulation(prices_jax, weights_jax, shocks_jax).block_until_ready()
    out_jit = jax_jit_portfolio_simulation(prices_jax, weights_jax, shocks_jax).block_until_ready()

    max_diff_np_jit = float(np.max(np.abs(out_np - np.asarray(out_jit))))
    max_diff_eager_jit = float(jnp.max(jnp.abs(out_eager - out_jit)))
    max_diff_hoisted_jit = float(np.max(np.abs(out_hoisted - np.asarray(out_jit))))

    # Median per-call time of each implementation
    numpy_args = (prices_np, weights_np, shocks_np)
    jax_args = (prices_jax, weights_jax, shocks_jax)
    numpy_ms = _median_ms(lambda: numpy_portfolio_simulation(*numpy_args), n_iters)
    numpy_hoisted_ms = _median_ms(lambda: numpy_hoisted_portfolio_simulation(*numpy_args), n_iters)
    jax_eager_ms = _median_ms(
        lambda: jax_portfolio_simulation(*jax_args).block_until_ready(), n_iters
    )
    jax_jit_ms = _median_ms(
        lambda: jax_jit_portfolio_simulation(*jax_args).block_until_ready(), n_iters
    )

    speedup_vs_numpy = numpy_ms / jax_jit_ms
    speedup_vs_eager = jax_eager_ms / jax_jit_ms
    speedup_vs_numpy_hoisted = numpy_hoisted_ms / jax_jit_ms
    eager_vs_numpy = numpy_ms / jax_eager_ms

    return {
        "experiment": "01_numpy_to_jax_arrays",
        "backend": jax.default_backend(),
        "shape": [n_paths, n_assets],
        "immutability_error_raised": immutability_error_raised,
        "original_unmodified": original_unmodified,
        "numpy_ms": round(numpy_ms, 4),
        "numpy_hoisted_ms": round(numpy_hoisted_ms, 4),
        "jax_eager_ms": round(jax_eager_ms, 4),
        "jax_jit_ms": round(jax_jit_ms, 4),
        "speedup_vs_numpy": round(speedup_vs_numpy, 2),
        "speedup_vs_eager": round(speedup_vs_eager, 2),
        "speedup_vs_numpy_hoisted": round(speedup_vs_numpy_hoisted, 2),
        "eager_vs_numpy": round(eager_vs_numpy, 2),
        "max_diff_np_jit": max_diff_np_jit,
        "max_diff_eager_jit": max_diff_eager_jit,
        "max_diff_hoisted_jit": max_diff_hoisted_jit,
    }


def print_report(res: dict[str, Any]) -> None:
    """Print the human-readable report for one run (also used by `verify_fundamentals.py --verbose`)."""
    print("=" * 72)
    print("Experiment 01: NumPy to jax.numpy & Functional Immutability")
    print("=" * 72)
    print(f"Backend Device             : {res['backend']}")
    print(f"Simulation Shape           : {tuple(res['shape'])} float32")
    print(f"In-Place Mutation Blocked  : {res['immutability_error_raised']} (TypeError raised)")
    print(f".at[].set() Pure Semantics : {res['original_unmodified']} (input buffer untouched)")
    print("-" * 72)
    print(f"NumPy Median Time          : {res['numpy_ms']:.4f} ms")
    print(f"NumPy Hoisted Median Time  : {res['numpy_hoisted_ms']:.4f} ms (invariants out of loop)")
    print(f"JAX Eager Median Time      : {res['jax_eager_ms']:.4f} ms")
    print(f"JAX @jit Fused Median Time : {res['jax_jit_ms']:.4f} ms")
    print(f"Speedup vs NumPy           : {res['speedup_vs_numpy']:.2f}x")
    print(f"Speedup vs JAX Eager       : {res['speedup_vs_eager']:.2f}x (both use .at[].set())")
    print(f"Speedup vs Hoisted NumPy   : {res['speedup_vs_numpy_hoisted']:.2f}x (fair baseline)")
    print(f"Eager .at[].set() vs NumPy : {res['eager_vs_numpy']:.2f}x (no jit, no fusion)")
    print(f"Max |NumPy - JAX @jit|     : {res['max_diff_np_jit']:.2e}")


def main() -> None:
    print_report(run_benchmark())


if __name__ == "__main__":
    main()
