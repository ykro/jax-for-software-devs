"""Experiment 05: Deterministic & Parallelizable PRNG Keys (jax.random.key & split).

Demonstrates:
1. Why stateful/sequential random generators depend on execution order and break
   reproducibility when workers execute out of order or in parallel.
2. How JAX's counter-based Threefry PRNG (`jax.random.key`, `jax.random.split`)
   turns random state into an explicit value:
   - Reusing the same key without splitting is a deterministic antipattern
     (produces identical random draws).
   - Splitting keys (`jax.random.split(master_key, N)`) guarantees bitwise-identical
     results across sequential loops, reversed execution order, and parallel `vmap`.
"""

from __future__ import annotations

import time
from typing import Any

import jax
import jax.numpy as jnp


def simulate_worker_rollout(worker_key: jax.Array) -> jax.Array:
    """Simulate a stochastic latency/throughput stress test for a single worker."""
    k_norm, k_exp = jax.random.split(worker_key, 2)
    jitter = jax.random.normal(k_norm, shape=(256,), dtype=jnp.float32)
    burst = jax.random.exponential(k_exp, shape=(256,), dtype=jnp.float32)
    service_curve = jnp.cumsum(0.8 * burst + 0.2 * jnp.abs(jitter))
    return jnp.mean(jnp.tanh(service_curve / 128.0))


parallel_worker_rollouts = jax.jit(jax.vmap(simulate_worker_rollout))


def run_experiment(n_workers: int = 2048, n_iters: int = 100) -> dict[str, Any]:
    """Verify PRNG key splitting semantics, order invariance, and parallel vmap speedup."""
    master_key = jax.random.key(42)

    # 1. Antipattern demonstration: reusing the exact same key without split()
    draw_a = jax.random.normal(master_key, shape=(8,))
    draw_b = jax.random.normal(master_key, shape=(8,))
    reused_key_identical = bool(jnp.array_equal(draw_a, draw_b))

    # Proper key splitting produces independent streams
    k1, k2 = jax.random.split(master_key, 2)
    draw_split_1 = jax.random.normal(k1, shape=(8,))
    draw_split_2 = jax.random.normal(k2, shape=(8,))
    split_keys_distinct = bool(not jnp.array_equal(draw_split_1, draw_split_2))

    # 2. Split master key into N independent worker keys
    worker_keys = jax.random.split(master_key, n_workers)

    # Compare sequential forward order vs reversed execution order on a subset of 64 workers
    subset = 64
    sub_keys = worker_keys[:subset]
    seq_forward = jnp.stack([simulate_worker_rollout(sub_keys[i]) for i in range(subset)])
    seq_reversed_list = [simulate_worker_rollout(sub_keys[i]) for i in range(subset - 1, -1, -1)]
    seq_reversed = jnp.stack(list(reversed(seq_reversed_list)))

    # Parallel execution via @jax.jit(jax.vmap)
    vmap_all = parallel_worker_rollouts(worker_keys).block_until_ready()
    vmap_subset = vmap_all[:subset]

    max_diff_forward_reversed = float(jnp.max(jnp.abs(seq_forward - seq_reversed)))
    max_diff_sequential_vmap = float(jnp.max(jnp.abs(seq_forward - vmap_subset)))
    bitwise_order_invariant = bool(max_diff_forward_reversed == 0.0)

    # 3. Benchmark sequential Python loop over N workers vs parallel vmap over N keys
    # Measure 64-worker loop and scale linearly to n_workers to keep runtime < 1s
    t0 = time.perf_counter()
    for i in range(subset):
        _ = simulate_worker_rollout(sub_keys[i]).block_until_ready()
    loop_subset_ms = (time.perf_counter() - t0) * 1000.0
    estimated_full_loop_ms = loop_subset_ms * (n_workers / subset)

    t0 = time.perf_counter()
    for _ in range(n_iters):
        _ = parallel_worker_rollouts(worker_keys).block_until_ready()
    vmap_parallel_ms = ((time.perf_counter() - t0) / n_iters) * 1000.0

    speedup = estimated_full_loop_ms / vmap_parallel_ms

    return {
        "experiment": "05_deterministic_prng_keys",
        "backend": jax.default_backend(),
        "n_workers": n_workers,
        "reused_key_identical": reused_key_identical,
        "split_keys_distinct": split_keys_distinct,
        "bitwise_order_invariant": bitwise_order_invariant,
        "max_diff_forward_reversed": max_diff_forward_reversed,
        "max_diff_sequential_vmap": max_diff_sequential_vmap,
        "sequential_loop_ms": round(estimated_full_loop_ms, 4),
        "vmap_parallel_ms": round(vmap_parallel_ms, 4),
        "speedup": round(speedup, 2),
    }


def main() -> None:
    res = run_experiment()
    print("=" * 72)
    print("Experiment 05: Deterministic & Parallelizable PRNG Keys (Threefry)")
    print("=" * 72)
    print(f"Backend Device                     : {res['backend']}")
    print(f"Parallel Worker Streams            : {res['n_workers']} independent keys")
    print(f"Reused Key Antipattern Identical   : {res['reused_key_identical']} (must split!)")
    print(f"Split Keys Produce Distinct Draws  : {res['split_keys_distinct']}")
    print(f"Order-Invariant (Fwd vs Reversed)  : {res['bitwise_order_invariant']} (diff={res['max_diff_forward_reversed']:.2e})")
    print(f"Max |Sequential - Parallel vmap|   : {res['max_diff_sequential_vmap']:.2e}")
    print("-" * 72)
    print(f"Sequential Worker Loop Time (est.) : {res['sequential_loop_ms']:.4f} ms")
    print(f"Parallel @jit(vmap) PRNG Time      : {res['vmap_parallel_ms']:.4f} ms")
    print(f"Speedup (Parallel vmap vs Loop)    : {res['speedup']:.2f}x")


if __name__ == "__main__":
    main()
