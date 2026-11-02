"""Experiment 05: Deterministic, Batchable PRNG Keys (jax.random.key & split).

Demonstrates:
1. Why stateful/sequential random generators depend on execution order and break
   reproducibility when workers execute out of order or in parallel.
2. How JAX's counter-based Threefry PRNG (`jax.random.key`, `jax.random.split`)
   turns random state into an explicit value:
   - Reusing the same key without splitting is a deterministic antipattern
     (produces identical random draws).
   - Splitting keys (`jax.random.split(master_key, N)`) gives every worker its own
     stream, so the random bits do not depend on execution order. The script checks
     forward vs reversed order (bitwise identical) and loop vs batched `vmap` (equal up
     to float32 rounding between eager and compiled code).
3. Timing: `vmap` vectorizes the 2048 workers into one compiled batch (it does not spread
   work across cores by itself). Two baselines are reported: the naive one (an eager loop
   that indexes `worker_keys[i]` on every step) and a fairer one (a loop over the compiled
   single-worker function with the keys already split into a list).
"""

from __future__ import annotations

import statistics
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


batched_worker_rollouts = jax.jit(jax.vmap(simulate_worker_rollout))
single_worker_rollout_jit = jax.jit(simulate_worker_rollout)


def run_experiment(n_workers: int = 2048, n_iters: int = 100, repeats: int = 7) -> dict[str, Any]:
    """Verify PRNG key splitting semantics, order invariance, and batched vmap speedup."""
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

    # Batched execution via @jax.jit(jax.vmap)
    vmap_all = batched_worker_rollouts(worker_keys).block_until_ready()
    vmap_subset = vmap_all[:subset]

    max_diff_forward_reversed = float(jnp.max(jnp.abs(seq_forward - seq_reversed)))
    max_diff_sequential_vmap = float(jnp.max(jnp.abs(seq_forward - vmap_subset)))
    bitwise_order_invariant = bool(max_diff_forward_reversed == 0.0)

    # 3. Benchmark the full sequential Python loop over N workers vs batched vmap over N keys
    def sequential_loop() -> None:
        for i in range(n_workers):
            simulate_worker_rollout(worker_keys[i]).block_until_ready()

    loop_times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        sequential_loop()
        loop_times.append((time.perf_counter() - t0) * 1000.0)
    sequential_loop_ms = statistics.median(loop_times)

    # Fairer baseline: loop over the compiled single-worker function, keys pre-split
    key_list = list(worker_keys)
    single_worker_rollout_jit(key_list[0]).block_until_ready()  # warmup (compile)

    def compiled_loop() -> None:
        for key in key_list:
            single_worker_rollout_jit(key).block_until_ready()

    compiled_times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        compiled_loop()
        compiled_times.append((time.perf_counter() - t0) * 1000.0)
    compiled_loop_ms = statistics.median(compiled_times)

    vmap_times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        batched_worker_rollouts(worker_keys).block_until_ready()
        vmap_times.append((time.perf_counter() - t0) * 1000.0)
    vmap_batched_ms = statistics.median(vmap_times)

    speedup = sequential_loop_ms / vmap_batched_ms
    speedup_vs_compiled_loop = compiled_loop_ms / vmap_batched_ms

    return {
        "experiment": "05_deterministic_prng_keys",
        "backend": jax.default_backend(),
        "n_workers": n_workers,
        "reused_key_identical": reused_key_identical,
        "split_keys_distinct": split_keys_distinct,
        "bitwise_order_invariant": bitwise_order_invariant,
        "max_diff_forward_reversed": max_diff_forward_reversed,
        "max_diff_sequential_vmap": max_diff_sequential_vmap,
        "sequential_loop_ms": round(sequential_loop_ms, 4),
        "compiled_loop_ms": round(compiled_loop_ms, 4),
        "vmap_batched_ms": round(vmap_batched_ms, 4),
        "speedup": round(speedup, 2),
        "speedup_vs_compiled_loop": round(speedup_vs_compiled_loop, 2),
    }


def main() -> None:
    res = run_experiment()
    print("=" * 72)
    print("Experiment 05: Deterministic, Batchable PRNG Keys (Threefry)")
    print("=" * 72)
    print(f"Backend Device                     : {res['backend']}")
    print(f"Worker Streams                     : {res['n_workers']} independent keys")
    print(f"Reused Key Antipattern Identical   : {res['reused_key_identical']} (must split!)")
    print(f"Split Keys Produce Distinct Draws  : {res['split_keys_distinct']}")
    print(f"Order-Invariant (Fwd vs Reversed)  : {res['bitwise_order_invariant']} (diff={res['max_diff_forward_reversed']:.2e})")
    print(f"Max |Sequential - Batched vmap|    : {res['max_diff_sequential_vmap']:.2e}")
    print("-" * 72)
    print(f"Sequential Eager Loop Time         : {res['sequential_loop_ms']:.4f} ms")
    print(f"Loop over jit(single worker) Time  : {res['compiled_loop_ms']:.4f} ms (keys pre-split)")
    print(f"Batched @jit(vmap) PRNG Time       : {res['vmap_batched_ms']:.4f} ms")
    print(f"Speedup (Batched vmap vs Loop)     : {res['speedup']:.2f}x")
    print(f"Speedup vs Loop over jit(single)   : {res['speedup_vs_compiled_loop']:.2f}x (fair baseline)")


if __name__ == "__main__":
    main()
