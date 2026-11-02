"""Experiment 04: Goodbye for-loops with jax.vmap.

Demonstrates:
1. Writing clean, maintainable domain logic for a SINGLE 1D record (`score_single_transaction`)
   without manual batch dimensions or broadcasting gymnastics.
2. Transforming that single-record function into a parallel batch processor over 10,000
   records with `jax.jit(jax.vmap(score_single_transaction, in_axes=(0, None, None)))`.
3. Measuring the speedup of compiled `vmap` vs a Python `for-loop`.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


def score_single_transaction(
    tx_features: jax.Array,
    reference_profile: jax.Array,
    risk_matrix: jax.Array,
) -> jax.Array:
    """Compute an anomaly risk score for a single 1D transaction vector `(D,)`."""
    deviation = tx_features - reference_profile
    projected = risk_matrix @ deviation
    mahalanobis_like = jnp.sqrt(jnp.dot(projected, projected) + 1e-6)
    cosine_sim = jnp.dot(tx_features, reference_profile) / (
        jnp.linalg.norm(tx_features) * jnp.linalg.norm(reference_profile) + 1e-6
    )
    raw_score = 0.65 * mahalanobis_like - 1.25 * cosine_sim
    return jax.nn.sigmoid(raw_score)


# Automatically vectorize across the 0-th axis of tx_features while sharing reference & matrix
batch_score_transactions = jax.jit(
    jax.vmap(score_single_transaction, in_axes=(0, None, None))
)


def _median_ms(fn: Callable[[], Any], n_iters: int) -> float:
    """Median wall time of `fn()` in ms (robust to noisy neighbours on a shared machine)."""
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times)


def python_for_loop_scoring(
    batch_tx: np.ndarray,
    ref_profile: np.ndarray,
    risk_mat: np.ndarray,
) -> np.ndarray:
    """Baseline Python for-loop scoring 10,000 transactions one by one."""
    n_records = batch_tx.shape[0]
    scores = np.empty(n_records, dtype=np.float32)
    ref_norm = float(np.linalg.norm(ref_profile))
    for i in range(n_records):
        tx = batch_tx[i]
        dev = tx - ref_profile
        proj = risk_mat @ dev
        maha = np.sqrt(np.dot(proj, proj) + 1e-6)
        cos = np.dot(tx, ref_profile) / (np.linalg.norm(tx) * ref_norm + 1e-6)
        raw = 0.65 * maha - 1.25 * cos
        scores[i] = 1.0 / (1.0 + np.exp(-raw))
    return scores


def run_benchmark(
    batch_size: int = 10000,
    feature_dim: int = 16,
    n_iters_vmap: int = 100,
    n_iters_loop: int = 5,
) -> dict[str, Any]:
    """Benchmark Python for-loop vs compiled `jax.vmap` over 10,000 records."""
    rng = np.random.default_rng(2026)
    batch_tx_np = rng.standard_normal((batch_size, feature_dim)).astype(np.float32)
    ref_np = rng.uniform(-0.5, 0.5, size=(feature_dim,)).astype(np.float32)
    risk_mat_np = (rng.standard_normal((feature_dim, feature_dim)) / np.sqrt(feature_dim)).astype(
        np.float32
    )

    batch_tx_jax = jnp.asarray(batch_tx_np)
    ref_jax = jnp.asarray(ref_np)
    risk_mat_jax = jnp.asarray(risk_mat_np)

    # Warmup both
    scores_loop = python_for_loop_scoring(batch_tx_np, ref_np, risk_mat_np)
    scores_vmap = batch_score_transactions(batch_tx_jax, ref_jax, risk_mat_jax).block_until_ready()

    max_diff = float(np.max(np.abs(scores_loop - np.asarray(scores_vmap))))

    # Median per-call time: Python for-loop vs compiled jax.vmap
    for_loop_ms = _median_ms(
        lambda: python_for_loop_scoring(batch_tx_np, ref_np, risk_mat_np), n_iters_loop
    )
    vmap_ms = _median_ms(
        lambda: batch_score_transactions(batch_tx_jax, ref_jax, risk_mat_jax).block_until_ready(),
        n_iters_vmap,
    )

    speedup = for_loop_ms / vmap_ms

    return {
        "experiment": "04_vmap_goodbye_for_loops",
        "backend": jax.default_backend(),
        "batch_size": batch_size,
        "feature_dim": feature_dim,
        "for_loop_ms": round(for_loop_ms, 4),
        "vmap_jit_ms": round(vmap_ms, 4),
        "speedup": round(speedup, 2),
        "max_diff": max_diff,
    }


def main() -> None:
    res = run_benchmark()
    print("=" * 72)
    print("Experiment 04: Goodbye for-loops with jax.vmap (10,000 Records)")
    print("=" * 72)
    print(f"Backend Device                : {res['backend']}")
    print(f"Batch Size & Feature Dim      : ({res['batch_size']}, {res['feature_dim']}) float32")
    print("-" * 72)
    print(f"Python for-loop Median Time   : {res['for_loop_ms']:.4f} ms")
    print(f"JAX @jit(vmap(f)) Median Time : {res['vmap_jit_ms']:.4f} ms")
    print(f"Speedup vs Python for-loop    : {res['speedup']:.2f}x")
    print(f"Max |for-loop - vmap|         : {res['max_diff']:.2e}")


if __name__ == "__main__":
    main()
