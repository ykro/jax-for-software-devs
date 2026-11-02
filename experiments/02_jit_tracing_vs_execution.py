"""Experiment 02: How @jax.jit Thinks (Tracing vs Execution).

Run with `--show-ir` to also print the Jaxpr and the head of the StableHLO module.

Demonstrates:
1. Why Python side effects inside `@jax.jit` only execute once per `(shape, dtype)`
   during `ShapedArray` tracing.
2. How to inspect the intermediate representation (`jax.make_jaxpr`) and the
   OpenXLA `StableHLO` MLIR module.
3. Why data-dependent Python `if` statements fail on traced values, and how
   `jnp.where` and `jax.lax.cond` express branches inside compiled XLA kernels.
"""

from __future__ import annotations

import statistics
import sys
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


def run_experiment(n_elements: int = 65536, n_iters: int = 150) -> dict[str, Any]:
    """Run tracing inspection, cache hit count, and conditional branching benchmark."""
    trace_log: list[tuple[tuple[int, ...], str]] = []

    def raw_signal_filter(x: jax.Array, threshold: jax.Array) -> jax.Array:
        # Python side effect: only runs when JAX traces a new (shape, dtype)
        trace_log.append((x.shape, str(x.dtype)))
        centered = x - jnp.mean(x)
        scaled = centered * jax.lax.rsqrt(jnp.var(x) + 1e-5)
        # Data-dependent elementwise selection inside XLA via jnp.where
        clipped = jnp.where(jnp.abs(scaled) > threshold, jnp.sign(scaled) * threshold, scaled)
        return jnp.tanh(clipped) + 0.1 * scaled

    signal_filter = jax.jit(raw_signal_filter)

    rng = np.random.default_rng(7)
    x1 = jnp.asarray(rng.standard_normal(n_elements, dtype=np.float32))
    x2 = jnp.asarray(rng.standard_normal(n_elements, dtype=np.float32))
    x3 = jnp.asarray(rng.standard_normal(n_elements, dtype=np.float32))
    thresh = jnp.array(1.75, dtype=jnp.float32)

    # First call triggers tracing + XLA compilation
    t0 = time.perf_counter()
    y1 = signal_filter(x1, thresh).block_until_ready()
    first_call_ms = (time.perf_counter() - t0) * 1000.0

    # Subsequent calls with the same (shape, dtype) bypass Python tracing completely
    _ = signal_filter(x2, thresh).block_until_ready()
    _ = signal_filter(x3, thresh).block_until_ready()
    trace_count_same_shape = len(trace_log)

    # Calling with a new shape triggers exactly 1 new trace
    x_other_shape = jnp.asarray(rng.standard_normal(n_elements // 2, dtype=np.float32))
    _ = signal_filter(x_other_shape, thresh).block_until_ready()
    trace_count_after_new_shape = len(trace_log)

    # Inspect the Jaxpr of the very same function. make_jaxpr traces it once more, so the
    # trace_log side effect fires again; the counts above were already captured.
    closed_jaxpr = jax.make_jaxpr(raw_signal_filter)(x1, thresh)
    num_jaxpr_eqns = len(closed_jaxpr.jaxpr.eqns)
    lowered = signal_filter.lower(x1, thresh)
    stablehlo_text = lowered.as_text("stablehlo")
    stablehlo_lines = len(stablehlo_text.strip().splitlines())

    # Scalar branch with jax.lax.cond (true_fun vs false_fun)
    @jax.jit
    def dynamic_router(vec: jax.Array, use_l2_norm: jax.Array) -> jax.Array:
        return jax.lax.cond(
            use_l2_norm,
            lambda v: v / (jnp.linalg.norm(v) + 1e-6),
            lambda v: v / (jnp.max(jnp.abs(v)) + 1e-6),
            vec,
        )

    routed_l2 = dynamic_router(x1, jnp.array(True)).block_until_ready()
    routed_linf = dynamic_router(x1, jnp.array(False)).block_until_ready()
    l2_norm_check = float(jnp.linalg.norm(routed_l2))
    linf_max_check = float(jnp.max(jnp.abs(routed_linf)))

    # Benchmark cached @jit execution vs unfused Python/NumPy
    def numpy_signal_filter(x_np: np.ndarray, t_val: float) -> np.ndarray:
        c = x_np - np.mean(x_np)
        s = c / np.sqrt(np.var(x_np) + 1e-5)
        cl = np.where(np.abs(s) > t_val, np.sign(s) * t_val, s)
        return np.tanh(cl) + 0.1 * s

    x1_np = np.asarray(x1)
    y_np = numpy_signal_filter(x1_np, 1.75)
    max_diff = float(np.max(np.abs(y_np - np.asarray(y1))))

    def median_ms(fn: Any) -> float:
        times = []
        for _ in range(n_iters):
            t0 = time.perf_counter()
            fn()
            times.append((time.perf_counter() - t0) * 1000.0)
        return statistics.median(times)

    numpy_ms = median_ms(lambda: numpy_signal_filter(x1_np, 1.75))
    cached_jit_ms = median_ms(lambda: signal_filter(x1, thresh).block_until_ready())

    speedup = numpy_ms / cached_jit_ms

    return {
        "experiment": "02_jit_tracing_vs_execution",
        "backend": jax.default_backend(),
        "n_elements": n_elements,
        "trace_count_for_3_same_shape_calls": trace_count_same_shape,
        "trace_count_after_new_shape": trace_count_after_new_shape,
        "jaxpr_num_equations": num_jaxpr_eqns,
        "stablehlo_lines": stablehlo_lines,
        "first_call_compile_ms": round(first_call_ms, 3),
        "numpy_ms": round(numpy_ms, 4),
        "cached_jit_ms": round(cached_jit_ms, 4),
        "speedup": round(speedup, 2),
        "max_diff": max_diff,
        "lax_cond_l2_norm": round(l2_norm_check, 5),
        "lax_cond_linf_max": round(linf_max_check, 5),
        "jaxpr_text": str(closed_jaxpr),
        "stablehlo_text": stablehlo_text,
    }


def main() -> None:
    show_ir = "--show-ir" in sys.argv[1:]
    res = run_experiment()
    print("=" * 72)
    print("Experiment 02: How @jax.jit Thinks (Tracing vs Cached Execution)")
    print("=" * 72)
    print(f"Backend Device                   : {res['backend']}")
    print(f"Input Vector Length              : {res['n_elements']} float32")
    print(
        f"Traces on 3 Same-Shape Calls     : {res['trace_count_for_3_same_shape_calls']} "
        "(Python body ran once)"
    )
    print(f"Traces After New Shape           : {res['trace_count_after_new_shape']}")
    print(f"Jaxpr Top-Level Equations        : {res['jaxpr_num_equations']} eqns")
    print(f"Lowered StableHLO Lines          : {res['stablehlo_lines']} lines")
    print(
        f"lax.cond Branches (L2 / L-inf)   : "
        f"norm={res['lax_cond_l2_norm']:.4f}, max={res['lax_cond_linf_max']:.4f}"
    )
    print("-" * 72)
    print(f"1st Call (Trace + XLA Compile)   : {res['first_call_compile_ms']:.3f} ms")
    print(f"NumPy Eager Median Time          : {res['numpy_ms']:.4f} ms")
    print(f"Cached @jit Median Time          : {res['cached_jit_ms']:.4f} ms")
    print(f"Speedup (Cached @jit vs NumPy)   : {res['speedup']:.2f}x")
    print(f"Max |NumPy - JAX @jit|           : {res['max_diff']:.2e}")
    if show_ir:
        print("=" * 72)
        print(f"Jaxpr ({res['jaxpr_num_equations']} top-level equations)")
        print("=" * 72)
        print(res["jaxpr_text"])
        print("=" * 72)
        print(f"StableHLO (first 14 of {res['stablehlo_lines']} lines)")
        print("=" * 72)
        print("\n".join(res["stablehlo_text"].strip().splitlines()[:14]))


if __name__ == "__main__":
    main()
