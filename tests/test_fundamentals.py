"""Automated pytest suite verifying all 5 JAX for Software Devs experiments."""

from __future__ import annotations

import importlib

exp01 = importlib.import_module("experiments.01_numpy_to_jax_arrays")
exp02 = importlib.import_module("experiments.02_jit_tracing_vs_execution")
exp03 = importlib.import_module("experiments.03_grad_as_code_transformation")
exp04 = importlib.import_module("experiments.04_vmap_goodbye_for_loops")
exp05 = importlib.import_module("experiments.05_deterministic_prng_keys")


def test_01_numpy_to_jax_arrays() -> None:
    res = exp01.run_benchmark(n_paths=4096, n_assets=32, n_iters=60)
    assert res["immutability_error_raised"] is True
    assert res["original_unmodified"] is True
    assert res["max_diff_np_jit"] < 1e-4
    assert res["max_diff_eager_jit"] < 1e-4
    assert res["max_diff_hoisted_jit"] < 1e-4
    assert res["numpy_hoisted_ms"] > 0.0
    assert res["speedup_vs_eager"] >= 2.0


def test_02_jit_tracing_vs_execution() -> None:
    res = exp02.run_experiment(n_elements=32768, n_iters=60)
    assert res["trace_count_for_3_same_shape_calls"] == 1
    assert res["trace_count_after_new_shape"] == 2
    assert res["jaxpr_num_equations"] >= 10
    assert res["stablehlo_lines"] >= 15
    assert "tanh" in res["jaxpr_text"]
    assert "stablehlo" in res["stablehlo_text"]
    assert res["max_diff"] < 1e-5
    assert abs(res["lax_cond_l2_norm"] - 1.0) < 1e-3
    assert abs(res["lax_cond_linf_max"] - 1.0) < 1e-3


def test_03_grad_as_code_transformation() -> None:
    res = exp03.run_benchmark(n_samples=1024, n_params=32, n_steps=20, n_iters=40)
    assert res["grad_max_diff_vs_finite_diff"] < 1e-4
    assert res["finite_diff_jit_max_diff_vs_exact"] < 1e-4
    assert res["finite_diff_jit_ms"] > 0.0
    assert res["final_loss"] < res["initial_loss"]
    assert res["loss_reduction_ratio"] >= 5.0
    assert res["speedup"] >= 5.0


def test_04_vmap_goodbye_for_loops() -> None:
    res = exp04.run_benchmark(batch_size=5000, feature_dim=16, n_iters_vmap=50, n_iters_loop=3)
    assert res["max_diff"] < 1e-5
    assert res["max_diff_vectorized"] < 1e-5
    assert res["speedup"] >= 20.0


def test_05_deterministic_prng_keys() -> None:
    res = exp05.run_experiment(n_workers=1024, n_iters=50, repeats=1)
    assert res["reused_key_identical"] is True
    assert res["split_keys_distinct"] is True
    assert res["bitwise_order_invariant"] is True
    assert res["max_diff_sequential_vmap"] < 1e-6
    assert res["compiled_loop_ms"] > 0.0
    assert res["speedup"] >= 10.0
