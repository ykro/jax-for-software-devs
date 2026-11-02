"""The Outside Judge: Deterministic Verification & Benchmark Runner for Software Devs.

Executes all 5 JAX fundamentals experiments, verifies numerical parity,
immutability guarantees, tracing cache behavior, autodiff accuracy vs finite
differences, `vmap` batching speedup, and counter-based PRNG determinism, and
writes `benchmarks/latest_results.json` and `benchmarks/terminal_output.txt`.

The checks (parity, immutability, trace counts, determinism) are deterministic. The
timings are medians per call after a warmup and vary run to run; each speedup row also
reports a second, fairer baseline (see "Cómo leer estas cifras" in the README).
"""

from __future__ import annotations

import importlib
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

import jax
import numpy as np

exp01 = importlib.import_module("experiments.01_numpy_to_jax_arrays")
exp02 = importlib.import_module("experiments.02_jit_tracing_vs_execution")
exp03 = importlib.import_module("experiments.03_grad_as_code_transformation")
exp04 = importlib.import_module("experiments.04_vmap_goodbye_for_loops")
exp05 = importlib.import_module("experiments.05_deterministic_prng_keys")


def _cpu_model() -> str:
    """Human-readable CPU model (platform.processor() only says 'arm' on macOS)."""
    try:
        if platform.system() == "Darwin":
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
            )
            return out.stdout.strip()
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or "unknown"


def _load_average() -> list[float]:
    try:
        return [round(x, 2) for x in os.getloadavg()]
    except OSError:
        return []


def run_all_verifications() -> dict[str, Any]:
    """Run all 5 experiments and enforce deterministic verification assertions."""
    load_before = _load_average()
    r01 = exp01.run_benchmark()
    r02 = exp02.run_experiment()
    # Keep the JSON report compact: the IR dumps are printed by `02_*.py --show-ir`
    r02.pop("jaxpr_text", None)
    r02.pop("stablehlo_text", None)
    r03 = exp03.run_benchmark()
    r04 = exp04.run_benchmark()
    r05 = exp05.run_experiment()

    # Assertions for Experiment 01: NumPy to jax.numpy & Immutability
    assert r01["immutability_error_raised"] is True, "Exp01 must block in-place mutation"
    assert r01["original_unmodified"] is True, "Exp01 .at[].set() must preserve original buffer"
    assert r01["max_diff_np_jit"] < 1e-4, f"Exp01 parity failed: {r01['max_diff_np_jit']}"
    assert r01["max_diff_hoisted_jit"] < 1e-4, (
        f"Exp01 hoisted NumPy parity failed: {r01['max_diff_hoisted_jit']}"
    )
    assert r01["speedup_vs_eager"] >= 2.0, f"Exp01 speedup too low: {r01['speedup_vs_eager']}x"

    # Assertions for Experiment 02: @jax.jit Tracing vs Execution
    assert r02["trace_count_for_3_same_shape_calls"] == 1, (
        f"Exp02 traced more than once for same shape: {r02['trace_count_for_3_same_shape_calls']}"
    )
    assert r02["trace_count_after_new_shape"] == 2, "Exp02 should re-trace on new shape"
    assert r02["max_diff"] < 1e-5, f"Exp02 parity failed: {r02['max_diff']}"
    assert r02["speedup"] >= 1.5, f"Exp02 speedup too low: {r02['speedup']}x"

    # Assertions for Experiment 03: jax.grad as Code Transformation
    assert r03["grad_max_diff_vs_finite_diff"] < 1e-4, (
        f"Exp03 grad parity vs finite diff failed: {r03['grad_max_diff_vs_finite_diff']}"
    )
    assert r03["loss_reduction_ratio"] >= 5.0, (
        f"Exp03 calibration did not converge enough: {r03['loss_reduction_ratio']}x"
    )
    assert r03["finite_diff_jit_max_diff_vs_exact"] < 1e-4, (
        f"Exp03 compiled finite diff parity failed: {r03['finite_diff_jit_max_diff_vs_exact']}"
    )
    assert r03["speedup"] >= 5.0, f"Exp03 autodiff speedup < 5.0x: {r03['speedup']}x"

    # Assertions for Experiment 04: Goodbye for-loops with jax.vmap
    assert r04["max_diff"] < 1e-5, f"Exp04 vmap parity failed: {r04['max_diff']}"
    assert r04["max_diff_vectorized"] < 1e-5, (
        f"Exp04 vectorized NumPy parity failed: {r04['max_diff_vectorized']}"
    )
    assert r04["speedup"] >= 20.0, f"Exp04 vmap speedup < 20.0x: {r04['speedup']}x"

    # Assertions for Experiment 05: Deterministic PRNG Keys
    assert r05["reused_key_identical"] is True, "Exp05 reused key must be identical"
    assert r05["split_keys_distinct"] is True, "Exp05 split keys must be distinct"
    assert r05["bitwise_order_invariant"] is True, "Exp05 PRNG must be order-invariant"
    assert r05["max_diff_sequential_vmap"] < 1e-6, (
        f"Exp05 sequential vs vmap diff failed: {r05['max_diff_sequential_vmap']}"
    )
    assert r05["speedup"] >= 10.0, f"Exp05 batched PRNG speedup < 10.0x: {r05['speedup']}x"

    summary = {
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_model": _cpu_model(),
            "cpu_count": os.cpu_count(),
            "load_average_before": load_before,
            "load_average_after": _load_average(),
            "timing": "median ms per call after one warmup call; varies run to run",
            "jax_version": jax.__version__,
            "numpy_version": np.__version__,
            "jax_backend": jax.default_backend(),
            "jax_device": str(jax.devices()[0]),
        },
        "experiments": {
            "01_numpy_to_jax_arrays": r01,
            "02_jit_tracing_vs_execution": r02,
            "03_grad_as_code_transformation": r03,
            "04_vmap_goodbye_for_loops": r04,
            "05_deterministic_prng_keys": r05,
        },
        "all_verifications_passed": True,
    }
    return summary


def main() -> None:
    summary = run_all_verifications()
    bench_dir = Path(__file__).resolve().parent / "benchmarks"
    bench_dir.mkdir(parents=True, exist_ok=True)
    out_json = bench_dir / "latest_results.json"
    out_txt = bench_dir / "terminal_output.txt"
    out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    env = summary["environment"]
    e01 = summary["experiments"]["01_numpy_to_jax_arrays"]
    e02 = summary["experiments"]["02_jit_tracing_vs_execution"]
    e03 = summary["experiments"]["03_grad_as_code_transformation"]
    e04 = summary["experiments"]["04_vmap_goodbye_for_loops"]
    e05 = summary["experiments"]["05_deterministic_prng_keys"]

    def ms(value: float) -> str:
        return f"{value} ms"

    def load1(values: list[float]) -> str:
        return f"{values[0]:.2f}" if values else "n/a"

    rows = [
        (
            "01. jit vs eager (portfolio sim)",
            ms(e01["jax_eager_ms"]),
            ms(e01["jax_jit_ms"]),
            f"{e01['speedup_vs_eager']}x (PASS)",
            f"{e01['speedup_vs_numpy_hoisted']}x vs hoisted NumPy",
        ),
        (
            "02. Cached @jit vs NumPy",
            ms(e02["numpy_ms"]),
            ms(e02["cached_jit_ms"]),
            f"{e02['speedup']}x (1 trace)",
            "same (already vs NumPy)",
        ),
        (
            "03. value_and_grad vs Finite Diff",
            ms(e03["finite_diff_ms"]),
            ms(e03["jax_value_and_grad_ms"]),
            f"{e03['speedup']}x (PASS)",
            f"{e03['speedup_vs_compiled_fd']}x vs jit finite diff",
        ),
        (
            "04. vmap vs for-loop (10k records)",
            ms(e04["for_loop_ms"]),
            ms(e04["vmap_jit_ms"]),
            f"{e04['speedup']}x (PASS)",
            f"{e04['speedup_vs_numpy_vectorized']}x vs vectorized NumPy",
        ),
        (
            "05. Batched Threefry keys (vmap)",
            ms(e05["sequential_loop_ms"]),
            ms(e05["vmap_batched_ms"]),
            f"{e05['speedup']}x (Δ=0.0)",
            f"{e05['speedup_vs_compiled_loop']}x vs loop over jit",
        ),
    ]
    width = 110
    header = (
        f"{'Experiment':<34} | {'Baseline':<13} | {'JAX/OpenXLA':<11} | "
        f"{'Speedup / Check':<15} | {'Fair baseline':<25}"
    )
    lines = [
        "=" * width,
        "THE OUTSIDE JUDGE — JAX for Software Devs Verification Report",
        "=" * width,
        (
            f"Runtime : Python {env['python_version']} | JAX {env['jax_version']} | "
            f"NumPy {env['numpy_version']} | Backend: {env['jax_backend']}"
        ),
        (
            f"CPU     : {env['cpu_model']} ({env['cpu_count']} cores) | 1-min load average "
            f"{load1(env['load_average_before'])} before, {load1(env['load_average_after'])} after"
        ),
        "-" * width,
        header,
        "-" * width,
    ]
    for name, base, jax_t, speed, fair in rows:
        lines.append(f"{name:<34} | {base:<13} | {jax_t:<11} | {speed:<15} | {fair:<25}")
    lines += [
        "-" * width,
        "Times are medians per call after a warmup and vary run to run (about ±15-20%).",
        f"Saved JSON report to: {out_json.relative_to(Path(__file__).resolve().parent)}",
        "ALL DETERMINISTIC CHECKS PASSED.",
    ]
    report = "\n".join(lines)
    out_txt.write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
