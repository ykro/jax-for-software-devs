"""The snippets in practica/ are the ones typed in the practice video: each must run and print
what the video shows (the two *_roto.py files must fail with the error the video explains)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PRACTICA = Path(__file__).resolve().parent.parent / "practica"


def run(name: str) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, str(PRACTICA / name)]
    return subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=300)


def test_01_mutar_roto_raises_type_error() -> None:
    r = run("01_mutar_roto.py")
    assert r.returncode != 0
    assert "TypeError: JAX arrays are immutable" in r.stderr


def test_01_mutar_keeps_original() -> None:
    out = run("01_mutar.py").stdout
    assert "original: [10.  12.5  9.  11. ]" in out
    assert "nuevos:   [10. 13.  9. 11.]" in out


def test_02_trazar_traces_once_per_shape() -> None:
    lines = run("02_trazar.py").stdout.splitlines()
    assert lines == ["trazando con (30,) float32", "tres llamadas listas", "trazando con (60,) float32"]


def test_02_if_roto_raises_tracer_bool_error() -> None:
    r = run("02_if_roto.py")
    assert r.returncode != 0
    assert "TracerBoolConversionError" in r.stderr


def test_02_if_uses_where() -> None:
    lines = run("02_if.py").stdout.splitlines()
    assert lines == ["[ 0.01 -0.03  0.04]", "[0. 0. 0.]"]


def test_03_calibrar_converges() -> None:
    lines = run("03_calibrar.py").stdout.splitlines()
    assert lines[0].startswith("gradiente:")
    last = float(lines[-1].split("error")[1].split()[0])
    assert last < 1e-3


def test_04_vmap_scores_every_transaction() -> None:
    out = run("04_vmap.py").stdout
    assert "una: ()" in out
    assert "todas: (10000,)" in out


def test_05_llaves_same_key_same_numbers() -> None:
    lines = run("05_llaves.py").stdout.splitlines()
    first = lines[0].removeprefix("misma llave:").strip()
    a, b = first.split("] [")
    assert a + "]" == "[" + b
    assert lines[-1].startswith("4 workers:")


def test_06_componer_one_gradient_per_sensor() -> None:
    out = run("06_componer.py").stdout
    assert "un gradiente por sensor: (1000, 2)" in out
