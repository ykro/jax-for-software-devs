# JAX para Software Devs: De Python y NumPy a Cómputo Acelerado con OpenXLA

- **Por qué es relevante:** JAX permite escribir funciones matemáticas en Python con la misma API de `numpy` y transformarlas mediante un compilador (`@jax.jit`, `jax.grad`, `jax.vmap`) en binarios nativos de OpenXLA que corren en CPU, GPU y Cloud TPU sin escribir un solo kernel en C++ o CUDA.
- **Para quién es:** Ingenieros de software, desarrolladores backend y científicos de datos que dominan Python y NumPy, pero nunca han usado JAX ni necesitan venir del mundo de PyTorch o Deep Learning.
- **Quién debería verlo:** Desarrolladores que construyen simulaciones de Montecarlo, motores de riesgo o precios, calibración de sensores, optimización numérica o procesamiento de datos por `batches` y buscan eliminar `for-loops` lentos de Python.
- **Serie completa:** Punto de entrada de la Serie B [JAX & ML for Software Devs Field Guide](https://github.com/ykro/jax-ml-for-software-devs-field-guide) (y compañero de la Serie A [JAX & OpenXLA Field Guide](https://github.com/ykro/jax-openxla-field-guide)).

---

## Arquitectura y Modelo Mental

### 1. De `numpy` a `jax.numpy` e Inmutabilidad Funcional (`.at[].set()`)

En NumPy, una asignación como `arr[:, 0] = val` muta el buffer en RAM in-place y arriesga efectos secundarios si otra función comparte la referencia. En JAX, los arrays son inmutables por diseño para que el compilador OpenXLA pueda fusionar operaciones y reutilizar buffers sin race conditions. Las actualizaciones se expresan de forma pura con `arr.at[:, 0].set(val)`. El contrato de pureza es lo que habilita la optimización, pero la velocidad la da `@jax.jit`. En el experimento 01, JAX eager con el mismo `.at[].set()` apenas le gana a NumPy, y la versión compilada es mucho más rápida porque XLA fusiona la función en 3 kernels y calcula una sola vez el `tanh` y el `exp` que NumPy repite en cada paso del loop.

![Inmutabilidad funcional de JAX frente a mutación in-place de NumPy](diagrams/01-numpy-vs-jax-immutability.svg)

### 2. Cómo Piensa `@jax.jit`: Tracing vs Ejecución Compilada

Cuando decoras una función con `@jax.jit`, la primera llamada con una firma `(shape, dtype)` ejecuta el código Python pasando objetos abstractos `ShapedArray` para grabar las operaciones en un grafo `Jaxpr` y compilarlo a `StableHLO`. En las llamadas subsecuentes con la misma forma geométrica, JAX salta el intérprete de Python y hace `dispatch` directo al binario XLA en caché.

![Modelo mental de tracing y ejecución en @jax.jit](diagrams/02-jit-tracing-mental-model.svg)

### 3. Transformaciones Componibles: `jit(vmap(value_and_grad(f)))`

En lugar de programar derivadas manuales o `for-loops` para recorrer un `batch`, en JAX escribes una función escalar limpia para un solo registro (`1D`) y compones transformaciones de orden superior como `jax.value_and_grad`, `jax.vmap` y `@jax.jit` en una sola línea.

![Pipeline de transformaciones componibles en JAX](diagrams/03-composable-transforms-pipeline.svg)

---

## Resultados Reales de Verificación (`verify_fundamentals.py`)

Corrida con Python `3.12.11`, JAX `0.12.0.1` y NumPy `2.5.4` en el CPU de una laptop (Apple M1 Max, 10 núcleos, sin GPU), con un load average de 7.6 al empezar y 7.5 al terminar. Las **verificaciones** son deterministas (paridad numérica, inmutabilidad, conteo de traces y determinismo del PRNG) y dan lo mismo en cada corrida. Los **tiempos** son la mediana por llamada después de una llamada de calentamiento y cambian de una corrida a otra; el juez solo exige mínimos holgados de speedup. Cada fila trae dos speedups, contra el código ingenuo y contra un baseline justo (ver [Cómo leer estas cifras](#cómo-leer-estas-cifras)).

| Experimento | Archivo | Baseline ingenuo | JAX + OpenXLA | Speedup | Contra el baseline justo |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **01. `jit` vs eager (simulación de portafolio)** | `experiments/01_numpy_to_jax_arrays.py` | `4.4308 ms` (JAX eager, también con `.at[].set()`) / `6.1670 ms` (NumPy) | `0.3467 ms` (`@jax.jit`) | **12.78x** vs eager / **17.79x** vs NumPy (`max_diff = 1.53e-05`) | **4.25x** vs NumPy con lo invariante fuera del loop (`1.4747 ms`) |
| **02. `@jit` en caché vs NumPy** | `experiments/02_jit_tracing_vs_execution.py` | `0.5132 ms` (NumPy) | `0.1020 ms` (`@jit` en caché) | **5.03x** (`1 trace` para 3 llamadas con el mismo `shape`, `2` con un `shape` nuevo; Jaxpr de `15` ecuaciones de primer nivel, `61` líneas de StableHLO) | Igual (ya compara contra NumPy) |
| **03. `value_and_grad` vs diferencias finitas** | `experiments/03_grad_as_code_transformation.py` | `1.8369 ms` (diferencias finitas en NumPy, 64 evaluaciones) | `0.0533 ms` (`jit(value_and_grad)`) | **34.48x** (`loss` baja **376.48x** en 20 steps) | **6.47x** vs diferencias finitas compiladas con `jit(vmap)` (`0.3449 ms`) |
| **04. `vmap` vs for-loop (10,000 registros)** | `experiments/04_vmap_goodbye_for_loops.py` | `33.5893 ms` (`for-loop` de Python) | `0.1571 ms` (`jit(vmap)`) | **213.77x** (`max_diff = 1.19e-07`) | **2.04x** vs NumPy vectorizado a mano (`0.3204 ms`) |
| **05. Keys Threefry en batch (`vmap`)** | `experiments/05_deterministic_prng_keys.py` | `419.0313 ms` (loop eager completo, 2048 workers) | `8.9766 ms` (`jit(vmap)` sobre 2048 keys) | **46.68x** (orden-invariante, `Δ = 0.0`) | **11.03x** vs loop sobre la función compilada de un worker (`98.9980 ms`) |

Salida íntegra de terminal generada por `verify_fundamentals.py` (la misma corrida que `benchmarks/latest_results.json`):

```text
==============================================================================================================
THE OUTSIDE JUDGE — JAX for Software Devs Verification Report
==============================================================================================================
Runtime : Python 3.12.11 | JAX 0.12.0.1 | NumPy 2.5.4 | Backend: cpu
CPU     : Apple M1 Max (10 cores) | 1-min load average 7.58 before, 7.45 after
--------------------------------------------------------------------------------------------------------------
Experiment                         | Baseline      | JAX/OpenXLA | Speedup / Check | Fair baseline            
--------------------------------------------------------------------------------------------------------------
01. jit vs eager (portfolio sim)   | 4.4308 ms     | 0.3467 ms   | 12.78x (PASS)   | 4.25x vs hoisted NumPy   
02. Cached @jit vs NumPy           | 0.5132 ms     | 0.102 ms    | 5.03x (1 trace) | same (already vs NumPy)  
03. value_and_grad vs Finite Diff  | 1.8369 ms     | 0.0533 ms   | 34.48x (PASS)   | 6.47x vs jit finite diff 
04. vmap vs for-loop (10k records) | 33.5893 ms    | 0.1571 ms   | 213.77x (PASS)  | 2.04x vs vectorized NumPy
05. Batched Threefry keys (vmap)   | 419.0313 ms   | 8.9766 ms   | 46.68x (Δ=0.0)  | 11.03x vs loop over jit  
--------------------------------------------------------------------------------------------------------------
Times are medians per call after a warmup and vary run to run (about ±15-20%).
Saved JSON report to: benchmarks/latest_results.json
ALL DETERMINISTIC CHECKS PASSED.
```

### Cómo leer estas cifras

- **Qué compara cada speedup.** La fila 01 compara `@jax.jit` contra JAX eager, y **los dos usan `.at[].set()`**, así que mide lo que aporta compilar. El speedup viene de que `jit` le entrega la función completa a XLA, que la fusiona en 3 kernels y calcula `tanh(shocks)` y `exp(...)` una sola vez en lugar de una vez por paso del loop. `.at[].set()` por sí solo no acelera, y JAX eager con el mismo `.at[].set()` apenas le gana 1.39x a NumPy en esta corrida. La 02 compara la llamada en caché contra NumPy. La 03 compara diferencias finitas en NumPy interpretado contra `value_and_grad` compilado, la 04 un `for-loop` de Python contra `jit(vmap)` y la 05 un loop eager que indexa `worker_keys[i]` en cada vuelta contra `jit(vmap)`.
- **Baselines justos.** Contra la mejor versión razonable sin la transformación, las ventajas son mucho menores y más útiles para decidir: unas **4.8x** de `jit` frente a NumPy con lo invariante fuera del loop, unas **5.8x** de autodiff frente a diferencias finitas también compiladas (con 32 parámetros; la ventaja crece con el número de parámetros), unas **2.2x** de `vmap` frente a NumPy vectorizado a mano (lo valioso es llegar ahí sin reescribir la función de un registro) y unas **12x** frente a un loop sobre la versión compilada de un worker. Son medianas de una corrida aparte con los scripts de sonda (con `nice` y un load average de 6 a 10 en los 10 núcleos), que coinciden con la columna «Contra el baseline justo» dentro del margen de variación.
- **Mediana por llamada después del warmup.** La primera llamada a una función con `@jax.jit` incluye tracing y compilación (unos 37 ms en el experimento 02) y no entra en las medianas.
- **CPU de una laptop.** Todo corre en CPU (Apple M1 Max, 10 núcleos), sin GPU ni TPU. Con estos tamaños, la diferencia entre XLA multihilo y NumPy de un hilo no explica los speedups (desactivar los hilos de XLA no cambia los tiempos de forma apreciable). `vmap` vectoriza en un solo batch compilado; no reparte trabajo entre núcleos por sí mismo.
- **Variación entre corridas.** En la misma máquina, los speedups cambian ±15–20 % de una corrida a otra (por ejemplo, la fila 01 contra NumPy ha dado entre 15x y 22x). Los dos decimales que imprime el juez no implican esa precisión.
- **Videos.** La tabla del juez que aparece en el [video de práctica](https://youtu.be/p1dv3q2INGQ) (13.78x / 5.16x / 32.39x / 207.02x / 47.52x) es una corrida anterior de este mismo código, antes de agregar los baselines justos. El [video de teoría](https://youtu.be/kIRbH33BGBo) usa cifras de una versión anterior que medía la media en lugar de la mediana y extrapolaba el loop del experimento 05.

---

## Estructura del Repositorio

```text
.
├── benchmarks/
│   ├── latest_results.json                  # Métricas verificadas por verify_fundamentals.py
│   └── terminal_output.txt                  # Salida íntegra del Outside Judge
├── diagrams/
│   ├── 01-numpy-vs-jax-immutability.{html,svg,png}
│   ├── 02-jit-tracing-mental-model.{html,svg,png}
│   └── 03-composable-transforms-pipeline.{html,svg,png}
├── experiments/
│   ├── 01_numpy_to_jax_arrays.py            # De NumPy a jax.numpy e inmutabilidad con .at[].set()
│   ├── 02_jit_tracing_vs_execution.py       # Tracing con ShapedArray, Jaxpr, StableHLO y lax.cond
│   ├── 03_grad_as_code_transformation.py    # Autodiff exacto vs diferencias finitas (sin Deep Learning)
│   ├── 04_vmap_goodbye_for_loops.py         # Vectorización automática de 10,000 registros sin for-loops
│   └── 05_deterministic_prng_keys.py        # PRNG Threefry determinista con split + vmap en un solo batch
├── gcp/
│   └── run_on_demand_benchmark.sh           # Runner on-demand con auto-teardown en Cloud TPU v5e y L4 GPU
├── tests/
│   └── test_fundamentals.py                 # Suite pytest (5 tests, verificaciones deterministas)
├── Makefile
├── pyproject.toml
└── verify_fundamentals.py                   # El Outside Judge (aserciones de paridad y speedup)
```

---

## Onboarding Paso a Paso

### 1. Instalación desde cero (macOS / Linux)

```bash
git clone https://github.com/ykro/jax-for-software-devs.git
cd jax-for-software-devs
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

### 2. Recorrido guiado por los 5 experimentos (Orden recomendado)

Cada script toma unos pocos segundos y explica un pilar del modelo mental de JAX para ingenieros de software:

```bash
# Paso 1 — De numpy a jax.numpy, inmutabilidad y actualizaciones puras con .at[].set()
python experiments/01_numpy_to_jax_arrays.py

# Paso 2 — Cómo piensa @jax.jit: ShapedArray tracing, Jaxpr, StableHLO y condicionales
# (--show-ir imprime además el Jaxpr completo y el inicio del módulo StableHLO)
python experiments/02_jit_tracing_vs_execution.py --show-ir

# Paso 3 — jax.grad como transformación de código: calibración de sensores vs diferencias finitas
python experiments/03_grad_as_code_transformation.py

# Paso 4 — Adiós a los for-loops: procesar un batch de 10,000 registros con jax.vmap
python experiments/04_vmap_goodbye_for_loops.py

# Paso 5 — Aleatoriedad determinista con jax.random.key y split, en batch con vmap
python experiments/05_deterministic_prng_keys.py
```

### 3. Verificación completa (Linter + Tests + Outside Judge)

```bash
# Ejecutar únicamente el Outside Judge y actualizar benchmarks/latest_results.json
python verify_fundamentals.py

# Ejecutar ruff, pytest (5 tests) y verify_fundamentals.py en un solo comando
make all
```

### 4. Ejecución on-demand con auto-teardown en Google Cloud (Cloud TPU v5e o NVIDIA L4 GPU)

El script `gcp/run_on_demand_benchmark.sh` aprovisiona instancias on-demand con auto-teardown en tu propio proyecto de Google Cloud (variable `GCP_PROJECT_ID`) y garantiza la limpieza automática de los recursos mediante `trap cleanup EXIT INT TERM`.

```bash
# Ejecutar benchmark en una instancia on-demand de Cloud TPU v5e-1 (us-central1-a)
export GCP_PROJECT_ID=<tu-proyecto>
make gcp-tpu

# Ejecutar benchmark en un Vertex AI Custom Job on-demand con GPU NVIDIA L4
make gcp-gpu
```
