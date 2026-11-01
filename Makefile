PYTHON ?= $(shell if [ -x ../../.venv-shared/bin/python ]; then echo ../../.venv-shared/bin/python; else command -v python3; fi)
PYTEST ?= $(shell if [ -x ../../.venv-shared/bin/pytest ]; then echo ../../.venv-shared/bin/pytest; else command -v pytest; fi)
RUFF ?= $(shell if [ -x ../../.venv-shared/bin/ruff ]; then echo ../../.venv-shared/bin/ruff; else command -v ruff; fi)

.PHONY: all experiments verify test lint clean gcp-gpu gcp-tpu

all: lint test verify

experiments:
	$(PYTHON) experiments/01_numpy_to_jax_arrays.py
	$(PYTHON) experiments/02_jit_tracing_vs_execution.py
	$(PYTHON) experiments/03_grad_as_code_transformation.py
	$(PYTHON) experiments/04_vmap_goodbye_for_loops.py
	$(PYTHON) experiments/05_deterministic_prng_keys.py

verify:
	$(PYTHON) verify_fundamentals.py

test:
	$(PYTEST) -v

lint:
	$(RUFF) check .

gcp-gpu:
	bash gcp/run_on_demand_benchmark.sh gpu

gcp-tpu:
	bash gcp/run_on_demand_benchmark.sh tpu

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache experiments/__pycache__ tests/__pycache__
