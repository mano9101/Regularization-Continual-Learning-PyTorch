# Elastic Weight Consolidation (EWC)

## Overview
- Focuses on protecting prior knowledge while sequentially learning Split CIFAR-100 tasks.
- Provides both continual learning training and an explicit unlearning workflow.
- Results from earlier experiments are archived under `results/`.

## Layout
- `src/ewc/` – package with training entrypoints (`main.py`, `ewc_unlearning.py`) plus helpers.
- `results/` – text summaries collected from previous experiment runs.
- `requirements.txt` – minimal dependency set for the method.

## Quickstart
1. `cd repositories/Regularization-Continual-Learning-PyTorch`
2. `python -m venv .venv && source .venv/bin/activate`
3. `pip install -r requirements.txt`
4. `export PYTHONPATH=src`
5. Run continual training: `python -m ewc.main`

### Unlearning demo
```
export PYTHONPATH=src
python -m ewc.ewc_unlearning
```

# Elastic Weight Consolidation (EWC)

Overview
--------

This repository provides a reference implementation of Elastic Weight Consolidation (EWC)
for continual learning experiments on Split CIFAR-100. It includes training scripts,
an explicit unlearning workflow, helper utilities for dataset splits, and example
experiment outputs in `results/`.

Repository layout
-----------------

- `src/ewc/` – Python package with training and unlearning entry points (`main.py`,
	`ewc_unlearning.py`) and supporting modules (`utils.py`, `strategies.py`).
- `results/` – experiment output summaries and logs.
- `requirements.txt` – Python dependencies used for development and experiments.

Quick start
-----------

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Make the package importable and run the training example:

```bash
export PYTHONPATH=src
python -m ewc.main
```

Unlearning demo
---------------

```bash
export PYTHONPATH=src
python -m ewc.ewc_unlearning
```

Data
----

By default the code downloads CIFAR-100 into `./data`. If you prefer a different
location, update the `root` argument in `src/ewc/utils.py` or set up a data
directory and point the scripts to it.

Contributing and development
----------------------------

- Tests: A minimal smoke test is included under `tests/` that verifies the package
	import. Run tests with `pytest` (install `pytest` in your environment).
- Packaging: Basic `pyproject.toml` and `setup.cfg` are provided for local
	installation using `pip install -e .`.
- Style and linting: The project does not enforce a style guide currently; adding
	`pre-commit` and `flake8`/`ruff` is recommended for collaborators.

Notes
-----

- The implementation is intended as a research reference and is not production
	hardened. Review training loops, device placement, and checkpointing before
	using it for large-scale experiments.
- If you want, I can add CI (GitHub Actions) for tests and linting, and a more
	complete example notebook demonstrating an experiment run.

