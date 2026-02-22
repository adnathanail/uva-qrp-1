# Practical quantum-hardware verification: testing the Clifford hierarchy
_Alex Nathanail UvA QuSoft Research Project 1_

[![Run tests](https://github.com/adnathanail/uva-qrp-1/actions/workflows/ci.yml/badge.svg)](https://github.com/adnathanail/quantum-error-correcting-codes/actions/workflows/ci.yml)
[![ty](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json)](https://github.com/astral-sh/ty)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![prek](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/j178/prek/master/docs/assets/badge-v0.json)](https://github.com/j178/prek)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)

Install dependencies and install `lib/` folder so scripts can access it

```shell
uv sync
```

Run formatting, linting, typechecking, and tests

```shell
uv run ruff format --check  # Remove check for some auto-fixes
uv run ty check
uv run ruff check
uv run pytest
```

Log in to QI

```shell
qi login
# Then restart Jupyter
```

## Algorithm 1 scripts

### 1. Collect full results for standard unitaries

Runs the paired and batched algorithms on the 9 "standard" unitaries, declared in `algorithm-1/lib/unitaries/standard.py`

```shell
uv run python algorithm-1/scripts/01_collect_full_results_for_standard_unitaries.py
```

Used to get a baseline set of results, to compare approaches/backends, and refine future approaches

### 1b. Compare Tuna-9 results on standard unitaries

Uses the saved results from script 1a, and produces a table comparing the results of the paired vs batched algorithms on the Tuna-9 backend

```shell
uv run python algorithm-1/scripts/01b_standard_unitaries_tuna_result_comparison.py
```

### Calculate how many n-qubit Cliffords there are

Based on the formula from [Robert Koenig, John A. Smolin](https://arxiv.org/abs/1406.2170), calculates how many Clifford unitaries there are for 1-10 qubits

```shell
uv run python algorithm-1/scripts/02_how_many_n_qubit_cliffords.py
```

### 3. Shot count vs execution time on Tuna-9

Benchmarks how shot count affects execution time on Tuna-9. Submits random Clifford circuits at 1/2/4-qubit gate sizes with 1–1000 shots, 10 reps each (120 jobs total). Resumable — saves progress after each job. Results and plot go to `algorithm-1/results/shot_timing/`.

```shell
uv run python algorithm-1/scripts/03_num_shots_time_comparison_tuna_9.py       # collect + plot
uv run python algorithm-1/scripts/03_num_shots_time_comparison_tuna_9.py plot  # plot only
```
