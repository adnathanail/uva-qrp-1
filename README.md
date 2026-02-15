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

Run the result collection harness

```shell
uv run python algorithm-1/scripts/run_harness.py
```

Log in to QI

```shell
qi login
# Then restart Jupyter
```

Projects
- [Algorithm 1](algorithm-1/README.md)
