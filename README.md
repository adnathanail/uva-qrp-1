# Practical quantum-hardware verification: testing the Clifford hierarchy
_Alex Nathanail UvA QuSoft Research Project 1_

Install dependencies and install `lib/` folder so scripts can access it

```shell
uv sync
```

Run linting, typechecking, and tests

```shell
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
