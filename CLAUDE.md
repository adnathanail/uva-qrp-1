# Claude Code Context

## Project Overview

Research project implementing quantum algorithms for testing whether a unitary gate belongs to the Clifford hierarchy. Based on https://arxiv.org/abs/2510.07164.

## Structure

```
algorithm-1/
├── README.md              # Mathematical breakdown of the algorithm
├── algorithm-1.ipynb      # Main implementation and tests
├── qi-testing.ipynb       # Testing on Quantum Inspire hardware
├── lib/
│   ├── __init__.py
│   ├── gates.py           # Reusable quantum gate functions
│   ├── measurements.py    # Bell basis measurement
│   ├── qi_transpilation.py # Quantum Inspire backend helpers
│   ├── expected_acceptance_probability.py  # Theoretical p_acc computation
│   └── clifford_tester/
│       ├── __init__.py
│       ├── testers.py     # paired_runs and batched tester implementations
│       ├── utils.py       # Circuit building, collision probability
│       └── results.py     # Pydantic models for raw results + save/load/summarise
├── scripts/
│   ├── expected_acceptance_probability.py  # Compute p_acc for specific gates
│   └── run_harness.py     # Result collection harness (multi-backend, skip-if-exists)
├── results/               # Generated output from run_harness.py (gitignored)
└── tests/
    ├── test_gates.py      # pytest tests with explicit matrix comparisons
    └── utils.py           # Test utilities
```

## Key Technical Notes

### Qiskit Conventions
- **Little-endian ordering**: Qubit 0 is the least significant (rightmost) in tensor products
- For a 2-qubit system: `(operator on qubit 1) ⊗ (operator on qubit 0)`
- This affects how we construct expected matrices in tests

### Custom Gates and Simulation
- Custom gates created with `.to_gate()` need `.decompose(reps=N)` before running on AerSimulator
- Use `reps=3` for nested gates (e.g., `weyl_choi_state` contains `maximally_entangled_state` contains Bell pairs)

### Simulator Memory Limits
- The Clifford tester uses 4n qubits for an n-qubit gate under test
- Statevector simulation memory: 2^(4n) × 16 bytes
- Practical limits on 32GB machine: ~7 qubits (28 total, ~4GB)

## Running

```shell
uv sync                                               # Install dependencies (also installs lib/ as a package)
pytest algorithm-1/tests -v                           # Run tests
uv run python algorithm-1/scripts/run_harness.py      # Run result collection harness
jupyter notebook algorithm-1/                         # Open notebook
```

### Result Collection Harness

`run_harness.py` runs both tester variants (paired_runs, batched) across configured backends and writes results to `algorithm-1/results/`. It skips runs if `raw_results.json` already exists, so re-running is safe and fast. Configure gate, shots, and backends in the script's configuration section.

### Package Setup

`lib/` is installed as a Python package via hatchling (configured in `pyproject.toml`), so `from lib import ...` works everywhere — scripts, tests, and notebooks — without `sys.path` hacks.

## Testing Philosophy

Tests compare gate matrices against explicit expected values (not recomputed quantum operations). This ensures we're testing the implementation, not just that quantum math is self-consistent.
