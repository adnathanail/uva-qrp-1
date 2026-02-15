# Claude Code Context

## Project Overview

Research project implementing quantum algorithms for testing whether a unitary gate belongs to the Clifford hierarchy. Based on https://arxiv.org/abs/2510.07164.

## Structure

```
algorithm-1/           # Four-query Clifford tester (Algorithm 1 from paper)
├── README.md          # Mathematical breakdown of the algorithm
├── algorithm-1.ipynb  # Main implementation and tests
├── lib/
│   └── gates.py       # Reusable quantum gate functions
└── tests/
    ├── test_gates.py  # pytest tests with explicit matrix comparisons
    └── utils.py       # Test utilities
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
uv sync                              # Install dependencies
pytest algorithm-1/tests -v          # Run tests
jupyter notebook algorithm-1/        # Open notebook
```

## Testing Philosophy

Tests compare gate matrices against explicit expected values (not recomputed quantum operations). This ensures we're testing the implementation, not just that quantum math is self-consistent.
