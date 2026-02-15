# Claude Code Context

## Project Overview

Research project implementing quantum algorithms for testing whether a unitary gate belongs to the Clifford hierarchy. Based on https://arxiv.org/abs/2510.07164.

## Structure

```
algorithm-1/
├── README.md                                       # Mathematical breakdown of the algorithm
├── lib/
│   ├── __init__.py
│   ├── gates.py                                    # Reusable quantum gate functions
│   ├── measurements.py                             # Bell basis measurement
│   ├── qi_transpilation.py                         # Quantum Inspire backend helpers
│   ├── expected_acceptance_probability.py          # Theoretical p_acc computation
│   └── clifford_tester/
│       ├── __init__.py
│       ├── testers.py                              # paired_runs and batched tester implementations
│       ├── utils.py                                # Circuit building, collision probability
│       └── results.py                              # Pydantic models for raw results + checkpoint state + save/load
├── scripts/
│   ├── expected_acceptance_probability.py          # Compute p_acc for specific gates
│   └── run_harness.py                              # Result collection harness (multi-backend, skip-if-exists)
├── results/                                        # Generated output from run_harness.py (gitignored)
└── tests/
    ├── test_gates.py                               # pytest tests with explicit matrix comparisons
    ├── test_expected_acceptance_probability.py     # Tests for theoretical p_acc values
    └── utils.py                                    # Test utilities
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

### Tester Approaches

There are two tester implementations in `clifford_tester/testers.py`:

- **`clifford_tester_batched`**: Enumerates all 4^n Weyl operators, builds one circuit per operator, and runs them all in a single `backend.run()` call with many shots each. The acceptance probability is computed from collision probabilities across the full counts distribution.
- **`clifford_tester_paired_runs`**: Randomly samples Weyl operators, runs the circuit for each sample, and pairs individual measurement outcomes (y1, y2) to check for collisions. Deduplicates circuits so each unique Weyl operator is only built/transpiled once.

On a **noiseless simulator**, both approaches produce statistically equivalent results — they're sampling from the same distribution.

On **noisy hardware**, the batched approach gives better data.
It runs each circuit with many shots in one job, so all shots for a given circuit experience the same calibration state.
The paired_runs approach submits many smaller jobs, which may span different calibration windows and accumulate more overhead-related noise.
The batched approach also avoids the statistical subtlety of pairing expanded counts (where shuffling is needed to prevent bias).

## Running

```shell
uv sync                                               # Install dependencies (also installs lib/ as a package)
pytest algorithm-1/tests -v                           # Run tests
uv run ty check                                       # Type checking
uv run python algorithm-1/scripts/run_harness.py      # Run result collection harness
jupyter notebook algorithm-1/                         # Open notebook
```

### Result Collection Harness

`run_harness.py` runs both tester variants (paired_runs, batched) across configured backends and writes results to `algorithm-1/results/`. It skips runs if `raw_results.json` already exists, so re-running is safe and fast. Configure gate, shots, and backends in the script's configuration section.

### Checkpoint / Resumption

Both testers support checkpoint files via `checkpoint_dir` (passed automatically by the harness). If a run is interrupted, re-running resumes from where it left off:

- **`plan.json`** — saves the testing plan (which Weyl operators, how many shots) so resumed runs use the same random samples.
- **`jobs.json`** (paired only) — tracks per-x progress: which have counts collected, which have a job submitted but not yet collected.
- **`job_{id}.qpy`** — serialized QI job (via `QIJob.serialize()`), allowing retrieval of results from jobs still running on QI hardware. Named with the batch job ID for easy identification.

On completion, checkpoint files are cleaned up automatically. On AerSimulator, jobs are ephemeral so incomplete x values are simply resubmitted (fast). On QI hardware, the serialized job is reconstructed via `_load_job()` (a lightweight alternative to `QIJob.deserialize()` that takes a backend directly instead of requiring a provider).

### Package Setup

`lib/` is installed as a Python package via hatchling (configured in `pyproject.toml`), so `from lib import ...` works everywhere — scripts, tests, and notebooks — without `sys.path` hacks.

### Quantum Inspire
- `qi_transpilation.py` lazily initializes `QIProvider` (only on first call to `get_qi_backend_and_transpilation_function`), so importing the module doesn't require a QI connection.
- Run `qi login` before using QI backends.

## Testing Philosophy

Tests compare gate matrices against explicit expected values (not recomputed quantum operations). This ensures we're testing the implementation, not just that quantum math is self-consistent.
