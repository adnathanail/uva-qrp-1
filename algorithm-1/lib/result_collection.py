"""Reusable result collection for the Clifford tester.

Provides ``collect_results_for_unitary`` which runs both tester variants
(paired_runs and batched) on a single backend and writes results to disk.
Skips computations if raw_results.json already exists for a given run.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, get_args

from qiskit import QuantumCircuit

from lib.clifford_tester import clifford_tester_batched, clifford_tester_paired_runs
from lib.expected_acceptance_probability import expected_acceptance_probability_from_circuit
from lib.state import (
    BatchedRawResults,
    ExpectedAcceptanceProbability,
    PairedRawResults,
    PairedSample,
    load_batched_raw,
    load_paired_raw,
    save_batched_raw,
    save_paired_raw,
    save_summary,
)
from lib.unitaries import gate_source

BackendName = Literal["aer_simulator", "qi_tuna_9"]
_VALID_BACKENDS = set(get_args(BackendName))

DEFAULT_RESULTS_DIR = Path(__file__).parent.parent / "results" / "clifford_tester"

EXPECTED_FILE = "expected_acceptance_probability.json"


def _resolve_backend(name: BackendName) -> tuple[Any, Callable[..., Any] | None, float | None]:
    """Map a backend name to ``(backend_instance, transpile_fn, timeout)``.

    Imports are lazy so that importing this module doesn't require QI or Aer.
    """
    if name not in _VALID_BACKENDS:
        raise ValueError(f"Unknown backend '{name}'. Valid: {', '.join(sorted(_VALID_BACKENDS))}")
    if name == "aer_simulator":
        from qiskit_aer import AerSimulator

        return AerSimulator(), None, None
    if name == "qi_tuna_9":
        from lib.qi_transpilation import get_qi_backend_and_transpilation_function

        backend, transpile_fn = get_qi_backend_and_transpilation_function("Tuna-9")
        return backend, transpile_fn, 300

    raise ValueError(f"Unhandled backend '{name}'")  # unreachable, keeps type-checkers happy


def collect_results_for_unitary(
    gate_name: str,
    U: QuantumCircuit,
    backend: BackendName,
    *,
    shots: int = 1000,
    timeout: float | None = None,
    results_dir: Path = DEFAULT_RESULTS_DIR,
) -> None:
    """Run both tester variants on *one* backend and save results.

    Parameters
    ----------
    gate_name:
        Name of the gate (must be present in ``UNITARIES``).
    U:
        The quantum circuit implementing the gate under test.
    backend:
        Which backend to run on (``"aer_simulator"`` or ``"qi_tuna_9"``).
    shots:
        Number of measurement shots.
    timeout:
        Optional timeout override.  When *None* the backend default is used.
    results_dir:
        Root directory for results (default: ``algorithm-1/results/clifford_tester``).
    """
    source = gate_source(gate_name)
    n = U.num_qubits
    gate_dir = results_dir / source / gate_name / f"{shots}_shots"
    gate_dir.mkdir(parents=True, exist_ok=True)

    # --- Expected acceptance probability ---
    expected_path = gate_dir / EXPECTED_FILE
    if expected_path.exists():
        data = ExpectedAcceptanceProbability.model_validate_json(expected_path.read_text())
        expected = data.expected_acceptance_probability
        print(f"[skip] Expected acceptance probability already computed: {expected:.6f}")
    else:
        expected = expected_acceptance_probability_from_circuit(U)
        data = ExpectedAcceptanceProbability(expected_acceptance_probability=expected)
        expected_path.write_text(data.model_dump_json(indent=2))
        print(f"[done] Expected acceptance probability: {expected:.6f}")

    # --- Resolve backend ---
    backend_instance, transpile_fn, default_timeout = _resolve_backend(backend)
    effective_timeout = timeout if timeout is not None else default_timeout

    backend_dir = gate_dir / backend

    # --- Paired runs ---
    paired_dir = backend_dir / "paired"
    paired_raw = load_paired_raw(paired_dir)
    if paired_raw is not None:
        print(f"[skip] {backend}/paired: raw_results.json exists")
    else:
        print(f"[run]  {backend}/paired: running {shots} shots...")
        raw_dicts = clifford_tester_paired_runs(
            U, n, shots=shots, backend=backend_instance, transpilation_function=transpile_fn, timeout=effective_timeout, checkpoint_dir=paired_dir
        )
        paired_raw = PairedRawResults(samples=[PairedSample(**d) for d in raw_dicts])
        save_paired_raw(paired_raw, paired_dir)
        print(f"[done] {backend}/paired: saved raw results")

    paired_rate = paired_raw.summarise()
    save_summary(paired_rate, paired_dir)

    # --- Batched ---
    batched_dir = backend_dir / "batched"
    batched_raw = load_batched_raw(batched_dir)
    if batched_raw is not None:
        print(f"[skip] {backend}/batched: raw_results.json exists")
    else:
        print(f"[run]  {backend}/batched: running {shots} shots...")
        raw_dict = clifford_tester_batched(
            U, n, shots=shots, backend=backend_instance, transpilation_function=transpile_fn, timeout=effective_timeout, checkpoint_dir=batched_dir
        )
        batched_raw = BatchedRawResults.from_tuples(raw_dict)
        save_batched_raw(batched_raw, batched_dir)
        print(f"[done] {backend}/batched: saved raw results")

    batched_rate = batched_raw.summarise()
    save_summary(batched_rate, batched_dir)

    # --- Summary ---
    print()
    print(f"Gate: {gate_name} ({n}-qubit), Shots: {shots}")
    print(f"Expected acceptance probability: {expected:.6f}")
    print()
    print(f"{'Backend':<20} {'Paired':>10} {'Batched':>10}")
    print("-" * 42)
    print(f"{backend:<20} {paired_rate:>10.6f} {batched_rate:>10.6f}")
