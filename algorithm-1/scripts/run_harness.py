"""
Result collection harness for the Clifford tester.

Runs both tester variants (paired_runs and batched) across multiple backends,
stores raw results to disk, and prints a summary table.
Skips computations if raw_results.json already exists for a given run.

Usage:
    uv run python algorithm-1/scripts/run_harness.py              # run all unitaries
    uv run python algorithm-1/scripts/run_harness.py hadamard t_gate  # run specific ones
"""

import sys
from pathlib import Path

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from scripts.unitaries import UNITARIES

from lib import clifford_tester_batched, clifford_tester_paired_runs, get_qi_backend_and_transpilation_function
from lib.clifford_tester.results import (
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
from lib.expected_acceptance_probability import expected_acceptance_probability_from_circuit

# === Configuration ===
SHOTS = 1000
RESULTS_DIR = Path(__file__).parent.parent / "results"

BACKENDS = [
    ("aer_simulator", AerSimulator(), None, None),  # (name, backend, transpilation_fn, timeout)
    ("qi_tuna_9", *get_qi_backend_and_transpilation_function("Tuna-9"), 300),
]

EXPECTED_FILE = "01_expected_acceptance_probability.json"


# === Execution ===


def run_gate(
    gate_name: str,
    U: QuantumCircuit,
    shots: int,
    backends: list,
    results_dir: Path,
):
    n = U.num_qubits
    gate_dir = results_dir / f"{gate_name}_{shots}"
    gate_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Expected acceptance probability
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

    # Step 2: Run testers on each backend
    summaries = []

    for idx, (backend_name, backend, transpile_fn, timeout) in enumerate(backends, start=2):
        backend_dir = gate_dir / f"{idx:02d}_{backend_name}"

        # --- Paired runs ---
        paired_dir = backend_dir / "01_paired"
        paired_raw = load_paired_raw(paired_dir)
        if paired_raw is not None:
            print(f"[skip] {backend_name}/paired: raw_results.json exists")
        else:
            print(f"[run]  {backend_name}/paired: running {shots} shots...")
            raw_dicts = clifford_tester_paired_runs(
                U, n, shots=shots, backend=backend, transpilation_function=transpile_fn, timeout=timeout, checkpoint_dir=paired_dir
            )
            paired_raw = PairedRawResults(samples=[PairedSample(**d) for d in raw_dicts])
            save_paired_raw(paired_raw, paired_dir)
            print(f"[done] {backend_name}/paired: saved raw results")

        paired_rate = paired_raw.summarise()
        save_summary(paired_rate, paired_dir)

        # --- Batched ---
        batched_dir = backend_dir / "02_batched"
        batched_raw = load_batched_raw(batched_dir)
        if batched_raw is not None:
            print(f"[skip] {backend_name}/batched: raw_results.json exists")
        else:
            print(f"[run]  {backend_name}/batched: running {shots} shots...")
            raw_dict = clifford_tester_batched(
                U, n, shots=shots, backend=backend, transpilation_function=transpile_fn, timeout=timeout, checkpoint_dir=batched_dir
            )
            batched_raw = BatchedRawResults.from_tuples(raw_dict)
            save_batched_raw(batched_raw, batched_dir)
            print(f"[done] {backend_name}/batched: saved raw results")

        batched_rate = batched_raw.summarise()
        save_summary(batched_rate, batched_dir)

        summaries.append((backend_name, paired_rate, batched_rate))

    # Step 3: Print summary table
    print()
    print(f"Gate: {gate_name} ({n}-qubit), Shots: {shots}")
    print(f"Expected acceptance probability: {expected:.6f}")
    print()
    print(f"{'Backend':<20} {'Paired':>10} {'Batched':>10}")
    print("-" * 42)
    for name, paired_rate, batched_rate in summaries:
        print(f"{name:<20} {paired_rate:>10.6f} {batched_rate:>10.6f}")


def main():
    gates = UNITARIES
    if len(sys.argv) > 1:
        names = sys.argv[1:]
        unknown = set(names) - set(UNITARIES)
        if unknown:
            print(f"Unknown unitaries: {', '.join(sorted(unknown))}")
            print(f"Available: {', '.join(UNITARIES)}")
            sys.exit(1)
        gates = {k: v for k, v in UNITARIES.items() if k in names}

    for name, make_circuit in gates.items():
        print(f"\n{'=' * 50}")
        print(f"Gate: {name}")
        print(f"{'=' * 50}")
        run_gate(name, make_circuit(), SHOTS, BACKENDS, RESULTS_DIR)


if __name__ == "__main__":
    main()
