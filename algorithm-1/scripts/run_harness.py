"""
Result collection harness for the Clifford tester.

Runs both tester variants (paired_runs and batched) across multiple backends,
stores raw results to disk, and prints a summary table.
Skips computations if raw_results.json already exists for a given run.
"""

import json
from pathlib import Path

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from lib import clifford_tester_batched, clifford_tester_paired_runs
from lib.clifford_tester.results import (
    load_batched_raw,
    load_paired_raw,
    save_batched_raw,
    save_paired_raw,
    save_summary,
    summarise_batched,
    summarise_paired,
)
from lib.expected_acceptance_probability import expected_acceptance_probability_from_circuit

# === Configuration ===
GATE_NAME = "hadamard"
N = 1  # number of qubits

U = QuantumCircuit(1)
U.h(0)

SHOTS = 1000
RESULTS_DIR = Path(__file__).parent.parent / "results"

BACKENDS = [
    ("aer_simulator", AerSimulator(), None),  # (name, backend, transpilation_fn)
    # ("qi_tuna_9", *get_backend_and_transpilation_function("Tuna-9")),
]

# === Execution (no edits below) ===


def main():
    gate_dir = RESULTS_DIR / GATE_NAME
    gate_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Expected acceptance probability
    expected_path = gate_dir / "01_expected_acceptance_probability.json"
    if expected_path.exists():
        with open(expected_path) as f:
            expected = json.load(f)["expected_acceptance_probability"]
        print(f"[skip] Expected acceptance probability already computed: {expected:.6f}")
    else:
        expected = expected_acceptance_probability_from_circuit(U)
        with open(expected_path, "w") as f:
            json.dump({"expected_acceptance_probability": expected}, f, indent=2)
        print(f"[done] Expected acceptance probability: {expected:.6f}")

    # Step 2: Run testers on each backend
    summaries = []

    for idx, (backend_name, backend, transpile_fn) in enumerate(BACKENDS, start=2):
        backend_dir = gate_dir / f"{idx:02d}_{backend_name}"

        # --- Paired runs ---
        paired_dir = backend_dir / "01_paired"
        paired_raw = load_paired_raw(paired_dir)
        if paired_raw is not None:
            print(f"[skip] {backend_name}/paired: raw_results.json exists")
        else:
            print(f"[run]  {backend_name}/paired: running {SHOTS} shots...")
            paired_raw = clifford_tester_paired_runs(U, N, shots=SHOTS, backend=backend, transpilation_function=transpile_fn)
            save_paired_raw(paired_raw, paired_dir)
            print(f"[done] {backend_name}/paired: saved raw results")

        paired_rate = summarise_paired(paired_raw)
        save_summary(paired_rate, paired_dir)

        # --- Batched ---
        batched_dir = backend_dir / "02_batched"
        batched_raw = load_batched_raw(batched_dir)
        if batched_raw is not None:
            print(f"[skip] {backend_name}/batched: raw_results.json exists")
        else:
            print(f"[run]  {backend_name}/batched: running {SHOTS} shots...")
            batched_raw = clifford_tester_batched(U, N, shots=SHOTS, backend=backend, transpilation_function=transpile_fn)
            save_batched_raw(batched_raw, batched_dir)
            print(f"[done] {backend_name}/batched: saved raw results")

        batched_rate = summarise_batched(batched_raw)
        save_summary(batched_rate, batched_dir)

        summaries.append((backend_name, paired_rate, batched_rate))

    # Step 3: Print summary table
    print()
    print(f"Gate: {GATE_NAME} ({N}-qubit), Shots: {SHOTS}")
    print(f"Expected acceptance probability: {expected:.6f}")
    print()
    print(f"{'Backend':<20} {'Paired':>10} {'Batched':>10}")
    print("-" * 42)
    for name, paired_rate, batched_rate in summaries:
        print(f"{name:<20} {paired_rate:>10.6f} {batched_rate:>10.6f}")


if __name__ == "__main__":
    main()
