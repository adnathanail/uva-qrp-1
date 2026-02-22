# uv run algorithm-1/scripts/03_num_shots_time_comparison_tuna_9.py         # collect + plot
# uv run algorithm-1/scripts/03_num_shots_time_comparison_tuna_9.py plot     # plot only
"""Benchmark shot count vs execution time on Tuna-9.

Submits random Clifford circuits of varying qubit counts and shot counts,
records execution times via the QI API, and generates a summary plot.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import requests
from qiskit import QuantumCircuit
from qiskit.quantum_info import random_clifford

from lib.backends import resolve_backend
from lib.clifford_tester.utils import get_clifford_tester_circuit
from lib.jobs import get_job_id
from lib.state.utils import atomic_write

# ── Configuration ────────────────────────────────────────────────────────────

N_QUBITS_LIST = [1, 2, 4]
SHOTS_LIST = [1, 10, 100, 1000]
N_REPS = 10

RESULTS_DIR = Path("algorithm-1/results/shot_timing")
RESULTS_FILE = RESULTS_DIR / "results.json"
PLOT_FILE = RESULTS_DIR / "shots_vs_time.png"

QI_CONFIG = Path.home() / ".quantuminspire" / "config.json"
QI_HOST = "https://api.quantum-inspire.com"


# ── QI API helpers ───────────────────────────────────────────────────────────


def get_token() -> str:
    config = json.loads(QI_CONFIG.read_text())
    return config["auths"][QI_HOST]["tokens"]["access_token"]


def get_execution_time(job_id: int, token: str) -> float | None:
    r = requests.get(
        f"{QI_HOST}/results/job/{job_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    r.raise_for_status()
    items = r.json().get("items", [])
    if items:
        return items[0]["execution_time_in_seconds"]
    return None


# ── State helpers ────────────────────────────────────────────────────────────


def load_results() -> dict[str, dict[str, object]]:
    if RESULTS_FILE.exists():
        return json.loads(RESULTS_FILE.read_text())
    return {}


def save_results(results: dict[str, dict[str, object]]) -> None:
    atomic_write(RESULTS_FILE, json.dumps(results, indent=2) + "\n")


def make_key(n_qubits: int, shots: int, rep: int) -> str:
    return f"{n_qubits}_qubits/{shots}_shots/rep_{rep}"


# ── Collection ───────────────────────────────────────────────────────────────


def collect() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    backend, transpile_fn, timeout = resolve_backend("qi_tuna_9")
    token = get_token()
    results = load_results()

    total = len(N_QUBITS_LIST) * len(SHOTS_LIST) * N_REPS
    done = sum(1 for v in results.values() if v.get("execution_time_seconds") is not None)
    print(f"Progress: {done}/{total} complete")

    for n_qubits in N_QUBITS_LIST:
        for shots in SHOTS_LIST:
            for rep in range(N_REPS):
                key = make_key(n_qubits, shots, rep)
                entry = results.get(key)

                # Already have timing → skip
                if entry and entry.get("execution_time_seconds") is not None:
                    continue

                # Have job_id but no timing → poll API until result appears
                if entry and entry.get("job_id") is not None:
                    job_id_str = entry["job_id"]
                    print(f"[{key}] Polling job {job_id_str}...")
                    poll_interval = 10  # seconds
                    elapsed = 0
                    exec_time = None
                    while elapsed < timeout:
                        exec_time = get_execution_time(int(job_id_str), token)
                        if exec_time is not None:
                            break
                        print(f"[{key}] Not ready, retrying in {poll_interval}s ({elapsed}s elapsed)...")
                        time.sleep(poll_interval)
                        elapsed += poll_interval

                    if exec_time is not None:
                        entry["execution_time_seconds"] = exec_time
                        save_results(results)
                        done += 1
                        print(f"[{key}] {exec_time:.4f}s  ({done}/{total})")
                    else:
                        print(f"[{key}] Timed out after {timeout}s, will retry next run")
                    continue

                # Submit new job
                if entry is None:
                    cliff = random_clifford(n_qubits)
                    U = QuantumCircuit(n_qubits)
                    U.append(cliff.to_instruction(), list(range(n_qubits)))

                    x = (0,) * (2 * n_qubits)
                    qc = get_clifford_tester_circuit(U, n_qubits, x)
                    qc = transpile_fn(qc)

                    print(f"[{key}] Submitting ({2 * n_qubits} circuit qubits, {shots} shots)...")
                    job = backend.run(qc, shots=shots)
                    job_id = get_job_id(job)

                    # Save job_id immediately so we can resume
                    results[key] = {"job_id": job_id, "execution_time_seconds": None}
                    save_results(results)

                    print(f"[{key}] Job {job_id}, waiting...")
                    job.result(timeout=timeout)

                    exec_time = get_execution_time(int(job_id), token)
                    results[key]["execution_time_seconds"] = exec_time
                    save_results(results)

                    done += 1
                    time_str = f"{exec_time:.4f}s" if exec_time is not None else "N/A"
                    print(f"[{key}] {time_str}  ({done}/{total})")

    print(f"\nCollection complete. {done}/{total} results.")


# ── Plot ─────────────────────────────────────────────────────────────────────


def plot() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = load_results()
    if not results:
        print("No results to plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for n_qubits in N_QUBITS_LIST:
        means = []
        stds = []
        valid_shots = []

        for shots in SHOTS_LIST:
            times = []
            for rep in range(N_REPS):
                key = make_key(n_qubits, shots, rep)
                entry = results.get(key)
                if entry and entry.get("execution_time_seconds") is not None:
                    times.append(entry["execution_time_seconds"])

            if times:
                valid_shots.append(shots)
                means.append(np.mean(times))
                stds.append(np.std(times) / np.sqrt(len(times)))

                # Scatter individual points
                ax.scatter(
                    [shots] * len(times),
                    times,
                    alpha=0.2,
                    s=15,
                    color=f"C{N_QUBITS_LIST.index(n_qubits)}",
                )

        if valid_shots:
            ax.errorbar(
                valid_shots,
                means,
                yerr=stds,
                marker="o",
                capsize=4,
                label=f"{n_qubits}-qubit gate ({2 * n_qubits}-qubit circuit)",
                color=f"C{N_QUBITS_LIST.index(n_qubits)}",
            )

    ax.set_xscale("log")
    ax.set_xlabel("Number of shots")
    ax.set_ylabel("Execution time (seconds)")
    ax.set_title("Shot count vs execution time on Tuna-9")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_FILE, dpi=150)
    print(f"Plot saved to {PLOT_FILE}")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "plot":
        plot()
    else:
        collect()
        plot()


if __name__ == "__main__":
    main()
