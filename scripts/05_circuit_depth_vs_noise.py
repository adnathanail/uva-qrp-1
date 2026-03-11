# uv run scripts/05_circuit_depth_vs_noise.py         # collect + plot
# uv run scripts/05_circuit_depth_vs_noise.py plot     # plot only
"""Circuit depth vs acceptance rate for random 2-qubit Cliffords on Tuna-9.

Generates 50 random 2-qubit Cliffords, runs the batched tester on each via
Tuna-9, and plots transpiled circuit depth against acceptance rate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate

from cliff_lib.backends import resolve_backend
from cliff_lib.clifford_tester.testers import clifford_tester_batched
from cliff_lib.state.outputs import BatchedRawResults
from cliff_lib.state.utils import atomic_write
from cliff_lib.unitaries.generators.stim import stim_random_clifford_gate

# -- Configuration -----------------------------------------------------------

N_QUBITS = 2
N_UNITARIES = 50
SHOTS = 1000

RESULTS_DIR = Path("results/depth_vs_acceptance")
STATE_FILE = RESULTS_DIR / "state.json"
PLOT_FILE = RESULTS_DIR / "depth_vs_acceptance.png"


# -- State helpers -----------------------------------------------------------


def _complex_encoder(obj: object) -> object:
    if isinstance(obj, complex):
        return [obj.real, obj.imag]
    raise TypeError(f"Not serializable: {type(obj)}")


def _decode_matrix(raw: list[list[list[float]]]) -> np.ndarray:
    return np.array([[complex(c[0], c[1]) for c in row] for row in raw])


def load_state() -> dict[str, dict]:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def save_state(state: dict[str, dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write(STATE_FILE, json.dumps(state, indent=2, default=_complex_encoder) + "\n")


# -- Collection --------------------------------------------------------------


def collect() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    backend, transpile_fn, timeout = resolve_backend("qi_tuna_9")
    state = load_state()

    # Phase 1: Generate unitaries if needed (first run)
    if not state:
        _state = {}
        print(f"Generating {N_UNITARIES} random {N_QUBITS}-qubit Cliffords...")
        for i in range(N_UNITARIES):
            gate = stim_random_clifford_gate(N_QUBITS)
            matrix = gate.to_matrix().tolist()

            qc = QuantumCircuit(N_QUBITS)
            qc.append(gate, list(range(N_QUBITS)))
            transpiled = transpile_fn(qc)

            _state[str(i)] = {
                "matrix": matrix,
                "depth": transpiled.depth(),
                "acceptance_rate": None,
            }
        save_state(_state)
        state = load_state()
        print(f"Saved {N_UNITARIES} unitaries with depths to {STATE_FILE}")

    done = sum(1 for v in state.values() if v["acceptance_rate"] is not None)
    total = len(state)
    print(f"Progress: {done}/{total} complete")

    # Phase 2: Run batched tester for each unitary
    for idx_str, entry in state.items():
        if entry["acceptance_rate"] is not None:
            continue

        idx = int(idx_str)
        matrix = _decode_matrix(entry["matrix"])
        gate = UnitaryGate(matrix)

        U_circuit = QuantumCircuit(N_QUBITS)
        U_circuit.append(gate, list(range(N_QUBITS)))

        checkpoint_dir = RESULTS_DIR / "checkpoints" / idx_str

        print(f"\n[{idx + 1}/{total}] depth={entry['depth']}")
        raw = clifford_tester_batched(
            U_circuit,
            N_QUBITS,
            shots=SHOTS,
            backend=backend,
            transpilation_function=transpile_fn,
            timeout=timeout,
            checkpoint_dir=checkpoint_dir,
        )

        acceptance_rate = BatchedRawResults.from_tuples(raw).summarise()
        entry["acceptance_rate"] = acceptance_rate
        save_state(state)

        done += 1
        print(f"[{idx + 1}/{total}] acceptance_rate={acceptance_rate:.4f}  ({done}/{total})")

    print(f"\nCollection complete. {done}/{total} results.")


# -- Plot --------------------------------------------------------------------


def plot() -> None:
    state = load_state()
    if not state:
        print("No results to plot.")
        return

    completed = {k: v for k, v in state.items() if v["acceptance_rate"] is not None}
    if not completed:
        print("No completed results to plot.")
        return

    depths = [v["depth"] for v in completed.values()]
    rates = [v["acceptance_rate"] for v in completed.values()]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(depths, rates, alpha=0.6, edgecolors="black", linewidths=0.5)

    # Trend line
    if len(depths) >= 2:
        z = np.polyfit(depths, rates, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(depths), max(depths), 100)
        ax.plot(x_line, p(x_line), "--", color="red", alpha=0.7, label=f"Linear fit (slope={z[0]:.4f})")
        ax.legend()

    ax.set_xlabel("Transpiled circuit depth")
    ax.set_ylabel("Acceptance rate")
    ax.set_title("Circuit depth vs acceptance rate (2-qubit Cliffords, Tuna-9)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_FILE, dpi=150)
    print(f"Plot saved to {PLOT_FILE}")
    plt.close(fig)


# -- Main --------------------------------------------------------------------


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "plot":
        plot()
    else:
        collect()
        plot()


if __name__ == "__main__":
    main()
