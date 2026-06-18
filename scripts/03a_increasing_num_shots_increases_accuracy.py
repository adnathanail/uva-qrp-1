# uv run scripts/03a_increasing_num_shots_increases_accuracy.py        # collect + plot
# uv run scripts/03a_increasing_num_shots_increases_accuracy.py plot    # plot only
"""4-qubit T⊗T⊗T⊗T on aer_simulator: shot count vs measured acceptance rate.

Runs the batched Clifford tester on the 4-qubit T⊗T⊗T⊗T for a range of
shot counts, with multiple reps per shot count, and plots the empirical
acceptance rate against the theoretical expected value.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cliff_lib.backends import resolve_backend
from cliff_lib.clifford_tester import clifford_tester_batched
from cliff_lib.expected_acceptance_probability import expected_acceptance_probability_from_circuit
from cliff_lib.state import (
    BatchedRawResults,
    load_batched_raw,
    save_batched_raw,
)
from cliff_lib.unitaries.standard import nc_4_t_gate

SHOTS_LIST = [10, 100, 1000, 10000]
N_REPS = 10

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "shot_accuracy_nc_4_t_gate"
PLOT_FILE = RESULTS_DIR / "shots_vs_acceptance.png"


def collect() -> None:
    U = nc_4_t_gate()
    n = U.num_qubits
    expected = expected_acceptance_probability_from_circuit(U)
    print(f"Expected acceptance probability for 4-qubit T⊗T⊗T⊗T: {expected:.6f}")

    backend, transpile_fn, timeout = resolve_backend("aer_simulator")

    for shots in SHOTS_LIST:
        for rep in range(N_REPS):
            batched_dir = RESULTS_DIR / f"{shots}_shots" / f"rep_{rep}" / "batched"

            if load_batched_raw(batched_dir) is None:
                print(f"[run]  batched  shots={shots} rep={rep}")
                raw = clifford_tester_batched(
                    U,
                    n,
                    shots=shots,
                    backend=backend,
                    transpilation_function=transpile_fn,
                    timeout=timeout,
                    checkpoint_dir=batched_dir,
                )
                save_batched_raw(BatchedRawResults.from_tuples(raw), batched_dir)
            else:
                print(f"[skip] batched  shots={shots} rep={rep}")


def plot() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    U = nc_4_t_gate()
    expected = expected_acceptance_probability_from_circuit(U)

    fig, ax = plt.subplots(figsize=(8, 5))

    valid_shots: list[int] = []
    means: list[float] = []
    sems: list[float] = []

    for shots in SHOTS_LIST:
        rates: list[float] = []
        for rep in range(N_REPS):
            raw = load_batched_raw(RESULTS_DIR / f"{shots}_shots" / f"rep_{rep}" / "batched")
            if raw is not None:
                rates.append(raw.summarise())

        if rates:
            valid_shots.append(shots)
            means.append(float(np.mean(rates)))
            sems.append(float(np.std(rates) / np.sqrt(len(rates))))
            ax.scatter([shots] * len(rates), rates, alpha=0.3, s=15, color="C0")

    if valid_shots:
        ax.errorbar(
            valid_shots,
            means,
            yerr=sems,
            marker="o",
            capsize=4,
            label="Simulator",
            color="C0",
        )

    ax.axhline(
        expected,
        color="black",
        linestyle="--",
        label=f"Theoretical p_acc = {expected:.4f}",
    )
    ax.set_xscale("log")
    ax.set_ylim(0.25, 0.5)
    ax.set_xlabel("Number of shots")
    ax.set_ylabel("Acceptance rate")
    ax.set_title("4-qubit T⊗T⊗T⊗T on aer_simulator: shots vs acceptance rate")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_FILE, dpi=150)
    print(f"Plot saved to {PLOT_FILE}")
    plt.close(fig)


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "plot":
        plot()
    else:
        collect()
        plot()


if __name__ == "__main__":
    main()
