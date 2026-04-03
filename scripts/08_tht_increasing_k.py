"""Test THT (T·H·T) for increasing k levels of the Clifford hierarchy.

THT is believed to not lie in any finite level. This script runs the k-th
tester for k=2,3,4,... and prints acceptance rates to see if they converge
toward 1 (would indicate membership) or stay below 1.
"""

from pathlib import Path

from qiskit import QuantumCircuit

from cliff_lib.backends import resolve_backend
from cliff_lib.clifford_tester.testers import kth_clifford_tester

RESULTS_DIR = Path("results/tht_increasing_k")
RESULTS_FILE = RESULTS_DIR / "results.txt"

MAX_K = 10
NUM_SHOTS = 4000
BASE_A_SAMPLES = 10  # scaled exponentially: base * 2^(k-2)


def tht_circuit() -> QuantumCircuit:
    """T · H · T on a single qubit."""
    qc = QuantumCircuit(1)
    qc.t(0)
    qc.h(0)
    qc.t(0)
    return qc


def main():
    U = tht_circuit()
    n = U.num_qubits
    backend, transpile_fn, _timeout = resolve_backend("aer_simulator")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    header_info = f"Gate: THT (T·H·T), n={n}\nShots: {NUM_SHOTS}, base a_samples: {BASE_A_SAMPLES} (scaled as base * 2^(k-2))\n"
    header = f"{'k':<6}{'a_samples':>10}{'acceptance rate':>16}"
    separator = "-" * 32

    print(header_info)
    print(header)
    print(separator)

    with open(RESULTS_FILE, "w") as f:
        f.write(header_info + "\n")
        f.write(header + "\n")
        f.write(separator + "\n")

        for k in range(2, MAX_K + 1):
            num_a_samples = BASE_A_SAMPLES * 2 ** (k - 2)
            rate = kth_clifford_tester(
                U,
                n,
                k,
                shots=NUM_SHOTS,
                num_a_samples=num_a_samples,
                backend=backend,
                transpilation_function=transpile_fn,
            )
            line = f"{k:<6}{num_a_samples:>10}{rate:>16.6f}"
            print(line)
            f.write(line + "\n")
            f.flush()


if __name__ == "__main__":
    main()
