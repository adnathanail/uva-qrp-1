"""Test THT (T·H·T) for increasing k levels of the Clifford hierarchy.

THT is believed to not lie in any finite level. This script runs the k-th
tester for k=2,3,4,... and prints acceptance rates to see if they converge
toward 1 (would indicate membership) or stay below 1.
"""

from qiskit import QuantumCircuit

from cliff_lib.backends import resolve_backend
from cliff_lib.clifford_tester.testers import kth_clifford_tester

MAX_K = 10
NUM_SHOTS = 4000
NUM_A_SAMPLES = 30


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

    print(f"Gate: THT (T·H·T), n={n}")
    print(f"Shots: {NUM_SHOTS}, a_vector samples: {NUM_A_SAMPLES}")
    print()
    print(f"{'k':<6}{'acceptance rate':>16}")
    print("-" * 22)

    for k in range(2, MAX_K + 1):
        rate = kth_clifford_tester(
            U,
            n,
            k,
            shots=NUM_SHOTS,
            num_a_samples=NUM_A_SAMPLES,
            backend=backend,
            transpilation_function=transpile_fn,
        )
        print(f"{k:<6}{rate:>16.6f}")


if __name__ == "__main__":
    main()
