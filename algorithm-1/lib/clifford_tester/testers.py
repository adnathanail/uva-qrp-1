from collections.abc import Callable
from itertools import product

import numpy as np
from qiskit import QuantumCircuit

from .utils import default_backend_and_transpilation, get_clifford_tester_circuit


def clifford_tester_batched(
    U_circuit: QuantumCircuit, n: int, shots: int = 1000, backend=None, transpilation_function: Callable[[QuantumCircuit], QuantumCircuit] | None = None
) -> dict[tuple, dict]:
    """
    Four-query Clifford tester algorithm (batched).

    Tests whether a unitary U is a Clifford gate by enumerating all 4^n
    Weyl operators, running each circuit with the given number of shots,
    and returning the raw counts for each Weyl operator.

    Args:
        U_circuit: A quantum circuit implementing the n-qubit unitary U
        n: Number of qubits U acts on
        shots: Number of backend shots per Weyl operator circuit
        backend: Qiskit backend to run on (defaults to AerSimulator)
        transpilation_function: Optional function to transpile circuits

    Returns:
        dict mapping each Weyl operator x (tuple) to its Qiskit counts dict
    """
    backend, transpilation_function = default_backend_and_transpilation(backend, transpilation_function)

    # Generate all 4^n Weyl operators (all 2n-bit strings over F_2)
    all_x = list(product([0, 1], repeat=2 * n))

    # Build one circuit per Weyl operator
    circuits = [transpilation_function(get_clifford_tester_circuit(U_circuit, n, x)) for x in all_x]

    # Run all circuits in a single backend call
    result = backend.run(circuits, shots=shots).result(timeout=None)

    # Collect raw counts for each Weyl operator
    raw_results = {}
    for i, x in enumerate(all_x):
        counts = result.get_counts(i)
        raw_results[x] = counts

    return raw_results


def clifford_tester_paired_runs(
    U_circuit: QuantumCircuit, n: int, shots: int = 1000, backend=None, transpilation_function: Callable[[QuantumCircuit], QuantumCircuit] | None = None
) -> list[dict]:
    """
    Four-query Clifford tester algorithm (paired runs).

    Tests whether a unitary U is a Clifford gate by:
    1. Sampling random x from F_2^{2n}
    2. Running U^{⊗2}|P_x⟩⟩ twice with Bell basis measurement
    3. Recording both outcomes y1, y2

    Args:
        U_circuit: A quantum circuit implementing the n-qubit unitary U
        n: Number of qubits U acts on
        shots: Number of times to run the test
        backend: Qiskit backend to run on (defaults to AerSimulator)
        transpilation_function: Optional function to transpile circuits

    Returns:
        list of dicts, each with keys "x", "y1", "y2"
    """
    backend, transpilation_function = default_backend_and_transpilation(backend, transpilation_function)

    raw_results = []

    for _ in range(shots):
        # Sample random x from F_2^{2n}
        x = [int(v) for v in np.random.randint(0, 2, size=2 * n)]

        # Build and fully decompose the circuit (reps=3 handles nested gates)
        qc = get_clifford_tester_circuit(U_circuit, n, x)
        qc_transpiled = transpilation_function(qc)

        # Run the same circuit twice
        result = backend.run(qc_transpiled, shots=2).result(timeout=None)
        counts = result.get_counts()

        # Extract the two measurement outcomes
        keys = list(counts.keys())
        if len(keys) == 1:
            y1 = y2 = keys[0]
        else:
            # Two different outcomes, one count each
            y1, y2 = keys[0], keys[1]

        raw_results.append({"x": x, "y1": y1, "y2": y2})

    return raw_results
