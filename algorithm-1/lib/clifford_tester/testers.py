from collections import Counter
from collections.abc import Callable
from itertools import product
from typing import Any

import numpy as np
from qiskit import QuantumCircuit

from .utils import default_backend_and_transpilation, get_clifford_tester_circuit


def clifford_tester_batched(
    U_circuit: QuantumCircuit,
    n: int,
    shots: int = 1000,
    backend: Any = None,
    transpilation_function: Callable[[QuantumCircuit], QuantumCircuit] | None = None,
    timeout: float | None = None,
) -> dict[tuple[int, ...], dict[str, int]]:
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
    result = backend.run(circuits, shots=shots).result(timeout=timeout)

    # Collect raw counts for each Weyl operator
    raw_results = {}
    for i, x in enumerate(all_x):
        counts = result.get_counts(i)
        raw_results[x] = counts

    return raw_results


def clifford_tester_paired_runs(
    U_circuit: QuantumCircuit,
    n: int,
    shots: int = 1000,
    backend: Any = None,
    transpilation_function: Callable[[QuantumCircuit], QuantumCircuit] | None = None,
    timeout: float | None = None,
) -> list[dict[str, Any]]:
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

    # Pre-generate all random Weyl strings and deduplicate
    xs = [tuple(int(v) for v in np.random.randint(0, 2, size=2 * n)) for _ in range(shots)]
    x_counts = Counter(xs)

    # Build & transpile one circuit per unique x
    circuits: dict[tuple[int, ...], QuantumCircuit] = {}
    for x in x_counts:
        qc = get_clifford_tester_circuit(U_circuit, n, x)
        circuits[x] = transpilation_function(qc)

    # Run each unique circuit with exact shot count, expand & pair
    raw_results: list[dict[str, Any]] = []
    for x, count in x_counts.items():
        result = backend.run(circuits[x], shots=2 * count).result(timeout=timeout)
        counts = result.get_counts()

        # Expand counts dict to individual outcomes and shuffle
        outcomes: list[str] = []
        for bitstring, freq in counts.items():
            outcomes.extend([bitstring] * freq)
        np.random.shuffle(outcomes)

        # Pair sequentially
        for i in range(0, len(outcomes) - 1, 2):
            raw_results.append({"x": x, "y1": outcomes[i], "y2": outcomes[i + 1]})

    return raw_results
