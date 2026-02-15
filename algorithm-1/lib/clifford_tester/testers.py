from collections.abc import Callable
from itertools import product

import numpy as np
from qiskit import QuantumCircuit

from .utils import collision_probability, default_backend_and_transpilation, get_clifford_tester_circuit


def clifford_tester_batched(
    U_circuit: QuantumCircuit, n: int, shots: int = 1000, backend=None, transpilation_function: Callable[[QuantumCircuit], QuantumCircuit] | None = None
):
    """
    Four-query Clifford tester algorithm (batched).

    Tests whether a unitary U is a Clifford gate by enumerating all 4^n
    Weyl operators, running each circuit with the given number of shots,
    and computing the average collision probability.

    For Clifford gates, each circuit produces a deterministic outcome,
    so the collision probability is 1.0 for every Weyl operator.

    Args:
        U_circuit: A quantum circuit implementing the n-qubit unitary U
        n: Number of qubits U acts on
        shots: Number of backend shots per Weyl operator circuit
        backend: Qiskit backend to run on (defaults to AerSimulator)
        transpilation_function: Optional function to transpile circuits

    Returns:
        acceptance_rate: Average collision probability across all Weyl operators
    """
    backend, transpilation_function = default_backend_and_transpilation(backend, transpilation_function)

    # Generate all 4^n Weyl operators (all 2n-bit strings over F_2)
    all_x = list(product([0, 1], repeat=2 * n))

    # Build one circuit per Weyl operator
    circuits = [transpilation_function(get_clifford_tester_circuit(U_circuit, n, x)) for x in all_x]

    # Run all circuits in a single backend call
    result = backend.run(circuits, shots=shots).result(timeout=None)

    # Compute average collision probability across all Weyl operators
    total_collision_prob = 0.0
    for i in range(len(circuits)):
        counts = result.get_counts(i)
        total_collision_prob += collision_probability(counts)

    return total_collision_prob / len(circuits)


def clifford_tester_paired_runs(
    U_circuit: QuantumCircuit, n: int, shots: int = 1000, backend=None, transpilation_function: Callable[[QuantumCircuit], QuantumCircuit] | None = None
):
    """
    Four-query Clifford tester algorithm (sampled).

    Tests whether a unitary U is a Clifford gate by:
    1. Sampling random x from F_2^{2n}
    2. Running U^{⊗2}|P_x⟩⟩ twice with Bell basis measurement
    3. Accepting if both runs give the same outcome y = y'

    Args:
        U_circuit: A quantum circuit implementing the n-qubit unitary U
        n: Number of qubits U acts on
        shots: Number of times to run the test
        backend: Qiskit backend to run on (defaults to AerSimulator)
        transpilation_function: Optional function to transpile circuits

    Returns:
        acceptance_rate: Fraction of runs where y = y'
    """
    backend, transpilation_function = default_backend_and_transpilation(backend, transpilation_function)

    accepts = 0

    for _ in range(shots):
        # Sample random x from F_2^{2n}
        x = list(np.random.randint(0, 2, size=2 * n))

        # Build and fully decompose the circuit (reps=3 handles nested gates)
        qc = get_clifford_tester_circuit(U_circuit, n, x)
        qc_transpiled = transpilation_function(qc)

        # Run the same circuit twice
        result = backend.run(qc_transpiled, shots=2).result(timeout=None)
        counts = result.get_counts()

        # Accept if both shots gave the same outcome (y == y' therefore only one key in counts)
        if len(counts) == 1:
            accepts += 1

    return accepts / shots
