from collections.abc import Callable

import numpy as np
from qiskit import QuantumCircuit

from .clifford_tester import _default_backend_and_transpilation, get_clifford_tester_circuit


def clifford_tester_sampled(
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
    backend, transpilation_function = _default_backend_and_transpilation(backend, transpilation_function)

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
