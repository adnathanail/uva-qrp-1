import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from lib import weyl_choi_state
from lib.measurements import measure_bell_basis


def get_clifford_tester_circuit(U_circuit: QuantumCircuit, n: int, x: list) -> QuantumCircuit:
    """
    Build the circuit for one sample of the Clifford tester.

    Creates a circuit that:
    1. Prepares |P_x⟩⟩ (Choi state)
    2. Applies U^{⊗2} (U to both halves)
    3. Measures in the Bell basis

    Args:
        U_circuit: A quantum circuit implementing the n-qubit unitary U
        n: Number of qubits U acts on
        x: 2n-bit string specifying which Weyl operator to use

    Returns:
        QuantumCircuit with 2n qubits and 2n classical bits
    """
    qc = QuantumCircuit(2 * n, 2 * n)

    # Qubit layout: qubits 0 to n-1 (A), qubits n to 2n-1 (B)
    A = list(range(0, n))
    B = list(range(n, 2 * n))

    # Step 1: Prepare |P_x⟩⟩
    choi = weyl_choi_state(n, x)
    qc.append(choi, A + B)

    qc.barrier()

    # Step 2: Apply U^{⊗2} = U ⊗ U (U to both registers)
    for qubits in [A, B]:
        qc.compose(U_circuit, qubits=qubits, inplace=True)

    qc.barrier()

    # Step 3: Measure in Bell basis
    clbits = list(range(2 * n))
    measure_bell_basis(qc, A, B, clbits)

    return qc


def clifford_tester(U_circuit: QuantumCircuit, n: int, shots: int = 1000):
    """
    Four-query Clifford tester algorithm.

    Tests whether a unitary U is a Clifford gate by:
    1. Sampling random x from F_2^{2n}
    2. Running U^{⊗2}|P_x⟩⟩ twice with Bell basis measurement
    3. Accepting if both runs give the same outcome y = y'

    Args:
        U_circuit: A quantum circuit implementing the n-qubit unitary U
        n: Number of qubits U acts on
        shots: Number of times to run the test

    Returns:
        acceptance_rate: Fraction of runs where y = y'
    """
    accepts = 0
    simulator = AerSimulator()

    for _ in range(shots):
        # Sample random x from F_2^{2n}
        x = list(np.random.randint(0, 2, size=2 * n))

        # Build and fully decompose the circuit (reps=3 handles nested gates)
        qc = get_clifford_tester_circuit(U_circuit, n, x)
        qc = qc.decompose(reps=3)

        # Run the same circuit twice
        result = simulator.run(qc, shots=2).result()
        counts = result.get_counts()

        # Accept if both shots gave the same outcome (y == y' therefore only one key in counts)
        if len(counts) == 1:
            accepts += 1

    return accepts / shots
