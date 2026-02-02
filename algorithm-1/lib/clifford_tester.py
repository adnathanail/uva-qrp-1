import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from lib import weyl_choi_state
from lib.measurements import measure_bell_basis

def get_clifford_tester_circuit(U_circuit: QuantumCircuit, n: int, x: list) -> QuantumCircuit:
    """
    Build the circuit for one run of the four-query Clifford tester.

    Creates a circuit that:
    1. Prepares two copies of |P_x⟩⟩ (Choi state)
    2. Applies U^{⊗2} to each copy
    3. Measures each in the Bell basis

    Args:
        U_circuit: A quantum circuit implementing the n-qubit unitary U
        n: Number of qubits U acts on
        x: 2n-bit string specifying which Weyl operator to use

    Returns:
        QuantumCircuit with 4n qubits and 4n classical bits
    """
    # We need 4n qubits total:
    # |P_x⟩⟩ is a 2n-qubit state (Choi state of n-qubit operator)
    # U^{⊗2}|P_x⟩⟩ applies U to both halves of the Choi state
    # We need two independent copies of this, so 4n qubits total

    qc = QuantumCircuit(4*n, 4*n)

    # Qubit layout:
    # Copy 1: qubits 0 to n-1 (A1), qubits n to 2n-1 (B1)
    # Copy 2: qubits 2n to 3n-1 (A2), qubits 3n to 4n-1 (B2)
    A1 = list(range(0, n))
    B1 = list(range(n, 2*n))
    A2 = list(range(2*n, 3*n))
    B2 = list(range(3*n, 4*n))

    # Step 1: Prepare |P_x⟩⟩ for both copies
    choi = weyl_choi_state(n, x)
    qc.append(choi, A1 + B1)
    qc.append(choi, A2 + B2)

    qc.barrier()

    # Step 2: Apply U^{⊗2} = U ⊗ U to each copy
    # This means applying U to A1, U to B1, U to A2, U to B2
    for qubits in [A1, B1, A2, B2]:
        qc.compose(U_circuit, qubits=qubits, inplace=True)

    qc.barrier()

    # Step 3: Measure each copy in Bell basis
    c1 = list(range(0, 2*n))      # Classical bits for copy 1
    c2 = list(range(2*n, 4*n))    # Classical bits for copy 2

    measure_bell_basis(qc, A1, B1, c1)
    measure_bell_basis(qc, A2, B2, c2)

    return qc


def clifford_tester(U_circuit: QuantumCircuit, n: int, shots: int = 1000):
    """
    Four-query Clifford tester algorithm.

    Tests whether a unitary U is a Clifford gate by:
    1. Sampling random x from F_2^{2n}
    2. Preparing two copies of U^{⊗2}|P_x⟩⟩
    3. Measuring each in the Bell basis to get y and y'
    4. Accepting if y = y'

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

        # Run the circuit
        result = simulator.run(qc, shots=1).result()
        counts = result.get_counts()

        # Get the measurement outcome (should be just one since shots=1)
        outcome = list(counts.keys())[0]

        # Split into y and y' (Qiskit returns bits in reverse order)
        # outcome is a 4n-bit string, first 2n bits are c2, last 2n bits are c1
        y_prime = outcome[:2 * n]  # Copy 2 result
        y = outcome[2 * n:]  # Copy 1 result

        # Accept if y = y'
        if y == y_prime:
            accepts += 1

    return accepts / shots
