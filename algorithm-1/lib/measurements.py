from qiskit import QuantumCircuit


def measure_bell_basis(qc: QuantumCircuit, qubits_A: list, qubits_B: list, clbits: list):
    """
    Measure in the Bell basis {|P_y⟩⟩⟨⟨P_y|}_y

    To measure in the Bell basis, we undo the Bell state preparation
    (CNOT then H) and measure in the computational basis.

    The measurement outcome directly gives us y = (a, b) where:
    - The B register measurement gives 'a' (which controlled Z)
    - The A register measurement gives 'b' (which controlled X)

    Args:
        qc: Quantum circuit to modify
        qubits_A: First register (n qubits)
        qubits_B: Second register (n qubits)
        clbits: Classical bits to store measurement results (2n bits)
    """
    n = len(qubits_A)
    assert len(qubits_B) == n
    assert len(clbits) == 2 * n

    # Undo Bell state preparation: CNOT then H
    for i in range(n):
        qc.cx(qubits_A[i], qubits_B[i])
        qc.h(qubits_A[i])

    # Measure: B register gives 'a', A register gives 'b'
    # Store as y = (a, b) = (B measurement, A measurement)
    for i in range(n):
        qc.measure(qubits_B[i], clbits[i])  # a bits
        qc.measure(qubits_A[i], clbits[n + i])  # b bits
