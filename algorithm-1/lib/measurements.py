from qiskit import QuantumCircuit


def measure_bell_basis(qc: QuantumCircuit, qubits_A: list, qubits_B: list, clbits: list):
    """
    Measure in the Bell basis {|P_y⟩⟩⟨⟨P_y|}_y

    To measure in the Bell basis, we undo the Bell state preparation
    (CNOT then H) and measure in the computational basis.

    The measurement outcome directly gives us y = (a, b) where:
    - The A register measurement gives 'a' (Z info, via H converting phase to bit)
    - The B register measurement gives 'b' (X info, via CNOT propagating bit flips)

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

    # Measure: A register gives 'a' (Z info), B register gives 'b' (X info)
    # Store as y = (a, b) = (A measurement, B measurement)
    for i in range(n):
        qc.measure(qubits_A[i], clbits[i])  # a bits (Z info)
        qc.measure(qubits_B[i], clbits[n + i])  # b bits (X info)
