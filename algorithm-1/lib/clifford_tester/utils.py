from qiskit import QuantumCircuit

from .gates import weyl_choi_state
from .measurements import measure_bell_basis


def default_transpilation_function(qcc: QuantumCircuit) -> QuantumCircuit:
    return qcc.decompose(reps=3)


def get_clifford_tester_circuit(U_circuit: QuantumCircuit, n: int, x: tuple[int, ...]) -> QuantumCircuit:
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
    A = list(range(n))
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


def collision_probability(counts: dict[str, int]) -> float:
    """
    Compute the collision probability from a Qiskit counts dict.

    P(collision) = Σᵢ (count_i / total)²

    This estimates the probability that two independent samples from
    the same distribution would give the same outcome.
    """
    total = sum(counts.values())
    # Protect against DivisionByZero
    if total == 0:
        return 0.0
    return sum((c / total) ** 2 for c in counts.values())
