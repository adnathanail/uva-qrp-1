from qiskit import QuantumCircuit

from .gates import (
    convolution_3_gate,
    kth_discrete_derivative_circuit,
    maximally_entangled_state,
    weyl_choi_state,
)
from .measurements import measure_bell_basis


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


def get_kth_clifford_tester_circuit(
    U_circuit: QuantumCircuit,
    n: int,
    k: int,
    a_vectors: list[tuple[int, ...]],
) -> QuantumCircuit:
    """
    Build the circuit for the k-th level Clifford hierarchy test.

    Uses discrete derivatives, ⊠_3 convolution, and a swap test.
    Based on Section 12 of arXiv:2508.15908.

    Args:
        U_circuit: n-qubit circuit implementing the unitary under test
        n: number of qubits
        k: hierarchy level (k ≥ 2)
        a_vectors: k-2 direction vectors, each a 2n-bit tuple from F_2^{2n}

    Returns:
        QuantumCircuit with 12n+1 qubits and 1 classical bit (ancilla measurement)
    """
    assert k >= 2
    assert len(a_vectors) == k - 2

    total_qubits = 12 * n + 1
    qc = QuantumCircuit(total_qubits, 1)

    # Register layout (each copy has 2n qubits for the Choi state: n for I, n for V)
    # Triple 1: copies A [0,2n), B [2n,4n), C [4n,6n)
    # Triple 2: copies A [6n,8n), B [8n,10n), C [10n,12n)
    # Swap ancilla: 12n

    # Step 1: Compute V = (k-2)-fold discrete derivative of U
    V_circuit = kth_discrete_derivative_circuit(U_circuit, n, a_vectors) if a_vectors else U_circuit

    # Step 2: Prepare 6 Choi states (I ⊗ V)|Φ⟩
    # Each copy occupies 2n qubits: first n = identity register, last n = V register
    for copy_start in range(0, 12 * n, 2 * n):
        I_reg = list(range(copy_start, copy_start + n))
        V_reg = list(range(copy_start + n, copy_start + 2 * n))

        # Prepare maximally entangled state |Φ⟩ on 2n qubits
        bell = maximally_entangled_state(n)
        qc.append(bell, I_reg + V_reg)

        # Apply V to the second register
        qc.compose(V_circuit, qubits=V_reg, inplace=True)

    qc.barrier()

    # Step 3: Apply ⊠_3 convolution to each triple
    # Triple 1: copies at offsets 0, 2n, 4n (each 2n qubits)
    conv = convolution_3_gate(2 * n)
    triple1_qubits = list(range(0, 6 * n))
    qc.append(conv, triple1_qubits)

    # Triple 2: copies at offsets 6n, 8n, 10n (each 2n qubits)
    triple2_qubits = list(range(6 * n, 12 * n))
    qc.append(conv, triple2_qubits)

    qc.barrier()

    # Step 4: Swap test between kept copies (copy A of each triple)
    # Triple 1 copy A: [0, 2n), Triple 2 copy A: [6n, 8n)
    ancilla = 12 * n

    qc.h(ancilla)

    for i in range(2 * n):
        q1 = i  # Triple 1, copy A
        q2 = 6 * n + i  # Triple 2, copy A
        qc.cswap(ancilla, q1, q2)

    qc.h(ancilla)

    # Step 5: Measure ancilla
    qc.measure(ancilla, 0)

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
