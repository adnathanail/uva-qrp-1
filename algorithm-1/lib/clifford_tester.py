from collections.abc import Callable
from itertools import product

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from .gates import weyl_choi_state
from .measurements import measure_bell_basis


def get_clifford_tester_circuit(U_circuit: QuantumCircuit, n: int, x: tuple[int]) -> QuantumCircuit:
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


def collision_probability(counts: dict) -> float:
    """
    Compute the collision probability from a Qiskit counts dict.

    P(collision) = Σᵢ (count_i / total)²

    This estimates the probability that two independent samples from
    the same distribution would give the same outcome.
    """
    total = sum(counts.values())
    return sum((c / total) ** 2 for c in counts.values())


def clifford_tester(
    U_circuit: QuantumCircuit, n: int, shots: int = 1000, backend=None, transpilation_function: Callable[[QuantumCircuit], QuantumCircuit] | None = None
):
    """
    Four-query Clifford tester algorithm.

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
    if backend is None:
        backend = AerSimulator()

    if transpilation_function is None:

        def transpilation_function(qcc: QuantumCircuit):
            return qcc.decompose(reps=3)

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
