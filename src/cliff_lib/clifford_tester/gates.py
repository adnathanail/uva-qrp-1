from qiskit import QuantumCircuit
from qiskit.circuit import Gate


def get_weyl_operator(a: tuple[int, ...], b: tuple[int, ...]) -> Gate:
    """
    Create a gate implementing Weyl operator P_{a,b}.

    P_{a,b} = i^{<a,b>} Z^{a_1} ⊗ ... ⊗ Z^{a_n} X^{b_1} ⊗ ... ⊗ X^{b_n}

    The global phase i^{<a,b>} doesn't affect measurement outcomes, so we omit it.

    Args:
        a: List of n bits controlling Z gates
        b: List of n bits controlling X gates

    Returns:
        Gate implementing P_{a,b}
    """
    assert len(a) == len(b)

    qc = QuantumCircuit(len(a))

    # Apply Z gates where a_i = 1
    for i, ai in enumerate(a):
        if ai:
            qc.z(i)

    # Apply X gates where b_i = 1
    for i, bi in enumerate(b):
        if bi:
            qc.x(i)

    return qc.to_gate(label=f"P_{a},{b}")


def maximally_entangled_state(n: int) -> Gate:
    """
    Create a gate that prepares the maximally entangled state (tensor product of Bell pairs):
    |ψ⟩ = (1/√n) Σ_i |i⟩|i⟩

    This is done by applying H then CNOT to each pair of qubits.

    Args:
        n: Number of qubit pairs (total gate acts on 2n qubits)

    Returns:
        Gate that prepares the maximally entangled state on 2n qubits
    """
    qc = QuantumCircuit(2 * n)

    for i in range(n):
        qc.h(i)
        qc.cx(i, n + i)

    return qc.to_gate(label="Bell")


def weyl_choi_state(n: int, x: tuple[int, ...]) -> Gate:
    """
    Create a gate that prepares the Choi state of Weyl operator P_x: |P_x⟩⟩

    This is done by:
    1. Preparing the maximally entangled state |ψ⟩
    2. Applying (P_x ⊗ I)|ψ⟩

    Args:
        n: Number of qubits per register (gate acts on 2n qubits)
        x: 2n-bit string, split as x = (a, b) where a controls Z, b controls X

    Returns:
        Gate that prepares |P_x⟩⟩ on 2n qubits (first n = register A, last n = register B)
    """
    assert len(x) == 2 * n

    a = x[:n]  # First n bits control Z gates
    b = x[n:]  # Last n bits control X gates

    qc = QuantumCircuit(2 * n)

    # Step 1: Prepare maximally entangled state
    bell = maximally_entangled_state(n)
    qc.append(bell, range(2 * n))

    # Step 2: Apply P_x to the first register only
    P_x = get_weyl_operator(a, b)
    qc.append(P_x, range(n))

    return qc.to_gate(label="|P_x⟩⟩")


def discrete_derivative_circuit(U_circuit: QuantumCircuit, n: int, a_vec: tuple[int, ...]) -> QuantumCircuit:
    """
    Build circuit for the discrete derivative ∂_a⃗ U = w(a⃗) · U · w(a⃗)† · U†.

    Since w(a⃗) is a Weyl/Pauli operator it is self-inverse (up to phase),
    so w(a⃗)† = w(a⃗) (global phase doesn't matter).

    Args:
        U_circuit: n-qubit circuit implementing U
        n: number of qubits
        a_vec: 2n-bit tuple (a, b) specifying the Weyl operator direction

    Returns:
        n-qubit QuantumCircuit implementing ∂_a⃗ U
    """
    assert len(a_vec) == 2 * n

    a = a_vec[:n]
    b = a_vec[n:]

    qc = QuantumCircuit(n)

    # U†
    qc.compose(U_circuit.inverse(), inplace=True)

    # w(a⃗)† = w(a⃗) (Pauli operators are self-inverse up to phase)
    w = get_weyl_operator(a, b)
    qc.append(w, range(n))

    # U
    qc.compose(U_circuit, inplace=True)

    # w(a⃗)
    qc.append(w, range(n))

    return qc


def kth_discrete_derivative_circuit(U_circuit: QuantumCircuit, n: int, a_vectors: list[tuple[int, ...]]) -> QuantumCircuit:
    """
    Build circuit for the k-fold discrete derivative ∂_{a⃗_1} ... ∂_{a⃗_k} U.

    Applied recursively: each derivative wraps the previous result.

    Args:
        U_circuit: n-qubit circuit implementing U
        n: number of qubits
        a_vectors: list of 2n-bit direction vectors

    Returns:
        n-qubit QuantumCircuit implementing the k-fold derivative
    """
    result = U_circuit
    for a_vec in a_vectors:
        result = discrete_derivative_circuit(result, n, a_vec)
    return result


def convolution_3_gate(m: int) -> Gate:
    """
    Build the ⊠_3 convolution gate on 3 copies of m qubits (3m qubits total).

    From Definition 58: per-triple transformation |a,b,c⟩ → |a⊕b⊕c, a⊕b, a⊕c⟩.

    Qubit layout: copy1 [0,m), copy2 [m,2m), copy3 [2m,3m).
    For each qubit position i, the triple is (copy1[i], copy2[i], copy3[i]).

    Circuit per triple (q1, q2, q3):
        1. CNOT(q1→q2), CNOT(q1→q3)   — now q2=a⊕b, q3=a⊕c
        2. CNOT(q2→q1), CNOT(q3→q1)   — now q1=a⊕(a⊕b)⊕(a⊕c)=a⊕b⊕c

    Args:
        m: number of qubits per copy

    Returns:
        Gate on 3m qubits implementing V = U^{⊗m}
    """
    qc = QuantumCircuit(3 * m)

    for i in range(m):
        q1 = i  # copy 1
        q2 = m + i  # copy 2
        q3 = 2 * m + i  # copy 3

        # Step 1: CNOT(q1→q2), CNOT(q1→q3)
        qc.cx(q1, q2)
        qc.cx(q1, q3)

        # Step 2: CNOT(q2→q1), CNOT(q3→q1)
        qc.cx(q2, q1)
        qc.cx(q3, q1)

    return qc.to_gate(label="⊠_3")
