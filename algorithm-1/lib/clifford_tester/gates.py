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
