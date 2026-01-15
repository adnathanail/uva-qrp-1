import numpy as np
from lib import get_weyl_operator, maximally_entangled_state, weyl_choi_state
from tests.utils import QuantumGateMatrixTest


# Pauli matrices
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


class TestTwoQubitWeylOperator(QuantumGateMatrixTest):
    """
    Tests for 2-qubit Weyl operators.

    P_{a,b} applies Z^{a_i} then X^{b_i} to each qubit i.

    Qiskit uses little-endian ordering: qubit 0 is the least significant (rightmost)
    in tensor products. So for a 2-qubit system, the matrix is:
        (operator on qubit 1) ⊗ (operator on qubit 0)
    """

    def test_identity(self):
        """a=[0,0], b=[0,0] → I ⊗ I"""
        self.assert_gate_matrix_equal(get_weyl_operator([0, 0], [0, 0]), np.kron(I, I))

    def test_z_on_qubit_1(self):
        """a=[0,1], b=[0,0] → Z ⊗ I (Z on qubit 1 only)"""
        self.assert_gate_matrix_equal(get_weyl_operator([0, 1], [0, 0]), np.kron(Z, I))

    def test_mixed_zx(self):
        """a=[0,1], b=[1,0] → Z ⊗ X (Z on qubit 1, X on qubit 0)"""
        self.assert_gate_matrix_equal(get_weyl_operator([0, 1], [1, 0]), np.kron(Z, X))

    def test_zx_on_both(self):
        """a=[1,1], b=[1,1] → ZX ⊗ ZX"""
        self.assert_gate_matrix_equal(get_weyl_operator([1, 1], [1, 1]), np.kron(Z @ X, Z @ X))


class TestMaximallyEntangledState(QuantumGateMatrixTest):
    """
    Tests for maximally entangled state preparation.

    For n qubit pairs (2n total qubits), the gate prepares:
        |ψ⟩ = (1/√2^n) Σ_i |i⟩|i⟩

    Each column of the unitary is the image of a basis state.
    """

    def test_1_qubit_pair(self):
        """
        n=1: 2 qubits total (Bell state preparation)

        The gate maps basis states to Bell states:
            |00⟩ → (1/√2)(|00⟩ + |11⟩)
            |01⟩ → (1/√2)(|00⟩ - |11⟩)
            |10⟩ → (1/√2)(|01⟩ + |10⟩)
            |11⟩ → (1/√2)(|10⟩ - |01⟩)
        """
        expected = (1 / np.sqrt(2)) * np.array([
            [1,  1,  0,  0],  # |00⟩ component
            [0,  0,  1, -1],  # |01⟩ component
            [0,  0,  1,  1],  # |10⟩ component
            [1, -1,  0,  0],  # |11⟩ component
        ], dtype=complex)
        self.assert_gate_matrix_equal(maximally_entangled_state(1), expected)

    def test_2_qubit_pairs(self):
        """
        n=2: 4 qubits total

        Circuit: H on qubits 0,1, then CNOT(0→2), CNOT(1→3)
        This creates Bell pairs between (0,2) and (1,3).

        For input |q₃q₂q₁q₀⟩, output is superposition over q₀',q₁' ∈ {0,1}:
            (1/2) Σ (-1)^{q₀·q₀' + q₁·q₁'} |q₃⊕q₁', q₂⊕q₀', q₁', q₀'⟩

        The state |0000⟩ maps to (1/2)(|0000⟩ + |0101⟩ + |1010⟩ + |1111⟩),
        which is the maximally entangled state with registers A={0,1}, B={2,3}.
        """
        # fmt: off
        expected = 0.5 * np.array([
            #  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15   (input columns)
            [  1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # row 0  |0000⟩
            [  0, 0, 0, 0, 1,-1, 1,-1, 0, 0, 0, 0, 0, 0, 0, 0],  # row 1  |0001⟩
            [  0, 0, 0, 0, 0, 0, 0, 0, 1, 1,-1,-1, 0, 0, 0, 0],  # row 2  |0010⟩
            [  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,-1,-1, 1],  # row 3  |0011⟩
            [  0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # row 4  |0100⟩
            [  1,-1, 1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # row 5  |0101⟩
            [  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,-1,-1],  # row 6  |0110⟩
            [  0, 0, 0, 0, 0, 0, 0, 0, 1,-1,-1, 1, 0, 0, 0, 0],  # row 7  |0111⟩
            [  0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],  # row 8  |1000⟩
            [  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,-1, 1,-1],  # row 9  |1001⟩
            [  1, 1,-1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # row 10 |1010⟩
            [  0, 0, 0, 0, 1,-1,-1, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # row 11 |1011⟩
            [  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],  # row 12 |1100⟩
            [  0, 0, 0, 0, 0, 0, 0, 0, 1,-1, 1,-1, 0, 0, 0, 0],  # row 13 |1101⟩
            [  0, 0, 0, 0, 1, 1,-1,-1, 0, 0, 0, 0, 0, 0, 0, 0],  # row 14 |1110⟩
            [  1,-1,-1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # row 15 |1111⟩
        ], dtype=complex)
        # fmt: on
        self.assert_gate_matrix_equal(maximally_entangled_state(2), expected)
