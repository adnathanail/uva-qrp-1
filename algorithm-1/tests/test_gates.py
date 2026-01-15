import numpy as np
from lib import get_weyl_operator, maximally_entangled_state, weyl_choi_state
from qiskit.quantum_info import Operator


# Pauli matrices
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


class TestTwoQubitWeylOperator:
    """
    Tests for 2-qubit Weyl operators.

    P_{a,b} applies Z^{a_i} then X^{b_i} to each qubit i.

    Qiskit uses little-endian ordering: qubit 0 is the least significant (rightmost)
    in tensor products. So for a 2-qubit system, the matrix is:
        (operator on qubit 1) ⊗ (operator on qubit 0)
    """

    def assert_weyl_operator_correct(self, expected: np.ndarray, a: list, b: list):
        """Assert that get_weyl_operator(a, b) produces the expected matrix."""
        gate = get_weyl_operator(a, b)
        actual = Operator(gate).data
        np.testing.assert_array_almost_equal(actual, expected)

    def test_identity(self):
        """a=[0,0], b=[0,0] → I ⊗ I"""
        expected = np.kron(I, I)
        self.assert_weyl_operator_correct(expected, a=[0, 0], b=[0, 0])

    def test_z_on_qubit_1(self):
        """a=[0,1], b=[0,0] → Z ⊗ I (Z on qubit 1 only)"""
        expected = np.kron(Z, I)
        self.assert_weyl_operator_correct(expected, a=[0, 1], b=[0, 0])

    def test_mixed_zx(self):
        """a=[0,1], b=[1,0] → Z ⊗ X (Z on qubit 1, X on qubit 0)"""
        expected = np.kron(Z, X)
        self.assert_weyl_operator_correct(expected, a=[0, 1], b=[1, 0])

    def test_zx_on_both(self):
        """a=[1,1], b=[1,1] → ZX ⊗ ZX"""
        expected = np.kron(Z @ X, Z @ X)
        self.assert_weyl_operator_correct(expected, a=[1, 1], b=[1, 1])


class TestMaximallyEntangledState:
    """
    Tests for maximally entangled state preparation.

    For n qubit pairs (2n total qubits), the gate prepares:
        |ψ⟩ = (1/√2^n) Σ_i |i⟩|i⟩

    Each column of the unitary is the image of a basis state.
    """

    def assert_maximally_entangled_correct(self, expected: np.ndarray, n: int):
        """Assert that maximally_entangled_state(n) produces the expected matrix."""
        gate = maximally_entangled_state(n)
        actual = Operator(gate).data
        np.testing.assert_array_almost_equal(actual, expected)

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
        self.assert_maximally_entangled_correct(expected, n=1)
