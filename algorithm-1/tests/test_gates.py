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
