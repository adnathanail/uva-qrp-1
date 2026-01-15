import numpy as np
from qiskit.circuit import Gate
from qiskit.quantum_info import Operator


class QuantumGateMatrixTest:
    """Base class for tests that compare quantum gate matrices."""

    def assert_gate_matrix_equal(self, gate: Gate, expected: np.ndarray):
        """Assert that a gate's matrix representation matches the expected matrix."""
        actual = Operator(gate).data
        np.testing.assert_array_almost_equal(actual, expected)
