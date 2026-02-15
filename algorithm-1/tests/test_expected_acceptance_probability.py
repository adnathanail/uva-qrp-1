import numpy as np
import pytest
from qiskit import QuantumCircuit

from lib.expected_acceptance_probability import (
    expected_acceptance_probability,
    expected_acceptance_probability_from_circuit,
    get_p_table,
)

# Common gate matrices
HADAMARD = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)

TOFFOLI = np.array(
    [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 0],
    ],
    dtype=complex,
)


class TestProbabilityTableProperties:
    """The p_U(x, y) table must satisfy known mathematical properties."""

    def test_hadamard_table_sums_to_one(self):
        """Sum of all p_U(x, y) must equal 1 for any unitary."""
        table = get_p_table(HADAMARD, 1)
        assert table.sum() == pytest.approx(1.0)

    def test_toffoli_table_sums_to_one(self):
        table = get_p_table(TOFFOLI, 3)
        assert table.sum() == pytest.approx(1.0)

    def test_all_entries_non_negative(self):
        table = get_p_table(HADAMARD, 1)
        assert (table >= 0).all()


class TestCliffordGatesAcceptWithProbabilityOne:
    """Clifford gates must have acceptance probability exactly 1.0."""

    def test_hadamard(self):
        assert expected_acceptance_probability(HADAMARD, 1) == pytest.approx(1.0)

    def test_identity_1q(self):
        assert expected_acceptance_probability(np.eye(2, dtype=complex), 1) == pytest.approx(1.0)

    def test_s_gate(self):
        S = np.array([[1, 0], [0, 1j]], dtype=complex)
        assert expected_acceptance_probability(S, 1) == pytest.approx(1.0)

    def test_cnot(self):
        CNOT = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
            dtype=complex,
        )
        assert expected_acceptance_probability(CNOT, 2) == pytest.approx(1.0)

    def test_hadamard_from_circuit(self):
        H = QuantumCircuit(1)
        H.h(0)
        assert expected_acceptance_probability_from_circuit(H) == pytest.approx(1.0)


class TestNonCliffordGates:
    """Non-Clifford gates must have acceptance probability < 1.0."""

    def test_t_gate(self):
        T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
        assert expected_acceptance_probability(T, 1) == pytest.approx(0.75)

    def test_toffoli(self):
        assert expected_acceptance_probability(TOFFOLI, 3) == pytest.approx(0.34375)  # 11/32

    def test_t_gate_from_circuit(self):
        T = QuantumCircuit(1)
        T.t(0)
        assert expected_acceptance_probability_from_circuit(T) == pytest.approx(0.75)
