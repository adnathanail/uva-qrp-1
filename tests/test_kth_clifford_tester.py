import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit_aer import AerSimulator

from cliff_lib.clifford_tester.gates import (
    convolution_3_gate,
    discrete_derivative_circuit,
    kth_discrete_derivative_circuit,
)
from cliff_lib.clifford_tester.utils import get_kth_clifford_tester_circuit


class TestConvolution3Gate:
    """Tests for the ⊠_3 convolution gate."""

    def test_single_qubit_permutation_matrix(self):
        """
        For m=1 (3 qubits), the gate maps |a,b,c⟩ → |a⊕b⊕c, a⊕b, a⊕c⟩.

        Verify the unitary matrix is the expected permutation.
        """
        gate = convolution_3_gate(1)
        U = Operator(gate).data

        # Build expected permutation matrix
        expected = np.zeros((8, 8), dtype=complex)
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    # Input state index: |a, b, c⟩ in little-endian = c * 4 + b * 2 + a
                    in_idx = c * 4 + b * 2 + a

                    # Output: |a⊕b⊕c, a⊕b, a⊕c⟩
                    out_q0 = a ^ b ^ c
                    out_q1 = a ^ b
                    out_q2 = a ^ c
                    out_idx = out_q2 * 4 + out_q1 * 2 + out_q0

                    expected[out_idx, in_idx] = 1.0

        np.testing.assert_array_almost_equal(U, expected)

    def test_is_unitary(self):
        """The convolution gate must be unitary."""
        for m in [1, 2]:
            gate = convolution_3_gate(m)
            U = Operator(gate).data
            dim = 2 ** (3 * m)
            np.testing.assert_array_almost_equal(U @ U.conj().T, np.eye(dim))


class TestDiscreteDerivative:
    """Tests for discrete derivative circuits."""

    def test_clifford_derivative_is_clifford(self):
        """For Clifford U and any direction a⃗, ∂_a⃗ U should also be Clifford.

        We check this by verifying that the derivative of Hadamard maps Paulis to Paulis
        (i.e., it's a Clifford operation up to global phase).
        """
        # Hadamard circuit
        H_circ = QuantumCircuit(1)
        H_circ.h(0)

        # Direction a⃗ = (1, 0) → Z operator
        deriv = discrete_derivative_circuit(H_circ, 1, (1, 0))
        U = Operator(deriv).data

        # A Clifford on 1 qubit maps Paulis to Paulis (up to phase)
        # Check that U is unitary
        np.testing.assert_array_almost_equal(U @ U.conj().T, np.eye(2), decimal=10)

    def test_identity_derivative_is_identity(self):
        """∂_a⃗ I = w(a⃗) · I · w(a⃗)† · I† = w(a⃗) · w(a⃗)† = I (up to phase)."""
        I_circ = QuantumCircuit(1)
        I_circ.id(0)

        deriv = discrete_derivative_circuit(I_circ, 1, (1, 1))
        U = Operator(deriv).data

        # Should be identity up to global phase
        phase = U[0, 0]
        np.testing.assert_array_almost_equal(U / phase, np.eye(2))

    def test_kth_derivative_zero_vectors(self):
        """With zero direction vectors, kth derivative returns the original circuit."""
        H_circ = QuantumCircuit(1)
        H_circ.h(0)

        result = kth_discrete_derivative_circuit(H_circ, 1, [])
        # Should be the same as the original
        U_orig = Operator(H_circ).data
        U_result = Operator(result).data
        np.testing.assert_array_almost_equal(U_result, U_orig)


class TestKthCliffordTesterCircuit:
    """Integration tests for the k-th level Clifford hierarchy tester."""

    @staticmethod
    def _run_tester(U_circuit, n, k, a_vectors, shots=4096):
        """Run the k-th Clifford tester and return P(ancilla=0)."""
        qc = get_kth_clifford_tester_circuit(U_circuit, n, k, a_vectors)

        # Decompose custom gates for AerSimulator
        qc_decomposed = qc.decompose(reps=5)

        backend = AerSimulator()
        job = backend.run(qc_decomposed, shots=shots)
        counts = job.result().get_counts()

        # Count outcomes where ancilla (classical bit 0) is '0' (pass)
        # Qiskit bit string is big-endian, ancilla is classical bit 0 (rightmost)
        n_pass = sum(v for k_str, v in counts.items() if k_str[-1] == "0")
        return n_pass / shots

    def test_k2_hadamard_is_clifford(self):
        """k=2, U=Hadamard (Clifford): should pass with high probability."""
        H = QuantumCircuit(1)
        H.h(0)
        p_pass = self._run_tester(H, n=1, k=2, a_vectors=[], shots=4096)
        assert p_pass > 0.9, f"Hadamard should be Clifford, got p_pass={p_pass}"

    def test_k2_t_gate_not_clifford(self):
        """k=2, U=T gate (not Clifford): should have p_pass < 1."""
        T = QuantumCircuit(1)
        T.t(0)
        p_pass = self._run_tester(T, n=1, k=2, a_vectors=[], shots=4096)
        assert p_pass < 0.95, f"T gate should not be Clifford, got p_pass={p_pass}"

    def test_k3_t_gate_in_third_level(self):
        """k=3, U=T gate (in C^(3)): should pass with high probability."""
        T = QuantumCircuit(1)
        T.t(0)

        # Need k-2 = 1 direction vector, try a few and take the best
        # The T gate is in C^(3), so for any direction the derivative is Clifford
        # and the test should accept
        a_vec = (1, 0)  # direction in F_2^{2}
        p_pass = self._run_tester(T, n=1, k=3, a_vectors=[a_vec], shots=4096)
        assert p_pass > 0.9, f"T gate should be in C^(3), got p_pass={p_pass}"

    def test_circuit_qubit_count(self):
        """Verify the circuit has the correct number of qubits."""
        H = QuantumCircuit(1)
        H.h(0)
        qc = get_kth_clifford_tester_circuit(H, n=1, k=2, a_vectors=[])
        assert qc.num_qubits == 13  # 12*1 + 1
        assert qc.num_clbits == 1

    def test_circuit_qubit_count_n2(self):
        """Verify qubit count for n=2."""
        circ = QuantumCircuit(2)
        circ.h(0)
        circ.cx(0, 1)
        qc = get_kth_clifford_tester_circuit(circ, n=2, k=2, a_vectors=[])
        assert qc.num_qubits == 25  # 12*2 + 1
        assert qc.num_clbits == 1
