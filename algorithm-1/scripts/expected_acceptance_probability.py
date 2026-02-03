"""
Compute expected acceptance probability for the Clifford hierarchy tester.

Based on Lemma 3.6 and Eq. 90 from https://arxiv.org/abs/2510.07164.
"""

import itertools
from functools import reduce

import numpy as np

# Pauli matrices indexed 0=I, 1=X, 2=Y, 3=Z
PAULI = {
    0: np.array([[1, 0], [0, 1]], dtype=complex),  # I
    1: np.array([[0, 1], [1, 0]], dtype=complex),  # X
    2: np.array([[0, -1j], [1j, 0]], dtype=complex),  # Y
    3: np.array([[1, 0], [0, -1]], dtype=complex),  # Z
}


def pauli_n(labels: list[int]) -> np.ndarray:
    """Generate n-qubit Pauli operator from list of indices.

    E.g. pauli_n([1, 0, 3]) = X ⊗ I ⊗ Z
    """
    return reduce(np.kron, [PAULI[i] for i in labels])


def p_u(U: np.ndarray, x: list[int], y: list[int]) -> float:
    """Compute p_U(x, y) from Lemma 3.6.

    p_U(x,y) = 2^(-4n) * |Tr(P_x U P_y U†)|^2
    """
    nn = len(x)
    Px = pauli_n(x)
    Py = pauli_n(y)
    U_dag = U.conj().T
    trace = np.trace(Px @ U @ Py @ U_dag)
    return (2 ** (-4 * nn)) * np.abs(trace) ** 2


def pauli_labels_for_n(nn: int) -> list[tuple[int, ...]]:
    """Generate all combinations of {0,1,2,3} for length n."""
    return list(itertools.product([0, 1, 2, 3], repeat=nn))


def get_p_table(U: np.ndarray, nn: int) -> np.ndarray:
    """Compute all p_U(x, y) values as a 4^n x 4^n table."""
    labels = pauli_labels_for_n(nn)
    num_labels = len(labels)
    out = np.zeros((num_labels, num_labels))
    for i, x in enumerate(labels):
        for j, y in enumerate(labels):
            out[i, j] = p_u(U, list(x), list(y))
    return out


def p_acc_from_table(p_table: np.ndarray, nn: int) -> float:
    """Compute acceptance probability from Eq. 90.

    p_acc = 2^(2n) * sum(p_U(x,y)^2)
    """
    return (2 ** (2 * nn)) * np.sum(p_table**2)


# Toffoli gate (CCX)
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


if __name__ == "__main__":
    n = 3  # Toffoli is a 3-qubit gate

    print("Computing p_U(x,y) table for Toffoli gate...")
    table = get_p_table(TOFFOLI, n)

    # Check probabilities sum to 1
    prob_sum = np.sum(table)
    print(f"Sum of probabilities: {prob_sum:.6f} (should be 1.0)")

    # Calculate acceptance probability
    p_acc = p_acc_from_table(table, n)
    print(f"Acceptance probability: {p_acc:.6f}")
    print(f"As fraction: {p_acc} = {int(p_acc * 32)}/32 = 11/32")
