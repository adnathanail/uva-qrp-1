"""
Compute expected acceptance probability for the Clifford hierarchy tester.

Based on Lemma 3.6 and Eq. 90 from https://arxiv.org/abs/2510.07164.
"""

import numpy as np

from lib.expected_acceptance_probability import expected_acceptance_probability, get_p_table

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
    p_acc = expected_acceptance_probability(TOFFOLI, n)
    print(f"Acceptance probability: {p_acc:.6f}")
    print(f"As fraction: {p_acc} = {int(p_acc * 32)}/32 = 11/32")
