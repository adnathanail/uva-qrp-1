"""Draw the Clifford paired tester circuit for a 1-qubit unitary."""

from cliff_lib.clifford_tester.utils import get_clifford_tester_circuit
from cliff_lib.unitaries import UNITARIES

# Use Hadamard as an example 1-qubit unitary
U_circuit = UNITARIES["cnot"]()
n = 2
x = (1, 1, 0, 1)  # Example Weyl operator

qc = get_clifford_tester_circuit(U_circuit, n, x)


def draw_circuit(num_reps: int):
    # Decompose custom gates so the drawing shows primitive gates
    qc_decomposed = qc.decompose(reps=num_reps)

    fig = qc_decomposed.draw(output="mpl", fold=-1)
    fig.savefig(f"../results/clifford_tester_circuit_{num_reps}.png", dpi=150, bbox_inches="tight")


for i in range(3):
    draw_circuit(i)
