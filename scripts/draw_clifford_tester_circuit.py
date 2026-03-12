"""Draw the Clifford paired tester circuit for a 1-qubit unitary."""

from cliff_lib.clifford_tester.utils import get_kth_clifford_tester_circuit
from cliff_lib.unitaries import UNITARIES

# Use Hadamard as an example 1-qubit unitary
U_circuit = UNITARIES["cnot"]()
n = 2
x = (0, 1, 0, 1)  # Example Weyl operator

qc = get_kth_clifford_tester_circuit(U_circuit, n, 3, [(0, 1, 1, 0)])

# Decompose custom gates so the drawing shows primitive gates
qc_decomposed = qc.decompose(reps=2)

# Draw to file
fig = qc_decomposed.draw(output="mpl", fold=-1)
fig.savefig("../results/clifford_tester_circuit.png", dpi=150, bbox_inches="tight")
print("Saved to results/clifford_tester_circuit.png")
