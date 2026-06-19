"""Generate 5 random 2-qubit Clifford unitaries, transpile for Tuna-9, and print."""

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate

from cliff_lib.backends import resolve_backend
from cliff_lib.unitaries.generators import stim_random_clifford_gate

_, transpile_fn, _ = resolve_backend("qi_tuna_9")

for i in range(5):
    U = stim_random_clifford_gate(2)
    qc = QuantumCircuit(2)
    qc.append(UnitaryGate(U), [0, 1])
    transpiled = transpile_fn(qc)

    print(f"=== Unitary {i + 1} ===")
    print(transpiled.draw())
    print(transpiled.depth())
    print()
