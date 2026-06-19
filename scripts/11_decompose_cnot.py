"""Build a CNOT circuit and print its Tuna-9 transpilation."""

from qiskit.circuit import QuantumCircuit

from cliff_lib.backends import resolve_backend

_, transpile_fn, _ = resolve_backend("qi_tuna_9")

qc = QuantumCircuit(2)
qc.cx(0, 1)

transpiled = transpile_fn(qc)

print("=== Original ===")
print(qc.draw())
print()
print("=== Transpiled for Tuna-9 ===")
print(transpiled.draw())
print(f"depth: {transpiled.depth()}")
