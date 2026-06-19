from qiskit import QuantumCircuit

qc = QuantumCircuit(2)
qc.z(0)
qc.z(1)
qc.x(1)

fig = qc.draw(output="mpl", fold=-1)
fig.savefig(f"../results/example_qiskit_circuit.png", dpi=150, bbox_inches="tight")
