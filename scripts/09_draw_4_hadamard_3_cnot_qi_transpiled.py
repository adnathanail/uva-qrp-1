# uv run scripts/09_draw_4_hadamard_3_cnot_qi_transpiled.py
"""Draw the 4-Hadamard 3-CNOT circuit, both as written and transpiled for QI Tuna-9."""

# Workaround: the installed compute_api_client BackendStatus enum is missing
# 'benchmarking', which the QI server currently returns for Tuna-9 — without
# this the QIProvider() constructor fails with a pydantic ValidationError.
from compute_api_client.models import backend_status as _bs

# Alias 'benchmarking' to IDLE so pydantic enum validation accepts it.
_bs.BackendStatus._value2member_map_.setdefault("benchmarking", _bs.BackendStatus.IDLE)

from qiskit import transpile  # noqa: E402

from cliff_lib.backends import resolve_backend  # noqa: E402
from cliff_lib.unitaries import UNITARIES  # noqa: E402

qc = UNITARIES["c_4_hadamard_3_cnot"]()

backend, _, _ = resolve_backend("qi_tuna_9")
qubit_priorities = [4, 1, 2, 6, 7, 0, 3, 5, 8]
basis_gates = ["rx", "ry", "rz", "cz"]
qc_transpiled = transpile(
    qc,
    backend=backend,
    basis_gates=basis_gates,
    initial_layout=qubit_priorities[: qc.num_qubits],
    optimization_level=3,
)

print(f"Original depth: {qc.depth()}, gates: {qc.count_ops()}")
print(f"Transpiled depth: {qc_transpiled.depth()}, gates: {qc_transpiled.count_ops()}")

original_path = "results/c_4_hadamard_3_cnot.png"
fig = qc.draw(output="mpl", fold=-1)
fig.savefig(original_path, dpi=150, bbox_inches="tight")
print(f"Saved to {original_path}")

transpiled_path = "results/c_4_hadamard_3_cnot_qi_tuna_9_transpiled.png"
fig = qc_transpiled.draw(output="mpl", fold=-1, idle_wires=False)
fig.savefig(transpiled_path, dpi=150, bbox_inches="tight")
print(f"Saved to {transpiled_path}")
