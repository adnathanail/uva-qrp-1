"""Generic gate freezing utility."""

import hashlib
import textwrap
from pathlib import Path

import numpy as np
from qiskit.circuit.library import UnitaryGate


def _gate_name(prefix: str, gate: UnitaryGate) -> str:
    """Generate a unique name for a gate based on its matrix content.

    Uses SHA-256 of matrix bytes, first 8 hex chars. Combined with
    the prefix this gives names like ``stim_clifford_1q_a3b4c5d6``.
    """
    matrix = np.array(gate.to_matrix())
    matrix_hash = hashlib.sha256(matrix.tobytes()).hexdigest()[:8]
    return f"{prefix}_{gate.num_qubits}q_{matrix_hash}"


def freeze_gate(
    gate: UnitaryGate,
    *,
    name_prefix: str,
    target_file: Path,
    dict_name: str,
) -> str:
    """Append a frozen gate function and dict registration to target_file.

    The gate is assigned a unique name based on ``name_prefix`` and a hash
    of its matrix content. Returns the generated name.

    The target file must already import numpy as np, QuantumCircuit,
    and UnitaryGate at the top level.
    """
    # Check that the generated Python code will be valid
    if not name_prefix.isidentifier():
        raise ValueError(f"name_prefix must be a valid identifier, got {name_prefix!r}")
    if not dict_name.isidentifier():
        raise ValueError(f"dict_name must be a valid identifier, got {dict_name!r}")

    name = _gate_name(name_prefix, gate)
    matrix = np.array(gate.to_matrix())
    n = gate.num_qubits
    matrix_repr = repr(matrix.tolist())

    code = textwrap.dedent(f"""\

        def {name}() -> QuantumCircuit:
            matrix = np.array({matrix_repr}, dtype=np.complex128)
            qc = QuantumCircuit({n})
            qc.append(UnitaryGate(matrix), range({n}))
            return qc

        {dict_name}["{name}"] = {name}
    """)

    with open(target_file, "a") as f:
        f.write(code)

    return name
