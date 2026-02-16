"""Generate and store random Clifford unitaries via Stim."""

import hashlib
import textwrap
from pathlib import Path

import numpy as np
import stim
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate


def stim_random_clifford_gate(n: int) -> UnitaryGate:
    """Generate a uniformly random n-qubit Clifford as a Qiskit UnitaryGate.

    Uses Stim's Tableau.random (Bravyi & Maslov algorithm) for uniform sampling.
    The endian='little' parameter matches Qiskit's qubit ordering convention.
    """
    tab = stim.Tableau.random(n)
    unitary = tab.to_unitary_matrix(endian="little").astype(np.complex128)
    return UnitaryGate(unitary)


def stim_random_clifford_circuit(n: int) -> QuantumCircuit:
    """Generate a uniformly random n-qubit Clifford as a QuantumCircuit.

    Returns a single-gate circuit wrapping the random Clifford unitary,
    compatible with the tester harness.
    """
    gate = stim_random_clifford_gate(n)
    qc = QuantumCircuit(n)
    qc.append(gate, range(n))
    return qc


def freeze_stim_clifford(n: int) -> str:
    """Generate a random n-qubit Clifford and freeze it into stim_randoms.py.

    Returns the unique name assigned to the frozen Clifford.
    """
    gate = stim_random_clifford_gate(n)
    matrix = np.array(gate.to_matrix())

    # Compute unique name from matrix content
    matrix_hash = hashlib.sha256(matrix.tobytes()).hexdigest()[:8]
    name = f"stim_clifford_{n}q_{matrix_hash}"

    # Format the matrix as a reproducible numpy expression
    matrix_repr = repr(matrix.tolist())

    # Generate the function definition and registration
    code = textwrap.dedent(f"""\

        def {name}() -> QuantumCircuit:
            matrix = np.array({matrix_repr}, dtype=np.complex128)
            qc = QuantumCircuit({n})
            qc.append(UnitaryGate(matrix), range({n}))
            return qc

        STIM_UNITARIES["{name}"] = {name}
    """)

    # Append to stim_randoms.py
    stim_randoms_path = Path(__file__).parent / "stim_randoms.py"
    with open(stim_randoms_path, "a") as f:
        f.write(code)

    return name
