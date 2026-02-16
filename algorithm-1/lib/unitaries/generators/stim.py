"""Generate random Clifford unitaries via Stim."""

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
    """Generate a random n-qubit Clifford and freeze it into stim_random_cliffords.py.

    Returns the unique name assigned to the frozen Clifford.
    """
    from pathlib import Path

    from lib.unitaries.generators.freeze import freeze_gate

    gate = stim_random_clifford_gate(n)
    target = Path(__file__).resolve().parent.parent / "stim_random_cliffords.py"
    return freeze_gate(gate, name_prefix="stim_clifford", target_file=target, dict_name="STIM_UNITARIES")
