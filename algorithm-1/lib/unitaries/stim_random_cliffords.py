"""Frozen random Clifford unitaries generated via Stim.

Imports below are used by generated code appended by freeze_gate().
"""

from collections.abc import Callable

import numpy as np  # noqa: F401
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate  # noqa: F401

STIM_UNITARIES: dict[str, Callable[[], QuantumCircuit]] = {}
