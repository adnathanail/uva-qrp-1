"""Registry of unitary gates for the Clifford tester harness."""

from lib.unitaries.generators import (
    freeze_stim_clifford,
    stim_random_clifford_circuit,
    stim_random_clifford_gate,
)
from lib.unitaries.standard import STANDARD_UNITARIES
from lib.unitaries.stim_randoms import STIM_UNITARIES

UNITARIES = {**STANDARD_UNITARIES, **STIM_UNITARIES}

__all__ = [
    "STANDARD_UNITARIES",
    "STIM_UNITARIES",
    "UNITARIES",
    "freeze_stim_clifford",
    "stim_random_clifford_circuit",
    "stim_random_clifford_gate",
]
