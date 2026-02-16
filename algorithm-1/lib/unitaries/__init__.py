"""Registry of unitary gates for the Clifford tester harness."""

from lib.unitaries.generators import (
    freeze_gate,
    stim_random_clifford_circuit,
    stim_random_clifford_gate,
)
from lib.unitaries.generators.stim import freeze_stim_clifford
from lib.unitaries.standard import STANDARD_UNITARIES
from lib.unitaries.stim_random_cliffords import STIM_UNITARIES

UNITARIES = {**STANDARD_UNITARIES, **STIM_UNITARIES}

__all__ = [
    "STANDARD_UNITARIES",
    "STIM_UNITARIES",
    "UNITARIES",
    "freeze_gate",
    "freeze_stim_clifford",
    "stim_random_clifford_circuit",
    "stim_random_clifford_gate",
]
