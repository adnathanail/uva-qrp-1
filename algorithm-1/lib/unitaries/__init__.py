"""Registry of unitary gates for the Clifford tester harness."""

from lib.unitaries.standard import STANDARD_UNITARIES
from lib.unitaries.stim_random_cliffords import STIM_UNITARIES

UNITARIES = {**STANDARD_UNITARIES, **STIM_UNITARIES}

__all__ = [
    "STANDARD_UNITARIES",
    "STIM_UNITARIES",
    "UNITARIES",
]
