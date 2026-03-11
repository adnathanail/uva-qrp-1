"""Registry of unitary gates for the Clifford tester harness."""

from .standard import STANDARD_UNITARIES
from .stim_random_cliffords import STIM_UNITARIES
from .utils import gate_source

# Ensure no collisions between the dictionaries
_overlap = set(STANDARD_UNITARIES) & set(STIM_UNITARIES)
if _overlap:
    raise ValueError(f"Duplicate unitary names: {_overlap}")

UNITARIES = {**STANDARD_UNITARIES, **STIM_UNITARIES}

__all__ = [
    "STANDARD_UNITARIES",
    "STIM_UNITARIES",
    "UNITARIES",
    "gate_source",
]
