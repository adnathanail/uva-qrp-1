from lib.unitaries.standard import STANDARD_UNITARIES
from lib.unitaries.stim_random_cliffords import STIM_UNITARIES


def gate_source(name: str) -> str:
    if name in STANDARD_UNITARIES:
        return "standard"
    if name in STIM_UNITARIES:
        return "stim_random_cliffords"
    raise ValueError(f"Unknown gate source for '{name}'")
