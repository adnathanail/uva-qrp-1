"""Frozen random Clifford unitaries generated via Stim."""

from collections.abc import Callable

from qiskit import QuantumCircuit

STIM_UNITARIES: dict[str, Callable[[], QuantumCircuit]] = {}
