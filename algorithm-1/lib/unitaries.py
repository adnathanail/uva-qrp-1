"""Registry of unitary gates for the Clifford tester harness."""

from collections.abc import Callable

from qiskit import QuantumCircuit


def hadamard() -> QuantumCircuit:
    qc = QuantumCircuit(1)
    qc.h(0)
    return qc


def s_gate() -> QuantumCircuit:
    qc = QuantumCircuit(1)
    qc.s(0)
    return qc


def identity() -> QuantumCircuit:
    return QuantumCircuit(1)


def t_gate() -> QuantumCircuit:
    qc = QuantumCircuit(1)
    qc.t(0)
    return qc


def rx_0_3() -> QuantumCircuit:
    qc = QuantumCircuit(1)
    qc.rx(0.3, 0)
    return qc


def cnot() -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    return qc


def toffoli() -> QuantumCircuit:
    qc = QuantumCircuit(3)
    qc.ccx(0, 1, 2)
    return qc


UNITARIES: dict[str, Callable[[], QuantumCircuit]] = {
    "hadamard": hadamard,
    "s_gate": s_gate,
    "identity": identity,
    "t_gate": t_gate,
    "rx_0_3": rx_0_3,
    "cnot": cnot,
    "toffoli": toffoli,
}
