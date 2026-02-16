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


def _n_hadamard_cnot(n: int) -> QuantumCircuit:
    out = QuantumCircuit(n)
    for i in range(n):
        out.h(i)
    for i in range(n - 1):
        out.cx(i, i + 1)
    return out


def c_4_hadamard_3_cnot() -> QuantumCircuit:
    return _n_hadamard_cnot(4)


def _n_t_gate(n: int):
    out = QuantumCircuit(n)
    for i in range(n):
        out.t(i)
    return out


def c_4_t_gate():
    return _n_t_gate(4)


STANDARD_UNITARIES: dict[str, Callable[[], QuantumCircuit]] = {
    "hadamard": hadamard,
    "s_gate": s_gate,
    "identity": identity,
    "t_gate": t_gate,
    "rx_0_3": rx_0_3,
    "cnot": cnot,
    "toffoli": toffoli,
    "c_4_hadamard_3_cnot": c_4_hadamard_3_cnot,
    "c_4_t_gate": c_4_t_gate,
}
