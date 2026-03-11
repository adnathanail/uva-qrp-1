"""Backend resolution and transpilation.

Consolidates backend name types, transpilation functions, and QI provider
logic into a single module.  All QI-specific imports are lazy so that
importing this module never requires a QI connection or the
``qiskit_quantuminspire`` package.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import cache
from typing import Any, Literal, get_args

from qiskit import QuantumCircuit

BackendName = Literal["aer_simulator", "qi_tuna_9"]
_VALID_BACKENDS = set(get_args(BackendName))


def default_transpilation_function(qc: QuantumCircuit) -> QuantumCircuit:
    """Decompose custom gates so AerSimulator can execute them."""
    return qc.decompose(reps=3)


@cache
def _qi_provider() -> Any:
    from qiskit_quantuminspire.qi_provider import QIProvider

    return QIProvider()


def _get_qi_backend_and_transpilation_function(
    backend_name: str,
) -> tuple[Any, Callable[[QuantumCircuit], QuantumCircuit]]:
    from qiskit import transpile

    backend = _qi_provider().get_backend(backend_name)

    match backend_name:
        case "Tuna-9":
            # https://www.quantum-inspire.com/kbase/tuna-operational-specifics/
            qubit_priorities = [4, 1, 2, 6, 7, 0, 3, 5, 8]
        case _:
            qubit_priorities = list(range(100))

    return backend, lambda qc: transpile(qc, backend=backend, initial_layout=qubit_priorities[: qc.num_qubits])


def resolve_backend(
    name: BackendName,
) -> tuple[Any, Callable[[QuantumCircuit], QuantumCircuit], float | None]:
    """Map a backend name to ``(backend_instance, transpile_fn, timeout)``.

    Unlike the previous ``_resolve_backend``, the transpilation function is
    **never** None — AER gets ``default_transpilation_function`` and QI
    backends get their own transpile lambda.

    Imports are lazy so that importing this module doesn't require QI or Aer.
    """
    if name not in _VALID_BACKENDS:
        raise ValueError(f"Unknown backend '{name}'. Valid: {', '.join(sorted(_VALID_BACKENDS))}")

    if name == "aer_simulator":
        from qiskit_aer import AerSimulator

        return AerSimulator(), default_transpilation_function, None

    if name == "qi_tuna_9":
        backend, transpile_fn = _get_qi_backend_and_transpilation_function("Tuna-9")
        return backend, transpile_fn, 300

    raise ValueError(f"Unhandled backend '{name}'")  # unreachable, keeps type-checkers happy
