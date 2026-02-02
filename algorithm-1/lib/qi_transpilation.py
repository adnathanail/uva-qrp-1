from qiskit import transpile
from qiskit_quantuminspire.qi_provider import QIProvider

provider = QIProvider()


def get_backend_and_transpilation_function(backend_name: str):
    backend = provider.get_backend("QX emulator")

    match backend_name:
        case "Tuna-9":
            # https://www.quantum-inspire.com/kbase/tuna-operational-specifics/
            qubit_priorities = [4, 1, 2, 6, 7, 0, 3, 5, 8]
        case _:
            # Default to numerical order
            qubit_priorities = list(range(100))

    return backend, lambda qc: transpile(qc, backend=backend, initial_layout=qubit_priorities[: qc.num_qubits])
