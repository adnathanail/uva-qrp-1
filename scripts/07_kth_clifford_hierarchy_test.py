from cliff_lib.backends import resolve_backend
from cliff_lib.clifford_tester.testers import kth_clifford_tester
from cliff_lib.unitaries import STANDARD_UNITARIES

SHOTS = 1000
NUM_A_SAMPLES = 10
K_LEVELS = [2, 3]
MAX_N = 1  # 12n+1 = 13 qubits for n=1

gate_names = [name for name, factory in STANDARD_UNITARIES.items() if factory().num_qubits <= MAX_N]

backend, transpile_fn, _timeout = resolve_backend("aer_simulator")

# Header
k_cols = "".join(f"{'k=' + str(k):>12}" for k in K_LEVELS)
print(f"{'Gate':<24}{k_cols}")
print("-" * (24 + 12 * len(K_LEVELS)))

# Run tests
for name in gate_names:
    U_circuit = STANDARD_UNITARIES[name]()
    n = U_circuit.num_qubits
    row = f"{name:<24}"
    for k in K_LEVELS:
        rate = kth_clifford_tester(U_circuit, n, k, shots=SHOTS, num_a_samples=NUM_A_SAMPLES, backend=backend, transpilation_function=transpile_fn)
        row += f"{rate:>12.6f}"
    print(row)
