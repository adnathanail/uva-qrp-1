"""Run Clifford tester on all standard unitaries across all backends.

Usage:
    uv run python algorithm-1/scripts/collect_full_results_for_standard_unitaries.py              # run all
    uv run python algorithm-1/scripts/collect_full_results_for_standard_unitaries.py hadamard t_gate  # run specific ones
"""

import sys

from lib.backends import BackendName
from lib.result_collection import collect_results_for_unitary
from lib.unitaries import STANDARD_UNITARIES

SHOTS = 1000
BACKENDS: list[BackendName] = ["aer_simulator", "qi_tuna_9"]


def main():
    gates = STANDARD_UNITARIES
    if len(sys.argv) > 1:
        names = sys.argv[1:]
        unknown = set(names) - set(STANDARD_UNITARIES)
        if unknown:
            print(f"Unknown unitaries: {', '.join(sorted(unknown))}")
            print(f"Available: {', '.join(STANDARD_UNITARIES)}")
            sys.exit(1)
        gates = {k: v for k, v in STANDARD_UNITARIES.items() if k in names}

    for name, make_circuit in gates.items():
        print(f"\n{'=' * 50}")
        print(f"Gate: {name}")
        print(f"{'=' * 50}")
        for backend in BACKENDS:
            collect_results_for_unitary(name, make_circuit(), backend, shots=SHOTS)


if __name__ == "__main__":
    main()
