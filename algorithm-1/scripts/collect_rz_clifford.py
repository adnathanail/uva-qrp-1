"""Collect Rz(theta) Clifford tester data using AerSimulator.

Saves one JSON file per backend to algorithm-1/results/rz_clifford/.

File format (e.g. aer_depol_0.0100.json):
    {
        "backend_label": "aer_depol_0.0100",
        "depolarizing": 0.01,
        "theta_values": [0.0, ...],
        "shots": 1000,
        "repeats": 5,
        "acceptance_rates": [
            [0.95, 0.93, 0.94, 0.96, 0.92],   // one list per theta, one float per repeat
            [null, null, null, null, null],     // null = not yet collected
            ...
        ]
    }

Usage:
    uv run python algorithm-1/scripts/collect_rz_clifford.py --theta-steps 9 --depolarizing-list 0.01,0.05,0.1 --repeats 5
"""

import argparse
import json
import math
import tempfile
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

from lib.clifford_tester import clifford_tester_batched
from lib.state import BatchedRawResults

RESULTS_DIR = Path(__file__).parent.parent / "results" / "rz_clifford"


def _parse_float_list(raw: str) -> list[float]:
    parts = [p.strip() for p in raw.replace(";", ",").split(",")]
    return [float(p) for p in parts if p]


def _build_backend(depolarizing: float | None):
    if depolarizing is None or depolarizing <= 0:
        return AerSimulator()

    noise_model = NoiseModel()
    error_1q = depolarizing_error(depolarizing, 1)
    error_2q = depolarizing_error(depolarizing, 2)
    noise_model.add_all_qubit_quantum_error(
        error_1q,
        ["u", "u1", "u2", "u3", "rx", "ry", "rz", "x", "y", "z", "h", "s", "sdg", "t", "tdg", "sx", "id"],
    )
    noise_model.add_all_qubit_quantum_error(error_2q, ["cx", "cz", "swap"])
    return AerSimulator(noise_model=noise_model)


def _backend_label(depolarizing: float | None) -> str:
    if depolarizing is None or depolarizing <= 0:
        return "aer_noiseless"
    return f"aer_depol_{depolarizing:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Rz(theta) Clifford tester data using AerSimulator.")
    parser.add_argument("--shots", type=int, default=1000, help="Shots per Weyl operator.")
    parser.add_argument("--theta-steps", type=int, default=9, help="Number of theta points from 0 to 2pi (inclusive).")
    parser.add_argument(
        "--depolarizing-list",
        type=str,
        default="0.001,0.01,0.05,0.1",
        help="Comma-separated depolarizing rates. Include 0 for noiseless.",
    )
    parser.add_argument("--repeats", type=int, default=5, help="Number of repeats per theta (for mean/std bands).")
    args = parser.parse_args()

    theta_values: list[float] = np.linspace(0.0, 2 * math.pi, args.theta_steps).tolist()
    depol_values = sorted({float(v) for v in _parse_float_list(args.depolarizing_list)})

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for depol in depol_values:
        label = _backend_label(depol)
        backend = _build_backend(depol)

        def transpile_fn(qc: QuantumCircuit, _backend=backend) -> QuantumCircuit:
            return transpile(qc, backend=_backend)

        data_file = RESULTS_DIR / f"{label}.json"

        # Load existing data if parameters match, otherwise start fresh
        record = None
        if data_file.exists():
            existing = json.loads(data_file.read_text())
            if existing["theta_values"] == theta_values and existing["shots"] == args.shots and existing["repeats"] == args.repeats:
                record = existing
            else:
                print(f"  Parameters changed — overwriting existing data for [{label}]")
        if record is None:
            record = {
                "backend_label": label,
                "depolarizing": depol,
                "theta_values": theta_values,
                "shots": args.shots,
                "repeats": args.repeats,
                "acceptance_rates": [[None] * args.repeats for _ in theta_values],
            }

        print(f"\n[{label}] {len(theta_values)} theta values x {args.repeats} repeats")

        for i, theta in enumerate(theta_values):
            for rep in range(args.repeats):
                if record["acceptance_rates"][i][rep] is not None:
                    print(f"  theta={theta:.4f} rep={rep} — skipping (already done)")
                    continue

                print(f"  theta={theta:.4f} rep={rep} — running...")
                qc = QuantumCircuit(1)
                qc.rz(theta, 0)

                with tempfile.TemporaryDirectory() as tmp:
                    raw = clifford_tester_batched(
                        qc,
                        1,
                        shots=args.shots,
                        backend=backend,
                        transpilation_function=transpile_fn,
                        checkpoint_dir=Path(tmp),
                    )

                rate = BatchedRawResults.from_tuples(raw).summarise()
                record["acceptance_rates"][i][rep] = rate
                data_file.write_text(json.dumps(record, indent=2))
                print(f"    p_acc = {rate:.4f}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
