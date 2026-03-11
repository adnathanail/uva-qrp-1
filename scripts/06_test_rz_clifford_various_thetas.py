"""Collect Rz(theta) Clifford tester data and plot results.

Saves one JSON file per backend to results/rz_clifford/.
Re-running is safe: completed repeats are skipped automatically.
Parameters changed? The existing file is overwritten and collection restarts.

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
    # Collect + plot (default)
    uv run python scripts/collect_rz_clifford.py --theta-steps 9 --depolarizing-list 0.01,0.05,0.1 --repeats 5

    # Plot only from existing data
    uv run python scripts/collect_rz_clifford.py plot

    # Real hardware via cliff_lib/backends.py (ignores --depolarizing-list)
    uv run python scripts/collect_rz_clifford.py --backend qi_tuna_9 --theta-steps 9 --repeats 1
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import TypedDict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

from cliff_lib.clifford_tester import clifford_tester_batched
from cliff_lib.expected_acceptance_probability import expected_acceptance_probability_from_circuit
from cliff_lib.state import BatchedRawResults
from cliff_lib.state.utils import atomic_write

RESULTS_DIR = Path(__file__).parent.parent / "results" / "rz_clifford"


class RzRecord(TypedDict):
    backend_label: str
    depolarizing: float | None
    theta_values: list[float]
    shots: int
    repeats: int
    acceptance_rates: list[list[float | None]]


def _parse_float_list(raw: str) -> list[float]:
    parts = [p.strip() for p in raw.replace(";", ",").split(",")]
    return [float(p) for p in parts if p]


def _build_aer_backend(depolarizing: float | None):
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


def _aer_label(depolarizing: float | None) -> str:
    if depolarizing is None or depolarizing <= 0:
        return "aer_noiseless"
    return f"aer_depol_{depolarizing:.4f}"


def _run_backend(
    label: str,
    backend,
    transpile_fn,
    theta_values: list[float],
    shots: int,
    repeats: int,
    depolarizing: float | None = None,
    timeout: float | None = None,
) -> None:
    data_file = RESULTS_DIR / f"{label}.json"

    record: RzRecord | None = None
    if data_file.exists():
        existing: RzRecord = json.loads(data_file.read_text())
        # Check theta grid is the same size, and the theta values are the same(ish)
        same_theta_grid = len(existing["theta_values"]) == len(theta_values) and np.allclose(existing["theta_values"], theta_values)
        if same_theta_grid and existing["shots"] == shots and existing["repeats"] == repeats:
            record = existing
        else:
            print(f"  Parameters changed — overwriting existing data for [{label}]")
    if record is None:
        record = RzRecord(
            backend_label=label,
            depolarizing=depolarizing,
            theta_values=theta_values,
            shots=shots,
            repeats=repeats,
            acceptance_rates=[[None] * repeats for _ in theta_values],
        )

    print(f"\n[{label}] {len(theta_values)} theta values x {repeats} repeats")

    for i, theta in enumerate(theta_values):
        for rep in range(repeats):
            if record["acceptance_rates"][i][rep] is not None:
                print(f"  theta={theta:.4f} rep={rep} — skipping (already done)")
                continue

            print(f"  theta={theta:.4f} rep={rep} — running...")
            qc = QuantumCircuit(1)
            qc.rz(theta, 0)

            checkpoint_dir = RESULTS_DIR / "checkpoints" / f"{label}_theta{i:04d}_rep{rep:02d}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            raw = clifford_tester_batched(
                qc,
                1,
                shots=shots,
                backend=backend,
                transpilation_function=transpile_fn,
                checkpoint_dir=checkpoint_dir,
                timeout=timeout,
            )

            rate = BatchedRawResults.from_tuples(raw).summarise()
            record["acceptance_rates"][i][rep] = rate
            atomic_write(data_file, json.dumps(record, indent=2))
            print(f"    p_acc = {rate:.4f}")


def _load_backend_data(data_file: Path) -> dict | None:
    record = json.loads(data_file.read_text())
    theta_values: list[float] = record["theta_values"]
    rates: list[list[float | None]] = record["acceptance_rates"]

    means: list[float] = []
    stds: list[float] = []
    n_collected: list[int] = []

    for row in rates:
        samples = [v for v in row if v is not None]
        means.append(float(np.mean(samples)) if samples else float("nan"))
        stds.append(float(np.std(samples, ddof=1)) if len(samples) > 1 else 0.0)
        n_collected.append(len(samples))

    return {
        "backend_label": record["backend_label"],
        "depolarizing": record.get("depolarizing"),
        "theta_values": theta_values,
        "means": means,
        "stds": stds,
        "n_collected": n_collected,
    }


def plot(plot_file: Path | None = None) -> None:
    if not RESULTS_DIR.exists():
        print(f"Results directory not found: {RESULTS_DIR}")
        return

    backends_data = []
    for data_file in sorted(RESULTS_DIR.glob("*.json")):
        data = _load_backend_data(data_file)
        if data is not None:
            backends_data.append(data)

    if not backends_data:
        print(f"No backend data found in {RESULTS_DIR}")
        return

    print(f"Found {len(backends_data)} backend(s): {[d['backend_label'] for d in backends_data]}")

    # Theory line over a fine grid
    theory_thetas = np.linspace(0.0, 2 * math.pi, 300).tolist()
    theory_values = []
    for theta_rad in theory_thetas:
        qc = QuantumCircuit(1)
        qc.rz(theta_rad, 0)
        theory_values.append(expected_acceptance_probability_from_circuit(qc))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(theory_thetas, theory_values, color="black", linewidth=2.0, linestyle="--", label="Theory")

    depol_values = [d["depolarizing"] for d in backends_data if d.get("depolarizing") is not None]
    use_colormap = len(depol_values) > 1
    if use_colormap:
        cmap = plt.colormaps["viridis"]
        norm = plt.Normalize(vmin=min(depol_values), vmax=max(depol_values))

    for i, data in enumerate(backends_data):
        depol = data.get("depolarizing")
        thetas = data["theta_values"]
        means = np.array(data["means"], dtype=float)
        stds = np.array(data["stds"], dtype=float)
        has_std = not np.all(stds == 0) and any(n > 1 for n in data["n_collected"])

        color = cmap(norm(depol)) if (use_colormap and depol is not None) else f"C{i}"

        ax.plot(thetas, means, color=color, linewidth=1.6, alpha=0.9, label=data["backend_label"])
        if has_std:
            ax.fill_between(
                thetas,
                np.clip(means - stds, 0.0, 1.0),
                np.clip(means + stds, 0.0, 1.0),
                color=color,
                alpha=0.15,
                linewidth=0,
            )

    ax.set_ylabel(r"$p_{\mathrm{acc}}$")
    ax.set_xlabel(r"$\theta$ (radians)")
    ax.set_title(r"$R_z(\theta)$ Clifford test acceptance probability")
    tick_positions = [0.0, math.pi / 4, math.pi / 2, math.pi, 3 * math.pi / 2, 2 * math.pi]
    tick_labels = [
        r"$0\ (\mathrm{I})$",
        r"$\pi/4\ (\mathrm{T})$",
        r"$\pi/2\ (\mathrm{S})$",
        r"$\pi\ (\mathrm{Z})$",
        r"$3\pi/2\ (\mathrm{S}^\dagger)$",
        r"$2\pi\ (-\mathrm{I})$",
    ]
    ax.set_xticks(tick_positions, tick_labels)
    ax.set_xlim(0, 2 * math.pi)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plot_path = plot_file if plot_file is not None else RESULTS_DIR / "rz_clifford_plot.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    plt.savefig(plot_path, dpi=260)
    print(f"Plot saved to {plot_path}")


def collect() -> None:
    parser = argparse.ArgumentParser(description="Collect Rz(theta) Clifford tester data.")
    parser.add_argument("--shots", type=int, default=1000, help="Shots per Weyl operator.")
    parser.add_argument("--theta-steps", type=int, default=100, help="Number of theta points from 0 to 2pi (inclusive).")
    parser.add_argument("--repeats", type=int, default=10, help="Number of repeats per theta (for mean/std bands).")
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="Named backend from cliff_lib/backends.py (e.g. qi_tuna_9). If omitted, uses AerSimulator with --depolarizing-list.",
    )
    parser.add_argument(
        "--depolarizing-list",
        type=str,
        default="0.001,0.01,0.05,0.1",
        help="Aer only. Comma-separated depolarizing rates. Include 0 for noiseless.",
    )
    parser.add_argument("--plot-file", type=Path, default=None, help="Output path for the plot image.")
    args = parser.parse_args()
    if args.shots <= 0 or args.theta_steps <= 0 or args.repeats <= 0:
        parser.error("--shots, --theta-steps, and --repeats must all be positive integers")

    theta_values: list[float] = np.linspace(0.0, 2 * math.pi, args.theta_steps).tolist()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.backend is not None:
        from cliff_lib.backends import resolve_backend

        backend, transpile_fn, timeout = resolve_backend(args.backend)
        _run_backend(args.backend, backend, transpile_fn, theta_values, args.shots, args.repeats, timeout=timeout)
    else:
        depol_values = sorted({float(v) for v in _parse_float_list(args.depolarizing_list)})
        if not depol_values:
            parser.error("--depolarizing-list must contain at least one value")
        if any(v < 0 for v in depol_values):
            parser.error("--depolarizing-list values must be >= 0")

        for depol in depol_values:
            backend = _build_aer_backend(depol)

            def transpile_fn(qc: QuantumCircuit, _backend=backend) -> QuantumCircuit:
                return transpile(qc, backend=_backend)

            _run_backend(_aer_label(depol), backend, transpile_fn, theta_values, args.shots, args.repeats, depolarizing=depol)

    print("\nDone collecting.")
    plot(plot_file=args.plot_file)


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "plot":
        sys.argv.pop(1)  # Remove "plot" so argparse in plot() doesn't see it
        parser = argparse.ArgumentParser(description="Plot Rz(theta) Clifford tester results.")
        parser.add_argument("--plot-file", type=Path, default=None, help="Output path for the plot image.")
        args = parser.parse_args()
        plot(plot_file=args.plot_file)
    else:
        collect()


if __name__ == "__main__":
    main()
