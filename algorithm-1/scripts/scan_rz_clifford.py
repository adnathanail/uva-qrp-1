"""Scan Rz(theta) with the Clifford tester on an Aer simulator.

Usage:
    uv run python algorithm-1/scripts/scan_rz_clifford.py --theta-steps 9
    uv run python algorithm-1/scripts/scan_rz_clifford.py --theta-steps 13 --depolarizing-list 0.01
    uv run python algorithm-1/scripts/scan_rz_clifford.py --theta-steps 13 --depolarizing-list 0.01,0.05,0.1 --repeats 5
"""

from __future__ import annotations

import argparse
import math
import tempfile
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile

from lib.clifford_tester import clifford_tester_batched
from lib.expected_acceptance_probability import expected_acceptance_probability_from_circuit
from lib.state import BatchedRawResults, save_batched_raw, save_summary


def _parse_theta_list(raw: str) -> list[float]:
    parts = [p.strip() for p in raw.replace(";", ",").split(",")]
    values = [p for p in parts if p]
    if not values:
        raise ValueError("No theta values provided")
    return [float(v) for v in values]


def _parse_float_list(raw: str) -> list[float]:
    return _parse_theta_list(raw)


def _build_backend(depolarizing: float | None):
    from qiskit_aer import AerSimulator

    if depolarizing is None or depolarizing <= 0:
        return AerSimulator()

    from qiskit_aer.noise import NoiseModel, depolarizing_error

    noise_model = NoiseModel()
    error_1q = depolarizing_error(depolarizing, 1)
    error_2q = depolarizing_error(depolarizing, 2)

    noise_model.add_all_qubit_quantum_error(
        error_1q,
        ["u", "u1", "u2", "u3", "rx", "ry", "rz", "x", "y", "z", "h", "s", "sdg", "t", "tdg", "sx", "id"],
    )
    noise_model.add_all_qubit_quantum_error(error_2q, ["cx", "cz", "swap"])
    return AerSimulator(noise_model=noise_model)


def _theta_label(theta_rad: float) -> str:
    return f"theta_{theta_rad:.6f}"


def _depolarizing_label(depolarizing: float | None) -> str:
    if depolarizing is None or depolarizing <= 0:
        return "noiseless"
    return f"depol_{depolarizing:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Clifford tester on Rz(theta) using AerSimulator.")
    parser.add_argument("--shots", type=int, default=1000, help="Shots per Weyl operator (batched) or total shots (paired).")
    parser.add_argument("--theta-steps", type=int, default=9, help="Number of theta points from 0 to 2pi (inclusive).")
    parser.add_argument("--degrees", action="store_true", help="Interpret theta values as degrees instead of radians.")
    parser.add_argument(
        "--depolarizing-list",
        type=str,
        default="0.001,0.01,0.05,0.1",
        help="Comma-separated depolarizing rates to plot (overrides --depolarizing).",
    )
    parser.add_argument("--repeats", type=int, default=5, help="Number of repeats per theta (for mean/std bands).")
    parser.add_argument("--results-dir", type=Path, default=None, help="Optional directory to save raw results and summaries.")
    parser.add_argument("--plot-file", type=Path, default=None, help="Optional output path for the plot image.")
    args = parser.parse_args()

    theta_values = np.linspace(0.0, 2 * math.pi, args.theta_steps).tolist()

    if args.depolarizing_list:
        depol_values: list[float] = _parse_float_list(args.depolarizing_list)
    else:
        depol_values = [0.001, 0.01, 0.05, 0.1]

    depol_values = sorted({float(v) for v in depol_values})

    theta_rad_values = [math.radians(t) if args.degrees else t for t in theta_values]
    expected_values: list[float] = []
    for theta_rad in theta_rad_values:
        qc = QuantumCircuit(1)
        qc.rz(theta_rad, 0)
        expected_values.append(expected_acceptance_probability_from_circuit(qc))

    results_by_depol: list[dict[str, object]] = []

    for depol in depol_values:
        backend = _build_backend(depol)
        def transpile_fn(qc, backend=backend):
            return transpile(qc, backend=backend)

        paired_means: list[float] = []
        paired_stds: list[float] = []
        batched_means: list[float] = []
        batched_stds: list[float] = []

        for theta_rad in theta_rad_values:
            qc = QuantumCircuit(1)
            qc.rz(theta_rad, 0)

            if args.results_dir is not None and args.repeats == 1:
                base_dir = args.results_dir / _depolarizing_label(depol) / _theta_label(theta_rad)
                batched_dir = base_dir / "batched"
            else:
                base_dir = None
                batched_dir = None

            samples = []
            for _ in range(args.repeats):
                if base_dir is None:
                    with tempfile.TemporaryDirectory() as tmp:
                        raw = clifford_tester_batched(
                            qc, 1, shots=args.shots, backend=backend, transpilation_function=transpile_fn, checkpoint_dir=Path(tmp)
                        )
                else:
                    raw = clifford_tester_batched(qc, 1, shots=args.shots, backend=backend, transpilation_function=transpile_fn, checkpoint_dir=batched_dir)
                batched_raw = BatchedRawResults.from_tuples(raw)
                rate = batched_raw.summarise()
                samples.append(rate)
            batched_means.append(float(np.mean(samples)))
            batched_stds.append(float(np.std(samples, ddof=1)) if args.repeats > 1 else 0.0)

            if base_dir is not None:
                save_batched_raw(batched_raw, batched_dir)
                save_summary(batched_means[-1], batched_dir)

        results_by_depol.append(
            {
                "depol": depol,
                "paired_mean": paired_means,
                "paired_std": paired_stds,
                "batched_mean": batched_means,
                "batched_std": batched_stds,
            }
        )

    print()
    print("Rz(theta) Clifford tester results")
    if args.degrees:
        print("Angles: degrees (input) / radians (computed)")
    print(f"Shots: {args.shots} | Repeats: {args.repeats}")
    print(f"Depolarizing rates: {', '.join(f'{v:g}' for v in depol_values)}")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8.5, 4.8))

    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=min(depol_values), vmax=max(depol_values))

    ax.plot(
        theta_rad_values,
        expected_values,
        color="black",
        linewidth=2.0,
        linestyle="--",
    )
    for depol_entry in results_by_depol:
        depol = float(depol_entry["depol"])
        color = cmap(norm(depol))
        means = np.array(depol_entry["batched_mean"], dtype=float)
        stds = np.array(depol_entry["batched_std"], dtype=float)
        ax.plot(theta_rad_values, means, color=color, linewidth=1.6, alpha=0.9, label=f"p = {depol}")
        if args.repeats > 1:
            ax.fill_between(
                theta_rad_values,
                np.clip(means - stds, 0.0, 1.0),
                np.clip(means + stds, 0.0, 1.0),
                color=color,
                alpha=0.15,
                linewidth=0,
            )

    ax.set_ylabel(r"$p_{\mathrm{acc}}$")
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

    ax.set_xlabel(r"$\theta$ (radians)")
    ax.legend()

    if args.plot_file is not None:
        plot_path = args.plot_file
    elif args.results_dir is not None:
        plot_path = args.results_dir / "rz_clifford_plot.png"
    else:
        plot_path = Path("rz_clifford_plot.png")

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    plt.savefig(plot_path, dpi=260)
    print()
    print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    main()
