"""Plot Rz(theta) Clifford tester results from algorithm-1/results/rz_clifford/.

Reads data saved by collect_rz_clifford.py and overlays all backends on one plot.
Re-run at any time — adding new backend JSON files (e.g. hardware data) updates the plot.

Usage:
    uv run python algorithm-1/scripts/plot_rz_clifford.py
    uv run python algorithm-1/scripts/plot_rz_clifford.py --plot-file my_plot.png
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit

from lib.expected_acceptance_probability import expected_acceptance_probability_from_circuit

RESULTS_DIR = Path(__file__).parent.parent / "results" / "rz_clifford"


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Rz(theta) Clifford tester results.")
    parser.add_argument("--plot-file", type=Path, default=None, help="Output path for the plot image.")
    args = parser.parse_args()

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
        cmap = plt.cm.viridis
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

    plot_path = args.plot_file if args.plot_file is not None else RESULTS_DIR / "rz_clifford_plot.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    plt.savefig(plot_path, dpi=260)
    print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    main()
