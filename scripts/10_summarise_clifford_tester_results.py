"""Clustered bar chart summarising Clifford tester results per standard unitary.

For each standard unitary, plots 5 bars:
    - expected acceptance probability
    - aer_simulator paired
    - aer_simulator batched
    - qi_tuna_9 paired
    - qi_tuna_9 batched
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_ROOT = Path(__file__).resolve().parent.parent / "results" / "clifford_tester" / "standard"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "results" / "standard_unitaries_summary.png"


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _select_shots_dir(gate_dir: Path) -> Path | None:
    if not gate_dir.is_dir():
        return None
    shot_dirs = sorted(p for p in gate_dir.iterdir() if p.is_dir())
    return shot_dirs[0] if shot_dirs else None


def _acceptance_rate(shots_dir: Path, platform: str, tester: str) -> float | None:
    summary = _load_json(shots_dir / platform / tester / "summary.json")
    if summary is None:
        return None
    return summary.get("acceptance_rate")


def collect() -> list[dict]:
    rows: list[dict] = []
    for gate_name in [
        "identity",
        "hadamard",
        "s_gate",
        "t_gate",
        "rx_0_3",
        "cnot",
        "toffoli",
        "c_4_hadamard_3_cnot",
        "nc_4_t_gate",
    ]:  # order to match the paper
        shots_dir = _select_shots_dir(RESULTS_ROOT / gate_name)
        if shots_dir is None:
            rows.append({"gate": gate_name})
            continue

        exp = _load_json(shots_dir / "expected_acceptance_probability.json")
        rows.append(
            {
                "gate": gate_name,
                "expected": exp["expected_acceptance_probability"] if exp else None,
                "aer_paired": _acceptance_rate(shots_dir, "aer_simulator", "paired"),
                "aer_batched": _acceptance_rate(shots_dir, "aer_simulator", "batched"),
                "tuna_paired": _acceptance_rate(shots_dir, "qi_tuna_9", "paired"),
                "tuna_batched": _acceptance_rate(shots_dir, "qi_tuna_9", "batched"),
            }
        )
    return rows


def plot(rows: list[dict], out_path: Path) -> None:
    series = [
        ("Expected", "expected", "#444444"),
        ("Aer simulator paired", "aer_paired", "#4c78a8"),
        ("Aer simulator batched", "aer_batched", "#72b7e8"),
        ("Tuna-9 paired", "tuna_paired", "#e45756"),
        ("Tuna-9 batched", "tuna_batched", "#f1a6a4"),
    ]

    gates = [r["gate"] for r in rows]
    n_gates = len(gates)
    n_series = len(series)
    bar_width = 0.8 / n_series
    x = np.arange(n_gates)

    fig, ax = plt.subplots(figsize=(max(10.0, n_gates * 1.4), 5.5))

    for i, (label, key, color) in enumerate(series):
        values = [r.get(key) if r.get(key) is not None else np.nan for r in rows]
        offsets = x - 0.4 + bar_width * (i + 0.5)
        ax.bar(offsets, values, width=bar_width, label=label, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(gates, rotation=30, ha="right")
    ax.set_ylabel("Acceptance probability")
    ax.set_ylim(0, 1.05)
    ax.set_title("Clifford tester results per standard unitary")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper right", ncol=2, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rows = collect()
    plot(rows, OUTPUT_PATH)
    print(f"Saved {OUTPUT_PATH.relative_to(OUTPUT_PATH.parent.parent)}")


if __name__ == "__main__":
    main()
