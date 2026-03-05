# uv run algorithm-1/scripts/04_can_we_sample_a_subset_of_weyl_operators.py         # compute + plot
# uv run algorithm-1/scripts/04_can_we_sample_a_subset_of_weyl_operators.py plot     # plot only
"""Subsampling analysis: can we estimate acceptance rate from a subset of Weyl operators?

The batched tester enumerates ALL 4^n Weyl operators. This script resamples from
existing batched results to measure how the acceptance rate estimate converges as
we increase the number of sampled operators.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lib.clifford_tester.utils import collision_probability
from lib.state.outputs import load_batched_raw
from lib.state.utils import atomic_write

# ── Configuration ────────────────────────────────────────────────────────────

CLIFFORD_TESTER_DIR = Path("algorithm-1/results/clifford_tester")
RESULTS_DIR = Path("algorithm-1/results/weyl_subset_sampling")
RESULTS_FILE = RESULTS_DIR / "results.json"
CONVERGENCE_PLOT = RESULTS_DIR / "convergence.png"
RELATIVE_ERROR_PLOT = RESULTS_DIR / "relative_error.png"

N_SUBSET_SIZES = 15
N_TRIALS = 500
MIN_SUBSET_SIZE = 4
MIN_WEYL_OPS = 5  # skip 1-qubit gates (4 ops)
CP_VARIANCE_THRESHOLD = 1e-9


# ── Discovery ────────────────────────────────────────────────────────────────


def discover_interesting_cases() -> list[dict]:
    """Walk batched result directories and find cases with varying collision probs.

    Skips cases where all collision probabilities are identical (e.g. noiseless
    Clifford gates on aer_simulator) and cases with too few Weyl operators.
    """
    cases: list[dict] = []

    for source_dir in sorted(CLIFFORD_TESTER_DIR.iterdir()):
        if not source_dir.is_dir():
            continue
        for gate_dir in sorted(source_dir.iterdir()):
            if not gate_dir.is_dir():
                continue
            for shots_dir in sorted(gate_dir.iterdir()):
                if not shots_dir.is_dir():
                    continue

                exp_path = shots_dir / "expected_acceptance_probability.json"
                if not exp_path.exists():
                    continue
                exp_data = json.loads(exp_path.read_text())
                p_expected = exp_data["expected_acceptance_probability"]

                for backend_dir in sorted(shots_dir.iterdir()):
                    if not backend_dir.is_dir():
                        continue

                    batched_dir = backend_dir / "batched"
                    batched = load_batched_raw(batched_dir)
                    if batched is None:
                        continue

                    n_ops = len(batched.counts_by_x)
                    if n_ops < MIN_WEYL_OPS:
                        continue

                    cps = {x_key: collision_probability(counts) for x_key, counts in batched.counts_by_x.items()}

                    cp_values = list(cps.values())
                    if max(cp_values) - min(cp_values) <= CP_VARIANCE_THRESHOLD:
                        continue

                    full_rate = sum(cp_values) / len(cp_values)
                    label = f"{gate_dir.name} ({backend_dir.name})"

                    cases.append(
                        {
                            "label": label,
                            "gate": gate_dir.name,
                            "source": source_dir.name,
                            "backend": backend_dir.name,
                            "shots": shots_dir.name,
                            "n_weyl_ops": n_ops,
                            "full_rate": full_rate,
                            "p_expected": p_expected,
                            "collision_probs": cps,
                        }
                    )

    return sorted(cases, key=lambda c: (c["n_weyl_ops"], c["label"]))


# ── Subsampling ──────────────────────────────────────────────────────────────


def run_subsampling_trials(
    collision_probs: dict[str, float],
    n_subset_sizes: int = N_SUBSET_SIZES,
    n_trials: int = N_TRIALS,
) -> list[dict]:
    """Run subsampling trials at various subset sizes.

    Returns a list of dicts with keys: subset_size, trial_mean, trial_std, trial_relative_std.
    """
    rng = np.random.default_rng(42)
    keys = list(collision_probs.keys())
    values = np.array([collision_probs[k] for k in keys])
    total = len(keys)
    full_rate = float(np.mean(values))

    subset_sizes = np.unique(np.geomspace(MIN_SUBSET_SIZE, total, n_subset_sizes).astype(int))

    results = []
    for size in subset_sizes:
        size = int(size)
        trial_means = np.empty(n_trials)
        for t in range(n_trials):
            indices = rng.choice(total, size=size, replace=False)
            trial_means[t] = np.mean(values[indices])

        mean_of_trials = float(np.mean(trial_means))
        std_of_trials = float(np.std(trial_means))
        relative_std = std_of_trials / full_rate if full_rate > 0 else 0.0

        results.append(
            {
                "subset_size": size,
                "trial_mean": mean_of_trials,
                "trial_std": std_of_trials,
                "trial_relative_std": relative_std,
            }
        )

    return results


# ── Collect ──────────────────────────────────────────────────────────────────


def collect() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Discovering interesting cases...")
    cases = discover_interesting_cases()
    print(f"Found {len(cases)} interesting cases:")
    for c in cases:
        print(f"  {c['label']} — {c['n_weyl_ops']} Weyl ops, full rate={c['full_rate']:.4f}")

    all_results = []
    for c in cases:
        print(f"\nSubsampling {c['label']}...")
        trials = run_subsampling_trials(c["collision_probs"])

        all_results.append(
            {
                "label": c["label"],
                "gate": c["gate"],
                "source": c["source"],
                "backend": c["backend"],
                "shots": c["shots"],
                "n_weyl_ops": c["n_weyl_ops"],
                "full_rate": c["full_rate"],
                "p_expected": c["p_expected"],
                "subsampling": trials,
            }
        )

    atomic_write(RESULTS_FILE, json.dumps(all_results, indent=2) + "\n")
    print(f"\nResults saved to {RESULTS_FILE}")


# ── Plot ─────────────────────────────────────────────────────────────────────


def plot() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not RESULTS_FILE.exists():
        print("No results to plot. Run without 'plot' argument first.")
        return

    all_results = json.loads(RESULTS_FILE.read_text())
    if not all_results:
        print("No results to plot.")
        return

    plot_convergence(all_results)
    plot_relative_error(all_results)


def plot_convergence(all_results: list[dict]) -> None:
    n = len(all_results)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)

    for idx, case in enumerate(all_results):
        ax = axes[idx // cols][idx % cols]
        trials = case["subsampling"]

        sizes = [t["subset_size"] for t in trials]
        means = [t["trial_mean"] for t in trials]
        stds = [t["trial_std"] for t in trials]

        means_arr = np.array(means)
        stds_arr = np.array(stds)

        ax.fill_between(sizes, means_arr - stds_arr, means_arr + stds_arr, alpha=0.3)
        ax.plot(sizes, means, "o-", markersize=3, label="subsample mean")
        ax.axhline(case["full_rate"], color="C1", linestyle="--", label="full result")
        ax.axhline(case["p_expected"], color="C2", linestyle=":", alpha=0.7, label="theoretical")

        ax.set_xscale("log")
        ax.set_xlabel("# Weyl ops sampled")
        ax.set_ylabel("acceptance rate")
        ax.set_title(f"{case['label']}\n({case['n_weyl_ops']} total ops)", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.suptitle("Convergence of acceptance rate with Weyl operator subsampling", y=1.02)
    fig.tight_layout()
    fig.savefig(CONVERGENCE_PLOT, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {CONVERGENCE_PLOT}")
    plt.close(fig)


def plot_relative_error(all_results: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    for case in all_results:
        trials = case["subsampling"]
        fractions = [t["subset_size"] / case["n_weyl_ops"] for t in trials]
        rel_stds = [t["trial_relative_std"] for t in trials]

        ax.plot(fractions, rel_stds, "o-", markersize=3, label=case["label"])

    # Reference lines
    for pct, style in [(0.10, ":"), (0.05, "--"), (0.01, "-.")]:
        ax.axhline(pct, color="gray", linestyle=style, alpha=0.5)
        ax.text(1.01, pct, f"{pct:.0%}", va="center", fontsize=8, color="gray", transform=ax.get_yaxis_transform())

    ax.set_xlabel("fraction of Weyl operators sampled")
    ax.set_ylabel("relative std (std / full rate)")
    ax.set_title("Relative error vs fraction of Weyl operators sampled")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(RELATIVE_ERROR_PLOT, dpi=150)
    print(f"Plot saved to {RELATIVE_ERROR_PLOT}")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "plot":
        plot()
    else:
        collect()
        plot()


if __name__ == "__main__":
    main()
