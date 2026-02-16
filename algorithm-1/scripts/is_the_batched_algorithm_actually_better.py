"""Compare qi_tuna_9 paired vs batched results for Clifford gates.

Scans all results, identifies gates with expected acceptance probability ≈ 1
(i.e. Cliffords), and compares their qi_tuna_9 paired and batched outcomes.
"""

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "clifford_tester"
THRESHOLD = 0.99  # Expected acceptance probability above this → Clifford


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def main() -> None:
    # Collect all gate directories across all sources (standard, stim, etc.)
    gate_dirs: list[Path] = []
    for source_dir in RESULTS_DIR.iterdir():
        if not source_dir.is_dir():
            continue
        for gate_dir in sorted(source_dir.iterdir()):
            if gate_dir.is_dir():
                gate_dirs.append(gate_dir)

    cliffords: list[dict] = []

    for gate_dir in sorted(gate_dirs):
        # Find shot directories (e.g. 1000_shots)
        for shots_dir in gate_dir.iterdir():
            if not shots_dir.is_dir():
                continue

            exp_prob_file = shots_dir / "expected_acceptance_probability.json"
            exp_data = load_json(exp_prob_file)
            if exp_data is None:
                continue

            p_expected = exp_data["expected_acceptance_probability"]
            if p_expected < THRESHOLD:
                continue

            # This is a Clifford — look for qi_tuna_9 results
            qi_dir = shots_dir / "qi_tuna_9"
            if not qi_dir.is_dir():
                continue

            paired_summary = load_json(qi_dir / "paired" / "summary.json")
            batched_summary = load_json(qi_dir / "batched" / "summary.json")

            gate_name = gate_dir.name
            source = gate_dir.parent.name
            shots = shots_dir.name

            entry = {
                "gate": gate_name,
                "source": source,
                "shots": shots,
                "p_expected": p_expected,
                "paired": paired_summary["acceptance_rate"] if paired_summary else None,
                "batched": batched_summary["acceptance_rate"] if batched_summary else None,
            }
            cliffords.append(entry)

    if not cliffords:
        print("No Clifford gates with qi_tuna_9 results found.")
        return

    # Print comparison table
    print(f"{'Gate':<30} {'Source':<10} {'p_expected':>10} {'Paired':>10} {'Batched':>10} {'Diff (P-B)':>10}")
    print("-" * 82)

    for entry in cliffords:
        paired_str = f"{entry['paired']:.4f}" if entry["paired"] is not None else "N/A"
        batched_str = f"{entry['batched']:.4f}" if entry["batched"] is not None else "N/A"

        if entry["paired"] is not None and entry["batched"] is not None:
            diff = entry["paired"] - entry["batched"]
            diff_str = f"{diff:+.4f}"
        else:
            diff_str = "N/A"

        print(f"{entry['gate']:<30} {entry['source']:<10} {entry['p_expected']:>10.4f} {paired_str:>10} {batched_str:>10} {diff_str:>10}")

    # Summary stats
    both_available = [e for e in cliffords if e["paired"] is not None and e["batched"] is not None]
    if both_available:
        print(f"\n--- Summary ({len(both_available)} gates with both results) ---")
        avg_paired = sum(e["paired"] for e in both_available) / len(both_available)
        avg_batched = sum(e["batched"] for e in both_available) / len(both_available)
        print(f"Average paired acceptance rate:  {avg_paired:.4f}")
        print(f"Average batched acceptance rate: {avg_batched:.4f}")
        print(f"Average difference (P-B):        {avg_paired - avg_batched:+.4f}")


if __name__ == "__main__":
    main()
