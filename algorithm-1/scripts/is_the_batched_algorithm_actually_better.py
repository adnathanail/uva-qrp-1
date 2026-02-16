"""Compare qi_tuna_9 paired vs batched results.

Scans all results and compares qi_tuna_9 paired and batched outcomes,
grouped into Cliffords (expected p_acc ≈ 1) and non-Cliffords.
"""

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "clifford_tester"
CLIFFORD_THRESHOLD = 0.99


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def collect_entries() -> list[dict]:
    """Scan results directory and collect all gates with qi_tuna_9 results."""
    entries: list[dict] = []

    for source_dir in RESULTS_DIR.iterdir():
        if not source_dir.is_dir():
            continue
        for gate_dir in sorted(source_dir.iterdir()):
            if not gate_dir.is_dir():
                continue
            for shots_dir in gate_dir.iterdir():
                if not shots_dir.is_dir():
                    continue

                exp_data = load_json(shots_dir / "expected_acceptance_probability.json")
                if exp_data is None:
                    continue

                qi_dir = shots_dir / "qi_tuna_9"
                if not qi_dir.is_dir():
                    continue

                paired_summary = load_json(qi_dir / "paired" / "summary.json")
                batched_summary = load_json(qi_dir / "batched" / "summary.json")

                entries.append(
                    {
                        "gate": gate_dir.name,
                        "source": source_dir.name,
                        "shots": shots_dir.name,
                        "p_expected": exp_data["expected_acceptance_probability"],
                        "paired": paired_summary["acceptance_rate"] if paired_summary else None,
                        "batched": batched_summary["acceptance_rate"] if batched_summary else None,
                    }
                )

    return sorted(entries, key=lambda e: e["gate"])


def print_table(entries: list[dict]) -> None:
    header = f"{'Gate':<30} {'Source':<10} {'p_expected':>10} {'Paired':>10} {'Batched':>10} {'Diff (B-P)':>10} {'% Diff':>8}"
    print(header)
    print("-" * len(header))

    for entry in entries:
        paired_str = f"{entry['paired']:.4f}" if entry["paired"] is not None else "N/A"
        batched_str = f"{entry['batched']:.4f}" if entry["batched"] is not None else "N/A"

        if entry["paired"] is not None and entry["batched"] is not None:
            diff = entry["batched"] - entry["paired"]
            diff_str = f"{diff:+.4f}"
            pct = (diff / entry["paired"]) * 100 if entry["paired"] != 0 else float("inf")
            pct_str = f"{pct:+.1f}%"
        else:
            diff_str = "N/A"
            pct_str = "N/A"

        print(f"{entry['gate']:<30} {entry['source']:<10} {entry['p_expected']:>10.4f} {paired_str:>10} {batched_str:>10} {diff_str:>10} {pct_str:>8}")


def print_summary(entries: list[dict]) -> None:
    both = [e for e in entries if e["paired"] is not None and e["batched"] is not None]
    if not both:
        return
    avg_paired = sum(e["paired"] for e in both) / len(both)
    avg_batched = sum(e["batched"] for e in both) / len(both)
    print(f"\n  {len(both)} gates with both results:")
    print(f"  Average paired acceptance rate:  {avg_paired:.4f}")
    print(f"  Average batched acceptance rate: {avg_batched:.4f}")
    print(f"  Average difference (B-P):        {avg_batched - avg_paired:+.4f}")


def main() -> None:
    entries = collect_entries()
    if not entries:
        print("No gates with qi_tuna_9 results found.")
        return

    cliffords = [e for e in entries if e["p_expected"] >= CLIFFORD_THRESHOLD]
    non_cliffords = [e for e in entries if e["p_expected"] < CLIFFORD_THRESHOLD]

    print("=== Cliffords ===\n")
    # print("=== Cliffords (p_expected ≈ 1) ===\n")
    if cliffords:
        print_table(cliffords)
        # print_summary(cliffords)
    else:
        print("  (none)")

    print("\n\n=== Non-Cliffords ===\n")
    # print("\n\n=== Non-Cliffords (p_expected < 1) ===\n")
    if non_cliffords:
        print_table(non_cliffords)
        # print_summary(non_cliffords)
    else:
        print("  (none)")


if __name__ == "__main__":
    main()
