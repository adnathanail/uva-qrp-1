"""Generate bitstring-occurrence graphs for standard unitary raw results.

Two modes are supported:

1) Focused mode (recommended for hardware non-determinism demos):
   uv run python scripts/01c_graph_standard_unitaries.py \
       --unitary hadamard --platform qi_tuna_9 --x 0,0 --tester batched

   Writes one graph for that exact (unitary, platform, x, tester) slice.

2) Bulk mode:
   uv run python scripts/01c_graph_standard_unitaries.py

   Scans all available standard-unitary results and writes aggregate graphs.
"""

from __future__ import annotations

import argparse
import ast
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS_ROOT = Path(__file__).resolve().parent.parent / "results" / "clifford_tester" / "standard"


def _sort_bitstrings(values: list[str]) -> list[str]:
    return sorted(values, key=lambda s: (len(s), int(s, 2) if s else -1))


def _plot_counts(counts: Counter[str], out_path: Path, title: str) -> None:
    if not counts:
        return

    labels = _sort_bitstrings(list(counts.keys()))
    values = [counts[label] for label in labels]

    fig_width = max(8.0, len(labels) * 0.65)
    fig, ax = plt.subplots(figsize=(fig_width, 4.6))
    ax.bar(labels, values, color="#2f5d8a")
    ax.set_title(title)
    ax.set_xlabel("Bit string")
    ax.set_ylabel("Occurrences")

    # Keep bit strings on the bottom and occurrence axis on the right.
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.spines["left"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _parse_x_arg(x_arg: str) -> list[int]:
    text = x_arg.strip()
    if not text:
        raise ValueError("x cannot be empty")

    if text.startswith("["):
        parsed = ast.literal_eval(text)
        if not isinstance(parsed, list) or not all(isinstance(v, int) for v in parsed):
            raise ValueError("x must be a list of integers")
        return parsed

    parts = [part.strip() for part in text.split(",") if part.strip()]
    if not parts:
        raise ValueError("x must contain at least one integer")
    return [int(part) for part in parts]


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _paired_counts(raw_data: dict) -> tuple[Counter[str], Counter[str]]:
    y1_counts: Counter[str] = Counter()
    y2_counts: Counter[str] = Counter()

    for sample in raw_data.get("samples", []):
        y1 = sample.get("y1")
        y2 = sample.get("y2")
        if isinstance(y1, str):
            y1_counts[y1] += 1
        if isinstance(y2, str):
            y2_counts[y2] += 1

    return y1_counts, y2_counts


def _paired_counts_for_x(raw_data: dict, x_value: list[int]) -> tuple[Counter[str], Counter[str]]:
    y1_counts: Counter[str] = Counter()
    y2_counts: Counter[str] = Counter()

    for sample in raw_data.get("samples", []):
        sample_x = sample.get("x")
        if sample_x != x_value:
            continue

        y1 = sample.get("y1")
        y2 = sample.get("y2")
        if isinstance(y1, str):
            y1_counts[y1] += 1
        if isinstance(y2, str):
            y2_counts[y2] += 1

    return y1_counts, y2_counts


def _batched_counts(raw_data: dict) -> Counter[str]:
    aggregated: Counter[str] = Counter()
    counts_by_x = raw_data.get("counts_by_x", {})
    if not isinstance(counts_by_x, dict):
        return aggregated

    for per_x in counts_by_x.values():
        if not isinstance(per_x, dict):
            continue
        for bitstring, count in per_x.items():
            if isinstance(bitstring, str) and isinstance(count, int):
                aggregated[bitstring] += count

    return aggregated


def _batched_counts_for_x(raw_data: dict, x_value: list[int]) -> Counter[str]:
    counts_by_x = raw_data.get("counts_by_x", {})
    if not isinstance(counts_by_x, dict):
        return Counter()

    for x_key, per_x in counts_by_x.items():
        try:
            parsed_x = ast.literal_eval(x_key)
        except (SyntaxError, ValueError):
            continue
        if parsed_x != x_value:
            continue

        if not isinstance(per_x, dict):
            return Counter()

        selected: Counter[str] = Counter()
        for bitstring, count in per_x.items():
            if isinstance(bitstring, str) and isinstance(count, int):
                selected[bitstring] += count
        return selected

    return Counter()


def process_platform_dir(unitary: str, shots: str, platform_dir: Path) -> int:
    plots_created = 0
    graphs_dir = platform_dir / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)

    paired_raw = _load_json(platform_dir / "paired" / "raw_results.json")
    if isinstance(paired_raw, dict):
        y1_counts, y2_counts = _paired_counts(paired_raw)
        if y1_counts:
            _plot_counts(
                y1_counts,
                graphs_dir / "paired_y1_counts.png",
                f"{unitary} | {platform_dir.name} | paired y1 ({shots})",
            )
            plots_created += 1
        if y2_counts:
            _plot_counts(
                y2_counts,
                graphs_dir / "paired_y2_counts.png",
                f"{unitary} | {platform_dir.name} | paired y2 ({shots})",
            )
            plots_created += 1

    batched_raw = _load_json(platform_dir / "batched" / "raw_results.json")
    if isinstance(batched_raw, dict):
        counts = _batched_counts(batched_raw)
        if counts:
            _plot_counts(
                counts,
                graphs_dir / "batched_counts.png",
                f"{unitary} | {platform_dir.name} | batched ({shots})",
            )
            plots_created += 1

    return plots_created


def _select_shots_dir(unitary: str, shots: str | None) -> Path:
    unitary_dir = RESULTS_ROOT / unitary
    if not unitary_dir.is_dir():
        raise FileNotFoundError(f"Unitary not found: {unitary}")

    if shots:
        chosen = unitary_dir / shots
        if not chosen.is_dir():
            raise FileNotFoundError(f"Shots folder not found: {chosen}")
        return chosen

    shot_dirs = sorted(p for p in unitary_dir.iterdir() if p.is_dir())
    if not shot_dirs:
        raise FileNotFoundError(f"No shots folders under {unitary_dir}")
    return shot_dirs[0]


def focused_plot(unitary: str, platform: str, x_value: list[int], tester: str, channel: str, shots: str | None) -> Path:
    shots_dir = _select_shots_dir(unitary, shots)
    platform_dir = shots_dir / platform
    if not platform_dir.is_dir():
        raise FileNotFoundError(f"Platform folder not found: {platform_dir}")

    graphs_dir = platform_dir / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)

    x_label = "[" + ", ".join(str(v) for v in x_value) + "]"

    if tester == "batched":
        raw = _load_json(platform_dir / "batched" / "raw_results.json")
        if not isinstance(raw, dict):
            raise FileNotFoundError("Missing batched raw results JSON")

        counts = _batched_counts_for_x(raw, x_value)
        if not counts:
            raise ValueError(f"No batched results found for x={x_label}")

        out_path = graphs_dir / f"batched_x_{'_'.join(str(v) for v in x_value)}_counts.png"
        _plot_counts(
            counts,
            out_path,
            f"{unitary} | {platform} | batched | x={x_label} ({shots_dir.name})",
        )
        return out_path

    raw = _load_json(platform_dir / "paired" / "raw_results.json")
    if not isinstance(raw, dict):
        raise FileNotFoundError("Missing paired raw results JSON")

    y1_counts, y2_counts = _paired_counts_for_x(raw, x_value)
    counts = y1_counts if channel == "y1" else y2_counts
    if not counts:
        raise ValueError(f"No paired {channel} results found for x={x_label}")

    out_path = graphs_dir / f"paired_{channel}_x_{'_'.join(str(v) for v in x_value)}_counts.png"
    _plot_counts(
        counts,
        out_path,
        f"{unitary} | {platform} | paired {channel} | x={x_label} ({shots_dir.name})",
    )
    return out_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Graph standard unitary raw results")
    parser.add_argument("--unitary", help="Unitary name (e.g. hadamard)")
    parser.add_argument("--platform", help="Platform name (e.g. qi_tuna_9)")
    parser.add_argument("--x", help="Weyl x as comma list or Python list (e.g. 0,0 or [0, 0])")
    parser.add_argument("--tester", choices=["batched", "paired"], default="batched", help="Tester raw results to use")
    parser.add_argument("--channel", choices=["y1", "y2"], default="y1", help="Paired channel to graph (ignored for batched)")
    parser.add_argument("--shots", help="Shots folder name (e.g. 1000_shots). Defaults to first available.")
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    focused_args_present = any([args.unitary, args.platform, args.x])
    if focused_args_present:
        if not (args.unitary and args.platform and args.x):
            raise SystemExit("Focused mode requires --unitary, --platform, and --x together.")

        x_value = _parse_x_arg(args.x)
        out_path = focused_plot(args.unitary, args.platform, x_value, args.tester, args.channel, args.shots)
        print(f"Created focused graph: {out_path}")
        return

    total_plots = 0
    total_platforms = 0

    for unitary_dir in sorted(RESULTS_ROOT.iterdir()):
        if not unitary_dir.is_dir():
            continue

        for shots_dir in sorted(unitary_dir.iterdir()):
            if not shots_dir.is_dir():
                continue

            for platform_dir in sorted(shots_dir.iterdir()):
                if not platform_dir.is_dir():
                    continue

                created = process_platform_dir(unitary_dir.name, shots_dir.name, platform_dir)
                if created:
                    total_platforms += 1
                    total_plots += created
                    print(f"Created {created} graph(s) in {platform_dir / 'graphs'}")

    if total_plots == 0:
        print("No raw result files found to graph.")
    else:
        print(f"Done. Created {total_plots} graph(s) across {total_platforms} platform folder(s).")


if __name__ == "__main__":
    main()
