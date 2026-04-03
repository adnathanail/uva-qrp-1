"""Generate a focused bitstring-occurrence graph for standard unitary raw results.

Required arguments:
    uv run python scripts/01c_graph_standard_unitaries.py \
        --unitary hadamard --platform qi_tuna_9 --x 0,0 --tester batched

This writes one graph for that exact (unitary, platform, x, tester) slice.
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

    unitary_dir = RESULTS_ROOT / unitary
    unitary_dir.mkdir(parents=True, exist_ok=True)

    x_label = "[" + ", ".join(str(v) for v in x_value) + "]"
    x_slug = "_".join(str(v) for v in x_value)
    base_name = f"{unitary}__{platform}__{shots_dir.name}"

    if tester == "batched":
        raw = _load_json(platform_dir / "batched" / "raw_results.json")
        if not isinstance(raw, dict):
            raise FileNotFoundError("Missing batched raw results JSON")

        counts = _batched_counts_for_x(raw, x_value)
        if not counts:
            raise ValueError(f"No batched results found for x={x_label}")

        out_path = unitary_dir / f"{base_name}__batched__x_{x_slug}_counts.png"
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

    out_path = unitary_dir / f"{base_name}__paired_{channel}__x_{x_slug}_counts.png"
    _plot_counts(
        counts,
        out_path,
        f"{unitary} | {platform} | paired {channel} | x={x_label} ({shots_dir.name})",
    )
    return out_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Graph standard unitary raw results")
    parser.add_argument("--unitary", required=True, help="Unitary name (e.g. hadamard)")
    parser.add_argument("--platform", required=True, help="Platform name (e.g. qi_tuna_9)")
    parser.add_argument("--x", required=True, help="Weyl x as comma list or Python list (e.g. 0,0 or [0, 0])")
    parser.add_argument("--tester", choices=["batched", "paired"], default="batched", help="Tester raw results to use")
    parser.add_argument("--channel", choices=["y1", "y2"], default="y1", help="Paired channel to graph (ignored for batched)")
    parser.add_argument("--shots", help="Shots folder name (e.g. 1000_shots). Defaults to first available.")
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    x_value = _parse_x_arg(args.x)
    out_path = focused_plot(args.unitary, args.platform, x_value, args.tester, args.channel, args.shots)
    print(f"Created focused graph: {out_path}")


if __name__ == "__main__":
    main()
