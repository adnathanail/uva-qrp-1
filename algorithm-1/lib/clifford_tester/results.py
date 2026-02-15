import ast
import json
from pathlib import Path

from .utils import collision_probability

# --- Saving ---


def save_paired_raw(results: list[dict], path: Path):
    """Save paired-runs raw results to raw_results.json."""
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "raw_results.json", "w") as f:
        json.dump(results, f, indent=2)


def save_batched_raw(results: dict[tuple, dict], path: Path):
    """Save batched raw results to raw_results.json.

    Keys are stringified tuples, e.g. "(0, 1)".
    """
    path.mkdir(parents=True, exist_ok=True)
    serializable = {str(k): v for k, v in results.items()}
    with open(path / "raw_results.json", "w") as f:
        json.dump(serializable, f, indent=2)


def save_summary(acceptance_rate: float, path: Path):
    """Save summary with acceptance rate to summary.json."""
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "summary.json", "w") as f:
        json.dump({"acceptance_rate": acceptance_rate}, f, indent=2)


# --- Summarising ---


def summarise_paired(results: list[dict]) -> float:
    """Compute acceptance rate from paired-runs results (fraction where y1 == y2)."""
    if not results:
        return 0.0
    accepts = sum(1 for r in results if r["y1"] == r["y2"])
    return accepts / len(results)


def summarise_batched(results: dict[tuple, dict]) -> float:
    """Compute average collision probability from batched results."""
    if not results:
        return 0.0
    total = sum(collision_probability(counts) for counts in results.values())
    return total / len(results)


# --- Loading ---


def load_paired_raw(path: Path) -> list[dict] | None:
    """Load paired-runs raw results, or None if file doesn't exist."""
    filepath = path / "raw_results.json"
    if not filepath.exists():
        return None
    with open(filepath) as f:
        return json.load(f)


def load_batched_raw(path: Path) -> dict[tuple, dict] | None:
    """Load batched raw results, or None if file doesn't exist.

    Converts stringified tuple keys back to tuples.
    """
    filepath = path / "raw_results.json"
    if not filepath.exists():
        return None
    with open(filepath) as f:
        data = json.load(f)
    return {ast.literal_eval(k): v for k, v in data.items()}
