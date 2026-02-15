import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .utils import collision_probability

# --- Models ---


class PairedSample(BaseModel):
    x: list[int]
    y1: str
    y2: str


class PairedRawResults(BaseModel):
    samples: list[PairedSample]

    def summarise(self) -> float:
        """Compute acceptance rate (fraction where y1 == y2)."""
        if not self.samples:
            return 0.0
        accepts = sum(1 for s in self.samples if s.y1 == s.y2)
        return accepts / len(self.samples)


class BatchedRawResults(BaseModel):
    counts_by_x: dict[str, dict[str, int]]

    def summarise(self) -> float:
        """Compute average collision probability across all Weyl operators."""
        if not self.counts_by_x:
            return 0.0
        total = sum(collision_probability(counts) for counts in self.counts_by_x.values())
        return total / len(self.counts_by_x)

    def to_tuples(self) -> dict[tuple[int, ...], dict[str, int]]:
        """Convert back to the dict[tuple, dict] format."""
        return {tuple(json.loads(k)): v for k, v in self.counts_by_x.items()}

    @classmethod
    def from_tuples(cls, results: dict[tuple[int, ...], dict[str, Any]]) -> "BatchedRawResults":
        """Convert from the dict[tuple, dict] returned by the tester."""
        return cls(counts_by_x={json.dumps(k): v for k, v in results.items()})


class Summary(BaseModel):
    acceptance_rate: float


class ExpectedAcceptanceProbability(BaseModel):
    expected_acceptance_probability: float


# --- Save / Load ---

PAIRED_RAW_RESULTS_FILE = "raw_results.json"
BATCHED_RAW_RESULTS_FILE = "raw_results.json"
SUMMARY_FILE = "summary.json"


def save_paired_raw(results: PairedRawResults, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / PAIRED_RAW_RESULTS_FILE).write_text(results.model_dump_json(indent=2))


def load_paired_raw(path: Path) -> PairedRawResults | None:
    filepath = path / PAIRED_RAW_RESULTS_FILE
    if not filepath.exists():
        return None
    return PairedRawResults.model_validate_json(filepath.read_text())


def save_batched_raw(results: BatchedRawResults, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / BATCHED_RAW_RESULTS_FILE).write_text(results.model_dump_json(indent=2))


def load_batched_raw(path: Path) -> BatchedRawResults | None:
    filepath = path / BATCHED_RAW_RESULTS_FILE
    if not filepath.exists():
        return None
    return BatchedRawResults.model_validate_json(filepath.read_text())


def save_summary(acceptance_rate: float, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    summary = Summary(acceptance_rate=acceptance_rate)
    (path / SUMMARY_FILE).write_text(summary.model_dump_json(indent=2))


def load_summary(path: Path) -> Summary | None:
    filepath = path / SUMMARY_FILE
    if not filepath.exists():
        return None
    return Summary.model_validate_json(filepath.read_text())
