import json
import os
from collections import Counter
from pathlib import Path

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
    def from_tuples(cls, results: dict[tuple[int, ...], dict[str, int]]) -> "BatchedRawResults":
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


# --- Checkpoint Models ---

PLAN_FILE = "plan.json"
JOBS_FILE = "jobs.json"


def _key(x: tuple[int, ...]) -> str:
    return json.dumps(list(x))


def _atomic_write(path: Path, content: str) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(content)
    os.replace(tmp, path)


class PairedPlan(BaseModel):
    type: str = "paired_runs"
    n: int
    total_shots: int
    x_counts: dict[str, int]

    @classmethod
    def from_counter(cls, n: int, total_shots: int, counter: Counter[tuple[int, ...]]) -> "PairedPlan":
        return cls(n=n, total_shots=total_shots, x_counts={_key(x): count for x, count in counter.items()})

    def to_counter(self) -> Counter[tuple[int, ...]]:
        return Counter({tuple(json.loads(k)): v for k, v in self.x_counts.items()})


class BatchedPlan(BaseModel):
    type: str = "batched"
    n: int
    shots_per_x: int
    all_x: list[list[int]]

    def to_tuples(self) -> list[tuple[int, ...]]:
        return [tuple(x) for x in self.all_x]


class PairedJobEntry(BaseModel):
    job_id: str | None = None
    counts: dict[str, int] | None = None


class PairedJobsState(BaseModel):
    jobs: dict[str, PairedJobEntry] = {}

    def get_entry(self, x: tuple[int, ...]) -> PairedJobEntry | None:
        return self.jobs.get(_key(x))

    def set_entry(self, x: tuple[int, ...], entry: PairedJobEntry) -> None:
        self.jobs[_key(x)] = entry


class BatchedJobsState(BaseModel):
    job_id: str | None = None


# --- Checkpoint Save / Load ---


def save_plan(plan: PairedPlan | BatchedPlan, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    _atomic_write(path / PLAN_FILE, plan.model_dump_json(indent=2))


def load_paired_plan(path: Path) -> PairedPlan | None:
    filepath = path / PLAN_FILE
    if not filepath.exists():
        return None
    return PairedPlan.model_validate_json(filepath.read_text())


def load_batched_plan(path: Path) -> BatchedPlan | None:
    filepath = path / PLAN_FILE
    if not filepath.exists():
        return None
    return BatchedPlan.model_validate_json(filepath.read_text())


def save_paired_jobs(state: PairedJobsState, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    _atomic_write(path / JOBS_FILE, state.model_dump_json(indent=2))


def load_paired_jobs(path: Path) -> PairedJobsState | None:
    filepath = path / JOBS_FILE
    if not filepath.exists():
        return None
    return PairedJobsState.model_validate_json(filepath.read_text())


def save_batched_jobs(state: BatchedJobsState, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    _atomic_write(path / JOBS_FILE, state.model_dump_json(indent=2))


def load_batched_jobs(path: Path) -> BatchedJobsState | None:
    filepath = path / JOBS_FILE
    if not filepath.exists():
        return None
    return BatchedJobsState.model_validate_json(filepath.read_text())


def cleanup_checkpoint(path: Path) -> None:
    for filename in (PLAN_FILE, JOBS_FILE):
        f = path / filename
        if f.exists():
            f.unlink()
