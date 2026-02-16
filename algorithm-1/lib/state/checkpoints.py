import json
from collections import Counter
from pathlib import Path

from pydantic import BaseModel

from .utils import atomic_write, serialize_key

PLAN_FILE = "plan.json"
JOBS_FILE = "jobs.json"
JOB_GLOB = "job_*.qpy"


# --- Models ---


class PairedPlan(BaseModel):
    type: str = "paired_runs"
    n: int
    total_shots: int
    x_counts: dict[str, int]

    @classmethod
    def from_counter(cls, n: int, total_shots: int, counter: Counter[tuple[int, ...]]) -> "PairedPlan":
        return cls(n=n, total_shots=total_shots, x_counts={serialize_key(x): count for x, count in counter.items()})

    def to_counter(self) -> Counter[tuple[int, ...]]:
        return Counter({tuple(json.loads(k)): v for k, v in self.x_counts.items()})


class BatchedPlan(BaseModel):
    type: str = "batched"
    n: int
    shots_per_x: int
    all_x: list[list[int]]

    def to_tuples(self) -> list[tuple[int, ...]]:
        return [tuple(x) for x in self.all_x]


class JobEntry(BaseModel):
    job_id: str | None = None
    counts: dict[str, int] | None = None


class JobsState(BaseModel):
    jobs: dict[str, JobEntry] = {}

    def get_entry(self, x: tuple[int, ...]) -> JobEntry | None:
        return self.jobs.get(serialize_key(x))

    def set_entry(self, x: tuple[int, ...], entry: JobEntry) -> None:
        self.jobs[serialize_key(x)] = entry


# --- Save / Load ---


def save_plan(plan: PairedPlan | BatchedPlan, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    atomic_write(path / PLAN_FILE, plan.model_dump_json(indent=2))


def load_paired_plan(path: Path) -> PairedPlan | None:
    filepath = path / PLAN_FILE
    if not filepath.exists():
        return None
    plan = PairedPlan.model_validate_json(filepath.read_text())
    if plan.type != "paired_runs":
        raise ValueError(f"Expected paired_runs plan, got {plan.type}")
    return plan


def load_batched_plan(path: Path) -> BatchedPlan | None:
    filepath = path / PLAN_FILE
    if not filepath.exists():
        return None
    plan = BatchedPlan.model_validate_json(filepath.read_text())
    if plan.type != "batched":
        raise ValueError(f"Expected batched plan, got {plan.type}")
    return plan


def save_jobs(state: JobsState, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    atomic_write(path / JOBS_FILE, state.model_dump_json(indent=2))


def load_jobs(path: Path) -> JobsState | None:
    filepath = path / JOBS_FILE
    if not filepath.exists():
        return None
    return JobsState.model_validate_json(filepath.read_text())


def cleanup_checkpoint(path: Path) -> None:
    for filename in (PLAN_FILE, JOBS_FILE):
        f = path / filename
        if f.exists():
            f.unlink()
    for f in path.glob(JOB_GLOB):
        f.unlink()
