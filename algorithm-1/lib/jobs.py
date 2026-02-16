from pathlib import Path

from qiskit import qpy
from qiskit.providers import BackendV2
from qiskit_aer import AerJob
from qiskit_quantuminspire.qi_jobs import QIJob

from .state import JOB_GLOB


class JobManagementError(Exception):
    pass


def get_job_id(job: AerJob | QIJob) -> str:
    """Extract a usable job identifier from a backend job.

    - QI jobs use
        - circuits_run_data[0].job_id when there is only 1 circuit (as this matches the ID's on My QI)
        - batch_job_id otherwise
    - Aer jobs use job_id()
    """
    if isinstance(job, AerJob):
        return job.job_id()
    elif isinstance(job, QIJob):
        if len(job.circuits_run_data) == 1:
            if (job_circuit_id := job.circuits_run_data[0].job_id) is not None:
                return str(job_circuit_id)
            else:
                raise JobManagementError("Error getting job ID from QIJob: Circuit job ID was None")
        else:
            if (job_batch_id := job.batch_job_id) is not None:
                return str(job_batch_id)
            else:
                raise JobManagementError("Error getting job ID from QIJob: Job batch ID was None")
    else:
        msg = f"Error getting job ID: Invalid job type {type(job)}"
        raise JobManagementError(msg)


def save_job(job: AerJob | QIJob, checkpoint_dir: Path) -> None:
    """Serialize a job to checkpoint_dir if it supports serialization (QI jobs).

    Saves as job_{id}.qpy. Removes any previous job_*.qpy first.
    """
    if isinstance(job, QIJob):
        # Clean old serialized jobs
        for old in checkpoint_dir.glob(JOB_GLOB):
            old.unlink()
        jid = get_job_id(job)
        job.serialize(checkpoint_dir / f"job_{jid}.qpy")
    elif not isinstance(job, AerJob):
        msg = f"Error saving job: Invalid job type {type(job)}"
        raise JobManagementError(msg)


def load_job(backend: BackendV2, checkpoint_dir: Path, job_id: str) -> QIJob | None:
    """Try to reconstruct a QI job from a serialized checkpoint.

    QIJob.deserialize() requires a provider just to call provider.get_backend(),
    but we already have the backend. So we reconstruct the job directly.
    """
    job_path = checkpoint_dir / f"job_{job_id}.qpy"
    if not job_path.exists():
        return None

    with open(job_path, "rb") as f:
        circuits = qpy.load(f)
    if not circuits:
        return None
    batch_job_id = circuits[0].metadata.get("batch_job_id")
    if batch_job_id is None:
        return None
    job = QIJob(circuits, backend)
    job.batch_job_id = batch_job_id
    for cd in job.circuits_run_data:
        cd.job_id = cd.circuit.metadata.get("job_id")
    return job
