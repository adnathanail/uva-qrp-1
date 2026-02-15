from collections import Counter
from collections.abc import Callable
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.providers import BackendV2

from .results import (
    JOB_GLOB,
    BatchedPlan,
    PairedJobEntry,
    PairedJobsState,
    PairedPlan,
    cleanup_checkpoint,
    load_batched_plan,
    load_paired_jobs,
    load_paired_plan,
    save_paired_jobs,
    save_plan,
)
from .utils import default_transpilation_function, get_clifford_tester_circuit


def _get_job_id(job: Any) -> str | None:
    """Extract a usable job identifier from a backend job.

    QI jobs use batch_job_id (job_id() returns ""), Aer jobs use job_id().
    Returns None if no identifier is available.
    """
    batch_id = getattr(job, "batch_job_id", None)
    if batch_id is not None:
        return str(batch_id)
    jid = job.job_id()
    if jid:
        return jid
    return None


def _save_job(job: Any, checkpoint_dir: Path) -> None:
    """Serialize a job to checkpoint_dir if it supports serialization (QI jobs).

    Saves as job_{id}.qpy. Removes any previous job_*.qpy first.
    """
    if not hasattr(job, "serialize"):
        return
    # Clean old serialized jobs
    for old in checkpoint_dir.glob(JOB_GLOB):
        old.unlink()
    jid = _get_job_id(job) or "unknown"
    job.serialize(checkpoint_dir / f"job_{jid}.qpy")


def _load_job(backend: Any, checkpoint_dir: Path) -> Any | None:
    """Try to reconstruct a QI job from a serialized checkpoint.

    QIJob.deserialize() requires a provider just to call provider.get_backend(),
    but we already have the backend. So we reconstruct the job directly.
    """
    matches = list(checkpoint_dir.glob(JOB_GLOB))
    if not matches:
        return None
    job_path = matches[0]
    try:
        from qiskit import qpy
        from qiskit_quantuminspire.qi_jobs import QIJob

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
    except Exception:
        return None


def clifford_tester_batched(
    U_circuit: QuantumCircuit,
    n: int,
    *,
    shots: int = 1000,
    backend: BackendV2,
    transpilation_function: Callable[[QuantumCircuit], QuantumCircuit] | None = None,
    timeout: float | None = None,
    checkpoint_dir: Path,
) -> dict[tuple[int, ...], dict[str, int]]:
    """
    Four-query Clifford tester algorithm (batched).

    Tests whether a unitary U is a Clifford gate by enumerating all 4^n
    Weyl operators, running each circuit with the given number of shots,
    and returning the raw counts for each Weyl operator.

    Args:
        U_circuit: A quantum circuit implementing the n-qubit unitary U
        n: Number of qubits U acts on
        shots: Number of backend shots per Weyl operator circuit
        backend: Qiskit backend to run on (defaults to AerSimulator)
        transpilation_function: Optional function to transpile circuits
        checkpoint_dir: Directory for checkpoint files (plan.json, jobs.json)

    Returns:
        dict mapping each Weyl operator x (tuple) to its Qiskit counts dict
    """
    transpilation_function = transpilation_function or default_transpilation_function

    # Phase 1: Load or generate plan
    plan = load_batched_plan(checkpoint_dir)
    if plan is not None:
        all_x = plan.to_tuples()
        print(f"       loaded existing plan ({len(all_x)} Weyl operators)")
    else:
        all_x = list(product([0, 1], repeat=2 * n))
        plan = BatchedPlan(n=n, shots_per_x=shots, all_x=[list(x) for x in all_x])
        save_plan(plan, checkpoint_dir)
        print(f"       created new plan ({len(all_x)} Weyl operators, {shots} shots each)")

    # Phase 2: Build circuits
    print(f"       building {len(all_x)} circuits...")
    circuits = [transpilation_function(get_clifford_tester_circuit(U_circuit, n, x)) for x in all_x]

    # Phase 3: Check for existing job or submit new one
    result = None

    saved_job = _load_job(backend, checkpoint_dir)
    if saved_job is not None:
        jid = _get_job_id(saved_job)
        print(f"       loaded saved job (id={jid}), retrieving result...")
        try:
            result = saved_job.result(timeout=timeout)
            print("       retrieved results from saved job")
        except Exception as e:
            print(f"       retrieval failed ({e}), will resubmit")

    if result is None:
        print(f"       submitting job ({len(circuits)} circuits, {shots} shots each)...")
        job = backend.run(circuits, shots=shots)
        jid = _get_job_id(job)
        _save_job(job, checkpoint_dir)
        print(f"       job submitted (id={jid}), waiting for result...")
        result = job.result(timeout=timeout)
        print("       result received")

    # Phase 4: Collect raw counts
    raw_results = {}
    for i, x in enumerate(all_x):
        counts = result.get_counts(i)
        raw_results[x] = counts

    # Phase 5: Clean up checkpoint files
    cleanup_checkpoint(checkpoint_dir)
    print("       checkpoint cleaned up")

    return raw_results


def clifford_tester_paired_runs(
    U_circuit: QuantumCircuit,
    n: int,
    *,
    shots: int = 1000,
    backend: BackendV2,
    transpilation_function: Callable[[QuantumCircuit], QuantumCircuit] | None = None,
    timeout: float | None = None,
    checkpoint_dir: Path,
) -> list[dict[str, Any]]:
    """
    Four-query Clifford tester algorithm (paired runs).

    Tests whether a unitary U is a Clifford gate by:
    1. Sampling random x from F_2^{2n}
    2. Running U^{⊗2}|P_x⟩⟩ twice with Bell basis measurement
    3. Recording both outcomes y1, y2

    Args:
        U_circuit: A quantum circuit implementing the n-qubit unitary U
        n: Number of qubits U acts on
        shots: Number of times to run the test
        backend: Qiskit backend to run on (defaults to AerSimulator)
        transpilation_function: Optional function to transpile circuits
        checkpoint_dir: Directory for checkpoint files (plan.json, jobs.json)

    Returns:
        list of dicts, each with keys "x", "y1", "y2"
    """
    transpilation_function = transpilation_function or default_transpilation_function

    # Phase 1: Load or generate plan
    plan = load_paired_plan(checkpoint_dir)
    if plan is not None:
        x_counts = plan.to_counter()
        print(f"       loaded existing plan ({len(x_counts)} unique x, {sum(x_counts.values())} total shots)")
    else:
        xs = [tuple(int(v) for v in np.random.randint(0, 2, size=2 * n)) for _ in range(shots)]
        x_counts = Counter(xs)
        save_plan(PairedPlan.from_counter(n, shots, x_counts), checkpoint_dir)
        print(f"       created new plan ({len(x_counts)} unique x, {shots} total shots)")

    # Phase 2: Build & transpile one circuit per unique x
    print(f"       building {len(x_counts)} circuits...")
    circuits: dict[tuple[int, ...], QuantumCircuit] = {}
    for x in x_counts:
        qc = get_clifford_tester_circuit(U_circuit, n, x)
        circuits[x] = transpilation_function(qc)

    # Phase 3: Load existing jobs state
    jobs_state = load_paired_jobs(checkpoint_dir)
    if jobs_state is not None:
        already_done = sum(1 for k in jobs_state.jobs if jobs_state.jobs[k].counts is not None)
        print(f"       loaded jobs checkpoint ({already_done}/{len(x_counts)} already collected)")
    else:
        jobs_state = PairedJobsState()

    # Phase 4: For each x, collect results (skip/retrieve/submit as needed)
    for idx, (x, count) in enumerate(x_counts.items(), 1):
        if (entry := jobs_state.get_entry(x)) is not None:
            # Already have counts — skip
            if entry.counts is not None:
                continue

            # Have a saved job file — try to retrieve result
            if entry.job_id:
                saved_job = _load_job(backend, checkpoint_dir)
                if saved_job is not None:
                    jid = _get_job_id(saved_job)
                    print(f"       [{idx}/{len(x_counts)}] x={list(x)}: loaded saved job (id={jid}), retrieving...")
                    try:
                        counts = saved_job.result(timeout=timeout).get_counts()
                        jobs_state.set_entry(x, PairedJobEntry(job_id=entry.job_id, counts=counts))
                        save_paired_jobs(jobs_state, checkpoint_dir)
                        print(f"       [{idx}/{len(x_counts)}] x={list(x)}: retrieved")
                        continue
                    except Exception as e:
                        print(f"       [{idx}/{len(x_counts)}] x={list(x)}: retrieval failed ({e}), resubmitting")

        # Submit new job
        print(f"       [{idx}/{len(x_counts)}] x={list(x)}: submitting ({2 * count} shots)...")
        job = backend.run(circuits[x], shots=2 * count)
        jid = _get_job_id(job)
        jobs_state.set_entry(x, PairedJobEntry(job_id=jid))
        _save_job(job, checkpoint_dir)
        save_paired_jobs(jobs_state, checkpoint_dir)

        counts = job.result(timeout=timeout).get_counts()
        jobs_state.set_entry(x, PairedJobEntry(job_id=jid, counts=counts))
        save_paired_jobs(jobs_state, checkpoint_dir)
        print(f"       [{idx}/{len(x_counts)}] x={list(x)}: done (id={jid})")

    # Phase 5: Expand counts → shuffle → pair
    raw_results: list[dict[str, Any]] = []
    for x, _ in x_counts.items():
        entry = jobs_state.get_entry(x)
        assert entry is not None and entry.counts is not None
        counts = entry.counts

        outcomes: list[str] = []
        for bitstring, freq in counts.items():
            outcomes.extend([bitstring] * freq)
        np.random.shuffle(outcomes)

        for i in range(0, len(outcomes) - 1, 2):
            raw_results.append({"x": x, "y1": outcomes[i], "y2": outcomes[i + 1]})

    # Phase 6: Clean up checkpoint files
    cleanup_checkpoint(checkpoint_dir)
    print("       checkpoint cleaned up")

    return raw_results
