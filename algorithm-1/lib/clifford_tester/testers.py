from collections import Counter
from collections.abc import Callable
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
from qiskit import QuantumCircuit

from .results import (
    BatchedJobsState,
    BatchedPlan,
    PairedJobEntry,
    PairedJobsState,
    PairedPlan,
    cleanup_checkpoint,
    load_batched_jobs,
    load_batched_plan,
    load_paired_jobs,
    load_paired_plan,
    save_batched_jobs,
    save_paired_jobs,
    save_plan,
)
from .utils import default_backend_and_transpilation, get_clifford_tester_circuit


def clifford_tester_batched(
    U_circuit: QuantumCircuit,
    n: int,
    shots: int = 1000,
    backend: Any = None,
    transpilation_function: Callable[[QuantumCircuit], QuantumCircuit] | None = None,
    timeout: float | None = None,
    checkpoint_dir: Path | None = None,
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
    backend, transpilation_function = default_backend_and_transpilation(backend, transpilation_function)

    # Phase 1: Load or generate plan
    plan = load_batched_plan(checkpoint_dir) if checkpoint_dir else None
    if plan is not None:
        all_x = plan.to_tuples()
    else:
        all_x = list(product([0, 1], repeat=2 * n))
        if checkpoint_dir:
            plan = BatchedPlan(n=n, shots_per_x=shots, all_x=[list(x) for x in all_x])
            save_plan(plan, checkpoint_dir)

    # Phase 2: Build circuits
    circuits = [transpilation_function(get_clifford_tester_circuit(U_circuit, n, x)) for x in all_x]

    # Phase 3: Check for existing job or submit new one
    jobs_state = load_batched_jobs(checkpoint_dir) if checkpoint_dir else None
    result = None

    if jobs_state is not None:
        try:
            job = backend.retrieve_job(jobs_state.job_id)
            result = job.result(timeout=timeout)
        except Exception:
            jobs_state = None

    if result is None:
        job = backend.run(circuits, shots=shots)
        if checkpoint_dir:
            save_batched_jobs(BatchedJobsState(job_id=job.job_id()), checkpoint_dir)
        result = job.result(timeout=timeout)

    # Phase 4: Collect raw counts
    raw_results = {}
    for i, x in enumerate(all_x):
        counts = result.get_counts(i)
        raw_results[x] = counts

    # Phase 5: Clean up checkpoint files
    if checkpoint_dir:
        cleanup_checkpoint(checkpoint_dir)

    return raw_results


def clifford_tester_paired_runs(
    U_circuit: QuantumCircuit,
    n: int,
    shots: int = 1000,
    backend: Any = None,
    transpilation_function: Callable[[QuantumCircuit], QuantumCircuit] | None = None,
    timeout: float | None = None,
    checkpoint_dir: Path | None = None,
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
    backend, transpilation_function = default_backend_and_transpilation(backend, transpilation_function)

    # Phase 1: Load or generate plan
    plan = load_paired_plan(checkpoint_dir) if checkpoint_dir else None
    if plan is not None:
        x_counts = plan.to_counter()
    else:
        xs = [tuple(int(v) for v in np.random.randint(0, 2, size=2 * n)) for _ in range(shots)]
        x_counts = Counter(xs)
        if checkpoint_dir:
            save_plan(PairedPlan.from_counter(n, shots, x_counts), checkpoint_dir)

    # Phase 2: Build & transpile one circuit per unique x
    circuits: dict[tuple[int, ...], QuantumCircuit] = {}
    for x in x_counts:
        qc = get_clifford_tester_circuit(U_circuit, n, x)
        circuits[x] = transpilation_function(qc)

    # Phase 3: Load existing jobs state
    jobs_state = load_paired_jobs(checkpoint_dir) if checkpoint_dir else None
    if jobs_state is None:
        jobs_state = PairedJobsState()

    # Phase 4: For each x, collect results (skip/retrieve/submit as needed)
    for x, count in x_counts.items():
        entry = jobs_state.get_entry(x)

        # Already have counts — skip
        if entry is not None and entry.counts is not None:
            continue

        # Have job_id but no counts — try to retrieve
        if entry is not None and entry.job_id:
            try:
                job = backend.retrieve_job(entry.job_id)
                counts = job.result(timeout=timeout).get_counts()
                jobs_state.set_entry(x, PairedJobEntry(job_id=entry.job_id, counts=counts))
                if checkpoint_dir:
                    save_paired_jobs(jobs_state, checkpoint_dir)
                continue
            except Exception:
                pass  # Fall through to resubmit

        # Submit new job
        job = backend.run(circuits[x], shots=2 * count)
        jobs_state.set_entry(x, PairedJobEntry(job_id=job.job_id()))
        if checkpoint_dir:
            save_paired_jobs(jobs_state, checkpoint_dir)

        counts = job.result(timeout=timeout).get_counts()
        jobs_state.set_entry(x, PairedJobEntry(job_id=job.job_id(), counts=counts))
        if checkpoint_dir:
            save_paired_jobs(jobs_state, checkpoint_dir)

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
    if checkpoint_dir:
        cleanup_checkpoint(checkpoint_dir)

    return raw_results
