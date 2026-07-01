"""Live bake-off predict_fn: retry+cost-aware classify_one and factory.

Offline/$0 contract: every test mocks HttpRunPodClient — no network, no pod,
no real model call.  The live run is U8.
"""
from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from engine.classify.cost_tracker import CostTracker
from engine.classify.runpod_client import HttpRunPodClient, RunPodError
from engine.classify.stage2 import parse_stage2_response
from engine.classify.stage2_prompt import build_messages
from engine.schema import IncidentRecord

# PredictFn: config_name → {incident_id: entry_id}
PredictFn = Callable[[str], dict[str, str]]


class RetryExhaustedError(RunPodError):
    """Raised by ``classify_one`` after exhausting all retry attempts."""


def _build_valid_entry_ids(rubric_json: str) -> frozenset[str]:
    """Extract the set of valid entry_ids from a rubric JSON string.

    Always includes ``"out-of-scope"`` as a valid classification target.
    """
    try:
        rubric = json.loads(rubric_json)
        entries = rubric.get("entries", [])
        return frozenset(e.get("entry_id", "") for e in entries) | {"out-of-scope"}
    except (json.JSONDecodeError, AttributeError):
        return frozenset({"out-of-scope"})


def classify_one(
    client: object,
    incident: IncidentRecord,
    rubric_json: str,
    seed: int,
    cost_tracker: CostTracker,
    cost_per_call_usd: float,
    *,
    max_retries: int = 3,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> str:
    """Classify a single incident with exponential-backoff retry and cost accounting.

    Parameters
    ----------
    client:
        Any object with ``run_sync(messages, seed) -> RunPodResponse``.
    incident:
        The incident to classify; ``incident.text`` is delimiter-neutralized
        by ``build_messages`` before being sent to the model.
    rubric_json:
        Serialised rubric; defines the valid ``entry_id`` values.
    seed:
        PRNG seed forwarded to the model endpoint.
    cost_tracker:
        Shared ``CostTracker``; receives one record per attempt.
    cost_per_call_usd:
        Fixed per-call cost estimate (USD).  Recorded on EVERY attempt —
        success AND each ``RunPodError`` — so aborted/retried calls are never
        under-counted (R1).  A synthetic job_id ``f"{incident.id}-attempt{n}"``
        is used when no ``RunPodResponse`` is available.
    max_retries:
        Maximum attempts before raising ``RetryExhaustedError``.
    sleep_fn:
        Injected sleep callable; pass ``lambda _: None`` in tests to skip
        real sleeps.

    Returns
    -------
    str
        The classified ``entry_id``, or ``"out-of-scope"`` on fallback.

    Raises
    ------
    RetryExhaustedError
        After ``max_retries`` consecutive ``RunPodError`` failures.
    CostCeilingExceeded
        Propagated immediately from ``cost_tracker.check_or_abort()``.
    """
    valid_entry_ids = _build_valid_entry_ids(rubric_json)
    messages = build_messages(incident, rubric_json)
    last_error: RunPodError | None = None

    for attempt in range(max_retries):
        try:
            resp = client.run_sync(messages, seed)  # type: ignore[attr-defined]
            # R1: record cost on success using the real job_id
            cost_tracker.record(
                job_id=resp.job_id,
                cost_usd=cost_per_call_usd,
                execution_time_ms=resp.execution_time_ms,
            )
            cost_tracker.check_or_abort()
            return parse_stage2_response(resp.output_text, valid_entry_ids)
        except RunPodError as exc:
            last_error = exc
            # R1: record cost even on failure; use synthetic job_id
            cost_tracker.record(
                job_id=f"{incident.id}-attempt{attempt}",
                cost_usd=cost_per_call_usd,
                execution_time_ms=0.0,
            )
            cost_tracker.check_or_abort()
            if attempt < max_retries - 1:
                sleep_fn(2.0 ** attempt)

    raise RetryExhaustedError(str(last_error)) from last_error


def build_live_predict_fn(
    pod_urls: dict[str, str],
    model_names: dict[str, str],
    goldset_incidents: dict[str, IncidentRecord],
    rubric_json: str,
    cost_tracker: CostTracker,
    cost_per_call: dict[str, float],
    *,
    seed: int,
    checkpoint_dir: Path | None = None,
    client_factory: object = HttpRunPodClient,
    max_workers: int = 18,
) -> PredictFn:
    """Build a label-blind ``predict_fn`` for the live bake-off.

    The returned ``predict_fn(config_name) -> dict[incident_id, entry_id]``
    classifies every incident in ``goldset_incidents`` via the named config's
    pod and returns the full mapping.  Labels are never passed in or out.

    Two-level checkpoint contract
    ==============================
    OUTER (``run_bakeoff`` level):
        ``checkpoint_dir/{config_name}.json`` — written by ``run_bakeoff``
        itself as a full per-config result dict.  On resume, ``run_bakeoff``
        short-circuits ``predict_fn`` entirely for already-complete configs.

    INNER (this function):
        ``checkpoint_dir/predict/predict_{config_name}.jsonl`` — one JSON line
        per completed incident (``{"incident_id": ..., "entry_id": ...}``),
        appended as each future finishes.  On re-entry, already-done incident
        ids are loaded and skipped so a mid-config crash resumes without
        re-invoking the LLM for finished work.  The ``predict/`` subdirectory
        isolates inner files from outer checkpoint files to avoid naming
        conflicts.

    Parameters
    ----------
    pod_urls:
        Mapping of config_name → pod base URL.
    model_names:
        Mapping of config_name → HuggingFace model identifier forwarded to
        the pod.
    goldset_incidents:
        Mapping of incident_id → IncidentRecord.  Records carry only ``id``
        and ``text`` — no labels are threaded through.
    rubric_json:
        Serialised rubric shared across all configs.
    cost_tracker:
        Shared tracker; receives one record per classify_one attempt across
        all configs.
    cost_per_call:
        Mapping of config_name → per-call cost estimate (USD).
    seed:
        PRNG seed forwarded to all model endpoints.
    checkpoint_dir:
        Directory for checkpoint files.  ``None`` disables checkpointing.
    client_factory:
        Callable that produces a RunPodClient; defaults to ``HttpRunPodClient``.
        Injectable for offline tests.
    max_workers:
        Thread-pool size for concurrent incident classification.

    Returns
    -------
    PredictFn
        ``predict_fn(config_name) -> dict[incident_id, entry_id]``

    Raises
    ------
    ValueError
        If ``config_name`` is not in ``pod_urls`` (R2 — raised before any
        thread is spawned; zero client calls).
    """

    def predict_fn(config_name: str) -> dict[str, str]:
        # R2: validate config has a pod_url BEFORE spawning any threads
        if config_name not in pod_urls:
            raise ValueError(
                f"Config {config_name!r} has no entry in pod_urls. "
                f"Available configs: {sorted(pod_urls)}"
            )

        url = pod_urls[config_name]
        model_name = model_names.get(config_name, "")
        call_cost = cost_per_call[config_name]
        client = client_factory(base_url=url, model_name=model_name)  # type: ignore[operator]

        # R5: load inner per-incident checkpoint
        done: dict[str, str] = {}
        checkpoint_file: Path | None = None
        write_lock = threading.Lock()

        if checkpoint_dir is not None:
            predict_subdir = checkpoint_dir / "predict"
            predict_subdir.mkdir(parents=True, exist_ok=True)
            checkpoint_file = predict_subdir / f"predict_{config_name}.jsonl"
            if checkpoint_file.exists():
                with checkpoint_file.open() as fh:
                    for line in fh:
                        line = line.strip()
                        if line:
                            rec = json.loads(line)
                            done[rec["incident_id"]] = rec["entry_id"]

        # Only classify incidents not already done
        todo = {
            inc_id: inc
            for inc_id, inc in goldset_incidents.items()
            if inc_id not in done
        }

        results: dict[str, str] = dict(done)

        def _worker(inc_id: str, inc: IncidentRecord) -> tuple[str, str]:
            entry_id = classify_one(
                client, inc, rubric_json, seed, cost_tracker, call_cost
            )
            return inc_id, entry_id

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_worker, inc_id, inc): inc_id
                for inc_id, inc in todo.items()
            }
            for future in as_completed(futures):
                # R2: .result() propagates worker exceptions instead of swallowing
                inc_id, entry_id = future.result()
                results[inc_id] = entry_id
                if checkpoint_file is not None:
                    with write_lock, checkpoint_file.open("a") as fh:
                        fh.write(
                            json.dumps({"incident_id": inc_id, "entry_id": entry_id})
                            + "\n"
                        )

        return results

    return predict_fn
