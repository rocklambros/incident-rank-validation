"""Live bake-off predict_fn: retry+cost-aware classify_one and factory.

Offline/$0 contract: every test mocks HttpRunPodClient — no network, no pod,
no real model call.  The live run is U8.
"""
from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path  # noqa: F401  — used by Task 2 (build_live_predict_fn)

from engine.classify.cost_tracker import CostTracker
from engine.classify.runpod_client import RunPodError
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
