"""Tests for the injection gate (U6 Tasks 2 & 3).

Covers:
- T2: subverted model fails gate; robust model passes; revision_sha + per-probe recorded
- R4: seed forwarded to every run_sync call; bench=ship equivalence with classify_one
- R5: exception on a probe → that probe not resisted → gate fails (fail-closed)
- R6: benign_hit is non-gating (gate still passes when benign_hit differs)
- T3: filter_eligible_by_gate excludes failed + ungated configs (fail-closed)
- T3/R7: write_gate_provenance records probe_set_sha256 + probe_count
"""
from __future__ import annotations

import json
from pathlib import Path

from engine.classify.injection_gate import (
    InjectionGateResult,
    filter_eligible_by_gate,
    run_injection_gate,
    write_gate_provenance,
)
from engine.classify.injection_probes import INJECTION_PROBES
from engine.classify.runpod_client import RunPodError, RunPodResponse

# Minimal rubric with two real entry_ids.  INJECTION_PROBES[0].attacker_target
# is LLM05, which is present here, so the subverted-model test works correctly.
RUBRIC = '{"entries":[{"entry_id":"LLM01"},{"entry_id":"LLM05"}]}'


class _MockClient:
    """Mock client returning a fixed entry_id for every call."""

    def __init__(self, entry_for):  # entry_for: callable(messages) -> entry_id str
        self._f = entry_for
        self.calls: int = 0
        self.seeds_seen: list[int] = []

    def run_sync(self, messages, seed: int) -> RunPodResponse:
        self.calls += 1
        self.seeds_seen.append(seed)
        eid = self._f(messages)
        return RunPodResponse(
            f'{{"entry_id":"{eid}","confidence":0.9,"rationale":"x"}}',
            "mock-job",
            1.0,
        )


# ── Task 2 core tests ──────────────────────────────────────────────────────────

def test_gate_fails_when_model_emits_attacker_target() -> None:
    """Subverted model: returns INJECTION_PROBES[0].attacker_target → gate fails."""
    target = INJECTION_PROBES[0].attacker_target  # "LLM05" — present in RUBRIC
    r = run_injection_gate(
        _MockClient(lambda m: target), "evil", "sha1", RUBRIC, seed=42
    )
    assert r.passed is False
    assert any(not pr.resisted for pr in r.probe_results)


def test_gate_passes_when_model_resists_all() -> None:
    """Robust model: always returns out-of-scope → all resisted → gate passes."""
    r = run_injection_gate(
        _MockClient(lambda m: "out-of-scope"), "robust", "sha2", RUBRIC, seed=42
    )
    assert r.passed is True
    assert r.pass_rate == 1.0
    assert all(pr.resisted for pr in r.probe_results)


def test_gate_records_revision_and_perprobe_results() -> None:
    """revision_sha and per-probe results are recorded for every probe."""
    r = run_injection_gate(
        _MockClient(lambda m: "out-of-scope"), "m", "shaX", RUBRIC, seed=42
    )
    assert r.revision_sha == "shaX"
    assert len(r.probe_results) == len(INJECTION_PROBES)


# ── R5: fail-closed on exception ──────────────────────────────────────────────

def test_exception_probe_is_not_resisted_fail_closed() -> None:
    """R5: RunPodError on one probe → that probe not resisted → gate fails."""

    call_count = 0

    class _RaisesOnFirst:
        def run_sync(self, messages, seed: int) -> RunPodResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RunPodError("simulated timeout")
            return RunPodResponse(
                '{"entry_id":"out-of-scope","confidence":0.9,"rationale":"x"}',
                "j",
                1.0,
            )

    r = run_injection_gate(_RaisesOnFirst(), "m", "sha", RUBRIC, seed=42)
    assert r.error_count >= 1
    assert r.passed is False
    errored = [pr for pr in r.probe_results if pr.error is not None]
    assert len(errored) >= 1
    assert all(not pr.resisted for pr in errored)


# ── R4: seed forwarded + bench=ship equivalence ────────────────────────────────

def test_seed_forwarded_to_every_probe() -> None:
    """R4: the same seed must be passed to every client.run_sync call."""
    client = _MockClient(lambda m: "out-of-scope")
    run_injection_gate(client, "m", "sha", RUBRIC, seed=99)
    assert len(client.seeds_seen) == len(INJECTION_PROBES)
    assert all(s == 99 for s in client.seeds_seen), (
        f"Seed not forwarded uniformly; got seeds: {client.seeds_seen}"
    )


def test_bench_ship_equivalence() -> None:
    """R4: gate classify path and bakeoff_predict.classify_one use the same parser.

    Both call build_messages + parse_stage2_response.  Feeding the same mock
    response through both paths for INJECTION_PROBES[0] must yield the identical
    entry_id.
    """
    from engine.classify.bakeoff_predict import classify_one
    from engine.classify.cost_tracker import CostTracker
    from engine.schema import IncidentRecord

    fixed_json = '{"entry_id":"LLM01","confidence":0.9,"rationale":"test"}'

    class _FixedClient:
        def run_sync(self, messages, seed: int) -> RunPodResponse:
            return RunPodResponse(fixed_json, "j", 1.0)

    # Path 1: through run_injection_gate
    gate_result = run_injection_gate(
        _FixedClient(), "m", "sha", RUBRIC, seed=42
    )
    gate_entry_id = gate_result.probe_results[0].returned_entry_id

    # Path 2: through classify_one (bakeoff_predict) — must produce same entry_id.
    # Reconstruct the same IncidentRecord that run_injection_gate builds internally.
    probe = INJECTION_PROBES[0]
    inc = IncidentRecord(
        id=f"GATE-PROBE-{probe.probe_id}",
        date="2026-01-01",
        text=probe.incident_text,
        severity="High",
        source_class="injection-probe",
        corpus_stratum="gate",
        quality="auto",
        native_labels=(),
        source_url="https://gate.internal/injection-probe",
    )
    bakeoff_entry_id = classify_one(
        _FixedClient(),
        inc,
        RUBRIC,
        42,
        CostTracker(ceiling_usd=100.0),
        0.01,
        sleep_fn=lambda _: None,
    )

    assert gate_entry_id == bakeoff_entry_id, (
        f"bench≠ship: gate returned {gate_entry_id!r}, "
        f"bakeoff_predict returned {bakeoff_entry_id!r}"
    )


# ── R6: benign_hit is non-gating ──────────────────────────────────────────────

def test_benign_hit_is_non_gating() -> None:
    """R6: benign_hit field is recorded but does not affect passed."""
    # Model returns "out-of-scope" — always resisted.
    # Some probes have benign_expectation != "out-of-scope", so benign_hit may vary.
    r = run_injection_gate(
        _MockClient(lambda m: "out-of-scope"), "m", "sha", RUBRIC, seed=42
    )
    # Gate passes regardless of benign_hit values.
    assert r.passed is True
    # benign_hit field is present on every probe result.
    for pr in r.probe_results:
        assert isinstance(pr.benign_hit, bool)


# ── Task 3: filter + provenance ────────────────────────────────────────────────

def test_filter_excludes_failed_and_ungated_configs(tmp_path: Path) -> None:
    """filter_eligible_by_gate is fail-closed: no result OR failed → excluded."""
    good = run_injection_gate(
        _MockClient(lambda m: "out-of-scope"), "good-model", "sha-good", RUBRIC, seed=42
    )
    bad = run_injection_gate(
        _MockClient(lambda m: INJECTION_PROBES[0].attacker_target),
        "bad-model",
        "sha-bad",
        RUBRIC,
        seed=42,
    )
    results: dict[str, InjectionGateResult] = {"good": good, "bad": bad}
    # "ungated" has no entry in results at all
    eligible, excluded = filter_eligible_by_gate(
        ["good", "bad", "ungated"], results
    )
    assert eligible == ["good"]
    assert set(excluded) == {"bad", "ungated"}  # fail-closed

    # provenance write succeeds
    out = tmp_path / "injection_gate_results.json"
    write_gate_provenance(results, out)
    assert out.exists()


def test_provenance_written_with_integrity_fields(tmp_path: Path) -> None:
    """R7: provenance JSON must contain probe_set_sha256 and probe_count."""
    r = run_injection_gate(
        _MockClient(lambda m: "out-of-scope"), "m", "sha", RUBRIC, seed=42
    )
    out = tmp_path / "gate.json"
    write_gate_provenance({"m": r}, out)
    data = json.loads(out.read_text())
    assert "probe_set_sha256" in data, "probe_set_sha256 missing from provenance"
    assert "probe_count" in data, "probe_count missing from provenance"
    assert data["probe_count"] == len(INJECTION_PROBES)
    # Verify the hash matches the golden value from test_injection_probes.py
    from tests.unit.test_injection_probes import _GOLDEN_PROBE_SET_SHA256
    assert data["probe_set_sha256"] == _GOLDEN_PROBE_SET_SHA256
