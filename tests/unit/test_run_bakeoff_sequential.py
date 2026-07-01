"""Unit tests for tools/run_bakeoff_sequential.py (offline / mock, $0).

Three test areas:
1. ``bakeoff`` mode — pre-written predictions + gate files, replay path, winner selection.
2. ``score`` mode  — injected mock client_factory, no network, verifies outputs written.
3. Gate filter     — a config whose gate_<config>.json records passed=False is excluded.
"""
from __future__ import annotations

import json
from pathlib import Path

from engine.classify.bakeoff import BakeoffResult
from engine.classify.runpod_client import RunPodResponse
from tools.run_bakeoff_sequential import (
    _deserialize_gate_result,
    _serialize_gate_result,
    cmd_bakeoff,
    cmd_score,
)

# ---------------------------------------------------------------------------
# Probe attacker_targets: rubric must contain every LLM0x used by INJECTION_PROBES.
# ---------------------------------------------------------------------------
_PROBE_TARGETS = [
    "LLM01", "LLM02", "LLM03", "LLM04", "LLM05",
    "LLM06", "LLM07", "LLM08", "LLM09", "LLM10",
]


# ---------------------------------------------------------------------------
# Shared fixture builder
# ---------------------------------------------------------------------------

def _write_cycle(
    base: Path,
    *,
    configs: list[dict[str, object]],
    n_class_a: int = 12,
    n_class_b: int = 12,
    cost_ceiling_usd: float = 500.0,
    lockbox_fraction: float = 0.3,
    seed: int = 42,
    min_cell: int = 5,
) -> tuple[Path, list[str]]:
    """Write a minimal locked cycle fixture.  Returns (cycle_dir, all_incident_ids)."""
    cycle_dir = base / "cycle"
    ids_a = [f"a{i}" for i in range(n_class_a)]
    ids_b = [f"b{i}" for i in range(n_class_b)]
    all_ids = ids_a + ids_b

    (cycle_dir / "prereg").mkdir(parents=True, exist_ok=True)

    # bakeoff_grid.json
    grid: dict[str, object] = {
        "grid_version": "test.1",
        "selection": {
            "lockbox_fraction": lockbox_fraction,
            "alpha": 0.05,
            "min_cell": min_cell,
            "seed": seed,
            "winner_none_rule": "empty_eligible_set",
        },
        "configs": configs,
    }
    (cycle_dir / "prereg" / "bakeoff_grid.json").write_text(json.dumps(grid))

    # stage2_manifest.json (extra fields tolerated by _load_manifest_cost_fields)
    manifest: dict[str, object] = {
        "model_identity": None,
        "weight_provenance_hash": None,
        "selected_from": "bakeoff_grid.json",
        "prompt_hash": "abc",
        "prompt_template_version": "1.0.0",
        "batch_size": 1,
        "prng_seed": seed,
        "temperature": 0.0,
        "top_p": 1.0,
        "cost_ceiling_usd": cost_ceiling_usd,
        "abort_factor": 1.2,
        "provider": "runpod",
        "gpu_type": None,
        "gpu_count": None,
        "region": None,
        "runpod_job_ids": [],
        "wall_time_seconds": None,
        "actual_cost_usd": None,
        "incidents_classified": None,
        "injection_gate_passed": None,
        "injection_gate_revision_sha": None,
    }
    (cycle_dir / "prereg" / "stage2_manifest.json").write_text(json.dumps(manifest))

    # rubric.json — must include all INJECTION_PROBES attacker_targets (LLM01–LLM10)
    # plus "A" and "B" for the goldset classes used in tests.
    entries = [{"entry_id": eid} for eid in _PROBE_TARGETS + ["A", "B"]]
    rubric: dict[str, object] = {"entries": entries}
    (cycle_dir / "prereg" / "rubric.json").write_text(json.dumps(rubric))

    # adjudicated_goldset.jsonl
    (cycle_dir / "calibration").mkdir(parents=True, exist_ok=True)
    lines = []
    for inc_id in ids_a:
        lines.append(json.dumps({
            "incident_id": inc_id, "llm_consensus": "A",
            "adjudicated": "accept", "labels": ["A"],
            "blind_label": "A", "notes": None,
        }))
    for inc_id in ids_b:
        lines.append(json.dumps({
            "incident_id": inc_id, "llm_consensus": "B",
            "adjudicated": "accept", "labels": ["B"],
            "blind_label": "B", "notes": None,
        }))
    (cycle_dir / "calibration" / "adjudicated_goldset.jsonl").write_text(
        "\n".join(lines) + "\n"
    )

    # corpora snapshot — text-only, no labels
    (cycle_dir / "corpora").mkdir(parents=True, exist_ok=True)
    snapshot: list[dict[str, object]] = [
        {"id": inc_id, "title": f"incident {inc_id}", "description": "", "impact": ""}
        for inc_id in all_ids
    ]
    (cycle_dir / "corpora" / "incidents.json").write_text(
        json.dumps({"incidents": snapshot})
    )

    return cycle_dir, all_ids


def _make_config(name: str, gpu_count: int = 2) -> dict[str, object]:
    """Return a minimal ModelConfig dict for bakeoff_grid.json."""
    return {
        "name": name,
        "model_id": f"test-org/{name}",
        "revision_sha": "a" * 40,
        "gpu_type": "NVIDIA H200",
        "gpu_count": gpu_count,
    }


def _write_floor(tmp: Path, all_ids: list[str]) -> Path:
    """Write a floor labeled_incidents.json (all incidents → 'A')."""
    records = [
        {"incident_id": inc_id, "entry_id": "A", "confidence": 0.5, "stage": 1}
        for inc_id in all_ids
    ]
    p = tmp / "floor_labeled_incidents.json"
    p.write_text(json.dumps(records))
    return p


def _gate_payload(*, passed: bool) -> dict[str, object]:
    """Minimal gate JSON compatible with _deserialize_gate_result."""
    return {
        "model_name": "test-model",
        "revision_sha": "a" * 40,
        "passed": passed,
        "pass_rate": 1.0 if passed else 0.0,
        "threshold": 1.0,
        "error_count": 0,
        "probe_results": [],
    }


# ---------------------------------------------------------------------------
# 1. bakeoff mode: picks the perfect config, writes classify_provenance.json
# ---------------------------------------------------------------------------

def test_bakeoff_mode_picks_perfect_config(tmp_path: Path) -> None:
    """Pre-written predictions (2 gate-passing, 2 gate-failing) → perfect config wins."""
    configs = [
        _make_config("cfg-perfect"),
        _make_config("cfg-decent"),
        _make_config("cfg-fail1"),
        _make_config("cfg-fail2"),
    ]
    cycle_dir, all_ids = _write_cycle(tmp_path, configs=configs)
    seq_dir = cycle_dir / "classify" / "seq"
    seq_dir.mkdir(parents=True, exist_ok=True)

    ids_a = [i for i in all_ids if i.startswith("a")]
    ids_b = [i for i in all_ids if i.startswith("b")]

    # cfg-perfect: correct labels for all incidents
    perfect_preds = {k: "A" for k in ids_a} | {k: "B" for k in ids_b}
    # cfg-decent: all incidents → "A" (same as floor; won't beat floor)
    decent_preds = {k: "A" for k in all_ids}

    (seq_dir / "predictions_cfg-perfect.json").write_text(
        json.dumps(perfect_preds, sort_keys=True)
    )
    (seq_dir / "predictions_cfg-decent.json").write_text(
        json.dumps(decent_preds, sort_keys=True)
    )
    # Gate: cfg-perfect and cfg-decent pass; cfg-fail1/2 fail
    (seq_dir / "gate_cfg-perfect.json").write_text(
        json.dumps(_gate_payload(passed=True))
    )
    (seq_dir / "gate_cfg-decent.json").write_text(
        json.dumps(_gate_payload(passed=True))
    )
    (seq_dir / "gate_cfg-fail1.json").write_text(
        json.dumps(_gate_payload(passed=False))
    )
    (seq_dir / "gate_cfg-fail2.json").write_text(
        json.dumps(_gate_payload(passed=False))
    )

    floor_path = _write_floor(tmp_path, all_ids)
    result = cmd_bakeoff(cycle_dir, floor_path=floor_path)

    assert isinstance(result, BakeoffResult)
    assert result.winner == "cfg-perfect"
    prov_path = cycle_dir / "results" / "bakeoff_seq" / "classify_provenance.json"
    assert prov_path.exists(), f"classify_provenance.json not written at {prov_path}"
    prov = json.loads(prov_path.read_text())
    assert prov["winner"] == "cfg-perfect"


# ---------------------------------------------------------------------------
# 2. score mode: mock client → predictions + gate files written, no network
# ---------------------------------------------------------------------------

def test_score_mode_writes_predictions_and_gate(tmp_path: Path) -> None:
    """Injected mock client returns 'out-of-scope'; score writes both output files."""
    cycle_dir, all_ids = _write_cycle(
        tmp_path,
        configs=[_make_config("test-cfg")],
    )

    # Mock client: always returns "out-of-scope" (valid entry_id; never an
    # attacker_target, so every injection probe records resisted=True → gate passes).
    class _MockClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.calls = 0

        def run_sync(self, messages: object, seed: int) -> RunPodResponse:
            self.calls += 1
            return RunPodResponse(
                output_text=json.dumps(
                    {"entry_id": "out-of-scope", "confidence": 0.9, "rationale": "mock"}
                ),
                job_id=f"mock-job-{self.calls}",
                execution_time_ms=1.0,
            )

    mock_clients: list[_MockClient] = []

    def mock_factory(*args: object, **kwargs: object) -> _MockClient:
        c = _MockClient(*args, **kwargs)
        mock_clients.append(c)
        return c

    cmd_score(
        cycle_dir=cycle_dir,
        config_name="test-cfg",
        pod_url="http://mock-pod",
        client_factory=mock_factory,
    )

    seq_dir = cycle_dir / "classify" / "seq"

    # Predictions file: must exist and contain all goldset incident IDs
    preds_path = seq_dir / "predictions_test-cfg.json"
    assert preds_path.exists(), "predictions_test-cfg.json not written"
    preds: dict[str, str] = json.loads(preds_path.read_text())
    assert set(preds.keys()) == set(all_ids), (
        f"predictions missing {set(all_ids) - set(preds.keys())}"
    )

    # Gate file: must exist and gate passed (all probes returned "out-of-scope"
    # which != any attacker_target)
    gate_path = seq_dir / "gate_test-cfg.json"
    assert gate_path.exists(), "gate_test-cfg.json not written"
    gate_data: dict[str, object] = json.loads(gate_path.read_text())
    assert gate_data["passed"] is True, (
        f"gate should pass but passed={gate_data['passed']}, "
        f"pass_rate={gate_data['pass_rate']}"
    )

    # Verify mock was actually called (not short-circuited somehow)
    assert sum(c.calls for c in mock_clients) > 0


# ---------------------------------------------------------------------------
# 3. Gate filter: a config with passed=False is excluded from bakeoff
# ---------------------------------------------------------------------------

def test_bakeoff_mode_excludes_gate_failed_configs(tmp_path: Path) -> None:
    """A config whose gate_<config>.json has passed=False is excluded; passing one wins."""
    configs = [
        _make_config("pass-cfg"),
        _make_config("fail-cfg"),
    ]
    cycle_dir, all_ids = _write_cycle(tmp_path, configs=configs)
    seq_dir = cycle_dir / "classify" / "seq"
    seq_dir.mkdir(parents=True, exist_ok=True)

    ids_a = [i for i in all_ids if i.startswith("a")]
    ids_b = [i for i in all_ids if i.startswith("b")]

    # Both configs would produce perfect predictions, but fail-cfg is gate-excluded
    perfect_preds = {k: "A" for k in ids_a} | {k: "B" for k in ids_b}
    (seq_dir / "predictions_pass-cfg.json").write_text(json.dumps(perfect_preds))
    (seq_dir / "predictions_fail-cfg.json").write_text(json.dumps(perfect_preds))

    (seq_dir / "gate_pass-cfg.json").write_text(json.dumps(_gate_payload(passed=True)))
    (seq_dir / "gate_fail-cfg.json").write_text(json.dumps(_gate_payload(passed=False)))

    floor_path = _write_floor(tmp_path, all_ids)
    result = cmd_bakeoff(cycle_dir, floor_path=floor_path)

    # fail-cfg excluded → only pass-cfg in the eligible set
    assert result.winner == "pass-cfg"
    assert "fail-cfg" not in result.eligible_configs


# ---------------------------------------------------------------------------
# 4. Round-trip: _serialize_gate_result / _deserialize_gate_result
# ---------------------------------------------------------------------------

def test_gate_serialization_round_trip() -> None:
    """Serialized InjectionGateResult survives a JSON round-trip intact."""
    from engine.classify.injection_gate import InjectionGateResult, ProbeResult

    original = InjectionGateResult(
        model_name="test-model",
        revision_sha="b" * 40,
        passed=True,
        pass_rate=1.0,
        threshold=1.0,
        error_count=0,
        probe_results=(
            ProbeResult(
                probe_id="io-1",
                attacker_target="LLM05",
                returned_entry_id="out-of-scope",
                resisted=True,
                benign_hit=False,
                error=None,
            ),
        ),
    )
    serialized = _serialize_gate_result(original)
    restored = _deserialize_gate_result(serialized)

    assert restored.model_name == original.model_name
    assert restored.passed == original.passed
    assert restored.pass_rate == original.pass_rate
    assert len(restored.probe_results) == 1
    pr = restored.probe_results[0]
    assert pr.probe_id == "io-1"
    assert pr.resisted is True
    assert pr.error is None
