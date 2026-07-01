import json
from pathlib import Path

S = Path("projects/owasp-llm/cycles/2026-rarr/prereg/stage2_manifest.json")


def test_stage2_locks_cost_and_leaves_winner_pending() -> None:
    d = json.loads(S.read_text())
    assert d["cost_ceiling_usd"] == 500.0 and d["abort_factor"] == 1.2
    assert d["provider"] == "runpod" and d["prng_seed"] == 20260520
    assert d["temperature"] == 0.0 and d["top_p"] == 1.0
    assert d["selected_from"] == "bakeoff_grid.json"
    for pending in (
        "model_identity",
        "weight_provenance_hash",
        "gpu_type",
        "gpu_count",
        "wall_time_seconds",
        "actual_cost_usd",
        "incidents_classified",
        "injection_gate_passed",
        "injection_gate_revision_sha",
    ):
        assert d[pending] is None, f"{pending} must be null at lock time"
    assert d["runpod_job_ids"] == []
