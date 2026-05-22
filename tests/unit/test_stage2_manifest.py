from __future__ import annotations

import json
from pathlib import Path

from engine.classify.stage2_manifest import Stage2Manifest


def _make_manifest() -> Stage2Manifest:
    return Stage2Manifest(
        model_identity="meta-llama/Llama-3.1-70B-Instruct",
        weight_provenance_hash="abc123" * 8,
        prompt_hash="def456" * 8,
        prompt_template_version="1.0.0",
        batch_size=50,
        prng_seed=20260520,
        temperature=0.0,
        top_p=1.0,
        cost_ceiling_usd=500.0,
        provider="runpod",
        gpu_type="H100_80GB",
        gpu_count=1,
        region="US-TX-3",
        runpod_job_ids=(),
        wall_time_seconds=None,
        actual_cost_usd=None,
        incidents_classified=None,
    )


class TestStage2Manifest:
    def test_round_trip_json(self, tmp_path: Path) -> None:
        m = _make_manifest()
        path = tmp_path / "stage2_manifest.json"
        m.write(path)
        loaded = Stage2Manifest.read(path)
        assert loaded.model_identity == m.model_identity
        assert loaded.weight_provenance_hash == m.weight_provenance_hash
        assert loaded.prompt_hash == m.prompt_hash
        assert loaded.batch_size == m.batch_size
        assert loaded.temperature == 0.0
        assert loaded.top_p == 1.0
        assert loaded.cost_ceiling_usd == 500.0

    def test_determinism_fields(self) -> None:
        m = _make_manifest()
        assert m.temperature == 0.0
        assert m.top_p == 1.0
        assert m.prng_seed == 20260520

    def test_to_json_is_canonical(self) -> None:
        m = _make_manifest()
        j1 = m.to_json()
        j2 = m.to_json()
        assert j1 == j2
        parsed = json.loads(j1)
        assert parsed["model_identity"] == "meta-llama/Llama-3.1-70B-Instruct"

    def test_immutable(self) -> None:
        m = _make_manifest()
        try:
            m.model_identity = "other"  # type: ignore[misc]
            raise AssertionError("should be frozen")
        except AttributeError:
            pass

    def test_with_execution_results(self, tmp_path: Path) -> None:
        m = Stage2Manifest(
            model_identity="meta-llama/Llama-3.1-70B-Instruct",
            weight_provenance_hash="abc123" * 8,
            prompt_hash="def456" * 8,
            prompt_template_version="1.0.0",
            batch_size=50,
            prng_seed=20260520,
            temperature=0.0,
            top_p=1.0,
            cost_ceiling_usd=500.0,
            provider="runpod",
            gpu_type="H100_80GB",
            gpu_count=1,
            region="US-TX-3",
            runpod_job_ids=("job-001", "job-002"),
            wall_time_seconds=3600.0,
            actual_cost_usd=87.50,
            incidents_classified=7000,
        )
        path = tmp_path / "s2m.json"
        m.write(path)
        loaded = Stage2Manifest.read(path)
        assert loaded.runpod_job_ids == ("job-001", "job-002")
        assert loaded.wall_time_seconds == 3600.0
        assert loaded.actual_cost_usd == 87.50
        assert loaded.incidents_classified == 7000
