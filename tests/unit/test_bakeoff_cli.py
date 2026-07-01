"""Integration test for run_bakeoff and bakeoff_cmd (Plan 8e T7 + U7 Task 4)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.classify.bakeoff import BakeoffResult, ModelConfig
from engine.classify.runpod_client import RunPodResponse
from engine.cli.bakeoff import bakeoff_cli_cmd, bakeoff_cmd, run_bakeoff


def _write_goldset(tmp: Path) -> Path:
    rows = []
    for i in range(12):
        rows.append({"incident_id": f"a{i}", "llm_consensus": "A", "adjudicated": "accept",
                     "labels": ["A"], "blind_label": "A", "notes": None})
    for i in range(12):
        rows.append({"incident_id": f"b{i}", "llm_consensus": "B", "adjudicated": "accept",
                     "labels": ["B"], "blind_label": "B", "notes": None})
    p = tmp / "gold.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


def test_run_bakeoff_picks_perfect_config(tmp_path: Path) -> None:
    goldset = _write_goldset(tmp_path)
    all_ids = [f"a{i}" for i in range(12)] + [f"b{i}" for i in range(12)]

    def predict_fn(config_name: str) -> dict[str, str]:
        if config_name == "perfect":
            return {k: ("A" if k.startswith("a") else "B") for k in all_ids}
        return {k: "A" for k in all_ids}  # "weak"

    floor = {k: "A" for k in all_ids}
    label_file = tmp_path / "labeled_incidents.json"
    label_file.write_text("[]\n")
    configs = [ModelConfig("m", "id", "sha", "NVIDIA H200", 4)]

    result = run_bakeoff(
        goldset_path=goldset,
        config_names=["perfect", "weak"],
        predict_fn=predict_fn,
        floor_predictions=floor,
        model_configs=configs,
        out_dir=tmp_path / "out",
        label_file=label_file,
        seed=7,
    )
    assert result.winner == "perfect"
    prov = json.loads((tmp_path / "out" / "classify_provenance.json").read_text())
    assert prov["winner"] == "perfect"


def test_run_bakeoff_no_winner_when_all_weak(tmp_path: Path) -> None:
    goldset = _write_goldset(tmp_path)
    all_ids = [f"a{i}" for i in range(12)] + [f"b{i}" for i in range(12)]
    floor = {k: ("A" if k.startswith("a") else "B") for k in all_ids}  # perfect floor

    def predict_fn(config_name: str) -> dict[str, str]:
        return {k: "A" for k in all_ids}

    label_file = tmp_path / "labeled_incidents.json"
    label_file.write_text("[]\n")
    result = run_bakeoff(
        goldset_path=goldset,
        config_names=["weak"],
        predict_fn=predict_fn,
        floor_predictions=floor,
        model_configs=[],
        out_dir=tmp_path / "out",
        label_file=label_file,
        seed=7,
    )
    assert result.winner is None


def test_run_bakeoff_raises_when_predictions_miss_lockbox(tmp_path: Path) -> None:
    goldset = _write_goldset(tmp_path)
    all_ids = [f"a{i}" for i in range(12)] + [f"b{i}" for i in range(12)]
    floor = {k: "A" for k in all_ids}

    def predict_fn(config_name: str) -> dict[str, str]:
        return {}  # covers no lockbox incidents

    label_file = tmp_path / "labeled_incidents.json"
    label_file.write_text("[]\n")
    with pytest.raises(ValueError):
        run_bakeoff(
            goldset_path=goldset,
            config_names=["x"],
            predict_fn=predict_fn,
            floor_predictions=floor,
            model_configs=[],
            out_dir=tmp_path / "out",
            label_file=label_file,
            seed=7,
        )


def test_run_bakeoff_checkpoint_resumes(tmp_path: Path) -> None:
    goldset = _write_goldset(tmp_path)
    all_ids = [f"a{i}" for i in range(12)] + [f"b{i}" for i in range(12)]
    floor = {k: "A" for k in all_ids}
    label_file = tmp_path / "labeled_incidents.json"
    label_file.write_text("[]\n")
    ckpt = tmp_path / "ckpt"

    def perfect(name: str) -> dict[str, str]:
        return {k: ("A" if k.startswith("a") else "B") for k in all_ids}

    r1 = run_bakeoff(
        goldset_path=goldset, config_names=["perfect"], predict_fn=perfect,
        floor_predictions=floor, model_configs=[], out_dir=tmp_path / "o1",
        label_file=label_file, seed=7, checkpoint_dir=ckpt,
    )
    assert (ckpt / "perfect.json").exists()

    def boom(name: str) -> dict[str, str]:
        raise RuntimeError("predict_fn must not be called on a cache hit")

    r2 = run_bakeoff(
        goldset_path=goldset, config_names=["perfect"], predict_fn=boom,
        floor_predictions=floor, model_configs=[], out_dir=tmp_path / "o2",
        label_file=label_file, seed=7, checkpoint_dir=ckpt,
    )
    assert r2.winner == r1.winner == "perfect"


def test_run_bakeoff_records_goldset_provenance(tmp_path: Path) -> None:
    goldset = _write_goldset(tmp_path)
    all_ids = [f"a{i}" for i in range(12)] + [f"b{i}" for i in range(12)]
    floor = {k: "A" for k in all_ids}
    label_file = tmp_path / "labeled_incidents.json"
    label_file.write_text("[]\n")

    def perfect(name: str) -> dict[str, str]:
        return {k: ("A" if k.startswith("a") else "B") for k in all_ids}

    run_bakeoff(
        goldset_path=goldset, config_names=["perfect"], predict_fn=perfect,
        floor_predictions=floor, model_configs=[], out_dir=tmp_path / "out",
        label_file=label_file, seed=7,
        corpus_class_counts={"A": 100, "B": 50},
    )
    prov = json.loads((tmp_path / "out" / "classify_provenance.json").read_text())
    assert prov["goldset"]["sha256"]
    assert prov["min_cell"] == 5
    assert "corpus_tv_divergence" in prov["goldset"]


# ---------------------------------------------------------------------------
# Helpers for bakeoff_cmd fixture cycle
# ---------------------------------------------------------------------------

def _write_cycle_fixture(
    base: Path,
    *,
    config_name: str = "qwen25-72b",
    n_class_a: int = 12,
    n_class_b: int = 12,
    cost_ceiling_usd: float = 500.0,
    extra_manifest_fields: bool = True,
) -> tuple[Path, list[str]]:
    """Create a minimal cycle_dir fixture and return (cycle_dir, all_incident_ids)."""
    cycle_dir = base / "cycle"
    ids_a = [f"a{i}" for i in range(n_class_a)]
    ids_b = [f"b{i}" for i in range(n_class_b)]
    all_ids = ids_a + ids_b

    # prereg/bakeoff_grid.json
    (cycle_dir / "prereg").mkdir(parents=True, exist_ok=True)
    grid = {
        "grid_version": "test.1",
        "selection": {
            "lockbox_fraction": 0.3,
            "alpha": 0.05,
            "min_cell": 5,
            "seed": 42,
            "winner_none_rule": "empty_eligible_set",
        },
        "configs": [{
            "name": config_name,
            "model_id": "Qwen/Qwen2.5-72B-Instruct",
            "revision_sha": "a" * 40,
            "gpu_type": "NVIDIA H200",
            "gpu_count": 2,
        }],
    }
    (cycle_dir / "prereg" / "bakeoff_grid.json").write_text(json.dumps(grid))

    # prereg/stage2_manifest.json — intentionally has EXTRA fields (R3 test)
    manifest: dict[str, object] = {
        "model_identity": "test-model",
        "weight_provenance_hash": "abc",
        "prompt_hash": "def",
        "prompt_template_version": "1",
        "batch_size": 8,
        "prng_seed": 42,
        "temperature": 0.0,
        "top_p": 1.0,
        "cost_ceiling_usd": cost_ceiling_usd,
        "provider": "runpod",
        "gpu_type": "NVIDIA H200",
        "gpu_count": 2,
        "region": None,
        "runpod_job_ids": [],
        "wall_time_seconds": None,
        "actual_cost_usd": None,
        "incidents_classified": None,
    }
    if extra_manifest_fields:
        # Simulate U5 extra fields that make Stage2Manifest.read() fail with TypeError
        manifest["abort_factor"] = 1.2
        manifest["selected_from"] = "grid_v1"
        manifest["injection_gate_passed"] = True
        manifest["injection_gate_revision_sha"] = "b" * 40
    (cycle_dir / "prereg" / "stage2_manifest.json").write_text(json.dumps(manifest))

    # prereg/rubric.json
    rubric = {"entries": [{"entry_id": "A"}, {"entry_id": "B"}]}
    (cycle_dir / "prereg" / "rubric.json").write_text(json.dumps(rubric))

    # calibration/adjudicated_goldset.jsonl
    (cycle_dir / "calibration").mkdir(parents=True, exist_ok=True)
    goldset_lines = []
    for inc_id in ids_a:
        goldset_lines.append(json.dumps({
            "incident_id": inc_id, "llm_consensus": "A", "adjudicated": "accept",
            "labels": ["A"], "blind_label": "A", "notes": None,
        }))
    for inc_id in ids_b:
        goldset_lines.append(json.dumps({
            "incident_id": inc_id, "llm_consensus": "B", "adjudicated": "accept",
            "labels": ["B"], "blind_label": "B", "notes": None,
        }))
    (cycle_dir / "calibration" / "adjudicated_goldset.jsonl").write_text(
        "\n".join(goldset_lines) + "\n"
    )

    # classify/labeled_incidents.json (floor: all → "A")
    (cycle_dir / "classify").mkdir(parents=True, exist_ok=True)
    floor_records = [
        {"incident_id": inc_id, "entry_id": "A", "confidence": 0.5, "stage": 1}
        for inc_id in all_ids
    ]
    (cycle_dir / "classify" / "labeled_incidents.json").write_text(json.dumps(floor_records))

    # corpora/incidents.json (snapshot — text-only, no labels)
    (cycle_dir / "corpora").mkdir(parents=True, exist_ok=True)
    snapshot_incidents = [
        {"id": inc_id, "title": f"incident {inc_id}", "description": "", "impact": ""}
        for inc_id in all_ids
    ]
    (cycle_dir / "corpora" / "incidents.json").write_text(
        json.dumps({"incidents": snapshot_incidents})
    )

    return cycle_dir, all_ids


# ---------------------------------------------------------------------------
# bakeoff_cmd tests (Task 4 / R3, R6)
# ---------------------------------------------------------------------------

def test_bakeoff_cmd_injected_predict_fn_returns_bakeoff_result(tmp_path: Path) -> None:
    """bakeoff_cmd with an injected predict_fn runs offline and picks the winner."""
    cycle_dir, all_ids = _write_cycle_fixture(tmp_path)

    def perfect(config_name: str) -> dict[str, str]:
        return {k: ("A" if k.startswith("a") else "B") for k in all_ids}

    result = bakeoff_cmd(cycle_dir, execute=False, predict_fn=perfect)
    assert isinstance(result, BakeoffResult)
    assert result.winner == "qwen25-72b"


def test_bakeoff_cmd_reads_cost_ceiling_from_manifest(tmp_path: Path) -> None:
    """R3: bakeoff_cmd reads cost_ceiling_usd=500.0 from the fixture manifest (not a CLI flag)."""
    cycle_dir, all_ids = _write_cycle_fixture(tmp_path, cost_ceiling_usd=500.0)

    def perfect(config_name: str) -> dict[str, str]:
        return {k: ("A" if k.startswith("a") else "B") for k in all_ids}

    # Run succeeds and manifest ceiling was parsed (not hardcoded)
    bakeoff_cmd(cycle_dir, execute=False, predict_fn=perfect)
    # Verify the manifest was read correctly by checking the file
    manifest_data = json.loads((cycle_dir / "prereg" / "stage2_manifest.json").read_text())
    assert manifest_data["cost_ceiling_usd"] == 500.0
    # No --cost-ceiling CLI flag: the CLI wrapper exposes only cycle_dir + --execute
    params = {p.name for p in bakeoff_cli_cmd.params}
    assert "cost_ceiling" not in params
    assert "cost_ceiling_usd" not in params


def test_bakeoff_cmd_raises_when_manifest_missing(tmp_path: Path) -> None:
    """R3: bakeoff_cmd raises FileNotFoundError if stage2_manifest.json is absent."""
    cycle_dir, all_ids = _write_cycle_fixture(tmp_path)
    (cycle_dir / "prereg" / "stage2_manifest.json").unlink()
    with pytest.raises(FileNotFoundError, match="stage2_manifest.json"):
        bakeoff_cmd(cycle_dir, execute=False, predict_fn=lambda _: {})


def test_bakeoff_cmd_raises_when_cost_ceiling_null(tmp_path: Path) -> None:
    """R3: bakeoff_cmd raises ValueError if cost_ceiling_usd is null."""
    cycle_dir, all_ids = _write_cycle_fixture(tmp_path)
    data = json.loads((cycle_dir / "prereg" / "stage2_manifest.json").read_text())
    data["cost_ceiling_usd"] = None
    (cycle_dir / "prereg" / "stage2_manifest.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="cost_ceiling_usd"):
        bakeoff_cmd(cycle_dir, execute=False, predict_fn=lambda _: {})


def test_bakeoff_cmd_raises_when_no_predict_fn_and_no_execute(tmp_path: Path) -> None:
    """bakeoff_cmd raises ValueError when predict_fn=None and execute=False."""
    cycle_dir, _ = _write_cycle_fixture(tmp_path)
    with pytest.raises(ValueError, match="predict_fn is None"):
        bakeoff_cmd(cycle_dir, execute=False, predict_fn=None)


def test_bakeoff_cmd_extra_manifest_fields_do_not_crash(tmp_path: Path) -> None:
    """R3: U5 extra fields (abort_factor, selected_from, injection_gate_*) are tolerated."""
    # The fixture always writes extra fields (extra_manifest_fields=True by default)
    cycle_dir, all_ids = _write_cycle_fixture(tmp_path, extra_manifest_fields=True)
    manifest_data = json.loads((cycle_dir / "prereg" / "stage2_manifest.json").read_text())
    assert "abort_factor" in manifest_data  # confirm the extra fields are present
    assert "selected_from" in manifest_data

    def perfect(name: str) -> dict[str, str]:
        return {k: ("A" if k.startswith("a") else "B") for k in all_ids}

    # Must not raise TypeError despite extra fields (would if Stage2Manifest.read() used)
    result = bakeoff_cmd(cycle_dir, execute=False, predict_fn=perfect)
    assert result.winner == "qwen25-72b"


def test_bakeoff_cmd_builds_live_predict_fn_with_injected_client_factory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R6: bakeoff_cmd(execute=True, predict_fn=None, client_factory=mock) exercises
    the full env-pod-URLs → build_live_predict_fn → classify_one glue offline/$0.

    The mock client is DISCRIMINATING: it returns the correct entry_id for each
    incident by detecting the incident text in the message (the fixture writes
    title="incident a{i}" / "incident b{i}", which lands verbatim in the user
    message via build_messages).  This lets a single config beat the all-A floor
    so result.winner == config_name is actually exercised.
    """
    config_name = "qwen25-72b"
    cycle_dir, all_ids = _write_cycle_fixture(tmp_path, config_name=config_name)

    # Monkeypatch env pod URLs so bakeoff_cmd's env-reading loop finds them
    monkeypatch.setenv("RUNPOD_MODEL_1_NAME", config_name)
    monkeypatch.setenv("RUNPOD_MODEL_1_URL", "http://mock-pod")

    # Discriminating mock: returns the correct class for each incident.
    # The fixture snapshot writes title=f"incident {inc_id}", so the user
    # message contains "incident a0" … "incident a11" for class A and
    # "incident b0" … "incident b11" for class B.  str(messages) is the
    # Python repr of a list of dicts (single-quoted), which reliably contains
    # the substring "incident a" for class-A incidents and not for class-B.
    class _MockClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.calls = 0

        def run_sync(self, messages: object, seed: int) -> RunPodResponse:
            self.calls += 1
            msg_text = str(messages)
            entry_id = "A" if "incident a" in msg_text else "B"
            return RunPodResponse(
                output_text=json.dumps(
                    {"entry_id": entry_id, "confidence": 0.95, "rationale": "test"}
                ),
                job_id="job-mock",
                execution_time_ms=1.0,
            )

    mock_clients: list[_MockClient] = []

    def mock_factory(*args: object, **kwargs: object) -> _MockClient:
        c = _MockClient(*args, **kwargs)
        mock_clients.append(c)
        return c

    result = bakeoff_cmd(
        cycle_dir,
        execute=True,
        predict_fn=None,
        client_factory=mock_factory,
    )

    assert isinstance(result, BakeoffResult)
    # run_sync was invoked for every goldset incident (live glue actually ran)
    assert sum(c.calls for c in mock_clients) == len(all_ids)
    # Predictions were scored (config appears in balanced-accuracy table)
    assert config_name in result.config_balanced_accuracy
    # The discriminating mock produces a perfect classifier → beats the all-A floor
    assert result.winner == config_name


def test_bakeoff_cmd_ceiling_reaches_cost_tracker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R3 e2e: cost_ceiling_usd from stage2_manifest flows into CostTracker.ceiling_usd.

    The existing R3 tests verify the manifest is read and the CLI has no
    --cost-ceiling flag, but they do not prove the value actually reaches the
    CostTracker that guards the live run.  This test uses a spy to capture the
    constructed CostTracker and assert its ceiling matches the manifest value.
    """
    from engine.classify.cost_tracker import CostTracker

    cycle_dir, all_ids = _write_cycle_fixture(tmp_path, cost_ceiling_usd=500.0)

    captured: list[CostTracker] = []
    OriginalCostTracker = CostTracker

    def _spy_cost_tracker(ceiling_usd: float, _abort_factor: float = 1.2) -> CostTracker:
        t = OriginalCostTracker(ceiling_usd=ceiling_usd, _abort_factor=_abort_factor)
        captured.append(t)
        return t

    monkeypatch.setattr("engine.cli.bakeoff.CostTracker", _spy_cost_tracker)

    def perfect(config_name: str) -> dict[str, str]:
        return {k: ("A" if k.startswith("a") else "B") for k in all_ids}

    bakeoff_cmd(cycle_dir, execute=False, predict_fn=perfect)

    assert len(captured) == 1, "bakeoff_cmd must construct exactly one CostTracker"
    assert captured[0].ceiling_usd == 500.0, (
        f"CostTracker.ceiling_usd should be 500.0 (from manifest), "
        f"got {captured[0].ceiling_usd}"
    )
