"""Integration test for run_bakeoff with a stub predict_fn (Plan 8e T7)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.classify.bakeoff import ModelConfig
from engine.cli.bakeoff import run_bakeoff


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
