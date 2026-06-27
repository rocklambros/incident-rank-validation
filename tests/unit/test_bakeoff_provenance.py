"""Tests for bake-off provenance writer (Plan 8e T6)."""
from __future__ import annotations

import json
from pathlib import Path

from engine.classify.bakeoff import (
    BakeoffResult,
    ModelConfig,
    write_bakeoff_provenance,
)


def _result() -> BakeoffResult:
    return BakeoffResult(
        winner="good",
        floor_balanced_accuracy=0.5,
        config_balanced_accuracy={"good": 1.0},
        selection_classes=("A", "out-of-scope"),
        sparse_classes=(),
        lockbox_cell_sizes={"A": 6, "out-of-scope": 4},
        eligible_configs=("good",),
        alpha=0.05,
    )


def test_provenance_records_winner_shas_and_label_hash(tmp_path: Path) -> None:
    label_file = tmp_path / "labeled_incidents.json"
    label_file.write_text('[{"incident_id": "i1", "entry_id": "A"}]\n')
    configs = [
        ModelConfig("qwen3-235b", "Qwen/Qwen3-235B-A22B", "abc123", "NVIDIA H200", 4),
    ]
    out = write_bakeoff_provenance(tmp_path, _result(), configs, label_file)
    assert out == tmp_path / "classify_provenance.json"
    data = json.loads(out.read_text())
    assert data["winner"] == "good"
    assert data["label_file_sha256"]  # non-empty
    assert data["models"][0]["revision_sha"] == "abc123"
    assert data["floor_balanced_accuracy"] == 0.5


def test_label_hash_is_content_addressed(tmp_path: Path) -> None:
    import hashlib

    label_file = tmp_path / "labeled_incidents.json"
    content = '[{"incident_id": "i1", "entry_id": "A"}]\n'
    label_file.write_text(content)
    out = write_bakeoff_provenance(tmp_path, _result(), [], label_file)
    data = json.loads(out.read_text())
    assert data["label_file_sha256"] == hashlib.sha256(content.encode()).hexdigest()


def test_provenance_records_seed_and_fraction(tmp_path: Path) -> None:
    label_file = tmp_path / "labeled_incidents.json"
    label_file.write_text("[]\n")
    out = write_bakeoff_provenance(
        tmp_path, _result(), [], label_file, seed=7, lockbox_fraction=0.3
    )
    data = json.loads(out.read_text())
    assert data["seed"] == 7
    assert data["lockbox_fraction"] == 0.3


def test_goldset_provenance_records_hash_and_disagreement(tmp_path: Path) -> None:
    from engine.classify.bakeoff import goldset_provenance

    rows = [
        {"incident_id": "i1", "llm_consensus": "A", "adjudicated": "accept",
         "labels": ["A"], "blind_label": "A", "notes": None},
        {"incident_id": "i2", "llm_consensus": "A", "adjudicated": "override",
         "labels": ["B"], "blind_label": "B", "notes": None},
    ]
    gp = tmp_path / "gold.jsonl"
    gp.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    meta = goldset_provenance(gp)
    import hashlib
    assert meta["sha256"] == hashlib.sha256(gp.read_bytes()).hexdigest()
    assert meta["n_records"] == 2
    # i2 has blind_label "B" != llm_consensus "A" -> 1/2 disagreement
    assert meta["blind_consensus_disagreement_rate"] == 0.5
    assert meta["adjudicated_counts"] == {"accept": 1, "override": 1}
    assert meta["adjudicator"] == "single-author"


def test_provenance_records_min_cell_and_goldset_block(tmp_path: Path) -> None:
    label_file = tmp_path / "labeled_incidents.json"
    label_file.write_text("[]\n")
    out = write_bakeoff_provenance(
        tmp_path, _result(), [], label_file,
        seed=7, lockbox_fraction=0.3, min_cell=5,
        goldset_meta={"sha256": "abc", "n_records": 2},
    )
    data = json.loads(out.read_text())
    assert data["min_cell"] == 5
    assert data["goldset"]["sha256"] == "abc"
