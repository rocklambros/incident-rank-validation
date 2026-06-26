"""Tests for bake-off truth loading + OOS-inclusive balanced accuracy (Plan 8e T1)."""
from __future__ import annotations

import json
from pathlib import Path

from engine.classify.bakeoff import (
    OOS_CLASS,
    balanced_accuracy_oos,
    load_bakeoff_truth,
    per_class_recall,
    sparse_classes,
    truth_cell_sizes,
)


def _write_goldset(tmp: Path) -> Path:
    rows = [
        {"incident_id": "i1", "llm_consensus": "LLM01", "adjudicated": "accept",
         "labels": ["LLM01"], "blind_label": "LLM01", "notes": None},
        {"incident_id": "i2", "llm_consensus": "LLM01", "adjudicated": "accept",
         "labels": ["LLM01", "LLM02"], "blind_label": "LLM01", "notes": None},
        {"incident_id": "i3", "llm_consensus": "out-of-scope", "adjudicated": "accept",
         "labels": [], "blind_label": "out-of-scope", "notes": None},
    ]
    p = tmp / "gold.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


def test_load_truth_handles_multilabel_and_oos(tmp_path: Path) -> None:
    truth = load_bakeoff_truth(_write_goldset(tmp_path))
    assert truth["i1"] == frozenset({"LLM01"})
    assert truth["i2"] == frozenset({"LLM01", "LLM02"})
    assert truth["i3"] == frozenset({OOS_CLASS})


def test_truth_cell_sizes(tmp_path: Path) -> None:
    truth = load_bakeoff_truth(_write_goldset(tmp_path))
    sizes = truth_cell_sizes(truth)
    assert sizes["LLM01"] == 2  # i1, i2
    assert sizes["LLM02"] == 1  # i2
    assert sizes[OOS_CLASS] == 1  # i3


def test_sparse_classes(tmp_path: Path) -> None:
    truth = load_bakeoff_truth(_write_goldset(tmp_path))
    # min_n=2: LLM02 (1) and out-of-scope (1) are sparse; LLM01 (2) is not.
    assert sparse_classes(truth, min_n=2) == frozenset({"LLM02", OOS_CLASS})


def test_per_class_recall_multilabel_semantics() -> None:
    truth = {"i1": frozenset({"A"}), "i2": frozenset({"A", "B"}), "i3": frozenset({"B"})}
    # pred i2=A: hit for A, miss for B.
    predictions = {"i1": "A", "i2": "A", "i3": "B"}
    rec = per_class_recall(predictions, truth, ["A", "B"])
    assert rec["A"] == 1.0  # both A-truth incidents predicted A
    assert rec["B"] == 0.5  # i2 missed, i3 hit


def test_balanced_accuracy_oos_includes_oos() -> None:
    truth = {"i1": frozenset({"A"}), "i2": frozenset({OOS_CLASS})}
    # A model that never predicts OOS scores 0 on the OOS class -> drags the mean.
    predictions = {"i1": "A", "i2": "A"}
    ba = balanced_accuracy_oos(predictions, truth, ["A", OOS_CLASS])
    assert ba == 0.5  # recall A=1.0, recall OOS=0.0 -> mean 0.5
