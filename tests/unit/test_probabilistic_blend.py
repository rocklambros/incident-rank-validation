from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from engine.decide.blend import (
    EXPECTED_FRAME_BLIND,
    EXPECTED_ROLLUP,
    INCUMBENTS,
    load_inputs,
)

MANIFEST = Path("projects/owasp-llm/cycles/2026/blend/blend_manifest.json")


def test_load_inputs_shapes_and_labels() -> None:
    inp = load_inputs(MANIFEST)
    assert inp.lambda_samples.shape == (16000, 20)
    assert inp.vote_rank_samples.shape == (5000, 20)
    assert inp.entry_ids[:10] == INCUMBENTS
    assert inp.rollup == EXPECTED_ROLLUP
    assert inp.frame_blind == EXPECTED_FRAME_BLIND


def test_load_inputs_rejects_bad_hash(tmp_path: Path) -> None:
    m = json.loads(MANIFEST.read_text())
    m["inputs"]["taxonomy"]["sha256"] = "0" * 64
    bad = tmp_path / "m.json"
    bad.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="hash mismatch"):
        load_inputs(bad)


def test_load_inputs_rejects_frame_blind_drift(tmp_path: Path) -> None:
    # Copy inputs, tamper not_measurable, rehash, and repoint the manifest at tmp_path.
    src = json.loads(MANIFEST.read_text())
    root = Path.cwd()
    rb_path = Path(src["inputs"]["rankings_baselines"]["path"])
    rb = json.loads(rb_path.read_text())
    rb["not_measurable"] = ["LLM04", "LLM08"]  # drop LLM10
    bad_rb = tmp_path / "rb.json"
    bad_rb.write_text(json.dumps(rb))
    src["inputs"]["rankings_baselines"]["path"] = str(bad_rb)
    src["inputs"]["rankings_baselines"]["sha256"] = hashlib.sha256(bad_rb.read_bytes()).hexdigest()
    # keep other paths absolute so they still resolve from tmp manifest
    for k, spec in src["inputs"].items():
        if k != "rankings_baselines":
            spec["path"] = str((root / spec["path"]).resolve())
    bad_m = tmp_path / "m.json"
    bad_m.write_text(json.dumps(src))
    with pytest.raises(ValueError, match="frame-blind drift"):
        load_inputs(bad_m)
