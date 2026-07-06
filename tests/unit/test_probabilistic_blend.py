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


def test_blend_reproduces_approved_order() -> None:
    from engine.decide.blend import blend
    r = blend(load_inputs(MANIFEST))
    assert r.order == (
        "LLM01", "LLM02", "LLM06", "LLM03", "LLM04",
        "LLM10", "LLM09", "LLM07", "LLM08", "LLM05",
    )
    assert r.tiers["pair"] == ("LLM01", "LLM02")
    assert r.tiers["band"] == ("LLM06", "LLM03", "LLM04")
    assert r.tiers["tail"] == ("LLM10", "LLM09", "LLM07", "LLM08", "LLM05")


def test_blend_uncertainty_numbers() -> None:
    from engine.decide.blend import blend
    r = blend(load_inputs(MANIFEST))
    assert r.p_top3["LLM01"] == pytest.approx(0.99, abs=0.02)
    assert r.p_top3["LLM02"] == pytest.approx(0.95, abs=0.03)
    assert r.p_top5["LLM04"] == pytest.approx(0.76, abs=0.03)
    assert r.mean_position["LLM03"] < r.mean_position["LLM04"]
    assert r.p_top5["LLM10"] == pytest.approx(0.33, abs=0.05)  # borderline tail
    for e in ("LLM09", "LLM07", "LLM08", "LLM05"):
        assert r.p_top5[e] < 0.05  # deep blur


def test_fold_sum_prevalence_and_min_rank() -> None:
    # Exercise the real _fold: LLM01 folds in ROLL-CMSB (sum on data, min on vote).
    from engine.decide.blend import _fold
    inp = load_inputs(MANIFEST)
    idx = {e: i for i, e in enumerate(inp.entry_ids)}
    lam_arr, vote_arr = _fold(inp, inp.lambda_samples, inp.vote_rank_samples)
    assert lam_arr[0, 0] == pytest.approx(
        inp.lambda_samples[0, idx["LLM01"]] + inp.lambda_samples[0, idx["ROLL-CMSB"]])
    assert vote_arr[0, 0] == min(
        inp.vote_rank_samples[0, idx["LLM01"]], inp.vote_rank_samples[0, idx["ROLL-CMSB"]])


def test_zpopulation_alternative_bounded() -> None:
    # Order-neutral, P(top-k) shift under measurable-only z is small (< 0.05).
    from engine.decide.blend import _blend_measurable_z, blend
    r10 = blend(load_inputs(MANIFEST))
    r7 = _blend_measurable_z(load_inputs(MANIFEST))
    assert r10.order == r7.order
    assert max(abs(r10.p_top5[e] - r7.p_top5[e]) for e in INCUMBENTS) < 0.05
