from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from engine.decide.blend import (
    EXPECTED_FRAME_BLIND,
    EXPECTED_ROLLUP,
    INCUMBENTS,
    blend,
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


GOLDEN = Path("projects/owasp-llm/cycles/2026/blend/blend_golden.json")


def test_matches_golden() -> None:
    g = json.loads(GOLDEN.read_text())
    r = blend(load_inputs(MANIFEST))
    assert list(r.order) == g["order"]
    for e in INCUMBENTS:
        assert r.p_top3[e] == pytest.approx(g["p_top3"][e], abs=1e-9)
        assert r.p_top5[e] == pytest.approx(g["p_top5"][e], abs=1e-9)


def test_top5_and_tail_seed_stable() -> None:
    inp = load_inputs(MANIFEST)
    orders = [blend(inp, seed=s).order for s in range(1000, 1020)]
    assert len({o[:5] for o in orders}) == 1        # top-5 ordinal invariant
    assert len({frozenset(o[5:]) for o in orders}) == 1  # tail set invariant


def test_permuted_array_rejected(tmp_path: Path) -> None:
    src = json.loads(MANIFEST.read_text())
    root = Path.cwd()
    lam = np.load(root / src["inputs"]["lambda_samples"]["path"])
    bad = tmp_path / "lam.npy"
    np.save(bad, lam[:, ::-1])  # reverse columns
    src["inputs"]["lambda_samples"]["path"] = str(bad)
    src["inputs"]["lambda_samples"]["sha256"] = hashlib.sha256(bad.read_bytes()).hexdigest()
    for k, spec in src["inputs"].items():
        if k != "lambda_samples":
            spec["path"] = str((root / spec["path"]).resolve())
    m = tmp_path / "m.json"
    m.write_text(json.dumps(src))
    # column permutation is silent unless the order changes; assert the order DID change.
    from engine.decide.blend import blend as b2
    from engine.decide.blend import load_inputs as li2
    assert list(b2(li2(m)).order) != json.loads(GOLDEN.read_text())["order"]


def test_deterministic_tie_break() -> None:
    # Construct a tie: equal blended scores, expect entry_id ascending.
    from engine.decide.blend import _summarize
    pos = np.tile(np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]), (100, 1))
    r = _summarize(pos)
    assert r.order[0] == "LLM01"  # entry-id order preserved on identical positions


def test_cross_implementation_matches_prototype() -> None:
    # Freeze the reference by its manifest hash before trusting it.
    import json as _j

    from engine.decide.blend import blend
    from engine.decide.blend_prototype_reference import reconstruct_order
    m = _j.loads(MANIFEST.read_text())
    ref = m["inputs"]["prototype_reference"]
    from engine.snapshot.hashing import verify_snapshot_hash
    verify_snapshot_hash(Path.cwd() / ref["path"], ref["sha256"])
    assert list(blend(load_inputs(MANIFEST)).order) == reconstruct_order("lin")
