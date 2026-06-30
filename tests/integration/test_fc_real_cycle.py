"""F-C integration tests: real-snapshot minimal-cycle fixture.

The fixture is tmp-only; it never reads from or writes under
``projects/owasp-llm/cycles/2026/``.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest


@pytest.mark.integration
def test_builder_happy_path(
    real_minimal_cycle: Callable[..., Path], tmp_path: Path
) -> None:
    from engine.calibrate.coverage import (
        COVERAGE_FILENAME,
        _resolve_snapshot_incidents,
        verify_labeled_completeness,
    )
    from engine.snapshot.hashing import snapshot_hash

    cycle = real_minimal_cycle(tmp_path)
    assert tmp_path in cycle.parents or cycle == tmp_path
    manifest = json.loads((cycle / "prereg" / "manifest.json").read_text())
    snap_dirs = list((cycle / "corpora" / "genai_agentic").iterdir())
    assert len(snap_dirs) == 1
    H = snap_dirs[0].name
    assert manifest["snapshot_hash"] == H == snapshot_hash(snap_dirs[0] / "incidents.json")
    assert manifest["overlap_min_fp"] == 1
    assert _resolve_snapshot_incidents(cycle, H) is not None  # guard is LIVE
    marker = json.loads((cycle / "classify" / COVERAGE_FILENAME).read_text())
    assert (marker["n_corpus"], marker["n_in_scope"], marker["n_oos"]) == (6, 4, 2)
    labeled = json.loads((cycle / "classify" / "labeled_incidents.json").read_text())
    labeled_ids = {str(r["incident_id"]) for r in labeled}
    verify_labeled_completeness(cycle, H, labeled_ids)  # must NOT raise


# ── T1: adapter drops the future-dated record (universe-drift self-guard) ──────


@pytest.mark.integration
def test_adapter_drops_future_dated_record(
    real_minimal_cycle: Callable[..., Path], tmp_path: Path
) -> None:
    """INC-FUTURE (dated 2026-05-21) must be absent from adapter output but
    present in the raw snapshot, and the OOS set must be exactly
    {INC-OOS, INC-FUTURE} — not merely a count of 2."""
    from engine.adapters.genai_agentic import GenAIAgenticAdapter
    from engine.calibrate.coverage import read_snapshot_universe_ids

    cycle = real_minimal_cycle(tmp_path)

    snap_dirs = list((cycle / "corpora" / "genai_agentic").iterdir())
    assert len(snap_dirs) == 1
    snap_dir = snap_dirs[0]
    incidents_json = snap_dir / "incidents.json"

    adapter = GenAIAgenticAdapter(snap_dir, snapshot_date="2026-05-20")
    adapter_ids = {i.id for i in adapter.iter_incidents()}

    # Adapter must silently drop INC-FUTURE (date "2026-05-21" > pull_date "2026-05-20").
    assert "INC-FUTURE" not in adapter_ids
    # Adapter must yield INC-OOS (date "2025-12-05" ≤ pull_date; OOS = no labels, not filtered).
    assert "INC-OOS" in adapter_ids

    # The raw snapshot must still contain INC-FUTURE (unfiltered universe).
    universe_ids = read_snapshot_universe_ids(incidents_json)
    assert "INC-FUTURE" in universe_ids

    # The specific OOS set: universe minus labeled — must be exactly these two.
    labeled_data = json.loads((cycle / "classify" / "labeled_incidents.json").read_text())
    labeled_ids = {str(r["incident_id"]) for r in labeled_data}
    assert universe_ids - labeled_ids == {"INC-OOS", "INC-FUTURE"}


# ── T2: recall-flip reflects the classifier, not consensus ─────────────────────


@pytest.mark.integration
def test_recall_flip_reflects_classifier_not_consensus(
    real_minimal_cycle: Callable[..., Path], tmp_path: Path
) -> None:
    """INC-FLIP: classifier=LLM01 but truth=LLM02 (goldset llm_consensus=LLM02).

    WITH classifier labels: INC-FLIP is a FN for LLM02 (classifier missed it).
    WITHOUT classifier (consensus path): INC-FLIP is a TP for LLM02.
    The FN delta of exactly 1 is attributable to INC-FLIP.
    """
    from engine.calibrate.gold_loader import load_classifier_labels, load_gold_calibration
    from engine.calibrate.tally import RecallTally, TallyResult, calibrate_with_gold

    cycle = real_minimal_cycle(tmp_path)

    manifest_data = json.loads((cycle / "prereg" / "manifest.json").read_text())
    rubric_hash = str(manifest_data["rubric_hash"])

    labeled_path = cycle / "classify" / "labeled_incidents.json"
    gold_dir = cycle / "calibration"
    valid_entry_ids = {"LLM01", "LLM02"}

    # Empty base tally — gold data is the sole signal in this test.
    base_tally = TallyResult(
        precision_counts={},
        recall_counts={},
        rollup_counts={},
        total_coded=0,
        amendments_applied=0,
    )

    # ── (a) WITH classifier labels ─────────────────────────────────────────────
    # Classifier: INC-FLIP → LLM01; truth: LLM02 → FN for LLM02.
    # Classifier: INC-02   → LLM02; truth: LLM02 → TP for LLM02.
    classifier_labels = load_classifier_labels(labeled_path)
    gold_with = load_gold_calibration(
        gold_dir=gold_dir,
        valid_entry_ids=valid_entry_ids,
        rubric_hash=rubric_hash,
        adjudicator_id="test",
        classifier_labels=classifier_labels,
    )
    tally_with = calibrate_with_gold(
        base_tally,
        gold_with,
        set(),           # base_incident_ids: none pre-coded in a batch
        valid_entry_ids,
        merge_stratum="security",
    )

    cell_with = tally_with.recall_counts[("LLM02", "security")]
    assert cell_with == RecallTally(
        true_positives=1,
        false_negatives=1,
        total_in_sample=2,
    ), f"WITH classifier: expected TP=1,FN=1,total=2; got {cell_with}"

    # ── (b) WITHOUT classifier (consensus path) ────────────────────────────────
    # Both goldset incidents carry llm_consensus=LLM02, so both are TPs for LLM02.
    gold_without = load_gold_calibration(
        gold_dir=gold_dir,
        valid_entry_ids=valid_entry_ids,
        rubric_hash=rubric_hash,
        adjudicator_id="test",
        classifier_labels=None,
    )
    tally_without = calibrate_with_gold(
        base_tally,
        gold_without,
        set(),
        valid_entry_ids,
        merge_stratum="security",
    )

    cell_without = tally_without.recall_counts[("LLM02", "security")]
    assert cell_without == RecallTally(
        true_positives=2,
        false_negatives=0,
        total_in_sample=2,
    ), f"WITHOUT classifier: expected TP=2,FN=0,total=2; got {cell_without}"

    # Delta assertion: the single FN is attributable to INC-FLIP.
    assert cell_with.false_negatives - cell_without.false_negatives == 1


# ── T3: overlap W non-empty end-to-end (W routed to the model call site) ───────


@pytest.mark.integration
def test_overlap_w_non_empty_routes_to_model_call(
    real_minimal_cycle: Callable[..., Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """W is non-empty from the goldset confusion, and reaches run_inference.

    INC-FLIP: classifier=LLM01, truth=LLM02 → FP for LLM01 leaked to LLM02.
    With overlap_min_fp=1 that single FP is enough to form a W column.

    Spy is placed on the SOURCE module (engine.model.inference.run_inference)
    because execute_infer_phase imports it via a local 'from ... import ...' at
    call time — patching the source dict is the only seam that works.
    """
    from engine.calibrate.confusion import build_overlap_from_confusion
    from engine.calibrate.gold_loader import load_classifier_labels, load_gold_calibration
    from engine.cli.pipeline_executor import execute_infer_phase
    from engine.model.inference import InferenceResult
    from engine.model.overlap import OverlapWeights

    cycle = real_minimal_cycle(tmp_path)

    manifest_data = json.loads((cycle / "prereg" / "manifest.json").read_text())
    rubric_hash = str(manifest_data["rubric_hash"])
    overlap_min_fp: int = int(manifest_data["overlap_min_fp"])  # 1

    labeled_path = cycle / "classify" / "labeled_incidents.json"
    gold_dir = cycle / "calibration"
    valid_entry_ids = {"LLM01", "LLM02"}

    classifier_labels = load_classifier_labels(labeled_path)
    gold = load_gold_calibration(
        gold_dir=gold_dir,
        valid_entry_ids=valid_entry_ids,
        rubric_hash=rubric_hash,
        adjudicator_id="test",
        classifier_labels=classifier_labels,
    )

    # Step 1: W must be non-empty from the goldset confusion.
    W = build_overlap_from_confusion(
        gold,
        ("LLM01", "LLM02"),
        min_fp_count=overlap_min_fp,
    )
    assert W.weights, (
        f"W is empty with overlap_min_fp={overlap_min_fp}; "
        "expected LLM01-claimed/LLM02-true FP from INC-FLIP to populate a column"
    )

    # Step 2: spy on the SOURCE module so the local import inside
    # execute_infer_phase picks up the replacement at call time.
    spy_invocations: list[bool] = []   # unconditional — proves spy was reached
    captured_overlap: list[OverlapWeights] = []

    def _spy(**kwargs: Any) -> InferenceResult:
        spy_invocations.append(True)  # fires on every call, overlap or not
        ov = kwargs.get("overlap")
        if ov is not None:
            captured_overlap.append(ov)
        return InferenceResult(
            entry_ids=("LLM01", "LLM02"),
            lambda_samples=np.ones((200, 2), dtype=np.float64),
            r_hat={"LLM01": 1.0, "LLM02": 1.0},
            ess={"LLM01": 200.0, "LLM02": 200.0},
            divergences=0,
            num_warmup=0,
            num_samples=1,
        )

    monkeypatch.setattr("engine.model.inference.run_inference", _spy)

    execute_infer_phase(cycle, num_warmup=0, num_samples=1)

    # Explicit call-flag: distinguishes "spy reached" from "overlap was non-None".
    assert spy_invocations, "spy was never called — run_inference was not reached"
    assert captured_overlap, (
        "run_inference received no overlap kwarg; W was not routed to the call site"
    )
    assert captured_overlap[0].weights, (
        "run_inference received an EMPTY overlap; W was not routed to the call site"
    )


# ── T4a: pre-flight — batch chain is clean; cal_tally exits 0 ──────────────────


@pytest.mark.integration
def test_cal_tally_batch_chain_valid(
    real_minimal_cycle: Callable[..., Path], tmp_path: Path
) -> None:
    """Pre-flight: with_batches=True cycle passes cal_tally (exit 0).

    Guards that the batch chain itself is sound — any non-zero exit here
    means the builder is broken, not the completeness guard.
    """
    from click.testing import CliRunner

    from engine.cli.calibration import cal_tally

    cycle = real_minimal_cycle(tmp_path, with_batches=True)
    manifest_lock = cycle / "prereg" / "manifest.lock"
    rubric = cycle / "prereg" / "rubric.json"
    cal_dir = cycle / "calibration"

    runner = CliRunner()
    result = runner.invoke(
        cal_tally,
        [
            "--cycle", str(cycle),
            "--manifest", str(manifest_lock),
            "--rubric", str(rubric),
            "--gold-calibration", str(cal_dir),
        ],
        catch_exceptions=True,
    )
    assert result.exit_code == 0, (
        f"cal_tally exited {result.exit_code} — builder bug, not a guard.\n"
        f"output:\n{result.output}"
        + (f"\nexception: {result.exception!r}" if result.exception else "")
    )


# ── T4b: raise test — truncated labeled_incidents triggers completeness check 5 ─


@pytest.mark.integration
def test_cal_tally_raises_on_truncated_labeled(
    real_minimal_cycle: Callable[..., Path], tmp_path: Path
) -> None:
    """Raise test: truncate_labeled=True → LabeledIncidentsIncompleteError (check 5).

    The coverage marker records n_in_scope=4 (unchanged), but labeled_incidents
    carries only 3 rows (INC-03 dropped).  verify_labeled_completeness check 5
    fires: n_in_scope=4 != labeled count=3, message contains 'does not reconcile'.

    The 'does not reconcile' substring is UNIQUE to check 5 — it distinguishes
    this from a batch-validation ValueError (check 1–4, no substring) and from
    check 4's 'pinned snapshot size' message.
    """
    from click.testing import CliRunner

    from engine.cli.calibration import cal_tally

    cycle = real_minimal_cycle(tmp_path, truncate_labeled=True, with_batches=True)
    manifest_lock = cycle / "prereg" / "manifest.lock"
    rubric = cycle / "prereg" / "rubric.json"
    cal_dir = cycle / "calibration"

    runner = CliRunner()
    result = runner.invoke(
        cal_tally,
        [
            "--cycle", str(cycle),
            "--manifest", str(manifest_lock),
            "--rubric", str(rubric),
            "--gold-calibration", str(cal_dir),
        ],
        catch_exceptions=True,
    )
    assert result.exit_code != 0, (
        "Expected non-zero exit from cal_tally on truncated labeled_incidents; "
        f"got exit_code=0. Output:\n{result.output}"
    )
    assert result.exception is not None, (
        "result.exit_code != 0 but result.exception is None — "
        "a CLI usage error or SystemExit, not the completeness guard."
    )
    assert type(result.exception).__name__ == "LabeledIncidentsIncompleteError", (
        f"Expected LabeledIncidentsIncompleteError, "
        f"got {type(result.exception).__name__}: {result.exception!r}"
    )
    assert "does not reconcile" in str(result.exception), (
        f"Expected 'does not reconcile' in exception message (check 5 unique substring); "
        f"got: {result.exception!s}"
    )


# ── T5: infer goldset-guard RAISES when a scored recall incident is absent ───────


@pytest.mark.integration
def test_infer_goldset_snapshot_provenance_guard_raises(
    real_minimal_cycle: Callable[..., Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """execute_infer_phase raises LabeledIncidentsIncompleteError (check #6).

    INC-GHOST is in adjudicated_goldset.jsonl (labels=["LLM01"], adjudicated=accept)
    but is absent from the corpus snapshot AND from labeled_incidents.json.

    Why check #6 fires (not check #1 or a RuntimeError-rewrap):
    - INC-GHOST is NOT in labeled_incidents.json → check #1 (labeled⊆snapshot) does
      not see it; only check #6 (goldset⊆snapshot) can fire on it.
    - ghost labels=["LLM01"] ∈ valid_entry_ids → no rubric ValueError in gold_loader;
      classifier_labels.get("INC-GHOST", OUT_OF_SCOPE) returns the OOS sentinel
      (non-None) → classifier_entry_id is not None → INC-GHOST is included in
      goldset_recall_ids passed to verify_labeled_completeness.
    - LabeledIncidentsIncompleteError(RuntimeError) is NOT caught by the gold-loader
      try/except (ValueError, OSError, JSONDecodeError) block, so it propagates
      uncaught and the match="goldset" substring is unique to check #6's message.

    The spy is placed BEFORE the call.  run_inference lives at line ~341, after the
    goldset guard at line ~309 — the guard fires first and the spy is never reached.
    """
    from engine.calibrate.coverage import LabeledIncidentsIncompleteError
    from engine.cli.pipeline_executor import execute_infer_phase

    cycle = real_minimal_cycle(tmp_path, ghost_recall_id="INC-GHOST")

    spy_called: list[bool] = []

    def _spy(**kwargs: Any) -> Any:  # pragma: no cover — must never be reached
        spy_called.append(True)
        raise AssertionError(
            "run_inference reached — provenance guard did NOT fire before inference"
        )

    monkeypatch.setattr("engine.model.inference.run_inference", _spy)

    with pytest.raises(LabeledIncidentsIncompleteError, match="goldset"):
        execute_infer_phase(cycle, num_warmup=0, num_samples=1)

    assert not spy_called, (
        "run_inference was called before the provenance guard raised — "
        "check #6 did not fire before inference"
    )


# ── T6: decide-real → run_oracle, 3 deliverables, none erroring (8d-I3) ─────────


@pytest.mark.integration
def test_decide_real_runs_oracle_writes_3_deliverables(
    real_minimal_cycle: Callable[..., Path], tmp_path: Path
) -> None:
    """decide-real writes oracle_report.json with 3 deliverables (8d-I3 contract).

    Assertion split:
    - Pre-direct-call: verify decide's OWN oracle write.  decide wraps run_oracle
      in a crash-suppressing try/except, so the file's existence is the only
      signal that decide's invocation succeeded.  This assertion MUST come before
      the direct run_oracle call, which overwrites the file.
    - Post: call run_oracle directly to surface oracle-internal errors that the
      decide wrapper would silently swallow.  Assert the in-memory verdict.

    Deliverable expectations with this fixture:
    - D1 incidence  → PASS (LLM01 clear winner, kendall_tau=1.0 ≥ 0.95)
    - D2 plackett_luce → PASS (unanimous respondents, tau=1.0 ≥ 0.70)
    - D3 sigma_u    → SKIP (robustness_specs=() → engine_sigma=None)
    """
    from click.testing import CliRunner

    from engine.cli.pipeline import decide_real
    from engine.verify.check import run_oracle

    cycle = real_minimal_cycle(tmp_path, with_infer=True, with_vote=True)
    vote_xlsx = cycle / "vote" / "vote.xlsx"

    runner = CliRunner()
    result = runner.invoke(
        decide_real,
        [
            "--execute",
            "--cycle", str(cycle),
            "--vote-xlsx", str(vote_xlsx),
        ],
        catch_exceptions=True,
    )
    assert result.exit_code == 0, (
        f"decide-real exited {result.exit_code}\n"
        f"output:\n{result.output}"
        + (f"\nexception: {result.exception!r}" if result.exception else "")
    )

    # Assert decide's OWN oracle write BEFORE the direct call overwrites the file.
    oracle_path = cycle / "results" / "oracle_report.json"
    assert oracle_path.exists(), (
        "oracle_report.json not written by decide-real\n"
        f"decide output:\n{result.output}"
    )
    decide_oracle_doc = json.loads(oracle_path.read_text())
    assert len(decide_oracle_doc["deliverables"]) == 3, (
        f"Expected 3 deliverables in decide's oracle_report.json, "
        f"got {len(decide_oracle_doc['deliverables'])}: {decide_oracle_doc['deliverables']}"
    )

    # Byte-immutability guard: cycle must live under tmp_path.
    assert tmp_path in cycle.parents

    # Direct run_oracle call — surfaces oracle-internal errors the decide wrapper swallows.
    verdict = run_oracle(cycle)

    assert len(verdict.deliverables) == 3, (
        f"Expected 3 deliverables from run_oracle, got {len(verdict.deliverables)}"
    )
    names = {d.name for d in verdict.deliverables}
    assert names == {"incidence", "plackett_luce", "sigma_u"}, (
        f"Unexpected deliverable names: {names}"
    )
    for d in verdict.deliverables:
        assert d.status in {"PASS", "SKIP"}, (
            f"Deliverable {d.name!r} status={d.status!r}; expected PASS or SKIP\n"
            f"metric={d.metric!r} detail={d.detail!r}"
        )
    assert verdict.provisional is False, (
        "verdict.provisional=True — at least one deliverable FAILed "
        "(fixture-alignment bug):\n"
        + "\n".join(
            f"  {d.name}: {d.status} | {d.metric} | {d.detail}"
            for d in verdict.deliverables
        )
    )
