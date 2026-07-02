# F1 — Winner→Recall-Calibration Wiring (capability + contract proof) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Checkbox steps. Two small tasks on one file pair; one implementer with per-task commits, then a review.

**Goal:** Close premortem finding **F1** at the capability level: let recall calibration measure the classifier whose labels build the incidence counts (the bake-off winner in Phase 3) instead of the goldset's stored `llm_consensus`, and prove end-to-end that the winner's labels actually drive the recall posteriors. **Additive + backward-compatible — no live CLI behavior change** (that flip is a deliberate reassessment decision).

**Background:** `_load_recall_from_adjudicated` (`engine/calibrate/gold_loader.py:106`) hardcodes `classifier_entry_id = llm_consensus` (the old 3-model consensus). But incidence counts come from `labeled_incidents.json` (`_build_counts_from_labeled`). So today recall-corrects classifier-X's counts using classifier-Y's recall, AND the bake-off winner can never reach calibration. This plan adds an optional `classifier_labels` channel; `classifier_labels=None` preserves today's exact behavior.

**Scope (in):** `load_classifier_labels` helper; optional `classifier_labels` param on `_load_recall_from_adjudicated` + `load_gold_calibration`; coverage + vocab guards on that channel; unit tests + an end-to-end contract test. **Scope (out, → reassessment/Phase 3):** flipping the two CLI call sites (`pipeline_executor.py:294`, `calibration.py:401`) to pass `classifier_labels` live (changes current numbers — oracle corrections); the OOS-labeled-incident semantic (an incident the classifier put out-of-scope, hence absent from `labeled_incidents.json`); binding the scored-classifier identity into provenance/lock (F3b).

## Global Constraints
- NO new dependencies. mypy strict + ruff clean. Run `uv run ruff check . && uv run mypy engine tests` before each commit; FULL `uv run pytest -q` before push.
- **Backward compatibility is mandatory:** `classifier_labels=None` (the default, what every existing caller passes) must produce byte-identical behavior to today (`classifier_entry_id = llm_consensus`). The existing `tests/unit/test_gold_loader.py` must pass unchanged.
- No AI attribution in commits. Branch `plan7/engine-upgrade-recall-pl` (PR #22).

---

### Task 1: `load_classifier_labels` + optional `classifier_labels` channel in the gold loader

**Files:** `engine/calibrate/gold_loader.py` ; **Test:** `tests/unit/test_gold_f1_winner_recall.py`

**Interfaces produced:**
- `def load_classifier_labels(labeled_incidents_path: Path) -> dict[str, str]` — `{incident_id: entry_id}` from `labeled_incidents.json` (a list of `{"incident_id","entry_id",...}`).
- `_load_recall_from_adjudicated(path, valid_entry_ids, classifier_labels=None)` and `load_gold_calibration(..., classifier_labels=None)` — when `classifier_labels` is provided, `classifier_entry_id` (and the precision claim) come from `classifier_labels[incident_id]`; coverage guard raises if a scored (non-`uncertain`) goldset incident is absent; vocab guard raises if a classifier label is not in `valid_entry_ids`. When `None`, behavior is today's `llm_consensus`.

- [ ] **Step 1: failing tests** — create `tests/unit/test_gold_f1_winner_recall.py`:
```python
"""F1: recall calibration scores the classifier's labels, not the old consensus."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.calibrate.gold_loader import (
    load_classifier_labels,
    load_gold_calibration,
)


def _write_adjudicated(tmp: Path) -> Path:
    gold_dir = tmp / "calibration"
    gold_dir.mkdir(parents=True)
    rows = [
        {"incident_id": "INC-1", "llm_consensus": "A", "adjudicated": "accept",
         "labels": ["A"], "blind_label": "A", "notes": None},
        {"incident_id": "INC-2", "llm_consensus": "A", "adjudicated": "override",
         "labels": ["B"], "blind_label": "B", "notes": None},
    ]
    (gold_dir / "adjudicated_goldset.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n"
    )
    return gold_dir


def test_load_classifier_labels_maps_incident_to_entry(tmp_path: Path) -> None:
    p = tmp_path / "labeled_incidents.json"
    p.write_text(json.dumps([
        {"incident_id": "INC-1", "entry_id": "A", "stage": 2},
        {"incident_id": "INC-2", "entry_id": "B", "stage": 1},
    ]))
    assert load_classifier_labels(p) == {"INC-1": "A", "INC-2": "B"}


def test_classifier_labels_drive_classifier_entry_id(tmp_path: Path) -> None:
    gold_dir = _write_adjudicated(tmp_path)
    # Winner DISAGREES with consensus on INC-1 (consensus A, winner B).
    winner = {"INC-1": "B", "INC-2": "B"}
    gold = load_gold_calibration(
        gold_dir=gold_dir, valid_entry_ids={"A", "B"},
        rubric_hash="r", adjudicator_id="t", classifier_labels=winner,
    )
    by_id = {r.incident_id: r for r in gold.recall_labels}
    assert by_id["INC-1"].classifier_entry_id == "B"  # winner, not consensus "A"
    assert by_id["INC-2"].classifier_entry_id == "B"


def test_default_is_backward_compatible_consensus(tmp_path: Path) -> None:
    gold_dir = _write_adjudicated(tmp_path)
    gold = load_gold_calibration(
        gold_dir=gold_dir, valid_entry_ids={"A", "B"},
        rubric_hash="r", adjudicator_id="t",  # no classifier_labels
    )
    by_id = {r.incident_id: r for r in gold.recall_labels}
    assert by_id["INC-1"].classifier_entry_id == "A"  # consensus preserved
    assert by_id["INC-2"].classifier_entry_id == "A"


def test_coverage_guard_raises_on_missing_incident(tmp_path: Path) -> None:
    gold_dir = _write_adjudicated(tmp_path)
    with pytest.raises(ValueError, match="coverage guard"):
        load_gold_calibration(
            gold_dir=gold_dir, valid_entry_ids={"A", "B"},
            rubric_hash="r", adjudicator_id="t",
            classifier_labels={"INC-1": "A"},  # INC-2 missing
        )


def test_vocab_guard_raises_on_unknown_classifier_label(tmp_path: Path) -> None:
    gold_dir = _write_adjudicated(tmp_path)
    with pytest.raises(ValueError, match="not in rubric"):
        load_gold_calibration(
            gold_dir=gold_dir, valid_entry_ids={"A", "B"},
            rubric_hash="r", adjudicator_id="t",
            classifier_labels={"INC-1": "A", "INC-2": "B-TYPO"},
        )
```

- [ ] **Step 2: run** `uv run pytest tests/unit/test_gold_f1_winner_recall.py -v` → FAIL (`load_classifier_labels` not importable).

- [ ] **Step 3a: add `load_classifier_labels`** — `engine/calibrate/gold_loader.py` (after `parse_entry_id_from_prefix`; uses already-imported `json`, `Path`):
```python
def load_classifier_labels(labeled_incidents_path: Path) -> dict[str, str]:
    """Map incident_id -> the classifier's assigned entry_id (Plan 8e F1).

    Source of the labels that BUILD the incidence counts; recall must be
    measured for THIS classifier, not the goldset's stored llm_consensus.  In
    Phase 3 this file is the bake-off winner's labeled_incidents.json.
    """
    data = json.loads(labeled_incidents_path.read_text(encoding="utf-8"))
    return {str(rec["incident_id"]): str(rec["entry_id"]) for rec in data}
```

- [ ] **Step 3b: thread `classifier_labels` through `_load_recall_from_adjudicated`** — replace the whole function body's per-record block. The new function:
```python
def _load_recall_from_adjudicated(
    path: Path,
    valid_entry_ids: set[str],
    classifier_labels: dict[str, str] | None = None,
) -> tuple[list[GoldRecallLabel], list[GoldPrecisionLabel]]:
    recall: list[GoldRecallLabel] = []
    precision: list[GoldPrecisionLabel] = []

    for line in path.read_text(encoding="utf-8").strip().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        incident_id = record["incident_id"]
        labels = record.get("labels", [])
        adjudicated = record.get("adjudicated", "")

        if adjudicated == "uncertain":
            continue

        for eid in labels:
            if eid not in valid_entry_ids:
                raise ValueError(
                    f"Entry ID '{eid}' from adjudicated incident "
                    f"'{incident_id}' not in rubric."
                )

        if classifier_labels is None:
            # Backward-compatible: score the goldset's stored 3-model consensus.
            predicted = record.get("llm_consensus")
        else:
            # Plan 8e F1: score the classifier whose labels build the incidence
            # counts (the bake-off winner in Phase 3).
            if incident_id not in classifier_labels:
                raise ValueError(
                    f"adjudicated incident '{incident_id}' absent from "
                    f"classifier_labels (F1 coverage guard): every scored "
                    f"goldset incident must have a classifier label."
                )
            predicted = classifier_labels[incident_id]
            if predicted not in valid_entry_ids:
                raise ValueError(
                    f"classifier label '{predicted}' for incident "
                    f"'{incident_id}' not in rubric."
                )

        recall.append(GoldRecallLabel(
            incident_id=incident_id,
            true_entry_ids=labels if labels else [],
            classifier_entry_id=predicted,
            source="llm-adjudicated",
        ))

        if predicted and labels:
            precision.append(GoldPrecisionLabel(
                incident_id=incident_id,
                claimed_entry_id=predicted,
                is_correct=(predicted in labels),
                source="llm-adjudicated",
            ))

    return recall, precision
```

- [ ] **Step 3c: thread `classifier_labels` through `load_gold_calibration`** — add `classifier_labels: dict[str, str] | None = None` to its signature (after `session_count`), and pass it at the call site:
```python
    if adjudicated_path is not None:
        adj_recall, adj_precision = _load_recall_from_adjudicated(
            adjudicated_path, valid_entry_ids, classifier_labels,
        )
```

- [ ] **Step 4: run** `uv run pytest tests/unit/test_gold_f1_winner_recall.py tests/unit/test_gold_loader.py -v` → PASS (new F1 tests + existing loader tests both green — backward compat holds).

- [ ] **Step 5: gate + commit**
```bash
uv run ruff check . && uv run mypy engine tests
git add engine/calibrate/gold_loader.py tests/unit/test_gold_f1_winner_recall.py
git commit -m "feat(calibrate): optional classifier_labels channel for recall (Plan 8e F1)"
```

---

### Task 2: End-to-end contract proof — the winner drives recall, not the consensus

**File:** `tests/unit/test_gold_f1_winner_recall.py` (append) — uses the real `load_gold_calibration → calibrate_with_gold` chain.

This is the decisive F1 proof: an incident where the winner DISAGREES with the consensus produces a recall **miss** under the winner channel where the consensus channel would have scored a **hit**.

- [ ] **Step 1: failing test** — append:
```python
def test_winner_recall_differs_from_consensus_end_to_end(tmp_path: Path) -> None:
    from engine.calibrate.tally import TallyResult, calibrate_with_gold

    gold_dir = _write_adjudicated(tmp_path)
    base = TallyResult(
        precision_counts={}, recall_counts={}, rollup_counts={},
        total_coded=0, amendments_applied=0,
    )

    # Consensus channel: INC-1 consensus "A" == truth "A" -> recall HIT for A.
    gold_consensus = load_gold_calibration(
        gold_dir=gold_dir, valid_entry_ids={"A", "B"},
        rubric_hash="r", adjudicator_id="t",
    )
    tally_c = calibrate_with_gold(base, gold_consensus, set(), {"A", "B"})
    rc_a = tally_c.recall_counts[("A", "security")]
    assert rc_a.true_positives == 1 and rc_a.false_negatives == 0

    # Winner channel: INC-1 winner "B" != truth "A" -> recall MISS for A.
    winner = {"INC-1": "B", "INC-2": "B"}
    gold_winner = load_gold_calibration(
        gold_dir=gold_dir, valid_entry_ids={"A", "B"},
        rubric_hash="r", adjudicator_id="t", classifier_labels=winner,
    )
    tally_w = calibrate_with_gold(base, gold_winner, set(), {"A", "B"})
    rw_a = tally_w.recall_counts[("A", "security")]
    assert rw_a.true_positives == 0 and rw_a.false_negatives == 1

    # The two channels genuinely disagree -> F1 wiring changes the recall.
    assert rc_a != rw_a
```

- [ ] **Step 2: run** → FAIL only if the chain is wrong (it should pass once Task 1 lands; if it fails, the contract is broken — investigate, do not weaken).

- [ ] **Step 3:** no new implementation — this test consumes Task 1's capability. If it fails, the F1 wiring is incorrect; fix Task 1.

- [ ] **Step 4: run** the full suite `uv run pytest -q` → PASS.

- [ ] **Step 5: gate + commit**
```bash
uv run ruff check . && uv run mypy engine tests
git add tests/unit/test_gold_f1_winner_recall.py
git commit -m "test(calibrate): prove winner labels drive recall vs consensus (Plan 8e F1)"
```

---

## Reassessment notes (for the post-F1 Phase-2 scoping)
Record in `LESSONS-rarr.md` after review:
- F1 **capability + contract proof DONE**; the live CLI flip (pass `classifier_labels=load_classifier_labels(cycle/classify/labeled_incidents.json)` at `pipeline_executor.py:294` and `calibration.py:401`) is the next decision — it changes current recall numbers (recall now measures the counted classifier, not consensus) → expect oracle corrections to synthetic/parity tests; sequence it with the re-run/cycle strategy + RM14.
- **Open semantic surfaced by F1:** an incident the classifier labeled out-of-scope is absent from `labeled_incidents.json`, so the coverage guard would raise. Decide at the flip: treat a missing label as an explicit recall-miss sentinel (so OOS-on-a-truly-in-scope incident counts as `recall_fn`) vs require full in-scope coverage. `calibrate_with_gold` currently SKIPS `classifier_entry_id is None`, which would wrongly drop such incidents — so the flip needs an explicit OOS policy.
- **Provenance gap (→ F3b):** the scored-classifier identity is not yet in `provenance_hash`/the lock; bind it when flipping.

## Self-review
Backward compat: `classifier_labels=None` → `predicted = llm_consensus` → identical to today (existing `test_gold_loader.py` unchanged). Coverage + vocab guards only fire on the provided channel. Contract proof uses the real loader→tally chain and asserts the winner/consensus recall genuinely differ. No placeholders; full code each step. Out-of-scope items (CLI flip, OOS semantic, provenance binding) explicitly deferred with rationale.
