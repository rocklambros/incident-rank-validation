# Unit A — Recall CLI flip + OOS policy (a) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Checkbox steps. This is a BEHAVIOR-CHANGE plan (recall calibration switches from the goldset's stored `llm_consensus` to the classifier whose labels build the counts) — expect oracle corrections to synthetic/parity tests; the reviewer judges correction-vs-weakening.

**Goal:** Complete F1 end-to-end: wire the `classifier_labels` channel into the calibrate + infer phases so recall (and the overlap `W`) measure the classifier whose labels build the incidence counts, with **OOS policy (a)**: a goldset incident with no in-scope classifier label (out-of-scope / absent from `labeled_incidents.json`) is a **recall miss**, not a coverage error and not a precision false-positive.

**Why:** F1 built the capability + proof (commits `fc72c98..a2f4b5a`); this flips the live behavior. It is a current-pipeline correctness fix — today recall-corrects classifier-X's counts using classifier-Y's recall. The byte-immutable 2026 cycle is NOT re-run; this corrects future (Phase-3) runs.

**Owner decision encoded:** OOS policy = **(a) missing → recall-miss** (chosen 2026-06-27). A classifier that says out-of-scope made no positive claim → recall miss for each true entry, NO precision FP.

## Global Constraints
- NO new dependencies. mypy strict + ruff clean. Run `uv run ruff check . && uv run mypy engine tests` before each commit; FULL `uv run pytest -q` before push.
- The F4 recall semantics (`tests/unit/test_recall_single_label_semantics.py`) must stay green — the OOS change must not alter in-scope recall semantics.
- **Oracle corrections:** when an existing test's asserted recall/precision/W numbers change because recall now reflects `labeled_incidents` (not consensus), update the oracle to the NEW correct value — do NOT weaken the assertion (e.g., do not delete it or replace `== n` with `>= 0`). If you cannot determine the correct new value by reasoning, set DONE_WITH_CONCERNS and list the test.
- No AI attribution in commits. Branch `plan7/engine-upgrade-recall-pl` (PR #22).

## Out of scope (recorded, NOT fixed here)
- **Precision FP double-count (discovered):** for the adjudicated path, `calibrate_with_gold` derives a precision FP from the recall label (`tally.py:260-263`) AND from the loader-generated precision label (`tally.py:265-271`), double-counting precision FP for in-scope misclassifications. Orthogonal to this unit; record in LESSONS as a new Phase-2 finding.
- Binding the scored-classifier identity into provenance/the lock (F3b — Phase 3).

---

### Task 1: OOS sentinel + OOS-aware adjudicated loader

**Files:** `engine/calibrate/gold_schema.py`, `engine/calibrate/gold_loader.py` ; **Test:** `tests/unit/test_gold_f1_winner_recall.py`

Replace the F1 coverage-guard *raise* with OOS policy (a): a missing classifier label becomes the OOS sentinel (a recall miss), not an error.

- [ ] **Step 1: add the shared sentinel** — in `engine/calibrate/gold_schema.py`, at module top (after the module docstring / `from __future__` line), add:
```python
OUT_OF_SCOPE = "out-of-scope"
"""Sentinel classifier prediction: the classifier assigned no in-scope entry.

A recall MISS for each true entry, and NOT a precision false-positive (no
positive claim was made).  Matches the bake-off harness's OOS class string.
"""
```

- [ ] **Step 2: update the loader** — in `engine/calibrate/gold_loader.py`, import the sentinel (add `OUT_OF_SCOPE` to the existing `from engine.calibrate.gold_schema import (...)` block), and replace the `else:` branch of the `classifier_labels` handling in `_load_recall_from_adjudicated` (the F1 coverage-guard block) with the OOS-default version. The block currently reads:
```python
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
```
Replace with:
```python
        else:
            # Plan 8e F1 + OOS policy (a): score the classifier whose labels
            # build the incidence counts (the bake-off winner in Phase 3).  A
            # goldset incident with no in-scope classifier label (out-of-scope,
            # hence absent from labeled_incidents.json) is a recall MISS, not a
            # coverage error: predict the OOS sentinel.
            predicted = classifier_labels.get(incident_id, OUT_OF_SCOPE)
            if predicted != OUT_OF_SCOPE and predicted not in valid_entry_ids:
                raise ValueError(
                    f"classifier label '{predicted}' for incident "
                    f"'{incident_id}' not in rubric."
                )
```

- [ ] **Step 3: do not emit a precision label for OOS** — in the same function, the precision-label generation reads:
```python
        if predicted and labels:
            precision.append(GoldPrecisionLabel(
                incident_id=incident_id,
                claimed_entry_id=predicted,
                is_correct=(predicted in labels),
                source="llm-adjudicated",
            ))
```
Change the condition to skip the OOS sentinel (no positive claim → no precision row):
```python
        if predicted and predicted != OUT_OF_SCOPE and labels:
            precision.append(GoldPrecisionLabel(
                incident_id=incident_id,
                claimed_entry_id=predicted,
                is_correct=(predicted in labels),
                source="llm-adjudicated",
            ))
```

- [ ] **Step 4: replace the coverage-guard test with the OOS-miss test** — in `tests/unit/test_gold_f1_winner_recall.py`, DELETE `test_coverage_guard_raises_on_missing_incident` and add:
```python
def test_missing_classifier_label_becomes_oos_recall_miss(tmp_path: Path) -> None:
    from engine.calibrate.gold_schema import OUT_OF_SCOPE

    gold_dir = _write_adjudicated(tmp_path)
    # INC-2 has no classifier label -> OOS sentinel (a recall miss for its truth).
    gold = load_gold_calibration(
        gold_dir=gold_dir, valid_entry_ids={"A", "B"},
        rubric_hash="r", adjudicator_id="t",
        classifier_labels={"INC-1": "A"},  # INC-2 deliberately absent
    )
    by_id = {r.incident_id: r for r in gold.recall_labels}
    assert by_id["INC-2"].classifier_entry_id == OUT_OF_SCOPE
    # No precision row is generated for an OOS prediction.
    assert all(p.claimed_entry_id != OUT_OF_SCOPE for p in gold.precision_labels)
```

- [ ] **Step 5: run + gate + commit**
```bash
uv run pytest tests/unit/test_gold_f1_winner_recall.py tests/unit/test_gold_loader.py -q
uv run ruff check . && uv run mypy engine tests
git add engine/calibrate/gold_schema.py engine/calibrate/gold_loader.py tests/unit/test_gold_f1_winner_recall.py
git commit -m "feat(calibrate): OOS policy (a) — missing classifier label is a recall miss (Plan 8e F1)"
```

---

### Task 2: `calibrate_with_gold` — OOS counts as recall miss, not precision FP

**File:** `engine/calibrate/tally.py` ; **Test:** `tests/unit/test_gold_f1_winner_recall.py`

The recall loop already treats the OOS sentinel as a miss (sentinel ≠ any in-scope true entry). The only change needed: the recall-derived precision FP (line 260) must NOT fire for the OOS sentinel.

- [ ] **Step 1: failing test** — append to `tests/unit/test_gold_f1_winner_recall.py`:
```python
def test_oos_prediction_is_recall_miss_no_precision_fp() -> None:
    from engine.calibrate.gold_schema import (
        GoldCalibration,
        GoldRecallLabel,
        OUT_OF_SCOPE,
    )
    from engine.calibrate.tally import TallyResult, calibrate_with_gold

    base = TallyResult(
        precision_counts={}, recall_counts={}, rollup_counts={},
        total_coded=0, amendments_applied=0,
    )
    gold = GoldCalibration(
        recall_labels=[GoldRecallLabel(
            incident_id="i0", true_entry_ids=["A"],
            classifier_entry_id=OUT_OF_SCOPE, source="g",
        )],
        precision_labels=[],
        provenance_hash="h", rubric_hash="r", adjudicator_id="t", session_count=1,
    )
    merged = calibrate_with_gold(base, gold, set(), {"A"})
    # Recall: a miss for A (it was truly A, classifier said out-of-scope).
    rc = merged.recall_counts[("A", "security")]
    assert rc.true_positives == 0 and rc.false_negatives == 1
    # Precision: NO false-positive cell for the OOS sentinel.
    assert (OUT_OF_SCOPE, "security") not in merged.precision_counts
```

- [ ] **Step 2: run** → FAIL (a precision_fp cell for OUT_OF_SCOPE currently appears).

- [ ] **Step 3: implement** — in `engine/calibrate/tally.py`, add `OUT_OF_SCOPE` to the `from engine.calibrate.gold_schema import (...)` import, and change the recall-derived precision-FP condition:
```python
        if label.classifier_entry_id not in label.true_entry_ids:
```
to:
```python
        if (
            label.classifier_entry_id not in label.true_entry_ids
            and label.classifier_entry_id != OUT_OF_SCOPE
        ):
```

- [ ] **Step 4: run + gate** — `uv run pytest tests/unit/test_gold_f1_winner_recall.py tests/unit/test_recall_single_label_semantics.py -q` (F4 pin must stay green); `uv run ruff check . && uv run mypy engine tests`.

- [ ] **Step 5: commit**
```bash
git add engine/calibrate/tally.py tests/unit/test_gold_f1_winner_recall.py
git commit -m "feat(calibrate): OOS sentinel is a recall miss, not a precision FP (Plan 8e F1)"
```

---

### Task 3: CLI flip — calibrate + infer phases score the actual classifier

**Files:** `engine/cli/calibration.py`, `engine/cli/pipeline_executor.py` ; **Tests:** existing suite (oracle corrections)

Wire `classifier_labels=load_classifier_labels(cycle/classify/labeled_incidents.json)` into both gold-load call sites so recall (calibrate phase) and the overlap `W` (infer phase) reflect the classifier whose labels build the counts.

- [ ] **Step 1: infer phase (overlap `W`)** — in `engine/cli/pipeline_executor.py` `execute_infer_phase`, `labeled_path = cycle/"classify"/"labeled_incidents.json"` already exists and is checked. Add the import (`from engine.calibrate.gold_loader import load_classifier_labels`, beside the existing `load_gold_calibration` import in that block) and thread the labels into the `load_gold_calibration(...)` call. Change:
```python
            _gold = load_gold_calibration(
                gold_dir=_gold_dir,
                valid_entry_ids=set(measurable_entries),
                rubric_hash=_rubric_hash,
                adjudicator_id="executor",
            )
```
to:
```python
            _classifier_labels = load_classifier_labels(labeled_path)
            _gold = load_gold_calibration(
                gold_dir=_gold_dir,
                valid_entry_ids=set(measurable_entries),
                rubric_hash=_rubric_hash,
                adjudicator_id="executor",
                classifier_labels=_classifier_labels,
            )
```

- [ ] **Step 2: calibrate command** — in `engine/cli/calibration.py`, the `calibrate` command has `cycle` in scope (`cal_dir = cycle / "calibration"`). Before the `load_gold_calibration(...)` call (line ~401), load the classifier labels if present, and pass them. Add the import to the existing `from engine.calibrate.gold_loader import load_gold_calibration` line (make it `import load_classifier_labels, load_gold_calibration`), then change the call:
```python
        gold = load_gold_calibration(
            **gold_kwargs,
            valid_entry_ids=all_entry_ids,
            rubric_hash=rubric_hash,
            adjudicator_id="cli",
        )
```
to:
```python
        _labeled = cycle / "classify" / "labeled_incidents.json"
        _classifier_labels = (
            load_classifier_labels(_labeled) if _labeled.exists() else None
        )
        gold = load_gold_calibration(
            **gold_kwargs,
            valid_entry_ids=all_entry_ids,
            rubric_hash=rubric_hash,
            adjudicator_id="cli",
            classifier_labels=_classifier_labels,
        )
```

- [ ] **Step 3: run the FULL suite and fix oracle corrections** — `uv run pytest -q`. Some synthetic/parity/integration tests that asserted consensus-based recall/precision/W numbers will now see the classifier-based numbers. For each failure:
  - Confirm the failure is *because recall/W now reflects the classifier's labels* (the intended change), not an unintended break.
  - Update the asserted number to the NEW correct value (compute it from the test's own fixture: recall for entry X = fraction of truly-X goldset incidents whose `labeled_incidents` entry_id == X; an absent/OOS label is a miss). Do NOT weaken the assertion.
  - If a test's correct new value cannot be determined by reasoning from its fixture, STOP and set DONE_WITH_CONCERNS listing that test (do not guess).
  - The F4 pin (`test_recall_single_label_semantics.py`) and `test_gold_loader.py` (consensus path) must NOT need changes — if they fail, you broke something; fix the code.

- [ ] **Step 4: gate + commit**
```bash
uv run ruff check . && uv run mypy engine tests
git add -A
git commit -m "feat(cli): recall + overlap-W score the classifier's labels, not consensus (Plan 8e F1)"
```

---

### Task 4: End-to-end OOS proof through the loader→tally chain

**File:** `tests/unit/test_gold_f1_winner_recall.py` (append)

- [ ] **Step 1: test** — append:
```python
def test_oos_missing_label_recall_miss_end_to_end(tmp_path: Path) -> None:
    from engine.calibrate.tally import TallyResult, calibrate_with_gold

    gold_dir = _write_adjudicated(tmp_path)
    base = TallyResult(
        precision_counts={}, recall_counts={}, rollup_counts={},
        total_coded=0, amendments_applied=0,
    )
    # INC-1 truly {A}; classifier gives NO label (OOS) -> recall miss for A.
    gold = load_gold_calibration(
        gold_dir=gold_dir, valid_entry_ids={"A", "B"},
        rubric_hash="r", adjudicator_id="t",
        classifier_labels={"INC-2": "B"},  # INC-1 absent -> OOS
    )
    merged = calibrate_with_gold(base, gold, set(), {"A", "B"})
    rc_a = merged.recall_counts[("A", "security")]
    assert rc_a.true_positives == 0 and rc_a.false_negatives == 1
```

- [ ] **Step 2: run + full suite + gate + commit**
```bash
uv run pytest tests/unit/test_gold_f1_winner_recall.py -q && uv run pytest -q
uv run ruff check . && uv run mypy engine tests
git add tests/unit/test_gold_f1_winner_recall.py
git commit -m "test(calibrate): OOS missing label is a recall miss end-to-end (Plan 8e F1)"
```

---

## Self-review
F4 recall semantics untouched (OOS suppression only affects the new sentinel path). OOS sentinel is shared (`gold_schema.OUT_OF_SCOPE`), counts as recall miss in the existing recall loop, and is excluded from BOTH precision paths (loader precision-label generation + the recall-derived FP in `calibrate_with_gold`). CLI flip threads `classifier_labels` from the cycle's `labeled_incidents.json` at both gold-load sites. Oracle corrections are expected and bounded to tests asserting consensus-based numbers; the reviewer judges correction-vs-weakening. Discovered precision double-count recorded as out-of-scope. No placeholders; full code each step.
