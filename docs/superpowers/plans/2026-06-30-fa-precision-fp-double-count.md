# F-A — Fix the adjudicated-path precision FP double-count Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Checkbox steps. This touches the F4-pinned `calibrate_with_gold` and changes precision numbers — expect oracle corrections; the reviewer judges correction-vs-weakening.

**Goal:** Stop double-counting precision false-positives for adjudicated in-scope misclassifications. Verified on the real 2026 goldset: total precision FP = 288 = 108 (recall-derived, `tally.py:260-266`) + 180 (loader precision-labels, `tally.py:268-274`) with NO dedup; the overlap (incidents that are both a recall-derived FP and have a precision_label) is counted twice → precision posteriors → the W/precision term of the measurement-error model → the RARR headline are corrupted.

**Fix (dedup, not deletion):** the loader's `precision_labels` are the complete precision accounting for the adjudicated path (TP for correct claims, FP for wrong claims, one per claim). The recall-derived FP exists for inputs that have recall labels but NO precision labels (curation / the F4 pin). So the recall-derived FP must fire **only when the incident has no precision_label**. This:
- removes the double-count (adjudicated misclassification → its precision_label is the single source),
- preserves the F4 path (`test_recall_single_label_semantics` builds recall labels with `precision_labels=[]` → the dedup set is empty → recall-derived fires),
- preserves the truth-OOS-predicted-in-scope FP (a labels-empty record gets NO precision_label from the loader → recall-derived correctly counts the FP).

Deleting either path would be wrong: dropping the loader precision_labels loses precision TP (correct claims get no TP); dropping the recall-derived FP breaks the curation/F4 path.

## Global Constraints
- NO new dependencies. mypy strict + ruff clean. `uv run ruff check . && uv run mypy engine tests` before each commit; FULL `uv run pytest -q` before push.
- **F4 pin `tests/unit/test_recall_single_label_semantics.py` MUST stay green UNCHANGED** (it relies on the recall-derived FP and uses empty precision_labels → unaffected by the dedup).
- **Oracle corrections:** any existing test asserting the OLD (doubled) precision FP/total from the adjudicated path must be updated to the correct single-count value — that is a CORRECTION (the old value was the bug), not a weakening. Do NOT delete assertions or replace `== n` with `>= 0`. If a correct new value can't be derived from the test's own fixture, set DONE_WITH_CONCERNS and list it.
- No AI attribution in commits. Branch `plan7/engine-upgrade-recall-pl` (PR #22).

---

### Task 1: Dedup the recall-derived precision FP against explicit precision_labels

**File:** `engine/calibrate/tally.py` ; **Test:** `tests/unit/test_precision_double_count.py`

**Interfaces:** `calibrate_with_gold` unchanged signature; `GoldRecallLabel.incident_id` and `GoldPrecisionLabel.incident_id` already exist.

- [ ] **Step 1: failing tests** — create `tests/unit/test_precision_double_count.py`:
```python
"""F-A: adjudicated misclassifications must count ONE precision FP, not two."""
from __future__ import annotations

from engine.calibrate.gold_schema import (
    GoldCalibration,
    GoldPrecisionLabel,
    GoldRecallLabel,
)
from engine.calibrate.tally import TallyResult, calibrate_with_gold


def _base() -> TallyResult:
    return TallyResult(
        precision_counts={}, recall_counts={}, rollup_counts={},
        total_coded=0, amendments_applied=0,
    )


def test_adjudicated_misclassification_is_one_precision_fp() -> None:
    # Adjudicated path: the SAME incident produces a recall label (classifier=B,
    # truth=A) AND a loader precision_label (claimed=B, is_correct=False).
    # Before the fix both fire -> FP=2; after the fix -> FP=1.
    gold = GoldCalibration(
        recall_labels=[GoldRecallLabel(
            incident_id="i0", true_entry_ids=["A"],
            classifier_entry_id="B", source="llm-adjudicated",
        )],
        precision_labels=[GoldPrecisionLabel(
            incident_id="i0", claimed_entry_id="B",
            is_correct=False, source="llm-adjudicated",
        )],
        provenance_hash="h", rubric_hash="r", adjudicator_id="t", session_count=1,
    )
    merged = calibrate_with_gold(_base(), gold, set(), {"A", "B"})
    p_b = merged.precision_counts[("B", "security")]
    assert p_b.false_positives == 1  # not 2
    assert p_b.total == 1


def test_recall_only_path_still_derives_precision_fp() -> None:
    # Curation / F4 path: recall label, NO precision_labels -> recall-derived FP
    # MUST still fire (the dedup set is empty).
    gold = GoldCalibration(
        recall_labels=[GoldRecallLabel(
            incident_id="i0", true_entry_ids=["A"],
            classifier_entry_id="B", source="manual",
        )],
        precision_labels=[],
        provenance_hash="h", rubric_hash="r", adjudicator_id="t", session_count=1,
    )
    merged = calibrate_with_gold(_base(), gold, set(), {"A", "B"})
    assert merged.precision_counts[("B", "security")].false_positives == 1


def test_correct_adjudicated_claim_is_one_precision_tp() -> None:
    # Correct claim: precision_label is_correct=True -> 1 TP; recall-derived does
    # not fire (B in truth). No change from the fix, asserted for completeness.
    gold = GoldCalibration(
        recall_labels=[GoldRecallLabel(
            incident_id="i0", true_entry_ids=["B"],
            classifier_entry_id="B", source="llm-adjudicated",
        )],
        precision_labels=[GoldPrecisionLabel(
            incident_id="i0", claimed_entry_id="B",
            is_correct=True, source="llm-adjudicated",
        )],
        provenance_hash="h", rubric_hash="r", adjudicator_id="t", session_count=1,
    )
    merged = calibrate_with_gold(_base(), gold, set(), {"A", "B"})
    p_b = merged.precision_counts[("B", "security")]
    assert p_b.true_positives == 1 and p_b.false_positives == 0 and p_b.total == 1
```

- [ ] **Step 2: run** `uv run pytest tests/unit/test_precision_double_count.py -v` → `test_adjudicated_misclassification_is_one_precision_fp` FAILS (FP==2).

- [ ] **Step 3: implement the dedup** — in `engine/calibrate/tally.py` `calibrate_with_gold`, after the dict initializations (the `precision_total: dict... = {}` line, ~229) and before `for label in gold.recall_labels:`, add:
```python
    # F-A: incidents that carry an EXPLICIT precision verdict (a GoldPrecisionLabel,
    # e.g. every adjudicated incident) are scored for precision via that label
    # below; deriving a second precision FP from the recall label would
    # double-count.  The recall-derived FP therefore fires only for incidents
    # WITHOUT a precision_label (the curation / F4 path).
    precision_label_incident_ids = {p.incident_id for p in gold.precision_labels}
```
Then change the recall-derived precision-FP condition from:
```python
        if (
            label.classifier_entry_id not in label.true_entry_ids
            and label.classifier_entry_id != OUT_OF_SCOPE
        ):
```
to:
```python
        if (
            label.classifier_entry_id not in label.true_entry_ids
            and label.classifier_entry_id != OUT_OF_SCOPE
            and label.incident_id not in precision_label_incident_ids
        ):
```

- [ ] **Step 4: run** `uv run pytest tests/unit/test_precision_double_count.py tests/unit/test_recall_single_label_semantics.py -v` → all PASS (the 3 new tests + the F4 pin unchanged).

- [ ] **Step 5: full suite + oracle corrections** — `uv run pytest -q`. For each failure: confirm it asserts the OLD doubled precision FP/total from the adjudicated path; recompute the correct single-count value from the test's own fixture (each adjudicated claim = ONE precision observation: TP if predicted∈truth else FP); update the assertion. Do NOT weaken. `test_recall_single_label_semantics.py` and any recall-only-path test must NOT need changes. Record every changed oracle (old→new) in the report.

- [ ] **Step 6: gate + commit**
```bash
uv run ruff check . && uv run mypy engine tests
git add engine/calibrate/tally.py tests/unit/test_precision_double_count.py
# plus any oracle-corrected test files
git commit -m "fix(calibrate): dedup adjudicated precision FP against explicit precision labels (F-A)"
```

---

## Self-review
The dedup keys on `incident_id`: an incident with a `GoldPrecisionLabel` is scored for precision by that label only (TP or FP); the recall-derived FP fires only when there is no precision_label for the incident. Cases: (1) adjudicated in-scope misclassification → has precision_label → recall-derived suppressed → 1 FP (was 2); (2) adjudicated correct claim → predicted∈truth → recall-derived doesn't fire anyway; precision_label → 1 TP; (3) curation/F4 (no precision_labels) → recall-derived fires → 1 FP; (4) labels-empty truth-OOS, predicted in-scope → loader makes NO precision_label → recall-derived fires → 1 FP (correct: claiming in-scope on a truly-OOS incident is an FP); (5) predicted OOS → recall-derived suppressed by the `!= OUT_OF_SCOPE` guard + no precision_label → 0. F4 pin preserved (empty precision_labels). No placeholders; full code. The fix changes numbers (the bug's whole point) → oracle corrections expected and bounded to adjudicated-precision assertions.
