# Plan 8a — Engine Correctness Foundations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the existing primary concordance pipeline *correct and honestly reproducible* — fix recall calibration, schema-version the manifest so new fields don't break the frozen 2026 lock, bind provenance, rank by incidence λ·size, activate the precision/FP term, and wire+gate the robustness-spec spread — all TDD-tested on fixtures with no RunPod.

**Architecture:** Six independent engine changes, each a failing-test→implement→pass→commit cycle. No new models or data runs (those are Plans 8b–8e). Recall posteriors are re-sourced from the gold truth-vs-prediction path; the manifest gains a schema version so `goldset_hash` can be added without invalidating the immutable 2026 cycle; the concordance ranks incidence (λ·size) instead of bare λ; the overlap matrix `W` is built from the goldset confusion so the precision posteriors stop being inert; and `run_robustness_inference` is finally wired into the CLI behind a decide-time completeness gate.

**Tech Stack:** Python 3.12, NumPyro/JAX (CPU only), NumPy, pytest, frozen dataclasses, Click CLI. Determinism: `JAX_PLATFORM_NAME=cpu`, `JAX_ENABLE_X64=true`.

## Global Constraints

- **CPU only for all Bayesian fits** — `assert jax.default_backend() == "cpu"` (`engine/model/inference.py:121`). Never run NUTS on GPU.
- **Reproducibility is within MCSE, not bit-exact** across machines; pin `OMP_NUM_THREADS`/`XLA_FLAGS` where determinism is asserted.
- **The frozen `cycles/2026/` is byte-immutable** — no edits to it; the 2026 manifest lock MUST still verify after every change in this plan.
- **No new heavy dependencies** — use pinned `scipy`/`numpy` only.
- **No AI attribution** in any commit message or GitHub-visible content.
- **CI gates must stay green:** `uv run ruff check .`, `uv run mypy engine tests`, `uv run pytest -v`, `uv run semgrep --config .semgrep.yml --error engine/`, pre-push gitleaks.
- **Canonical hashing** is always `json.dumps(obj, sort_keys=True, separators=(",", ":"))` then `hashlib.sha256(...).hexdigest()`.
- **Recall truth lives in the adjudicated goldset** (`GoldRecallLabel.true_entry_ids` vs `classifier_entry_id`), never the recall-frame batches (which carry no classifier prediction).

---

### Task 1: Recall posteriors from gold truth-vs-prediction (kill the frame-size denominator)

**Files:**
- Modify: `engine/calibrate/tally.py:203-293` (`calibrate_with_gold` — recall merge)
- Test: `tests/unit/test_calibrate_with_gold_recall.py`

**Interfaces:**
- Consumes: `TallyResult` (with `recall_counts: dict[tuple[str,str], RecallTally]`), `GoldCalibration` (`recall_labels: list[GoldRecallLabel]` with `true_entry_ids: list[str]`, `classifier_entry_id: str | None`).
- Produces: `calibrate_with_gold(...) -> TallyResult` whose `recall_counts` derive **solely from gold** (per-entry truth-cell denominators); precision behavior unchanged. Downstream `compute_calibration` then yields wide recall posteriors for sparse entries.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_calibrate_with_gold_recall.py
from engine.calibrate.beta import BetaPosterior
from engine.calibrate.calibrate import compute_calibration
from engine.calibrate.gold_schema import GoldCalibration, GoldRecallLabel
from engine.calibrate.tally import PrecisionTally, RecallTally, TallyResult, calibrate_with_gold


def _base_tally_with_frame_padding() -> TallyResult:
    # Simulates the frame-size-padded recall the recall branch produces today:
    # ROLL-CFAS has exactly 1 true incident but a denominator of 100.
    return TallyResult(
        precision_counts={},
        recall_counts={("ROLL-CFAS", "ai-harm"): RecallTally(0, 100, 100)},
        rollup_counts={},
        total_coded=100,
        amendments_applied=0,
    )


def test_recall_derives_solely_from_gold_not_frame_padding():
    base = _base_tally_with_frame_padding()
    gold = GoldCalibration(
        recall_labels=[
            GoldRecallLabel(
                incident_id="INC-1",
                true_entry_ids=["ROLL-CFAS"],
                classifier_entry_id="out-of-scope",  # classifier MISSED it -> FN
                source="goldset",
            )
        ],
        precision_labels=[],
        provenance_hash="h",
        rubric_hash="r",
        adjudicator_id="RL",
        session_count=1,
    )
    merged = calibrate_with_gold(
        base_tally=base,
        gold=gold,
        base_incident_ids=set(),
        all_entry_ids={"ROLL-CFAS"},
        merge_stratum="ai-harm",
    )
    # Recall denominator must be the truth cell (1), NOT the frame size (100).
    rt = merged.recall_counts[("ROLL-CFAS", "ai-harm")]
    assert rt == RecallTally(true_positives=0, false_negatives=1, total_in_sample=1)

    cal, _ = compute_calibration(
        merged, all_entry_ids=["ROLL-CFAS"], frame_blind_ids=set(),
    )
    # Wide posterior: Beta(1, 2), mean 1/3 -- NOT the falsely-precise Beta(1, 101).
    assert cal.recall[("ROLL-CFAS", "ai-harm")] == BetaPosterior(alpha=1.0, beta=2.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_calibrate_with_gold_recall.py -v`
Expected: FAIL — current merge adds base+gold (`false_negatives=100+1=101`, `total_in_sample=100+1=101`), so the assertion on `RecallTally(0,1,1)` fails.

- [ ] **Step 3: Write minimal implementation**

In `engine/calibrate/tally.py`, change the recall seeding and merge inside `calibrate_with_gold`.

Replace the seeding line:
```python
    recall_counts = dict(base_tally.recall_counts)
```
with:
```python
    # Recall posteriors derive SOLELY from gold truth-vs-prediction (per-entry
    # truth-cell denominators). The recall-frame tally has no classifier
    # prediction, so its frame-size-padded recall counts are not real recall and
    # are intentionally dropped here. (Plan 8a, SD1/RM2.)
    recall_counts: dict[tuple[str, str], RecallTally] = {}
```

Replace the recall merge block:
```python
    for k in recall_total:
        existing = recall_counts.get(k)
        if existing:
            recall_counts[k] = RecallTally(
                true_positives=existing.true_positives + recall_tp.get(k, 0),
                false_negatives=existing.false_negatives + recall_fn.get(k, 0),
                total_in_sample=existing.total_in_sample + recall_total[k],
            )
        else:
            recall_counts[k] = RecallTally(
                true_positives=recall_tp.get(k, 0),
                false_negatives=recall_fn.get(k, 0),
                total_in_sample=recall_total[k],
            )
```
with:
```python
    for k in recall_total:
        recall_counts[k] = RecallTally(
            true_positives=recall_tp.get(k, 0),
            false_negatives=recall_fn.get(k, 0),
            total_in_sample=recall_total[k],
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_calibrate_with_gold_recall.py -v`
Expected: PASS.
Then run the existing calibration suite to catch regressions: `uv run pytest tests/unit -k calibrat -v` — Expected: PASS (precision paths unchanged).

- [ ] **Step 5: Commit**

```bash
git add engine/calibrate/tally.py tests/unit/test_calibrate_with_gold_recall.py
git commit -m "fix(calibrate): derive recall posteriors solely from gold truth-vs-prediction

The recall-frame tally has no classifier prediction, so its frame-size
denominator manufactured falsely-precise near-zero recall (Beta(1,101)) for
sparse entries. Recall now comes only from the adjudicated goldset's
per-truth-cell TP/FN, so sparse entries get honest wide posteriors."
```

---

### Task 2: Schema-version the manifest so `goldset_hash` doesn't break the 2026 lock

**Files:**
- Modify: `engine/prereg/manifest.py:28-89` (add fields + version-aware `to_dict`)
- Test: `tests/unit/test_manifest_schema_version.py`

**Interfaces:**
- Consumes: existing `PreregManifest` constructor (all current call sites pass no `schema_version`/`goldset_hash`).
- Produces: `PreregManifest` with new optional fields `schema_version: int = 1` and `goldset_hash: str | None = None`; `to_dict()` emits the **v1 field set** (no `schema_version`, no `goldset_hash`) when `schema_version == 1`, so v1 lock hashes are byte-stable; v2 manifests include both new keys.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_manifest_schema_version.py
from engine.prereg.lock import compute_lock_hash
from engine.prereg.manifest import PreregManifest

_BASE = dict(
    engine_version="1.2.0", engine_version_range_min="1.0.0",
    engine_version_range_max="2.0.0", cycle_id="t", taxonomy_hash="tx",
    snapshot_hash="sn", primary_spec="negative_binomial_per_stratum",
    robustness_specs=(), flag_threshold_tau=0.8, statistic="weighted_cohens_kappa",
    measurability_minimum=4, prior_scale=0.5, concentration_shape=5.0,
    concentration_rate=0.1, ess_fraction=0.4, meaningful_kappa_n=4, prng_seed=42,
    confidence_threshold=0.3, rubric_drafting_attestation=None,
    rubric_reviewer=None, statistical_reviewer=None, classifier_rule_hash=None,
    rubric_hash=None, post_hoc_register_path=None,
)


def test_v1_manifest_excludes_new_fields_from_hash():
    m = PreregManifest(**_BASE)  # schema_version defaults to 1
    d = m.to_dict()
    assert "schema_version" not in d
    assert "goldset_hash" not in d


def test_adding_goldset_hash_under_v1_does_not_change_hash():
    # A v1 manifest with a goldset_hash set but schema_version still 1 must hash
    # identically to one without it -- the field is not part of the v1 canonical form.
    m_plain = PreregManifest(**_BASE)
    m_with = PreregManifest(**_BASE, goldset_hash="abc123")
    assert compute_lock_hash(m_plain) == compute_lock_hash(m_with)


def test_v2_manifest_includes_new_fields_and_changes_hash():
    m1 = PreregManifest(**_BASE)
    m2 = PreregManifest(**_BASE, schema_version=2, goldset_hash="abc123")
    d2 = m2.to_dict()
    assert d2["schema_version"] == 2
    assert d2["goldset_hash"] == "abc123"
    assert compute_lock_hash(m1) != compute_lock_hash(m2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_manifest_schema_version.py -v`
Expected: FAIL — `PreregManifest` has no `schema_version`/`goldset_hash` kwargs (TypeError).

- [ ] **Step 3: Write minimal implementation**

In `engine/prereg/manifest.py`, add two defaulted fields after `lambda_min`:
```python
    lambda_min: float | None = None  # noise floor; default: prior_scale * 0.02
    schema_version: int = 1  # 1 = original field set; 2 = adds goldset_hash
    goldset_hash: str | None = None  # bound only when schema_version >= 2
```

Replace `to_dict` with a version-aware form:
```python
    def to_dict(self) -> dict[str, object]:
        """Canonical dict for JSON serialization and hashing.

        When schema_version == 1 the canonical form is the ORIGINAL field set
        (no schema_version, no goldset_hash) so pre-existing v1 locks stay
        byte-stable. v2+ includes the new fields. (Plan 8a, SD3/RM13.)
        """
        result: dict[str, object] = {}
        for field in dataclasses.fields(self):
            result[field.name] = _dc_to_dict(getattr(self, field.name))
        if self.schema_version == 1:
            result.pop("schema_version", None)
            result.pop("goldset_hash", None)
        return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_manifest_schema_version.py -v`
Expected: PASS.
Regression — the real 2026 lock must still verify:
Run: `uv run python -c "import json,pathlib; from engine.prereg.manifest import PreregManifest; from engine.prereg.lock import compute_lock_hash; d=json.loads(pathlib.Path('projects/owasp-llm/cycles/2026/prereg/manifest.json').read_text()); fn={f.name for f in __import__('dataclasses').fields(PreregManifest)}; m=PreregManifest(**{k:(tuple(v) if isinstance(v,list) else v) for k,v in d.items() if k in fn}); print('lock-ok', compute_lock_hash(m))"`
Expected: prints `lock-ok <hash>` matching `projects/owasp-llm/cycles/2026/prereg/manifest.lock` (no exception).

- [ ] **Step 5: Commit**

```bash
git add engine/prereg/manifest.py tests/unit/test_manifest_schema_version.py
git commit -m "feat(prereg): schema-version the manifest so goldset_hash is additive

Adding a field to the frozen dataclass would rehash every prior manifest and
break the immutable 2026 lock. to_dict() now emits the v1 field set when
schema_version==1, so v1 locks stay byte-stable while v2 manifests bind
goldset_hash."
```

---

### Task 3: Bind classifier labels + goldset + snapshot in provenance

**Files:**
- Modify: `engine/repro/bundle.py:9-16` (record `goldset_hash`)
- Modify: calibration provenance writer call site (in `engine/cli/calibration.py`, the `cal_calibrate` command) to add the classifier-label hash to `input_hashes`
- Test: `tests/unit/test_provenance_binding.py`

**Interfaces:**
- Consumes: `StageProvenance(stage_name, manifest_lock_hash, input_hashes: dict[str,str], output_hash, timestamp, engine_version)` (`engine/calibrate/provenance.py`); `hash_json(obj)` (`engine/snapshot/provenance.py:72`); `ReproductionBundle`.
- Produces: `ReproductionBundle` gains a `goldset_hash: str` field; the calibrate provenance `input_hashes` includes `"classifier_labels": <sha256>`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_provenance_binding.py
from pathlib import Path

from engine.repro.bundle import ReproductionBundle


def test_bundle_records_goldset_hash(tmp_path: Path):
    b = ReproductionBundle(
        cycle_id="2026-rarr", engine_version="1.3.0",
        snapshot_hash="snap", manifest_hash="man", lockfile_hash="lf",
        goldset_hash="gold-sha", provenance={"calibration_hash": "c"},
    )
    p = tmp_path / "bundle.json"
    b.write(p)
    rt = ReproductionBundle.read(p)
    assert rt.goldset_hash == "gold-sha"
    assert rt.snapshot_hash == "snap"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_provenance_binding.py -v`
Expected: FAIL — `ReproductionBundle.__init__() got an unexpected keyword argument 'goldset_hash'`.

- [ ] **Step 3: Write minimal implementation**

In `engine/repro/bundle.py`, add the field to the dataclass and serializer:
```python
@dataclass(frozen=True, slots=True)
class ReproductionBundle:
    cycle_id: str
    engine_version: str
    snapshot_hash: str
    manifest_hash: str
    lockfile_hash: str
    goldset_hash: str
    provenance: dict[str, str]

    def to_json(self) -> str:
        return (
            json.dumps(
                {
                    "cycle_id": self.cycle_id,
                    "engine_version": self.engine_version,
                    "snapshot_hash": self.snapshot_hash,
                    "manifest_hash": self.manifest_hash,
                    "lockfile_hash": self.lockfile_hash,
                    "goldset_hash": self.goldset_hash,
                    "provenance": self.provenance,
                },
                sort_keys=True,
                indent=2,
            )
            + "\n"
        )
```
(`write`/`read` are unchanged; `read` uses `cls(**d)` so the new key round-trips.)

Then, in `engine/cli/calibration.py` `cal_calibrate`, where the calibrate `StageProvenance` is built, add the classifier-label hash to `input_hashes`:
```python
        from engine.snapshot.provenance import hash_json
        labeled = json.loads((cycle / "classify" / "labeled_incidents.json").read_text())
        input_hashes = {
            "tally": tally_output_hash,
            "classifier_labels": hash_json(labeled),
        }
```
(Use the existing `tally` hash variable name already in scope; add the `classifier_labels` entry.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_provenance_binding.py -v`
Expected: PASS.
Regression: `uv run pytest tests/unit -k "bundle or provenance" -v` — Expected: PASS (update any existing bundle constructor in tests to pass `goldset_hash="..."`).

- [ ] **Step 5: Commit**

```bash
git add engine/repro/bundle.py engine/cli/calibration.py tests/unit/test_provenance_binding.py
git commit -m "feat(repro): bind goldset + classifier labels in provenance

Reproduction bundle records goldset_hash; calibrate provenance hashes the
classifier label file so a result is traceable to the exact classifier."
```

---

### Task 4: Rank the primary by incidence λ·size, not bare λ

**Files:**
- Modify: `engine/decide/concordance.py:53-249` (add `_ranks_from_incidence`, thread `stratum_sizes` through `compute_concordance`)
- Modify: `engine/cli/pipeline_executor.py` (pass `stratum_sizes` into `compute_concordance`)
- Test: `tests/unit/test_concordance_incidence_ranking.py`

**Interfaces:**
- Consumes: `InferenceResult.lambda_samples` (shape `(draws, n_entries)`), `InferenceResult.entry_ids`, a new `stratum_sizes: dict[str,int]` argument, and per-entry stratum membership (entries carry one stratum; reuse the `(entry, stratum)` keys already in `observed_counts`).
- Produces: `compute_concordance(..., entry_strata: dict[str, str], stratum_sizes: dict[str, int])`; ranking uses incidence `λ_e · size(stratum(e))`. New helper `_ranks_from_incidence(lam_draw, idx_map, common, entry_strata, stratum_sizes)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_concordance_incidence_ranking.py
import numpy as np

from engine.decide.concordance import _ranks_from_incidence


def test_incidence_ranking_uses_lambda_times_size():
    # Entry A has higher lambda but lives in a tiny stratum; entry B has lower
    # lambda in a huge stratum. By incidence (lambda*size), B outranks A.
    lam = np.array([0.9, 0.4])  # A, B
    idx = {"A": 0, "B": 1}
    common = ["A", "B"]
    entry_strata = {"A": "small", "B": "big"}
    sizes = {"small": 10, "big": 1000}
    ranks = _ranks_from_incidence(lam, idx, common, entry_strata, sizes)
    # incidence: A=9, B=400 -> B is rank 1, A is rank 2
    assert ranks[common.index("B")] == 1.0
    assert ranks[common.index("A")] == 2.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_concordance_incidence_ranking.py -v`
Expected: FAIL — `cannot import name '_ranks_from_incidence'`.

- [ ] **Step 3: Write minimal implementation**

In `engine/decide/concordance.py`, add the helper next to `_ranks_from_lambda`:
```python
def _ranks_from_incidence(
    lam_draw: npt.NDArray[np.float64],
    idx_map: dict[str, int],
    common: list[str],
    entry_strata: dict[str, str],
    stratum_sizes: dict[str, int],
) -> npt.NDArray[np.float64]:
    """Rank entries by incidence (lambda_e * stratum_size_e), descending."""
    vals = np.array(
        [lam_draw[idx_map[e]] * float(stratum_sizes[entry_strata[e]]) for e in common]
    )
    order = np.argsort(-vals)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(common) + 1, dtype=np.float64)
    return ranks
```
Add `entry_strata: dict[str, str]` and `stratum_sizes: dict[str, int]` parameters to `compute_concordance`, and replace every call `_ranks_from_lambda(inference_result.lambda_samples[s], inf_idx, common)` with `_ranks_from_incidence(inference_result.lambda_samples[s], inf_idx, common, entry_strata, stratum_sizes)` (three call sites: kappa loop, flags loop, comparisons loop). In `pipeline_executor.py`, pass `entry_strata` (built from the `(entry, stratum)` keys of `observed_counts`) and `stratum_sizes` into the `compute_concordance(...)` call.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_concordance_incidence_ranking.py -v`
Expected: PASS.
Regression: `uv run pytest tests/unit -k concordance -v` — update existing concordance tests to pass `entry_strata`/`stratum_sizes`. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add engine/decide/concordance.py engine/cli/pipeline_executor.py tests/unit/test_concordance_incidence_ranking.py
git commit -m "feat(decide): rank concordance by incidence lambda*size

The decision-relevant quantity is incidence, not per-corpus intensity. The
primary now ranks by lambda_e * stratum_size_e; the baseline is recomputed
under the same estimator for an apples-to-apples comparison (Plan 8a, SD8)."
```

---

### Task 5: Build the overlap matrix `W` from the goldset confusion (activate the FP/precision term)

**Files:**
- Create: `engine/calibrate/confusion.py`
- Modify: `engine/cli/pipeline_executor.py:253` (replace `OverlapWeights(weights={})`)
- Test: `tests/unit/test_confusion_overlap.py`

**Interfaces:**
- Consumes: `GoldCalibration.recall_labels` (each has `true_entry_ids`, `classifier_entry_id`); `measurable_entries: tuple[str,...]`.
- Produces: `build_overlap_from_confusion(gold: GoldCalibration, measurable_entries: tuple[str,...]) -> OverlapWeights` where `W[target][source]` = fraction of `source`'s false positives (predicted `source`, true `target`) — column-normalized, self-loops dropped, capped at column-sum ≤ 1.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_confusion_overlap.py
from engine.calibrate.confusion import build_overlap_from_confusion
from engine.calibrate.gold_schema import GoldCalibration, GoldRecallLabel


def _gold(*triples) -> GoldCalibration:
    return GoldCalibration(
        recall_labels=[
            GoldRecallLabel(incident_id=f"i{n}", true_entry_ids=[t],
                            classifier_entry_id=p, source="g")
            for n, (t, p) in enumerate(triples)
        ],
        precision_labels=[], provenance_hash="h", rubric_hash="r",
        adjudicator_id="RL", session_count=1,
    )


def test_overlap_built_from_misclassifications():
    # Two incidents truly LLM02 but the classifier predicted LLM01 -> LLM01's FPs
    # leak from LLM02. One correct LLM01. So source=LLM01 has 2 FPs, all from LLM02.
    gold = _gold(("LLM02", "LLM01"), ("LLM02", "LLM01"), ("LLM01", "LLM01"))
    W = build_overlap_from_confusion(gold, ("LLM01", "LLM02"))
    # W[target=LLM02][source=LLM01] = 2 FPs from LLM02 / 2 total FPs of LLM01 = 1.0
    assert abs(W.weights["LLM02"]["LLM01"] - 1.0) < 1e-9
    # no self-loop
    assert "LLM01" not in W.weights.get("LLM01", {})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_confusion_overlap.py -v`
Expected: FAIL — module `engine.calibrate.confusion` does not exist.

- [ ] **Step 3: Write minimal implementation**

```python
# engine/calibrate/confusion.py
"""Build the FP-leakage overlap matrix W from goldset confusion (Plan 8a, RM12)."""
from __future__ import annotations

from engine.calibrate.gold_schema import GoldCalibration
from engine.model.overlap import OverlapWeights


def build_overlap_from_confusion(
    gold: GoldCalibration,
    measurable_entries: tuple[str, ...],
) -> OverlapWeights:
    entries = set(measurable_entries)
    # fp_counts[source][target] = # incidents predicted `source` but truly `target`
    fp_counts: dict[str, dict[str, int]] = {}
    for label in gold.recall_labels:
        pred = label.classifier_entry_id
        if pred is None or pred not in entries:
            continue
        for true_eid in label.true_entry_ids:
            if true_eid == pred or true_eid not in entries:
                continue
            fp_counts.setdefault(pred, {}).setdefault(true_eid, 0)
            fp_counts[pred][true_eid] += 1

    weights: dict[str, dict[str, float]] = {}
    for source, targets in fp_counts.items():
        total = sum(targets.values())
        if total == 0:
            continue
        for target, n in targets.items():
            weights.setdefault(target, {})[source] = n / total
    return OverlapWeights(weights=weights)
```
In `engine/cli/pipeline_executor.py`, replace:
```python
    overlap = OverlapWeights(weights={})
```
with:
```python
    from engine.calibrate.confusion import build_overlap_from_confusion
    overlap = build_overlap_from_confusion(gold, measurable_entries)
```
(where `gold` is the `GoldCalibration` already loaded for calibration in the executor; if it is not in scope at that point, load it via the same `load_gold_calibration(...)` call the calibrate stage uses, keyed to the cycle's gold dir).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_confusion_overlap.py -v`
Expected: PASS.
Regression: `uv run pytest tests/unit -k "overlap or inference" -v` — Expected: PASS (the `OverlapWeights.__post_init__` column-stochastic check holds since each column sums to ≤ 1 by construction).

- [ ] **Step 5: Commit**

```bash
git add engine/calibrate/confusion.py engine/cli/pipeline_executor.py tests/unit/test_confusion_overlap.py
git commit -m "feat(calibrate): build overlap matrix W from goldset confusion

The production executor passed an empty W, zeroing the false-positive term and
making the precision posteriors inert. W is now the column-normalized
off-diagonal of the goldset confusion matrix, so precision correction is live."
```

---

### Task 6: Wire `run_robustness_inference` into the CLI behind a heterogeneous spread + decide-time gate

**Files:**
- Modify: `engine/decide/robustness_multiplicity.py:30-37` (extend `SpecResult` with optional non-kappa fields)
- Modify: `engine/cli/pipeline.py:580-615` (run robustness specs; pass executed `primary_spec`; build `RobustnessSpread`; gate)
- Modify: `tests/proofs/test_two_cycle_parity.py` (assert executed spec identity)
- Test: `tests/unit/test_robustness_wiring.py`

**Interfaces:**
- Consumes: `manifest.robustness_specs: tuple[str,...]`, `run_robustness_inference(manifest, spec_name, measurable_entries, strata, observed_counts, stratum_sizes, calibration, overlap, ...) -> InferenceResult`, `compute_concordance(...)`, `SpecResult`, `RobustnessSpread`.
- Produces: `build_robustness_spread(manifest, primary_concordance, primary_spec_name, run_specs) -> RobustnessSpread`; a `decide`-time check `assert_robustness_complete(manifest, spread)` that raises if declared specs are missing; `SpecResult` gains `sigma_u: float | None = None` and `extra_rankings: dict[str, tuple[str,...]] | None = None` (populated by Plans 8b/8c).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_robustness_wiring.py
import pytest

from engine.decide.robustness_multiplicity import RobustnessSpread, SpecResult


def _spec(name, k=0.2):
    return SpecResult(spec_name=name, weighted_kappa_median=k,
                      weighted_kappa_ci=(0.0, 0.4), flags=())


def test_assert_robustness_complete_raises_when_declared_spec_missing():
    from engine.cli.pipeline import assert_robustness_complete

    class M:
        robustness_specs = ("poisson_flat", "hierarchical_pooling")

    spread = RobustnessSpread(primary=_spec("negative_binomial_per_stratum"),
                              robustness=(_spec("poisson_flat"),))  # missing hierarchical
    with pytest.raises(ValueError, match="hierarchical_pooling"):
        assert_robustness_complete(M(), spread)


def test_specresult_carries_optional_heterogeneous_fields():
    s = SpecResult(spec_name="hierarchical_pooling", weighted_kappa_median=0.2,
                   weighted_kappa_ci=(0.0, 0.4), flags=(), sigma_u=2.1,
                   extra_rankings={"incidence": ("LLM09", "LLM02")})
    assert s.sigma_u == 2.1
    assert s.extra_rankings["incidence"] == ("LLM09", "LLM02")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_robustness_wiring.py -v`
Expected: FAIL — `assert_robustness_complete` does not exist; `SpecResult` has no `sigma_u`/`extra_rankings`.

- [ ] **Step 3: Write minimal implementation**

Extend `SpecResult` in `engine/decide/robustness_multiplicity.py`:
```python
@dataclass(frozen=True, slots=True)
class SpecResult:
    """Concordance result for one spec (primary or robustness)."""

    spec_name: str
    weighted_kappa_median: float | None
    weighted_kappa_ci: tuple[float, float] | None
    flags: tuple[FlagFinding, ...]
    sigma_u: float | None = None  # populated by hierarchical spec (Plan 8b)
    extra_rankings: dict[str, tuple[str, ...]] | None = None  # e.g. PL worth, incidence
```
Add to `engine/cli/pipeline.py`:
```python
def assert_robustness_complete(manifest: object, spread: "RobustnessSpread") -> None:
    """Refuse a report whose declared robustness specs were not all run (Plan 8a, SD4)."""
    declared = set(getattr(manifest, "robustness_specs", ()))
    present = {s.spec_name for s in spread.robustness}
    missing = declared - present
    if missing:
        raise ValueError(
            f"declared robustness specs not run: {sorted(missing)}"
        )
```
In the report-assembly section of `pipeline.py`, replace the tautological diff call and `robustness=None`:
```python
        from engine.model.robustness import run_robustness_inference
        from engine.decide.robustness_multiplicity import RobustnessSpread, SpecResult

        primary_spec_result = SpecResult(
            spec_name=manifest.primary_spec,
            weighted_kappa_median=concordance.weighted_kappa_median,
            weighted_kappa_ci=concordance.weighted_kappa_ci,
            flags=concordance.flags,
        )
        robustness_results: list[SpecResult] = []
        for spec_name in manifest.robustness_specs:
            r_inf = run_robustness_inference(
                manifest=manifest, spec_name=spec_name,
                measurable_entries=measurable_entries, strata=strata,
                observed_counts=observed_counts, stratum_sizes=stratum_sizes,
                calibration=calibration, overlap=overlap,
            )
            r_conc = compute_concordance(
                r_inf, vote_posterior, tier_boundaries,
                manifest.flag_threshold_tau, measurable_count, total_count,
                manifest.meaningful_kappa_n, manifest.measurability_minimum,
                entry_strata=entry_strata, stratum_sizes=stratum_sizes,
            )
            robustness_results.append(SpecResult(
                spec_name=spec_name,
                weighted_kappa_median=r_conc.weighted_kappa_median,
                weighted_kappa_ci=r_conc.weighted_kappa_ci,
                flags=r_conc.flags,
            ))
        spread = RobustnessSpread(primary=primary_spec_result,
                                  robustness=tuple(robustness_results))
        assert_robustness_complete(manifest, spread)

        prereg_diff = compute_prereg_diff(
            prereg_primary_spec=manifest.primary_spec,
            actual_primary_spec=manifest.primary_spec,  # executed == declared (primary unchanged)
            prereg_flag_tau=manifest.flag_threshold_tau,
            actual_flag_tau=manifest.flag_threshold_tau,
            prereg_measurability_min=manifest.measurability_minimum,
            actual_measurability_min=manifest.measurability_minimum,
        )
```
and set `robustness=spread` in the `ReportInputs(...)` constructor (replacing `robustness=None`).

Add a model-identity assertion to `tests/proofs/test_two_cycle_parity.py` (so a silent fallback to the wrong model fails the proof):
```python
    # Plan 8a: the executed primary spec must match the declared manifest spec.
    assert s1["primary_spec"] == "negative_binomial_per_stratum"
    assert s1["primary_spec"] == s2["primary_spec"]
```
(Add `"primary_spec": manifest.primary_spec` to the synthetic `summary.json` writer in `engine/cli/synthetic.py` if not already present.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_robustness_wiring.py -v`
Expected: PASS.
Then: `uv run pytest tests/unit -k robustness -v` and `uv run pytest tests/proofs/test_two_cycle_parity.py -v` — Expected: PASS (the parity test is `@pytest.mark.slow`; run with `-m slow` if needed).

- [ ] **Step 5: Commit**

```bash
git add engine/decide/robustness_multiplicity.py engine/cli/pipeline.py engine/cli/synthetic.py tests/unit/test_robustness_wiring.py tests/proofs/test_two_cycle_parity.py
git commit -m "feat(decide): wire robustness specs into the report behind a completeness gate

run_robustness_inference was defined but never called. The pipeline now runs
each declared robustness spec, assembles a RobustnessSpread, and refuses a
report missing a declared spec. SpecResult carries optional sigma_u and
extra_rankings for the hierarchical/PL specs added in Plans 8b/8c."
```

---

## Self-Review

**1. Spec coverage (8a's slice of RARR):**
- §5.1 recall fix → Task 1 ✓
- §5.9 manifest schema-version + goldset_hash → Task 2 ✓
- §5.9 provenance binding (classifier-label hash, goldset_hash in bundle) → Task 3 ✓
- §2.1/§5.4 λ·size ranking → Task 4 ✓ (baseline recompute under λ·size is exercised in Plan 8e's cycle run, which reuses this estimator)
- §5.4 live W → Task 5 ✓
- §5.5/§5.8 robustness mechanism + wiring + completeness gate + drift-diff/parity → Task 6 ✓
- Out of 8a scope (later plans): hierarchical model + ESS-gate parameterization + σ_u persistence (8b), tie-aware PL (8b), oracle + Merkle gate (8c), bake-off + security (8d), cycle run + report + power statement + curation-dir move + real snapshot_hash population (8e). Noted, not gaps.

**2. Placeholder scan:** No "TBD"/"add error handling"/"similar to Task N". Every code step shows real code. One intentional implementer judgment is flagged in Task 5 (locating the in-scope `gold` object in the executor) and Task 3 (reusing the existing `tally` hash variable) — both name the exact existing symbol to use.

**3. Type consistency:** `RecallTally(true_positives, false_negatives, total_in_sample)` matches `tally.py:18-22`. `BetaPosterior(alpha, beta)` matches `beta.py`. `GoldRecallLabel(incident_id, true_entry_ids, classifier_entry_id, source)` matches `gold_schema.py`. `OverlapWeights(weights=...)` matches `overlap.py`. `SpecResult`/`RobustnessSpread` field names match `robustness_multiplicity.py`. `compute_concordance` new params (`entry_strata`, `stratum_sizes`) are threaded consistently through Tasks 4 and 6. `ReproductionBundle` new field `goldset_hash` is positioned before `provenance` and used consistently in `to_json`/`read`.

---

## Execution Handoff

This is **Plan 8a of the 8a–8e roadmap** (engine correctness foundations; no RunPod). On completion it should be merged before Plan 8b (new robustness models) begins, since 8b/8c plug into Task 6's spread and Task 2's schema-versioned manifest.
