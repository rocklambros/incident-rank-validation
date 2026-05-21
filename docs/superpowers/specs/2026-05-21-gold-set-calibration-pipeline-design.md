# Plan 4: Gold-Set Calibration Pipeline — Design Spec

## Goal

Build the gold-set calibration pipeline for the OWASP LLM Top 10 2026 incident-rank validation cycle. The pipeline produces per-entry, per-stratum Beta posteriors for the Stage-1 classifier's precision and recall, which feed into the Plan 5 Bayesian inference model. It uses a two-frame sampling approach (precision frame + recall frame), supports single-coder non-publishable mode, and emits provenance-linked artifacts at every stage.

## Constraints Inherited from Plans 1-3

### Pre-registration discipline

- The frozen rubric (hash `2383f39825e4949251d514df49d1fed586874714aa9396fa24f3c0774534b120`, 20 entries) is the classification authority.
- The manifest lock (`engine/prereg/lock.py`) hash-locks all analysis parameters before any concordance number exists (HANDOFF §6 control 1).
- Vote-blindness (HANDOFF §6 control 2) is maintained — no access to `2026/polling/` or vote results.
- Rubric hash match is verified at classify time via `require_rubric_hash_match()` in `engine/prereg/gates.py`.

### Existing stubs to promote

| Stub | File | Plan 4 action |
|------|------|---------------|
| `Sampler` protocol | `engine/calibrate/sampler.py:17-24` | **Breaking change**: redesign for two-frame sampling |
| `StratifiedSampler` | `engine/calibrate/sampler.py:27-39` | Replace with real implementation |
| `cross_validate_calibration()` | `engine/calibrate/cv.py:21-26` | Implement with new signature |
| `CVResult` | `engine/calibrate/cv.py:13-18` | Extend with `interpretation` and `min_per_fold` fields |
| `classify_stub()` | `engine/classify/stub.py:67-111` | Kept for synthetic cycles; real Stage-1 built alongside |
| `BetaPosterior.from_counts()` | `engine/calibrate/beta.py:37-45` | Used as-is |
| `Calibration` | `engine/calibrate/beta.py:48-53` | Used as-is |

### Existing proof tests that must remain green

| Test | File | Constraint |
|------|------|------------|
| Never-falsely-low | `tests/proofs/test_never_falsely_low.py` | Wide posteriors or DiagnosticsFailure for low-count/unknown-recall entries |
| Frame-blind gate | `tests/proofs/test_frame_blind_gate.py` | Frame-blind entries excluded from inference |
| Calibrate shape | `tests/unit/test_calibrate.py` | Protocol conformance (signature changes must update these) |

### Single-coder non-publishable mode

Rock codes solo. No inter-rater reliability computed. Results are non-publishable per HANDOFF §4. The batch format includes `coder_id` to support future multi-coder runs without discarding Rock's coding. The waiver is documented in the manifest and the report.

### Corpus characteristics

- 7,714 total incidents: security=7,350, ai-harm=364
- 907 bare-LLM03 singletons (contamination in security stratum)
- 2,652 multi-label incidents
- 10 legacy entries have native labels (LLM01-LLM10)
- 10 new entries (NEW-\*, ROLL-\*) have zero native labels

---

## Architecture: Staged Pipeline with Provenance Chain

Six CLI commands, each producing a provenance-linked artifact:

```
classify → sample → code → tally → calibrate → cv-stability
```

### Stage provenance

Every CLI stage writes a provenance record binding its inputs to its outputs:

```python
@dataclass(frozen=True, slots=True)
class StageProvenance:
    stage_name: str
    manifest_lock_hash: str
    input_hashes: dict[str, str]
    output_hash: str
    timestamp: str  # ISO 8601
    engine_version: str
```

Each subsequent stage reads the previous stage's provenance and verifies `input_hashes` match. If any intermediate artifact was regenerated without re-running downstream stages, the next stage refuses to proceed. This prevents stale intermediate artifacts from propagating silently through the pipeline.

The `output_hash` covers the full serialized output (including incident text for batch files), not just identifiers. This ensures that any modification to batch content between stages is detected by the provenance chain.

### Classifier iteration requires a new cycle

If calibration reveals the classifier is inadequate, revising the classifier requires a new pre-registration cycle: new `classifier_rule_hash`, new manifest, new lock. No mid-cycle classifier changes. Each classifier version is a distinct pre-registration.

### Overlap weights are pre-registered

Overlap weights, like classifier rules and the confidence threshold, are pre-registered parameters locked in the manifest. Changes to overlap weights require a new cycle.

### Synthetic coding path for testing

A synthetic coding function `code_synthetic(sample_results, incidents)` fills batch labels from `native_labels` for testing purposes. It enables end-to-end pipeline testing (classify → sample → code_synthetic → tally → calibrate → cv-stability) without manual coding. The synthetic coder uses `coder_id: "synthetic"` and results are non-publishable. This parallels `classify_stub()` in `engine/classify/stub.py` — a test-only path through the manual stage.

The synthetic coding path is the primary guard against the 40-hour coding investment being wasted on a pipeline bug discovered after coding is complete.

---

## Stage 1: Classify

### Deterministic keyword/indicator classifier

The Stage-1 classifier is built from the frozen rubric's `positive_indicators` and `negative_indicators`. For each incident, it scans the text for indicator matches and assigns entry labels with a confidence score based on indicator hit ratio.

```python
@dataclass(frozen=True, slots=True)
class EntryClassifierRule:
    entry_id: str
    positive_patterns: tuple[str, ...]
    negative_patterns: tuple[str, ...]
    confidence_threshold: float

@dataclass(frozen=True, slots=True)
class ClassifierRules:
    rules_by_entry: dict[str, EntryClassifierRule]
    rule_hash: str  # SHA-256 of canonical JSON of all rules
```

**Indicator matching semantics**: case-insensitive substring search. For each indicator string P and incident text T, a match occurs when `P.lower()` is found in `T.lower()`. No regex, no word-boundary constraints. This definition is locked by the `classifier_rule_hash`.

Confidence formula: `confidence = max(0, positive_hits - negative_hits) / len(positive_patterns)`. An incident is assigned to an entry when `confidence >= confidence_threshold` (default 0.3, pre-registered as `confidence_threshold` in `PreregManifest`). Negative-pattern hits suppress, not veto — a high positive-hit count can overcome negative signals.

The classifier produces a `ClassificationResult` (existing schema in `engine/classify/stub.py:38-64`) with the real `classifier_rule_hash` derived from the rules, not the hardcoded stub value.

### Classifier rule hash gate

A new gate `require_classifier_rule_hash_match(manifest, actual_hash)` is added to `engine/prereg/gates.py`. The classify CLI command calls this gate before running, verifying that the classifier rules match what was pre-registered in the manifest. This closes the integrity gap where classifier rules could change after pre-registration without detection (HANDOFF §6 control 11(b)).

### Adapter update: 20 entries

`_PROVISIONAL_2025_ENTRIES` in `engine/adapters/genai_agentic.py:38-55` is replaced with all 20 entries from the frozen rubric:

| Entry | frame_blind | Rationale |
|-------|------------|-----------|
| LLM01, LLM02, LLM03, LLM05, LLM06, LLM07, LLM09 | False | Corpus has native labels; sampling frame observes them |
| LLM04, LLM08, LLM10 | True | Already frame_blind in Plan 1 — corpus cannot observe |
| NEW-PMP, NEW-MTIE, NEW-MA, NEW-ITSCD, NEW-WLA, NEW-MSDA | False | Not frame-blind (corpus CAN contain these incidents), but classifier-blind (zero native labels = no training signal). They enter calibration via the recall frame only. |
| ROLL-CMSB, ROLL-LAPTF, ROLL-SICG, ROLL-CFAS | False | Rolled into parents; classifier assigns parent label, rollup sub-test checks child indicators |

### Coverage gap disclosure

The 10 new/rollup entries have zero native labels in the corpus:
- **10 legacy entries**: calibrated via both precision and recall frames
- **6 new standalone entries**: calibrated via recall frame only; precision frame empty; posteriors will be wide
- **4 rollup entries**: calibrated via rollup sub-test within parent's precision frame

This is disclosed in the calibration diagnostic output and the final report.

---

## Stage 2: Sample (Two-Frame Design)

### Sampler protocol redesign

The current `Sampler.draw()` signature (`engine/calibrate/sampler.py:18-24`) is incompatible with two-frame sampling. Breaking change:

```python
class SampleFrame(Enum):
    PRECISION = "precision"
    RECALL = "recall"

@dataclass(frozen=True, slots=True)
class SampleRequest:
    frame: SampleFrame
    entry_id: str | None       # required for PRECISION, None for RECALL
    stratum: str | None        # optional stratum filter
    n: int                     # target sample size

@dataclass(frozen=True, slots=True)
class SampleResult:
    incidents: tuple[IncidentRecord, ...]
    request: SampleRequest
    actual_n: int
    sample_hash: str           # SHA-256 of sorted incident IDs

@runtime_checkable
class Sampler(Protocol):
    def draw(
        self,
        request: SampleRequest,
        incidents: list[IncidentRecord],
        seed: int,
    ) -> SampleResult:
        ...
```

### Target sample sizes

Target posterior widths and derived minimum gold-set sizes:

- **Target**: 90% credible interval width < 0.30 for recall posteriors of high-count entries (LLM05, LLM09, LLM03, LLM01)
- **Derivation**: Beta(a, b) with 90% CI width ~ 0.30 requires a+b ≈ 40 (approximately 40 coded labels per entry per stratum in the precision frame)
- **Precision frame**: target 40 incidents per entry per stratum for entries with native labels. Actual size is `min(40, classifier_positive_count_for_entry_in_stratum)`. If classifier-positive count < 20, use census (all classifier-positive incidents). Budget: up to 10 entries × 2 strata × 40 = 800 codings.
- **Recall frame**: 200 incidents total (100 per stratum), coded for all 20 entries.
- **Total coding budget**: ~1,000 incident codings.
- **Estimated coding time**: ~2-3 minutes per incident × 1,000 ≈ 33-50 hours.

### Contamination stratum protocol

The 907 bare-LLM03 singletons remain in the security stratum (not a separate stratum). In the LLM03 precision frame, contaminated singletons will be sampled proportionally. The coder codes them normally. The Beta posteriors for LLM03/security will naturally reflect the contamination rate. The calibration diagnostic flags LLM03 if its precision posterior is anomalous.

### ai-harm stratum small-sample handling

The ai-harm stratum has only 364 incidents. If a stratum has fewer than 20 classifier-positive incidents for an entry, the precision-frame sample for that entry+stratum is the entire set of classifier-positive incidents (census, not sample). Wider posteriors for ai-harm entries are disclosed in the calibration-adequacy report.

---

## Stage 3: Code (Batch Files)

### Batch file format with integrity header

```json
{
  "batch_header": {
    "cycle_id": "2026",
    "batch_id": "precision-LLM01-security-001",
    "frame": "precision",
    "entry_id": "LLM01",
    "stratum": "security",
    "sample_hash": "<SHA-256 of sorted incident IDs>",
    "rubric_hash": "2383f39825e4949251d514df49d1fed586874714aa9396fa24f3c0774534b120",
    "manifest_lock_hash": "<from manifest.lock>",
    "coder_id": "rock-lambros",
    "generated_at": "2026-06-01T10:00:00-06:00"
  },
  "incidents": [
    {
      "incident_id": "GA-00123",
      "text": "<incident text>",
      "labels": null,
      "rollup_sub_labels": null,
      "notes": null
    }
  ]
}
```

The coder fills in `labels` (list of entry_ids) and optional `notes`.

**Recall-frame coding checklist**: recall-frame batches include a `coding_checklist` field in the batch header containing all 20 entry names and short descriptions from the rubric. This reduces cognitive load during the 20-way multi-label classification task by providing a structured reference directly in the batch file. Precision-frame batches do not need this (the framing question is binary: "does this incident match [entry_id]?").

The tally stage verifies:
- `sample_hash` matches the sample stage's output hash
- `rubric_hash` matches the manifest
- `manifest_lock_hash` matches the lock file
- `coder_id` is present and non-empty
- No incidents were added or removed (incident IDs match the sample)

### Coding correction protocol

If the coder discovers an error after initial coding, the correction is recorded as an amendment:

```json
{
  "incident_id": "GA-00123",
  "labels": ["LLM01", "LLM06"],
  "notes": "corrected from [LLM01] — missed LLM06 co-occurrence",
  "amendment": {
    "original_labels": ["LLM01"],
    "correction_reason": "missed co-occurrence with LLM06",
    "corrected_at": "2026-06-02T14:30:00-06:00"
  }
}
```

The tally stage uses the latest labels. Amendments are preserved for audit. The post-hoc register records all amendments.

### Batch validation rules

The tally stage validates every coded batch file before counting. Validation collects all errors across all batches and reports them as a structured error list (not fail-fast):

| Field state | Meaning | Tally behavior |
|-------------|---------|----------------|
| `labels: null` | Uncoded — coder has not filled in this incident | Skip with warning; incident excluded from counts |
| `labels: []` | Coded as "no match" — incident does not match the entry | Precision frame: count as false positive. Recall frame: count as true negative for this entry |
| `labels: ["LLM01", ...]` | Coded labels | Validated against rubric `entry_ids`; unknown IDs raise `ValueError` with file + incident context |
| `rollup_sub_labels: [...]` | Rollup child labels | Validated against rollup entry IDs only; non-rollup IDs raise `ValueError` |
| Duplicate labels | Coder listed same entry twice | Deduplicated silently |
| `amendment.original_labels` | Correction chain | Must match the incident's labels from the previous version (if amendment chain exists); mismatch raises `ValueError` |

Error behavior: the tally stage scans all batches, collects all validation errors, and emits a structured error report before failing. This lets the coder fix all errors in one pass rather than iterating on individual batch files.

### Batch re-coding policy

If a pipeline error invalidates a sample (changing `sample_hash`), all coded batches for that sample are invalidated and must be re-coded. If the sample stage is re-run with the same seed and inputs, producing the same `sample_hash`, existing coded batches remain valid. The provenance chain enforces this: the tally stage rejects coded batches whose `sample_hash` doesn't match the current sample provenance.

### Single-coder waiver

Single-coder mode: no inter-rater reliability computed. Non-publishable per HANDOFF §4. When external coders are recruited, their batches are added alongside Rock's with distinct `coder_id` values, and inter-rater reliability is computed across all coders.

---

## Stage 4: Tally

Aggregates coded labels into per-entry per-stratum counts, separated by frame:

```python
@dataclass(frozen=True, slots=True)
class PrecisionTally:
    true_positives: int
    false_positives: int
    total: int

@dataclass(frozen=True, slots=True)
class RecallTally:
    true_positives: int
    false_negatives: int
    total_in_sample: int

@dataclass(frozen=True, slots=True)
class TallyResult:
    precision_counts: dict[tuple[str, str], PrecisionTally]
    recall_counts: dict[tuple[str, str], RecallTally]
    total_coded: int
    amendments_applied: int
```

---

## Stage 5: Calibrate

Computes per-entry per-stratum Beta posteriors from tally counts using `BetaPosterior.from_counts()` (`engine/calibrate/beta.py:37-45`):

- **Precision**: `Beta(TP + prior_a, FP + prior_b)`
- **Recall**: `Beta(TP_recall + prior_a, FN + prior_b)`

The `Calibration` dataclass (`engine/calibrate/beta.py:48-53`) is populated with the computed posteriors.

### Calibration-adequacy diagnostic

After computing all posteriors, the calibrate stage emits a diagnostic report. This is **disclosure, not a gate** — the pipeline continues regardless.

```python
@dataclass(frozen=True, slots=True)
class EntryCalibrationReport:
    entry_id: str
    has_precision_data: bool
    has_recall_data: bool
    precision_ci_width: float | None
    recall_ci_width: float | None
    recall_sample_size: int       # recall-frame labels for this entry
    precision_sample_size: int    # precision-frame labels for this entry
    min_fold_count: int
    flag: str     # "adequate", "wide", "no-data"
    reason: str   # structured reason for the flag value

@dataclass(frozen=True, slots=True)
class CalibrationDiagnostic:
    entries_with_both_frames: int
    entries_recall_only: int
    entries_no_data: int
    entry_reports: dict[str, EntryCalibrationReport]
```

**Flag reason values** (the `reason` field distinguishes root cause from symptom):
- `"adequate"` — both frames have sufficient data, CI widths within target
- `"wide: small-sample (n=N)"` — posteriors are wide due to small gold-set count, not due to genuinely poor classifier performance
- `"wide: recall-frame-only"` — no precision-frame data (new/rollup entry); recall posterior only
- `"no-data: no-classifier-positives"` — classifier produced zero positives for this entry+stratum; precision frame empty
- `"no-data: frame-blind"` — entry is frame-blind; excluded from calibration by design

This distinction is critical: "insufficient sample to measure recall" and "measured low recall from adequate sample" are different methodological statements with different implications for the inference model and the report.

Each entry's diagnostic includes `min_fold_count` — the minimum number of coded labels per fold. If `min_fold_count < 5`, the CV fold variance is annotated as "unstable — fewer than 5 labels per fold."

---

## Stage 6: CV Stability

k=5 cross-validation per HANDOFF §6 control 11(c). **This is a transparency disclosure, not a quality gate.** High fold variance does not block the pipeline.

```python
@dataclass(frozen=True, slots=True)
class CVResult:
    n_folds: int
    fold_variances: dict[tuple[str, str], float]
    interpretation: dict[tuple[str, str], str]
    min_per_fold: dict[tuple[str, str], int]
```

Interpretation thresholds:
- fold_variance < 0.01: "stable"
- 0.01 ≤ fold_variance < 0.05: "moderate"
- fold_variance ≥ 0.05: "unstable — interpret with caution"

---

## Output Artifact Paths

Each stage writes its output to the cycle directory:

| Stage | Output path | Format |
|-------|------------|--------|
| classify | `cycles/2026/calibration/classifications.json` | `ClassificationResult` as JSON |
| sample | `cycles/2026/calibration/samples/` | One `SampleResult` JSON per batch |
| code | `cycles/2026/calibration/coded/` | Coder-filled batch JSON files |
| tally | `cycles/2026/calibration/tally.json` | `TallyResult` as JSON |
| calibrate | `cycles/2026/calibration/calibration.json` + `calibration_diagnostic.json` | `Calibration` + `CalibrationDiagnostic` |
| cv-stability | `cycles/2026/calibration/cv_result.json` | `CVResult` as JSON |

Each stage also writes `cycles/2026/calibration/<stage>_provenance.json` (the `StageProvenance` record).

## Rollup Sub-Test

Rollup candidates (ROLL-CMSB, ROLL-LAPTF, ROLL-SICG, ROLL-CFAS) do not get their own precision-frame sample. Instead, within the parent entry's precision frame, the coder is asked: "Does this incident specifically involve [rollup child concept]?" For example, in the LLM01 precision-frame batch, each incident has an additional field `rollup_sub_labels: list[str] | null` where the coder can assign `ROLL-CMSB` if the incident involves cross-modal safety bypass specifically. The tally stage counts rollup sub-labels separately. The rollup sub-test produces its own Beta posterior for the child entry, conditioned on the parent entry's precision frame.

Rollup precision posteriors are **conditional on the parent** — they measure P(rollup correctly identified | incident is parent-positive), not unconditional precision. FPs from outside the parent's domain are already counted in the parent entry's FP rate and do not affect the rollup sub-test.

---

## Residual Risk Register

| Risk | Status | Rationale |
|------|--------|-----------|
| Git timestamp forgery (signoff.py:37-43) | Accepted | Discipline-based control per REVIEWERS.md:56. Mechanical check catches accidental mismatch. Cryptographic timestamping is overkill for OWASP working group audience. |
| Inference FP term semantic mismatch (inference.py:177-181) | Documented; fix deferred to Plan 5 | The inference model's FP leakage term uses `true_rate * (1-precision)`. Standard formulation is `true_rate * recall * (1-precision) / precision`. These differ by factor `recall/precision`. With current sparse overlap matrix (single entry: LLM05→LLM03 at 0.2), practical impact is negligible. **Hard gate**: the overlap matrix MUST NOT be expanded beyond its current single non-zero entry until this formula is corrected in `inference.py`. Plan 4 calibration outputs are the first to exercise this code path with non-trivial precision values. |
| Recall frame insufficient for rare entries | Accepted | Censoring module (`censoring.py:72-82`) excludes entries with mean recall < 0.1 from ranked inference. CalibrationDiagnostic's `reason` field now distinguishes "insufficient sample" from "measured low recall." Doubling recall-frame sample to 400 would add ~20 hours coding for marginal value (confirming what the coverage gap analysis already shows). |
| Single-coder non-publishable | Accepted | Deliberate project decision per HANDOFF §4. Pipeline ready for multi-coder via `coder_id`. |
| Recall-frame cognitive asymmetry | Partially mitigated | Inherent to two-frame methodology. Mitigated by recall-frame coding checklist in batch files (20 entry names + descriptions). Rock's rubric familiarity further reduces risk. |

All other premortem findings (first + second premortem) have been mitigated in this design.

---

## Premortem Finding Traceability

### First premortem (design phase)

| Finding | Severity | Remediation | Spec section |
|---------|----------|-------------|--------------|
| F1.1 | CRITICAL | R11: Coverage gap disclosure | Stage 1 — Coverage gap disclosure |
| F1.2 | HIGH | Census sampling + disclosure | Stage 2 — ai-harm small-sample handling |
| F1.3 | HIGH | R4: Adapter → 20 entries | Stage 1 — Adapter update |
| F1.4 | LOW | R5: coder_id + waiver | Stage 3 — Single-coder waiver |
| F1.5 | MEDIUM | R7: Code normally, flag in diagnostic | Stage 2 — Contamination stratum protocol |
| F2.1 | MEDIUM | min_fold_count annotation | Stage 5 — Calibration-adequacy diagnostic |
| F2.2 | HIGH | R1: New Sampler protocol | Stage 2 — Sampler protocol redesign |
| F2.4 | MEDIUM | R6: Target widths + sizes | Stage 2 — Target sample sizes |
| F3.2 | HIGH | R2: Classifier rule hash gate | Stage 1 — Classifier rule hash gate |
| F3.3 | MEDIUM | R5: Integrity header | Stage 3 — Batch file format |
| F3.5 | LOW | Accepted — documented | Residual risk register |
| F4.3 | HIGH | R3: StageProvenance | Architecture — Stage provenance |
| F4.4 | HIGH | R4: Replace provisional entries | Stage 1 — Adapter update |
| F4.5 | MEDIUM | R9: Amendment protocol | Stage 3 — Coding correction protocol |
| F4.6 | MEDIUM | R6: Budget specified | Stage 2 — Target sample sizes |
| F5.1 | MEDIUM | R8: CalibrationDiagnostic | Stage 5 — Calibration-adequacy diagnostic |
| F5.2 | MEDIUM | R5: Batch header coder_id | Stage 3 — Batch file format |
| F5.3 | MEDIUM | R10: New cycle required | Architecture — Classifier iteration |
| F5.5 | LOW | Disclosure-only + interpretation | Stage 6 — CV Stability |
| F5.6 | MEDIUM | R11: Explicit disclosure | Stage 1 — Coverage gap disclosure |

### Second premortem (spec review)

| Finding | Severity | Remediation | Spec section |
|---------|----------|-------------|--------------|
| 2-F1.1 | CRITICAL | Add `confidence_threshold` to PreregManifest | Stage 1 — Classify (manifest field) |
| 2-F1.2 | HIGH | Diagnostic `reason` + `recall_sample_size` fields | Stage 5 — Calibration-adequacy diagnostic |
| 2-F1.3 | HIGH | Define indicator match semantics | Stage 1 — Classify (matching definition) |
| 2-F1.4 | MEDIUM | Coding instructions (not spec) | — |
| 2-F1.5 | MEDIUM | Recall-frame coding checklist in batches | Stage 3 — Batch file format |
| 2-F1.6 | MEDIUM | Conservative by design (empty overlap for new entries) | Architecture — Overlap weights |
| 2-F2.1 | HIGH | Document FP term mismatch + hard gate on overlap expansion | Residual risk register |
| 2-F2.4 | MEDIUM | Clarify rollup precision is conditional on parent | Rollup Sub-Test |
| 2-F3.1 | MEDIUM | Clarify output_hash covers full content | Architecture — Stage provenance |
| 2-F4.1 | HIGH | Synthetic coding path (code_synthetic) | Architecture — Synthetic coding path |
| 2-F4.2 | HIGH | Batch validation rules table | Stage 3 — Batch validation rules |
| 2-F4.5 | MEDIUM | Batch re-coding policy | Stage 3 — Batch re-coding policy |
| 2-F5.1 | MEDIUM | CalibrationDiagnostic `reason` field | Stage 5 — Calibration-adequacy diagnostic |
| 2-F5.3 | MEDIUM | Overlap weights are pre-registered | Architecture — Overlap weights |
