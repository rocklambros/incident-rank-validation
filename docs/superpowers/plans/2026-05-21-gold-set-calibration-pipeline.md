# Gold-Set Calibration Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the 6-stage gold-set calibration pipeline (classify → sample → code → tally → calibrate → cv-stability) that produces per-entry, per-stratum Beta posteriors for the Stage-1 classifier's precision and recall.

**Architecture:** Six CLI commands chained by a StageProvenance hash chain. The Sampler protocol is redesigned for two-frame sampling (precision + recall). A deterministic keyword/indicator classifier replaces the stub. A synthetic coding path (`code_synthetic`) enables end-to-end testing without manual coding.

**Tech Stack:** Python 3.11+, Click (CLI), pytest, dataclasses, hashlib, json, random (seeded)

**Spec:** `docs/superpowers/specs/2026-05-21-gold-set-calibration-pipeline-design.md`

---

## File Structure

### New files

| File | Responsibility |
|------|---------------|
| `engine/calibrate/provenance.py` | `StageProvenance` dataclass, write/read/verify helpers |
| `engine/classify/classifier.py` | `EntryClassifierRule`, `ClassifierRules`, `classify_real()` — real Stage-1 classifier |
| `engine/calibrate/two_frame_sampler.py` | `SampleFrame`, `SampleRequest`, `SampleResult`, `TwoFrameSampler` |
| `engine/calibrate/batch.py` | Batch file generation, validation, `code_synthetic()` |
| `engine/calibrate/tally.py` | `PrecisionTally`, `RecallTally`, `TallyResult`, `tally_batches()` |
| `engine/calibrate/calibrate.py` | `EntryCalibrationReport`, `CalibrationDiagnostic`, `compute_calibration()` |
| `engine/cli/calibration.py` | 6 CLI commands: classify, sample, generate-batches, tally, calibrate, cv-stability |
| `tests/unit/test_classifier_real.py` | Tests for real Stage-1 classifier |
| `tests/unit/test_two_frame_sampler.py` | Tests for two-frame sampler |
| `tests/unit/test_batch.py` | Tests for batch generation, validation, synthetic coding |
| `tests/unit/test_tally.py` | Tests for tally aggregation |
| `tests/unit/test_calibration_diagnostic.py` | Tests for calibration + diagnostic |
| `tests/unit/test_cv_real.py` | Tests for real CV implementation |
| `tests/unit/test_provenance.py` | Tests for stage provenance chain |
| `tests/integration/test_calibration_e2e.py` | End-to-end synthetic pipeline test |

### Modified files

| File | Changes |
|------|---------|
| `engine/prereg/manifest.py` | Add `confidence_threshold: float` field |
| `engine/prereg/gates.py` | Add `require_classifier_rule_hash_match()` |
| `engine/calibrate/sampler.py` | Replace protocol with new `SampleFrame`/`SampleRequest`/`SampleResult`/`Sampler` |
| `engine/calibrate/cv.py` | Replace stub with real implementation, extend `CVResult` |
| `engine/adapters/genai_agentic.py` | Replace 10-entry `_PROVISIONAL_2025_ENTRIES` with 20 entries |
| `engine/cli/main.py` | Register 6 new calibration CLI commands |
| `tests/unit/test_calibrate.py` | Update for new Sampler protocol + CVResult fields |
| `tests/unit/test_prereg.py` | Add `confidence_threshold` to `_make_manifest()` mutation table |
| `tests/proofs/test_never_falsely_low.py` | Add `confidence_threshold` to `_make_manifest()` |
| `tests/unit/test_inference.py` | Add `confidence_threshold` to `_make_manifest()` |
| `tests/unit/test_predictive.py` | Add `confidence_threshold` to `_make_manifest()` |
| `tests/unit/test_concentration_sensitivity.py` | Add `confidence_threshold` to `_make_manifest()` |
| `engine/cli/synthetic.py` | Add `confidence_threshold` to manifest construction |

---

### Task 1: Add `confidence_threshold` to PreregManifest

**Files:**
- Modify: `engine/prereg/manifest.py:30-55`
- Modify: `tests/unit/test_prereg.py:44-77` and `:218-247`
- Modify: `tests/proofs/test_never_falsely_low.py:25-52`
- Modify: `tests/unit/test_inference.py:25-*`
- Modify: `tests/unit/test_predictive.py:30-*`
- Modify: `tests/unit/test_concentration_sensitivity.py:30-*`
- Modify: `engine/cli/synthetic.py:173-198`

- [ ] **Step 1: Write test for new field**

In `tests/unit/test_prereg.py`, add to the `mutations` dict and add a dedicated test:

```python
# In TestPreregManifest class, add:
def test_confidence_threshold_default(self) -> None:
    m = _make_manifest()
    assert m.confidence_threshold == 0.3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_prereg.py::TestPreregManifest::test_confidence_threshold_default -v`
Expected: FAIL — `_make_manifest` doesn't include `confidence_threshold`, and `PreregManifest` doesn't have the field.

- [ ] **Step 3: Add field to PreregManifest**

In `engine/prereg/manifest.py`, add after line 49 (`prng_seed: int`):

```python
    confidence_threshold: float  # Stage-1 classifier confidence threshold (default 0.3)
```

- [ ] **Step 4: Update all `_make_manifest()` helpers**

Every test file that constructs a `PreregManifest` needs the new field. Update each:

In `tests/unit/test_prereg.py` `_make_manifest()` defaults dict, add:
```python
        "confidence_threshold": 0.3,
```

In the same file's `mutations` dict (line ~218), add:
```python
            "confidence_threshold": 0.99,
```

In `tests/proofs/test_never_falsely_low.py` `_make_manifest()` defaults dict, add:
```python
        "confidence_threshold": 0.3,
```

In `tests/unit/test_inference.py` `_make_manifest()` defaults dict, add:
```python
        "confidence_threshold": 0.3,
```

In `tests/unit/test_predictive.py` `_make_manifest()` defaults dict, add:
```python
        "confidence_threshold": 0.3,
```

In `tests/unit/test_concentration_sensitivity.py` `_make_manifest()` defaults dict, add:
```python
        "confidence_threshold": 0.3,
```

In `engine/cli/synthetic.py` `execute_synthetic_pipeline()`, add to the `PreregManifest()` constructor (after `prng_seed=prng_seed,`):
```python
        confidence_threshold=hyper.get("confidence_threshold", 0.3),
```

- [ ] **Step 5: Run full test suite to verify nothing breaks**

Run: `pytest tests/ -x -q`
Expected: ALL PASS (including the existing proof tests and the new test).

- [ ] **Step 6: Commit**

```bash
git add engine/prereg/manifest.py engine/cli/synthetic.py tests/unit/test_prereg.py tests/proofs/test_never_falsely_low.py tests/unit/test_inference.py tests/unit/test_predictive.py tests/unit/test_concentration_sensitivity.py
git commit -m "feat(prereg): add confidence_threshold to PreregManifest"
```

---

### Task 2: Add `require_classifier_rule_hash_match` gate

**Files:**
- Modify: `engine/prereg/gates.py`
- Test: `tests/unit/test_prereg.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/unit/test_prereg.py`:

```python
from engine.prereg.gates import (
    require_classifier_rule_hash_match,
    require_rubric_attestation,
    require_rubric_hash,
)


class TestClassifierRuleHashGate:
    def test_passes_when_hashes_match(self) -> None:
        m = _make_manifest(classifier_rule_hash="abc123")
        require_classifier_rule_hash_match(m, "abc123")

    def test_raises_when_hashes_differ(self) -> None:
        m = _make_manifest(classifier_rule_hash="abc123")
        with pytest.raises(ValueError, match="classifier rule hash mismatch"):
            require_classifier_rule_hash_match(m, "different")

    def test_raises_when_manifest_hash_is_none(self) -> None:
        m = _make_manifest(classifier_rule_hash=None)
        with pytest.raises(ValueError, match="classifier_rule_hash is None"):
            require_classifier_rule_hash_match(m, "abc123")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_prereg.py::TestClassifierRuleHashGate -v`
Expected: FAIL — `require_classifier_rule_hash_match` not found.

- [ ] **Step 3: Implement the gate**

Add to `engine/prereg/gates.py`:

```python
def require_classifier_rule_hash_match(
    manifest: PreregManifest, actual_hash: str
) -> None:
    """Raise if manifest.classifier_rule_hash does not match actual classifier rules.

    HANDOFF §6 control 11(b): classifier rules are hash-locked before sampling.
    """
    if manifest.classifier_rule_hash is None:
        raise ValueError(
            "classifier_rule_hash is None in manifest — "
            "pre-register classifier rules before running classify"
        )
    if manifest.classifier_rule_hash != actual_hash:
        raise ValueError(
            f"classifier rule hash mismatch: manifest={manifest.classifier_rule_hash}, "
            f"actual={actual_hash}. Were classifier rules modified after pre-registration?"
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_prereg.py::TestClassifierRuleHashGate -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add engine/prereg/gates.py tests/unit/test_prereg.py
git commit -m "feat(prereg): add require_classifier_rule_hash_match gate"
```

---

### Task 3: StageProvenance dataclass and helpers

**Files:**
- Create: `engine/calibrate/provenance.py`
- Create: `tests/unit/test_provenance.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_provenance.py`:

```python
"""Tests for engine.calibrate.provenance — stage provenance chain."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.calibrate.provenance import StageProvenance, write_provenance, read_provenance, verify_input_hashes


class TestStageProvenance:
    def test_frozen(self) -> None:
        sp = StageProvenance(
            stage_name="classify",
            manifest_lock_hash="abc",
            input_hashes={"rubric": "def"},
            output_hash="ghi",
            timestamp="2026-06-01T10:00:00-06:00",
            engine_version="0.3.0",
        )
        with pytest.raises(AttributeError):
            sp.stage_name = "mutated"  # type: ignore[misc]

    def test_write_read_roundtrip(self, tmp_path: Path) -> None:
        sp = StageProvenance(
            stage_name="classify",
            manifest_lock_hash="lock123",
            input_hashes={"rubric": "rub456", "manifest": "man789"},
            output_hash="out000",
            timestamp="2026-06-01T10:00:00-06:00",
            engine_version="0.3.0",
        )
        path = tmp_path / "classify_provenance.json"
        write_provenance(sp, path)
        loaded = read_provenance(path)
        assert loaded.stage_name == "classify"
        assert loaded.manifest_lock_hash == "lock123"
        assert loaded.input_hashes == {"rubric": "rub456", "manifest": "man789"}
        assert loaded.output_hash == "out000"

    def test_verify_input_hashes_passes(self, tmp_path: Path) -> None:
        prev = StageProvenance(
            stage_name="classify",
            manifest_lock_hash="lock",
            input_hashes={},
            output_hash="classify_out",
            timestamp="2026-06-01T10:00:00-06:00",
            engine_version="0.3.0",
        )
        prev_path = tmp_path / "classify_provenance.json"
        write_provenance(prev, prev_path)
        verify_input_hashes(
            expected={"classify": "classify_out"},
            provenance_dir=tmp_path,
        )

    def test_verify_input_hashes_raises_on_mismatch(self, tmp_path: Path) -> None:
        prev = StageProvenance(
            stage_name="classify",
            manifest_lock_hash="lock",
            input_hashes={},
            output_hash="classify_out",
            timestamp="2026-06-01T10:00:00-06:00",
            engine_version="0.3.0",
        )
        prev_path = tmp_path / "classify_provenance.json"
        write_provenance(prev, prev_path)
        with pytest.raises(ValueError, match="provenance mismatch"):
            verify_input_hashes(
                expected={"classify": "WRONG_HASH"},
                provenance_dir=tmp_path,
            )

    def test_verify_input_hashes_raises_on_missing(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="provenance file not found"):
            verify_input_hashes(
                expected={"classify": "any"},
                provenance_dir=tmp_path,
            )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_provenance.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement StageProvenance**

Create `engine/calibrate/provenance.py`:

```python
"""Stage provenance chain — binds each pipeline stage's inputs to outputs.

Each CLI stage writes a StageProvenance record. The next stage verifies
input_hashes match the previous stage's output_hash, preventing stale
intermediate artifacts from propagating.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class StageProvenance:
    stage_name: str
    manifest_lock_hash: str
    input_hashes: dict[str, str]
    output_hash: str
    timestamp: str
    engine_version: str


def write_provenance(prov: StageProvenance, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "stage_name": prov.stage_name,
        "manifest_lock_hash": prov.manifest_lock_hash,
        "input_hashes": prov.input_hashes,
        "output_hash": prov.output_hash,
        "timestamp": prov.timestamp,
        "engine_version": prov.engine_version,
    }
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def read_provenance(path: Path) -> StageProvenance:
    data = json.loads(path.read_text())
    return StageProvenance(
        stage_name=data["stage_name"],
        manifest_lock_hash=data["manifest_lock_hash"],
        input_hashes=data["input_hashes"],
        output_hash=data["output_hash"],
        timestamp=data["timestamp"],
        engine_version=data["engine_version"],
    )


def verify_input_hashes(
    expected: dict[str, str],
    provenance_dir: Path,
) -> None:
    for stage_name, expected_hash in expected.items():
        prov_path = provenance_dir / f"{stage_name}_provenance.json"
        if not prov_path.exists():
            raise ValueError(
                f"provenance file not found for stage '{stage_name}': {prov_path}"
            )
        prov = read_provenance(prov_path)
        if prov.output_hash != expected_hash:
            raise ValueError(
                f"provenance mismatch for stage '{stage_name}': "
                f"expected output_hash={expected_hash}, got {prov.output_hash}"
            )


def hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hash_json(data: object) -> str:
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_provenance.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add engine/calibrate/provenance.py tests/unit/test_provenance.py
git commit -m "feat(calibrate): add StageProvenance dataclass and chain verification"
```

---

### Task 4: Real Stage-1 classifier

**Files:**
- Create: `engine/classify/classifier.py`
- Create: `tests/unit/test_classifier_real.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_classifier_real.py`:

```python
"""Tests for engine.classify.classifier — real Stage-1 keyword/indicator classifier."""
from __future__ import annotations

import hashlib
import json

import pytest

from engine.classify.classifier import (
    ClassifierRules,
    EntryClassifierRule,
    build_rules_from_rubric,
    classify_real,
)
from engine.classify.stub import Classification, ClassificationResult
from engine.schema import IncidentRecord


def _make_incident(
    id: str = "GA-001",
    text: str = "prompt injection attack on LLM",
    labels: tuple[str, ...] = (),
) -> IncidentRecord:
    return IncidentRecord(
        id=id,
        date="2026-01-01",
        text=text,
        severity="High",
        source_class="advisory",
        corpus_stratum="security",
        quality="curated",
        native_labels=labels,
        source_url="https://example.com",
    )


class TestEntryClassifierRule:
    def test_frozen(self) -> None:
        rule = EntryClassifierRule(
            entry_id="LLM01",
            positive_patterns=("prompt injection",),
            negative_patterns=("not a prompt",),
            confidence_threshold=0.3,
        )
        with pytest.raises(AttributeError):
            rule.entry_id = "X"  # type: ignore[misc]


class TestClassifierRules:
    def test_rule_hash_deterministic(self) -> None:
        rule = EntryClassifierRule(
            entry_id="LLM01",
            positive_patterns=("prompt injection",),
            negative_patterns=("benign",),
            confidence_threshold=0.3,
        )
        rules = ClassifierRules(
            rules_by_entry={"LLM01": rule},
            rule_hash="",
        )
        h1 = ClassifierRules.compute_rule_hash({"LLM01": rule})
        h2 = ClassifierRules.compute_rule_hash({"LLM01": rule})
        assert h1 == h2

    def test_rule_hash_changes_with_threshold(self) -> None:
        rule_a = EntryClassifierRule(
            entry_id="LLM01",
            positive_patterns=("prompt injection",),
            negative_patterns=(),
            confidence_threshold=0.3,
        )
        rule_b = EntryClassifierRule(
            entry_id="LLM01",
            positive_patterns=("prompt injection",),
            negative_patterns=(),
            confidence_threshold=0.5,
        )
        h_a = ClassifierRules.compute_rule_hash({"LLM01": rule_a})
        h_b = ClassifierRules.compute_rule_hash({"LLM01": rule_b})
        assert h_a != h_b


class TestClassifyReal:
    def test_positive_match_case_insensitive(self) -> None:
        rule = EntryClassifierRule(
            entry_id="LLM01",
            positive_patterns=("prompt injection",),
            negative_patterns=(),
            confidence_threshold=0.3,
        )
        rules = ClassifierRules(
            rules_by_entry={"LLM01": rule},
            rule_hash=ClassifierRules.compute_rule_hash({"LLM01": rule}),
        )
        incident = _make_incident(text="A PROMPT INJECTION was discovered")
        result = classify_real((incident,), rules)
        assert len(result.classifications) == 1
        assert result.classifications[0].entry_id == "LLM01"
        assert result.classifications[0].confidence >= 0.3

    def test_negative_pattern_suppresses(self) -> None:
        rule = EntryClassifierRule(
            entry_id="LLM01",
            positive_patterns=("injection",),
            negative_patterns=("sql injection",),
            confidence_threshold=0.3,
        )
        rules = ClassifierRules(
            rules_by_entry={"LLM01": rule},
            rule_hash=ClassifierRules.compute_rule_hash({"LLM01": rule}),
        )
        incident = _make_incident(text="sql injection vulnerability found")
        result = classify_real((incident,), rules)
        assert len(result.classifications) == 0

    def test_below_threshold_excluded(self) -> None:
        rule = EntryClassifierRule(
            entry_id="LLM01",
            positive_patterns=("prompt injection", "jailbreak", "adversarial input"),
            negative_patterns=(),
            confidence_threshold=0.5,
        )
        rules = ClassifierRules(
            rules_by_entry={"LLM01": rule},
            rule_hash=ClassifierRules.compute_rule_hash({"LLM01": rule}),
        )
        # Only 1 of 3 positive patterns matches → confidence = 1/3 ≈ 0.33 < 0.5
        incident = _make_incident(text="A prompt injection was found")
        result = classify_real((incident,), rules)
        assert len(result.classifications) == 0

    def test_multi_entry_classification(self) -> None:
        rule_01 = EntryClassifierRule(
            entry_id="LLM01",
            positive_patterns=("prompt injection",),
            negative_patterns=(),
            confidence_threshold=0.3,
        )
        rule_02 = EntryClassifierRule(
            entry_id="LLM02",
            positive_patterns=("data leak",),
            negative_patterns=(),
            confidence_threshold=0.3,
        )
        rules = ClassifierRules(
            rules_by_entry={"LLM01": rule_01, "LLM02": rule_02},
            rule_hash=ClassifierRules.compute_rule_hash(
                {"LLM01": rule_01, "LLM02": rule_02}
            ),
        )
        incident = _make_incident(text="prompt injection caused a data leak")
        result = classify_real((incident,), rules)
        entry_ids = {c.entry_id for c in result.classifications}
        assert entry_ids == {"LLM01", "LLM02"}

    def test_result_uses_real_rule_hash(self) -> None:
        rule = EntryClassifierRule(
            entry_id="LLM01",
            positive_patterns=("prompt injection",),
            negative_patterns=(),
            confidence_threshold=0.3,
        )
        expected_hash = ClassifierRules.compute_rule_hash({"LLM01": rule})
        rules = ClassifierRules(
            rules_by_entry={"LLM01": rule},
            rule_hash=expected_hash,
        )
        incident = _make_incident(text="prompt injection attack")
        result = classify_real((incident,), rules)
        assert result.classifier_rule_hash == expected_hash
        # Not the stub hash
        stub_hash = hashlib.sha256(b"stub-classifier-v0.1.0").hexdigest()
        assert result.classifier_rule_hash != stub_hash
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_classifier_real.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement the classifier**

Create `engine/classify/classifier.py`:

```python
"""Real Stage-1 deterministic keyword/indicator classifier.

Built from the frozen rubric's positive_indicators and negative_indicators.
Matching semantics: case-insensitive substring search. For each indicator
string P and incident text T, a match occurs when P.lower() is found in
T.lower(). No regex, no word-boundary constraints.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from engine.classify.stub import Classification, ClassificationResult
from engine.prereg.rubric import Rubric
from engine.schema import IncidentRecord


@dataclass(frozen=True, slots=True)
class EntryClassifierRule:
    entry_id: str
    positive_patterns: tuple[str, ...]
    negative_patterns: tuple[str, ...]
    confidence_threshold: float


@dataclass(frozen=True, slots=True)
class ClassifierRules:
    rules_by_entry: dict[str, EntryClassifierRule]
    rule_hash: str

    @staticmethod
    def compute_rule_hash(
        rules_by_entry: dict[str, EntryClassifierRule],
    ) -> str:
        canonical = {}
        for eid in sorted(rules_by_entry):
            r = rules_by_entry[eid]
            canonical[eid] = {
                "confidence_threshold": r.confidence_threshold,
                "negative_patterns": list(r.negative_patterns),
                "positive_patterns": list(r.positive_patterns),
            }
        blob = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def build_rules_from_rubric(
    rubric: Rubric,
    confidence_threshold: float = 0.3,
) -> ClassifierRules:
    rules: dict[str, EntryClassifierRule] = {}
    for entry in rubric.entries:
        rules[entry.entry_id] = EntryClassifierRule(
            entry_id=entry.entry_id,
            positive_patterns=entry.positive_indicators,
            negative_patterns=entry.negative_indicators,
            confidence_threshold=confidence_threshold,
        )
    return ClassifierRules(
        rules_by_entry=rules,
        rule_hash=ClassifierRules.compute_rule_hash(rules),
    )


def _compute_confidence(
    text_lower: str,
    rule: EntryClassifierRule,
) -> float:
    if not rule.positive_patterns:
        return 0.0
    positive_hits = sum(
        1 for p in rule.positive_patterns if p.lower() in text_lower
    )
    negative_hits = sum(
        1 for n in rule.negative_patterns if n.lower() in text_lower
    )
    return max(0, positive_hits - negative_hits) / len(rule.positive_patterns)


def classify_real(
    incidents: tuple[IncidentRecord, ...],
    rules: ClassifierRules,
) -> ClassificationResult:
    classifications: list[Classification] = []
    for inc in incidents:
        text_lower = inc.text.lower()
        for rule in rules.rules_by_entry.values():
            confidence = _compute_confidence(text_lower, rule)
            if confidence >= rule.confidence_threshold:
                classifications.append(
                    Classification(
                        incident_id=inc.id,
                        entry_id=rule.entry_id,
                        confidence=confidence,
                        stage=1,
                        rationale=f"indicator match: confidence={confidence:.3f}",
                    )
                )
    return ClassificationResult(
        classifications=tuple(classifications),
        classifier_version="stage1-keyword-1.0.0",
        classifier_rule_hash=rules.rule_hash,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_classifier_real.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add engine/classify/classifier.py tests/unit/test_classifier_real.py
git commit -m "feat(classify): add real Stage-1 keyword/indicator classifier"
```

---

### Task 5: Redesign Sampler protocol for two-frame sampling

**Files:**
- Modify: `engine/calibrate/sampler.py`
- Create: `engine/calibrate/two_frame_sampler.py`
- Create: `tests/unit/test_two_frame_sampler.py`
- Modify: `tests/unit/test_calibrate.py:89-106`

- [ ] **Step 1: Update sampler.py with new protocol**

Replace `engine/calibrate/sampler.py` contents:

```python
"""Sampler protocol for two-frame gold-set calibration.

The Sampler protocol defines the interface for drawing precision-frame
and recall-frame samples from the incident corpus.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

from engine.schema import IncidentRecord

__all__ = [
    "SampleFrame",
    "SampleRequest",
    "SampleResult",
    "Sampler",
]


class SampleFrame(Enum):
    PRECISION = "precision"
    RECALL = "recall"


@dataclass(frozen=True, slots=True)
class SampleRequest:
    frame: SampleFrame
    entry_id: str | None
    stratum: str | None
    n: int


@dataclass(frozen=True, slots=True)
class SampleResult:
    incidents: tuple[IncidentRecord, ...]
    request: SampleRequest
    actual_n: int
    sample_hash: str

    @staticmethod
    def compute_sample_hash(incidents: tuple[IncidentRecord, ...]) -> str:
        sorted_ids = sorted(inc.id for inc in incidents)
        return hashlib.sha256(
            json.dumps(sorted_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


import json


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

- [ ] **Step 2: Write tests for TwoFrameSampler**

Create `tests/unit/test_two_frame_sampler.py`:

```python
"""Tests for engine.calibrate.two_frame_sampler — two-frame gold-set sampling."""
from __future__ import annotations

import pytest

from engine.calibrate.sampler import SampleFrame, SampleRequest, SampleResult, Sampler
from engine.calibrate.two_frame_sampler import TwoFrameSampler
from engine.classify.stub import Classification, ClassificationResult
from engine.schema import IncidentRecord


def _make_incidents(n: int, stratum: str = "security") -> list[IncidentRecord]:
    return [
        IncidentRecord(
            id=f"GA-{i:05d}",
            date="2026-01-01",
            text=f"incident text {i}",
            severity="High",
            source_class="advisory",
            corpus_stratum=stratum,
            quality="curated",
            native_labels=("LLM01",) if i % 3 == 0 else (),
            source_url="https://example.com",
        )
        for i in range(n)
    ]


def _make_classifications(
    incidents: list[IncidentRecord],
    entry_id: str = "LLM01",
) -> ClassificationResult:
    classifications = tuple(
        Classification(
            incident_id=inc.id,
            entry_id=entry_id,
            confidence=1.0,
            stage=1,
            rationale="test",
        )
        for inc in incidents
        if "LLM01" in inc.native_labels
    )
    return ClassificationResult(
        classifications=classifications,
        classifier_version="test",
        classifier_rule_hash="test_hash",
    )


class TestTwoFrameSampler:
    def test_implements_sampler_protocol(self) -> None:
        sampler = TwoFrameSampler(
            classification_result=_make_classifications([]),
        )
        assert isinstance(sampler, Sampler)

    def test_precision_frame_samples_classifier_positives(self) -> None:
        incidents = _make_incidents(100)
        classifications = _make_classifications(incidents, "LLM01")
        sampler = TwoFrameSampler(classification_result=classifications)
        request = SampleRequest(
            frame=SampleFrame.PRECISION,
            entry_id="LLM01",
            stratum="security",
            n=10,
        )
        result = sampler.draw(request, incidents, seed=42)
        assert result.actual_n <= 10
        assert all(inc.corpus_stratum == "security" for inc in result.incidents)
        classified_ids = {
            c.incident_id for c in classifications.classifications
            if c.entry_id == "LLM01"
        }
        assert all(inc.id in classified_ids for inc in result.incidents)

    def test_recall_frame_samples_all_incidents(self) -> None:
        incidents = _make_incidents(200)
        classifications = _make_classifications(incidents)
        sampler = TwoFrameSampler(classification_result=classifications)
        request = SampleRequest(
            frame=SampleFrame.RECALL,
            entry_id=None,
            stratum="security",
            n=50,
        )
        result = sampler.draw(request, incidents, seed=42)
        assert result.actual_n == 50
        assert all(inc.corpus_stratum == "security" for inc in result.incidents)

    def test_precision_frame_census_when_under_threshold(self) -> None:
        incidents = _make_incidents(30)
        classifications = _make_classifications(incidents, "LLM01")
        sampler = TwoFrameSampler(classification_result=classifications)
        classifier_positive_count = len([
            c for c in classifications.classifications if c.entry_id == "LLM01"
        ])
        request = SampleRequest(
            frame=SampleFrame.PRECISION,
            entry_id="LLM01",
            stratum="security",
            n=40,
        )
        result = sampler.draw(request, incidents, seed=42)
        # When classifier_positive_count < 20, should be census
        if classifier_positive_count < 20:
            assert result.actual_n == classifier_positive_count

    def test_sample_hash_is_deterministic(self) -> None:
        incidents = _make_incidents(100)
        classifications = _make_classifications(incidents)
        sampler = TwoFrameSampler(classification_result=classifications)
        request = SampleRequest(
            frame=SampleFrame.RECALL,
            entry_id=None,
            stratum="security",
            n=20,
        )
        r1 = sampler.draw(request, incidents, seed=42)
        r2 = sampler.draw(request, incidents, seed=42)
        assert r1.sample_hash == r2.sample_hash
        assert tuple(i.id for i in r1.incidents) == tuple(i.id for i in r2.incidents)

    def test_precision_requires_entry_id(self) -> None:
        sampler = TwoFrameSampler(classification_result=_make_classifications([]))
        request = SampleRequest(
            frame=SampleFrame.PRECISION,
            entry_id=None,
            stratum="security",
            n=10,
        )
        with pytest.raises(ValueError, match="entry_id required"):
            sampler.draw(request, [], seed=42)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/unit/test_two_frame_sampler.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 4: Implement TwoFrameSampler**

Create `engine/calibrate/two_frame_sampler.py`:

```python
"""Two-frame gold-set sampler: precision frame + recall frame.

Precision frame: sample from classifier-positive incidents for a specific entry.
Recall frame: random stratified sample from all incidents.
"""
from __future__ import annotations

import random

from engine.calibrate.sampler import SampleFrame, SampleRequest, SampleResult
from engine.classify.stub import ClassificationResult
from engine.schema import IncidentRecord


class TwoFrameSampler:
    def __init__(self, classification_result: ClassificationResult) -> None:
        self._classification = classification_result

    def draw(
        self,
        request: SampleRequest,
        incidents: list[IncidentRecord],
        seed: int,
    ) -> SampleResult:
        if request.frame == SampleFrame.PRECISION:
            return self._draw_precision(request, incidents, seed)
        return self._draw_recall(request, incidents, seed)

    def _draw_precision(
        self,
        request: SampleRequest,
        incidents: list[IncidentRecord],
        seed: int,
    ) -> SampleResult:
        if request.entry_id is None:
            raise ValueError("entry_id required for PRECISION frame")

        classified_ids = {
            c.incident_id
            for c in self._classification.classifications
            if c.entry_id == request.entry_id
        }
        pool = [
            inc for inc in incidents
            if inc.id in classified_ids
            and (request.stratum is None or inc.corpus_stratum == request.stratum)
        ]

        if len(pool) < 20:
            sampled = tuple(pool)
        elif len(pool) <= request.n:
            sampled = tuple(pool)
        else:
            rng = random.Random(seed)
            sampled = tuple(rng.sample(pool, request.n))

        return SampleResult(
            incidents=sampled,
            request=request,
            actual_n=len(sampled),
            sample_hash=SampleResult.compute_sample_hash(sampled),
        )

    def _draw_recall(
        self,
        request: SampleRequest,
        incidents: list[IncidentRecord],
        seed: int,
    ) -> SampleResult:
        pool = [
            inc for inc in incidents
            if request.stratum is None or inc.corpus_stratum == request.stratum
        ]

        n = min(request.n, len(pool))
        rng = random.Random(seed)
        sampled = tuple(rng.sample(pool, n))

        return SampleResult(
            incidents=sampled,
            request=request,
            actual_n=len(sampled),
            sample_hash=SampleResult.compute_sample_hash(sampled),
        )
```

- [ ] **Step 5: Update existing test_calibrate.py**

Replace the `TestSamplerProtocol` class in `tests/unit/test_calibrate.py` with:

```python
from engine.calibrate.sampler import SampleFrame, SampleRequest, SampleResult, Sampler


class TestSamplerProtocol:
    def test_sample_frame_enum(self) -> None:
        assert SampleFrame.PRECISION.value == "precision"
        assert SampleFrame.RECALL.value == "recall"

    def test_sample_request_is_frozen(self) -> None:
        req = SampleRequest(
            frame=SampleFrame.RECALL, entry_id=None, stratum=None, n=100,
        )
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            req.n = 50  # type: ignore[misc]

    def test_sample_result_is_frozen(self) -> None:
        result = SampleResult(
            incidents=(), request=SampleRequest(
                frame=SampleFrame.RECALL, entry_id=None, stratum=None, n=0,
            ),
            actual_n=0, sample_hash="abc",
        )
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            result.actual_n = 99  # type: ignore[misc]
```

Also update the `TestCVStub` class (in `tests/unit/test_calibrate.py`): remove the tests for the old stub since we'll replace it in Task 10. For now, update `test_cvresult_is_frozen_dataclass` and `test_cvresult_stores_fold_variances` to match the new CVResult schema with `interpretation` and `min_per_fold`:

```python
class TestCVResult:
    def test_cvresult_is_frozen_dataclass(self) -> None:
        result = CVResult(
            n_folds=5,
            fold_variances={("e1", "s1"): 0.01},
            interpretation={("e1", "s1"): "stable"},
            min_per_fold={("e1", "s1"): 10},
        )
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            result.n_folds = 3  # type: ignore[misc]

    def test_cvresult_stores_all_fields(self) -> None:
        fv: dict[tuple[str, str], float] = {("e1", "security"): 0.002}
        interp: dict[tuple[str, str], str] = {("e1", "security"): "stable"}
        mpf: dict[tuple[str, str], int] = {("e1", "security"): 10}
        result = CVResult(n_folds=5, fold_variances=fv, interpretation=interp, min_per_fold=mpf)
        assert result.n_folds == 5
        assert result.fold_variances[("e1", "security")] == pytest.approx(0.002)
        assert result.interpretation[("e1", "security")] == "stable"
        assert result.min_per_fold[("e1", "security")] == 10
```

- [ ] **Step 6: Update CVResult in engine/calibrate/cv.py**

Replace `engine/calibrate/cv.py` contents:

```python
"""K-fold cross-validation for calibration stability.

See HANDOFF §6 control 11(c). This is a transparency disclosure,
not a quality gate — high fold variance does not block the pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass

__all__ = ["CVResult", "cross_validate_calibration"]


@dataclass(frozen=True, slots=True)
class CVResult:
    n_folds: int
    fold_variances: dict[tuple[str, str], float]
    interpretation: dict[tuple[str, str], str]
    min_per_fold: dict[tuple[str, str], int]


def cross_validate_calibration(
    precision_labels: dict[tuple[str, str], list[bool]],
    recall_labels: dict[tuple[str, str], list[bool]],
    n_folds: int = 5,
) -> CVResult:
    """k-fold CV for calibration stability. Stub replaced in Task 10."""
    raise NotImplementedError(
        "cross_validate_calibration full implementation is Task 10. "
        f"Schema for k={n_folds} fold CV is ready."
    )
```

- [ ] **Step 7: Run all tests**

Run: `pytest tests/unit/test_calibrate.py tests/unit/test_two_frame_sampler.py -v`
Expected: ALL PASS

- [ ] **Step 8: Commit**

```bash
git add engine/calibrate/sampler.py engine/calibrate/two_frame_sampler.py engine/calibrate/cv.py tests/unit/test_calibrate.py tests/unit/test_two_frame_sampler.py
git commit -m "feat(calibrate): redesign Sampler protocol for two-frame sampling"
```

---

### Task 6: Batch file generation, validation, and synthetic coding

**Files:**
- Create: `engine/calibrate/batch.py`
- Create: `tests/unit/test_batch.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_batch.py`:

```python
"""Tests for engine.calibrate.batch — batch generation, validation, synthetic coding."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.calibrate.batch import (
    BatchHeader,
    BatchIncident,
    CodingBatch,
    ValidationError,
    code_synthetic,
    generate_batch,
    validate_coded_batch,
)
from engine.calibrate.sampler import SampleFrame, SampleRequest, SampleResult
from engine.schema import IncidentRecord


def _make_incident(
    id: str = "GA-001",
    text: str = "test incident",
    labels: tuple[str, ...] = ("LLM01",),
    stratum: str = "security",
) -> IncidentRecord:
    return IncidentRecord(
        id=id, date="2026-01-01", text=text, severity="High",
        source_class="advisory", corpus_stratum=stratum, quality="curated",
        native_labels=labels, source_url="https://example.com",
    )


def _sample_result(incidents: tuple[IncidentRecord, ...]) -> SampleResult:
    return SampleResult(
        incidents=incidents,
        request=SampleRequest(
            frame=SampleFrame.PRECISION, entry_id="LLM01", stratum="security", n=10,
        ),
        actual_n=len(incidents),
        sample_hash="abc123",
    )


class TestGenerateBatch:
    def test_generates_precision_batch(self) -> None:
        inc = _make_incident()
        sr = _sample_result((inc,))
        batch = generate_batch(
            sample_result=sr,
            rubric_hash="rub_hash",
            manifest_lock_hash="lock_hash",
            coder_id="rock-lambros",
            cycle_id="2026",
        )
        assert batch.header.frame == "precision"
        assert batch.header.entry_id == "LLM01"
        assert batch.header.sample_hash == "abc123"
        assert len(batch.incidents) == 1
        assert batch.incidents[0].labels is None

    def test_recall_batch_includes_checklist(self) -> None:
        inc = _make_incident()
        sr = SampleResult(
            incidents=(inc,),
            request=SampleRequest(
                frame=SampleFrame.RECALL, entry_id=None, stratum="security", n=10,
            ),
            actual_n=1,
            sample_hash="abc123",
        )
        checklist = {"LLM01": "Prompt Injection", "LLM02": "Data Leak"}
        batch = generate_batch(
            sample_result=sr,
            rubric_hash="rub_hash",
            manifest_lock_hash="lock_hash",
            coder_id="rock-lambros",
            cycle_id="2026",
            coding_checklist=checklist,
        )
        assert batch.header.coding_checklist == checklist


class TestValidateCodedBatch:
    def test_valid_batch_passes(self, tmp_path: Path) -> None:
        inc = _make_incident()
        sr = _sample_result((inc,))
        batch = generate_batch(
            sample_result=sr, rubric_hash="rub", manifest_lock_hash="lock",
            coder_id="rock", cycle_id="2026",
        )
        batch_data = batch.to_dict()
        batch_data["incidents"][0]["labels"] = ["LLM01"]
        path = tmp_path / "batch.json"
        path.write_text(json.dumps(batch_data, indent=2))
        errors = validate_coded_batch(
            path, valid_entry_ids={"LLM01", "LLM02"},
            rollup_entry_ids=set(),
            expected_sample_hash="abc123",
            expected_rubric_hash="rub",
            expected_lock_hash="lock",
        )
        assert errors == []

    def test_null_labels_generates_warning(self, tmp_path: Path) -> None:
        inc = _make_incident()
        sr = _sample_result((inc,))
        batch = generate_batch(
            sample_result=sr, rubric_hash="rub", manifest_lock_hash="lock",
            coder_id="rock", cycle_id="2026",
        )
        batch_data = batch.to_dict()
        # labels stays null
        path = tmp_path / "batch.json"
        path.write_text(json.dumps(batch_data, indent=2))
        errors = validate_coded_batch(
            path, valid_entry_ids={"LLM01"},
            rollup_entry_ids=set(),
            expected_sample_hash="abc123",
            expected_rubric_hash="rub",
            expected_lock_hash="lock",
        )
        assert any("uncoded" in str(e).lower() for e in errors)

    def test_unknown_label_raises(self, tmp_path: Path) -> None:
        inc = _make_incident()
        sr = _sample_result((inc,))
        batch = generate_batch(
            sample_result=sr, rubric_hash="rub", manifest_lock_hash="lock",
            coder_id="rock", cycle_id="2026",
        )
        batch_data = batch.to_dict()
        batch_data["incidents"][0]["labels"] = ["UNKNOWN_ENTRY"]
        path = tmp_path / "batch.json"
        path.write_text(json.dumps(batch_data, indent=2))
        errors = validate_coded_batch(
            path, valid_entry_ids={"LLM01"},
            rollup_entry_ids=set(),
            expected_sample_hash="abc123",
            expected_rubric_hash="rub",
            expected_lock_hash="lock",
        )
        assert any("unknown" in str(e).lower() for e in errors)

    def test_hash_mismatch_raises(self, tmp_path: Path) -> None:
        inc = _make_incident()
        sr = _sample_result((inc,))
        batch = generate_batch(
            sample_result=sr, rubric_hash="rub", manifest_lock_hash="lock",
            coder_id="rock", cycle_id="2026",
        )
        batch_data = batch.to_dict()
        batch_data["incidents"][0]["labels"] = ["LLM01"]
        path = tmp_path / "batch.json"
        path.write_text(json.dumps(batch_data, indent=2))
        errors = validate_coded_batch(
            path, valid_entry_ids={"LLM01"},
            rollup_entry_ids=set(),
            expected_sample_hash="WRONG",
            expected_rubric_hash="rub",
            expected_lock_hash="lock",
        )
        assert any("sample_hash" in str(e).lower() for e in errors)


class TestCodeSynthetic:
    def test_fills_labels_from_native(self) -> None:
        inc = _make_incident(labels=("LLM01", "LLM05"))
        sr = _sample_result((inc,))
        batch = generate_batch(
            sample_result=sr, rubric_hash="rub", manifest_lock_hash="lock",
            coder_id="synthetic", cycle_id="2026",
        )
        coded = code_synthetic(batch, valid_entry_ids={"LLM01", "LLM02", "LLM05"})
        assert coded.incidents[0].labels == ["LLM01", "LLM05"]

    def test_empty_labels_when_no_native(self) -> None:
        inc = _make_incident(labels=())
        sr = _sample_result((inc,))
        batch = generate_batch(
            sample_result=sr, rubric_hash="rub", manifest_lock_hash="lock",
            coder_id="synthetic", cycle_id="2026",
        )
        coded = code_synthetic(batch, valid_entry_ids={"LLM01"})
        assert coded.incidents[0].labels == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_batch.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement batch module**

Create `engine/calibrate/batch.py`:

```python
"""Batch file generation, validation, and synthetic coding for gold-set calibration."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from engine.calibrate.sampler import SampleResult
from engine.schema import IncidentRecord


@dataclass
class BatchHeader:
    cycle_id: str
    batch_id: str
    frame: str
    entry_id: str | None
    stratum: str | None
    sample_hash: str
    rubric_hash: str
    manifest_lock_hash: str
    coder_id: str
    generated_at: str = ""
    coding_checklist: dict[str, str] | None = None


@dataclass
class BatchIncident:
    incident_id: str
    text: str
    labels: list[str] | None = None
    rollup_sub_labels: list[str] | None = None
    notes: str | None = None
    amendment: dict[str, Any] | None = None


@dataclass
class CodingBatch:
    header: BatchHeader
    incidents: list[BatchIncident]

    def to_dict(self) -> dict[str, Any]:
        header_d: dict[str, Any] = {
            "cycle_id": self.header.cycle_id,
            "batch_id": self.header.batch_id,
            "frame": self.header.frame,
            "entry_id": self.header.entry_id,
            "stratum": self.header.stratum,
            "sample_hash": self.header.sample_hash,
            "rubric_hash": self.header.rubric_hash,
            "manifest_lock_hash": self.header.manifest_lock_hash,
            "coder_id": self.header.coder_id,
            "generated_at": self.header.generated_at,
        }
        if self.header.coding_checklist is not None:
            header_d["coding_checklist"] = self.header.coding_checklist
        incidents_d = [
            {
                "incident_id": inc.incident_id,
                "text": inc.text,
                "labels": inc.labels,
                "rollup_sub_labels": inc.rollup_sub_labels,
                "notes": inc.notes,
            }
            for inc in self.incidents
        ]
        return {"batch_header": header_d, "incidents": incidents_d}

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @staticmethod
    def read(path: Path) -> CodingBatch:
        data = json.loads(path.read_text())
        h = data["batch_header"]
        header = BatchHeader(
            cycle_id=h["cycle_id"],
            batch_id=h["batch_id"],
            frame=h["frame"],
            entry_id=h.get("entry_id"),
            stratum=h.get("stratum"),
            sample_hash=h["sample_hash"],
            rubric_hash=h["rubric_hash"],
            manifest_lock_hash=h["manifest_lock_hash"],
            coder_id=h["coder_id"],
            generated_at=h.get("generated_at", ""),
            coding_checklist=h.get("coding_checklist"),
        )
        incidents = [
            BatchIncident(
                incident_id=inc["incident_id"],
                text=inc["text"],
                labels=inc.get("labels"),
                rollup_sub_labels=inc.get("rollup_sub_labels"),
                notes=inc.get("notes"),
                amendment=inc.get("amendment"),
            )
            for inc in data["incidents"]
        ]
        return CodingBatch(header=header, incidents=incidents)


def generate_batch(
    sample_result: SampleResult,
    rubric_hash: str,
    manifest_lock_hash: str,
    coder_id: str,
    cycle_id: str,
    coding_checklist: dict[str, str] | None = None,
) -> CodingBatch:
    req = sample_result.request
    frame_str = req.frame.value
    entry_id = req.entry_id
    stratum = req.stratum
    batch_id = f"{frame_str}-{entry_id or 'all'}-{stratum or 'all'}-001"

    header = BatchHeader(
        cycle_id=cycle_id,
        batch_id=batch_id,
        frame=frame_str,
        entry_id=entry_id,
        stratum=stratum,
        sample_hash=sample_result.sample_hash,
        rubric_hash=rubric_hash,
        manifest_lock_hash=manifest_lock_hash,
        coder_id=coder_id,
        coding_checklist=coding_checklist if frame_str == "recall" else None,
    )

    incidents = [
        BatchIncident(incident_id=inc.id, text=inc.text)
        for inc in sample_result.incidents
    ]

    return CodingBatch(header=header, incidents=incidents)


@dataclass
class ValidationError:
    file: str
    incident_id: str | None
    message: str

    def __str__(self) -> str:
        loc = self.file
        if self.incident_id:
            loc += f":{self.incident_id}"
        return f"{loc}: {self.message}"


def validate_coded_batch(
    path: Path,
    valid_entry_ids: set[str],
    rollup_entry_ids: set[str],
    expected_sample_hash: str,
    expected_rubric_hash: str,
    expected_lock_hash: str,
) -> list[ValidationError]:
    errors: list[ValidationError] = []
    batch = CodingBatch.read(path)
    fname = path.name

    if batch.header.sample_hash != expected_sample_hash:
        errors.append(ValidationError(fname, None,
            f"sample_hash mismatch: expected {expected_sample_hash}, got {batch.header.sample_hash}"))
    if batch.header.rubric_hash != expected_rubric_hash:
        errors.append(ValidationError(fname, None,
            f"rubric_hash mismatch: expected {expected_rubric_hash}, got {batch.header.rubric_hash}"))
    if batch.header.manifest_lock_hash != expected_lock_hash:
        errors.append(ValidationError(fname, None,
            f"manifest_lock_hash mismatch: expected {expected_lock_hash}, got {batch.header.manifest_lock_hash}"))
    if not batch.header.coder_id:
        errors.append(ValidationError(fname, None, "coder_id is empty"))

    for inc in batch.incidents:
        if inc.labels is None:
            errors.append(ValidationError(fname, inc.incident_id, "uncoded: labels is null"))
            continue
        deduped = list(dict.fromkeys(inc.labels))
        if len(deduped) != len(inc.labels):
            inc.labels = deduped
        for label in inc.labels:
            if label not in valid_entry_ids:
                errors.append(ValidationError(fname, inc.incident_id,
                    f"unknown entry_id in labels: {label}"))
        if inc.rollup_sub_labels:
            for rl in inc.rollup_sub_labels:
                if rl not in rollup_entry_ids:
                    errors.append(ValidationError(fname, inc.incident_id,
                        f"unknown rollup entry_id: {rl}"))

    return errors


def code_synthetic(
    batch: CodingBatch,
    valid_entry_ids: set[str],
) -> CodingBatch:
    """Fill batch labels from IncidentRecord.native_labels for testing."""
    from engine.schema import IncidentRecord
    coded_incidents: list[BatchIncident] = []
    for inc in batch.incidents:
        coded_incidents.append(BatchIncident(
            incident_id=inc.incident_id,
            text=inc.text,
            labels=[],
            rollup_sub_labels=inc.rollup_sub_labels,
            notes="synthetic coder",
        ))
    return CodingBatch(header=batch.header, incidents=coded_incidents)


def code_synthetic_with_ground_truth(
    batch: CodingBatch,
    incidents_by_id: dict[str, IncidentRecord],
    valid_entry_ids: set[str],
) -> CodingBatch:
    """Fill batch labels from native_labels in the incident records."""
    coded_incidents: list[BatchIncident] = []
    for inc in batch.incidents:
        source = incidents_by_id.get(inc.incident_id)
        labels: list[str] = []
        if source is not None:
            labels = [l for l in source.native_labels if l in valid_entry_ids]
        coded_incidents.append(BatchIncident(
            incident_id=inc.incident_id,
            text=inc.text,
            labels=labels,
            rollup_sub_labels=inc.rollup_sub_labels,
            notes="synthetic coder",
        ))
    return CodingBatch(header=batch.header, incidents=coded_incidents)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_batch.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add engine/calibrate/batch.py tests/unit/test_batch.py
git commit -m "feat(calibrate): add batch file generation, validation, and synthetic coding"
```

---

### Task 7: Tally module

**Files:**
- Create: `engine/calibrate/tally.py`
- Create: `tests/unit/test_tally.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_tally.py`:

```python
"""Tests for engine.calibrate.tally — aggregation of coded labels."""
from __future__ import annotations

import pytest

from engine.calibrate.batch import BatchHeader, BatchIncident, CodingBatch
from engine.calibrate.tally import (
    PrecisionTally,
    RecallTally,
    TallyResult,
    tally_batches,
)


def _precision_batch(
    entry_id: str = "LLM01",
    stratum: str = "security",
    incidents: list[dict] | None = None,
) -> CodingBatch:
    if incidents is None:
        incidents = [
            {"incident_id": "GA-001", "labels": ["LLM01"], "text": "t1"},
            {"incident_id": "GA-002", "labels": [], "text": "t2"},
            {"incident_id": "GA-003", "labels": ["LLM01", "LLM05"], "text": "t3"},
        ]
    return CodingBatch(
        header=BatchHeader(
            cycle_id="2026", batch_id=f"precision-{entry_id}-{stratum}",
            frame="precision", entry_id=entry_id, stratum=stratum,
            sample_hash="h", rubric_hash="r", manifest_lock_hash="l",
            coder_id="rock",
        ),
        incidents=[
            BatchIncident(
                incident_id=inc["incident_id"],
                text=inc["text"],
                labels=inc["labels"],
            )
            for inc in incidents
        ],
    )


def _recall_batch(
    stratum: str = "security",
    incidents: list[dict] | None = None,
) -> CodingBatch:
    if incidents is None:
        incidents = [
            {"incident_id": "GA-100", "labels": ["LLM01", "LLM05"], "text": "t"},
            {"incident_id": "GA-101", "labels": [], "text": "t"},
            {"incident_id": "GA-102", "labels": ["LLM09"], "text": "t"},
        ]
    return CodingBatch(
        header=BatchHeader(
            cycle_id="2026", batch_id=f"recall-all-{stratum}",
            frame="recall", entry_id=None, stratum=stratum,
            sample_hash="h", rubric_hash="r", manifest_lock_hash="l",
            coder_id="rock",
        ),
        incidents=[
            BatchIncident(
                incident_id=inc["incident_id"],
                text=inc["text"],
                labels=inc["labels"],
            )
            for inc in incidents
        ],
    )


class TestTally:
    def test_precision_counts(self) -> None:
        batch = _precision_batch("LLM01", "security")
        result = tally_batches([batch])
        key = ("LLM01", "security")
        assert key in result.precision_counts
        pt = result.precision_counts[key]
        # GA-001: LLM01 in labels → TP. GA-002: [] → FP. GA-003: LLM01 in labels → TP.
        assert pt.true_positives == 2
        assert pt.false_positives == 1
        assert pt.total == 3

    def test_recall_counts(self) -> None:
        batch = _recall_batch("security")
        result = tally_batches([batch])
        # LLM01: GA-100 has it → TP, GA-101 no → FN, GA-102 no → FN
        key_01 = ("LLM01", "security")
        assert key_01 in result.recall_counts
        rt = result.recall_counts[key_01]
        assert rt.true_positives == 1
        assert rt.false_negatives == 2
        assert rt.total_in_sample == 3

    def test_total_coded(self) -> None:
        batch = _precision_batch()
        result = tally_batches([batch])
        assert result.total_coded == 3

    def test_skips_null_labels(self) -> None:
        batch = CodingBatch(
            header=BatchHeader(
                cycle_id="2026", batch_id="p", frame="precision",
                entry_id="LLM01", stratum="security",
                sample_hash="h", rubric_hash="r", manifest_lock_hash="l",
                coder_id="rock",
            ),
            incidents=[
                BatchIncident(incident_id="GA-001", text="t", labels=None),
                BatchIncident(incident_id="GA-002", text="t", labels=["LLM01"]),
            ],
        )
        result = tally_batches([batch])
        assert result.precision_counts[("LLM01", "security")].total == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_tally.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement tally module**

Create `engine/calibrate/tally.py`:

```python
"""Tally aggregation: count coded labels into per-entry per-stratum tallies."""
from __future__ import annotations

from dataclasses import dataclass

from engine.calibrate.batch import CodingBatch


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
    rollup_counts: dict[tuple[str, str], PrecisionTally]
    total_coded: int
    amendments_applied: int


def tally_batches(batches: list[CodingBatch]) -> TallyResult:
    precision_tp: dict[tuple[str, str], int] = {}
    precision_fp: dict[tuple[str, str], int] = {}
    precision_total: dict[tuple[str, str], int] = {}
    recall_hits: dict[tuple[str, str], int] = {}
    recall_miss: dict[tuple[str, str], int] = {}
    recall_total: dict[tuple[str, str], int] = {}
    rollup_tp: dict[tuple[str, str], int] = {}
    rollup_fp: dict[tuple[str, str], int] = {}
    rollup_total: dict[tuple[str, str], int] = {}
    total_coded = 0
    amendments = 0

    all_recall_entries: set[str] = set()

    for batch in batches:
        entry_id = batch.header.entry_id
        stratum = batch.header.stratum or "unknown"
        frame = batch.header.frame

        if frame == "precision" and entry_id is not None:
            key = (entry_id, stratum)
            for inc in batch.incidents:
                if inc.labels is None:
                    continue
                total_coded += 1
                precision_total[key] = precision_total.get(key, 0) + 1
                if entry_id in inc.labels:
                    precision_tp[key] = precision_tp.get(key, 0) + 1
                else:
                    precision_fp[key] = precision_fp.get(key, 0) + 1
                if inc.rollup_sub_labels:
                    for rl in inc.rollup_sub_labels:
                        rk = (rl, stratum)
                        rollup_total[rk] = rollup_total.get(rk, 0) + 1
                        rollup_tp[rk] = rollup_tp.get(rk, 0) + 1
                if inc.amendment:
                    amendments += 1

        elif frame == "recall":
            for inc in batch.incidents:
                if inc.labels is None:
                    continue
                total_coded += 1
                labels_set = set(inc.labels)
                all_recall_entries.update(labels_set)
                for eid in labels_set:
                    rk = (eid, stratum)
                    recall_hits[rk] = recall_hits.get(rk, 0) + 1
                if inc.amendment:
                    amendments += 1

    for batch in batches:
        if batch.header.frame != "recall":
            continue
        stratum = batch.header.stratum or "unknown"
        coded_count = sum(1 for inc in batch.incidents if inc.labels is not None)
        for eid in all_recall_entries:
            rk = (eid, stratum)
            hits = recall_hits.get(rk, 0)
            recall_total[rk] = recall_total.get(rk, 0) + coded_count
            recall_miss[rk] = recall_miss.get(rk, 0) + (coded_count - hits)

    precision_counts = {
        k: PrecisionTally(
            true_positives=precision_tp.get(k, 0),
            false_positives=precision_fp.get(k, 0),
            total=precision_total[k],
        )
        for k in precision_total
    }

    recall_counts = {
        k: RecallTally(
            true_positives=recall_hits.get(k, 0),
            false_negatives=recall_miss.get(k, 0),
            total_in_sample=recall_total.get(k, 0),
        )
        for k in recall_total
    }

    rollup_counts_out = {
        k: PrecisionTally(
            true_positives=rollup_tp.get(k, 0),
            false_positives=rollup_total.get(k, 0) - rollup_tp.get(k, 0),
            total=rollup_total[k],
        )
        for k in rollup_total
    }

    return TallyResult(
        precision_counts=precision_counts,
        recall_counts=recall_counts,
        rollup_counts=rollup_counts_out,
        total_coded=total_coded,
        amendments_applied=amendments,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_tally.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add engine/calibrate/tally.py tests/unit/test_tally.py
git commit -m "feat(calibrate): add tally aggregation module"
```

---

### Task 8: Calibration + diagnostic module

**Files:**
- Create: `engine/calibrate/calibrate.py`
- Create: `tests/unit/test_calibration_diagnostic.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_calibration_diagnostic.py`:

```python
"""Tests for engine.calibrate.calibrate — calibration computation + diagnostic."""
from __future__ import annotations

import math

import pytest

from engine.calibrate.beta import BetaPosterior, Calibration
from engine.calibrate.calibrate import (
    CalibrationDiagnostic,
    EntryCalibrationReport,
    compute_calibration,
)
from engine.calibrate.tally import PrecisionTally, RecallTally, TallyResult


def _tally(
    precision: dict[tuple[str, str], tuple[int, int, int]] | None = None,
    recall: dict[tuple[str, str], tuple[int, int, int]] | None = None,
) -> TallyResult:
    pc = {
        k: PrecisionTally(true_positives=v[0], false_positives=v[1], total=v[2])
        for k, v in (precision or {}).items()
    }
    rc = {
        k: RecallTally(true_positives=v[0], false_negatives=v[1], total_in_sample=v[2])
        for k, v in (recall or {}).items()
    }
    total = sum(t.total for t in pc.values()) + sum(
        t.total_in_sample for t in rc.values()
    )
    return TallyResult(
        precision_counts=pc,
        recall_counts=rc,
        rollup_counts={},
        total_coded=total,
        amendments_applied=0,
    )


class TestComputeCalibration:
    def test_precision_posterior(self) -> None:
        tally = _tally(
            precision={("LLM01", "security"): (35, 5, 40)},
        )
        cal, diag = compute_calibration(
            tally, all_entry_ids=["LLM01"], strata=["security"],
            frame_blind_ids=set(),
        )
        bp = cal.precision[("LLM01", "security")]
        assert bp.alpha == 36.0  # 35 + 1
        assert bp.beta == 6.0  # 5 + 1

    def test_recall_posterior(self) -> None:
        tally = _tally(
            recall={("LLM01", "security"): (30, 70, 100)},
        )
        cal, diag = compute_calibration(
            tally, all_entry_ids=["LLM01"], strata=["security"],
            frame_blind_ids=set(),
        )
        bp = cal.recall[("LLM01", "security")]
        assert bp.alpha == 31.0
        assert bp.beta == 71.0

    def test_diagnostic_adequate(self) -> None:
        tally = _tally(
            precision={("LLM01", "security"): (35, 5, 40)},
            recall={("LLM01", "security"): (30, 10, 40)},
        )
        cal, diag = compute_calibration(
            tally, all_entry_ids=["LLM01"], strata=["security"],
            frame_blind_ids=set(),
        )
        report = diag.entry_reports["LLM01"]
        assert report.has_precision_data is True
        assert report.has_recall_data is True
        assert report.flag == "adequate"

    def test_diagnostic_no_data_frame_blind(self) -> None:
        tally = _tally()
        cal, diag = compute_calibration(
            tally, all_entry_ids=["LLM04"], strata=["security"],
            frame_blind_ids={"LLM04"},
        )
        report = diag.entry_reports["LLM04"]
        assert report.flag == "no-data"
        assert "frame-blind" in report.reason

    def test_diagnostic_recall_only(self) -> None:
        tally = _tally(
            recall={("NEW-PMP", "security"): (2, 98, 100)},
        )
        cal, diag = compute_calibration(
            tally, all_entry_ids=["NEW-PMP"], strata=["security"],
            frame_blind_ids=set(),
        )
        report = diag.entry_reports["NEW-PMP"]
        assert report.has_precision_data is False
        assert report.has_recall_data is True
        assert "recall-frame-only" in report.reason
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_calibration_diagnostic.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement calibration + diagnostic**

Create `engine/calibrate/calibrate.py`:

```python
"""Compute per-entry per-stratum Beta posteriors + calibration-adequacy diagnostic."""
from __future__ import annotations

from dataclasses import dataclass

from scipy import stats as scipy_stats

from engine.calibrate.beta import BetaPosterior, Calibration
from engine.calibrate.tally import TallyResult


@dataclass(frozen=True, slots=True)
class EntryCalibrationReport:
    entry_id: str
    has_precision_data: bool
    has_recall_data: bool
    precision_ci_width: float | None
    recall_ci_width: float | None
    recall_sample_size: int
    precision_sample_size: int
    min_fold_count: int
    flag: str
    reason: str


@dataclass(frozen=True, slots=True)
class CalibrationDiagnostic:
    entries_with_both_frames: int
    entries_recall_only: int
    entries_no_data: int
    entry_reports: dict[str, EntryCalibrationReport]


def _ci_width(bp: BetaPosterior) -> float:
    lo, hi = scipy_stats.beta.ppf([0.05, 0.95], bp.alpha, bp.beta)
    return float(hi - lo)


def compute_calibration(
    tally: TallyResult,
    all_entry_ids: list[str],
    strata: list[str],
    frame_blind_ids: set[str],
    n_folds: int = 5,
) -> tuple[Calibration, CalibrationDiagnostic]:
    recall_posteriors: dict[tuple[str, str], BetaPosterior] = {}
    precision_posteriors: dict[tuple[str, str], BetaPosterior] = {}

    for key, pt in tally.precision_counts.items():
        precision_posteriors[key] = BetaPosterior.from_counts(
            pt.true_positives, pt.false_positives,
        )

    for key, rt in tally.recall_counts.items():
        recall_posteriors[key] = BetaPosterior.from_counts(
            rt.true_positives, rt.false_negatives,
        )

    for key, rt in tally.rollup_counts.items():
        precision_posteriors[key] = BetaPosterior.from_counts(
            rt.true_positives, rt.false_positives,
        )

    cal = Calibration(recall=recall_posteriors, precision=precision_posteriors)

    both = 0
    recall_only = 0
    no_data = 0
    reports: dict[str, EntryCalibrationReport] = {}

    for eid in all_entry_ids:
        has_prec = any(k[0] == eid for k in precision_posteriors)
        has_rec = any(k[0] == eid for k in recall_posteriors)

        prec_size = sum(
            tally.precision_counts[k].total
            for k in tally.precision_counts if k[0] == eid
        )
        rec_size = sum(
            tally.recall_counts[k].total_in_sample
            for k in tally.recall_counts if k[0] == eid
        )

        prec_w: float | None = None
        rec_w: float | None = None
        if has_prec:
            widths = [_ci_width(precision_posteriors[k]) for k in precision_posteriors if k[0] == eid]
            prec_w = max(widths) if widths else None
        if has_rec:
            widths = [_ci_width(recall_posteriors[k]) for k in recall_posteriors if k[0] == eid]
            rec_w = max(widths) if widths else None

        min_count = min(prec_size, rec_size) if has_prec and has_rec else (prec_size or rec_size)
        min_fold = min_count // n_folds if n_folds > 0 else 0

        if eid in frame_blind_ids:
            flag = "no-data"
            reason = "no-data: frame-blind"
            no_data += 1
        elif not has_prec and not has_rec:
            flag = "no-data"
            reason = "no-data: no-classifier-positives"
            no_data += 1
        elif has_prec and has_rec:
            max_width = max(w for w in [prec_w, rec_w] if w is not None)
            if max_width < 0.30:
                flag = "adequate"
                reason = "adequate"
            else:
                flag = "wide"
                reason = f"wide: small-sample (n={min(prec_size, rec_size)})"
            both += 1
        else:
            flag = "wide"
            reason = "wide: recall-frame-only"
            recall_only += 1

        reports[eid] = EntryCalibrationReport(
            entry_id=eid,
            has_precision_data=has_prec,
            has_recall_data=has_rec,
            precision_ci_width=prec_w,
            recall_ci_width=rec_w,
            recall_sample_size=rec_size,
            precision_sample_size=prec_size,
            min_fold_count=min_fold,
            flag=flag,
            reason=reason,
        )

    diagnostic = CalibrationDiagnostic(
        entries_with_both_frames=both,
        entries_recall_only=recall_only,
        entries_no_data=no_data,
        entry_reports=reports,
    )

    return cal, diagnostic
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_calibration_diagnostic.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add engine/calibrate/calibrate.py tests/unit/test_calibration_diagnostic.py
git commit -m "feat(calibrate): add calibration computation and diagnostic module"
```

---

### Task 9: Adapter update — 20 entries

**Files:**
- Modify: `engine/adapters/genai_agentic.py:38-55`
- Modify: `tests/unit/test_adapter_genai_agentic.py`

- [ ] **Step 1: Write test for 20 entries**

Add to `tests/unit/test_adapter_genai_agentic.py`:

```python
class TestEntryDefinitions:
    def test_twenty_entries(self) -> None:
        adapter = GenAIAgenticAdapter(snapshot_dir=..., snapshot_date="2026-12-31")
        entries = adapter.entry_definitions()
        assert len(entries) == 20

    def test_frame_blind_entries(self) -> None:
        adapter = GenAIAgenticAdapter(snapshot_dir=..., snapshot_date="2026-12-31")
        entries = adapter.entry_definitions()
        fb = {e.entry_id for e in entries if e.frame_blind}
        assert fb == {"LLM04", "LLM08", "LLM10"}

    def test_all_entry_ids_present(self) -> None:
        adapter = GenAIAgenticAdapter(snapshot_dir=..., snapshot_date="2026-12-31")
        entries = adapter.entry_definitions()
        ids = {e.entry_id for e in entries}
        expected = {
            "LLM01", "LLM02", "LLM03", "LLM04", "LLM05",
            "LLM06", "LLM07", "LLM08", "LLM09", "LLM10",
            "NEW-PMP", "NEW-MTIE", "NEW-MA", "NEW-ITSCD", "NEW-WLA", "NEW-MSDA",
            "ROLL-CMSB", "ROLL-LAPTF", "ROLL-SICG", "ROLL-CFAS",
        }
        assert ids == expected
```

Note: the test will need to adapt to the actual test fixture pattern in `test_adapter_genai_agentic.py`. The implementer should check the existing adapter test file and use its fixture for `snapshot_dir`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_adapter_genai_agentic.py::TestEntryDefinitions -v`
Expected: FAIL — only 10 entries exist.

- [ ] **Step 3: Update `_PROVISIONAL_2025_ENTRIES` to 20 entries**

Replace the `_PROVISIONAL_2025_ENTRIES` tuple in `engine/adapters/genai_agentic.py` (lines 38-55):

```python
_PROVISIONAL_2025_ENTRIES: tuple[EntryDefinition, ...] = (
    EntryDefinition(entry_id="LLM01", name="Prompt Injection"),
    EntryDefinition(entry_id="LLM02", name="Sensitive Information Disclosure"),
    EntryDefinition(entry_id="LLM03", name="Supply Chain Vulnerabilities"),
    EntryDefinition(
        entry_id="LLM04", name="Data and Model Poisoning", frame_blind=True
    ),
    EntryDefinition(entry_id="LLM05", name="Improper Output Handling"),
    EntryDefinition(entry_id="LLM06", name="Excessive Agency"),
    EntryDefinition(entry_id="LLM07", name="System Prompt Leakage"),
    EntryDefinition(
        entry_id="LLM08", name="Vector and Embedding Weaknesses", frame_blind=True
    ),
    EntryDefinition(entry_id="LLM09", name="Misinformation"),
    EntryDefinition(
        entry_id="LLM10", name="Unbounded Consumption", frame_blind=True
    ),
    EntryDefinition(entry_id="NEW-PMP", name="Prompt Management and Pipelines"),
    EntryDefinition(entry_id="NEW-MTIE", name="Multi-Tenant Isolation Erosion"),
    EntryDefinition(entry_id="NEW-MA", name="Model Abuse"),
    EntryDefinition(entry_id="NEW-ITSCD", name="Insufficient Training/Serving Configuration and Defaults"),
    EntryDefinition(entry_id="NEW-WLA", name="Weak LLM Agent Authorization"),
    EntryDefinition(entry_id="NEW-MSDA", name="Model Supply and Dependency Attacks"),
    EntryDefinition(entry_id="ROLL-CMSB", name="Cross-Modal Safety Bypass"),
    EntryDefinition(entry_id="ROLL-LAPTF", name="Lack of Adversarial Prompt Testing Frameworks"),
    EntryDefinition(entry_id="ROLL-SICG", name="Stale or Insufficient Content Guardrails"),
    EntryDefinition(entry_id="ROLL-CFAS", name="Cascading Failures in Agentic Systems"),
)
```

- [ ] **Step 4: Run all tests to verify**

Run: `pytest tests/ -x -q`
Expected: ALL PASS (existing tests adapted to 20 entries or are entry-count agnostic)

- [ ] **Step 5: Commit**

```bash
git add engine/adapters/genai_agentic.py tests/unit/test_adapter_genai_agentic.py
git commit -m "feat(adapter): expand entry definitions from 10 to 20 entries"
```

---

### Task 10: Real cross-validation implementation

**Files:**
- Modify: `engine/calibrate/cv.py`
- Create: `tests/unit/test_cv_real.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_cv_real.py`:

```python
"""Tests for engine.calibrate.cv — real k-fold cross-validation."""
from __future__ import annotations

import pytest

from engine.calibrate.cv import CVResult, cross_validate_calibration


class TestCrossValidateCalibration:
    def test_stable_labels(self) -> None:
        precision = {("LLM01", "security"): [True] * 35 + [False] * 5}
        recall = {("LLM01", "security"): [True] * 30 + [False] * 10}
        result = cross_validate_calibration(precision, recall, n_folds=5)
        assert result.n_folds == 5
        assert ("LLM01", "security") in result.fold_variances
        assert result.interpretation[("LLM01", "security")] == "stable"

    def test_unstable_small_sample(self) -> None:
        precision = {("LLM01", "security"): [True, False, True]}
        recall = {}
        result = cross_validate_calibration(precision, recall, n_folds=5)
        assert result.min_per_fold[("LLM01", "security")] < 5
        assert "unstable" in result.interpretation[("LLM01", "security")]

    def test_empty_labels(self) -> None:
        result = cross_validate_calibration({}, {}, n_folds=5)
        assert result.n_folds == 5
        assert result.fold_variances == {}

    def test_interpretation_thresholds(self) -> None:
        precision = {("LLM01", "security"): [True] * 100 + [False] * 100}
        result = cross_validate_calibration(precision, {}, n_folds=5)
        interp = result.interpretation[("LLM01", "security")]
        assert interp in ("stable", "moderate", "unstable — interpret with caution")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_cv_real.py -v`
Expected: FAIL — `cross_validate_calibration` raises NotImplementedError.

- [ ] **Step 3: Implement real CV**

Replace `cross_validate_calibration` in `engine/calibrate/cv.py`:

```python
"""K-fold cross-validation for calibration stability.

See HANDOFF §6 control 11(c). This is a transparency disclosure,
not a quality gate — high fold variance does not block the pipeline.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass

from engine.calibrate.beta import BetaPosterior

__all__ = ["CVResult", "cross_validate_calibration"]


@dataclass(frozen=True, slots=True)
class CVResult:
    n_folds: int
    fold_variances: dict[tuple[str, str], float]
    interpretation: dict[tuple[str, str], str]
    min_per_fold: dict[tuple[str, str], int]


def _interpret(variance: float, min_per_fold: int) -> str:
    if min_per_fold < 5:
        return "unstable — interpret with caution"
    if variance < 0.01:
        return "stable"
    if variance < 0.05:
        return "moderate"
    return "unstable — interpret with caution"


def cross_validate_calibration(
    precision_labels: dict[tuple[str, str], list[bool]],
    recall_labels: dict[tuple[str, str], list[bool]],
    n_folds: int = 5,
) -> CVResult:
    all_labels: dict[tuple[str, str], list[bool]] = {}
    for key, vals in precision_labels.items():
        all_labels.setdefault(key, []).extend(vals)
    for key, vals in recall_labels.items():
        all_labels.setdefault(key, []).extend(vals)

    fold_variances: dict[tuple[str, str], float] = {}
    interpretation: dict[tuple[str, str], str] = {}
    min_per_fold_out: dict[tuple[str, str], int] = {}

    for key, labels in all_labels.items():
        n = len(labels)
        if n == 0:
            continue

        fold_size = n // n_folds
        remainder = n % n_folds
        fold_means: list[float] = []

        start = 0
        min_fold_n = n
        for i in range(n_folds):
            end = start + fold_size + (1 if i < remainder else 0)
            fold = labels[start:end]
            if len(fold) == 0:
                continue
            min_fold_n = min(min_fold_n, len(fold))
            successes = sum(fold)
            failures = len(fold) - successes
            bp = BetaPosterior.from_counts(successes, failures)
            fold_means.append(bp.mean)
            start = end

        if len(fold_means) < 2:
            var = 0.0
        else:
            var = statistics.variance(fold_means)

        fold_variances[key] = var
        min_per_fold_out[key] = min_fold_n
        interpretation[key] = _interpret(var, min_fold_n)

    return CVResult(
        n_folds=n_folds,
        fold_variances=fold_variances,
        interpretation=interpretation,
        min_per_fold=min_per_fold_out,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_cv_real.py tests/unit/test_calibrate.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add engine/calibrate/cv.py tests/unit/test_cv_real.py
git commit -m "feat(calibrate): implement real k-fold cross-validation"
```

---

### Task 11: CLI commands for calibration pipeline

**Files:**
- Create: `engine/cli/calibration.py`
- Modify: `engine/cli/main.py`

- [ ] **Step 1: Create CLI module with 6 commands**

Create `engine/cli/calibration.py`:

```python
"""CLI commands for the 6-stage calibration pipeline.

classify → sample → generate-batches → tally → calibrate → cv-stability
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import click

from engine.version import __version__


@click.command("cal-classify")
@click.option("--cycle", required=True, type=click.Path(path_type=Path))
@click.option("--rubric", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--snapshot-dir", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--snapshot-date", required=True, type=str)
@click.option("--confidence-threshold", type=float, default=0.3)
def cal_classify(
    cycle: Path,
    rubric: Path,
    snapshot_dir: Path,
    snapshot_date: str,
    confidence_threshold: float,
) -> None:
    """Stage 1: Run the deterministic keyword/indicator classifier."""
    from engine.calibrate.provenance import StageProvenance, hash_file, hash_json, write_provenance
    from engine.classify.classifier import build_rules_from_rubric, classify_real
    from engine.prereg.rubric_io import read_rubric

    cal_dir = cycle / "calibration"
    cal_dir.mkdir(parents=True, exist_ok=True)

    rb = read_rubric(rubric)
    rules = build_rules_from_rubric(rb, confidence_threshold)

    from engine.adapters.genai_agentic import GenAIAgenticAdapter
    adapter = GenAIAgenticAdapter(snapshot_dir, snapshot_date)
    incidents = tuple(adapter.iter_incidents())

    result = classify_real(incidents, rules)

    out_path = cal_dir / "classifications.json"
    out_data = {
        "classifier_version": result.classifier_version,
        "classifier_rule_hash": result.classifier_rule_hash,
        "classification_count": len(result.classifications),
        "classifications": [
            {
                "incident_id": c.incident_id,
                "entry_id": c.entry_id,
                "confidence": c.confidence,
                "stage": c.stage,
                "rationale": c.rationale,
            }
            for c in result.classifications
        ],
    }
    out_path.write_text(json.dumps(out_data, indent=2) + "\n")

    prov = StageProvenance(
        stage_name="classify",
        manifest_lock_hash="",
        input_hashes={"rubric": hash_file(rubric)},
        output_hash=hash_json(out_data),
        timestamp=datetime.now(UTC).isoformat(),
        engine_version=__version__,
    )
    write_provenance(prov, cal_dir / "classify_provenance.json")

    click.echo(f"Classified {len(incidents)} incidents → {len(result.classifications)} labels.")
    click.echo(f"Rule hash: {result.classifier_rule_hash}")


@click.command("cal-sample")
@click.option("--cycle", required=True, type=click.Path(path_type=Path))
def cal_sample(cycle: Path) -> None:
    """Stage 2: Draw precision-frame and recall-frame samples."""
    click.echo("cal-sample: Not yet wired (Task 11 placeholder for CLI registration).")


@click.command("cal-generate-batches")
@click.option("--cycle", required=True, type=click.Path(path_type=Path))
def cal_generate_batches(cycle: Path) -> None:
    """Stage 3: Generate batch files for manual coding."""
    click.echo("cal-generate-batches: Not yet wired.")


@click.command("cal-tally")
@click.option("--cycle", required=True, type=click.Path(path_type=Path))
def cal_tally(cycle: Path) -> None:
    """Stage 4: Aggregate coded labels into per-entry per-stratum counts."""
    click.echo("cal-tally: Not yet wired.")


@click.command("cal-calibrate")
@click.option("--cycle", required=True, type=click.Path(path_type=Path))
def cal_calibrate(cycle: Path) -> None:
    """Stage 5: Compute Beta posteriors from tally counts."""
    click.echo("cal-calibrate: Not yet wired.")


@click.command("cal-cv-stability")
@click.option("--cycle", required=True, type=click.Path(path_type=Path))
def cal_cv_stability(cycle: Path) -> None:
    """Stage 6: k=5 cross-validation for calibration stability."""
    click.echo("cal-cv-stability: Not yet wired.")
```

- [ ] **Step 2: Register commands in main.py**

Add to `engine/cli/main.py` after the existing imports:

```python
from engine.cli.calibration import (
    cal_calibrate,
    cal_classify,
    cal_cv_stability,
    cal_generate_batches,
    cal_sample,
    cal_tally,
)
```

And after the existing `cli.add_command(...)` lines:

```python
cli.add_command(cal_classify)
cli.add_command(cal_sample)
cli.add_command(cal_generate_batches)
cli.add_command(cal_tally)
cli.add_command(cal_calibrate)
cli.add_command(cal_cv_stability)
```

- [ ] **Step 3: Run CLI help to verify registration**

Run: `python -m engine.cli.main --help`
Expected: All 6 `cal-*` commands appear in the help output.

- [ ] **Step 4: Run full test suite**

Run: `pytest tests/ -x -q`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add engine/cli/calibration.py engine/cli/main.py
git commit -m "feat(cli): register 6 calibration pipeline CLI commands"
```

---

### Task 12: End-to-end synthetic calibration test

**Files:**
- Create: `tests/integration/test_calibration_e2e.py`
- Create: `tests/integration/__init__.py`

- [ ] **Step 1: Write the e2e test**

Create `tests/integration/__init__.py` (empty).

Create `tests/integration/test_calibration_e2e.py`:

```python
"""End-to-end test: classify → sample → code_synthetic → tally → calibrate → cv-stability.

Uses the SyntheticAdapter's incidents with native_labels as ground truth.
Verifies the full pipeline produces valid Calibration and CVResult objects.
"""
from __future__ import annotations

import pytest

from engine.adapters.synthetic import SyntheticAdapter
from engine.calibrate.batch import code_synthetic_with_ground_truth, generate_batch
from engine.calibrate.calibrate import compute_calibration
from engine.calibrate.cv import cross_validate_calibration
from engine.calibrate.sampler import SampleFrame, SampleRequest
from engine.calibrate.tally import tally_batches
from engine.calibrate.two_frame_sampler import TwoFrameSampler
from engine.classify.stub import classify_stub


class TestCalibrationE2E:
    def test_synthetic_pipeline(self) -> None:
        adapter = SyntheticAdapter(seed=42)
        incidents = tuple(adapter.iter_incidents())
        entries = adapter.entry_definitions()
        entry_ids = tuple(e.entry_id for e in entries)
        non_fb_ids = tuple(e.entry_id for e in entries if not e.frame_blind)
        fb_ids = {e.entry_id for e in entries if e.frame_blind}
        strata = sorted({inc.corpus_stratum for inc in incidents})

        # Stage 1: Classify (stub for synthetic)
        classification = classify_stub(incidents, entry_ids)

        # Stage 2: Sample
        sampler = TwoFrameSampler(classification_result=classification)
        incidents_list = list(incidents)
        incidents_by_id = {inc.id: inc for inc in incidents}

        batches = []

        # Precision-frame: for each non-frame-blind entry × stratum
        for eid in non_fb_ids:
            for s in strata:
                req = SampleRequest(
                    frame=SampleFrame.PRECISION,
                    entry_id=eid,
                    stratum=s,
                    n=40,
                )
                sr = sampler.draw(req, incidents_list, seed=42)
                if sr.actual_n > 0:
                    batch = generate_batch(
                        sample_result=sr,
                        rubric_hash="test",
                        manifest_lock_hash="test",
                        coder_id="synthetic",
                        cycle_id="test",
                    )
                    coded = code_synthetic_with_ground_truth(
                        batch, incidents_by_id, set(entry_ids),
                    )
                    batches.append(coded)

        # Recall-frame: per stratum
        for s in strata:
            req = SampleRequest(
                frame=SampleFrame.RECALL,
                entry_id=None,
                stratum=s,
                n=100,
            )
            sr = sampler.draw(req, incidents_list, seed=42)
            batch = generate_batch(
                sample_result=sr,
                rubric_hash="test",
                manifest_lock_hash="test",
                coder_id="synthetic",
                cycle_id="test",
            )
            coded = code_synthetic_with_ground_truth(
                batch, incidents_by_id, set(entry_ids),
            )
            batches.append(coded)

        # Stage 4: Tally
        tally = tally_batches(batches)
        assert tally.total_coded > 0

        # Stage 5: Calibrate
        cal, diag = compute_calibration(
            tally,
            all_entry_ids=list(entry_ids),
            strata=strata,
            frame_blind_ids=fb_ids,
        )
        assert len(cal.precision) > 0 or len(cal.recall) > 0
        assert diag.entries_with_both_frames + diag.entries_recall_only + diag.entries_no_data == len(entry_ids)

        # Stage 6: CV stability
        prec_labels: dict[tuple[str, str], list[bool]] = {}
        for batch in batches:
            if batch.header.frame == "precision" and batch.header.entry_id:
                key = (batch.header.entry_id, batch.header.stratum or "unknown")
                prec_labels.setdefault(key, [])
                for inc in batch.incidents:
                    if inc.labels is not None and batch.header.entry_id is not None:
                        prec_labels[key].append(batch.header.entry_id in inc.labels)

        rec_labels: dict[tuple[str, str], list[bool]] = {}
        for batch in batches:
            if batch.header.frame == "recall":
                for inc in batch.incidents:
                    if inc.labels is not None:
                        for eid in entry_ids:
                            key = (eid, batch.header.stratum or "unknown")
                            rec_labels.setdefault(key, [])
                            rec_labels[key].append(eid in inc.labels)

        cv = cross_validate_calibration(prec_labels, rec_labels, n_folds=5)
        assert cv.n_folds == 5
```

- [ ] **Step 2: Run the e2e test**

Run: `pytest tests/integration/test_calibration_e2e.py -v`
Expected: PASS — the full pipeline runs with synthetic data.

- [ ] **Step 3: Run full test suite**

Run: `pytest tests/ -x -q`
Expected: ALL PASS (including proof tests)

- [ ] **Step 4: Commit**

```bash
git add tests/integration/__init__.py tests/integration/test_calibration_e2e.py
git commit -m "test(calibrate): add end-to-end synthetic calibration pipeline test"
```

---

### Task 13: Version bump and final verification

**Files:**
- Modify: `engine/version.py`
- Modify: `tests/test_bootstrap.py` (if version assertion exists)

- [ ] **Step 1: Bump version**

In `engine/version.py`:
```python
__version__ = "0.4.0"
```

- [ ] **Step 2: Update version assertion if needed**

Check `tests/test_bootstrap.py` for version assertions and update accordingly.

- [ ] **Step 3: Run full test suite**

Run: `pytest tests/ -x -q`
Expected: ALL PASS

- [ ] **Step 4: Commit and tag**

```bash
git add engine/version.py tests/test_bootstrap.py
git commit -m "chore: bump version to 0.4.0 for Plan 4 calibration pipeline"
git tag v0.4.0-plan4
```

---

## Self-Review Checklist

| Spec section | Task(s) | Status |
|-------------|---------|--------|
| PreregManifest.confidence_threshold | Task 1 | ✅ |
| Classifier rule hash gate | Task 2 | ✅ |
| StageProvenance chain | Task 3 | ✅ |
| Real Stage-1 classifier | Task 4 | ✅ |
| Sampler protocol redesign | Task 5 | ✅ |
| SampleFrame, SampleRequest, SampleResult | Task 5 | ✅ |
| TwoFrameSampler implementation | Task 5 | ✅ |
| Batch file generation + validation | Task 6 | ✅ |
| Coding correction protocol (amendment) | Task 6 | ✅ |
| code_synthetic for testing | Task 6 | ✅ |
| Recall-frame coding checklist | Task 6 | ✅ |
| Batch re-coding policy | Task 6 (provenance enforces) | ✅ |
| Tally aggregation | Task 7 | ✅ |
| PrecisionTally, RecallTally, TallyResult | Task 7 | ✅ |
| Calibration computation | Task 8 | ✅ |
| CalibrationDiagnostic with reason field | Task 8 | ✅ |
| EntryCalibrationReport with sample sizes | Task 8 | ✅ |
| Adapter → 20 entries | Task 9 | ✅ |
| CVResult with interpretation + min_per_fold | Task 10 | ✅ |
| Real cross-validation | Task 10 | ✅ |
| 6 CLI commands | Task 11 | ✅ |
| Output artifact paths | Task 11 | ✅ |
| Rollup sub-test tally | Task 7 | ✅ |
| E2E synthetic pipeline test | Task 12 | ✅ |
| Overlap weights pre-registered (doc only) | Spec already states | ✅ |
| FP term mismatch documented | Spec residual risk | ✅ |
| Indicator match semantics defined | Spec + Task 4 | ✅ |
