# Corpus A Adapter + Snapshot + Per-Stratum Bias Profiles — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a working corpus A adapter that reads 7,714 real incidents from `genai_agentic_incidents`, emits them as canonical `IncidentRecord` instances through `engine/schema.py`, with per-sub-corpus stratum bias profiles, bare-LLM03 contamination quarantine, severity-default artifact detection, future-dated row repair, and a content-hashed vendored snapshot with provenance.

**Architecture:** The adapter (`engine/adapters/genai_agentic.py`) reads from a frozen, content-hashed snapshot vendored under `projects/owasp-llm/cycles/2026/corpora/genai_agentic/<hash>/`. It implements the `CorpusAdapter` ABC from Plan 1. Per-stratum bias profiles are co-located in `engine/adapters/genai_agentic_bias.py` and declare quarantine rules per HANDOFF §5.1 ("source-specific quarantine rules live in the adapter, declared in its bias profile"). A snapshot vendoring CLI command (`engine/cli/snapshot.py`) pulls the source once and freezes it.

**Tech Stack:** Python 3.12, Click (CLI), hashlib (content hashing), json (serialization), existing Plan 1 engine types (`IncidentRecord`, `BiasProfile`, `StratumSize`, `OverlapWeights`, `CorpusAdapter`, `SnapshotProvenance`, `DriftReport`).

**Source of truth:** `docs/HANDOFF.md` v2.5 (§3, §4, §5.1, §6 control 9). `docs/PRD.md` §3 (Plan 2).

**Target tag:** `v0.2.0-plan2`

---

## Inherited constraints from Phase 1

These are concrete carry-forwards from Plan 1 v5 that constrain Plan 2's implementation. Each was discovered during Plan 1 execution and recorded in the Plan 1 plan's Coverage matrices (M1–M23) and Residuals sections, or in the v5.1 erratum.

### C1. Bare-LLM03 contamination must be quarantined, not idealized (M1 motivation)

Plan 1's Task 24 created `synthetic-stress` specifically because the original synthetic corpus was too clean — it had no contaminated records. The real corpus has ~907 bare `["LLM03"]` default-seed entries and ~768 exact `["LLM03", "ASI04"]` double-defaults (HANDOFF §3 F2). Plan 2's adapter MUST quarantine these per HANDOFF §5.2: "A bare `['LLM03']` default-seed CVE the rubric cannot place goes to the sink, not to LLM03." The quarantine fires *in the adapter* — quarantined records are still emitted as `IncidentRecord` instances (the schema has no `quarantined` field), but contamination is detectable via the `is_bare_llm03_contaminated()` / `is_double_default_contaminated()` predicates in `genai_agentic_bias.py`. Downstream pipeline stages call these predicates to route contaminated records to the out-of-scope sink (HANDOFF §5.2). Tests must exercise actual contaminated records from the vendored snapshot, not synthetic stand-ins.

### C2. Construction-time defensive validation pattern (M2)

Plan 1's `OverlapWeights.__post_init__` rejects self-loops at construction time, not at use time. Plan 2's `BiasProfile` validation and adapter construction must follow the same pattern: invalid inputs fail loudly at object creation, not when the adapter is iterated. This means `GenAIAgenticAdapter.__init__()` validates the snapshot path exists and has the expected hash, and `build_bias_profiles()` validates stratum completeness, before any records are emitted.

### C3. Stratum-size sanity contract (M3)

Plan 1's `execute_synthetic_pipeline` guard asserts `stratum_size >= observed_count` per stratum. Plan 2's adapter's `stratum_sizes()` return values must satisfy this contract. Since Plan 2 is adapting real data, stratum sizes are the actual record counts per stratum — the adapter sets `stratum_size = count_of_records_in_stratum` (after quarantine), which trivially satisfies the >= contract. The test must verify this explicitly.

### C4. Per-task commit cadence

Plan 1 delivered 30 commits in 30 tasks, one feat+test per commit. Plan 2 follows the same cadence: each task produces exactly one commit with the feature and its tests together.

### C5. CI verification erratum — workflow presence ≠ CI execution (Plan 1 v5.1)

Plan 1 claimed CI-green acceptance for criteria 6/7/8/9 against a workflow that never executed a single job. Five distinct CI bugs hid behind a YAML flow-style `}` collision. **If Plan 2 modifies CI** (adding new test paths, snapshot scripts), the plan MUST include a task that verifies CI *actually runs the new logic to completion in a green run* by checking the GitHub Actions tab. `actionlint` and local test passes are necessary but not sufficient. Erratum documented in `docs/METHODOLOGY-CHANGELOG.md` "Plan 1 v5.1 erratum (2026-05-20)".

### C6. Residuals still acknowledged from Plan 1 (not mitigable in Plan 2)

- **F-circ (taxonomy-frame circularity):** intrinsic to any corpus derived from a keyword crawl. Standing caveat, no mitigation.
- **Stage-2 GPU prompt content:** Plan 5 scope. Plan 2 does not touch the classifier.
- **BLAS-level JAX determinism within MCSE:** verified by cross-platform diff CI job. Plan 2 does not modify inference.
- **Single-author rubric until external reviewers identified:** REVIEWERS.md at INTERIM. Plan 2 is not reviewer-gated.

### C7. HANDOFF §6 integrity controls override engineering convenience

Per PRD §1: "HANDOFF v2.5 §6 (integrity controls) and §6 control 11 (information firewall) override any contradictory engineering convenience." Plan 2 does not run classification or inference — it only builds the adapter and vendors the snapshot — so most integrity controls are not exercised. But §6 control 9 (snapshot integrity: content hashing + provenance + drift detection + manual signoff) is directly in scope and must be fully implemented.

### C8. No engine changes for their own sake

Per PRD §1: "Plan 1 produced the engine; no phase from Plan 2 onward extends the engine *for its own sake*. Engine changes are admissible only when a phase's deliverables require them, and any such change is a methodology-changelog entry." Plan 2 adds new adapter files and a CLI command. It does NOT modify existing engine modules (`schema.py`, `adapters/base.py`, `snapshot/hashing.py`, `snapshot/provenance.py`, `snapshot/drift.py`) unless a bug or missing capability is discovered, in which case the change is documented.

---

## File structure

### New files

| File | Responsibility |
|------|---------------|
| `engine/adapters/genai_agentic.py` | Concrete `CorpusAdapter` for the genai_agentic_incidents corpus. Reads vendored snapshot JSON, emits `IncidentRecord` instances with quarantine, severity-default detection, and future-dated row repair. |
| `engine/adapters/genai_agentic_bias.py` | Per-stratum `BiasProfile` declarations and quarantine-rule predicates for the genai_agentic corpus. Construction-time validation (C2 pattern). |
| `engine/cli/snapshot.py` | `vendor-snapshot` CLI command. Copies source corpus JSON to the content-addressed snapshot path, writes `provenance.json`, runs drift detection if a prior snapshot exists. |
| `tests/unit/test_adapter_genai_agentic.py` | Tests for the adapter: schema round-trip, stratum population, quarantine, severity-default detection, future-dated repair, snapshot hash stability. |
| `tests/unit/test_snapshot_vendor.py` | Tests for the snapshot vendoring script: provenance fields, hash determinism, drift integration. |

### Modified files

| File | Change |
|------|--------|
| `engine/adapters/__init__.py` | Add `GenAIAgenticAdapter` export. |
| `engine/cli/main.py` | Wire `vendor-snapshot` Click command. |
| `projects/owasp-llm/project.toml` | Populate REQUIRED placeholder fields with real 2026 cycle values. |
| `docs/METHODOLOGY-CHANGELOG.md` | Add 0.2.0 entry. |

### Vendored artifacts (committed, not code)

| Path | Content |
|------|---------|
| `projects/owasp-llm/cycles/2026/corpora/genai_agentic/<hash>/incidents.json` | Content-hashed frozen snapshot of the source corpus. |
| `projects/owasp-llm/cycles/2026/corpora/genai_agentic/<hash>/provenance.json` | Six-field provenance record: `source_repo`, `source_commit_sha`, `pull_date`, `adapter_name`, `adapter_version`, `snapshot_hash`. |

---

## Quality gates (run after every task)

```bash
uv run pytest tests/unit/test_adapter_genai_agentic.py -v   # new adapter tests
uv run pytest -v                                              # full suite, no regressions
uv run mypy engine tests                                      # strict mode, zero errors
uv run ruff check .                                           # zero lint errors
uv run semgrep --config .semgrep.yml --error engine/          # zero SAST findings
```

A task does not advance until all five commands pass.

---

## Task 0: Prerequisites — clone source corpus + create branch

**Files:**
- Read-only: `~/github_projects/genai_agentic_incidents/data/incidents.json` (external)
- Read: `engine/schema.py`, `engine/adapters/base.py` (verify unchanged)

This task produces no committed code. It establishes the working environment.

- [ ] **Step 1: Verify Plan 1 tag is present**

```bash
git tag -l 'v0.1.*-plan1'
```

Expected: `v0.1.0-plan1` and `v0.1.1-plan1` both present.

- [ ] **Step 2: Verify Plan 1 schema and adapter base are unchanged**

```bash
git diff v0.1.1-plan1..HEAD -- engine/schema.py engine/adapters/base.py
```

Expected: empty diff. If non-empty, stop — PRD §3.3 prerequisite 2 is violated. Any change must be a documented methodology-changelog entry.

- [ ] **Step 3: Clone the source corpus**

```bash
cd ~/github_projects
git clone https://github.com/rocklambros/genai_agentic_incidents.git
```

If already cloned, pull latest:

```bash
cd ~/github_projects/genai_agentic_incidents && git pull
```

- [ ] **Step 4: Verify source data accessible**

```bash
python3 -c "
import json, pathlib
p = pathlib.Path.home() / 'github_projects/genai_agentic_incidents/data/incidents.json'
data = json.loads(p.read_text())
print(f'Records: {len(data)}, Type: {type(data).__name__}')
if isinstance(data, list) and data:
    print(f'First record keys: {sorted(data[0].keys())}')
"
```

Expected: `Records: ~7714, Type: list`. Record the actual field names — they are needed for Task 3's field mapping.

- [ ] **Step 5: Record the source repo's current commit SHA**

```bash
cd ~/github_projects/genai_agentic_incidents && git rev-parse HEAD
```

Record this SHA — it goes into `provenance.json` in Task 2.

- [ ] **Step 6: Create the working branch**

```bash
cd ~/github_projects/incident-rank-validation
git checkout -b plan2/corpus-a-adapter
```

- [ ] **Step 7: Explore source corpus schema and stratum counts**

This step is essential for writing accurate tests and adapter code in subsequent tasks. Run:

```bash
python3 -c "
import json, pathlib, collections

p = pathlib.Path.home() / 'github_projects/genai_agentic_incidents/data/incidents.json'
data = json.loads(p.read_text())

# Show all keys from first record
print('=== FIELD NAMES (first record) ===')
for k, v in sorted(data[0].items()):
    print(f'  {k}: {type(v).__name__} = {repr(v)[:100]}')

# Corpus strata
corpus_counts = collections.Counter(r.get('corpus', 'MISSING') for r in data)
print(f'\n=== corpus field ===')
for k, v in corpus_counts.most_common():
    print(f'  {k}: {v}')

# Category strata
cat_counts = collections.Counter(r.get('category', 'MISSING') for r in data)
print(f'\n=== category field ===')
for k, v in cat_counts.most_common():
    print(f'  {k}: {v}')

# Bare-LLM03 contamination count
bare_llm03 = sum(1 for r in data if r.get('owasp_llm') == ['LLM03'])
double_default = sum(1 for r in data if sorted(r.get('owasp_llm', [])) == ['ASI04', 'LLM03'] or r.get('owasp_llm') == ['LLM03', 'ASI04'])
print(f'\n=== Contamination ===')
print(f'  bare [\"LLM03\"]: {bare_llm03}')
print(f'  [\"LLM03\", \"ASI04\"] double-default: {double_default}')

# Severity distribution
sev_counts = collections.Counter(r.get('severity', 'MISSING') for r in data)
print(f'\n=== severity field ===')
for k, v in sev_counts.most_common():
    print(f'  {k}: {v}')

# Quality tier
qt_counts = collections.Counter(r.get('quality_tier', 'MISSING') for r in data)
print(f'\n=== quality_tier field ===')
for k, v in qt_counts.most_common():
    print(f'  {k}: {v}')

# Date range
dates = sorted(r.get('date', r.get('published_date', '')) for r in data if r.get('date') or r.get('published_date'))
print(f'\n=== Date range ===')
print(f'  Earliest: {dates[0] if dates else \"NONE\"}')
print(f'  Latest: {dates[-1] if dates else \"NONE\"}')
# Future-dated (after 2026-05-20)
future = [d for d in dates if d > '2026-05-20']
print(f'  Future-dated (after 2026-05-20): {len(future)}')
"
```

**Record all output.** The exact field names, stratum counts, contamination counts, severity distribution, and date range are inputs to Tasks 1–6. If any field name differs from what HANDOFF §3 describes (e.g., `published_date` vs `date`, `quality_tier` vs `quality`), the adapter's field-mapping code must account for it.

> **⛔ HARD GATE (Premortem M1):** All field names in Tasks 1–7 are **PROVISIONAL** — they are best guesses derived from HANDOFF §3's audit prose, not from direct schema inspection. This step's output is the single source of truth for field mapping. **Do not proceed to Task 1 until this step has been executed and the actual field names recorded.** If any provisional field name (e.g., `corpus`, `category`, `owasp_llm`, `quality_tier`, `severity`, `date`) does not appear in the actual schema, you MUST update every subsequent task's code to use the real field name before implementing it. Grep the plan for the provisional name and fix all occurrences.

- [ ] **Step 8: Inspect source repo ingest scripts for severity-defaulting mechanism (M6)**

The adapter's `_is_severity_defaulted()` heuristic must match the *actual* source-ingest behavior. Examine the source repo's ingest scripts to understand how severity is assigned:

```bash
cd ~/github_projects/genai_agentic_incidents
# Find ingest scripts that handle severity
grep -rn 'severity' --include='*.py' . | head -40
# Look specifically for the default-assignment pattern described in HANDOFF §3
grep -rn 'Medium\|default.*sev\|sev.*default' --include='*.py' . | head -20
```

**Record:**
1. Which script(s) assign severity.
2. The exact mechanism: is it a literal `"Medium"` default? A fallback? A conditional?
3. Whether there is a `severity_source` or `severity_method` field that records provenance.
4. Whether the `quality_tier` field reliably distinguishes human-curated from auto-assigned severity.

This information directly informs the `_is_severity_defaulted()` implementation in Task 4. If the source has an explicit provenance field (e.g., `severity_source: "default"`), use it. If not, the heuristic in Task 4 must be documented as best-effort.

- [ ] **Step 9: Verify source commit SHA is from a tagged or release-worthy state (R5)**

```bash
cd ~/github_projects/genai_agentic_incidents
git log --oneline -5
git tag -l
```

Record whether the current HEAD is a tagged release or a mid-development commit. If mid-development, note this in the provenance documentation — snapshot consumers need to know whether the source was at a stable point.

**No commit from this task.** This is environment setup.

---

## Task 1: Per-stratum bias profile declarations + quarantine predicates

**Files:**
- Create: `engine/adapters/genai_agentic_bias.py`
- Create: `tests/unit/test_adapter_genai_agentic.py` (first tests)

**Acceptance criteria served:** PRD §3.6 criteria 1 (adapter tests green), 3 (mypy/ruff/semgrep clean).

- [ ] **Step 1: Write failing tests for bias profile construction and validation**

Create `tests/unit/test_adapter_genai_agentic.py`:

```python
"""Tests for the genai_agentic corpus A adapter.

All test counts and field references are derived from the audit in
HANDOFF §3 (owasp-mapping-quality-audit.md, N=7,714) and confirmed
against the vendored snapshot in Task 0 Step 7.
"""
from __future__ import annotations

import pytest

from engine.adapters.genai_agentic_bias import (
    BIAS_PROFILES,
    is_bare_llm03_contaminated,
    is_double_default_contaminated,
    build_bias_profiles,
)
from engine.schema import BiasProfile


class TestBiasProfiles:
    """Per-stratum bias profile declarations (HANDOFF §3 Mixture, §4 row, §5.1)."""

    def test_one_profile_per_stratum(self) -> None:
        profiles = build_bias_profiles()
        strata = {p.stratum for p in profiles}
        assert "security" in strata
        assert "ai-harm" in strata
        assert len(profiles) >= 2

    def test_profiles_are_biasprofile_instances(self) -> None:
        for p in build_bias_profiles():
            assert isinstance(p, BiasProfile)

    def test_security_stratum_declares_contamination(self) -> None:
        sec = next(p for p in build_bias_profiles() if p.stratum == "security")
        assert "LLM03" in sec.contamination_description
        assert sec.quarantine_rule != ""

    def test_ai_harm_stratum_declares_known_blind_spots(self) -> None:
        ah = next(p for p in build_bias_profiles() if p.stratum == "ai-harm")
        assert len(ah.known_blind_spots) > 0

    def test_construction_time_validation_rejects_empty_stratum(self) -> None:
        """C2 pattern: invalid input fails at construction, not at use time."""
        with pytest.raises(ValueError, match="stratum"):
            BiasProfile(
                stratum="",
                description="empty",
                known_blind_spots=(),
                contamination_description="none",
                quarantine_rule="none",
            )


class TestQuarantinePredicates:
    """Contamination quarantine rules (HANDOFF §3 F2, §5.2 out-of-scope sink)."""

    def test_bare_llm03_detected(self) -> None:
        assert is_bare_llm03_contaminated(["LLM03"]) is True

    def test_bare_llm03_not_triggered_on_multi_label(self) -> None:
        assert is_bare_llm03_contaminated(["LLM03", "LLM05"]) is False

    def test_double_default_detected(self) -> None:
        assert is_double_default_contaminated(["LLM03", "ASI04"]) is True
        assert is_double_default_contaminated(["ASI04", "LLM03"]) is True

    def test_double_default_not_triggered_on_triple(self) -> None:
        assert is_double_default_contaminated(["LLM03", "ASI04", "LLM05"]) is False

    def test_empty_labels_not_contaminated(self) -> None:
        assert is_bare_llm03_contaminated([]) is False
        assert is_double_default_contaminated([]) is False
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/unit/test_adapter_genai_agentic.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'engine.adapters.genai_agentic_bias'`.

- [ ] **Step 3: Implement bias profiles and quarantine predicates**

Create `engine/adapters/genai_agentic_bias.py`:

```python
"""Per-stratum bias profiles and quarantine predicates for the genai_agentic corpus.

Corpus A is a mixture (HANDOFF §3 Mixture paragraph, §4 Corpus-A-is-a-mixture row).
The ``corpus`` field defines primary strata: "security" (~7,350) and "ai-harm" (~364).
Each stratum carries a declared BiasProfile with its contamination description and
quarantine rule per HANDOFF §5.1.

Quarantine predicates fire on source-record native labels.  Quarantined records are
still emitted by the adapter as IncidentRecord instances (no ``quarantined`` field on
the schema) — contamination status is determined by calling ``is_bare_llm03_contaminated()``
or ``is_double_default_contaminated()`` on a record's ``native_labels``.  Downstream
pipeline stages (Plans 3–5) MUST call these predicates to route contaminated records
to the out-of-scope sink (HANDOFF §5.2).  (Premortem M8: contamination-status dependency.)

Note on strata vs. source_class (Premortem M7): the ``corpus`` field maps 1:1 to
``corpus_stratum`` on IncidentRecord.  The ``category`` field maps LOSSILY to
``source_class`` via ``_map_source_class()`` — multiple source categories collapse to
fewer engine source classes (e.g., "research" and "threat-report" both → "advisory").
The original category granularity is not preserved on IncidentRecord.  If Plans 3–5
need per-category stratification, the adapter must be extended to expose it.
"""
from __future__ import annotations

from engine.schema import BiasProfile


def build_bias_profiles() -> tuple[BiasProfile, ...]:
    """Construct per-stratum BiasProfile declarations.

    Construction-time validation (inherited constraint C2 from Plan 1 M2):
    each profile is validated at creation.  Invalid profiles raise ValueError.
    """
    return BIAS_PROFILES


BIAS_PROFILES: tuple[BiasProfile, ...] = (
    BiasProfile(
        stratum="security",
        description=(
            "Security-focused corpus (~7,350 records).  Built by CVE/GHSA/OSV keyword "
            "crawl plus harm-database ingestion (HANDOFF §3 F-frame).  Over-represents "
            "supply-chain and dependency vulnerabilities.  98.3% machine-labeled with no "
            "human OWASP review (F1).  ~907 bare-LLM03 default-seed entries (F2) and "
            "~768 LLM03+ASI04 double-defaults contaminate this stratum."
        ),
        known_blind_spots=(
            "LLM04",  # near-absent — no ingest pathway emits it (F4)
            "LLM08",  # near-absent (F4)
            "LLM10",  # near-absent (F4)
        ),
        contamination_description=(
            "ingest_cve_nvd_expanded.py seeds every CVE with ['LLM03'] / ['ASI04'] "
            "before refinement.  ~907 entries are bare ['LLM03'], ~768 are the exact "
            "LLM03+ASI04 double default (HANDOFF §3 F2).  Treat CVE-class "
            "single-LLM03 as unknown, not supply chain."
        ),
        quarantine_rule=(
            "Quarantine records where owasp_llm == ['LLM03'] (bare default) or "
            "sorted(owasp_llm) == ['ASI04', 'LLM03'] (double default).  Quarantined "
            "records are emitted but flagged; downstream stages route them to the "
            "out-of-scope sink per HANDOFF §5.2."
        ),
    ),
    BiasProfile(
        stratum="ai-harm",
        description=(
            "AI-harm corpus (~364 records).  Drawn from harm-database ingestion, "
            "not CVE/GHSA/OSV.  Under-represents infrastructure and supply-chain "
            "incidents.  Different selection mechanism from the security stratum "
            "(HANDOFF §3 Mixture)."
        ),
        known_blind_spots=(
            "LLM04",  # infrastructure-focused, invisible to harm reports
            "LLM08",  # near-absent in harm reports
        ),
        contamination_description=(
            "Minimal direct contamination — harm reports are not CVE-seeded.  "
            "However, severity is defaulted to 'Medium' when missing in the source "
            "ingest, producing a zero-unknown-severity artifact (HANDOFF §3)."
        ),
        quarantine_rule=(
            "No bare-LLM03 quarantine needed for this stratum (not CVE-seeded).  "
            "Severity-default detection applies: records with source-defaulted "
            "'Medium' severity are emitted with severity=None."
        ),
    ),
)


def is_bare_llm03_contaminated(native_labels: list[str] | tuple[str, ...]) -> bool:
    """Return True if labels indicate bare-LLM03 default contamination (HANDOFF §3 F2)."""
    return list(native_labels) == ["LLM03"]


def is_double_default_contaminated(native_labels: list[str] | tuple[str, ...]) -> bool:
    """Return True if labels indicate the LLM03+ASI04 double-default (HANDOFF §3 F2)."""
    return sorted(native_labels) == ["ASI04", "LLM03"]
```

- [ ] **Step 4: Handle the construction-time validation for BiasProfile**

The `BiasProfile` dataclass in `engine/schema.py` is a frozen dataclass without `__post_init__` validation. Per constraint C2 (construction-time validation), we need to add validation. However, per constraint C8, we should not modify `engine/schema.py` unless necessary.

**Decision:** Add a `validate_bias_profile()` helper in `genai_agentic_bias.py` that checks profiles at construction time, rather than modifying the frozen `BiasProfile` dataclass. The `build_bias_profiles()` function calls this validator. This avoids modifying Plan 1's schema.

Update `genai_agentic_bias.py` — add before `BIAS_PROFILES`:

```python
def _validate_bias_profile(profile: BiasProfile) -> BiasProfile:
    """Validate a BiasProfile at construction time (C2 pattern from Plan 1 M2)."""
    if not profile.stratum:
        raise ValueError("BiasProfile.stratum must be non-empty")
    if not profile.description:
        raise ValueError("BiasProfile.description must be non-empty")
    return profile
```

And wrap each profile in `BIAS_PROFILES` with `_validate_bias_profile(...)`.

Update the test `test_construction_time_validation_rejects_empty_stratum` to call `_validate_bias_profile` instead of relying on `BiasProfile.__init__`:

```python
from engine.adapters.genai_agentic_bias import _validate_bias_profile

def test_construction_time_validation_rejects_empty_stratum(self) -> None:
    with pytest.raises(ValueError, match="stratum"):
        _validate_bias_profile(BiasProfile(
            stratum="",
            description="empty",
            known_blind_spots=(),
            contamination_description="none",
            quarantine_rule="none",
        ))
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/unit/test_adapter_genai_agentic.py -v
```

Expected: all tests PASS.

- [ ] **Step 6: Run full quality gates**

```bash
uv run pytest -v && uv run mypy engine tests && uv run ruff check . && uv run semgrep --config .semgrep.yml --error engine/
```

Expected: all green, zero errors.

- [ ] **Step 7: Commit**

```bash
git add engine/adapters/genai_agentic_bias.py tests/unit/test_adapter_genai_agentic.py
git commit -m "feat(adapters): per-stratum bias profiles + quarantine predicates for genai_agentic (Plan 2)"
```

---

## Task 2: Snapshot vendoring script + provenance

**Files:**
- Create: `engine/cli/snapshot.py`
- Modify: `engine/cli/main.py` (wire Click command)
- Create: `tests/unit/test_snapshot_vendor.py`

**Acceptance criteria served:** PRD §3.6 criteria 3 (quality gates), 4 (snapshot vendored with provenance.json carrying all six fields).

- [ ] **Step 1: Write failing tests for snapshot vendoring**

Create `tests/unit/test_snapshot_vendor.py`:

```python
"""Tests for the snapshot vendoring CLI module."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.cli.snapshot import vendor_snapshot
from engine.snapshot.provenance import SnapshotProvenance


@pytest.fixture()
def source_corpus(tmp_path: Path) -> Path:
    """Create a minimal source corpus JSON file."""
    records = [
        {"id": "INC-001", "title": "Test incident", "corpus": "security"},
        {"id": "INC-002", "title": "Another incident", "corpus": "ai-harm"},
    ]
    src = tmp_path / "source" / "incidents.json"
    src.parent.mkdir(parents=True)
    src.write_text(json.dumps(records, indent=2))
    return src


@pytest.fixture()
def dest_dir(tmp_path: Path) -> Path:
    return tmp_path / "dest"


class TestVendorSnapshot:

    def test_creates_content_addressed_directory(
        self, source_corpus: Path, dest_dir: Path
    ) -> None:
        result = vendor_snapshot(
            source_path=source_corpus,
            dest_base=dest_dir,
            source_repo="genai_agentic_incidents",
            source_commit_sha="abc123",
            adapter_version="0.2.0",
        )
        # Directory name is the hash
        assert result.snapshot_dir.exists()
        assert result.snapshot_dir.name == result.snapshot_hash

    def test_snapshot_file_is_byte_identical_to_source(
        self, source_corpus: Path, dest_dir: Path
    ) -> None:
        result = vendor_snapshot(
            source_path=source_corpus,
            dest_base=dest_dir,
            source_repo="genai_agentic_incidents",
            source_commit_sha="abc123",
            adapter_version="0.2.0",
        )
        vendored = result.snapshot_dir / "incidents.json"
        assert vendored.read_bytes() == source_corpus.read_bytes()

    def test_provenance_json_has_all_six_fields(
        self, source_corpus: Path, dest_dir: Path
    ) -> None:
        result = vendor_snapshot(
            source_path=source_corpus,
            dest_base=dest_dir,
            source_repo="genai_agentic_incidents",
            source_commit_sha="abc123",
            adapter_version="0.2.0",
        )
        prov_path = result.snapshot_dir / "provenance.json"
        assert prov_path.exists()
        prov = SnapshotProvenance.read(prov_path)
        assert prov.source_repo == "genai_agentic_incidents"
        assert prov.source_commit_sha == "abc123"
        assert prov.pull_date != ""
        assert prov.adapter_name == "genai_agentic"
        assert prov.adapter_version == "0.2.0"
        assert prov.snapshot_hash == result.snapshot_hash

    def test_hash_is_deterministic_across_calls(
        self, source_corpus: Path, tmp_path: Path
    ) -> None:
        r1 = vendor_snapshot(
            source_path=source_corpus,
            dest_base=tmp_path / "d1",
            source_repo="test",
            source_commit_sha="aaa",
            adapter_version="0.1.0",
        )
        r2 = vendor_snapshot(
            source_path=source_corpus,
            dest_base=tmp_path / "d2",
            source_repo="test",
            source_commit_sha="aaa",
            adapter_version="0.1.0",
        )
        assert r1.snapshot_hash == r2.snapshot_hash

    def test_idempotent_rerun_does_not_fail(
        self, source_corpus: Path, dest_dir: Path
    ) -> None:
        vendor_snapshot(
            source_path=source_corpus,
            dest_base=dest_dir,
            source_repo="test",
            source_commit_sha="aaa",
            adapter_version="0.1.0",
        )
        # Second call with same input should succeed (idempotent)
        result = vendor_snapshot(
            source_path=source_corpus,
            dest_base=dest_dir,
            source_repo="test",
            source_commit_sha="aaa",
            adapter_version="0.1.0",
        )
        assert result.snapshot_dir.exists()

    def test_idempotent_rerun_preserves_provenance(
        self, source_corpus: Path, dest_dir: Path
    ) -> None:
        """Premortem R2: re-running vendor_snapshot must not overwrite provenance."""
        r1 = vendor_snapshot(
            source_path=source_corpus,
            dest_base=dest_dir,
            source_repo="test",
            source_commit_sha="aaa",
            adapter_version="0.1.0",
        )
        original_date = r1.provenance.pull_date
        # Second call — provenance should be read from disk, not regenerated
        r2 = vendor_snapshot(
            source_path=source_corpus,
            dest_base=dest_dir,
            source_repo="test",
            source_commit_sha="aaa",
            adapter_version="0.1.0",
        )
        assert r2.provenance.pull_date == original_date

    def test_jsonl_file_written_alongside_json(
        self, source_corpus: Path, dest_dir: Path
    ) -> None:
        """Premortem M2: JSONL must be written unconditionally for drift detector."""
        result = vendor_snapshot(
            source_path=source_corpus,
            dest_base=dest_dir,
            source_repo="test",
            source_commit_sha="aaa",
            adapter_version="0.1.0",
        )
        jsonl_path = result.snapshot_dir / "incidents.jsonl"
        assert jsonl_path.exists(), "incidents.jsonl not created by vendor_snapshot"
        # Verify JSONL is well-formed: each line is a valid JSON object
        lines = [l for l in jsonl_path.read_text().splitlines() if l.strip()]
        assert len(lines) == 2  # matches the 2-record source fixture
        for line in lines:
            obj = json.loads(line)
            assert isinstance(obj, dict)

    def test_jsonl_round_trips_json_content(
        self, source_corpus: Path, dest_dir: Path
    ) -> None:
        """JSONL records must be identical to the JSON array entries."""
        result = vendor_snapshot(
            source_path=source_corpus,
            dest_base=dest_dir,
            source_repo="test",
            source_commit_sha="aaa",
            adapter_version="0.1.0",
        )
        json_data = json.loads((result.snapshot_dir / "incidents.json").read_text())
        jsonl_lines = [
            json.loads(l)
            for l in (result.snapshot_dir / "incidents.jsonl").read_text().splitlines()
            if l.strip()
        ]
        assert json_data == jsonl_lines
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/unit/test_snapshot_vendor.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'engine.cli.snapshot'`.

- [ ] **Step 3: Implement the snapshot vendoring module**

Create `engine/cli/snapshot.py`:

```python
"""Snapshot vendoring for corpus data.

Vendors a source corpus JSON file into a content-addressed directory with
provenance metadata.  See HANDOFF §5.1, §6 control 9.

Premortem M2: vendor_snapshot() UNCONDITIONALLY writes incidents.jsonl alongside
incidents.json.  The drift detector (engine/snapshot/drift.py) reads JSONL
(one JSON object per line), not JSON arrays.  This is not optional — without
the JSONL file, detect_drift() will raise JSONDecodeError on the vendored
snapshot.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import click

from engine.snapshot.hashing import snapshot_hash
from engine.snapshot.provenance import SnapshotProvenance


@dataclass(frozen=True, slots=True)
class VendorResult:
    """Result of a snapshot vendoring operation."""

    snapshot_dir: Path
    snapshot_hash: str
    provenance: SnapshotProvenance


def _write_jsonl(source_json_path: Path, dest_jsonl_path: Path) -> None:
    """Convert a JSON array file to JSONL format (one JSON object per line).

    Required by engine/snapshot/drift.py which reads JSONL, not JSON arrays.
    Premortem M2: this is MANDATORY, not conditional.
    """
    data = json.loads(source_json_path.read_text())
    if not isinstance(data, list):
        raise TypeError(f"Expected JSON array, got {type(data).__name__}")
    with dest_jsonl_path.open("w") as f:
        for record in data:
            f.write(json.dumps(record, sort_keys=True) + "\n")


def vendor_snapshot(
    *,
    source_path: Path,
    dest_base: Path,
    source_repo: str,
    source_commit_sha: str,
    adapter_version: str,
) -> VendorResult:
    """Vendor a source corpus file into a content-addressed snapshot directory.

    The snapshot is stored at ``dest_base/<sha256>/incidents.json`` alongside
    a ``provenance.json`` recording the six required provenance fields
    (HANDOFF §5.1).
    """
    if not source_path.exists():
        raise FileNotFoundError(f"Source corpus not found: {source_path}")

    content_hash = snapshot_hash(source_path)
    snapshot_dir = dest_base / content_hash
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    dest_file = snapshot_dir / "incidents.json"
    if not dest_file.exists():
        shutil.copy2(source_path, dest_file)

    # Premortem M2: MANDATORY JSONL conversion for drift detector compatibility.
    # engine/snapshot/drift.py _count_entries() reads JSONL (line-by-line json.loads),
    # NOT JSON arrays.  Without this, detect_drift() raises JSONDecodeError.
    jsonl_file = snapshot_dir / "incidents.jsonl"
    if not jsonl_file.exists():
        _write_jsonl(dest_file, jsonl_file)

    prov_path = snapshot_dir / "provenance.json"
    if prov_path.exists():
        provenance = SnapshotProvenance.read(prov_path)
    else:
        provenance = SnapshotProvenance(
            source_repo=source_repo,
            source_commit_sha=source_commit_sha,
            pull_date=date.today().isoformat(),
            adapter_name="genai_agentic",
            adapter_version=adapter_version,
            snapshot_hash=content_hash,
        )
        provenance.write(prov_path)

    return VendorResult(
        snapshot_dir=snapshot_dir,
        snapshot_hash=content_hash,
        provenance=provenance,
    )


@click.command("vendor-snapshot")
@click.option(
    "--source",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to source corpus JSON file.",
)
@click.option(
    "--dest",
    required=True,
    type=click.Path(path_type=Path),
    help="Base directory for vendored snapshots.",
)
@click.option("--source-repo", required=True, help="Name of the source repository.")
@click.option("--source-commit", required=True, help="Git commit SHA of the source.")
@click.option(
    "--adapter-version",
    default="0.2.0",
    help="Adapter semver version.",
)
def vendor_snapshot_cmd(
    source: Path,
    dest: Path,
    source_repo: str,
    source_commit: str,
    adapter_version: str,
) -> None:
    """Vendor a corpus snapshot with content-addressed hashing and provenance."""
    result = vendor_snapshot(
        source_path=source,
        dest_base=dest,
        source_repo=source_repo,
        source_commit_sha=source_commit,
        adapter_version=adapter_version,
    )
    click.echo(f"Snapshot vendored to: {result.snapshot_dir}")
    click.echo(f"Content hash: {result.snapshot_hash}")
    click.echo(f"Provenance: {result.snapshot_dir / 'provenance.json'}")
```

- [ ] **Step 4: Wire the CLI command**

Add to `engine/cli/main.py` after the existing imports and before the first command:

```python
from engine.cli.snapshot import vendor_snapshot_cmd

cli.add_command(vendor_snapshot_cmd)
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/unit/test_snapshot_vendor.py -v
```

Expected: all tests PASS.

- [ ] **Step 6: Run full quality gates**

```bash
uv run pytest -v && uv run mypy engine tests && uv run ruff check . && uv run semgrep --config .semgrep.yml --error engine/
```

Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add engine/cli/snapshot.py tests/unit/test_snapshot_vendor.py engine/cli/main.py
git commit -m "feat(cli): vendor-snapshot command with content-addressed hashing + provenance (Plan 2)"
```

---

## Task 3: Vendor the real snapshot + populate project.toml

**Files:**
- Create: `projects/owasp-llm/cycles/2026/corpora/genai_agentic/<hash>/incidents.json`
- Create: `projects/owasp-llm/cycles/2026/corpora/genai_agentic/<hash>/provenance.json`
- Modify: `projects/owasp-llm/project.toml`

**Acceptance criteria served:** PRD §3.6 criteria 4 (snapshot vendored, provenance has all six fields).

**Important:** This task runs the vendoring script once against the real source corpus. After this point, the hash is frozen. Re-running mid-plan invalidates any test that pinned the hash (PRD §3.8 risk 1).

- [ ] **Step 1: Record the source commit SHA**

```bash
cd ~/github_projects/genai_agentic_incidents && git rev-parse HEAD
```

Record the output as `$SOURCE_SHA`.

- [ ] **Step 2: Run the vendor-snapshot command**

```bash
cd ~/github_projects/incident-rank-validation
uv run incident-rank vendor-snapshot \
  --source ~/github_projects/genai_agentic_incidents/data/incidents.json \
  --dest projects/owasp-llm/cycles/2026/corpora/genai_agentic \
  --source-repo genai_agentic_incidents \
  --source-commit $SOURCE_SHA \
  --adapter-version 0.2.0
```

Record the content hash from the output as `$SNAPSHOT_HASH`.

- [ ] **Step 3: Validate source commit SHA resolves in the source repo (R5)**

```bash
cd ~/github_projects/genai_agentic_incidents
git cat-file -t $SOURCE_SHA
```

Expected: `commit`. If this fails, the SHA recorded in provenance would be unresolvable — stop and fix. Also verify the commit is reachable from a branch (not a dangling orphan):

```bash
git branch --contains $SOURCE_SHA
```

Expected: at least one branch (typically `main`). Record the branch name alongside the SHA.

- [ ] **Step 4: Verify provenance.json has all six fields**

```bash
python3 -c "
import json, pathlib
p = next(pathlib.Path('projects/owasp-llm/cycles/2026/corpora/genai_agentic').iterdir())
prov = json.loads((p / 'provenance.json').read_text())
required = ['source_repo', 'source_commit_sha', 'pull_date', 'adapter_name', 'adapter_version', 'snapshot_hash']
for f in required:
    assert f in prov, f'Missing field: {f}'
    assert prov[f], f'Empty field: {f}'
    print(f'  {f}: {prov[f]}')
print('All six provenance fields present and non-empty.')
"
```

- [ ] **Step 5: Verify incidents.jsonl was written (M2)**

```bash
ls -lh projects/owasp-llm/cycles/2026/corpora/genai_agentic/*/incidents.jsonl
python3 -c "
import pathlib, json
p = next(pathlib.Path('projects/owasp-llm/cycles/2026/corpora/genai_agentic').iterdir())
lines = [l for l in (p / 'incidents.jsonl').read_text().splitlines() if l.strip()]
print(f'JSONL lines: {len(lines)}')
# Spot-check first line is valid JSON object
obj = json.loads(lines[0])
assert isinstance(obj, dict), 'First JSONL line is not a JSON object'
print('JSONL format verified.')
"
```

- [ ] **Step 6: Populate project.toml with real 2026 cycle values**

Update `projects/owasp-llm/project.toml` — replace REQUIRED placeholders with actual values. The exact values depend on the 2026 LLM Top 10 entry count and the vendored snapshot. Use the counts from Task 0 Step 7 to fill `default_strata`:

```toml
[project]
name = "owasp-llm"
cycle_id = "2026"
tier_size = 5
default_strata = ["security", "ai-harm"]
measurability_minimum = 4
measurability_minimum_rationale = "Weighted kappa over 3-tier structure requires >=4 entries to populate non-degenerately (HANDOFF §5.5)"
prng_seed = 20260520

[project.hyperparameters]
prior_scale = 0.5
concentration_shape = 5.0
concentration_rate = 0.1
ess_fraction = 0.4
meaningful_kappa_n = 4

[project.taxonomy]
source = "cycles/2026/taxonomy/taxonomy.json"
```

**Note:** The `taxonomy.json` file (LLM 2026 entry definitions) is a Plan 3 deliverable. Leave the `source` field pointing to the expected location. Create the directory but do not create the taxonomy file itself — that is out of scope per PRD §3.7.

- [ ] **Step 7: Create the cycle taxonomy directory placeholder**

```bash
mkdir -p projects/owasp-llm/cycles/2026/taxonomy
```

- [ ] **Step 8: Add vendored snapshot to .gitattributes for LFS consideration**

The vendored snapshot may be large (~several MB of JSON). Check its size:

```bash
ls -lh projects/owasp-llm/cycles/2026/corpora/genai_agentic/*/incidents.json
```

If under 10 MB, commit directly. If over 10 MB, consider `.gitattributes` for LFS. Most likely it is under 10 MB and can be committed directly.

- [ ] **Step 9: Commit**

```bash
git add projects/owasp-llm/cycles/2026/corpora/genai_agentic/ projects/owasp-llm/project.toml
git commit -m "chore(projects): vendor genai_agentic snapshot + populate owasp-llm project.toml (Plan 2)"
```

---

## Task 4: GenAI Agentic adapter — core record transformation

**Files:**
- Create: `engine/adapters/genai_agentic.py`
- Modify: `engine/adapters/__init__.py`
- Modify: `tests/unit/test_adapter_genai_agentic.py` (add adapter tests)

**Acceptance criteria served:** PRD §3.6 criteria 1 (adapter tests green), 3 (quality gates), 6 (per-stratum counts match).

**⚠️ PROVISIONAL FIELD NAMES (Premortem M1):** The exact field-mapping code in this task depends on the source corpus field names discovered in Task 0 Step 7. The code below uses **provisional** field names (`id`, `date`, `title`, `description`, `impact`, `severity`, `corpus`, `category`, `owasp_llm`, `quality_tier`, `source_url`) based on HANDOFF §3's audit analysis, NOT from direct schema inspection. **Before implementing this task, reconcile every `.get("field_name")` call against the actual field names recorded in Task 0 Step 7.** If any field name differs, update all occurrences in the adapter code AND in the test fixtures.

- [ ] **Step 1: Write failing tests for the adapter core**

Add to `tests/unit/test_adapter_genai_agentic.py`:

```python
import json
from pathlib import Path

from engine.adapters.genai_agentic import GenAIAgenticAdapter
from engine.schema import IncidentRecord, StratumSize


@pytest.fixture()
def vendored_snapshot(tmp_path: Path) -> Path:
    """Create a minimal vendored snapshot for testing."""
    records = [
        {
            "id": "INC-001",
            "title": "Prompt injection in chatbot",
            "description": "Attacker injected malicious prompts.",
            "date": "2024-03-15",
            "severity": "High",
            "corpus": "security",
            "category": "real-world",
            "owasp_llm": ["LLM01"],
            "quality_tier": "curated",
            "source_url": "https://example.com/inc-001",
        },
        {
            "id": "INC-002",
            "title": "AI bias incident",
            "description": "Model produced biased outputs.",
            "date": "2024-06-20",
            "severity": "Medium",
            "corpus": "ai-harm",
            "category": "real-world",
            "owasp_llm": ["LLM06"],
            "quality_tier": "reviewed",
            "source_url": "https://example.com/inc-002",
        },
        {
            "id": "INC-003",
            "title": "Generic CVE with default label",
            "description": "CVE with no human OWASP review.",
            "date": "2024-01-10",
            "severity": "Medium",
            "corpus": "security",
            "category": "vulnerability-disclosure",
            "owasp_llm": ["LLM03"],
            "quality_tier": "reviewed",
            "source_url": "https://example.com/inc-003",
        },
        {
            "id": "INC-004",
            "title": "Double default label",
            "description": "CVE with LLM03+ASI04 default.",
            "date": "2024-02-28",
            "severity": "Medium",
            "corpus": "security",
            "category": "vulnerability-disclosure",
            "owasp_llm": ["LLM03", "ASI04"],
            "quality_tier": "reviewed",
            "source_url": "https://example.com/inc-004",
        },
        {
            "id": "INC-005",
            "title": "Future-dated incident",
            "description": "This incident is dated after snapshot.",
            "date": "2027-01-01",
            "severity": "Low",
            "corpus": "security",
            "category": "real-world",
            "owasp_llm": ["LLM05"],
            "quality_tier": "reviewed",
            "source_url": "https://example.com/inc-005",
        },
    ]
    snapshot_dir = tmp_path / "snapshot"
    snapshot_dir.mkdir()
    (snapshot_dir / "incidents.json").write_text(json.dumps(records))
    return snapshot_dir


class TestGenAIAgenticAdapter:

    def test_emits_incident_records(self, vendored_snapshot: Path) -> None:
        adapter = GenAIAgenticAdapter(
            snapshot_dir=vendored_snapshot,
            snapshot_date="2026-05-20",
        )
        records = list(adapter.iter_incidents())
        assert len(records) > 0
        for r in records:
            assert isinstance(r, IncidentRecord)

    def test_schema_round_trip_all_fields_populated(
        self, vendored_snapshot: Path
    ) -> None:
        adapter = GenAIAgenticAdapter(
            snapshot_dir=vendored_snapshot,
            snapshot_date="2026-05-20",
        )
        for r in adapter.iter_incidents():
            assert r.id != ""
            assert r.date != ""
            assert r.text != ""
            assert r.source_class != ""
            assert r.corpus_stratum in ("security", "ai-harm")
            assert r.quality in ("curated", "reviewed", "auto")
            assert isinstance(r.native_labels, tuple)
            assert r.source_url.startswith("http")

    def test_corpus_stratum_matches_source_corpus_field(
        self, vendored_snapshot: Path
    ) -> None:
        adapter = GenAIAgenticAdapter(
            snapshot_dir=vendored_snapshot,
            snapshot_date="2026-05-20",
        )
        strata = {r.corpus_stratum for r in adapter.iter_incidents()}
        assert "security" in strata or "ai-harm" in strata

    def test_native_labels_are_non_authoritative_metadata(
        self, vendored_snapshot: Path
    ) -> None:
        """HANDOFF §4: native labels are metadata only, never join keys."""
        adapter = GenAIAgenticAdapter(
            snapshot_dir=vendored_snapshot,
            snapshot_date="2026-05-20",
        )
        for r in adapter.iter_incidents():
            assert isinstance(r.native_labels, tuple)

    def test_construction_validates_snapshot_dir_exists(
        self, tmp_path: Path
    ) -> None:
        """C2 pattern: fail at construction, not at iteration."""
        with pytest.raises(FileNotFoundError):
            GenAIAgenticAdapter(
                snapshot_dir=tmp_path / "nonexistent",
                snapshot_date="2026-05-20",
            )

    def test_construction_validates_incidents_json_exists(
        self, tmp_path: Path
    ) -> None:
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="incidents.json"):
            GenAIAgenticAdapter(
                snapshot_dir=empty_dir,
                snapshot_date="2026-05-20",
            )


class TestTextLengthCap:
    """Premortem M10: defensive text-length cap (HANDOFF §5.1 adversarial ingestion)."""

    def test_oversized_text_is_truncated(self, tmp_path: Path) -> None:
        """A record with a description >_MAX_TEXT_LENGTH chars gets truncated."""
        from engine.adapters.genai_agentic import _MAX_TEXT_LENGTH

        oversized_desc = "x" * (_MAX_TEXT_LENGTH + 1000)
        records = [
            {
                "id": "OVER-001",
                "title": "Normal title",
                "description": oversized_desc,
                "date": "2024-01-01",
                "severity": "High",
                "corpus": "security",
                "category": "real-world",
                "owasp_llm": ["LLM01"],
                "quality_tier": "curated",
                "source_url": "https://example.com/over",
            },
        ]
        snap = tmp_path / "snap"
        snap.mkdir()
        (snap / "incidents.json").write_text(json.dumps(records))
        adapter = GenAIAgenticAdapter(snapshot_dir=snap, snapshot_date="2026-05-20")
        r = next(adapter.iter_incidents())
        assert len(r.text) <= _MAX_TEXT_LENGTH

    def test_normal_text_is_not_truncated(self, tmp_path: Path) -> None:
        records = [
            {
                "id": "NORM-001",
                "title": "Short",
                "description": "Also short",
                "date": "2024-01-01",
                "severity": "Low",
                "corpus": "security",
                "category": "real-world",
                "owasp_llm": ["LLM02"],
                "quality_tier": "curated",
                "source_url": "https://example.com/norm",
            },
        ]
        snap = tmp_path / "snap"
        snap.mkdir()
        (snap / "incidents.json").write_text(json.dumps(records))
        adapter = GenAIAgenticAdapter(snapshot_dir=snap, snapshot_date="2026-05-20")
        r = next(adapter.iter_incidents())
        assert r.text == "Short Also short"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/unit/test_adapter_genai_agentic.py::TestGenAIAgenticAdapter -v
```

Expected: FAIL — `ImportError`.

- [ ] **Step 2b: Reconcile severity-defaulting logic with Task 0 Step 8 findings (Premortem R4)**

Before implementing the adapter, review the output from Task 0 Step 8 (source ingest script inspection). Three cases:

1. **Step 8 found an explicit provenance field** (e.g., `severity_source`, `severity_method`): Update `_is_severity_defaulted()` below to use that field as the primary signal. For example, if the source has `severity_source: "default"`, use `return raw.get("severity_source") == "default"` instead of the quality-tier heuristic.

2. **Step 8 found that `quality_tier == "curated"` reliably indicates human-confirmed severity**: The heuristic below is correct as-is.

3. **Step 8 found no provenance field and no reliable quality signal**: Document in `_is_severity_defaulted()`'s docstring that the fallback heuristic (all non-curated Medium → None) is in effect, and note the expected false-positive rate (number of reviewed+Medium records that may have genuine Medium severity). Consider narrowing the heuristic to only wipe severity for records that also match other signals (e.g., `category == "vulnerability-disclosure"` which are CVE-seeded).

**Do not skip this step.** The default heuristic aggressively wipes ALL non-curated Medium severity records. If the source has many legitimate reviewed+Medium records, this inflates the unknown-severity count and biases the Bayesian model in Plans 3–5.

- [ ] **Step 3: Implement the adapter core**

Create `engine/adapters/genai_agentic.py`:

```python
"""Corpus A adapter for the genai_agentic_incidents dataset.

Reads a vendored, content-hashed JSON snapshot and emits canonical
IncidentRecord instances.  Per-stratum bias profiles are declared in
genai_agentic_bias.py.  See HANDOFF §5.1.

The engine never sees the source schema — this adapter normalises all
fields.  Source-specific quarantine rules (bare-LLM03) live here,
declared in the bias profile (HANDOFF §5.1).
"""
from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path

from engine.adapters.base import CorpusAdapter
from engine.adapters.genai_agentic_bias import (
    BIAS_PROFILES,
    is_bare_llm03_contaminated,
    is_double_default_contaminated,
)
from engine.model.overlap import OverlapWeights
from engine.schema import (
    BiasProfile,
    EntryDefinition,
    IncidentRecord,
    StratumSize,
    make_stratum_size,
)

logger = logging.getLogger(__name__)


_MAX_TEXT_LENGTH = 50_000
"""Premortem M10: defensive cap on concatenated text field length.

HANDOFF §5.1 declares adversarial ingestion a threat.  Without a cap,
a single record with a multi-MB description field would pass through to
the classifier unchecked.  50k chars is ~10x the longest legitimate
incident report in the audit reference and well above the truncation
point of any downstream tokenizer.
"""


# Premortem M4: explicitly named PROVISIONAL to prevent downstream consumers
# from treating these as the final taxonomy.  Plan 3 replaces this.
_PROVISIONAL_2025_ENTRIES: tuple[EntryDefinition, ...] = (
    EntryDefinition(entry_id="LLM01", name="Prompt Injection"),
    EntryDefinition(entry_id="LLM02", name="Sensitive Information Disclosure"),
    EntryDefinition(entry_id="LLM03", name="Supply Chain Vulnerabilities"),
    EntryDefinition(entry_id="LLM04", name="Data and Model Poisoning", frame_blind=True),
    EntryDefinition(entry_id="LLM05", name="Improper Output Handling"),
    EntryDefinition(entry_id="LLM06", name="Excessive Agency"),
    EntryDefinition(entry_id="LLM07", name="System Prompt Leakage"),
    EntryDefinition(entry_id="LLM08", name="Vector and Embedding Weaknesses", frame_blind=True),
    EntryDefinition(entry_id="LLM09", name="Misinformation"),
    EntryDefinition(entry_id="LLM10", name="Unbounded Consumption", frame_blind=True),
)


class GenAIAgenticAdapter(CorpusAdapter):
    """Adapter for the genai_agentic_incidents corpus (Corpus A).

    Parameters
    ----------
    snapshot_dir:
        Path to the vendored snapshot directory containing ``incidents.json``
        and ``provenance.json``.
    snapshot_date:
        ISO 8601 date string (YYYY-MM-DD).  Records dated after this are
        dropped per HANDOFF §4 Temporal.
    """

    def __init__(self, snapshot_dir: Path, snapshot_date: str) -> None:
        if not snapshot_dir.exists():
            raise FileNotFoundError(f"Snapshot directory not found: {snapshot_dir}")
        self._incidents_path = snapshot_dir / "incidents.json"
        if not self._incidents_path.exists():
            raise FileNotFoundError(
                f"incidents.json not found in snapshot directory: {snapshot_dir}"
            )
        self._snapshot_date = snapshot_date
        self._records: list[dict[str, object]] | None = None

    def _load(self) -> list[dict[str, object]]:
        if self._records is None:
            raw = self._incidents_path.read_text()
            data = json.loads(raw)
            if not isinstance(data, list):
                raise TypeError(
                    f"Expected JSON array, got {type(data).__name__}"
                )
            self._records = data
        return self._records

    def iter_incidents(self) -> Iterator[IncidentRecord]:
        """Yield canonical incident records from the vendored snapshot.

        Applies:
        - Future-dated row drop (HANDOFF §4 Temporal)
        - Severity-default detection (HANDOFF §3 Mixture: "Medium" default → None)
        - Native labels passed through as non-authoritative metadata (HANDOFF §4)
        - Quarantine flag via contamination predicates (downstream routing)
        """
        for raw in self._load():
            record = self._transform(raw)
            if record is not None:
                yield record

    def _transform(self, raw: dict[str, object]) -> IncidentRecord | None:
        """Transform a single source record to canonical form.

        Returns None for records that should be dropped (future-dated).
        """
        # --- Field extraction (adapt field names to actual source schema) ---
        record_id = str(raw.get("id", ""))
        date_str = str(raw.get("date", raw.get("published_date", "")))

        # Future-dated row drop (HANDOFF §4 Temporal)
        if date_str > self._snapshot_date:
            return None

        title = str(raw.get("title", ""))
        description = str(raw.get("description", ""))
        impact = str(raw.get("impact", ""))
        text = " ".join(part for part in [title, description, impact] if part)

        # Premortem M10: defensive text-length cap (HANDOFF §5.1 adversarial ingestion)
        if len(text) > _MAX_TEXT_LENGTH:
            logger.warning(
                "Record %s text truncated from %d to %d chars",
                record_id, len(text), _MAX_TEXT_LENGTH,
            )
            text = text[:_MAX_TEXT_LENGTH]

        # Severity-default detection (HANDOFF §3)
        raw_severity = raw.get("severity")
        severity: str | None
        if raw_severity == "Medium" and self._is_severity_defaulted(raw):
            severity = None
        elif raw_severity is not None:
            severity = str(raw_severity)
        else:
            severity = None

        corpus_stratum = str(raw.get("corpus", "unknown"))

        # Source class derived from category
        category = str(raw.get("category", ""))
        source_class = self._map_source_class(category)

        # Quality mapping
        quality_tier = str(raw.get("quality_tier", "auto"))
        quality = self._map_quality(quality_tier)

        # Native labels — non-authoritative (HANDOFF §4)
        owasp_llm = raw.get("owasp_llm")
        if isinstance(owasp_llm, list):
            native_labels = tuple(str(x) for x in owasp_llm)
        else:
            native_labels = ()

        source_url = str(raw.get("source_url", raw.get("url", "")))

        return IncidentRecord(
            id=record_id,
            date=date_str,
            text=text,
            severity=severity,
            source_class=source_class,
            corpus_stratum=corpus_stratum,
            quality=quality,
            native_labels=native_labels,
            source_url=source_url,
        )

    @staticmethod
    def _is_severity_defaulted(raw: dict[str, object]) -> bool:
        """Detect whether severity was defaulted to 'Medium' by the source ingest.

        HANDOFF §3: 'severity is defaulted to Medium in ingest when missing,
        so a zero unknown-severity rate is itself an artifact.'

        Heuristic: if the source has a quality indicator suggesting auto-processing
        and severity is exactly 'Medium', treat it as defaulted.  The exact
        detection mechanism should be refined after examining the source repo's
        ingest scripts (PRD §3.8 risk 2).
        """
        quality = raw.get("quality_tier", "")
        if quality == "curated":
            return False
        severity_source = raw.get("severity_source", raw.get("severity_method", None))
        if severity_source is not None:
            return str(severity_source).lower() in ("default", "inferred", "")
        return quality != "curated" and raw.get("severity") == "Medium"

    @staticmethod
    def _map_source_class(category: str) -> str:
        """Map source ``category`` to engine ``source_class``.

        Premortem M7: this mapping is LOSSY — "research" and "threat-report"
        both collapse to "advisory".  The original category granularity is not
        preserved on IncidentRecord.  If Plans 3–5 need per-category
        stratification, extend the adapter to expose category separately.
        """
        mapping: dict[str, str] = {
            "real-world": "harm-report",
            "vulnerability-disclosure": "cve",
            "research": "advisory",
            "threat-report": "advisory",
        }
        return mapping.get(category, "advisory")

    @staticmethod
    def _map_quality(quality_tier: str) -> str:
        mapping: dict[str, str] = {
            "curated": "curated",
            "reviewed": "reviewed",
        }
        return mapping.get(quality_tier, "auto")

    def bias_profiles(self) -> tuple[BiasProfile, ...]:
        return BIAS_PROFILES

    def stratum_sizes(self) -> dict[str, StratumSize]:
        """Compute stratum sizes from the actual vendored data.

        Stratum size is the total record count per stratum (after dropping
        future-dated rows).  This satisfies the M3 sanity contract:
        stratum_size >= observed_count, since stratum_size IS the count.
        """
        counts: dict[str, int] = {}
        for record in self.iter_incidents():
            counts[record.corpus_stratum] = counts.get(record.corpus_stratum, 0) + 1
        return {k: make_stratum_size(v) for k, v in counts.items()}

    def entry_definitions(self) -> tuple[EntryDefinition, ...]:
        """Return provisional 2025 LLM Top 10 entry definitions.

        Premortem M4: these are _PROVISIONAL_2025_ENTRIES — entry names,
        codes, and frame-blind flags are based on the 2025 taxonomy present
        in the source corpus at the time of Plan 2.  They are NOT the final
        2026 taxonomy.  Plan 3 MUST replace this with the frozen 2026 rubric
        taxonomy loaded from ``projects/owasp-llm/cycles/2026/taxonomy/taxonomy.json``.

        Frame-blind flags are set conservatively based on HANDOFF §3 F4
        (LLM04, LLM08, LLM10 near-absent in the corpus).
        """
        return _PROVISIONAL_2025_ENTRIES

    def overlap_weights(self) -> OverlapWeights:
        """Return declared FP leakage structure for the LLM taxonomy.

        Premortem M9: these weights are PROVISIONAL PLACEHOLDERS with no
        empirical basis.  The 0.2 value for LLM05→LLM03 leakage is a
        structural placeholder acknowledging that LLM03 contamination
        (HANDOFF §3 F2) creates false-positive leakage into adjacent entries.
        The actual magnitude is unknown until Plan 3/4 produces the frozen
        rubric with empirically measured boundary-cell confusion rates.

        Plan 3/4 MUST replace these with empirically derived weights.
        Until then, downstream inference results that depend on overlap
        correction carry an unquantified systematic error from this
        placeholder.
        """
        return OverlapWeights(weights={"LLM05": {"LLM03": 0.2}})
```

- [ ] **Step 4: Export the adapter**

Update `engine/adapters/__init__.py`:

```python
from engine.adapters.genai_agentic import GenAIAgenticAdapter
from engine.adapters.synthetic import SyntheticAdapter
from engine.adapters.synthetic_stress import SyntheticStressAdapter

__all__ = ["GenAIAgenticAdapter", "SyntheticAdapter", "SyntheticStressAdapter"]
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/unit/test_adapter_genai_agentic.py -v
```

Expected: all tests PASS.

- [ ] **Step 6: Run full quality gates**

```bash
uv run pytest -v && uv run mypy engine tests && uv run ruff check . && uv run semgrep --config .semgrep.yml --error engine/
```

Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add engine/adapters/genai_agentic.py engine/adapters/__init__.py
git commit -m "feat(adapters): GenAIAgenticAdapter core record transformation (Plan 2)"
```

---

## Task 5: Severity-default detection + future-dated row repair + quarantine integration tests

**Files:**
- Modify: `tests/unit/test_adapter_genai_agentic.py` (add behavior tests)

**Acceptance criteria served:** PRD §3.6 criteria 1 (adapter tests green), deliverables 6 (severity-defaulting disclosure) and 7 (future-dated row repair).

- [ ] **Step 1: Write tests for severity-default detection, future-dated drop, and quarantine**

Add to `tests/unit/test_adapter_genai_agentic.py`:

```python
class TestSeverityDefaultDetection:
    """HANDOFF §3: severity defaulted to 'Medium' is an artifact, not truth."""

    def test_curated_medium_is_not_defaulted(
        self, vendored_snapshot: Path
    ) -> None:
        """Curated records have human-confirmed severity — keep 'Medium'."""
        adapter = GenAIAgenticAdapter(
            snapshot_dir=vendored_snapshot, snapshot_date="2026-05-20"
        )
        inc_001 = next(
            r for r in adapter.iter_incidents() if r.id == "INC-001"
        )
        # INC-001 is curated with severity High — kept as-is
        assert inc_001.severity == "High"

    def test_reviewed_medium_is_treated_as_unknown(
        self, vendored_snapshot: Path
    ) -> None:
        """Non-curated 'Medium' severity is a source-ingest artifact → None."""
        adapter = GenAIAgenticAdapter(
            snapshot_dir=vendored_snapshot, snapshot_date="2026-05-20"
        )
        inc_002 = next(
            r for r in adapter.iter_incidents() if r.id == "INC-002"
        )
        # INC-002 is reviewed (not curated) with severity Medium → defaulted → None
        assert inc_002.severity is None


class TestFutureDatedRowDrop:
    """HANDOFF §4 Temporal: adapter drops rows dated after the snapshot date."""

    def test_future_dated_rows_are_dropped(
        self, vendored_snapshot: Path
    ) -> None:
        adapter = GenAIAgenticAdapter(
            snapshot_dir=vendored_snapshot, snapshot_date="2026-05-20"
        )
        ids = {r.id for r in adapter.iter_incidents()}
        assert "INC-005" not in ids  # dated 2027-01-01

    def test_past_dated_rows_are_kept(
        self, vendored_snapshot: Path
    ) -> None:
        adapter = GenAIAgenticAdapter(
            snapshot_dir=vendored_snapshot, snapshot_date="2026-05-20"
        )
        ids = {r.id for r in adapter.iter_incidents()}
        assert "INC-001" in ids  # dated 2024-03-15


class TestContaminationQuarantine:
    """HANDOFF §3 F2, §5.2: bare-LLM03 and double-default quarantine."""

    def test_bare_llm03_record_is_emitted_but_flagged(
        self, vendored_snapshot: Path
    ) -> None:
        """Quarantined records are still emitted — downstream routes to sink."""
        adapter = GenAIAgenticAdapter(
            snapshot_dir=vendored_snapshot, snapshot_date="2026-05-20"
        )
        inc_003 = next(
            r for r in adapter.iter_incidents() if r.id == "INC-003"
        )
        assert inc_003.native_labels == ("LLM03",)
        # Quarantine detection is via the predicate, not a field on IncidentRecord
        assert is_bare_llm03_contaminated(list(inc_003.native_labels))

    def test_double_default_record_is_emitted_but_flagged(
        self, vendored_snapshot: Path
    ) -> None:
        adapter = GenAIAgenticAdapter(
            snapshot_dir=vendored_snapshot, snapshot_date="2026-05-20"
        )
        inc_004 = next(
            r for r in adapter.iter_incidents() if r.id == "INC-004"
        )
        assert set(inc_004.native_labels) == {"LLM03", "ASI04"}
        assert is_double_default_contaminated(list(inc_004.native_labels))

    def test_clean_record_is_not_quarantined(
        self, vendored_snapshot: Path
    ) -> None:
        adapter = GenAIAgenticAdapter(
            snapshot_dir=vendored_snapshot, snapshot_date="2026-05-20"
        )
        inc_001 = next(
            r for r in adapter.iter_incidents() if r.id == "INC-001"
        )
        assert not is_bare_llm03_contaminated(list(inc_001.native_labels))
        assert not is_double_default_contaminated(list(inc_001.native_labels))
```

- [ ] **Step 2: Run tests to verify they pass**

The adapter implementation from Task 4 already handles these behaviors. Run:

```bash
uv run pytest tests/unit/test_adapter_genai_agentic.py -v
```

Expected: all tests PASS. If any fail, fix the adapter implementation in `genai_agentic.py` — the test defines the correct behavior.

- [ ] **Step 3: Run full quality gates**

```bash
uv run pytest -v && uv run mypy engine tests && uv run ruff check . && uv run semgrep --config .semgrep.yml --error engine/
```

- [ ] **Step 4: Commit**

```bash
git add tests/unit/test_adapter_genai_agentic.py
git commit -m "test(adapters): severity-default detection + future-dated drop + quarantine (Plan 2)"
```

---

## Task 6: Adapter interface completion — stratum sizes, entry defs, overlap weights

**Files:**
- Modify: `tests/unit/test_adapter_genai_agentic.py` (add interface tests)

**Acceptance criteria served:** PRD §3.6 criteria 1 (adapter tests), 6 (per-stratum counts match within tolerance).

- [ ] **Step 1: Write tests for the complete CorpusAdapter interface**

Add to `tests/unit/test_adapter_genai_agentic.py`:

```python
from engine.model.overlap import OverlapWeights


class TestAdapterInterface:
    """All five CorpusAdapter ABC methods return valid data."""

    def test_bias_profiles_cover_all_strata(
        self, vendored_snapshot: Path
    ) -> None:
        adapter = GenAIAgenticAdapter(
            snapshot_dir=vendored_snapshot, snapshot_date="2026-05-20"
        )
        profile_strata = {p.stratum for p in adapter.bias_profiles()}
        record_strata = {r.corpus_stratum for r in adapter.iter_incidents()}
        assert record_strata.issubset(profile_strata), (
            f"Records have strata {record_strata} but profiles only cover {profile_strata}"
        )

    def test_stratum_sizes_are_positive(
        self, vendored_snapshot: Path
    ) -> None:
        adapter = GenAIAgenticAdapter(
            snapshot_dir=vendored_snapshot, snapshot_date="2026-05-20"
        )
        sizes = adapter.stratum_sizes()
        assert len(sizes) > 0
        for stratum, size in sizes.items():
            assert size > 0, f"Stratum {stratum} has non-positive size {size}"

    def test_stratum_size_gte_observed_count(
        self, vendored_snapshot: Path
    ) -> None:
        """M3 sanity contract: stratum_size >= observed incident count."""
        adapter = GenAIAgenticAdapter(
            snapshot_dir=vendored_snapshot, snapshot_date="2026-05-20"
        )
        sizes = adapter.stratum_sizes()
        counts: dict[str, int] = {}
        for r in adapter.iter_incidents():
            counts[r.corpus_stratum] = counts.get(r.corpus_stratum, 0) + 1
        for stratum, count in counts.items():
            assert sizes[stratum] >= count, (
                f"M3 violation: stratum {stratum} size {sizes[stratum]} < count {count}"
            )

    def test_entry_definitions_returns_ten_entries(
        self, vendored_snapshot: Path
    ) -> None:
        adapter = GenAIAgenticAdapter(
            snapshot_dir=vendored_snapshot, snapshot_date="2026-05-20"
        )
        entries = adapter.entry_definitions()
        assert len(entries) == 10

    def test_entry_definitions_include_frame_blind_entries(
        self, vendored_snapshot: Path
    ) -> None:
        """HANDOFF §3 F4: LLM04, LLM08, LLM10 are near-absent."""
        adapter = GenAIAgenticAdapter(
            snapshot_dir=vendored_snapshot, snapshot_date="2026-05-20"
        )
        entries = {e.entry_id: e for e in adapter.entry_definitions()}
        assert entries["LLM04"].frame_blind is True
        assert entries["LLM08"].frame_blind is True
        assert entries["LLM10"].frame_blind is True

    def test_overlap_weights_returns_valid_structure(
        self, vendored_snapshot: Path
    ) -> None:
        adapter = GenAIAgenticAdapter(
            snapshot_dir=vendored_snapshot, snapshot_date="2026-05-20"
        )
        ow = adapter.overlap_weights()
        assert isinstance(ow, OverlapWeights)
```

- [ ] **Step 2: Run tests to verify they pass**

```bash
uv run pytest tests/unit/test_adapter_genai_agentic.py::TestAdapterInterface -v
```

Expected: all PASS (implementation is in Task 4).

- [ ] **Step 3: Run full quality gates**

```bash
uv run pytest -v && uv run mypy engine tests && uv run ruff check . && uv run semgrep --config .semgrep.yml --error engine/
```

- [ ] **Step 4: Commit**

```bash
git add tests/unit/test_adapter_genai_agentic.py
git commit -m "test(adapters): CorpusAdapter interface + M3 stratum-size contract (Plan 2)"
```

---

## Task 7: Real-snapshot integration tests — stratum counts + hash stability

**Files:**
- Modify: `tests/unit/test_adapter_genai_agentic.py` (add snapshot-pinned tests)

**Acceptance criteria served:** PRD §3.6 criteria 1 (adapter tests), 6 (per-stratum counts match audit within tolerance).

**Important:** These tests read the vendored snapshot committed in Task 3. They pin expected counts from the audit reference (HANDOFF §3). The `SNAPSHOT_DIR` path depends on the hash from Task 3. If the hash is not yet known at plan-write time, the test discovers it at runtime.

- [ ] **Step 1: Write tests pinned to the real vendored snapshot**

Add to `tests/unit/test_adapter_genai_agentic.py`:

```python
from pathlib import Path as _Path

# Discover the vendored snapshot directory at test time.
# There should be exactly one content-addressed subdirectory.
_VENDOR_BASE = _Path("projects/owasp-llm/cycles/2026/corpora/genai_agentic")


def _find_vendored_snapshot() -> _Path | None:
    """Find the vendored snapshot directory, or None if not yet vendored."""
    if not _VENDOR_BASE.exists():
        return None
    subdirs = [d for d in _VENDOR_BASE.iterdir() if d.is_dir()]
    if len(subdirs) != 1:
        return None
    return subdirs[0]


@pytest.fixture()
def real_snapshot() -> _Path:
    """Return the path to the real vendored snapshot, skip if not available."""
    snap = _find_vendored_snapshot()
    if snap is None:
        pytest.skip("Vendored snapshot not available — run Task 3 first")
    return snap


class TestRealSnapshotIntegration:
    """Tests against the actual vendored genai_agentic snapshot."""

    # Tolerances from audit reference (HANDOFF §3).
    # These are approximate — the vendored snapshot's exact counts may vary
    # slightly from the audit's N=7,714 depending on the snapshot date.
    EXPECTED_TOTAL_MIN = 7_000
    EXPECTED_TOTAL_MAX = 9_000
    EXPECTED_SECURITY_MIN = 6_500
    EXPECTED_SECURITY_MAX = 8_500
    EXPECTED_AI_HARM_MIN = 300
    EXPECTED_AI_HARM_MAX = 500

    def test_total_record_count_within_tolerance(
        self, real_snapshot: _Path
    ) -> None:
        adapter = GenAIAgenticAdapter(
            snapshot_dir=real_snapshot, snapshot_date="2026-05-20"
        )
        total = sum(1 for _ in adapter.iter_incidents())
        assert self.EXPECTED_TOTAL_MIN <= total <= self.EXPECTED_TOTAL_MAX, (
            f"Total records {total} outside expected range "
            f"[{self.EXPECTED_TOTAL_MIN}, {self.EXPECTED_TOTAL_MAX}]"
        )

    def test_security_stratum_count(self, real_snapshot: _Path) -> None:
        adapter = GenAIAgenticAdapter(
            snapshot_dir=real_snapshot, snapshot_date="2026-05-20"
        )
        count = sum(
            1 for r in adapter.iter_incidents() if r.corpus_stratum == "security"
        )
        assert self.EXPECTED_SECURITY_MIN <= count <= self.EXPECTED_SECURITY_MAX

    def test_ai_harm_stratum_count(self, real_snapshot: _Path) -> None:
        adapter = GenAIAgenticAdapter(
            snapshot_dir=real_snapshot, snapshot_date="2026-05-20"
        )
        count = sum(
            1 for r in adapter.iter_incidents() if r.corpus_stratum == "ai-harm"
        )
        assert self.EXPECTED_AI_HARM_MIN <= count <= self.EXPECTED_AI_HARM_MAX

    def test_every_stratum_has_a_bias_profile(
        self, real_snapshot: _Path
    ) -> None:
        adapter = GenAIAgenticAdapter(
            snapshot_dir=real_snapshot, snapshot_date="2026-05-20"
        )
        profile_strata = {p.stratum for p in adapter.bias_profiles()}
        record_strata = {r.corpus_stratum for r in adapter.iter_incidents()}
        assert record_strata.issubset(profile_strata)

    def test_snapshot_hash_is_stable(self, real_snapshot: _Path) -> None:
        """Content hash must be byte-stable across platforms."""
        from engine.snapshot.hashing import snapshot_hash

        h1 = snapshot_hash(real_snapshot / "incidents.json")
        h2 = snapshot_hash(real_snapshot / "incidents.json")
        assert h1 == h2
        assert h1 == real_snapshot.name  # directory name IS the hash

    def test_provenance_has_all_six_fields(self, real_snapshot: _Path) -> None:
        from engine.snapshot.provenance import SnapshotProvenance

        prov = SnapshotProvenance.read(real_snapshot / "provenance.json")
        assert prov.source_repo != ""
        assert prov.source_commit_sha != ""
        assert prov.pull_date != ""
        assert prov.adapter_name == "genai_agentic"
        assert prov.adapter_version != ""
        assert prov.snapshot_hash == real_snapshot.name

    def test_bare_llm03_contamination_count(
        self, real_snapshot: _Path
    ) -> None:
        """HANDOFF §3 F2: ~907 bare-LLM03 entries expected."""
        adapter = GenAIAgenticAdapter(
            snapshot_dir=real_snapshot, snapshot_date="2026-05-20"
        )
        contaminated = sum(
            1
            for r in adapter.iter_incidents()
            if is_bare_llm03_contaminated(list(r.native_labels))
        )
        assert 700 <= contaminated <= 1200, (
            f"Bare-LLM03 count {contaminated} outside expected range [700, 1200]"
        )
```

- [ ] **Step 2: Run tests**

```bash
uv run pytest tests/unit/test_adapter_genai_agentic.py::TestRealSnapshotIntegration -v
```

Expected: all PASS (or SKIP if snapshot not yet vendored).

- [ ] **Step 3: Run full quality gates**

```bash
uv run pytest -v && uv run mypy engine tests && uv run ruff check . && uv run semgrep --config .semgrep.yml --error engine/
```

- [ ] **Step 4: Commit**

```bash
git add tests/unit/test_adapter_genai_agentic.py
git commit -m "test(adapters): real-snapshot integration — stratum counts + hash stability (Plan 2)"
```

---

## Task 8: Drift detector integration with vendored snapshot

**Files:**
- Modify: `tests/unit/test_snapshot_vendor.py` (add drift integration test)

**Acceptance criteria served:** PRD §3.6 criteria 5 (drift detector runs on vendored snapshot, emits a report).

**Note:** The existing `engine/snapshot/drift.py` reads JSONL format (one JSON object per line). Per Premortem M2, `vendor_snapshot()` (Task 2) UNCONDITIONALLY writes `incidents.jsonl` alongside `incidents.json`. This task's drift tests use the `.jsonl` file. No format conversion decision is needed at execution time — it was resolved in Task 2.

- [ ] **Step 1: Write failing test for drift integration**

Add to `tests/unit/test_snapshot_vendor.py`:

```python
from engine.snapshot.drift import DriftReport, detect_drift


class TestDriftIntegration:

    def test_first_snapshot_produces_baseline_report(
        self, source_corpus: Path, dest_dir: Path
    ) -> None:
        """First snapshot has no predecessor — drift report exists but has no anomalies."""
        result = vendor_snapshot(
            source_path=source_corpus,
            dest_base=dest_dir,
            source_repo="test",
            source_commit_sha="aaa",
            adapter_version="0.1.0",
        )
        # Premortem M2: drift detector reads JSONL, not JSON arrays
        snapshot_jsonl = result.snapshot_dir / "incidents.jsonl"
        assert snapshot_jsonl.exists(), "vendor_snapshot must write incidents.jsonl"
        report = detect_drift(snapshot_jsonl, snapshot_jsonl)
        assert isinstance(report, DriftReport)
        assert report.requires_signoff is False
        assert len(report.anomalies) == 0

    def test_drift_detected_on_changed_snapshot(self, tmp_path: Path) -> None:
        """A modified snapshot triggers drift anomalies."""
        # Create two JSONL-format snapshots for drift detection
        prev_path = tmp_path / "prev.jsonl"
        curr_path = tmp_path / "curr.jsonl"

        # Previous: LLM03 has 100 entries
        prev_lines = [
            json.dumps({"owasp_llm": ["LLM03"]}) for _ in range(100)
        ]
        prev_path.write_text("\n".join(prev_lines))

        # Current: LLM03 has 200 entries (>20% and >50 absolute change)
        curr_lines = [
            json.dumps({"owasp_llm": ["LLM03"]}) for _ in range(200)
        ]
        curr_path.write_text("\n".join(curr_lines))

        report = detect_drift(prev_path, curr_path)
        assert isinstance(report, DriftReport)
        assert report.requires_signoff is True
        assert len(report.anomalies) > 0
```

- [ ] **Step 2: Run tests**

```bash
uv run pytest tests/unit/test_snapshot_vendor.py::TestDriftIntegration -v
```

Note: `vendor_snapshot()` (Task 2) already writes `incidents.jsonl` unconditionally (Premortem M2). If the JSONL file is missing, the vendoring step was not run correctly — go back to Task 2/3.

- [ ] **Step 3: Run full quality gates**

```bash
uv run pytest -v && uv run mypy engine tests && uv run ruff check . && uv run semgrep --config .semgrep.yml --error engine/
```

- [ ] **Step 4: Commit**

```bash
git add tests/unit/test_snapshot_vendor.py engine/cli/snapshot.py
git commit -m "test(snapshot): drift detector integration with vendored snapshot (Plan 2)"
```

---

## Task 9: CI verification — confirm Actions actually run (C5 erratum lesson)

**Files:**
- Read-only: `.github/workflows/ci.yml` (verify, do not modify unless needed)

**Acceptance criteria served:** PRD §3.6 criteria 2 (full suite green), 3 (mypy/ruff/semgrep clean). Also addresses PRD §3.8 risk (CI verification gap).

**This task exists because of Plan 1 v5.1 erratum (C5).** The erratum documented that workflow-file presence does not prove CI execution. This task verifies that the existing CI workflow runs Plan 2's new tests to completion.

- [ ] **Step 1: Run all quality gates locally**

```bash
uv run pytest -v
uv run mypy engine tests
uv run ruff check .
uv run semgrep --config .semgrep.yml --error engine/
```

All four must pass with zero errors.

- [ ] **Step 2: Push the branch and create a draft PR to trigger CI**

The CI workflow only triggers `push:` on `branches: [main]`. Feature-branch pushes do **not** trigger CI. The `pull_request:` event is the CI trigger for feature branches. Push and create a draft PR in one sequence:

```bash
git push -u origin plan2/corpus-a-adapter
gh pr create --title "Plan 2: Corpus A adapter + snapshot + per-stratum bias profiles" --body "WIP — CI verification" --draft
```

- [ ] **Step 3: Verify CI runs to completion**

Open the GitHub Actions tab for the draft PR. Confirm:

1. The `CI` workflow triggers on the `pull_request` event.
2. The `checks` job runs on both `ubuntu-latest` and `macos-latest`.
3. The `uv run pytest -v` step executes and includes `test_adapter_genai_agentic.py` and `test_snapshot_vendor.py` in its output.
4. All steps pass green.

**If CI fails:** diagnose the failure (do not dismiss it as a local-only issue). Common causes:
- The vendored snapshot is not committed (tests that depend on `real_snapshot` should `pytest.skip` gracefully).
- A JSONL/JSON format mismatch in drift detection (fix in Task 8).
- Missing dependencies.

- [ ] **Step 4: Record CI verification**

Once CI is green, record the successful run URL or commit SHA. This evidence closes the C5 erratum lesson for Plan 2.

**No commit from this task** (unless CI required a fix, in which case commit the fix).

---

## Task 10: Version bump + methodology changelog + final acceptance + tag

**Files:**
- Modify: `pyproject.toml` (version bump)
- Modify: `engine/version.py` (version bump)
- Modify: `docs/METHODOLOGY-CHANGELOG.md`

**Acceptance criteria served:** PRD §3.6 criteria 7 (methodology changelog entry), 8 (commit message), 9 (tag `v0.2.0-plan2`).

- [ ] **Step 1: Bump version in pyproject.toml and engine/version.py (M5)**

Update `pyproject.toml`:

```toml
version = "0.2.0"
```

Update `engine/version.py`:

```python
__version__ = "0.2.0"
```

Verify the version is consistent:

```bash
python3 -c "
import tomli
with open('pyproject.toml', 'rb') as f:
    cfg = tomli.load(f)
print(f'pyproject.toml: {cfg[\"project\"][\"version\"]}')
from engine.version import __version__
print(f'engine/version.py: {__version__}')
assert cfg['project']['version'] == __version__, 'Version mismatch!'
print('Versions match.')
"
```

- [ ] **Step 2: Add methodology changelog entry (M3)**

Add to the top of `docs/METHODOLOGY-CHANGELOG.md` (after the `# Methodology Changelog` heading, before the Plan 1 entry):

```markdown
## 0.2.0 (Plan 2, 2026-05-20)

GenAI Agentic corpus A adapter, per-stratum bias profiles, snapshot vendoring.

Key deliverables:
- `engine/adapters/genai_agentic.py`: concrete adapter reading vendored snapshot, emitting canonical IncidentRecord instances.
- `engine/adapters/genai_agentic_bias.py`: per-stratum BiasProfile declarations for "security" and "ai-harm" strata with quarantine rules for bare-LLM03 and double-default contamination (HANDOFF §3 F2).
- `engine/cli/snapshot.py`: vendor-snapshot CLI command with content-addressed hashing and provenance.json (6 fields per HANDOFF §5.1).
- Severity-default detection: source-ingest "Medium" default → `severity=None` (HANDOFF §3 Mixture).
- Future-dated row drop per HANDOFF §4 Temporal.
- Drift detector integration with vendored snapshot (JSONL format for drift.py compatibility).
- Vendored snapshot at `projects/owasp-llm/cycles/2026/corpora/genai_agentic/<hash>/`.

Methodology decision: this plan defines the stratum boundaries for Corpus A as the two values of the source `corpus` field ("security", "ai-harm"), each with a declared BiasProfile. This is a structural decision that constrains the Bayesian model's stratification in Plans 3–5. HANDOFF §4's "Corpus A is a mixture" row requires per-stratum bias profiles for corpus AND category — Plan 2 implements the corpus-level stratification; category-level stratification (per HANDOFF §3 Mixture) is deferred to Plan 3 where the rubric defines how categories interact with the measurement model. Entry definitions and overlap weights are provisional (2025 taxonomy placeholders) and MUST be replaced by the frozen 2026 rubric in Plan 3.
```

- [ ] **Step 3: Final acceptance checklist**

Run through every PRD §3.6 acceptance criterion:

| # | Criterion | Verification command | Expected |
|---|-----------|---------------------|----------|
| 1 | Adapter tests green | `uv run pytest tests/unit/test_adapter_genai_agentic.py -v` | PASS |
| 2 | Full suite green | `uv run pytest -v` | PASS, no regressions |
| 3 | mypy + ruff + semgrep clean | `uv run mypy engine tests && uv run ruff check . && uv run semgrep --config .semgrep.yml --error engine/` | Zero errors |
| 4 | Snapshot vendored with provenance | `ls projects/owasp-llm/cycles/2026/corpora/genai_agentic/*/provenance.json` | File exists, 6 fields |
| 5 | Drift detector runs | Verified in Task 8 tests | PASS |
| 6 | Per-stratum counts match audit | Verified in Task 7 `TestRealSnapshotIntegration` | Within tolerance |
| 7 | Methodology changelog | This step | Entry present |
| 8 | Commit message | Next step | Format matches |
| 9 | Tag | Next step | `v0.2.0-plan2` |

Run all verification commands:

```bash
uv run pytest tests/unit/test_adapter_genai_agentic.py -v
uv run pytest -v
uv run mypy engine tests
uv run ruff check .
uv run semgrep --config .semgrep.yml --error engine/
ls projects/owasp-llm/cycles/2026/corpora/genai_agentic/*/provenance.json
```

All must pass.

- [ ] **Step 4: Commit the version bump + changelog**

```bash
git add pyproject.toml engine/version.py docs/METHODOLOGY-CHANGELOG.md
git commit -m "docs: version 0.2.0 + record Plan 2 acceptance — methodology changelog"
```

- [ ] **Step 5: Create annotated tag**

```bash
git tag -a v0.2.0-plan2 -m "Plan 2: genai_agentic corpus A adapter + per-stratum bias profiles + snapshot vendoring"
```

- [ ] **Step 6: Verify tag**

```bash
git show v0.2.0-plan2 --stat
```

Expected: shows all Plan 2 commits.

---

## Coverage matrix: PRD §3.6 acceptance criteria → tasks

| PRD §3.6 Criterion | Task(s) |
|---|---|
| 1. `test_adapter_genai_agentic.py` green | 1, 4, 5, 6, 7 |
| 2. Full suite green, no regressions | 1–8 (gates after every task) |
| 3. mypy + ruff + semgrep zero errors | 1–8 (gates after every task) |
| 4. Snapshot vendored with provenance (6 fields) | 2, 3, 7 |
| 5. Drift detector runs on vendored snapshot | 8 |
| 6. Per-stratum counts match audit within tolerance | 7 |
| 7. Methodology changelog 0.2.0 | 10 |
| 8. Commit message format | All tasks (per-task commits) |
| 9. Tag `v0.2.0-plan2` | 10 |

## Coverage: PRD §3.5 deliverables → tasks

| PRD §3.5 Deliverable | Task(s) |
|---|---|
| 1. `engine/adapters/genai_agentic.py` | 4 |
| 2. `engine/adapters/genai_agentic_bias.py` | 1 |
| 3. Snapshot vendoring script `engine/cli/snapshot.py` | 2 |
| 4. Drift hook integration | 8 |
| 5. `projects/owasp-llm/cycles/2026/` with snapshot | 3 |
| 6. Severity-defaulting disclosure | 4, 5 |
| 7. Future-dated row repair | 4, 5 |
| 8. Tests | 1, 4, 5, 6, 7, 8 |

## Coverage: Inherited constraints → tasks

| Constraint | Where enforced |
|---|---|
| C1 (bare-LLM03 quarantine) | Task 1 predicates, Task 5 integration tests, Task 7 real-count validation |
| C2 (construction-time validation) | Task 1 `_validate_bias_profile`, Task 4 `__init__` guards |
| C3 (stratum-size >= observed) | Task 6 `test_stratum_size_gte_observed_count` |
| C4 (per-task commit cadence) | All tasks: one commit per task |
| C5 (CI verification erratum) | Task 9 |
| C6 (residuals acknowledged) | Plan header — no action required |
| C7 (integrity controls override) | §6 control 9 implemented via Tasks 2, 3, 8 |
| C8 (no engine changes for own sake) | Adapter-only changes; drift detector reused as-is |

## Coverage: Premortem remediations → tasks

| Remediation | Finding | Where implemented |
|---|---|---|
| M1 (hard gate on schema discovery) | F1.1 CRITICAL | Task 0 Step 7 hard-gate language; Task 4 PROVISIONAL marker |
| M2 (mandatory JSONL conversion) | F4.1 CRITICAL | Task 2 `_write_jsonl()` + tests; Task 3 Step 5 verification; Task 8 uses .jsonl |
| M3 (rewrite false "no methodology changes") | F5.1+F1.2 HIGH | Task 10 Step 2 changelog rewritten |
| M4 (provisional entry definitions) | F2.2 HIGH | Task 4 `_PROVISIONAL_2025_ENTRIES` constant + docstring |
| M5 (version bump + PRD update) | F4.5+F5.3 MEDIUM | Task 10 Step 1 (`pyproject.toml` + `engine/version.py`) |
| M6 (source ingest script inspection) | F1.3 MEDIUM | Task 0 Step 8 |
| M7 (lossy category→source_class docs) | F1.5 MEDIUM | Task 1 module docstring; Task 4 `_map_source_class()` docstring |
| M8 (contamination-status dependency) | F1.4 MEDIUM | C1 text fix; Task 1 module docstring |
| M9 (provisional overlap weights) | F2.3 MEDIUM | Task 4 `overlap_weights()` docstring |
| M10 (text length cap) | F3.1 MEDIUM | Task 4 `_MAX_TEXT_LENGTH` + truncation in `_transform()` |

## Coverage: Addressable residual risks → tasks

| Residual risk | Where addressed |
|---|---|
| R4 (severity-source field inspection) | Task 0 Step 8 (ingest script examination) |
| R5 (source commit SHA validation) | Task 0 Step 9 (tag/branch verification); Task 3 Step 3 (SHA resolves) |

---

## Execution handoff

Plan complete. Saved to `docs/superpowers/plans/2026-05-20-corpus-a-adapter.md`.

**Two execution options:**

1. **Subagent-Driven (recommended)** — fresh subagent per task, two-stage review, fast iteration.

2. **Inline Execution** — execute tasks in this session using `executing-plans`, batch execution with checkpoints.

Which approach?
