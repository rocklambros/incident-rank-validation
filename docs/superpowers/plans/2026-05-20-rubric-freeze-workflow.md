# Rubric Freeze Workflow — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the rubric drafting, adjudication, and freeze workflow for the 2026 LLM Top 10 cycle — producing the frozen, vote-blind, hash-locked classification rubric with Rock's adjudication log and independent reviewer signoff scaffolding.

**Architecture:** Three-phase approach: (A) workflow scaffolding (rubric data model, serialization, freeze CLI, gate logic, tests), (B) rubric drafting from entry definitions with taxonomy vendoring, (C) freeze with external reviewer signoff. Phases A and B are runnable now; Phase C is BLOCKED on external rubric reviewer identification per `docs/REVIEWERS.md` INTERIM state.

**Tech Stack:** Python 3.12, dataclasses, JSON (canonical serialization), SHA-256 hashing, Click CLI, pytest

---

## Inherited Constraints from Phases 1-2

These carry-forwards are concrete technical and procedural constraints. Violating any is a plan bug.

### From Plan 1 (v0.1.0-plan1)

1. **Pre-registration schema stubs — promote, do not replace:**
   - `engine/prereg/rubric_attestation.py`: `RubricDraftingAttestation(viewed_corpus_before_drafting: bool, viewed_corpus_details: str)` — Plan 1 stub dataclass. Plan 3 adds `viewed_vote_data_before_drafting: bool` and `viewed_vote_data_details: str` fields (Premortem3 R3) for vote-blindness attestation, adds JSON serialization, and populates it as a committed artifact.
   - `engine/prereg/signoff.py`: `ReviewerSignoff` with `verify()` (hash + git-timestamp + content checks). Plan 3 consumes this for external reviewer workflow.
   - `engine/prereg/manifest.py`: `PreregManifest` with `non_publishable` property. Plan 3 adds `rubric_hash` field.
   - `engine/prereg/lock.py`: `compute_lock_hash`, `write_lock`, `verify_lock` — operational. Plan 3 uses these.
   - `engine/prereg/attestation.py`: `verify_committed` — git-tracked file check. Plan 3 uses this to verify rubric files are committed before freeze.
   - `engine/prereg/git_timestamp.py`: `attestation_signed_at` — M8 mechanism for reviewer signoff timing.

2. **Procedural vote-blindness (HANDOFF §6 control 2):** The engine prevents vote data from entering classify/calibrate/infer. The rubric *drafter* must also not read the vote. This is procedural: the drafting session must not open the `2026/polling/` directory in the source repo (`https://github.com/GenAI-Security-Project/GenAI-LLM-Top10`) or any vote-result file. The `RubricDraftingAttestation` records this.

3. **CI verification lesson (Plan 1 v5.1 erratum, C5):** Any CI modification must verify CI actually runs new logic to completion in a green run.

4. **Test baseline:** 422 tests passing (370 Plan 1 + 52 Plan 2), 10 xfailed Stage-2 security fixtures.

### From Plan 2 (v0.2.0-plan2)

5. **Snapshot-hash binding:** Vendored snapshot at `projects/owasp-llm/cycles/2026/corpora/genai_agentic/<hash>/`. The prereg manifest already carries `snapshot_hash`. Plan 3's rubric does not directly bind to the snapshot, but the manifest binds both.

6. **`_PROVISIONAL_2025_ENTRIES` must eventually be replaced:** `engine/adapters/genai_agentic.py:38-55` defines 10 provisional LLM entries. The frozen rubric covers 20 entries. Plan 3 does NOT modify the adapter (Plan 5 integrates the frozen rubric), but the rubric must define all 20 entries.

7. **Corpus messiness informs rubric boundary rules:**
   - Bare-LLM03 contamination (~907 entries): rubric LLM03 entry must exclude default-seeded records.
   - Severity-default artifact: `"Medium"` is unreliable — rubric indicators should not use severity.
   - Stratum heterogeneity: "security" (~7,350) vs "ai-harm" (~364).
   - Lossy category mapping: 8 source categories → 3 engine source_classes.

8. **project.toml references taxonomy.json:** Line 18: `source = "cycles/2026/taxonomy/taxonomy.json"`. This file does not exist yet — Plan 3 creates it.

9. **Entry definition paths:** Source repo: `https://github.com/GenAI-Security-Project/GenAI-LLM-Top10`
   - Incumbents (10): `2026/LLM01_PromptInjection.md` through `2026/LLM10_UnboundedConsumption.md`
   - Candidates (10): `2026/new_entry_candidates/*.md`
   - NOTE: The `www-project-top-10-for-large-language-model-applications` repo contains the ASI agentic incidents corpus (Plan 2 Corpus A), NOT the Top 10 candidate entries.

---

## The 20 Rubric Entries (from HANDOFF §2)

For reference during execution. All 20 entries need a rubric entry.

**10 incumbents (from `GenAI-LLM-Top10/2026/`):**
- LLM01 Prompt Injection
- LLM02 Sensitive Information Disclosure
- LLM03 Supply Chain Vulnerabilities
- LLM04 Data and Model Poisoning
- LLM05 Improper Output Handling
- LLM06 Excessive Agency
- LLM07 System Prompt Leakage (CASE 2 name: "Hidden Context Exposure")
- LLM08 Vector and Embedding Weaknesses
- LLM09 Misinformation
- LLM10 Unbounded Consumption

**6 standalone new candidates (from `GenAI-LLM-Top10/2026/new_entry_candidates/`):**
- Persistent Memory Poisoning (new candidate)
- MCP Tool Interface Exploitation (new candidate)
- Model Misalignment (new candidate)
- Inference-Time Side-Channel Disclosure (new candidate)
- Weaponized LLM Abuse (new candidate)
- Model Scheming and Deceptive Alignment (new candidate)

> **VOTE-BLINDNESS NOTE (Premortem R1):** CASE 2 rank ordinals are deliberately omitted from this plan. Subagents executing Phase B MUST NOT consult HANDOFF §2 for rank order. Entry names are needed for identification; rank positions are vote-derived data that would compromise rubric-drafting blindness per HANDOFF §6 control 2.

**4 rolled-up candidates (own rubric entry for rollup sub-test per §5.2):**
- Cross-Modal Safety Bypass (rolled into LLM01)
- LLM Artifact Promotion Trust Failure (rolled into LLM03)
- Systemic Insecure Code Generation (rolled into LLM05)
- Compositional Fine-tuning Alignment Subversion (rolled into LLM04)

---

## File Structure

### New files

| File | Responsibility |
|---|---|
| `engine/prereg/rubric.py` | `RubricEntry`, `BoundaryRule`, `Rubric` dataclasses with hash + validation |
| `engine/prereg/adjudication.py` | `AdjudicationEntry`, `AdjudicationLog` dataclasses |
| `engine/prereg/rubric_io.py` | JSON read/write for rubric, attestation, adjudication log |
| `engine/prereg/gates.py` | Pre-classify gate checks + publishability check + rubric-hash-match verification |
| `engine/cli/rubric.py` | Click commands: `freeze-rubric`, `validate-rubric` |
| `tests/unit/test_rubric.py` | Rubric schema, hash stability, gates, freeze tests |
| `projects/owasp-llm/cycles/2026/taxonomy/taxonomy.json` | Machine-readable 20-entry taxonomy |
| `projects/owasp-llm/cycles/2026/taxonomy/taxonomy_provenance.json` | Source repo commit hash + vendoring metadata |
| `projects/owasp-llm/cycles/2026/taxonomy/*.md` | Vendored entry definition source files |
| `projects/owasp-llm/cycles/2026/prereg/rubric.json` | Rubric content (Phase B) |
| `projects/owasp-llm/cycles/2026/prereg/adjudication_log.json` | Adjudication decisions (Phase B) |
| `projects/owasp-llm/cycles/2026/prereg/rubric_attestation.json` | Drafting attestation (Phase B) |
| `docs/RUBRIC-WORKFLOW.md` | Procedural documentation |

### Modified files

| File | Change |
|---|---|
| `engine/prereg/manifest.py` | Add `rubric_hash: str \| None` field |
| `engine/prereg/__init__.py` | Re-export new types |
| `engine/cli/main.py` | Wire rubric CLI commands |
| `tests/unit/test_prereg.py` | Add `rubric_hash` to lock mutation table |
| `engine/version.py` | Bump to `"0.3.0"` |
| `pyproject.toml` | Bump to `"0.3.0"` |
| `uv.lock` | Version field updated |
| `docs/METHODOLOGY-CHANGELOG.md` | Add 0.3.0 entry |
| `tests/test_bootstrap.py` | Version assertion `"0.3.0"` |

---

## Phase A: Workflow Scaffolding (runnable now)

Phase A builds all the infrastructure. It merges as `v0.3.0-rc1` without any rubric content.

---

### Task 0: Environment Setup

**Files:**
- None created or modified

- [ ] **Step 1: Create worktree**

Use the `superpowers:using-git-worktrees` skill to create an isolated workspace on branch `plan3/rubric-freeze-workflow`.

- [ ] **Step 2: Verify baseline**

Run: `uv run pytest -x -q 2>&1 | tail -5`
Expected: `422 passed, 10 xfailed`

Run: `uv run mypy engine tests --strict 2>&1 | tail -3`
Expected: `Success`

---

### Task 1: Rubric Data Model

**Files:**
- Create: `engine/prereg/rubric.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_rubric.py` with an import test:

```python
"""Tests for the rubric data model, serialization, gates, and freeze workflow."""
from __future__ import annotations

import json

import pytest

from engine.prereg.rubric import BoundaryRule, Rubric, RubricEntry


class TestRubricDataModel:
    """Tests for RubricEntry, BoundaryRule, and Rubric dataclasses."""

    def test_rubric_entry_all_fields_present(self) -> None:
        entry = RubricEntry(
            entry_id="LLM01",
            canonical_name="Prompt Injection",
            in_scope="Attacks that manipulate LLM behavior via crafted input.",
            exclusions=("Jailbreaking via social engineering",),
            boundary_rules=(
                BoundaryRule(
                    adjacent_entry_id="LLM02",
                    rule="If the attack extracts data, classify as LLM02 not LLM01.",
                    is_ambiguous=False,
                ),
            ),
            positive_indicators=("prompt injection", "instruction override"),
            negative_indicators=("data exfiltration only",),
            co_occurrence_pairs=(("LLM01", "LLM06"),),
            is_rollup_candidate=False,
            rolled_into=None,
        )
        assert entry.entry_id == "LLM01"
        assert entry.canonical_name == "Prompt Injection"
        assert len(entry.exclusions) == 1
        assert len(entry.boundary_rules) == 1
        assert entry.boundary_rules[0].adjacent_entry_id == "LLM02"

    def test_rubric_entry_rollup_candidate(self) -> None:
        entry = RubricEntry(
            entry_id="ROLL-CMSB",
            canonical_name="Cross-Modal Safety Bypass",
            in_scope="Attacks exploiting cross-modal input processing.",
            exclusions=(),
            boundary_rules=(
                BoundaryRule(
                    adjacent_entry_id="LLM01",
                    rule="If attack uses only text prompts, classify as LLM01.",
                    is_ambiguous=False,
                ),
            ),
            positive_indicators=("multi-modal", "image injection"),
            negative_indicators=("text-only prompt",),
            co_occurrence_pairs=(("ROLL-CMSB", "LLM01"),),
            is_rollup_candidate=True,
            rolled_into="LLM01",
        )
        assert entry.is_rollup_candidate is True
        assert entry.rolled_into == "LLM01"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_rubric.py::TestRubricDataModel -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'engine.prereg.rubric'`

- [ ] **Step 3: Write the rubric data model**

Create `engine/prereg/rubric.py`:

```python
"""Frozen classification rubric per HANDOFF §5.2 Artifact 1.

The rubric is the primary pre-registration artifact: per-entry classification
rules frozen, hash-locked, and independently reviewed before any concordance
number exists.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class BoundaryRule:
    """Pairwise boundary rule between two taxonomy entries."""

    adjacent_entry_id: str
    rule: str
    is_ambiguous: bool


def _to_serializable(obj: object) -> object:
    """Recursively convert frozen dataclasses and tuples to JSON-safe types."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {
            f.name: _to_serializable(getattr(obj, f.name))
            for f in dataclasses.fields(obj)
        }
    if isinstance(obj, tuple):
        return [_to_serializable(item) for item in obj]
    return obj


@dataclass(frozen=True, slots=True)
class RubricEntry:
    """Per-entry classification rubric per HANDOFF §5.2 Artifact 1."""

    entry_id: str
    canonical_name: str
    in_scope: str
    exclusions: tuple[str, ...]
    boundary_rules: tuple[BoundaryRule, ...]
    positive_indicators: tuple[str, ...]
    negative_indicators: tuple[str, ...]
    co_occurrence_pairs: tuple[tuple[str, str], ...]
    is_rollup_candidate: bool
    rolled_into: str | None


@dataclass(frozen=True, slots=True)
class Rubric:
    """Complete frozen classification rubric."""

    cycle_id: str
    version: str
    entries: tuple[RubricEntry, ...]

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-safe dict for serialization and hashing."""
        return _to_serializable(self)  # type: ignore[return-value]

    def compute_hash(self) -> str:
        """SHA-256 of canonical JSON representation."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def validate_completeness(
        self,
        expected_entry_ids: set[str],
        *,
        no_adjacency_attested: set[str] | None = None,
    ) -> None:
        """Verify all expected entries are present and required fields non-empty.

        Parameters
        ----------
        no_adjacency_attested:
            Entry IDs for which the drafter attests no genuine adjacency exists.
            These are allowed to have empty boundary_rules without raising.
            A warning is logged instead.
        """
        attested = no_adjacency_attested or set()
        actual = {e.entry_id for e in self.entries}
        missing = expected_entry_ids - actual
        extra = actual - expected_entry_ids
        if missing:
            raise ValueError(f"rubric missing entries: {sorted(missing)}")
        if extra:
            raise ValueError(f"rubric has unexpected entries: {sorted(extra)}")
        for entry in self.entries:
            if not entry.in_scope.strip():
                raise ValueError(f"{entry.entry_id}: in_scope is empty")
            if not entry.positive_indicators:
                raise ValueError(f"{entry.entry_id}: positive_indicators is empty")
            if not entry.negative_indicators:
                raise ValueError(f"{entry.entry_id}: negative_indicators is empty")
            if not entry.is_rollup_candidate and not entry.boundary_rules:
                if entry.entry_id in attested:
                    import logging
                    logging.getLogger(__name__).warning(
                        "%s: no boundary_rules (no-adjacency attested by drafter)",
                        entry.entry_id,
                    )
                else:
                    raise ValueError(
                        f"{entry.entry_id}: boundary_rules is empty for non-rollup entry "
                        f"(pass entry_id in no_adjacency_attested if no genuine adjacency exists)"
                    )

    def validate_boundary_rules(self) -> None:
        """Verify boundary rules are paired: if A->B exists, B->A must exist."""
        pairs: set[tuple[str, str]] = set()
        entry_ids = {e.entry_id for e in self.entries}
        for entry in self.entries:
            for br in entry.boundary_rules:
                if br.adjacent_entry_id not in entry_ids:
                    raise ValueError(
                        f"{entry.entry_id}: boundary rule references unknown "
                        f"entry {br.adjacent_entry_id}"
                    )
                pairs.add((entry.entry_id, br.adjacent_entry_id))
        for a, b in pairs:
            if (b, a) not in pairs:
                raise ValueError(
                    f"boundary rule {a}->{b} exists but {b}->{a} is missing"
                )

    def validate_co_occurrences(self) -> None:
        """Verify all co_occurrence_pairs reference valid entry IDs."""
        entry_ids = {e.entry_id for e in self.entries}
        for entry in self.entries:
            for pair in entry.co_occurrence_pairs:
                for eid in pair:
                    if eid not in entry_ids:
                        raise ValueError(
                            f"{entry.entry_id}: co_occurrence_pair references "
                            f"unknown entry {eid}"
                        )

    def rollup_candidates(self) -> tuple[RubricEntry, ...]:
        """Return only the rolled-up candidate entries."""
        return tuple(e for e in self.entries if e.is_rollup_candidate)

    def standalone_entries(self) -> tuple[RubricEntry, ...]:
        """Return entries that are NOT rollup candidates."""
        return tuple(e for e in self.entries if not e.is_rollup_candidate)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_rubric.py::TestRubricDataModel -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add engine/prereg/rubric.py tests/unit/test_rubric.py
git commit -m "feat(prereg): rubric data model — RubricEntry, BoundaryRule, Rubric (Plan 3)"
```

---

### Task 2: Rubric Hash Stability and Validation Tests

**Files:**
- Modify: `tests/unit/test_rubric.py`

- [ ] **Step 1: Write hash stability and validation tests**

Append to `tests/unit/test_rubric.py`:

```python
def _make_entry(
    entry_id: str = "LLM01",
    *,
    canonical_name: str = "Prompt Injection",
    is_rollup: bool = False,
    rolled_into: str | None = None,
    boundary_adjacent: str = "LLM02",
) -> RubricEntry:
    return RubricEntry(
        entry_id=entry_id,
        canonical_name=canonical_name,
        in_scope=f"In-scope statement for {entry_id}.",
        exclusions=(f"Exclusion for {entry_id}",),
        boundary_rules=(
            BoundaryRule(
                adjacent_entry_id=boundary_adjacent,
                rule=f"Boundary rule between {entry_id} and {boundary_adjacent}.",
                is_ambiguous=False,
            ),
        ),
        positive_indicators=(f"positive-{entry_id}",),
        negative_indicators=(f"negative-{entry_id}",),
        co_occurrence_pairs=(),
        is_rollup_candidate=is_rollup,
        rolled_into=rolled_into,
    )


def _make_rubric(entries: tuple[RubricEntry, ...] | None = None) -> Rubric:
    if entries is None:
        entries = (
            _make_entry("LLM01", boundary_adjacent="LLM02"),
            _make_entry("LLM02", boundary_adjacent="LLM01"),
        )
    return Rubric(cycle_id="2026", version="1.0.0", entries=entries)


class TestRubricHash:
    """Hash stability tests."""

    def test_hash_deterministic(self) -> None:
        r = _make_rubric()
        assert r.compute_hash() == r.compute_hash()

    def test_hash_changes_on_mutation(self) -> None:
        r1 = _make_rubric()
        r2 = Rubric(
            cycle_id="2026",
            version="1.0.1",
            entries=r1.entries,
        )
        assert r1.compute_hash() != r2.compute_hash()

    def test_hash_changes_on_entry_mutation(self) -> None:
        r1 = _make_rubric()
        mutated_entry = RubricEntry(
            entry_id="LLM01",
            canonical_name="MUTATED NAME",
            in_scope=r1.entries[0].in_scope,
            exclusions=r1.entries[0].exclusions,
            boundary_rules=r1.entries[0].boundary_rules,
            positive_indicators=r1.entries[0].positive_indicators,
            negative_indicators=r1.entries[0].negative_indicators,
            co_occurrence_pairs=r1.entries[0].co_occurrence_pairs,
            is_rollup_candidate=False,
            rolled_into=None,
        )
        r2 = Rubric(
            cycle_id="2026",
            version="1.0.0",
            entries=(mutated_entry, r1.entries[1]),
        )
        assert r1.compute_hash() != r2.compute_hash()

    def test_to_dict_roundtrips_through_json(self) -> None:
        r = _make_rubric()
        d = r.to_dict()
        serialized = json.dumps(d, sort_keys=True, separators=(",", ":"))
        roundtripped = json.loads(serialized)
        assert roundtripped == d


class TestRubricValidation:
    """Completeness and boundary-rule validation tests."""

    def test_validate_completeness_passes(self) -> None:
        r = _make_rubric()
        r.validate_completeness({"LLM01", "LLM02"})

    def test_validate_completeness_missing_entry(self) -> None:
        r = _make_rubric()
        with pytest.raises(ValueError, match="rubric missing entries"):
            r.validate_completeness({"LLM01", "LLM02", "LLM03"})

    def test_validate_completeness_extra_entry(self) -> None:
        r = _make_rubric()
        with pytest.raises(ValueError, match="rubric has unexpected entries"):
            r.validate_completeness({"LLM01"})

    def test_validate_completeness_empty_in_scope(self) -> None:
        bad_entry = RubricEntry(
            entry_id="LLM01",
            canonical_name="Prompt Injection",
            in_scope="",
            exclusions=(),
            boundary_rules=(),
            positive_indicators=("x",),
            negative_indicators=("y",),
            co_occurrence_pairs=(),
            is_rollup_candidate=False,
            rolled_into=None,
        )
        r = Rubric(cycle_id="2026", version="1.0.0", entries=(bad_entry,))
        with pytest.raises(ValueError, match="in_scope is empty"):
            r.validate_completeness({"LLM01"})

    def test_validate_completeness_empty_positive_indicators(self) -> None:
        bad_entry = RubricEntry(
            entry_id="LLM01",
            canonical_name="Prompt Injection",
            in_scope="In scope.",
            exclusions=(),
            boundary_rules=(),
            positive_indicators=(),
            negative_indicators=("y",),
            co_occurrence_pairs=(),
            is_rollup_candidate=False,
            rolled_into=None,
        )
        r = Rubric(cycle_id="2026", version="1.0.0", entries=(bad_entry,))
        with pytest.raises(ValueError, match="positive_indicators is empty"):
            r.validate_completeness({"LLM01"})

    def test_validate_completeness_empty_boundary_rules_non_rollup(self) -> None:
        bad_entry = RubricEntry(
            entry_id="LLM01",
            canonical_name="Prompt Injection",
            in_scope="In scope.",
            exclusions=(),
            boundary_rules=(),
            positive_indicators=("x",),
            negative_indicators=("y",),
            co_occurrence_pairs=(),
            is_rollup_candidate=False,
            rolled_into=None,
        )
        r = Rubric(cycle_id="2026", version="1.0.0", entries=(bad_entry,))
        with pytest.raises(ValueError, match="boundary_rules is empty for non-rollup entry"):
            r.validate_completeness({"LLM01"})

    def test_validate_completeness_empty_boundary_rules_no_adjacency_attested(self) -> None:
        entry = RubricEntry(
            entry_id="NEW-WLA",
            canonical_name="Weaponized LLM Abuse",
            in_scope="In scope.",
            exclusions=(),
            boundary_rules=(),
            positive_indicators=("x",),
            negative_indicators=("y",),
            co_occurrence_pairs=(),
            is_rollup_candidate=False,
            rolled_into=None,
        )
        r = Rubric(cycle_id="2026", version="1.0.0", entries=(entry,))
        r.validate_completeness(
            {"NEW-WLA"}, no_adjacency_attested={"NEW-WLA"}
        )  # should not raise — logs warning instead

    def test_validate_completeness_empty_boundary_rules_rollup_ok(self) -> None:
        rollup = RubricEntry(
            entry_id="ROLL-X",
            canonical_name="Rollup",
            in_scope="In scope.",
            exclusions=(),
            boundary_rules=(),
            positive_indicators=("x",),
            negative_indicators=("y",),
            co_occurrence_pairs=(),
            is_rollup_candidate=True,
            rolled_into="LLM01",
        )
        r = Rubric(cycle_id="2026", version="1.0.0", entries=(rollup,))
        r.validate_completeness({"ROLL-X"})  # should not raise

    def test_validate_boundary_rules_paired(self) -> None:
        r = _make_rubric()
        r.validate_boundary_rules()  # should not raise

    def test_validate_boundary_rules_unpaired(self) -> None:
        entry_a = _make_entry("LLM01", boundary_adjacent="LLM02")
        entry_b = RubricEntry(
            entry_id="LLM02",
            canonical_name="Sensitive Info",
            in_scope="In scope.",
            exclusions=(),
            boundary_rules=(),  # missing reciprocal rule
            positive_indicators=("x",),
            negative_indicators=("y",),
            co_occurrence_pairs=(),
            is_rollup_candidate=False,
            rolled_into=None,
        )
        r = Rubric(cycle_id="2026", version="1.0.0", entries=(entry_a, entry_b))
        with pytest.raises(ValueError, match="boundary rule LLM01->LLM02.*LLM02->LLM01 is missing"):
            r.validate_boundary_rules()

    def test_validate_boundary_rules_unknown_entry(self) -> None:
        entry = _make_entry("LLM01", boundary_adjacent="NONEXISTENT")
        r = Rubric(cycle_id="2026", version="1.0.0", entries=(entry,))
        with pytest.raises(ValueError, match="references unknown entry"):
            r.validate_boundary_rules()

    def test_validate_co_occurrences_valid(self) -> None:
        r = _make_rubric()
        r.validate_co_occurrences()  # should not raise (empty co_occurrence_pairs)

    def test_validate_co_occurrences_invalid_entry_id(self) -> None:
        bad_entry = RubricEntry(
            entry_id="LLM01",
            canonical_name="Prompt Injection",
            in_scope="In scope.",
            exclusions=(),
            boundary_rules=(
                BoundaryRule(
                    adjacent_entry_id="LLM02",
                    rule="Test rule.",
                    is_ambiguous=False,
                ),
            ),
            positive_indicators=("x",),
            negative_indicators=("y",),
            co_occurrence_pairs=(("LLM01", "NONEXISTENT"),),
            is_rollup_candidate=False,
            rolled_into=None,
        )
        entry_b = _make_entry("LLM02", boundary_adjacent="LLM01")
        r = Rubric(cycle_id="2026", version="1.0.0", entries=(bad_entry, entry_b))
        with pytest.raises(ValueError, match="co_occurrence_pair references unknown entry NONEXISTENT"):
            r.validate_co_occurrences()

    def test_rollup_candidates_filter(self) -> None:
        regular = _make_entry("LLM01", boundary_adjacent="ROLL-X")
        rollup = _make_entry(
            "ROLL-X",
            canonical_name="Rollup Entry",
            is_rollup=True,
            rolled_into="LLM01",
            boundary_adjacent="LLM01",
        )
        r = Rubric(cycle_id="2026", version="1.0.0", entries=(regular, rollup))
        assert len(r.rollup_candidates()) == 1
        assert r.rollup_candidates()[0].entry_id == "ROLL-X"
        assert len(r.standalone_entries()) == 1
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/unit/test_rubric.py -v`
Expected: All pass

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_rubric.py
git commit -m "test(prereg): rubric hash stability + completeness + boundary validation (Plan 3)"
```

---

### Task 3: Add `rubric_hash` to PreregManifest

**Files:**
- Modify: `engine/prereg/manifest.py`
- Modify: `tests/unit/test_prereg.py`

**NOTE (Premortem F2.1):** Adding `rubric_hash` changes the manifest's canonical JSON output (a new key appears in `to_dict()` even when `None`). This is a manifest schema break — any lock file computed against the old schema will fail `verify_lock`. No committed lock files exist (Plans 1-2 use transient `tmp_path` locks in tests), so this is safe. The version bump to 0.3.0 in Task 10 signals the schema change.

- [ ] **Step 1: Add the field to PreregManifest**

In `engine/prereg/manifest.py`, add `rubric_hash: str | None` after `classifier_rule_hash`:

```python
    classifier_rule_hash: str | None  # hash of Stage-1 classifier rules
    rubric_hash: str | None  # hash of frozen rubric (Plan 3)
    post_hoc_register_path: str | None  # path to Merkle-chained register
```

- [ ] **Step 2: Update `_make_manifest` fixture in test_prereg.py**

In `tests/unit/test_prereg.py`, add `"rubric_hash": None,` to the defaults dict in `_make_manifest()`:

```python
        "classifier_rule_hash": None,
        "rubric_hash": None,
        "post_hoc_register_path": None,
```

- [ ] **Step 3: Add `rubric_hash` to the lock mutation table**

In `tests/unit/test_prereg.py` class `TestLock.test_verify_lock_raises_on_mutation`, add to the `mutations` dict:

```python
            "rubric_hash": "mutated_rubric_hash",
```

And update the `_make_manifest` call in that test to include `rubric_hash="original_rubric_hash"`.

- [ ] **Step 4: Run all tests**

Run: `uv run pytest tests/unit/test_prereg.py -v`
Expected: All pass

Run: `uv run mypy engine/prereg/manifest.py --strict`
Expected: Success

- [ ] **Step 5: Add vote-data exposure fields to RubricDraftingAttestation (Premortem3 R3)**

HANDOFF §6 control 2 requires vote-blindness during rubric drafting. The existing `RubricDraftingAttestation` records corpus exposure but NOT vote-data exposure — these are orthogonal blinding controls. Add fields so vote-blindness violations are recordable and auditable.

In `engine/prereg/rubric_attestation.py`, add two fields after `viewed_corpus_details`:

```python
@dataclass(frozen=True, slots=True)
class RubricDraftingAttestation:
    """Attestation about corpus and vote-data exposure during rubric drafting."""

    viewed_corpus_before_drafting: bool
    viewed_corpus_details: str  # which samples, if any — empty string if none
    viewed_vote_data_before_drafting: bool  # HANDOFF §6 control 2
    viewed_vote_data_details: str  # which vote data, if any — empty string if none
    no_adjacency_attested_entry_ids: tuple[str, ...] = ()  # Premortem4 R3: audit trail
```

The `no_adjacency_attested_entry_ids` field records which entries were attested as having no genuine adjacency during rubric validation (entries that pass `validate_completeness()` with empty `boundary_rules` via the `--no-adjacency-attested` CLI flag). This persists the methodological decision in the hash-locked attestation artifact rather than leaving it as an ephemeral CLI parameter.

- [ ] **Step 6: Run tests to verify no regressions**

Run: `uv run pytest tests/unit/test_prereg.py -v`

Any existing test that constructs `RubricDraftingAttestation` with only 2 positional args will now fail. Fix by adding the two new fields (both `False` and `""` for tests that don't care about vote-data exposure). Scan with:

```bash
grep -rn "RubricDraftingAttestation(" tests/ engine/
```

And update each construction site to include the new fields.

- [ ] **Step 7: Commit**

```bash
git add engine/prereg/rubric_attestation.py tests/unit/test_prereg.py
git commit -m "feat(prereg): add rubric_hash to manifest + vote-data attestation fields (Plan 3)"
```

---

### Task 4: Adjudication Log Schema

**Files:**
- Create: `engine/prereg/adjudication.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_rubric.py`:

```python
from engine.prereg.adjudication import AdjudicationEntry, AdjudicationLog


class TestAdjudicationLog:
    """Tests for the adjudication log schema."""

    def test_adjudication_entry_fields(self) -> None:
        ae = AdjudicationEntry(
            entry_id_a="LLM01",
            entry_id_b="LLM02",
            decision="resolved:LLM01",
            rationale="The attack manipulates behavior, not extracts data.",
            adjudicator="Rock Lambros",
            date="2026-05-20",
        )
        assert ae.entry_id_a == "LLM01"
        assert ae.decision == "resolved:LLM01"

    def test_adjudication_entry_ambiguous(self) -> None:
        ae = AdjudicationEntry(
            entry_id_a="LLM03",
            entry_id_b="LLM04",
            decision="ambiguous-both-labels",
            rationale="Supply chain poisoning overlaps with data poisoning.",
            adjudicator="Rock Lambros",
            date="2026-05-20",
        )
        assert ae.decision == "ambiguous-both-labels"

    def test_adjudication_entry_invalid_decision(self) -> None:
        with pytest.raises(ValueError, match="invalid decision format"):
            AdjudicationEntry(
                entry_id_a="LLM01",
                entry_id_b="LLM02",
                decision="maybe:LLM01",
                rationale="Bad format.",
                adjudicator="Rock Lambros",
                date="2026-05-20",
            )

    def test_adjudication_entry_resolved_not_in_pair(self) -> None:
        with pytest.raises(ValueError, match="resolved entry.*not in adjudicated pair"):
            AdjudicationEntry(
                entry_id_a="LLM01",
                entry_id_b="LLM02",
                decision="resolved:LLM99",
                rationale="Wrong entry ID.",
                adjudicator="Rock Lambros",
                date="2026-05-20",
            )

    def test_adjudication_log_validate_coverage(self) -> None:
        rubric = _make_rubric()
        log = AdjudicationLog(
            rubric_hash=rubric.compute_hash(),
            entries=(
                AdjudicationEntry(
                    entry_id_a="LLM01",
                    entry_id_b="LLM02",
                    decision="resolved:LLM01",
                    rationale="Boundary adjudicated.",
                    adjudicator="Rock Lambros",
                    date="2026-05-20",
                ),
            ),
        )
        log.validate_coverage(rubric)  # should not raise

    def test_adjudication_log_missing_boundary(self) -> None:
        rubric = _make_rubric()
        log = AdjudicationLog(rubric_hash=rubric.compute_hash(), entries=())
        with pytest.raises(ValueError, match="boundary pairs without adjudication"):
            log.validate_coverage(rubric)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_rubric.py::TestAdjudicationLog -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the adjudication log module**

Create `engine/prereg/adjudication.py`:

```python
"""Adjudication log for rubric boundary-cell decisions.

Records Rock's per-cell adjudications per HANDOFF §5.2. Boundary cells
that are genuine 50/50 calls carry ``decision="ambiguous-both-labels"``
and propagate as label uncertainty into the measurement model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from engine.prereg.rubric import Rubric


_VALID_DECISIONS = frozenset({"ambiguous-both-labels"})
_RESOLVED_PREFIX = "resolved:"


@dataclass(frozen=True, slots=True)
class AdjudicationEntry:
    """A single boundary-cell adjudication decision."""

    entry_id_a: str
    entry_id_b: str
    decision: str  # "resolved:<winner_entry_id>" or "ambiguous-both-labels"
    rationale: str
    adjudicator: str
    date: str  # ISO 8601

    def __post_init__(self) -> None:
        if (
            self.decision not in _VALID_DECISIONS
            and not self.decision.startswith(_RESOLVED_PREFIX)
        ):
            raise ValueError(
                f"invalid decision format: {self.decision!r}. "
                f"Must be 'resolved:<entry_id>' or 'ambiguous-both-labels'."
            )
        if self.decision.startswith(_RESOLVED_PREFIX):
            winner = self.decision[len(_RESOLVED_PREFIX):]
            if winner not in {self.entry_id_a, self.entry_id_b}:
                raise ValueError(
                    f"resolved entry {winner!r} not in adjudicated pair "
                    f"({self.entry_id_a}, {self.entry_id_b})"
                )


@dataclass(frozen=True, slots=True)
class AdjudicationLog:
    """Complete adjudication log bound to a rubric hash."""

    rubric_hash: str
    entries: tuple[AdjudicationEntry, ...]

    def validate_coverage(self, rubric: Rubric) -> None:
        """Verify every boundary rule pair has an adjudication entry."""
        boundary_pairs: set[tuple[str, str]] = set()
        for entry in rubric.entries:
            for br in entry.boundary_rules:
                a, b = sorted([entry.entry_id, br.adjacent_entry_id])
                boundary_pairs.add((a, b))
        adjudicated: set[tuple[str, str]] = set()
        for ae in self.entries:
            a, b = sorted([ae.entry_id_a, ae.entry_id_b])
            if (a, b) in adjudicated:
                raise ValueError(
                    f"duplicate adjudication for pair ({a}, {b})"
                )
            adjudicated.add((a, b))
        missing = boundary_pairs - adjudicated
        if missing:
            raise ValueError(
                f"boundary pairs without adjudication: {sorted(missing)}"
            )
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/unit/test_rubric.py::TestAdjudicationLog -v`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add engine/prereg/adjudication.py tests/unit/test_rubric.py
git commit -m "feat(prereg): adjudication log schema for boundary-cell decisions (Plan 3)"
```

---

### Task 5: Rubric JSON I/O

**Files:**
- Create: `engine/prereg/rubric_io.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_rubric.py`:

```python
from pathlib import Path

from engine.prereg.rubric_attestation import RubricDraftingAttestation
from engine.prereg.rubric_io import (
    read_adjudication_log,
    read_rubric,
    read_rubric_attestation,
    write_adjudication_log,
    write_rubric,
    write_rubric_attestation,
)


class TestRubricIO:
    """JSON serialization roundtrip tests."""

    def test_rubric_write_read_roundtrip(self, tmp_path: Path) -> None:
        r = _make_rubric()
        p = tmp_path / "rubric.json"
        write_rubric(r, p)
        loaded = read_rubric(p)
        assert loaded.compute_hash() == r.compute_hash()
        assert loaded.entries[0].entry_id == r.entries[0].entry_id
        assert loaded.entries[0].boundary_rules[0].adjacent_entry_id == "LLM02"

    def test_rubric_json_is_human_readable(self, tmp_path: Path) -> None:
        r = _make_rubric()
        p = tmp_path / "rubric.json"
        write_rubric(r, p)
        content = p.read_text()
        assert '"entry_id": "LLM01"' in content

    def test_attestation_write_read_roundtrip(self, tmp_path: Path) -> None:
        att = RubricDraftingAttestation(
            viewed_corpus_before_drafting=False,
            viewed_corpus_details="",
            viewed_vote_data_before_drafting=False,
            viewed_vote_data_details="",
        )
        p = tmp_path / "rubric_attestation.json"
        write_rubric_attestation(att, p)
        loaded = read_rubric_attestation(p)
        assert loaded.viewed_corpus_before_drafting is False
        assert loaded.viewed_corpus_details == ""
        assert loaded.viewed_vote_data_before_drafting is False
        assert loaded.viewed_vote_data_details == ""

    def test_adjudication_log_write_read_roundtrip(self, tmp_path: Path) -> None:
        log = AdjudicationLog(
            rubric_hash="abc123",
            entries=(
                AdjudicationEntry(
                    entry_id_a="LLM01",
                    entry_id_b="LLM02",
                    decision="resolved:LLM01",
                    rationale="Test.",
                    adjudicator="Rock",
                    date="2026-05-20",
                ),
            ),
        )
        p = tmp_path / "adjudication_log.json"
        write_adjudication_log(log, p)
        loaded = read_adjudication_log(p)
        assert loaded.rubric_hash == "abc123"
        assert len(loaded.entries) == 1
        assert loaded.entries[0].decision == "resolved:LLM01"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_rubric.py::TestRubricIO -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the I/O module**

Create `engine/prereg/rubric_io.py`:

```python
"""JSON I/O for rubric, attestation, and adjudication log artifacts.

All writes produce human-readable, indented JSON.  Reads reconstruct
frozen dataclasses from the JSON.  Hashing uses the Rubric.compute_hash()
canonical form, not the pretty-printed file contents.
"""
from __future__ import annotations

import json
from pathlib import Path

from engine.prereg.adjudication import AdjudicationEntry, AdjudicationLog
from engine.prereg.rubric import BoundaryRule, Rubric, RubricEntry
from engine.prereg.rubric_attestation import RubricDraftingAttestation


def write_rubric(rubric: Rubric, path: Path) -> None:
    """Write *rubric* as human-readable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = rubric.to_dict()
    path.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n")


def _validate_pair(pair: list[str], entry_id: str) -> tuple[str, str]:
    """Validate and convert a co_occurrence_pair from JSON (Premortem3 R5)."""
    if len(pair) != 2:
        raise ValueError(
            f"{entry_id}: co_occurrence_pair must have exactly 2 elements, "
            f"got {len(pair)}: {pair}"
        )
    return (pair[0], pair[1])


def read_rubric(path: Path) -> Rubric:
    """Read a Rubric from JSON."""
    data = json.loads(path.read_text())
    entries = tuple(
        RubricEntry(
            entry_id=e["entry_id"],
            canonical_name=e["canonical_name"],
            in_scope=e["in_scope"],
            exclusions=tuple(e["exclusions"]),
            boundary_rules=tuple(
                BoundaryRule(
                    adjacent_entry_id=br["adjacent_entry_id"],
                    rule=br["rule"],
                    is_ambiguous=br["is_ambiguous"],
                )
                for br in e["boundary_rules"]
            ),
            positive_indicators=tuple(e["positive_indicators"]),
            negative_indicators=tuple(e["negative_indicators"]),
            co_occurrence_pairs=tuple(
                _validate_pair(pair, e["entry_id"])
                for pair in e["co_occurrence_pairs"]
            ),
            is_rollup_candidate=e["is_rollup_candidate"],
            rolled_into=e["rolled_into"],
        )
        for e in data["entries"]
    )
    return Rubric(
        cycle_id=data["cycle_id"],
        version=data["version"],
        entries=entries,
    )


def write_rubric_attestation(
    attestation: RubricDraftingAttestation, path: Path
) -> None:
    """Write drafting attestation as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "viewed_corpus_before_drafting": attestation.viewed_corpus_before_drafting,
        "viewed_corpus_details": attestation.viewed_corpus_details,
        "viewed_vote_data_before_drafting": attestation.viewed_vote_data_before_drafting,
        "viewed_vote_data_details": attestation.viewed_vote_data_details,
    }
    path.write_text(json.dumps(data, indent=2) + "\n")


def read_rubric_attestation(path: Path) -> RubricDraftingAttestation:
    """Read a RubricDraftingAttestation from JSON."""
    data = json.loads(path.read_text())
    return RubricDraftingAttestation(
        viewed_corpus_before_drafting=data["viewed_corpus_before_drafting"],
        viewed_corpus_details=data["viewed_corpus_details"],
        viewed_vote_data_before_drafting=data["viewed_vote_data_before_drafting"],
        viewed_vote_data_details=data["viewed_vote_data_details"],
    )


def write_adjudication_log(log: AdjudicationLog, path: Path) -> None:
    """Write adjudication log as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "rubric_hash": log.rubric_hash,
        "entries": [
            {
                "entry_id_a": e.entry_id_a,
                "entry_id_b": e.entry_id_b,
                "decision": e.decision,
                "rationale": e.rationale,
                "adjudicator": e.adjudicator,
                "date": e.date,
            }
            for e in log.entries
        ],
    }
    path.write_text(json.dumps(data, indent=2) + "\n")


def read_adjudication_log(path: Path) -> AdjudicationLog:
    """Read an AdjudicationLog from JSON."""
    data = json.loads(path.read_text())
    entries = tuple(
        AdjudicationEntry(
            entry_id_a=e["entry_id_a"],
            entry_id_b=e["entry_id_b"],
            decision=e["decision"],
            rationale=e["rationale"],
            adjudicator=e["adjudicator"],
            date=e["date"],
        )
        for e in data["entries"]
    )
    return AdjudicationLog(
        rubric_hash=data["rubric_hash"],
        entries=entries,
    )
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/unit/test_rubric.py::TestRubricIO -v`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add engine/prereg/rubric_io.py tests/unit/test_rubric.py
git commit -m "feat(prereg): rubric + attestation + adjudication JSON I/O (Plan 3)"
```

---

### Task 6: Pre-Classify Gate Logic

**Files:**
- Create: `engine/prereg/gates.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_rubric.py`:

```python
from engine.prereg.gates import (
    is_publishable,
    require_rubric_attestation,
    require_rubric_hash,
    require_rubric_hash_match,
)
from engine.prereg.manifest import PreregManifest
from engine.prereg.signoff import ReviewerSignoff


def _make_signoff(
    *,
    name: str = "Alice",
    path: str = "docs/REVIEWERS/alice-rubric.txt",
    sha: str = "abc123",
    ts: str = "2025-01-15T10:00:00+00:00",
    viewed: bool = False,
) -> ReviewerSignoff:
    return ReviewerSignoff(
        reviewer_name=name,
        reviewer_affiliation="Example Org",
        attestation_relative_path=path,
        attestation_sha256=sha,
        signed_at=ts,
        viewed_results_before_signoff=viewed,
    )


def _make_test_manifest(**overrides: object) -> PreregManifest:
    from typing import Any

    defaults: dict[str, Any] = {
        "engine_version": "0.3.0",
        "engine_version_range_min": "0.3.0",
        "engine_version_range_max": "0.3.0",
        "cycle_id": "2026",
        "taxonomy_hash": "aaa",
        "snapshot_hash": "bbb",
        "primary_spec": "negative_binomial_per_stratum",
        "robustness_specs": ("poisson_flat",),
        "flag_threshold_tau": 0.8,
        "statistic": "weighted_cohens_kappa",
        "measurability_minimum": 4,
        "prior_scale": 0.5,
        "concentration_shape": 5.0,
        "concentration_rate": 0.1,
        "ess_fraction": 0.4,
        "meaningful_kappa_n": 4,
        "prng_seed": 20260520,
        "rubric_drafting_attestation": None,
        "rubric_reviewer": None,
        "statistical_reviewer": None,
        "classifier_rule_hash": None,
        "rubric_hash": None,
        "post_hoc_register_path": None,
    }
    defaults.update(overrides)
    return PreregManifest(**defaults)


class TestGates:
    """Tests for pre-classify gate checks."""

    def test_require_rubric_attestation_passes(self) -> None:
        m = _make_test_manifest(
            rubric_drafting_attestation=RubricDraftingAttestation(
                viewed_corpus_before_drafting=False,
                viewed_corpus_details="",
                viewed_vote_data_before_drafting=False,
                viewed_vote_data_details="",
            ),
        )
        require_rubric_attestation(m)  # should not raise

    def test_require_rubric_attestation_fails_when_none(self) -> None:
        m = _make_test_manifest(rubric_drafting_attestation=None)
        with pytest.raises(ValueError, match="rubric drafting attestation required"):
            require_rubric_attestation(m)

    def test_require_rubric_hash_passes(self) -> None:
        m = _make_test_manifest(rubric_hash="abc123")
        require_rubric_hash(m)  # should not raise

    def test_require_rubric_hash_fails_when_none(self) -> None:
        m = _make_test_manifest(rubric_hash=None)
        with pytest.raises(ValueError, match="rubric hash required"):
            require_rubric_hash(m)

    def test_is_publishable_true_with_external_reviewers(self) -> None:
        m = _make_test_manifest(
            rubric_reviewer=_make_signoff(name="External-A"),
            statistical_reviewer=_make_signoff(
                name="External-B",
                path="docs/REVIEWERS/ext-b.txt",
                sha="def456",
            ),
        )
        assert is_publishable(m, ranking_author="Rock Lambros") is True

    def test_is_publishable_false_when_reviewer_is_author(self) -> None:
        m = _make_test_manifest(
            rubric_reviewer=_make_signoff(name="Rock Lambros"),
            statistical_reviewer=_make_signoff(
                name="External-B",
                path="docs/REVIEWERS/ext-b.txt",
                sha="def456",
            ),
        )
        assert is_publishable(m, ranking_author="Rock Lambros") is False

    def test_is_publishable_false_when_statistical_is_author(self) -> None:
        m = _make_test_manifest(
            rubric_reviewer=_make_signoff(name="External-A"),
            statistical_reviewer=_make_signoff(
                name="Rock Lambros",
                path="docs/REVIEWERS/rock.txt",
                sha="ghi789",
            ),
        )
        assert is_publishable(m, ranking_author="Rock Lambros") is False

    def test_is_publishable_false_when_reviewer_missing(self) -> None:
        m = _make_test_manifest(rubric_reviewer=None, statistical_reviewer=None)
        assert is_publishable(m, ranking_author="Rock Lambros") is False

    def test_is_publishable_name_normalization(self) -> None:
        m = _make_test_manifest(
            rubric_reviewer=_make_signoff(name="rock  lambros"),
            statistical_reviewer=_make_signoff(
                name="External-B",
                path="docs/REVIEWERS/ext-b.txt",
                sha="def456",
            ),
        )
        assert is_publishable(m, ranking_author="Rock Lambros") is False

    def test_require_rubric_hash_match_passes(self, tmp_path: Path) -> None:
        r = _make_rubric()
        p = tmp_path / "rubric.json"
        write_rubric(r, p)
        m = _make_test_manifest(rubric_hash=r.compute_hash())
        require_rubric_hash_match(m, p)  # should not raise

    def test_require_rubric_hash_match_fails_on_mismatch(self, tmp_path: Path) -> None:
        r = _make_rubric()
        p = tmp_path / "rubric.json"
        write_rubric(r, p)
        m = _make_test_manifest(rubric_hash="wrong_hash")
        with pytest.raises(ValueError, match="rubric hash mismatch"):
            require_rubric_hash_match(m, p)

    def test_require_rubric_hash_match_fails_when_none(self, tmp_path: Path) -> None:
        r = _make_rubric()
        p = tmp_path / "rubric.json"
        write_rubric(r, p)
        m = _make_test_manifest(rubric_hash=None)
        with pytest.raises(ValueError, match="rubric hash is None"):
            require_rubric_hash_match(m, p)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_rubric.py::TestGates -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the gates module**

Create `engine/prereg/gates.py`:

```python
"""Pre-classify gate checks and publishability verification.

These functions enforce HANDOFF §6 controls at the module level. The CLI
commands (engine/cli/rubric.py) call these; the functions are also
independently testable.
"""
from __future__ import annotations

from pathlib import Path

from engine.prereg.manifest import PreregManifest


def _normalize_name(name: str) -> str:
    """Collapse whitespace and lowercase for name comparison."""
    return " ".join(name.lower().split())


def require_rubric_attestation(manifest: PreregManifest) -> None:
    """Raise if rubric drafting attestation is missing (HANDOFF §6 control 11(d))."""
    if manifest.rubric_drafting_attestation is None:
        raise ValueError(
            "rubric drafting attestation required before classify — "
            "populate rubric_attestation.json first"
        )


def require_rubric_hash(manifest: PreregManifest) -> None:
    """Raise if rubric hash is missing from the manifest."""
    if manifest.rubric_hash is None:
        raise ValueError(
            "rubric hash required before classify — "
            "freeze the rubric first via `freeze-rubric`"
        )


def require_rubric_hash_match(
    manifest: PreregManifest, rubric_path: Path
) -> None:
    """Raise if manifest.rubric_hash does not match the actual rubric file.

    This closes the integrity gap where the rubric file could be modified
    after freeze without detection.  Called at classify time.
    """
    from engine.prereg.rubric_io import read_rubric

    if manifest.rubric_hash is None:
        raise ValueError("rubric hash is None in manifest")
    rubric = read_rubric(rubric_path)
    actual_hash = rubric.compute_hash()
    if manifest.rubric_hash != actual_hash:
        raise ValueError(
            f"rubric hash mismatch: manifest={manifest.rubric_hash}, "
            f"file={actual_hash}. Was rubric.json modified after freeze?"
        )


def is_publishable(manifest: PreregManifest, *, ranking_author: str) -> bool:
    """Check publication readiness including reviewer independence.

    Combines the manifest's mechanical ``non_publishable`` derivation with
    the discipline-based reviewer-independence check (HANDOFF §4 Crosswalk
    authorship + REVIEWERS.md PRE-PUBLISH CHECKLIST).

    Name comparison is normalized (case-insensitive, whitespace-collapsed)
    to prevent accidental bypass via formatting differences.
    """
    if manifest.non_publishable:
        return False
    author_norm = _normalize_name(ranking_author)
    if (
        manifest.rubric_reviewer is not None
        and _normalize_name(manifest.rubric_reviewer.reviewer_name) == author_norm
    ):
        return False
    if (
        manifest.statistical_reviewer is not None
        and _normalize_name(manifest.statistical_reviewer.reviewer_name)
        == author_norm
    ):
        return False
    return True
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/unit/test_rubric.py::TestGates -v`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add engine/prereg/gates.py tests/unit/test_rubric.py
git commit -m "feat(prereg): pre-classify gate logic + publishability check (Plan 3)"
```

---

### Task 7: Rubric CLI Commands

**Files:**
- Create: `engine/cli/rubric.py`
- Modify: `engine/cli/main.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_rubric.py`:

```python
from click.testing import CliRunner

from engine.cli.rubric import freeze_rubric_cmd, validate_rubric_cmd


class TestRubricCLI:
    """Smoke tests for rubric CLI commands."""

    def test_validate_rubric_missing_file(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(
            validate_rubric_cmd,
            ["--rubric", str(tmp_path / "nonexistent.json")],
        )
        assert result.exit_code != 0
        assert "does not exist" in result.output or "Error" in result.output

    def test_validate_rubric_valid_file(self, tmp_path: Path) -> None:
        r = _make_rubric()
        p = tmp_path / "rubric.json"
        write_rubric(r, p)
        runner = CliRunner()
        result = runner.invoke(
            validate_rubric_cmd,
            [
                "--rubric", str(p),
                "--expected-ids", "LLM01,LLM02",
            ],
        )
        assert result.exit_code == 0
        assert "valid" in result.output.lower() or "hash" in result.output.lower()

    def test_validate_rubric_with_taxonomy(self, tmp_path: Path) -> None:
        r = _make_rubric()
        rubric_path = tmp_path / "rubric.json"
        write_rubric(r, rubric_path)
        taxonomy_path = tmp_path / "taxonomy.json"
        taxonomy_path.write_text(json.dumps({
            "cycle_id": "2026",
            "entries": [
                {"entry_id": "LLM01", "canonical_name": "Prompt Injection"},
                {"entry_id": "LLM02", "canonical_name": "Sensitive Info"},
            ],
        }))
        runner = CliRunner()
        result = runner.invoke(
            validate_rubric_cmd,
            ["--rubric", str(rubric_path), "--taxonomy", str(taxonomy_path)],
        )
        assert result.exit_code == 0
        assert "valid" in result.output.lower() or "hash" in result.output.lower()

    def test_validate_rubric_taxonomy_and_expected_ids_exclusive(self, tmp_path: Path) -> None:
        r = _make_rubric()
        rubric_path = tmp_path / "rubric.json"
        write_rubric(r, rubric_path)
        taxonomy_path = tmp_path / "taxonomy.json"
        taxonomy_path.write_text(json.dumps({"cycle_id": "2026", "entries": []}))
        runner = CliRunner()
        result = runner.invoke(
            validate_rubric_cmd,
            [
                "--rubric", str(rubric_path),
                "--taxonomy", str(taxonomy_path),
                "--expected-ids", "LLM01,LLM02",
            ],
        )
        assert result.exit_code != 0
        assert "mutually exclusive" in result.output.lower() or "cannot" in result.output.lower()

    def test_freeze_rubric_missing_attestation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import subprocess

        r = _make_rubric()
        rubric_path = tmp_path / "rubric.json"
        write_rubric(r, rubric_path)
        # Initialize a git repo in tmp_path so verify_committed() can resolve
        # paths via git rev-parse --show-toplevel (Premortem3 R2).
        subprocess.run(
            ["git", "init"], cwd=tmp_path, capture_output=True, check=True
        )
        subprocess.run(
            ["git", "add", "rubric.json"],
            cwd=tmp_path, capture_output=True, check=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "init"],
            cwd=tmp_path, capture_output=True, check=True,
        )
        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            freeze_rubric_cmd,
            [
                "--rubric", str(rubric_path),
                "--cycle-dir", str(tmp_path),
            ],
        )
        assert result.exit_code != 0
        assert "attestation" in result.output.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_rubric.py::TestRubricCLI -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the CLI module**

Create `engine/cli/rubric.py`:

```python
"""CLI commands for rubric validation and freeze workflow."""
from __future__ import annotations

import json
from pathlib import Path

import click

from engine.prereg.rubric_io import read_rubric, read_rubric_attestation


@click.command("validate-rubric")
@click.option(
    "--rubric",
    type=click.Path(exists=False),
    required=True,
    help="Path to rubric.json",
)
@click.option(
    "--expected-ids",
    type=str,
    default=None,
    help="Comma-separated expected entry IDs for completeness check.",
)
@click.option(
    "--taxonomy",
    type=click.Path(exists=True),
    default=None,
    help="Path to taxonomy.json — reads entry IDs automatically (mutually exclusive with --expected-ids).",
)
@click.option(
    "--no-adjacency-attested",
    type=str,
    default=None,
    help="Comma-separated entry IDs with attested no-adjacency (allowed empty boundary_rules). Premortem3 R4.",
)
def validate_rubric_cmd(
    rubric: str,
    expected_ids: str | None,
    taxonomy: str | None,
    no_adjacency_attested: str | None,
) -> None:
    """Validate a rubric file: schema, completeness, boundary rules."""
    if expected_ids is not None and taxonomy is not None:
        raise click.ClickException(
            "--expected-ids and --taxonomy are mutually exclusive. "
            "Cannot specify both."
        )

    rubric_path = Path(rubric)
    if not rubric_path.exists():
        raise click.ClickException(f"rubric file does not exist: {rubric_path}")

    r = read_rubric(rubric_path)

    attested: set[str] | None = None
    if no_adjacency_attested is not None:
        attested = {i.strip() for i in no_adjacency_attested.split(",") if i.strip()}

    ids: set[str] | None = None
    if taxonomy is not None:
        tax_data = json.loads(Path(taxonomy).read_text())
        ids = {e["entry_id"] for e in tax_data["entries"]}
        click.echo(f"Loaded {len(ids)} entry IDs from taxonomy.json.")
    elif expected_ids is not None:
        ids = {i.strip() for i in expected_ids.split(",")}

    if ids is not None:
        r.validate_completeness(ids, no_adjacency_attested=attested)
        click.echo(f"Completeness: {len(r.entries)} entries match expected set.")

    r.validate_boundary_rules()
    click.echo(f"Boundary rules: all paired.")

    r.validate_co_occurrences()
    click.echo(f"Co-occurrence pairs: all reference valid entries.")

    h = r.compute_hash()
    click.echo(f"Rubric hash: {h}")
    click.echo("Rubric is valid.")


@click.command("freeze-rubric")
@click.option(
    "--rubric",
    type=click.Path(exists=True),
    required=True,
    help="Path to rubric.json",
)
@click.option(
    "--cycle-dir",
    type=click.Path(exists=True),
    required=True,
    help="Path to cycle directory (e.g., projects/owasp-llm/cycles/2026).",
)
@click.option(
    "--no-adjacency-attested",
    type=str,
    default=None,
    help="Comma-separated entry IDs with attested no-adjacency (allowed empty boundary_rules). Premortem3 R4.",
)
def freeze_rubric_cmd(
    rubric: str, cycle_dir: str, no_adjacency_attested: str | None
) -> None:
    """Freeze the rubric: validate, require attestation, verify committed, emit hash."""
    import json as _json
    import subprocess

    from engine.prereg.attestation import verify_committed

    rubric_path = Path(rubric)
    cycle = Path(cycle_dir)

    attested: set[str] | None = None
    if no_adjacency_attested is not None:
        attested = {i.strip() for i in no_adjacency_attested.split(",") if i.strip()}

    # Verify rubric file is committed to git (Premortem2 R7).
    repo_root = Path(
        subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], text=True
        ).strip()
    )
    verify_committed(rubric_path, repo_root)

    r = read_rubric(rubric_path)

    # Completeness check against taxonomy (Premortem2 R5).
    taxonomy_path = cycle / "taxonomy" / "taxonomy.json"
    if taxonomy_path.exists():
        tax_data = _json.loads(taxonomy_path.read_text())
        ids = {e["entry_id"] for e in tax_data["entries"]}
        r.validate_completeness(ids, no_adjacency_attested=attested)
        click.echo(f"Completeness: {len(r.entries)} entries match taxonomy.")
    else:
        click.echo(
            "WARNING: taxonomy.json not found — skipping completeness check."
        )

    r.validate_boundary_rules()
    r.validate_co_occurrences()

    attestation_path = cycle / "prereg" / "rubric_attestation.json"
    if not attestation_path.exists():
        raise click.ClickException(
            f"rubric attestation not found at {attestation_path} — "
            "populate it before freezing"
        )
    verify_committed(attestation_path, repo_root)
    att = read_rubric_attestation(attestation_path)

    rubric_hash = r.compute_hash()
    click.echo(f"Rubric hash: {rubric_hash}")
    click.echo(
        f"Viewed corpus before drafting: {att.viewed_corpus_before_drafting}"
    )
    if att.viewed_corpus_before_drafting:
        click.echo(
            "WARNING: corpus-informed rubric — report will carry caveat."
        )

    click.echo(f"Entries: {len(r.entries)}")
    click.echo(f"  Standalone: {len(r.standalone_entries())}")
    click.echo(f"  Rollup candidates: {len(r.rollup_candidates())}")
    click.echo("Rubric frozen. Add rubric_hash to prereg manifest to lock.")
```

- [ ] **Step 4: Wire commands to CLI**

In `engine/cli/main.py`, add:

```python
from engine.cli.rubric import freeze_rubric_cmd, validate_rubric_cmd
```

And in the command registration section:

```python
cli.add_command(validate_rubric_cmd)
cli.add_command(freeze_rubric_cmd)
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/unit/test_rubric.py::TestRubricCLI -v`
Expected: All pass

Run: `uv run mypy engine/cli/rubric.py --strict`
Expected: Success

- [ ] **Step 6: Commit**

```bash
git add engine/cli/rubric.py engine/cli/main.py tests/unit/test_rubric.py
git commit -m "feat(cli): validate-rubric + freeze-rubric commands (Plan 3)"
```

---

### Task 8: Update `engine/prereg/__init__.py` Exports

**Files:**
- Modify: `engine/prereg/__init__.py`

- [ ] **Step 1: Update exports**

The file is currently empty (1 line). Replace with:

```python
"""Pre-registration module — hash-locked artifacts for methodology integrity."""
```

(The module's public API is accessed via direct imports from submodules. An `__init__.py` with re-exports is not needed since all consumers import from specific submodules like `engine.prereg.rubric`, `engine.prereg.gates`, etc.)

- [ ] **Step 2: Add end-to-end integration test (Premortem2 R3)**

Append to `tests/unit/test_rubric.py`:

```python
class TestFreezeWorkflowIntegration:
    """End-to-end integration test for the rubric freeze workflow.

    Exercises: construct rubric → write → write attestation → write adjudication
    → validate-rubric CLI → freeze-rubric CLI → hash chain verification.
    """

    def test_full_freeze_workflow(self, tmp_path: Path) -> None:
        # 1. Construct a 3-entry rubric (2 regular + 1 rollup) with paired rules.
        entry_a = RubricEntry(
            entry_id="LLM01",
            canonical_name="Prompt Injection",
            in_scope="Attacks via crafted input.",
            exclusions=("Data exfiltration only",),
            boundary_rules=(
                BoundaryRule(
                    adjacent_entry_id="LLM02",
                    rule="If data extraction is primary goal, classify as LLM02.",
                    is_ambiguous=False,
                ),
            ),
            positive_indicators=("prompt injection",),
            negative_indicators=("data exfiltration only",),
            co_occurrence_pairs=(("LLM01", "LLM02"),),
            is_rollup_candidate=False,
            rolled_into=None,
        )
        entry_b = RubricEntry(
            entry_id="LLM02",
            canonical_name="Sensitive Info Disclosure",
            in_scope="Data extraction attacks.",
            exclusions=("Prompt manipulation only",),
            boundary_rules=(
                BoundaryRule(
                    adjacent_entry_id="LLM01",
                    rule="If prompt manipulation is primary mechanism, classify as LLM01.",
                    is_ambiguous=False,
                ),
            ),
            positive_indicators=("data leak",),
            negative_indicators=("no data extracted",),
            co_occurrence_pairs=(("LLM01", "LLM02"),),
            is_rollup_candidate=False,
            rolled_into=None,
        )
        rollup = RubricEntry(
            entry_id="ROLL-X",
            canonical_name="Cross-Modal Bypass",
            in_scope="Multi-modal injection.",
            exclusions=(),
            boundary_rules=(),
            positive_indicators=("multi-modal",),
            negative_indicators=("text-only",),
            co_occurrence_pairs=(),
            is_rollup_candidate=True,
            rolled_into="LLM01",
        )
        rubric = Rubric(
            cycle_id="2026", version="1.0.0",
            entries=(entry_a, entry_b, rollup),
        )

        # 2. All validations pass.
        rubric.validate_completeness({"LLM01", "LLM02", "ROLL-X"})
        rubric.validate_boundary_rules()
        rubric.validate_co_occurrences()

        # 3. Write rubric.
        rubric_path = tmp_path / "prereg" / "rubric.json"
        write_rubric(rubric, rubric_path)

        # 4. Read back and verify hash stability.
        loaded = read_rubric(rubric_path)
        assert loaded.compute_hash() == rubric.compute_hash()

        # 5. Write attestation.
        att = RubricDraftingAttestation(
            viewed_corpus_before_drafting=False,
            viewed_corpus_details="",
            viewed_vote_data_before_drafting=False,
            viewed_vote_data_details="",
        )
        att_path = tmp_path / "prereg" / "rubric_attestation.json"
        write_rubric_attestation(att, att_path)

        # 6. Write adjudication log covering the one boundary pair.
        log = AdjudicationLog(
            rubric_hash=rubric.compute_hash(),
            entries=(
                AdjudicationEntry(
                    entry_id_a="LLM01",
                    entry_id_b="LLM02",
                    decision="resolved:LLM01",
                    rationale="Prompt manipulation is primary.",
                    adjudicator="Rock Lambros",
                    date="2026-05-20",
                ),
            ),
        )
        log.validate_coverage(rubric)
        log_path = tmp_path / "prereg" / "adjudication_log.json"
        write_adjudication_log(log, log_path)

        # 7. Verify hash chain: manifest rubric_hash matches file.
        from engine.prereg.gates import require_rubric_hash_match

        m = _make_test_manifest(rubric_hash=rubric.compute_hash())
        require_rubric_hash_match(m, rubric_path)  # should not raise

        # 8. Verify mismatch detection.
        m_bad = _make_test_manifest(rubric_hash="tampered_hash")
        with pytest.raises(ValueError, match="rubric hash mismatch"):
            require_rubric_hash_match(m_bad, rubric_path)
```

- [ ] **Step 3: Run full test suite**

Run: `uv run pytest -x -q 2>&1 | tail -5`
Expected: All pass (422 + new tests from Tasks 1-8)

Run: `uv run mypy engine tests --strict 2>&1 | tail -3`
Expected: Success

- [ ] **Step 4: Commit**

```bash
git add engine/prereg/__init__.py tests/unit/test_rubric.py
git commit -m "chore(prereg): update module docstring + integration test (Plan 3)"
```

---

### Task 9: RUBRIC-WORKFLOW.md

**Files:**
- Create: `docs/RUBRIC-WORKFLOW.md`

- [ ] **Step 1: Write the workflow documentation**

Create `docs/RUBRIC-WORKFLOW.md`:

```markdown
# Rubric Drafting and Freeze Workflow

This document describes the procedural steps for drafting, adjudicating, and
freezing the classification rubric for an incident-rank-validation cycle.

## Prerequisites

- Engine v0.3.0+ with rubric data model, CLI commands, and gate logic.
- Entry definitions vendored in `projects/<project>/cycles/<cycle>/taxonomy/`.
- `docs/REVIEWERS.md` consulted for reviewer state (INTERIM vs EXTERNAL).

## Vote-Blindness Rule (HANDOFF §6 control 2)

The rubric drafter (Claude + Rock) MUST NOT view vote results during drafting.
This means:

1. Do not open the `2026/polling/` directory in the source repo
   (`https://github.com/GenAI-Security-Project/GenAI-LLM-Top10`) or any
   file containing vote results.
2. Do not read the `Analysis` or `Results` sheets from the voting spreadsheet.
3. The CASE 2 ranking order in HANDOFF §2 is metadata about *what* the entries
   are and how they were named, not vote-influence data. Reading HANDOFF §2
   for entry names is permitted; reading it to infer relative importance is not.

The `rubric_attestation.json` records whether vote-blindness was maintained.
If violated, the cycle is non-publishable regardless of other controls.

## Step-by-Step Procedure

### 1. Vendor Taxonomy

Copy the 20 entry definitions to `projects/<project>/cycles/<cycle>/taxonomy/`
and create `taxonomy.json`. Use the `vendor-snapshot` pattern from Plan 2.

### 2. Draft Rubric Entries

For each of the 20 entries, produce a `RubricEntry` with all 8 required fields:

- **entry_id**: Canonical identifier (e.g., `LLM01`).
- **canonical_name**: Official name from the taxonomy.
- **in_scope**: What incidents this entry covers.
- **exclusions**: What this entry explicitly does NOT cover.
- **boundary_rules**: Pairwise rules against adjacent/confusable entries.
- **positive_indicators**: Keywords, patterns, or signals that suggest classification.
- **negative_indicators**: Signals that suggest classification elsewhere.
- **co_occurrence_pairs**: Entry pairs expected to co-occur on the same incident.

Rolled-up candidates (4 entries) get their own rubric entry with
`is_rollup_candidate=true` and `rolled_into` pointing to the parent.

### 3. Identify Boundary Cells

For every pair of entries that could be confused:

- Write a boundary rule on BOTH sides (rules must be paired).
- If the boundary is genuinely ambiguous (50/50): mark `is_ambiguous=true`.
  This propagates as label uncertainty in the measurement model.
- If the boundary is clear: mark `is_ambiguous=false` and state the rule.

### 4. Rock Adjudicates

Rock reviews every boundary rule and either:

- Confirms the rule as written, or
- Marks it `ambiguous-both-labels` with rationale.

Adjudication decisions go into `adjudication_log.json`.

### 5. Populate Attestation

Create `rubric_attestation.json`:

```json
{
  "viewed_corpus_before_drafting": false,
  "viewed_corpus_details": "",
  "viewed_vote_data_before_drafting": false,
  "viewed_vote_data_details": ""
}
```

If the drafter viewed corpus samples, set `viewed_corpus_before_drafting` to `true` and list which samples.
The report will carry a "corpus-informed rubric" caveat (HANDOFF §6 control 11(d)).

If the drafter viewed any vote results (including rank ordinals from HANDOFF §2), set
`viewed_vote_data_before_drafting` to `true` and describe what was seen. A vote-data
exposure makes the cycle non-publishable per HANDOFF §6 control 2 (Premortem3 R3).

### 6. Validate

```bash
uv run python -m engine.cli.main validate-rubric \
  --rubric projects/owasp-llm/cycles/2026/prereg/rubric.json \
  --taxonomy projects/owasp-llm/cycles/2026/taxonomy/taxonomy.json
```

### 7. Freeze (requires external reviewer)

```bash
uv run python -m engine.cli.main freeze-rubric \
  --rubric projects/owasp-llm/cycles/2026/prereg/rubric.json \
  --cycle-dir projects/owasp-llm/cycles/2026
```

This emits the rubric hash. Add it to the prereg manifest and lock.

### 8. External Reviewer Signoff

Per HANDOFF §6 control 5 and REVIEWERS.md:

1. External reviewer reads the rubric.
2. Reviewer writes attestation file at `docs/REVIEWERS/<name>-rubric.txt`.
3. Compute SHA-256, update REVIEWERS.md, commit atomically.
4. Update the cycle manifest with reviewer identity and hash.

If no external reviewer is available, the run proceeds as `non_publishable=True`.

## Rubric Amendments

Pre-registration is a commitment, not a prison. If a boundary rule deficiency is discovered
after freeze (e.g., during gold-set labeling in Plan 4), the rubric may be amended with
full disclosure:

1. **Version bump:** Increment `Rubric.version` (e.g., `"1.0.0"` → `"1.1.0"`).
2. **Rationale:** Document the deficiency and the change in the pre-registration diff
   artifact (HANDOFF §5.5). Include which boundary rules changed and why.
3. **Re-freeze:** Run `freeze-rubric` to produce a new rubric hash.
4. **Re-lock manifest:** Update `rubric_hash` in the manifest and re-run `write_lock`.
5. **Disclosure:** The report's pre-reg diff section shows original vs amended rubric
   hashes and the rationale. The amendment is disclosed, not hidden.
6. **Re-review (Premortem2 R10):** If the run is targeted for publication, the amended
   rubric requires external reviewer re-signoff. If the reviewer is unavailable, the
   report carries an "amendment not independently reviewed" disclosure alongside the
   pre-reg diff. An un-reviewed amendment does not make the run non-publishable by
   itself (the original rubric WAS reviewed), but the disclosure is mandatory.

The default is "do not amend" — the rubric is frozen for a reason. Amendments are for
genuine deficiencies discovered during execution, not for preference changes or
optimization. Each amendment requires Rock's sign-off and is recorded in the
adjudication log.
```

- [ ] **Step 2: Commit**

```bash
git add docs/RUBRIC-WORKFLOW.md
git commit -m "docs: rubric drafting and freeze workflow procedure (Plan 3)"
```

---

### Task 10: CI Verification + Version Bump + Tag v0.3.0-rc1

**Files:**
- Modify: `engine/version.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `docs/METHODOLOGY-CHANGELOG.md`
- Modify: `tests/test_bootstrap.py`

- [ ] **Step 1: Run full test suite locally**

Run: `uv run pytest -v 2>&1 | tail -20`
Expected: All previous tests + new Plan 3 tests pass.

Run: `uv run mypy engine tests --strict`
Expected: Success

Run: `uv run ruff check .`
Expected: No errors

- [ ] **Step 2: Bump version**

In `engine/version.py`:
```python
__version__ = "0.3.0"
```

In `pyproject.toml`:
```toml
version = "0.3.0"
```

Update `uv.lock`:
```bash
uv lock
```

In `tests/test_bootstrap.py`, update version assertion to `"0.3.0"`.

- [ ] **Step 3: Add methodology changelog entry**

In `docs/METHODOLOGY-CHANGELOG.md`, add at the top (before the 0.2.0 entry):

```markdown
## 0.3.0 (Plan 3, 2026-05-20)

Rubric freeze workflow scaffolding: data model, serialization, CLI, gate logic.

Key deliverables:
- `engine/prereg/rubric.py`: `RubricEntry`, `BoundaryRule`, `Rubric` dataclasses with hash computation and completeness/boundary validation.
- `engine/prereg/adjudication.py`: `AdjudicationEntry`, `AdjudicationLog` with boundary-coverage validation.
- `engine/prereg/rubric_io.py`: JSON serialization/deserialization for rubric, attestation, and adjudication log artifacts.
- `engine/prereg/gates.py`: Pre-classify gate (`require_rubric_attestation`, `require_rubric_hash`) and publishability check (`is_publishable` with reviewer-independence verification).
- `engine/cli/rubric.py`: `validate-rubric` and `freeze-rubric` CLI commands.
- `rubric_hash` field added to `PreregManifest` (hash-locked alongside existing `taxonomy_hash` and `snapshot_hash`).

Methodology decision: the rubric data model defines 10 fields per entry (the 8 HANDOFF §5.2 Artifact 1 required fields plus `is_rollup_candidate` and `rolled_into` for the rollup sub-test). Boundary rules are paired: if A→B exists, B→A must exist, enforced by `validate_boundary_rules()`. Ambiguous boundaries carry `is_ambiguous=true` and propagate as label uncertainty (HANDOFF §5.2: "Genuine 50/50 calls are recorded as both labels with ambiguity"). The `is_publishable()` gate combines the manifest's mechanical `non_publishable` derivation with the discipline-based reviewer-independence check. Rubric content (Phase B) and freeze (Phase C) are deferred.
```

- [ ] **Step 4: Commit version bump**

```bash
git add engine/version.py pyproject.toml uv.lock tests/test_bootstrap.py docs/METHODOLOGY-CHANGELOG.md
git commit -m "docs: version 0.3.0 + Plan 3 scaffolding methodology changelog"
```

- [ ] **Step 5: Push branch and create PR**

```bash
git push -u origin plan3/rubric-freeze-workflow
```

Create a draft PR against main. Title: `feat(prereg): rubric freeze workflow scaffolding (Plan 3)`

- [ ] **Step 6: Verify CI green (C5 erratum lesson)**

Check GitHub Actions: both ubuntu-latest and macos-latest jobs must complete green. Do not assume CI passes because local tests passed. If CI fails, diagnose and fix before proceeding.

- [ ] **Step 7: Merge and tag v0.3.0-rc1 (Premortem2 R11)**

After CI is green, merge the PR to main (via GitHub or CLI). Then:

```bash
git checkout main && git pull origin main
git tag v0.3.0-rc1
git push origin v0.3.0-rc1
```

The tag MUST be created AFTER merge to main so it anchors to the merge commit, not the branch tip. Plan 2 required re-anchoring the tag twice because it was created before merge — do not repeat that pattern.

---

## Phase B: Rubric Drafting (runnable now; requires vote-blindness)

Phase B produces the actual rubric content. It requires a vote-blind session: do NOT open the `2026/polling/` directory in the source repo (`https://github.com/GenAI-Security-Project/GenAI-LLM-Top10`) or any vote data.

**Corpus-blindness decision:** Plan 2 is DONE, so corpus samples are available. The drafter MAY view corpus samples (e.g., to understand what a "bare LLM03 default-seed CVE" looks like). If corpus samples are viewed, `rubric_attestation.json` records `viewed_corpus_before_drafting=true` and the report carries a "corpus-informed rubric" caveat (HANDOFF §6 control 11(d)). This is a disclosure, not a prohibition.

---

### Task 11: Taxonomy Vendoring

**VOTE-BLINDNESS CONSTRAINT (Premortem4 R2):** Do NOT read, open, or reference any files under `2026/polling/` in the source repo. Do NOT access vote results, Google Forms responses, or the HANDOFF §2 rank ordinals. If the source directory listing shows `polling/`, ignore it completely. Violation invalidates the cycle's blinding guarantee and makes it non-publishable per HANDOFF §6 control 2.

**Files:**
- Create: `projects/owasp-llm/cycles/2026/taxonomy/taxonomy.json`
- Create: `projects/owasp-llm/cycles/2026/taxonomy/*.md` (20 files)

- [ ] **Step 1: Clone source repo and verify clean state**

Clone the source repo at a known commit for provenance (Premortem3 R6, Premortem4 R1):

```bash
# Source repo — clone to a temp directory (standalone, no local-path dependency)
SOURCE_REPO="https://github.com/GenAI-Security-Project/GenAI-LLM-Top10"
SOURCE_DIR=$(mktemp -d)
git clone --depth 50 "$SOURCE_REPO" "$SOURCE_DIR"

# Record the HEAD SHA for provenance
SOURCE_SHA=$(git -C "$SOURCE_DIR" rev-parse HEAD)
echo "Source repo HEAD: $SOURCE_SHA"

# Discover entry definition files
ls "$SOURCE_DIR"/2026/LLM*.md
ls "$SOURCE_DIR"/2026/new_entry_candidates/
```

- [ ] **Step 2: Copy 10 incumbent files**

```bash
mkdir -p projects/owasp-llm/cycles/2026/taxonomy
cp "$SOURCE_DIR"/2026/LLM01_*.md \
   "$SOURCE_DIR"/2026/LLM02_*.md \
   "$SOURCE_DIR"/2026/LLM03_*.md \
   "$SOURCE_DIR"/2026/LLM04_*.md \
   "$SOURCE_DIR"/2026/LLM05_*.md \
   "$SOURCE_DIR"/2026/LLM06_*.md \
   "$SOURCE_DIR"/2026/LLM07_*.md \
   "$SOURCE_DIR"/2026/LLM08_*.md \
   "$SOURCE_DIR"/2026/LLM09_*.md \
   "$SOURCE_DIR"/2026/LLM10_*.md \
   projects/owasp-llm/cycles/2026/taxonomy/
```

- [ ] **Step 3: Copy 10 candidate files**

```bash
cp "$SOURCE_DIR"/2026/new_entry_candidates/*.md \
   projects/owasp-llm/cycles/2026/taxonomy/
```

- [ ] **Step 4: Create taxonomy.json**

Create `projects/owasp-llm/cycles/2026/taxonomy/taxonomy.json` with the 20-entry mapping. Example structure:

```json
{
  "cycle_id": "2026",
  "source_repo": "https://github.com/GenAI-Security-Project/GenAI-LLM-Top10",
  "vendored_at": "2026-05-20",
  "entries": [
    {"entry_id": "LLM01", "canonical_name": "Prompt Injection", "source_file": "LLM01_PromptInjection.md", "is_incumbent": true, "is_rollup_candidate": false, "rolled_into": null},
    {"entry_id": "LLM02", "canonical_name": "Sensitive Information Disclosure", "source_file": "LLM02_SensitiveInformationDisclosure.md", "is_incumbent": true, "is_rollup_candidate": false, "rolled_into": null},
    {"entry_id": "LLM03", "canonical_name": "Supply Chain Vulnerabilities", "source_file": "LLM03_SupplyChain.md", "is_incumbent": true, "is_rollup_candidate": false, "rolled_into": null},
    {"entry_id": "LLM04", "canonical_name": "Data and Model Poisoning", "source_file": "LLM04_DataModelPoisoning.md", "is_incumbent": true, "is_rollup_candidate": false, "rolled_into": null},
    {"entry_id": "LLM05", "canonical_name": "Improper Output Handling", "source_file": "LLM05_ImproperOutputHandling.md", "is_incumbent": true, "is_rollup_candidate": false, "rolled_into": null},
    {"entry_id": "LLM06", "canonical_name": "Excessive Agency", "source_file": "LLM06_ExcessiveAgency.md", "is_incumbent": true, "is_rollup_candidate": false, "rolled_into": null},
    {"entry_id": "LLM07", "canonical_name": "Hidden Context Exposure", "source_file": "LLM07_HiddenContextExposure.md", "is_incumbent": true, "is_rollup_candidate": false, "rolled_into": null},
    {"entry_id": "LLM08", "canonical_name": "Vector and Embedding Weaknesses", "source_file": "LLM08_VectorAndEmbeddingWeaknesses.md", "is_incumbent": true, "is_rollup_candidate": false, "rolled_into": null},
    {"entry_id": "LLM09", "canonical_name": "Misinformation", "source_file": "LLM09_Misinformation.md", "is_incumbent": true, "is_rollup_candidate": false, "rolled_into": null},
    {"entry_id": "LLM10", "canonical_name": "Unbounded Consumption", "source_file": "LLM10_UnboundedConsumption.md", "is_incumbent": true, "is_rollup_candidate": false, "rolled_into": null},
    {"entry_id": "NEW-PMP", "canonical_name": "Persistent Memory Poisoning", "source_file": "persistent-memory-poisoning.md", "is_incumbent": false, "is_rollup_candidate": false, "rolled_into": null},
    {"entry_id": "NEW-MTIE", "canonical_name": "MCP Tool Interface Exploitation", "source_file": "mcp-tool-interface-exploitation.md", "is_incumbent": false, "is_rollup_candidate": false, "rolled_into": null},
    {"entry_id": "NEW-MA", "canonical_name": "Model Misalignment", "source_file": "model-misalignment.md", "is_incumbent": false, "is_rollup_candidate": false, "rolled_into": null},
    {"entry_id": "NEW-ITSCD", "canonical_name": "Inference-Time Side-Channel Disclosure", "source_file": "inference-time-side-channel-disclosure.md", "is_incumbent": false, "is_rollup_candidate": false, "rolled_into": null},
    {"entry_id": "NEW-WLA", "canonical_name": "Weaponized LLM Abuse", "source_file": "weaponized-llm-abuse.md", "is_incumbent": false, "is_rollup_candidate": false, "rolled_into": null},
    {"entry_id": "NEW-MSDA", "canonical_name": "Model Scheming and Deceptive Alignment", "source_file": "model-scheming-and-deceptive-alignment.md", "is_incumbent": false, "is_rollup_candidate": false, "rolled_into": null},
    {"entry_id": "ROLL-CMSB", "canonical_name": "Cross-Modal Safety Bypass", "source_file": "cross-modal-safety-bypass.md", "is_incumbent": false, "is_rollup_candidate": true, "rolled_into": "LLM01"},
    {"entry_id": "ROLL-LAPTF", "canonical_name": "LLM Artifact Promotion Trust Failure", "source_file": "llm-artifact-promotion-trust-failure.md", "is_incumbent": false, "is_rollup_candidate": true, "rolled_into": "LLM03"},
    {"entry_id": "ROLL-SICG", "canonical_name": "Systemic Insecure Code Generation", "source_file": "systemic-insecure-code-generation.md", "is_incumbent": false, "is_rollup_candidate": true, "rolled_into": "LLM05"},
    {"entry_id": "ROLL-CFAS", "canonical_name": "Compositional Fine-tuning Alignment Subversion", "source_file": "compositional-finetuning-alignment-subversion.md", "is_incumbent": false, "is_rollup_candidate": true, "rolled_into": "LLM04"}
  ]
}
```

**Entry ID assignment (Premortem4 R8):** The complete 20-entry ID mapping above is LOCKED. All IDs are deterministic — derived from filenames with prefixes `NEW-` (standalone candidates) and `ROLL-` (rolled-up candidates). Do not deviate from this mapping. The rubric, adjudication log, and all downstream artifacts reference these exact IDs.

**Missing-file fallback (Premortem F1.5):** If a finalist candidate does not have a matching file in `new_entry_candidates/`, check whether the entry was described in the OWASP project's discussion threads, meeting notes, or the polling archive. If no source definition exists, write a minimal definition file (`<entry-id>.md`) from the entry's name, HANDOFF §2 description, and any available community context. Record this provenance gap in `taxonomy_provenance.json` under a `"synthesized_entries"` key so downstream consumers know the definition is derived, not primary-source.

- [ ] **Step 5: Create taxonomy_provenance.json**

Create `projects/owasp-llm/cycles/2026/taxonomy/taxonomy_provenance.json`:

```json
{
  "source_repo": "https://github.com/GenAI-Security-Project/GenAI-LLM-Top10",
  "source_commit_sha": "$SOURCE_SHA",
  "vendored_at": "2026-05-20",
  "vendored_by": "Rock Lambros",
  "incumbent_source_dir": "2026/",
  "candidate_source_dir": "2026/new_entry_candidates/",
  "vendored_content_hash": "<computed in Step 6b>",
  "synthesized_entries": []
}
```

Populate `source_commit_sha` from the actual HEAD of the source repo. If any entries were synthesized per the missing-file fallback above, list their entry_ids in `synthesized_entries`.

- [ ] **Step 6: Verify taxonomy.json parses**

```bash
uv run python -c "import json; d = json.load(open('projects/owasp-llm/cycles/2026/taxonomy/taxonomy.json')); print(f'{len(d[\"entries\"])} entries'); assert len(d['entries']) == 20"
```

Expected: `20 entries`

- [ ] **Step 6b: Compute vendored content hash (Premortem4 R4)**

Compute a SHA-256 hash of all vendored taxonomy files and update `taxonomy_provenance.json`:

```bash
CONTENT_HASH=$(find projects/owasp-llm/cycles/2026/taxonomy/ \( -name "*.md" -o -name "*.json" \) | sort | xargs cat | sha256sum | cut -d' ' -f1)
echo "Vendored content hash: $CONTENT_HASH"
# Update taxonomy_provenance.json with the computed hash
uv run python -c "
import json; p='projects/owasp-llm/cycles/2026/taxonomy/taxonomy_provenance.json'
d=json.loads(open(p).read()); d['vendored_content_hash']='$CONTENT_HASH'
open(p,'w').write(json.dumps(d,indent=2)+'\n')
"
```

This creates a verifiable binding between the provenance record and the actual vendored files, closing the integrity gap identified in Premortem4 F3.1.

- [ ] **Step 6c: Verify rollup mappings against source definitions (Premortem4 R6)**

For each rolled-up candidate in taxonomy.json, verify the source definition mentions the parent entry:

```bash
# Cross-reference rollup mappings with source file content
for entry in cross-modal-safety-bypass:LLM01 llm-artifact-promotion-trust-failure:LLM03 systemic-insecure-code-generation:LLM05 compositional-finetuning-alignment-subversion:LLM04; do
  FILE=$(echo "$entry" | cut -d: -f1)
  PARENT=$(echo "$entry" | cut -d: -f2)
  if grep -qi "$PARENT" "projects/owasp-llm/cycles/2026/taxonomy/${FILE}.md"; then
    echo "✓ ${FILE} references ${PARENT}"
  else
    echo "⚠ ${FILE} does NOT reference ${PARENT} — verify rollup mapping with Rock"
  fi
done
```

Flag any unverified rollup mappings for Rock's review before proceeding.

- [ ] **Step 7: Commit**

```bash
git add projects/owasp-llm/cycles/2026/taxonomy/
git commit -m "chore(taxonomy): vendor 20 entry definitions + provenance for 2026 LLM cycle (Plan 3)"
```

---

### Task 12: Draft All 20 Rubric Entries

**Files:**
- Create: `projects/owasp-llm/cycles/2026/prereg/rubric.json`

**VOTE-BLINDNESS CONSTRAINT (Premortem4 R2):** Do NOT read, open, or reference any files under `2026/polling/` in the source repo (`https://github.com/GenAI-Security-Project/GenAI-LLM-Top10`). Do NOT access vote results, Google Forms responses, or the HANDOFF §2 rank ordinals. This task reads ONLY vendored entry definition files from `projects/owasp-llm/cycles/2026/taxonomy/`. The drafter reads each entry definition and drafts the rubric entry from the definition content alone. Violation invalidates the cycle.

**DOMAIN EXPERTISE REQUIRED (Premortem F4.1):** This task cannot be delegated to a generic subagent. It requires understanding of LLM security attack classes to write meaningful boundary rules. The drafter should read ALL 20 definitions before writing ANY boundary rules, to understand the full taxonomy space. If uncertain about a boundary, mark `is_ambiguous=True` — Rock adjudicates in Task 14.

- [ ] **Step 1: Read ALL 20 entry definitions first**

Read every vendored definition file before drafting any entry. Build a mental map of the taxonomy space. Note overlapping concepts.

- [ ] **Step 2: Identify adjacency pairs using the known-pairs map**

> **ENTRY ID COUPLING NOTE (Premortem2 R9):** The entry IDs used below (NEW-PMP, NEW-MTIE, ROLL-CMSB, etc.) are PROVISIONAL. After Task 11 assigns actual IDs in taxonomy.json, update all references in this step to match. A mismatch will cause `validate_boundary_rules()` to fail at Step 5, which is the intended safety net.

Not all 20×19/2 = 190 entry pairs are adjacent. Boundary rules are needed ONLY for pairs where genuine classification confusion could arise. Use this known-pairs map as the starting set (derived from HANDOFF §3 corpus findings and the taxonomy structure):

**Known high-priority adjacency pairs (from corpus contamination and taxonomy overlap):**
- LLM01 (Prompt Injection) ↔ LLM02 (Sensitive Info Disclosure) — injection that extracts data
- LLM01 (Prompt Injection) ↔ LLM06 (Excessive Agency) — injection enabling unauthorized actions
- LLM01 (Prompt Injection) ↔ LLM05 (Improper Output Handling) — injection via output processing
- LLM03 (Supply Chain) ↔ LLM04 (Data and Model Poisoning) — poisoned dependencies vs poisoned data
- LLM03 (Supply Chain) ↔ LLM05 (Improper Output Handling) — supply chain leading to output issues
- LLM04 (Data and Model Poisoning) ↔ LLM09 (Misinformation) — poisoning causing misinformation
- LLM05 (Improper Output Handling) ↔ LLM09 (Misinformation) — output errors vs systematic misinformation

**Known rollup adjacency pairs (parent ↔ rolled-up child):**
- LLM01 ↔ ROLL-CMSB (Cross-Modal Safety Bypass)
- LLM03 ↔ ROLL-LAPTF (LLM Artifact Promotion Trust Failure)
- LLM05 ↔ ROLL-SICG (Systemic Insecure Code Generation)
- LLM04 ↔ ROLL-CFAS (Compositional Fine-tuning Alignment Subversion)

**New candidate adjacency pairs (identify during Step 1 reading):**
- Each new standalone candidate should have at least one boundary rule against the incumbent it's most confusable with.
- NEW-PMP (Persistent Memory Poisoning) ↔ LLM04 (Data and Model Poisoning) is likely.
- NEW-MTIE (MCP Tool Interface Exploitation) ↔ LLM06 (Excessive Agency) is likely.

**Frame-blind entries (from `_PROVISIONAL_2025_ENTRIES` in `genai_agentic.py:38-55`):**
- LLM04, LLM08, LLM10 are marked `frame_blind=True` — the corpus cannot observe them well. Their `negative_indicators` should note that low corpus count is a measurement limitation, not evidence of low prevalence.

**Proportional guidance (Premortem F5.2):** Invest the most drafting effort in boundary rules for high-count entries where boundary confusion is empirically measurable: LLM05 (3,118 corpus incidents), LLM09 (1,929), LLM03 (1,928), LLM01 (366). For entries with <50 corpus incidents (LLM10: 45, LLM08: 25), boundary rules matter less for the measurement model but are still required for rubric completeness.

> **CORPUS STATISTICS DISCLOSURE (Premortem2 F1.4):** The counts above are derived from the vendored corpus (Plan 2). Reading this plan exposes the drafter to corpus-derived frequency information, which subtly informs effort allocation. This is methodologically defensible (knowing where confusion is empirically common improves boundary-rule quality) but constitutes corpus-informed drafting. If the drafter reads ONLY the entry definitions and NOT this proportional guidance, `rubric_attestation.json` may record `viewed_corpus_before_drafting=false`. If the drafter reads this section, the attestation should record `viewed_corpus_before_drafting=true` with `viewed_corpus_details="corpus frequency statistics from Plan 3 proportional guidance"`.

- [ ] **Step 3: Draft each entry using this field-derivation template**

For each of the 20 entries, produce a `RubricEntry`. Derive fields from the markdown as follows:

| Field | Derivation |
|---|---|
| `entry_id` | From `taxonomy.json` |
| `canonical_name` | From `taxonomy.json` |
| `in_scope` | First paragraph after `## Description` or equivalent heading. 1-3 sentences. |
| `exclusions` | Items from `## Not In Scope` heading if present, OR derived by negating adjacent entries' in_scope. Every non-rollup entry should have at least one exclusion. |
| `boundary_rules` | One `BoundaryRule` per adjacency pair from Step 2. Rule text: "If [distinguishing condition], classify as [this entry / adjacent entry]." Mark `is_ambiguous=True` only for genuine 50/50 calls. |
| `positive_indicators` | Keywords, CVE types, attack patterns from `## Examples`, `## Indicators`, or the body text. |
| `negative_indicators` | Inverse of adjacent entries' positive_indicators, plus explicit non-indicators from the definition. |
| `co_occurrence_pairs` | Entry pairs where a single incident could legitimately carry both labels (e.g., `("LLM01", "LLM06")` for injection-enabled excessive agency). |

- [ ] **Step 4: Write rubric.json using the I/O module**

```python
from engine.prereg.rubric import Rubric, RubricEntry, BoundaryRule
from engine.prereg.rubric_io import write_rubric
from pathlib import Path

# Construct all 20 RubricEntry instances from the drafting work in Steps 1-3.
# Each entry must have all fields populated per the template above.
# Example for one entry:
llm01 = RubricEntry(
    entry_id="LLM01",
    canonical_name="Prompt Injection",
    in_scope="Attacks that manipulate LLM behavior via crafted input prompts, including direct injection, indirect injection via retrieved context, and multi-turn manipulation.",
    exclusions=(
        "Data exfiltration where the primary mechanism is not prompt manipulation (→ LLM02)",
        "Social engineering of human operators without LLM involvement",
    ),
    boundary_rules=(
        BoundaryRule(
            adjacent_entry_id="LLM02",
            rule="If the attack's primary goal is data extraction and it uses prompt manipulation as the vector, classify as LLM02. If the primary mechanism is prompt manipulation with data extraction as a side effect, classify as LLM01.",
            is_ambiguous=False,
        ),
        BoundaryRule(
            adjacent_entry_id="LLM06",
            rule="If prompt injection causes the LLM to take unauthorized actions, classify as LLM01 (attack vector) not LLM06 (consequence). Co-label if both are independently present.",
            is_ambiguous=False,
        ),
        # ... additional boundary rules per Step 2 adjacency map
    ),
    positive_indicators=("prompt injection", "instruction override", "jailbreak", "system prompt bypass", "indirect prompt injection"),
    negative_indicators=("data exfiltration only", "no prompt manipulation involved", "hardware side-channel"),
    co_occurrence_pairs=(("LLM01", "LLM06"),),
    is_rollup_candidate=False,
    rolled_into=None,
)

# ... construct remaining 19 entries following the same pattern

entries = (llm01, ...)  # all 20

rubric = Rubric(cycle_id="2026", version="1.0.0", entries=entries)
write_rubric(rubric, Path("projects/owasp-llm/cycles/2026/prereg/rubric.json"))
```

- [ ] **Step 5: Validate the rubric**

```bash
uv run python -m engine.cli.main validate-rubric \
  --rubric projects/owasp-llm/cycles/2026/prereg/rubric.json \
  --taxonomy projects/owasp-llm/cycles/2026/taxonomy/taxonomy.json
```

Expected: "Rubric is valid." with rubric hash printed. The `--taxonomy` flag reads expected entry IDs directly from taxonomy.json — no manual comma-separated list needed.

(Exact entry IDs will be finalized during Task 11 taxonomy vendoring.)

- [ ] **Step 6: Commit draft rubric**

```bash
git add projects/owasp-llm/cycles/2026/prereg/rubric.json
git commit -m "feat(prereg): draft rubric — 20 entries, vote-blind (Plan 3)"
```

---

### Task 13: Populate Rubric Attestation

**Files:**
- Create: `projects/owasp-llm/cycles/2026/prereg/rubric_attestation.json`

- [ ] **Step 1: Determine corpus-viewing status**

If the drafter viewed ANY corpus samples during Task 12 (e.g., to understand contamination patterns, inspect bare-LLM03 records, or calibrate boundary rules against real data):
- Set `viewed_corpus_before_drafting` to `true`
- List which samples were viewed in `viewed_corpus_details`

If the drafter worked only from entry definitions:
- Set `viewed_corpus_before_drafting` to `false`
- Set `viewed_corpus_details` to `""`

- [ ] **Step 2: Write the attestation file**

```python
from engine.prereg.rubric_attestation import RubricDraftingAttestation
from engine.prereg.rubric_io import write_rubric_attestation
from pathlib import Path

attestation = RubricDraftingAttestation(
    viewed_corpus_before_drafting=False,  # or True if samples were viewed
    viewed_corpus_details="",  # or description of what was viewed
    viewed_vote_data_before_drafting=False,  # True if any vote results were seen
    viewed_vote_data_details="",  # describe what vote data was seen, if any
)
write_rubric_attestation(
    attestation,
    Path("projects/owasp-llm/cycles/2026/prereg/rubric_attestation.json"),
)
```

- [ ] **Step 3: Commit**

```bash
git add projects/owasp-llm/cycles/2026/prereg/rubric_attestation.json
git commit -m "feat(prereg): rubric drafting attestation — viewed_corpus=false (Plan 3)"
```

---

### Task 14: Rock Adjudication of Boundary Cells (HUMAN)

**Files:**
- Create: `projects/owasp-llm/cycles/2026/prereg/adjudication_log.json`

This task requires **Rock's manual review**. Claude cannot adjudicate boundary cells — that is Rock's methodological responsibility per HANDOFF §5.2.

- [ ] **Step 1: Present boundary cells to Rock**

List all boundary rule pairs from the rubric. For each pair, show:
- The two entries involved
- The proposed boundary rule
- Whether the rule is marked ambiguous

Rock reviews each and either confirms or marks as `ambiguous-both-labels`.

- [ ] **Step 2: Record adjudication decisions**

```python
from engine.prereg.adjudication import AdjudicationEntry, AdjudicationLog
from engine.prereg.rubric_io import read_rubric, write_adjudication_log
from pathlib import Path

rubric = read_rubric(Path("projects/owasp-llm/cycles/2026/prereg/rubric.json"))

# Rock's decisions (populated during review)
entries = (
    AdjudicationEntry(
        entry_id_a="LLM01",
        entry_id_b="LLM02",
        decision="resolved:LLM01",
        rationale="Prompt injection attacks that extract data classify as LLM01 if the primary mechanism is prompt manipulation.",
        adjudicator="Rock Lambros",
        date="2026-05-20",
    ),
    # ... one entry per boundary pair
)

log = AdjudicationLog(rubric_hash=rubric.compute_hash(), entries=entries)
log.validate_coverage(rubric)  # must not raise
write_adjudication_log(
    log,
    Path("projects/owasp-llm/cycles/2026/prereg/adjudication_log.json"),
)
```

- [ ] **Step 3: Commit**

```bash
git add projects/owasp-llm/cycles/2026/prereg/adjudication_log.json
git commit -m "feat(prereg): Rock's boundary-cell adjudication log (Plan 3)"
```

---

## Phase C: Freeze (BLOCKED on external reviewer)

Phase C requires an external rubric reviewer identified in `docs/REVIEWERS.md`. The current state is INTERIM (Rock = reviewer = ranking author). These tasks execute ONLY after the blocker clears.

**DECISION DEADLINE (Premortem R4):** If no independent rubric reviewer is identified by **2026-06-15**, Rock decides between:

- **(a) Publish as non_publishable** with "single-author rubric, uncontrolled" caveat per HANDOFF §4. The validation proceeds as a methodology demonstration. Plans 4/5 may start for internal use but cannot produce a publishable report.
- **(b) Pause the project** until a reviewer is identified.

This decision is recorded in `docs/METHODOLOGY-CHANGELOG.md`. The 4-week deadline (from Plan 3 completion) provides time for reviewer recruitment before Plans 4/5 would need to start for the 2026 cycle timeline. HANDOFF §4: "if no candidates are identified, Plan 4 and Plan 5 cannot start."

---

### Task 15: External Reviewer Signoff (BLOCKED)

**Files:**
- Create: `docs/REVIEWERS/<reviewer-name-slug>-rubric.txt`
- Modify: `docs/REVIEWERS.md`

**BLOCKER:** An independent OWASP working-group member who is not Rock must agree to review the rubric. This is a human-side action that engineering cannot unblock.

- [ ] **Step 1: Reviewer writes attestation**

The reviewer creates `docs/REVIEWERS/<reviewer-name-slug>-rubric.txt` containing:
- Their name and affiliation
- What they reviewed (the rubric at a specific hash)
- Date of review
- Whether they viewed any analysis results before signing (`viewed_results_before_signoff`)
- Statement of independence

- [ ] **Step 2: Compute SHA-256 and update REVIEWERS.md**

```bash
shasum -a 256 docs/REVIEWERS/<reviewer-name-slug>-rubric.txt
```

Update `docs/REVIEWERS.md` rubric reviewer section per the "Path to publishable" template (PRD §10.5).

- [ ] **Step 3: Commit atomically**

```bash
git add docs/REVIEWERS/<reviewer-name-slug>-rubric.txt docs/REVIEWERS.md
git commit -m "docs(reviewers): identify rubric reviewer <name> + attestation hash"
```

---

### Task 16: Hash-Lock Rubric in Manifest

**Files:**
- Create: `projects/owasp-llm/cycles/2026/prereg/manifest.json`
- Create: `projects/owasp-llm/cycles/2026/prereg/manifest.lock`

- [ ] **Step 1: Populate manifest with rubric hash**

```python
import hashlib
import json

from engine.prereg.manifest import PreregManifest
from engine.prereg.rubric_io import read_rubric, read_rubric_attestation
from engine.prereg.lock import write_lock
from pathlib import Path

rubric = read_rubric(Path("projects/owasp-llm/cycles/2026/prereg/rubric.json"))
attestation = read_rubric_attestation(
    Path("projects/owasp-llm/cycles/2026/prereg/rubric_attestation.json")
)

# Build manifest with rubric hash + reviewer info
manifest = PreregManifest(
    engine_version="0.3.0",
    engine_version_range_min="0.3.0",
    engine_version_range_max="0.3.0",
    cycle_id="2026",
    taxonomy_hash=hashlib.sha256(
        Path("projects/owasp-llm/cycles/2026/taxonomy/taxonomy.json").read_bytes()
    ).hexdigest(),
    snapshot_hash=json.loads(
        Path("projects/owasp-llm/cycles/2026/corpora/genai_agentic/"
             "24806f1a4f0917f85f7509d6cb2a34b12e56eb902714b37bc2b03a2cf1a246bb/"
             "provenance.json").read_text()
    )["snapshot_hash"],
    primary_spec="negative_binomial_per_stratum",
    robustness_specs=("poisson_flat",),
    flag_threshold_tau=0.8,  # pre-registered before decide (HANDOFF §9 item 4)
    statistic="weighted_cohens_kappa",
    measurability_minimum=4,
    prior_scale=0.5,
    concentration_shape=5.0,
    concentration_rate=0.1,
    ess_fraction=0.4,
    meaningful_kappa_n=4,
    prng_seed=20260520,
    rubric_drafting_attestation=attestation,
    rubric_reviewer=None,  # populated from Task 15 signoff
    statistical_reviewer=None,  # populated in Plan 5
    classifier_rule_hash=None,  # populated in Plan 5
    rubric_hash=rubric.compute_hash(),
    post_hoc_register_path=None,
)
```

- [ ] **Step 2: Verify rubric hash matches file (Premortem R2)**

```python
from engine.prereg.gates import require_rubric_hash_match

require_rubric_hash_match(
    manifest, Path("projects/owasp-llm/cycles/2026/prereg/rubric.json")
)
```

This gate closes the integrity gap: it verifies that `manifest.rubric_hash` was computed from the current rubric.json file contents, not from a stale or modified version.

- [ ] **Step 2b: Check for upstream source repo changes (Premortem4 R9)**

Before locking, verify the source repo hasn't advanced since vendoring:

```bash
VENDORED_SHA=$(uv run python -c "import json; print(json.load(open('projects/owasp-llm/cycles/2026/taxonomy/taxonomy_provenance.json'))['source_commit_sha'])")
CURRENT_SHA=$(git ls-remote https://github.com/GenAI-Security-Project/GenAI-LLM-Top10 HEAD | cut -f1)
if [ "$VENDORED_SHA" != "$CURRENT_SHA" ]; then
  echo "WARNING: source repo has advanced since vendoring ($VENDORED_SHA → $CURRENT_SHA)"
  echo "Review upstream changes before locking. Proceed only if changes do not affect entry definitions."
else
  echo "Source repo unchanged since vendoring."
fi
```

This is informational, not blocking. If the source repo has advanced, review the changes and decide whether to re-vendor or proceed.

- [ ] **Step 3: Write and lock manifest**

```python
import json

lock_path = Path("projects/owasp-llm/cycles/2026/prereg/manifest.lock")
manifest_path = Path("projects/owasp-llm/cycles/2026/prereg/manifest.json")
manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2) + "\n")
write_lock(manifest, lock_path)
```

- [ ] **Step 4: Run freeze-rubric CLI**

```bash
uv run python -m engine.cli.main freeze-rubric \
  --rubric projects/owasp-llm/cycles/2026/prereg/rubric.json \
  --cycle-dir projects/owasp-llm/cycles/2026
```

- [ ] **Step 5: Commit**

```bash
git add projects/owasp-llm/cycles/2026/prereg/manifest.json \
      projects/owasp-llm/cycles/2026/prereg/manifest.lock
git commit -m "feat(prereg): hash-lock rubric in prereg manifest (Plan 3)"
```

---

### Task 17: Final Tag v0.3.0-plan3

- [ ] **Step 1: Run full test suite**

```bash
uv run pytest -v
```

Expected: All pass. New Plan 3 tests should bring the total above 440.

- [ ] **Step 2: Verify CI green**

Push the branch, verify GitHub Actions green on both ubuntu and macOS.

- [ ] **Step 3: Merge and tag**

```bash
# Merge PR (or merge via GitHub)
git tag v0.3.0-plan3
git push origin v0.3.0-plan3
```

---

## Coverage Matrix

### PRD §4.6 Acceptance Criteria

| Criterion | Task | Verification |
|---|---|---|
| 1. rubric.json contains all 20 entries (16 ranked + 4 rolled-up) | Task 12 | `validate-rubric --taxonomy` |
| 2. Every entry has all 8 required fields | Task 12 | `validate_completeness()` (checks in_scope, positive/negative_indicators, boundary_rules for non-rollup) |
| 3. Every pairwise-adjacent entry pair has explicit boundary rule | Task 12 | `validate_boundary_rules()` |
| 4. adjudication_log covers every boundary cell | Task 14 | `validate_coverage()` |
| 5. rubric_attestation.json viewed_corpus_samples is set | Task 13 | `read_rubric_attestation()` non-None |
| 6. engine/prereg/lock.py accepts rubric, emits stable hash | Task 2, 3 | `test_hash_deterministic` + lock roundtrip |
| 7. CLI classify refuses without rubric attestation | Task 6 | `test_require_rubric_attestation_fails_when_none` |
| 8. non_publishable=True when reviewer missing or reviewer=author | Task 6 | `test_is_publishable_false_*` + `test_is_publishable_name_normalization` |
| 9. Tag v0.3.0-plan3 on freeze commit | Task 17 | Git tag present |
| 10. rubric_hash in manifest matches actual rubric file | Task 6, 16 | `require_rubric_hash_match` gate + `test_require_rubric_hash_match_*` |
| 11. co_occurrence_pairs reference valid entry IDs | Task 2, 7 | `validate_co_occurrences()` + `test_validate_co_occurrences_*` |
| 12. End-to-end freeze workflow hash chain | Task 8 | `TestFreezeWorkflowIntegration` |
| 13. freeze-rubric verifies files committed to git | Task 7 | `verify_committed()` calls in freeze-rubric CLI |
| 14. freeze-rubric runs completeness check against taxonomy | Task 7 | `validate_completeness()` call in freeze-rubric CLI |
| 15. Rock adjudicates all boundary cells and signs off on rubric quality (Premortem4 R10) | Task 14 | `adjudication_log.json` committed + Rock confirmation |
| 16. Taxonomy provenance includes vendored content hash (Premortem4 R4) | Task 11 | `taxonomy_provenance.json` contains `vendored_content_hash` |
| 17. No duplicate adjudications for same pair (Premortem4 R7) | Task 4 | `validate_coverage()` duplicate detection |

### HANDOFF §6 Control Compliance

| Control | How Plan 3 addresses it |
|---|---|
| §6.1 Pre-reg hash-locked | `rubric_hash` in manifest, locked via `write_lock()`, verified at classify via `require_rubric_hash_match()` |
| §6.2 Vote-blindness | Procedural: documented in RUBRIC-WORKFLOW.md, rank ordinals stripped from plan (Premortem R1), enforced by drafter discipline, attestation records vote-data exposure (Premortem3 R3) |
| §6.5 Independent reviewer | `is_publishable()` gate with normalized name comparison, REVIEWERS.md workflow, decision deadline 2026-06-15 |
| §6.11(b) Classifier rules hash-locked before gold-set | rubric frozen before classify (gate enforced) |
| §6.11(d) Rubric drafting attestation | `rubric_attestation.json` populated (corpus + vote-data fields per Premortem3 R3) |
| §6.11(e) Reviewer signoff timing | `ReviewerSignoff.verify()` checks git timestamp |

### Premortem Remediation Traceability

| Remediation | Finding | Where applied |
|---|---|---|
| R1: Strip vote ranks | F1.1 CRITICAL | Lines 70-82 entry list |
| R2: `require_rubric_hash_match` gate | F2.2, F2.3 HIGH | Task 6 (gates.py) + Task 16 Step 2 |
| R3: `--taxonomy` CLI flag | F1.3 HIGH | Task 7 (CLI) + Task 12 Step 5 |
| R4: Phase C deadline | F5.1 CRITICAL | Phase C header |
| R5: Task 12 drafting guidance | F4.1 HIGH | Task 12 Steps 1-3 |
| R6: boundary_rules non-empty check | F1.4 MEDIUM | Task 1 (validate_completeness) |
| R7: Name normalization | F3.2 MEDIUM | Task 6 (is_publishable) |
| R8: Taxonomy provenance | F3.1 MEDIUM | Task 11 Step 5 |
| R9: Fix type: ignore | F2.5 LOW | Task 4 (validate_coverage) |
| R10: Rubric amendments section | F5.3 MEDIUM | Task 9 (RUBRIC-WORKFLOW.md) |
| R11: Remove private cross-module import | F2.4 MEDIUM | Task 5 (rubric_io.py) |
| F1.5: Missing-file fallback | F1.5 MEDIUM | Task 11 header |
| F3.4: Decision format validation | F3.4 LOW | Task 4 (AdjudicationEntry.__post_init__) |

### Second Premortem Remediation Traceability

| Remediation | Finding | Where applied |
|---|---|---|
| P2-R1: ~~Remove phantom manifest fields~~ **REVERTED by P3-R1** | F4.4 HIGH | ~~Task 6 + Task 16~~ — fields exist in manifest.py:33-34, removal was iatrogenic |
| P2-R2: `validate_co_occurrences()` | F1.3 HIGH | Task 1 (Rubric class) + Task 2 (test) + Task 7 (CLI) |
| P2-R3: End-to-end integration test | F5.3 HIGH | Task 8 (TestFreezeWorkflowIntegration) |
| P2-R4: Relax boundary_rules check | F2.1 HIGH | Task 1 (validate_completeness no_adjacency_attested param) + Task 2 (test) |
| P2-R5: freeze-rubric completeness check | F2.4 MEDIUM | Task 7 (freeze-rubric reads taxonomy.json) |
| P2-R6: Resolved entry_id validation | F2.5 MEDIUM | Task 4 (AdjudicationEntry.__post_init__) + test |
| P2-R7: `verify_committed()` in freeze-rubric | F3.1 MEDIUM | Task 7 (freeze-rubric CLI) |
| P2-R8: Fix duplicate Step 4 | F4.1 MEDIUM | Task 12 (renumbered to Step 6) |
| P2-R9: Entry ID coupling note | F1.2 MEDIUM | Task 12 Step 2 header |
| P2-R10: Amendments re-review | F5.1 MEDIUM | Task 9 (RUBRIC-WORKFLOW.md amendments section) |
| P2-R11: Tag after merge | F4.3 MEDIUM | Task 10 Step 7 |
| P2-F1.4: Corpus statistics disclosure | F1.4 MEDIUM | Task 12 Step 2 proportional guidance note |

### Third Premortem Remediation Traceability

| Remediation | Finding | Where applied |
|---|---|---|
| P3-R1: Restore `engine_version_range_min/max` | F1.1 CRITICAL (iatrogenic from P2-R1) | Task 6 (`_make_test_manifest`) + Task 16 (manifest construction) |
| P3-R2: Fix freeze-rubric test git fixture | F2.1 HIGH | Task 7 (`test_freeze_rubric_missing_attestation`) |
| P3-R3: Vote-data attestation fields | F3.7 HIGH | Task 3 (`rubric_attestation.py`), Task 5 (`rubric_io.py`), Task 8/13 (constructions) |
| P3-R4: Wire `--no-adjacency-attested` through CLI | F4.7 HIGH | Task 7 (`validate-rubric` + `freeze-rubric` commands) |
| P3-R5: `co_occurrence_pair` length validation | F1.7 MEDIUM | Task 5 (`rubric_io.py` `_validate_pair()`) |
| P3-R6: Clean-tree check for taxonomy vendoring | F3.3 MEDIUM | Task 11 Step 1 |

### Phase Gate Summary

| Phase | Status | Merge tag | Depends on |
|---|---|---|---|
| A: Scaffolding | DONE (merged) | v0.3.0-rc1 | Plan 1, Plan 2 |
| B: Drafting | Runnable now (vote-blind) | (same branch) | Phase A + entry definitions |
| C: Freeze | BLOCKED (deadline: 2026-06-15) | v0.3.0-plan3 | Phase B + external reviewer |

### Fourth Premortem Remediation Traceability

| Remediation | Finding | Where applied |
|---|---|---|
| P4-R1: Replace local paths with GitHub repo URLs | F2.6 CRITICAL + F4.3 HIGH + F5.4 HIGH | Plan Tasks 11-14, RUBRIC-WORKFLOW.md, inherited constraint #2/#9 |
| P4-R2: Vote-blindness warning in task body text | F1.4 HIGH | Task 11 + Task 12 headers |
| P4-R3: `no_adjacency_attested_entry_ids` in attestation | F2.4 HIGH | Task 3 Step 5 (`RubricDraftingAttestation` field) |
| P4-R4: Vendored content hash in provenance | F3.1 HIGH + F1.5 MEDIUM | Task 11 Step 6b |
| P4-R5: Rollup sub-test consumption spec | F5.1 HIGH | RUBRIC-WORKFLOW.md new section |
| P4-R6: Rollup mapping verification against source | F1.2 HIGH | Task 11 Step 6c |
| P4-R7: Duplicate adjudication detection | F2.3 MEDIUM | Task 4 `validate_coverage()` code |
| P4-R8: Deterministic entry ID assignment | F1.3 MEDIUM | Task 11 Step 4 (full 20-entry taxonomy.json) |
| P4-R9: Upstream source repo change check before freeze | F5.3 MEDIUM | Task 16 Step 2b |
| P4-R10: Quality acceptance criterion | F5.2 MEDIUM | Coverage matrix criterion 15 |
| P4-R11: Adapter incompatibility note | F3.5 MEDIUM | RUBRIC-WORKFLOW.md new section |
