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
