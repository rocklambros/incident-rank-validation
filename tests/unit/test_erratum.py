"""Unit tests for engine.erratum — lineage, Merkle chain, post-hoc register (M16)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.erratum.lineage import CrossCycleComparisonError, verify_lineage
from engine.erratum.merkle import GENESIS, chain_link, verify_chain
from engine.erratum.models import Erratum
from engine.erratum.post_hoc import (
    PostHocAnalysis,
    PostHocRegister,
    append_analysis,
    read_and_verify_register,
    write_register,
)

# ---------------------------------------------------------------------------
# Lineage tests
# ---------------------------------------------------------------------------


class TestVerifyLineage:
    def test_same_ids_and_hashes_passes(self) -> None:
        """Identical cycle IDs and taxonomy hashes should not raise."""
        verify_lineage("cycle-1", "cycle-1", "abc123", "abc123")

    def test_different_cycle_ids_raises(self) -> None:
        """Different cycle IDs must raise CrossCycleComparisonError."""
        with pytest.raises(CrossCycleComparisonError, match="different IDs"):
            verify_lineage("cycle-1", "cycle-2", "abc123", "abc123")

    def test_different_taxonomy_hashes_raises(self) -> None:
        """Different taxonomy hashes must raise CrossCycleComparisonError."""
        with pytest.raises(CrossCycleComparisonError, match="different taxonomy hashes"):
            verify_lineage("cycle-1", "cycle-1", "abc123", "def456")


# ---------------------------------------------------------------------------
# Merkle chain tests
# ---------------------------------------------------------------------------


class TestMerkleChain:
    def test_chain_link_deterministic(self) -> None:
        """Same prev + payload should produce the same hash."""
        h1 = chain_link(GENESIS, {"foo": "bar"})
        h2 = chain_link(GENESIS, {"foo": "bar"})
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_chain_link_changes_with_payload(self) -> None:
        """Different payloads should produce different hashes."""
        h1 = chain_link(GENESIS, {"foo": "bar"})
        h2 = chain_link(GENESIS, {"foo": "baz"})
        assert h1 != h2

    def test_verify_chain_valid(self) -> None:
        """A correctly constructed chain should verify without error."""
        entries: list[dict[str, object]] = []
        prev = GENESIS
        for i in range(3):
            payload: dict[str, object] = {"index": i, "data": f"entry-{i}"}
            h = chain_link(prev, payload)
            entries.append({**payload, "chain_hash": h})
            prev = h

        terminal = verify_chain(entries)
        assert terminal == prev

    def test_verify_chain_empty(self) -> None:
        """Empty chain should return GENESIS."""
        terminal = verify_chain([])
        assert terminal == GENESIS

    def test_tampering_raises(self) -> None:
        """Modifying a payload after hashing should break the chain."""
        entries: list[dict[str, object]] = []
        prev = GENESIS
        for i in range(3):
            payload: dict[str, object] = {"index": i}
            h = chain_link(prev, payload)
            entries.append({**payload, "chain_hash": h})
            prev = h

        # Tamper with entry 1
        entries[1]["index"] = 999

        with pytest.raises(ValueError, match="chain break at entry 1"):
            verify_chain(entries)

    def test_reorder_raises(self) -> None:
        """Swapping entries should break the chain."""
        entries: list[dict[str, object]] = []
        prev = GENESIS
        for i in range(3):
            payload: dict[str, object] = {"index": i}
            h = chain_link(prev, payload)
            entries.append({**payload, "chain_hash": h})
            prev = h

        entries[0], entries[1] = entries[1], entries[0]

        with pytest.raises(ValueError, match="chain break"):
            verify_chain(entries)


# ---------------------------------------------------------------------------
# Post-hoc register tests
# ---------------------------------------------------------------------------


def _sample_analysis(n: int = 0) -> PostHocAnalysis:
    return PostHocAnalysis(
        cycle_id="cycle-1",
        title=f"Analysis {n}",
        description=f"Description of analysis {n}",
        rationale=f"Rationale for analysis {n}",
        added_at=f"2026-01-{15 + n:02d}T00:00:00Z",
        artifacts=(f"artifact_{n}.json",),
    )


class TestPostHocRegister:
    def test_write_and_read_roundtrip(self, tmp_path: Path) -> None:
        """Write a register, read it back, verify contents match."""
        a0 = _sample_analysis(0)
        a1 = _sample_analysis(1)

        reg = PostHocRegister(cycle_id="cycle-1", analyses=(a0, a1), chain_terminal_hash="")
        # Recompute proper terminal hash
        from engine.erratum.merkle import chain_link as cl

        prev = GENESIS
        for a in reg.analyses:
            prev = cl(prev, {
                "cycle_id": a.cycle_id,
                "title": a.title,
                "description": a.description,
                "rationale": a.rationale,
                "added_at": a.added_at,
                "artifacts": list(a.artifacts),
            })

        reg = PostHocRegister(cycle_id="cycle-1", analyses=(a0, a1), chain_terminal_hash=prev)

        p = tmp_path / "register.json"
        write_register(reg, p)
        loaded = read_and_verify_register(p)

        assert loaded.cycle_id == "cycle-1"
        assert len(loaded.analyses) == 2
        assert loaded.analyses[0].title == "Analysis 0"
        assert loaded.analyses[1].title == "Analysis 1"
        assert loaded.chain_terminal_hash == prev

    def test_tampered_file_raises(self, tmp_path: Path) -> None:
        """Tampering with the JSON file should raise on read."""
        a0 = _sample_analysis(0)
        reg = PostHocRegister(cycle_id="cycle-1", analyses=(a0,), chain_terminal_hash="")

        from engine.erratum.merkle import chain_link as cl

        prev = GENESIS
        prev = cl(prev, {
            "cycle_id": a0.cycle_id,
            "title": a0.title,
            "description": a0.description,
            "rationale": a0.rationale,
            "added_at": a0.added_at,
            "artifacts": list(a0.artifacts),
        })
        reg = PostHocRegister(cycle_id="cycle-1", analyses=(a0,), chain_terminal_hash=prev)

        p = tmp_path / "register.json"
        write_register(reg, p)

        # Tamper: change the title in the JSON
        data = json.loads(p.read_text())
        data["analyses"][0]["title"] = "TAMPERED"
        p.write_text(json.dumps(data))

        with pytest.raises(ValueError, match="chain break"):
            read_and_verify_register(p)

    def test_append_analysis(self) -> None:
        """append_analysis should add an entry and update terminal hash."""
        a0 = _sample_analysis(0)

        from engine.erratum.merkle import chain_link as cl

        prev = GENESIS
        prev = cl(prev, {
            "cycle_id": a0.cycle_id,
            "title": a0.title,
            "description": a0.description,
            "rationale": a0.rationale,
            "added_at": a0.added_at,
            "artifacts": list(a0.artifacts),
        })

        reg = PostHocRegister(cycle_id="cycle-1", analyses=(a0,), chain_terminal_hash=prev)
        a1 = _sample_analysis(1)

        new_reg = append_analysis(reg, a1)

        assert len(new_reg.analyses) == 2
        assert new_reg.analyses[1].title == "Analysis 1"
        # Terminal hash should differ from original
        assert new_reg.chain_terminal_hash != reg.chain_terminal_hash
        assert len(new_reg.chain_terminal_hash) == 64

    def test_append_preserves_chain_integrity(self, tmp_path: Path) -> None:
        """After append, writing and re-reading should still verify."""
        a0 = _sample_analysis(0)

        from engine.erratum.merkle import chain_link as cl

        prev = GENESIS
        prev = cl(prev, {
            "cycle_id": a0.cycle_id,
            "title": a0.title,
            "description": a0.description,
            "rationale": a0.rationale,
            "added_at": a0.added_at,
            "artifacts": list(a0.artifacts),
        })

        reg = PostHocRegister(cycle_id="cycle-1", analyses=(a0,), chain_terminal_hash=prev)
        a1 = _sample_analysis(1)
        reg2 = append_analysis(reg, a1)

        p = tmp_path / "register.json"
        write_register(reg2, p)
        loaded = read_and_verify_register(p)

        assert loaded.chain_terminal_hash == reg2.chain_terminal_hash
        assert len(loaded.analyses) == 2


# ---------------------------------------------------------------------------
# Erratum model tests
# ---------------------------------------------------------------------------


class TestErratumModel:
    def test_erratum_creation(self) -> None:
        """Erratum dataclass should be frozen and have expected fields."""
        e = Erratum(
            cycle_id="cycle-1",
            erratum_number=1,
            title="Fix measurability threshold",
            description="Threshold was off by one",
            impact="flips_measurability",
            issued_at="2026-01-15T00:00:00Z",
        )
        assert e.cycle_id == "cycle-1"
        assert e.impact == "flips_measurability"

        with pytest.raises(AttributeError):
            e.title = "changed"  # type: ignore[misc]
