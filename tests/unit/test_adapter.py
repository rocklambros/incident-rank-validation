"""Unit tests for engine.adapters.base and engine.adapters.synthetic."""

from __future__ import annotations

import pytest

from engine.adapters.base import CorpusAdapter
from engine.adapters.synthetic import SyntheticAdapter
from engine.model.overlap import OverlapWeights
from engine.schema import BiasProfile, EntryDefinition, IncidentRecord


class TestSyntheticAdapterInterface:
    def test_is_corpus_adapter_subclass(self) -> None:
        assert issubclass(SyntheticAdapter, CorpusAdapter)

    def test_instance_is_corpus_adapter(self) -> None:
        assert isinstance(SyntheticAdapter(), CorpusAdapter)


class TestIterIncidents:
    def test_yields_incident_records(self) -> None:
        adapter = SyntheticAdapter()
        incidents = list(adapter.iter_incidents())
        assert len(incidents) > 0
        for inc in incidents:
            assert isinstance(inc, IncidentRecord)

    def test_all_have_valid_stratum(self) -> None:
        adapter = SyntheticAdapter()
        strata = {"stratum_a", "stratum_b"}
        for inc in adapter.iter_incidents():
            assert inc.corpus_stratum in strata

    def test_no_frame_blind_incidents_generated(self) -> None:
        """E06 is frame_blind; no incidents should be generated for it."""
        adapter = SyntheticAdapter()
        for inc in adapter.iter_incidents():
            assert "E06" not in inc.native_labels


class TestBiasProfiles:
    def test_returns_nonempty_tuple(self) -> None:
        adapter = SyntheticAdapter()
        profiles = adapter.bias_profiles()
        assert isinstance(profiles, tuple)
        assert len(profiles) > 0

    def test_all_are_bias_profile_instances(self) -> None:
        adapter = SyntheticAdapter()
        for profile in adapter.bias_profiles():
            assert isinstance(profile, BiasProfile)

    def test_covers_both_strata(self) -> None:
        adapter = SyntheticAdapter()
        strata = {p.stratum for p in adapter.bias_profiles()}
        assert "stratum_a" in strata
        assert "stratum_b" in strata


class TestStratumSizes:
    def test_returns_dict(self) -> None:
        adapter = SyntheticAdapter()
        sizes = adapter.stratum_sizes()
        assert isinstance(sizes, dict)

    def test_all_positive(self) -> None:
        adapter = SyntheticAdapter()
        for stratum, size in adapter.stratum_sizes().items():
            assert size > 0, f"stratum {stratum!r} has non-positive size {size}"

    def test_expected_strata_present(self) -> None:
        adapter = SyntheticAdapter()
        sizes = adapter.stratum_sizes()
        assert "stratum_a" in sizes
        assert "stratum_b" in sizes

    def test_expected_sizes(self) -> None:
        adapter = SyntheticAdapter()
        sizes = adapter.stratum_sizes()
        assert sizes["stratum_a"] == 500
        assert sizes["stratum_b"] == 300


class TestEntryDefinitions:
    def test_returns_tuple(self) -> None:
        adapter = SyntheticAdapter()
        entries = adapter.entry_definitions()
        assert isinstance(entries, tuple)

    def test_all_are_entry_definitions(self) -> None:
        adapter = SyntheticAdapter()
        for entry in adapter.entry_definitions():
            assert isinstance(entry, EntryDefinition)

    def test_includes_frame_blind_entry(self) -> None:
        adapter = SyntheticAdapter()
        frame_blind = [e for e in adapter.entry_definitions() if e.frame_blind]
        assert len(frame_blind) >= 1, "must have at least one frame_blind entry"

    def test_e06_is_frame_blind(self) -> None:
        adapter = SyntheticAdapter()
        entries_by_id = {e.entry_id: e for e in adapter.entry_definitions()}
        assert "E06" in entries_by_id
        assert entries_by_id["E06"].frame_blind is True

    def test_e01_through_e05_not_frame_blind(self) -> None:
        adapter = SyntheticAdapter()
        entries_by_id = {e.entry_id: e for e in adapter.entry_definitions()}
        for eid in ("E01", "E02", "E03", "E04", "E05"):
            assert entries_by_id[eid].frame_blind is False


class TestOverlapWeights:
    def test_returns_overlap_weights(self) -> None:
        adapter = SyntheticAdapter()
        ow = adapter.overlap_weights()
        assert isinstance(ow, OverlapWeights)

    def test_no_self_loop(self) -> None:
        adapter = SyntheticAdapter()
        ow = adapter.overlap_weights()
        for target, sources in ow.weights.items():
            assert target not in sources, f"self-loop detected at W[{target}][{target}]"

    def test_column_stochastic(self) -> None:
        adapter = SyntheticAdapter()
        ow = adapter.overlap_weights()
        all_sources: set[str] = set()
        for tgt_map in ow.weights.values():
            all_sources.update(tgt_map.keys())
        for src in all_sources:
            col_sum = sum(ow.weights.get(tgt, {}).get(src, 0.0) for tgt in ow.weights)
            assert col_sum <= 1.0 + 1e-6, f"column sum for {src!r} = {col_sum:.4f} > 1"

    def test_e02_leaks_into_e01(self) -> None:
        adapter = SyntheticAdapter()
        ow = adapter.overlap_weights()
        assert "E01" in ow.weights
        assert ow.weights["E01"].get("E02") == pytest.approx(0.3)


class TestDeterminism:
    def test_same_seed_same_incidents(self) -> None:
        a1 = list(SyntheticAdapter(seed=7).iter_incidents())
        a2 = list(SyntheticAdapter(seed=7).iter_incidents())
        assert len(a1) == len(a2)
        for r1, r2 in zip(a1, a2, strict=True):
            assert r1 == r2

    def test_different_seed_different_incidents(self) -> None:
        a1 = list(SyntheticAdapter(seed=7).iter_incidents())
        a2 = list(SyntheticAdapter(seed=99).iter_incidents())
        # Same count (ground truth is fixed), but content differs
        assert len(a1) == len(a2)
        assert any(r1 != r2 for r1, r2 in zip(a1, a2, strict=True))

    def test_default_seed_is_stable(self) -> None:
        a1 = list(SyntheticAdapter().iter_incidents())
        a2 = list(SyntheticAdapter().iter_incidents())
        assert a1 == a2
