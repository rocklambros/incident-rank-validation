"""Unit tests for engine.adapters.synthetic_stress.SyntheticStressAdapter."""

from __future__ import annotations

import pytest

from engine.adapters.base import CorpusAdapter
from engine.adapters.synthetic_stress import SyntheticStressAdapter
from engine.model.overlap import OverlapWeights
from engine.schema import BiasProfile, EntryDefinition, IncidentRecord


class TestSyntheticStressAdapterInterface:
    def test_is_corpus_adapter_subclass(self) -> None:
        assert issubclass(SyntheticStressAdapter, CorpusAdapter)

    def test_instance_is_corpus_adapter(self) -> None:
        assert isinstance(SyntheticStressAdapter(), CorpusAdapter)


class TestDeterminism:
    def test_same_seed_same_incidents(self) -> None:
        a1 = list(SyntheticStressAdapter(seed=99).iter_incidents())
        a2 = list(SyntheticStressAdapter(seed=99).iter_incidents())
        assert len(a1) == len(a2)
        for r1, r2 in zip(a1, a2, strict=True):
            assert r1 == r2

    def test_different_seeds_differ(self) -> None:
        a1 = list(SyntheticStressAdapter(seed=99).iter_incidents())
        a2 = list(SyntheticStressAdapter(seed=1).iter_incidents())
        # Same count (counts are deterministic, not seed-dependent) but different content
        assert len(a1) == len(a2)
        assert any(r1 != r2 for r1, r2 in zip(a1, a2, strict=True))


class TestIterIncidents:
    def test_yields_incident_records(self) -> None:
        adapter = SyntheticStressAdapter()
        incidents = list(adapter.iter_incidents())
        assert len(incidents) > 0
        for inc in incidents:
            assert isinstance(inc, IncidentRecord)

    def test_correct_total_count(self) -> None:
        # stratum_a: 90+55+30 = 175; stratum_b: 50+30+15 = 95; total = 270
        adapter = SyntheticStressAdapter()
        incidents = list(adapter.iter_incidents())
        assert len(incidents) == 270

    def test_count_per_stratum(self) -> None:
        adapter = SyntheticStressAdapter()
        by_stratum: dict[str, int] = {}
        for inc in adapter.iter_incidents():
            by_stratum[inc.corpus_stratum] = by_stratum.get(inc.corpus_stratum, 0) + 1
        assert by_stratum["stratum_a"] == 175
        assert by_stratum["stratum_b"] == 95

    def test_all_have_valid_stratum(self) -> None:
        adapter = SyntheticStressAdapter()
        strata = {"stratum_a", "stratum_b"}
        for inc in adapter.iter_incidents():
            assert inc.corpus_stratum in strata

    def test_no_frame_blind_incidents_generated(self) -> None:
        """STR04-STR09 are frame_blind; no incidents should be generated for them."""
        adapter = SyntheticStressAdapter()
        frame_blind_ids = {"STR04", "STR05", "STR06", "STR07", "STR08", "STR09"}
        for inc in adapter.iter_incidents():
            for label in inc.native_labels:
                assert label not in frame_blind_ids

    def test_no_classifier_blind_incidents_generated(self) -> None:
        """STR10-STR12 are classifier-blind with zero ground-truth counts."""
        adapter = SyntheticStressAdapter()
        classifier_blind_ids = {"STR10", "STR11", "STR12"}
        for inc in adapter.iter_incidents():
            for label in inc.native_labels:
                assert label not in classifier_blind_ids


class TestEntryDefinitions:
    def test_returns_tuple(self) -> None:
        adapter = SyntheticStressAdapter()
        assert isinstance(adapter.entry_definitions(), tuple)

    def test_twelve_entries(self) -> None:
        adapter = SyntheticStressAdapter()
        assert len(adapter.entry_definitions()) == 12

    def test_all_are_entry_definitions(self) -> None:
        adapter = SyntheticStressAdapter()
        for entry in adapter.entry_definitions():
            assert isinstance(entry, EntryDefinition)

    def test_three_measurable_entries(self) -> None:
        adapter = SyntheticStressAdapter()
        measurable = [
            e for e in adapter.entry_definitions()
            if not e.frame_blind and e.entry_id in {"STR01", "STR02", "STR03"}
        ]
        assert len(measurable) == 3

    def test_six_frame_blind_entries(self) -> None:
        adapter = SyntheticStressAdapter()
        frame_blind = [e for e in adapter.entry_definitions() if e.frame_blind]
        assert len(frame_blind) == 6

    def test_frame_blind_ids(self) -> None:
        adapter = SyntheticStressAdapter()
        frame_blind_ids = {e.entry_id for e in adapter.entry_definitions() if e.frame_blind}
        assert frame_blind_ids == {"STR04", "STR05", "STR06", "STR07", "STR08", "STR09"}

    def test_three_classifier_blind_entries(self) -> None:
        """STR10-STR12 are frame_blind=False (in scope) but have zero incidents."""
        adapter = SyntheticStressAdapter()
        classifier_blind = [
            e for e in adapter.entry_definitions()
            if not e.frame_blind and e.entry_id in {"STR10", "STR11", "STR12"}
        ]
        assert len(classifier_blind) == 3

    def test_classifier_blind_not_frame_blind(self) -> None:
        adapter = SyntheticStressAdapter()
        entries_by_id = {e.entry_id: e for e in adapter.entry_definitions()}
        for eid in ("STR10", "STR11", "STR12"):
            assert entries_by_id[eid].frame_blind is False


class TestOverlapWeights:
    def test_returns_overlap_weights(self) -> None:
        adapter = SyntheticStressAdapter()
        assert isinstance(adapter.overlap_weights(), OverlapWeights)

    def test_no_self_loop(self) -> None:
        adapter = SyntheticStressAdapter()
        ow = adapter.overlap_weights()
        for target, sources in ow.weights.items():
            assert target not in sources

    def test_multi_target_leakage_structure(self) -> None:
        """STR07's FPs must split across two distinct targets."""
        adapter = SyntheticStressAdapter()
        ow = adapter.overlap_weights()
        # Both targets present
        assert "STR01" in ow.weights
        assert "STR02" in ow.weights
        # Correct source in each
        assert "STR07" in ow.weights["STR01"]
        assert "STR07" in ow.weights["STR02"]

    def test_str07_leaks_into_str01_at_0_6(self) -> None:
        adapter = SyntheticStressAdapter()
        ow = adapter.overlap_weights()
        assert ow.weights["STR01"]["STR07"] == pytest.approx(0.6)

    def test_str07_leaks_into_str02_at_0_4(self) -> None:
        adapter = SyntheticStressAdapter()
        ow = adapter.overlap_weights()
        assert ow.weights["STR02"]["STR07"] == pytest.approx(0.4)

    def test_column_stochastic_str07(self) -> None:
        """STR07 column sum must equal 1.0 (fully allocated)."""
        adapter = SyntheticStressAdapter()
        ow = adapter.overlap_weights()
        col_sum = sum(ow.weights.get(tgt, {}).get("STR07", 0.0) for tgt in ow.weights)
        assert col_sum == pytest.approx(1.0)

    def test_column_stochastic_all_sources(self) -> None:
        adapter = SyntheticStressAdapter()
        ow = adapter.overlap_weights()
        all_sources: set[str] = set()
        for tgt_map in ow.weights.values():
            all_sources.update(tgt_map.keys())
        for src in all_sources:
            col_sum = sum(ow.weights.get(tgt, {}).get(src, 0.0) for tgt in ow.weights)
            assert col_sum <= 1.0 + 1e-6, f"column sum for {src!r} = {col_sum:.4f} > 1"


class TestStratumSizes:
    def test_returns_dict(self) -> None:
        adapter = SyntheticStressAdapter()
        assert isinstance(adapter.stratum_sizes(), dict)

    def test_expected_strata_present(self) -> None:
        adapter = SyntheticStressAdapter()
        sizes = adapter.stratum_sizes()
        assert "stratum_a" in sizes
        assert "stratum_b" in sizes

    def test_all_positive(self) -> None:
        adapter = SyntheticStressAdapter()
        for stratum, size in adapter.stratum_sizes().items():
            assert size > 0, f"stratum {stratum!r} has non-positive size {size}"

    def test_expected_sizes(self) -> None:
        adapter = SyntheticStressAdapter()
        sizes = adapter.stratum_sizes()
        assert sizes["stratum_a"] == 400
        assert sizes["stratum_b"] == 200


class TestBiasProfiles:
    def test_returns_nonempty_tuple(self) -> None:
        adapter = SyntheticStressAdapter()
        profiles = adapter.bias_profiles()
        assert isinstance(profiles, tuple)
        assert len(profiles) > 0

    def test_all_are_bias_profile_instances(self) -> None:
        adapter = SyntheticStressAdapter()
        for profile in adapter.bias_profiles():
            assert isinstance(profile, BiasProfile)

    def test_covers_both_strata(self) -> None:
        adapter = SyntheticStressAdapter()
        strata = {p.stratum for p in adapter.bias_profiles()}
        assert "stratum_a" in strata
        assert "stratum_b" in strata
