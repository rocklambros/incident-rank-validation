"""U9 consumer-contract test for rankings_baselines.json.

Opens the committed artifact via the loader and asserts every key that U9
reads is present with the correct type and frozen value.  This test is the
guard that prevents U9 from drifting from the frozen baselines schema.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from engine.report.baselines_loader import (
    RankingsBaselines,
    load_rankings_baselines,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASELINES_PATH = (
    Path(__file__).resolve().parents[2]
    / "projects"
    / "owasp-llm"
    / "baselines"
    / "2026"
    / "rankings_baselines.json"
)


@pytest.fixture(scope="module")
def baselines() -> RankingsBaselines:
    if not BASELINES_PATH.exists():
        pytest.skip("rankings_baselines.json not present")
    return load_rankings_baselines(BASELINES_PATH)


# ---------------------------------------------------------------------------
# Contract assertions
# ---------------------------------------------------------------------------


class TestU9Contract:
    """Assert the exact schema U9 depends on is present and correctly typed."""

    def test_previous_ranking_kappa_median_frozen_value(
        self, baselines: RankingsBaselines
    ) -> None:
        """Frozen kappa median must equal the byte-pinned concordance.json value."""
        assert isinstance(baselines.previous_ranking.kappa_median, float)
        assert baselines.previous_ranking.kappa_median == pytest.approx(
            0.2028985507246377, abs=1e-12
        ), (
            f"previous_ranking.kappa_median drifted from the frozen concordance.json "
            f"value.  Got {baselines.previous_ranking.kappa_median!r}"
        )

    def test_previous_ranking_kappa_ci_present_and_typed(
        self, baselines: RankingsBaselines
    ) -> None:
        ci = baselines.previous_ranking.kappa_ci
        assert isinstance(ci, list)
        assert len(ci) == 2
        assert all(isinstance(v, float) for v in ci)

    def test_previous_ranking_kappa_ci_frozen_bounds(
        self, baselines: RankingsBaselines
    ) -> None:
        ci = baselines.previous_ranking.kappa_ci
        assert ci[0] == pytest.approx(-0.1594202898550725, abs=1e-12)
        assert ci[1] == pytest.approx(0.5652173913043478, abs=1e-12)

    def test_previous_ranking_ranking_present_and_typed(
        self, baselines: RankingsBaselines
    ) -> None:
        ranking = baselines.previous_ranking.ranking
        assert isinstance(ranking, list)
        assert len(ranking) == 20
        assert all(isinstance(e, str) for e in ranking)

    def test_bare_lambda_sensitivity_method_kappa_delta_present(
        self, baselines: RankingsBaselines
    ) -> None:
        delta = baselines.bare_lambda_sensitivity.method_kappa_delta
        assert isinstance(delta, float)

    def test_bare_lambda_sensitivity_method_kappa_delta_is_zero(
        self, baselines: RankingsBaselines
    ) -> None:
        """U9 must NOT credit a kappa gain to the ranking method — delta == 0.0."""
        assert baselines.bare_lambda_sensitivity.method_kappa_delta == 0.0, (
            "method_kappa_delta != 0.0: U9 must not credit a kappa gain from the "
            "ranking method on 2026 data.  If this changes, a dated amendment note "
            "is required before crediting the delta."
        )

    def test_measurable_subset_kappa_present_and_typed(
        self, baselines: RankingsBaselines
    ) -> None:
        assert isinstance(baselines.measurable_subset_kappa, float)

    def test_prospective_power_n_required_present_and_typed(
        self, baselines: RankingsBaselines
    ) -> None:
        assert isinstance(baselines.prospective_power.n_required, int)
        assert baselines.prospective_power.n_required > 0

    def test_prospective_power_disclosure_present_and_typed(
        self, baselines: RankingsBaselines
    ) -> None:
        assert isinstance(baselines.prospective_power.disclosure, str)
        assert len(baselines.prospective_power.disclosure) > 0

    def test_prospective_power_excludes_zero_at_current_n_present_and_typed(
        self, baselines: RankingsBaselines
    ) -> None:
        assert isinstance(
            baselines.prospective_power.excludes_zero_at_current_n, bool
        )

    def test_measurable_entry_ids_present_and_typed(
        self, baselines: RankingsBaselines
    ) -> None:
        ids = baselines.measurable_entry_ids
        assert isinstance(ids, list)
        assert len(ids) == 17
        assert all(isinstance(e, str) for e in ids)

    def test_entry_ids_present_and_typed(
        self, baselines: RankingsBaselines
    ) -> None:
        ids = baselines.entry_ids
        assert isinstance(ids, list)
        assert len(ids) == 20
        assert all(isinstance(e, str) for e in ids)

    def test_raw_dict_preserves_all_top_level_keys(
        self, baselines: RankingsBaselines
    ) -> None:
        """Spot-check that the raw dict exposes all top-level schema keys."""
        required_keys = {
            "artifact_type",
            "schema_version",
            "cycle",
            "generated_from",
            "entry_ids",
            "measurable_entry_ids",
            "not_measurable",
            "previous_ranking",
            "bare_lambda_sensitivity",
            "secondary_measurable_subset",
            "prospective_power",
            "disclosures",
        }
        missing = required_keys - set(baselines._raw.keys())
        assert not missing, f"Missing top-level keys in rankings_baselines.json: {missing}"
