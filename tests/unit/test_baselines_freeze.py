"""Unit tests for build_rankings_baselines (T7, U3 Cluster C).

Tests:
1. Schema completeness: all required top-level keys present.
2. Byte-pin check raises on kappa mismatch.
3. Disclosures: all 4 keys present and correctly typed.
4. n_measurable matches the measurable-ids slice.
5. _sha256_path is callable and returns a valid hex string.
6. Kappa values in output come from concordance.json file, not computation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from engine.baselines.freeze import _sha256_path, build_rankings_baselines
from tests.unit.fixtures.baselines import (
    ENTRY_IDS_A,
    ENTRY_STRATA_A,
    LAMBDA_SAMPLES_A,
    MEASURABLE_IDS_A,
    STRATUM_SIZES_A,
    VOTE_RANK_SAMPLES_A,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_NOT_MEASURABLE_A: tuple[str, ...] = tuple(
    e for e in ENTRY_IDS_A if e not in set(MEASURABLE_IDS_A)
)  # ("E3", "E4")

_GENERATED_FROM_STUB: dict[str, object] = {
    "lambda_samples": {"path": "test/lambda.npy", "shape": [20, 4], "sha256": "abc"},
    "inference_summary": {"path": "test/inf.json", "sha256": "def"},
    "labeled_incidents": {"path": "test/labeled.json", "sha256": "ghi"},
    "respondent_rankings": {"path": "test/resp.npy", "shape": [5, 4], "sha256": "jkl"},
    "concordance_json": {"path": "test/concordance.json", "sha256": "mno", "seed": 20260520},
}


def _make_concordance_json(tmp_path: Path, kappa: float, ci: list[float]) -> Path:
    """Write a fake concordance.json with the given kappa/CI values."""
    conc = {
        "weighted_kappa_median": kappa,
        "weighted_kappa_ci": ci,
        "total_count": 4,
        "measurable_count": 2,
        "ci_method": "paired_draw_percentile",
    }
    p = tmp_path / "concordance.json"
    p.write_text(json.dumps(conc))
    return p


def _compute_expected_kappa() -> float:
    """Compute the actual kappa for fixture A so we can write a matching concordance.json."""
    from engine.baselines.previous_ranking import compute_previous_ranking

    result = compute_previous_ranking(
        lambda_samples=LAMBDA_SAMPLES_A,
        vote_rank_samples=VOTE_RANK_SAMPLES_A,
        inf_entry_ids=ENTRY_IDS_A,
        vote_entry_ids=ENTRY_IDS_A,
        entry_strata=ENTRY_STRATA_A,
        stratum_sizes=STRATUM_SIZES_A,
    )
    return result.kappa_median


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_schema_completeness(tmp_path: Path) -> None:
    """build_rankings_baselines returns a dict with all required top-level keys."""
    kappa = _compute_expected_kappa()
    ci = [kappa, kappa]  # CI = [kappa, kappa] since all draws are constant
    conc_path = _make_concordance_json(tmp_path, kappa, ci)

    result = build_rankings_baselines(
        concordance_json_path=conc_path,
        lambda_samples=LAMBDA_SAMPLES_A,
        vote_rank_samples=VOTE_RANK_SAMPLES_A,
        inf_entry_ids=ENTRY_IDS_A,
        vote_entry_ids=ENTRY_IDS_A,
        measurable_entry_ids=MEASURABLE_IDS_A,
        not_measurable_entry_ids=_NOT_MEASURABLE_A,
        entry_strata=ENTRY_STRATA_A,
        stratum_sizes=STRATUM_SIZES_A,
        generated_from=_GENERATED_FROM_STUB,
    )

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
    assert required_keys <= set(result.keys()), (
        f"Missing keys: {required_keys - set(result.keys())}"
    )

    assert result["artifact_type"] == "rankings_baselines"
    assert result["schema_version"] == 4
    assert result["cycle"] == "2026"


def test_previous_ranking_schema(tmp_path: Path) -> None:
    """previous_ranking block has all required fields."""
    kappa = _compute_expected_kappa()
    ci = [kappa, kappa]
    conc_path = _make_concordance_json(tmp_path, kappa, ci)

    result = build_rankings_baselines(
        concordance_json_path=conc_path,
        lambda_samples=LAMBDA_SAMPLES_A,
        vote_rank_samples=VOTE_RANK_SAMPLES_A,
        inf_entry_ids=ENTRY_IDS_A,
        vote_entry_ids=ENTRY_IDS_A,
        measurable_entry_ids=MEASURABLE_IDS_A,
        not_measurable_entry_ids=_NOT_MEASURABLE_A,
        entry_strata=ENTRY_STRATA_A,
        stratum_sizes=STRATUM_SIZES_A,
        generated_from=_GENERATED_FROM_STUB,
    )

    pr = result["previous_ranking"]
    assert isinstance(pr, dict)
    assert pr["kappa_ci_method"] == "paired_draw_percentile"
    assert pr["n_common"] == 4
    assert pr["bootstrap_draws"] == 5000
    assert pr["bootstrap_seed"] == 20260520
    assert pr["method"] == "incidence_lambda_size"
    assert pr["function"] == "_ranks_from_incidence"

    # kappa_median must come from the file (== kappa we wrote), not hardcoded
    assert isinstance(pr["kappa_median"], float)
    kappa_in_output = float(pr["kappa_median"])
    assert abs(kappa_in_output - kappa) < 1e-12, (
        "kappa_median must come from concordance.json file"
    )


def test_disclosures_all_four_keys(tmp_path: Path) -> None:
    """disclosures block has all 4 required boolean keys."""
    kappa = _compute_expected_kappa()
    ci = [kappa, kappa]
    conc_path = _make_concordance_json(tmp_path, kappa, ci)

    result = build_rankings_baselines(
        concordance_json_path=conc_path,
        lambda_samples=LAMBDA_SAMPLES_A,
        vote_rank_samples=VOTE_RANK_SAMPLES_A,
        inf_entry_ids=ENTRY_IDS_A,
        vote_entry_ids=ENTRY_IDS_A,
        measurable_entry_ids=MEASURABLE_IDS_A,
        not_measurable_entry_ids=_NOT_MEASURABLE_A,
        entry_strata=ENTRY_STRATA_A,
        stratum_sizes=STRATUM_SIZES_A,
        generated_from=_GENERATED_FROM_STUB,
    )

    disc = result["disclosures"]
    assert isinstance(disc, dict)
    for key in ("incidence_kappa", "method_delta_zero", "ci_spans_zero", "omnibus_bridge"):
        assert key in disc, f"Missing disclosures key: {key}"
        assert isinstance(disc[key], bool), f"disclosures.{key} must be bool"


def test_secondary_measurable_subset_n(tmp_path: Path) -> None:
    """secondary_measurable_subset.n_measurable == len(MEASURABLE_IDS_A) == 2."""
    kappa = _compute_expected_kappa()
    ci = [kappa, kappa]
    conc_path = _make_concordance_json(tmp_path, kappa, ci)

    result = build_rankings_baselines(
        concordance_json_path=conc_path,
        lambda_samples=LAMBDA_SAMPLES_A,
        vote_rank_samples=VOTE_RANK_SAMPLES_A,
        inf_entry_ids=ENTRY_IDS_A,
        vote_entry_ids=ENTRY_IDS_A,
        measurable_entry_ids=MEASURABLE_IDS_A,
        not_measurable_entry_ids=_NOT_MEASURABLE_A,
        entry_strata=ENTRY_STRATA_A,
        stratum_sizes=STRATUM_SIZES_A,
        generated_from=_GENERATED_FROM_STUB,
    )

    sms = result["secondary_measurable_subset"]
    assert isinstance(sms, dict)
    assert sms["n_measurable"] == 2, (
        f"expected n_measurable=2 for MEASURABLE_IDS_A=('E1','E2'), got {sms['n_measurable']}"
    )


def test_byte_pin_raises_on_kappa_mismatch(tmp_path: Path) -> None:
    """build_rankings_baselines raises ValueError if concordance.json kappa is wrong."""
    # Write a concordance.json with a wrong kappa (0.9999 != actual ~0.636)
    wrong_kappa = 0.9999
    conc_path = _make_concordance_json(tmp_path, wrong_kappa, [wrong_kappa, wrong_kappa])
    # Also set total_count=4 to pass the n_common check
    conc_data = {
        "weighted_kappa_median": wrong_kappa,
        "weighted_kappa_ci": [wrong_kappa, wrong_kappa],
        "total_count": 4,
        "measurable_count": 2,
        "ci_method": "paired_draw_percentile",
    }
    conc_path.write_text(json.dumps(conc_data))

    with pytest.raises(ValueError, match="deviates from concordance.json"):
        build_rankings_baselines(
            concordance_json_path=conc_path,
            lambda_samples=LAMBDA_SAMPLES_A,
            vote_rank_samples=VOTE_RANK_SAMPLES_A,
            inf_entry_ids=ENTRY_IDS_A,
            vote_entry_ids=ENTRY_IDS_A,
            measurable_entry_ids=MEASURABLE_IDS_A,
            not_measurable_entry_ids=_NOT_MEASURABLE_A,
            entry_strata=ENTRY_STRATA_A,
            stratum_sizes=STRATUM_SIZES_A,
            generated_from=_GENERATED_FROM_STUB,
        )


def test_sha256_path_callable_and_valid(tmp_path: Path) -> None:
    """_sha256_path returns a valid 64-char hex string."""
    test_file = tmp_path / "test.bin"
    test_file.write_bytes(b"hello world")
    digest = _sha256_path(test_file)
    assert len(digest) == 64
    assert all(c in "0123456789abcdef" for c in digest)
    # Known SHA256 of b"hello world"
    expected = hashlib.sha256(b"hello world").hexdigest()
    assert digest == expected


def test_generated_from_passed_through(tmp_path: Path) -> None:
    """generated_from dict is passed through verbatim."""
    kappa = _compute_expected_kappa()
    ci = [kappa, kappa]
    conc_path = _make_concordance_json(tmp_path, kappa, ci)

    custom_gf: dict[str, object] = {
        "lambda_samples": {"path": "custom/path.npy", "sha256": "xyz123"},
        "inference_summary": {"path": "custom/inf.json", "sha256": "abc456"},
        "labeled_incidents": {"path": "custom/lab.json", "sha256": "def789"},
        "respondent_rankings": {"path": "custom/resp.npy", "sha256": "ghi012"},
        "concordance_json": {"path": "custom/conc.json", "sha256": "jkl345", "seed": 20260520},
    }

    result = build_rankings_baselines(
        concordance_json_path=conc_path,
        lambda_samples=LAMBDA_SAMPLES_A,
        vote_rank_samples=VOTE_RANK_SAMPLES_A,
        inf_entry_ids=ENTRY_IDS_A,
        vote_entry_ids=ENTRY_IDS_A,
        measurable_entry_ids=MEASURABLE_IDS_A,
        not_measurable_entry_ids=_NOT_MEASURABLE_A,
        entry_strata=ENTRY_STRATA_A,
        stratum_sizes=STRATUM_SIZES_A,
        generated_from=custom_gf,
    )

    assert result["generated_from"] is custom_gf, (
        "generated_from must be passed through verbatim (same object)"
    )


def test_entry_ids_lists_in_output(tmp_path: Path) -> None:
    """entry_ids, measurable_entry_ids, not_measurable are lists in the output."""
    kappa = _compute_expected_kappa()
    ci = [kappa, kappa]
    conc_path = _make_concordance_json(tmp_path, kappa, ci)

    result = build_rankings_baselines(
        concordance_json_path=conc_path,
        lambda_samples=LAMBDA_SAMPLES_A,
        vote_rank_samples=VOTE_RANK_SAMPLES_A,
        inf_entry_ids=ENTRY_IDS_A,
        vote_entry_ids=ENTRY_IDS_A,
        measurable_entry_ids=MEASURABLE_IDS_A,
        not_measurable_entry_ids=_NOT_MEASURABLE_A,
        entry_strata=ENTRY_STRATA_A,
        stratum_sizes=STRATUM_SIZES_A,
        generated_from=_GENERATED_FROM_STUB,
    )

    assert isinstance(result["entry_ids"], list)
    assert isinstance(result["measurable_entry_ids"], list)
    assert isinstance(result["not_measurable"], list)
    assert set(result["entry_ids"]) == set(ENTRY_IDS_A)
    assert set(result["measurable_entry_ids"]) == set(MEASURABLE_IDS_A)
    assert set(result["not_measurable"]) == set(_NOT_MEASURABLE_A)
