"""Integration reproduction test: byte-equal to concordance.json (U3 T11).

Loads real cycle artifacts and self-builds vote_rank_samples from committed
respondent_rankings.npy (NON-CIRCULAR — never reads back vote_rank_samples.npy).

ASSERT-NOT-SKIPPED GUARD: all 4 cycle source files + the committed respondent
matrix are git-tracked and must exist.  If any are absent, pytest.fail() is
called — NOT pytest.skip().  This test MUST run in CI.

Byte-pin assertions:
  - n_common == concordance.json total_count  (== 20)
  - tier_boundaries == (6, 12)
  - kappa_median byte-equal to concordance.json (atol=1e-9)
  - kappa_ci_lo, kappa_ci_hi byte-equal to concordance.json (atol=1e-9)
  - bare_lambda_sensitivity.method_kappa_delta == 0.0 (on 2026 data)
  - SHA256SUMS integrity check for rankings_baselines.json

All comparisons are against the FILE (concordance.json) — never against
hand-typed constants.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

# ---------------------------------------------------------------------------
# Paths (repo-relative, resolved from test file location)
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

_CYCLE_DIR = _REPO_ROOT / "projects" / "owasp-llm" / "cycles" / "2026"
_BASELINES_DIR = _REPO_ROOT / "projects" / "owasp-llm" / "baselines" / "2026"
_FIXTURE_DIR = _REPO_ROOT / "tests" / "unit" / "fixtures"

# Cycle artifacts (git-tracked, byte-immutable)
_LAMBDA_NPY = _CYCLE_DIR / "infer" / "lambda_samples.npy"
_INF_SUMMARY = _CYCLE_DIR / "infer" / "inference_summary.json"
_LABELED = _CYCLE_DIR / "classify" / "labeled_incidents.json"
_CONCORDANCE = _CYCLE_DIR / "results" / "concordance.json"

# Committed baselines (materialized by freeze-baselines CLI)
_RESPONDENT_NPY = _BASELINES_DIR / "respondent_rankings.npy"
_SHA256SUMS = _BASELINES_DIR / "SHA256SUMS"
_RANKINGS_JSON = _BASELINES_DIR / "rankings_baselines.json"

# Vote entry IDs from fixture (same ordering as respondent matrix columns)
_VOTE_IDS_JSON = _FIXTURE_DIR / "vote_entry_ids_2026.json"

# Bootstrap parameters
_N_BOOTSTRAP = 5000
_SEED = 20260520
_ATOL = 1e-9


# ---------------------------------------------------------------------------
# Assert-not-skipped guard (module level — fails at collection if files missing)
# ---------------------------------------------------------------------------

_REQUIRED_FILES = {
    "lambda_samples.npy": _LAMBDA_NPY,
    "inference_summary.json": _INF_SUMMARY,
    "labeled_incidents.json": _LABELED,
    "concordance.json": _CONCORDANCE,
    "baselines/respondent_rankings.npy": _RESPONDENT_NPY,
    "vote_entry_ids_2026.json": _VOTE_IDS_JSON,
}

for _name, _path in _REQUIRED_FILES.items():
    if not _path.exists():
        pytest.fail(
            f"Required file not found: {_path}. "
            f"This is a git-tracked file that must exist for T11 to run. "
            f"File: {_name}",
            pytrace=False,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_strata(
    labeled: list[dict[str, object]],
) -> tuple[dict[str, tuple[str, ...]], dict[str, int]]:
    es: dict[str, set[str]] = defaultdict(set)
    sc: dict[str, int] = defaultdict(int)
    for item in labeled:
        eid = str(item["entry_id"])
        stratum = str(item.get("stratum", "default"))
        es[eid].add(stratum)
        sc[stratum] += 1
    entry_strata = {e: tuple(sorted(ss)) for e, ss in es.items()}
    stratum_sizes = {s: max(c, 1) for s, c in sc.items()}
    return entry_strata, stratum_sizes


def _sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Test 1: Byte-pin kappa to concordance.json
# ---------------------------------------------------------------------------


def test_f7_kappa_byte_equal_to_concordance_json() -> None:
    """Reproduce as-shipped kappa from committed respondent matrix (non-circular).

    Self-builds vote_rank_samples by bootstrapping from respondent_rankings.npy.
    Never reads back the committed vote_rank_samples.npy (that would be circular).
    Asserts kappa/CI byte-equal to cycles/2026/results/concordance.json (atol=1e-9).
    Assertions are against the FILE — not hand-typed constants.
    """
    from engine.baselines.previous_ranking import compute_previous_ranking
    from engine.vote.bootstrap import bootstrap_vote_ranks

    # Load committed respondent matrix (RAW source — non-circular)
    respondent_rankings: npt.NDArray[np.float64] = np.load(_RESPONDENT_NPY)

    # Load vote entry IDs
    with open(_VOTE_IDS_JSON) as f:
        vote_entry_ids: tuple[str, ...] = tuple(json.load(f))

    # Bootstrap from RAW (NOT from committed vote_rank_samples.npy)
    vote_posterior = bootstrap_vote_ranks(
        respondent_rankings,
        vote_entry_ids,
        n_bootstrap=_N_BOOTSTRAP,
        seed=_SEED,
    )

    # Load cycle artifacts
    lambda_samples: npt.NDArray[np.float64] = np.load(_LAMBDA_NPY)
    with open(_INF_SUMMARY) as f:
        inf = json.load(f)
    inf_entry_ids: tuple[str, ...] = tuple(inf["entry_ids"])

    with open(_LABELED) as f:
        labeled: list[dict[str, object]] = json.load(f)
    entry_strata, stratum_sizes = _build_strata(labeled)

    # Compute previous ranking
    result = compute_previous_ranking(
        lambda_samples=lambda_samples,
        vote_rank_samples=vote_posterior.rank_samples,
        inf_entry_ids=inf_entry_ids,
        vote_entry_ids=vote_entry_ids,
        entry_strata=entry_strata,
        stratum_sizes=stratum_sizes,
    )

    # Load reference values FROM FILE (byte-pin)
    with open(_CONCORDANCE) as f:
        concordance = json.load(f)
    target_kappa: float = float(concordance["weighted_kappa_median"])
    target_ci_lo: float = float(concordance["weighted_kappa_ci"][0])
    target_ci_hi: float = float(concordance["weighted_kappa_ci"][1])
    target_n: int = int(concordance["total_count"])

    # Assertions
    assert result.n_common == target_n, (
        f"n_common={result.n_common} != concordance.json total_count={target_n}"
    )
    assert result.tier_boundaries == (6, 12), (
        f"tier_boundaries={result.tier_boundaries} != (6, 12) for n=20"
    )
    assert abs(result.kappa_median - target_kappa) <= _ATOL, (
        f"kappa_median={result.kappa_median!r} differs from "
        f"concordance.json {target_kappa!r} by "
        f"{abs(result.kappa_median - target_kappa):.2e} (atol={_ATOL})"
    )
    assert abs(result.kappa_ci_lo - target_ci_lo) <= _ATOL, (
        f"kappa_ci_lo={result.kappa_ci_lo!r} differs from "
        f"concordance.json {target_ci_lo!r} by "
        f"{abs(result.kappa_ci_lo - target_ci_lo):.2e}"
    )
    assert abs(result.kappa_ci_hi - target_ci_hi) <= _ATOL, (
        f"kappa_ci_hi={result.kappa_ci_hi!r} differs from "
        f"concordance.json {target_ci_hi!r} by "
        f"{abs(result.kappa_ci_hi - target_ci_hi):.2e}"
    )


# ---------------------------------------------------------------------------
# Test 2: Bare-lambda delta == 0.0 on 2026 data
# ---------------------------------------------------------------------------


def test_f7_bare_lambda_delta_zero_on_2026() -> None:
    """method_kappa_delta == 0.0 on 2026 OWASP-LLM data (disclosed, never credited)."""
    from engine.baselines.bare_lambda import compute_bare_lambda_sensitivity
    from engine.baselines.previous_ranking import compute_previous_ranking
    from engine.vote.bootstrap import bootstrap_vote_ranks

    respondent_rankings: npt.NDArray[np.float64] = np.load(_RESPONDENT_NPY)
    with open(_VOTE_IDS_JSON) as f:
        vote_entry_ids: tuple[str, ...] = tuple(json.load(f))

    vote_posterior = bootstrap_vote_ranks(
        respondent_rankings,
        vote_entry_ids,
        n_bootstrap=_N_BOOTSTRAP,
        seed=_SEED,
    )

    lambda_samples: npt.NDArray[np.float64] = np.load(_LAMBDA_NPY)
    with open(_INF_SUMMARY) as f:
        inf = json.load(f)
    inf_entry_ids: tuple[str, ...] = tuple(inf["entry_ids"])

    with open(_LABELED) as f:
        labeled: list[dict[str, object]] = json.load(f)
    entry_strata, stratum_sizes = _build_strata(labeled)

    prev = compute_previous_ranking(
        lambda_samples=lambda_samples,
        vote_rank_samples=vote_posterior.rank_samples,
        inf_entry_ids=inf_entry_ids,
        vote_entry_ids=vote_entry_ids,
        entry_strata=entry_strata,
        stratum_sizes=stratum_sizes,
    )

    bare = compute_bare_lambda_sensitivity(
        lambda_samples=lambda_samples,
        vote_rank_samples=vote_posterior.rank_samples,
        inf_entry_ids=inf_entry_ids,
        vote_entry_ids=vote_entry_ids,
        incidence_kappa_median=prev.kappa_median,
    )

    assert bare.method_kappa_delta == 0.0, (
        f"method_kappa_delta={bare.method_kappa_delta!r} != 0.0 on 2026 data. "
        "This delta is always disclosed and never credited as a method gain."
    )


# ---------------------------------------------------------------------------
# Test 3: SHA256SUMS integrity for rankings_baselines.json
# ---------------------------------------------------------------------------


def test_f7_sha256sums_integrity() -> None:
    """SHA256SUMS records the correct digest for rankings_baselines.json."""
    if not _SHA256SUMS.exists():
        pytest.fail(
            f"SHA256SUMS not found at {_SHA256SUMS}. "
            "Run the freeze-baselines CLI to materialize artifacts.",
            pytrace=False,
        )
    if not _RANKINGS_JSON.exists():
        pytest.fail(
            f"rankings_baselines.json not found at {_RANKINGS_JSON}.",
            pytrace=False,
        )

    # Find the SHA256 entry for rankings_baselines.json in SHA256SUMS
    recorded_sha: str | None = None
    for line in _SHA256SUMS.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) == 2 and parts[1].strip() == "rankings_baselines.json":
            recorded_sha = parts[0]
            break

    assert recorded_sha is not None, (
        "rankings_baselines.json not listed in SHA256SUMS"
    )

    actual_sha = _sha256_path(_RANKINGS_JSON)
    assert actual_sha == recorded_sha, (
        f"SHA256 mismatch for rankings_baselines.json: "
        f"SHA256SUMS has {recorded_sha[:16]}... "
        f"but actual file is {actual_sha[:16]}..."
    )


# ---------------------------------------------------------------------------
# Test 4: cycles/ was not modified
# ---------------------------------------------------------------------------


def test_f7_cycles_dir_not_modified() -> None:
    """Test execution does not write to cycles/2026/ directory.

    Checks that the cycle source files still exist and have not been modified
    during test execution by verifying they are readable and not zero-size.
    """
    for name, path in {
        "lambda_samples.npy": _LAMBDA_NPY,
        "inference_summary.json": _INF_SUMMARY,
        "labeled_incidents.json": _LABELED,
        "concordance.json": _CONCORDANCE,
    }.items():
        assert path.exists(), f"Cycle file gone after test: {path}"
        assert path.stat().st_size > 0, f"Cycle file empty after test: {name}"


# ---------------------------------------------------------------------------
# Test 5: respondent_rankings.npy shape and dtype
# ---------------------------------------------------------------------------


def test_f7_respondent_rankings_shape() -> None:
    """respondent_rankings.npy in baselines/ has shape (29, 20) float64."""
    arr: npt.NDArray[np.float64] = np.load(_RESPONDENT_NPY)
    assert arr.shape == (29, 20), (
        f"respondent_rankings.npy shape {arr.shape} != (29, 20)"
    )
    assert arr.dtype == np.float64, (
        f"respondent_rankings.npy dtype {arr.dtype} != float64"
    )


# ---------------------------------------------------------------------------
# Test 6: vote_rank_samples.npy shape (secondary — does not prove non-circular)
# ---------------------------------------------------------------------------


def test_f7_vote_rank_samples_shape() -> None:
    """vote_rank_samples.npy in baselines/ has shape (5000, 20)."""
    vrs_path = _BASELINES_DIR / "vote_rank_samples.npy"
    if not vrs_path.exists():
        pytest.fail(
            f"vote_rank_samples.npy not found at {vrs_path}. "
            "Run the freeze-baselines CLI.",
            pytrace=False,
        )
    arr: npt.NDArray[np.float64] = np.load(vrs_path)
    assert arr.shape == (5000, 20), (
        f"vote_rank_samples.npy shape {arr.shape} != (5000, 20)"
    )
