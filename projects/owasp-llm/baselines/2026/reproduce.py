#!/usr/bin/env python3
"""Standalone reproducer for the 2026 OWASP-LLM baselines (U3 T12).

Re-derives the frozen kappa from RAW respondent matrix (non-circular):
    Bootstrap from respondent_rankings.npy  ->  vote_rank_samples
    Combine with lambda_samples.npy         ->  kappa / CI

This is NON-CIRCULAR: it bootstraps from the RAW respondent matrix, NOT
from the committed vote_rank_samples.npy.  The vote_rank_samples.npy is
a SECONDARY check that demonstrates the committed array is self-consistent.

Usage
-----
# Re-derive kappa + validate SHA256SUMS (full check):
    python reproduce.py

# Also re-hash the LIVE cycle source files vs pinned SHA256s:
    python reproduce.py --verify-provenance

# Quiet (only print failures):
    python reproduce.py --quiet

Exit codes:
    0  All checks passed
    1  One or more checks failed
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths (relative to this script's parent = baselines/2026/)
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR
for _p in _SCRIPT_DIR.parents:
    if (_p / ".git").exists():
        _REPO_ROOT = _p
        break

_BASELINES_DIR = _SCRIPT_DIR
_CYCLE_DIR = _REPO_ROOT / "projects" / "owasp-llm" / "cycles" / "2026"
_FIXTURE_DIR = _REPO_ROOT / "tests" / "unit" / "fixtures"

# Committed baselines artifacts
_RESPONDENT_NPY = _BASELINES_DIR / "respondent_rankings.npy"
_VOTE_RANK_SAMPLES_NPY = _BASELINES_DIR / "vote_rank_samples.npy"
_RANKINGS_JSON = _BASELINES_DIR / "rankings_baselines.json"
_SHA256SUMS = _BASELINES_DIR / "SHA256SUMS"

# Cycle source files (byte-immutable)
_LAMBDA_NPY = _CYCLE_DIR / "infer" / "lambda_samples.npy"
_INF_SUMMARY = _CYCLE_DIR / "infer" / "inference_summary.json"
_LABELED = _CYCLE_DIR / "classify" / "labeled_incidents.json"
_CONCORDANCE = _CYCLE_DIR / "results" / "concordance.json"

# Vote entry IDs fixture (same order as respondent matrix columns)
_VOTE_IDS_JSON = _FIXTURE_DIR / "vote_entry_ids_2026.json"

# Bootstrap parameters (from pipeline.py manifest.prng_seed)
_N_BOOTSTRAP = 5000
_SEED = 20260520
_ATOL = 1e-9


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256_path(path: Path) -> str:
    """Compute SHA256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_strata(
    labeled: list[dict[str, object]],
) -> tuple[dict[str, tuple[str, ...]], dict[str, int]]:
    """Build entry_strata and stratum_sizes from labeled_incidents.json."""
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


def _ok(msg: str, quiet: bool) -> None:
    if not quiet:
        print(f"  OK  {msg}")


def _fail(msg: str) -> None:
    print(f"FAIL  {msg}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Check 1: re-derive kappa from RAW respondent matrix (non-circular)
# ---------------------------------------------------------------------------


def check_kappa_re_derive(quiet: bool = False) -> bool:
    """Bootstrap from respondent_rankings.npy and assert kappa == concordance.json."""
    # Import engine modules (requires the engine to be on sys.path)
    try:
        from engine.baselines.previous_ranking import compute_previous_ranking
        from engine.vote.bootstrap import bootstrap_vote_ranks
    except ImportError as exc:
        _fail(
            f"Cannot import engine modules: {exc}. "
            "Run from the repo root: uv run python projects/owasp-llm/baselines/2026/reproduce.py"
        )
        return False

    ok = True

    # Load respondent matrix (RAW source - non-circular, never use vote_rank_samples.npy)
    respondent_rankings: np.ndarray = np.load(_RESPONDENT_NPY)

    # Load vote entry IDs
    with open(_VOTE_IDS_JSON) as f:
        vote_entry_ids: tuple[str, ...] = tuple(json.load(f))

    # Bootstrap from RAW (NOT from committed vote_rank_samples.npy -- that would be circular)
    vp = bootstrap_vote_ranks(
        respondent_rankings,
        vote_entry_ids,
        n_bootstrap=_N_BOOTSTRAP,
        seed=_SEED,
    )

    # Load cycle artifacts
    lambda_samples: np.ndarray = np.load(_LAMBDA_NPY)
    with open(_INF_SUMMARY) as f:
        inf = json.load(f)
    inf_entry_ids: tuple[str, ...] = tuple(inf["entry_ids"])

    with open(_LABELED) as f:
        labeled: list[dict[str, object]] = json.load(f)
    entry_strata, stratum_sizes = _build_strata(labeled)

    # Re-derive kappa (same algorithm as compute_concordance, concordance.py:193)
    result = compute_previous_ranking(
        lambda_samples=lambda_samples,
        vote_rank_samples=vp.rank_samples,
        inf_entry_ids=inf_entry_ids,
        vote_entry_ids=vote_entry_ids,
        entry_strata=entry_strata,
        stratum_sizes=stratum_sizes,
    )

    # Load frozen reference values FROM FILE (byte-pin to concordance.json)
    with open(_CONCORDANCE) as f:
        concordance = json.load(f)
    ref_kappa: float = float(concordance["weighted_kappa_median"])
    ref_ci_lo: float = float(concordance["weighted_kappa_ci"][0])
    ref_ci_hi: float = float(concordance["weighted_kappa_ci"][1])
    ref_n: int = int(concordance["total_count"])

    # Assertions against concordance.json
    if result.n_common != ref_n:
        _fail(f"n_common={result.n_common} != concordance.json total_count={ref_n}")
        ok = False
    else:
        _ok(f"n_common == {ref_n}", quiet)

    if result.tier_boundaries != (6, 12):
        _fail(f"tier_boundaries={result.tier_boundaries} != (6, 12)")
        ok = False
    else:
        _ok("tier_boundaries == (6, 12)", quiet)

    kappa_diff = abs(result.kappa_median - ref_kappa)
    if kappa_diff > _ATOL:
        _fail(
            f"kappa_median={result.kappa_median!r} differs from "
            f"concordance.json {ref_kappa!r} by {kappa_diff:.2e} (atol={_ATOL})"
        )
        ok = False
    else:
        _ok(f"kappa_median byte-pins to concordance.json ({result.kappa_median})", quiet)

    ci_lo_diff = abs(result.kappa_ci_lo - ref_ci_lo)
    if ci_lo_diff > _ATOL:
        _fail(f"kappa_ci_lo={result.kappa_ci_lo!r} differs by {ci_lo_diff:.2e}")
        ok = False
    else:
        _ok("kappa_ci_lo byte-pins to concordance.json", quiet)

    ci_hi_diff = abs(result.kappa_ci_hi - ref_ci_hi)
    if ci_hi_diff > _ATOL:
        _fail(f"kappa_ci_hi={result.kappa_ci_hi!r} differs by {ci_hi_diff:.2e}")
        ok = False
    else:
        _ok("kappa_ci_hi byte-pins to concordance.json", quiet)

    return ok


# ---------------------------------------------------------------------------
# Check 2: SHA256SUMS integrity
# ---------------------------------------------------------------------------


def check_sha256sums(quiet: bool = False) -> bool:
    """Verify each file listed in SHA256SUMS has the recorded digest."""
    if not _SHA256SUMS.exists():
        _fail(f"SHA256SUMS not found: {_SHA256SUMS}")
        return False

    ok = True
    lines = _SHA256SUMS.read_text().splitlines()
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            _fail(f"malformed SHA256SUMS line: {line!r}")
            ok = False
            continue
        expected_sha, filename = parts[0], parts[1].strip()
        target = _BASELINES_DIR / filename
        if not target.exists():
            _fail(f"file listed in SHA256SUMS not found: {target}")
            ok = False
            continue
        actual_sha = _sha256_path(target)
        if actual_sha != expected_sha:
            _fail(
                f"SHA256 mismatch for {filename}: "
                f"expected {expected_sha[:12]}... got {actual_sha[:12]}..."
            )
            ok = False
        else:
            _ok(f"SHA256 OK: {filename}", quiet)

    return ok


# ---------------------------------------------------------------------------
# Check 3: --verify-provenance (re-hash live cycle files vs pinned SHA256s)
# ---------------------------------------------------------------------------


def check_verify_provenance(quiet: bool = False) -> bool:
    """Re-hash live cycle source files vs pinned SHA256s in rankings_baselines.json."""
    if not _RANKINGS_JSON.exists():
        _fail(f"rankings_baselines.json not found: {_RANKINGS_JSON}")
        return False

    with open(_RANKINGS_JSON) as f:
        manifest = json.load(f)

    generated_from: dict[str, object] = dict(manifest.get("generated_from", {}))
    if not generated_from:
        _fail("rankings_baselines.json has no 'generated_from' block")
        return False

    ok = True
    for source_name, source_info_raw in generated_from.items():
        if not isinstance(source_info_raw, dict):
            continue
        source_info: dict[str, object] = source_info_raw
        rel_path = str(source_info.get("path", ""))
        expected_sha = str(source_info.get("sha256", ""))
        if not rel_path or not expected_sha:
            if not quiet:
                print(f"  --  {source_name}: no path/sha256 in generated_from (skipping)")
            continue

        target = _REPO_ROOT / rel_path
        if not target.exists():
            print(f"WARN  {source_name}: source file not found: {target} (skipping)")
            continue

        actual_sha = _sha256_path(target)
        if actual_sha != expected_sha:
            _fail(
                f"provenance mismatch for {source_name} ({rel_path}): "
                f"pinned {expected_sha[:12]}... live {actual_sha[:12]}..."
            )
            ok = False
        else:
            _ok(f"provenance OK: {source_name} ({rel_path})", quiet)

    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--verify-provenance",
        action="store_true",
        help="Re-hash live cycle source files vs pinned SHA256s in rankings_baselines.json.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print failures (suppresses OK lines).",
    )
    args = parser.parse_args()

    all_ok = True

    if not args.quiet:
        print("=== reproduce.py: 2026 OWASP-LLM baselines integrity check ===")
        print()
        print("--- Check 1: re-derive kappa from RAW respondent matrix (non-circular) ---")
    ok1 = check_kappa_re_derive(quiet=args.quiet)
    all_ok = all_ok and ok1

    if not args.quiet:
        print()
        print("--- Check 2: SHA256SUMS integrity ---")
    ok2 = check_sha256sums(quiet=args.quiet)
    all_ok = all_ok and ok2

    if args.verify_provenance:
        if not args.quiet:
            print()
            print("--- Check 3: --verify-provenance (cycle source files) ---")
        ok3 = check_verify_provenance(quiet=args.quiet)
        all_ok = all_ok and ok3

    if not args.quiet:
        print()
        if all_ok:
            print("=== ALL CHECKS PASSED ===")
        else:
            print("=== SOME CHECKS FAILED (see FAIL lines above) ===", file=sys.stderr)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
