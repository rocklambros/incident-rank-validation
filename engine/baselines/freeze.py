"""Pure assembler for rankings_baselines.json (U3 T7).

Reads concordance.json to byte-pin kappa values — NEVER hardcodes them.
No I/O side effects; all writes are performed by the freeze CLI (T9).

The kappa values stored in the output dict come FROM the concordance.json
file, not from the computation.  The computation is used only to verify
that the reproduced kappa matches the file (raises ValueError on deviation
> 1e-9).

Schema version: 4 (adds prospective_power block per plan D6).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import numpy.typing as npt

from engine.baselines.bare_lambda import compute_bare_lambda_sensitivity
from engine.baselines.measurable_subset import compute_measurable_subset_kappa
from engine.baselines.previous_ranking import compute_previous_ranking
from engine.decide.prospective_power import prospective_power_statement

__all__ = ["build_rankings_baselines", "_sha256_path"]

_SCHEMA_VERSION: int = 4
_ARTIFACT_TYPE: str = "rankings_baselines"
_BOOTSTRAP_SEED: int = 20260520
_N_BOOTSTRAP: int = 5000
_ATOL: float = 1e-9


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def _sha256_path(path: Path) -> str:
    """Compute SHA256 hex digest of a file (for use by the freeze CLI)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _find_repo_root(start: Path) -> Path | None:
    """Walk parent directories to find the one containing a .git folder."""
    p = start.resolve()
    while p != p.parent:
        if (p / ".git").exists():
            return p
        p = p.parent
    return None


# ---------------------------------------------------------------------------
# Main assembler
# ---------------------------------------------------------------------------


def build_rankings_baselines(
    concordance_json_path: Path,
    lambda_samples: npt.NDArray[np.float64],
    vote_rank_samples: npt.NDArray[np.float64],
    inf_entry_ids: tuple[str, ...],
    vote_entry_ids: tuple[str, ...],
    measurable_entry_ids: tuple[str, ...],
    not_measurable_entry_ids: tuple[str, ...],
    entry_strata: dict[str, tuple[str, ...]],
    stratum_sizes: dict[str, int],
    generated_from: dict[str, object],
    cycle: str = "2026",
) -> dict[str, object]:
    """Assemble the rankings_baselines.json dict (U9 contract, schema_version=4).

    Reads ``concordance_json_path`` to byte-pin kappa values; raises ValueError
    if the reproduced kappa deviates from the file by more than 1e-9.  The
    kappa values stored in the output dict come **from the file**, not from a
    hand-typed constant.

    Parameters
    ----------
    concordance_json_path:
        Absolute path to ``cycles/.../results/concordance.json``; read to
        obtain the byte-pinned reference kappa / CI.
    lambda_samples:
        (N, n_entries) posterior lambda draws (e.g. 16000×20).
    vote_rank_samples:
        (M, n_entries) bootstrap vote rank draws (5000×20).
    inf_entry_ids:
        Ordered entry IDs from the inference summary.
    vote_entry_ids:
        Ordered entry IDs from the vote posterior.
    measurable_entry_ids:
        Entry IDs deemed measurable (e.g. 17 for 2026).
    not_measurable_entry_ids:
        Frame-blind entry IDs excluded from measurability (e.g. LLM04/08/10).
    entry_strata:
        Maps each entry ID to its observed strata names.
    stratum_sizes:
        Maps each stratum name to its incident count.
    generated_from:
        Pre-assembled provenance dict (paths + SHA256s); passed through
        verbatim into the output.
    cycle:
        Cycle identifier string, e.g. ``"2026"``.

    Returns
    -------
    dict[str, object]
        JSON-serialisable rankings_baselines dict (schema_version=4).

    Raises
    ------
    ValueError
        If the reproduced kappa deviates from concordance.json by > 1e-9.
    """
    # ------------------------------------------------------------------
    # 1. Read concordance.json for byte-pinned reference values
    # ------------------------------------------------------------------
    concordance_text = concordance_json_path.read_text()
    concordance = json.loads(concordance_text)

    # Values come FROM the file — never hardcoded
    ref_kappa_median: float = float(concordance["weighted_kappa_median"])
    ref_kappa_ci: list[float] = [float(x) for x in concordance["weighted_kappa_ci"]]
    ref_total_count: int = int(concordance["total_count"])
    ref_measurable_count: int = int(concordance["measurable_count"])
    ref_ci_method: str = str(concordance.get("ci_method", "paired_draw_percentile"))

    # ------------------------------------------------------------------
    # 2. Compute previous ranking (incidence, full common set — no measurability filter)
    # ------------------------------------------------------------------
    prev = compute_previous_ranking(
        lambda_samples=lambda_samples,
        vote_rank_samples=vote_rank_samples,
        inf_entry_ids=inf_entry_ids,
        vote_entry_ids=vote_entry_ids,
        entry_strata=entry_strata,
        stratum_sizes=stratum_sizes,
    )

    # ------------------------------------------------------------------
    # 3. Byte-pin validation (raises on deviation > 1e-9)
    # ------------------------------------------------------------------
    if prev.n_common != ref_total_count:
        raise ValueError(
            f"n_common={prev.n_common} != concordance.json total_count={ref_total_count}. "
            "Check inf_entry_ids and vote_entry_ids cover all 20 common entries."
        )

    kappa_diff = abs(prev.kappa_median - ref_kappa_median)
    if kappa_diff > _ATOL:
        raise ValueError(
            f"kappa_median={prev.kappa_median!r} deviates from concordance.json "
            f"{ref_kappa_median!r} by {kappa_diff:.2e} (threshold={_ATOL}). "
            "Verify lambda_samples (16000×20) and vote_rank_samples (5000×20, "
            "seed=20260520) are the correct inputs."
        )

    ci_lo_diff = abs(prev.kappa_ci_lo - ref_kappa_ci[0])
    if ci_lo_diff > _ATOL:
        raise ValueError(
            f"kappa_ci_lo={prev.kappa_ci_lo!r} deviates from concordance.json "
            f"{ref_kappa_ci[0]!r} by {ci_lo_diff:.2e}"
        )

    ci_hi_diff = abs(prev.kappa_ci_hi - ref_kappa_ci[1])
    if ci_hi_diff > _ATOL:
        raise ValueError(
            f"kappa_ci_hi={prev.kappa_ci_hi!r} deviates from concordance.json "
            f"{ref_kappa_ci[1]!r} by {ci_hi_diff:.2e}"
        )

    # ------------------------------------------------------------------
    # 4. Bare-lambda sensitivity
    # ------------------------------------------------------------------
    bare = compute_bare_lambda_sensitivity(
        lambda_samples=lambda_samples,
        vote_rank_samples=vote_rank_samples,
        inf_entry_ids=inf_entry_ids,
        vote_entry_ids=vote_entry_ids,
        incidence_kappa_median=prev.kappa_median,
    )

    # ------------------------------------------------------------------
    # 5. Secondary measurable-subset kappa (F14 footnote)
    # ------------------------------------------------------------------
    subset = compute_measurable_subset_kappa(
        lambda_samples=lambda_samples,
        vote_rank_samples=vote_rank_samples,
        inf_entry_ids=inf_entry_ids,
        vote_entry_ids=vote_entry_ids,
        measurable_entry_ids=measurable_entry_ids,
        entry_strata=entry_strata,
        stratum_sizes=stratum_sizes,
    )

    # ------------------------------------------------------------------
    # 6. Prospective power block (registered 2026-06-30, schema_version>=4)
    # ------------------------------------------------------------------
    power_block: dict[str, object] = prospective_power_statement()

    # ------------------------------------------------------------------
    # 7. Compute repo-relative path for byte_pinned_to
    # ------------------------------------------------------------------
    repo_root = _find_repo_root(concordance_json_path.parent)
    if repo_root is not None:
        try:
            byte_pinned_to = str(concordance_json_path.resolve().relative_to(repo_root))
        except ValueError:
            byte_pinned_to = str(concordance_json_path)
    else:
        byte_pinned_to = str(concordance_json_path)

    # ------------------------------------------------------------------
    # 8. Assemble output dict
    # The kappa values in previous_ranking come FROM concordance.json,
    # not from the computation above.
    # ------------------------------------------------------------------
    return {
        "artifact_type": _ARTIFACT_TYPE,
        "schema_version": _SCHEMA_VERSION,
        "cycle": cycle,
        "generated_from": generated_from,
        "entry_ids": list(inf_entry_ids),
        "measurable_entry_ids": list(measurable_entry_ids),
        "not_measurable": list(not_measurable_entry_ids),
        "previous_ranking": {
            "method": "incidence_lambda_size",
            "function": "_ranks_from_incidence",
            "tier_boundaries": list(prev.tier_boundaries),
            "n_common": prev.n_common,
            "bootstrap_draws": _N_BOOTSTRAP,
            "bootstrap_seed": _BOOTSTRAP_SEED,
            "ranking": list(prev.ranking),
            # Kappa values come FROM concordance.json (byte-pinned), NOT hardcoded
            "kappa_median": ref_kappa_median,
            "kappa_ci": ref_kappa_ci,
            "kappa_ci_method": ref_ci_method,
            "byte_pinned_to": byte_pinned_to,
        },
        "bare_lambda_sensitivity": {
            "method": "_ranks_from_lambda",
            "function": "_ranks_from_lambda",
            "ranking": list(bare.ranking),
            "kappa_median": bare.kappa_median,
            "method_kappa_delta": bare.method_kappa_delta,
            "disclosure": bare.disclosure,
        },
        "secondary_measurable_subset": {
            "measurable_kappa_median": subset.kappa_median,
            "measurable_kappa_ci": [subset.kappa_ci_lo, subset.kappa_ci_hi],
            "n_measurable": subset.n_measurable,
            "standing_caveat_contradiction": subset.standing_caveat_contradiction,
            "concordance_json_claims": f"measurable_count={ref_measurable_count}",
        },
        "prospective_power": power_block,
        "disclosures": {
            "incidence_kappa": True,
            "method_delta_zero": bool(bare.method_kappa_delta == 0.0),
            "ci_spans_zero": bool(ref_kappa_ci[0] < 0.0 < ref_kappa_ci[1]),
            "omnibus_bridge": True,
        },
    }
