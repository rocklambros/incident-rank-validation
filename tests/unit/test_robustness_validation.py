"""Tests for compute_robustness_validation (Task 1, arXiv preprint plan).

Checks key statistical properties of the robustness-validation JSON:
- The 2026 floor achieves Spearman ρ > 0.85 vs the adjudicated truth ranking.
- No frontier model/ensemble produces a ranking-delta CI that excludes 0 (cannot
  claim it beats the floor on incidence-ranking fidelity).
- The recall/precision correction does not produce a statistically significant
  difference between floor and ensemble (CI for delta crosses 0).
"""
from pathlib import Path

from tools.compute_robustness_validation import compute_robustness_validation

CYCLE = Path("projects/owasp-llm/cycles/2026-rarr")
FLOOR = Path("projects/owasp-llm/cycles/2026/classify/labeled_incidents.json")


def test_floor_ranks_high_no_frontier_beats_it() -> None:
    r = compute_robustness_validation(CYCLE, FLOOR)
    assert r["ranking_fidelity_spearman_vs_truth"]["floor"] > 0.85
    for m, d in r["ranking_delta_vs_floor_bootstrap"].items():
        lo, hi = d["ci95"]
        assert lo <= 0 <= hi, (
            f"Model {m!r}: CI [{lo}, {hi}] does not cross 0 — "
            "frontier would significantly beat floor, unexpectedly"
        )


def test_recall_correction_closes_gap() -> None:
    rc = compute_robustness_validation(CYCLE, FLOOR)["recall_correction_negL2"]
    lo, hi = rc["delta_ensemble_minus_floor_ci95"]
    assert lo <= 0 <= hi, (
        f"Recall-correction delta CI [{lo}, {hi}] does not cross 0 — "
        "correction does not neutralise the gap"
    )


def test_recall_correction_is_proportion_scale() -> None:
    """Guard against the raw-count scale bug: neg-L2 on proportion vectors must
    land in a small negative range, never the large magnitudes (~-10^4) that a
    raw-count L2 produces."""
    rc = compute_robustness_validation(CYCLE, FLOOR)["recall_correction_negL2"]
    assert -0.5 < rc["floor_cvcorrected"] < 0, (
        f"floor_cvcorrected={rc['floor_cvcorrected']} is out of proportion scale — "
        "neg-L2 is being computed on raw counts, not proportions"
    )
    assert -0.5 < rc["ensemble_cvcorrected"] < 0, (
        f"ensemble_cvcorrected={rc['ensemble_cvcorrected']} is out of proportion "
        "scale — neg-L2 is being computed on raw counts, not proportions"
    )
