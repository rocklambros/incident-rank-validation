"""F6 core (U2-1): thin/under-detected recall cell DETECTION + adequacy-flag fix + uniform floor.

Tests are written RED first, then the implementation is added to make them GREEN.

Invariants tested:
  1. Byte-identity when F6 is off (K=0, epsilon=0.0) — THE #1 INVARIANT.
  2. Thin cell (K>0, total_in_sample < K) never flag=='adequate'.
  3. TP==0 cell (under-detected) never flag=='adequate', independent of K.
  4. Uniform recall floor (epsilon>0) clips sampled recall ≥ epsilon; thick cells unaffected.
  5. F6 logic is recall-only: precision + rollup posteriors byte-identical with F6 on vs off;
     apply_empirical_precision_prior still fires on its (1.0,1.0) cells.
"""
from __future__ import annotations

import inspect
import math

import jax.numpy as jnp
import numpy as np
import pytest

from engine.calibrate.calibrate import (
    EntryCalibrationReport,
    compute_calibration,
)
from engine.calibrate.tally import PrecisionTally, RecallTally, TallyResult
from engine.model.inference import run_inference

# ---------------------------------------------------------------------------
# Minimal TallyResult helpers
# ---------------------------------------------------------------------------

def _tally(
    *,
    rec_tp: int,
    rec_fn: int,
    prec_tp: int = 5,
    prec_fp: int = 1,
    entry_id: str = "E1",
    stratum: str = "security",
) -> TallyResult:
    """Build a minimal TallyResult with one recall cell and one precision cell."""
    rec_total = rec_tp + rec_fn
    return TallyResult(
        recall_counts={(entry_id, stratum): RecallTally(
            true_positives=rec_tp,
            false_negatives=rec_fn,
            total_in_sample=rec_total,
        )},
        precision_counts={(entry_id, stratum): PrecisionTally(
            true_positives=prec_tp,
            false_positives=prec_fp,
            total=prec_tp + prec_fp,
        )},
        rollup_counts={},
        total_coded=rec_total + prec_tp + prec_fp,
        amendments_applied=0,
    )


# ---------------------------------------------------------------------------
# 1. Byte-identity when F6 is off (THE #1 INVARIANT)
# ---------------------------------------------------------------------------


class TestByteIdentityWhenOff:
    """With K=0 and epsilon=0.0 (defaults), output is indistinguishable from
    pre-F6 output: same posteriors, same adequacy flags, same EntryCalibrationReport."""

    def test_recall_posteriors_unchanged(self) -> None:
        """Recall Beta posteriors must be identical to default call."""
        tally = _tally(rec_tp=10, rec_fn=2)
        cal_default, _ = compute_calibration(
            tally, ["E1"], frame_blind_ids=set()
        )
        cal_off, _ = compute_calibration(
            tally, ["E1"], frame_blind_ids=set(),
            recall_min_denominator=0,
        )
        assert cal_default.recall == cal_off.recall

    def test_precision_posteriors_unchanged(self) -> None:
        """Precision Beta posteriors must be identical to default call."""
        tally = _tally(rec_tp=10, rec_fn=2)
        cal_default, _ = compute_calibration(
            tally, ["E1"], frame_blind_ids=set()
        )
        cal_off, _ = compute_calibration(
            tally, ["E1"], frame_blind_ids=set(),
            recall_min_denominator=0,
        )
        assert cal_default.precision == cal_off.precision

    def test_adequacy_flags_unchanged(self) -> None:
        """flag + reason fields must be identical to default call."""
        tally = _tally(rec_tp=10, rec_fn=2)
        _, diag_default = compute_calibration(
            tally, ["E1"], frame_blind_ids=set()
        )
        _, diag_off = compute_calibration(
            tally, ["E1"], frame_blind_ids=set(),
            recall_min_denominator=0,
        )
        r_default = diag_default.entry_reports["E1"]
        r_off = diag_off.entry_reports["E1"]
        assert r_default.flag == r_off.flag
        assert r_default.reason == r_off.reason

    def test_f6_fields_false_when_off(self) -> None:
        """With K=0, thin_denominator must be False (no threshold active)."""
        tally = _tally(rec_tp=10, rec_fn=2)
        _, diag = compute_calibration(
            tally, ["E1"], frame_blind_ids=set(), recall_min_denominator=0
        )
        rep: EntryCalibrationReport = diag.entry_reports["E1"]
        assert rep.thin_denominator is False

    def test_thick_cell_can_still_be_adequate_when_off(self) -> None:
        """A well-measured cell with K=0 must still reach 'adequate'."""
        # Build a high-n cell: TP=50, FN=5 → tight recall CI; TP=50, FP=5 → tight prec CI.
        tally = _tally(rec_tp=50, rec_fn=5, prec_tp=50, prec_fp=5)
        _, diag = compute_calibration(
            tally, ["E1"], frame_blind_ids=set(), recall_min_denominator=0
        )
        assert diag.entry_reports["E1"].flag == "adequate"


# ---------------------------------------------------------------------------
# 2. Thin cell (K > 0, total_in_sample < K) must NEVER be 'adequate'
# ---------------------------------------------------------------------------


class TestThinCellNeverAdequate:
    """A recall cell with total_in_sample < K is thin; must not report adequate."""

    def test_thin_cell_high_sample_count_but_below_K(self) -> None:
        """Even if the cell would otherwise be adequate (CI < 0.30), thin overrides."""
        # rec_tp=50, rec_fn=5 → Beta(51,6), very tight CI — would be 'adequate' with K=0.
        tally = _tally(rec_tp=50, rec_fn=5, prec_tp=50, prec_fp=5)
        _, diag_off = compute_calibration(
            tally, ["E1"], frame_blind_ids=set(), recall_min_denominator=0
        )
        assert diag_off.entry_reports["E1"].flag == "adequate", (
            "Precondition: this cell should be adequate with K=0"
        )

        # Now activate K = total_in_sample + 1 so cell is thin.
        K = 56 + 1  # total_in_sample = 50+5+1 = 56? No: rec_tp=50, rec_fn=5 → total=55.
        K = 56  # 55 < 56 → thin
        _, diag_on = compute_calibration(
            tally, ["E1"], frame_blind_ids=set(), recall_min_denominator=K
        )
        rep = diag_on.entry_reports["E1"]
        assert rep.flag != "adequate", (
            f"Thin cell must not be adequate; got flag={rep.flag!r}"
        )
        assert rep.thin_denominator is True

    def test_thin_cell_flag_is_wide(self) -> None:
        """The flag for a thin cell should be 'wide' (or similar non-adequate value)."""
        tally = _tally(rec_tp=50, rec_fn=5, prec_tp=50, prec_fp=5)
        K = 56  # total_in_sample=55 < 56
        _, diag = compute_calibration(
            tally, ["E1"], frame_blind_ids=set(), recall_min_denominator=K
        )
        rep = diag.entry_reports["E1"]
        assert rep.flag == "wide"

    def test_thick_cell_above_K_is_unaffected(self) -> None:
        """A cell with total_in_sample >= K is not thin and may still be adequate."""
        tally = _tally(rec_tp=50, rec_fn=5, prec_tp=50, prec_fp=5)
        K = 10  # total_in_sample=55 >= 10 → not thin
        _, diag = compute_calibration(
            tally, ["E1"], frame_blind_ids=set(), recall_min_denominator=K
        )
        rep = diag.entry_reports["E1"]
        assert rep.thin_denominator is False
        assert rep.flag == "adequate"

    def test_thin_reason_encodes_thin_denominator(self) -> None:
        """Reason string must indicate thin-denominator when flagged."""
        tally = _tally(rec_tp=50, rec_fn=5, prec_tp=50, prec_fp=5)
        K = 56
        _, diag = compute_calibration(
            tally, ["E1"], frame_blind_ids=set(), recall_min_denominator=K
        )
        rep = diag.entry_reports["E1"]
        assert "thin" in rep.reason.lower()


# ---------------------------------------------------------------------------
# 3. TP==0 cell (under-detected) must NEVER be 'adequate', independent of K
# ---------------------------------------------------------------------------


class TestUnderDetectedNeverAdequate:
    """A recall cell where true_positives == 0 and total_in_sample > 0 is under-detected."""

    def test_under_detected_forces_non_adequate(self) -> None:
        """TP=0 cell, regardless of CI width, must not be 'adequate'."""
        # rec_tp=0, rec_fn=1 → Beta(1,2), low recall.
        # prec is good to make it a candidate for adequate on the precision side.
        tally = _tally(rec_tp=0, rec_fn=1, prec_tp=50, prec_fp=5)
        _, diag = compute_calibration(
            tally, ["E1"], frame_blind_ids=set(), recall_min_denominator=0
        )
        rep = diag.entry_reports["E1"]
        assert rep.flag != "adequate", (
            f"Under-detected cell must not be adequate; got flag={rep.flag!r}"
        )
        assert rep.under_detected is True

    def test_under_detected_fires_without_K(self) -> None:
        """under_detected fires independently of K (even K=0)."""
        tally = _tally(rec_tp=0, rec_fn=5, prec_tp=50, prec_fp=5)
        for K in (0, 10, 100):
            _, diag = compute_calibration(
                tally, ["E1"], frame_blind_ids=set(), recall_min_denominator=K
            )
            rep = diag.entry_reports["E1"]
            assert rep.under_detected is True
            assert rep.flag != "adequate", f"Failed at K={K}: flag={rep.flag!r}"

    def test_detected_cell_not_under_detected(self) -> None:
        """TP > 0 cell is not under-detected."""
        tally = _tally(rec_tp=3, rec_fn=2)
        _, diag = compute_calibration(
            tally, ["E1"], frame_blind_ids=set(), recall_min_denominator=0
        )
        rep = diag.entry_reports["E1"]
        assert rep.under_detected is False

    def test_under_detected_reason_encodes_under_detected(self) -> None:
        """Reason string must indicate under-detected when flagged."""
        tally = _tally(rec_tp=0, rec_fn=3, prec_tp=50, prec_fp=5)
        _, diag = compute_calibration(
            tally, ["E1"], frame_blind_ids=set(), recall_min_denominator=0
        )
        rep = diag.entry_reports["E1"]
        assert "under-detected" in rep.reason.lower() or "under_detected" in rep.reason.lower()

    def test_tp_zero_total_zero_is_not_under_detected(self) -> None:
        """TP=0 with total_in_sample=0 is NOT under-detected (no observations)."""
        # An entry with no recall data at all should have has_recall_data=False.
        # Simulate: entry not in recall_counts.
        tally = TallyResult(
            recall_counts={},
            precision_counts={("E1", "security"): PrecisionTally(5, 1, 6)},
            rollup_counts={},
            total_coded=6,
            amendments_applied=0,
        )
        _, diag = compute_calibration(
            tally, ["E1"], frame_blind_ids=set(), recall_min_denominator=0
        )
        rep = diag.entry_reports["E1"]
        # No recall data → not under-detected (nothing in sample to detect)
        assert rep.under_detected is False


# ---------------------------------------------------------------------------
# 4. Uniform recall floor (epsilon > 0) clips sampled recall ≥ epsilon
# ---------------------------------------------------------------------------


class TestRecallFloor:
    """Uniform floor ε applied to sampled recall inside the NUTS model."""

    def test_floor_math_invariant(self) -> None:
        """jnp.maximum(recall_sample, epsilon) clips values below epsilon."""
        recall_samples = jnp.array([0.0, 1e-9, 0.005, 0.01, 0.5, 1.0])
        epsilon = 0.01
        floored = jnp.maximum(recall_samples, epsilon)
        assert float(floored[0]) == pytest.approx(epsilon)
        assert float(floored[1]) == pytest.approx(epsilon)
        assert float(floored[2]) == pytest.approx(epsilon)
        assert float(floored[3]) == pytest.approx(epsilon)  # exactly epsilon
        assert float(floored[4]) == pytest.approx(0.5)
        assert float(floored[5]) == pytest.approx(1.0)

    def test_floor_epsilon_zero_is_noop(self) -> None:
        """jnp.maximum(x, 0.0) must be identical to x for non-negative values."""
        recall_samples = jnp.array([0.0, 1e-9, 0.5, 1.0])
        floored = jnp.maximum(recall_samples, 0.0)
        np.testing.assert_array_equal(np.asarray(floored), np.asarray(recall_samples))

    def test_run_inference_accepts_recall_floor_epsilon_param(self) -> None:
        """run_inference must accept recall_floor_epsilon with default 0.0."""
        sig = inspect.signature(run_inference)
        assert "recall_floor_epsilon" in sig.parameters, (
            "run_inference must have a recall_floor_epsilon parameter"
        )
        default = sig.parameters["recall_floor_epsilon"].default
        assert default == 0.0, (
            f"recall_floor_epsilon default must be 0.0, got {default!r}"
        )

    def test_run_inference_default_path_unchanged(self) -> None:
        """With recall_floor_epsilon=0.0 (default), the model must not change
        the computation graph — verified by checking no extra JAX op is added
        for the default path (Python-level branch check)."""
        # Read source to verify the branch is Python-level (not jnp.where).
        # This is a structural test: the floor must be guarded by a Python if,
        # not a JAX conditional, so default path is byte-identical.
        src = inspect.getsource(run_inference)
        # The floor must be applied conditionally (Python if), not unconditionally.
        assert "recall_floor_epsilon > 0" in src or "recall_floor_epsilon > 0.0" in src, (
            "The recall floor must be guarded by a Python-level if "
            "so default path is byte-identical"
        )


# ---------------------------------------------------------------------------
# 5. F6 is recall-only: precision + rollup posteriors unchanged; empirical prior fires
# ---------------------------------------------------------------------------


class TestRecallOnlyIsolation:
    """F6 changes must not touch precision or rollup posteriors."""

    def _multi_entry_tally(self) -> TallyResult:
        """Two entries: E1 with good data (adequate), E2 with thin recall."""
        return TallyResult(
            recall_counts={
                ("E1", "security"): RecallTally(50, 5, 55),
                ("E2", "security"): RecallTally(2, 1, 3),
            },
            precision_counts={
                ("E1", "security"): PrecisionTally(50, 5, 55),
                ("E2", "security"): PrecisionTally(50, 5, 55),
            },
            rollup_counts={},
            total_coded=100,
            amendments_applied=0,
        )

    def test_precision_posteriors_identical_f6_on_vs_off(self) -> None:
        """Precision posteriors are byte-identical regardless of K > 0."""
        tally = self._multi_entry_tally()
        cal_off, _ = compute_calibration(
            tally, ["E1", "E2"], frame_blind_ids=set(), recall_min_denominator=0
        )
        cal_on, _ = compute_calibration(
            tally, ["E1", "E2"], frame_blind_ids=set(), recall_min_denominator=100
        )
        assert cal_off.precision == cal_on.precision

    def test_rollup_posteriors_identical_f6_on_vs_off(self) -> None:
        """Rollup precision posteriors are byte-identical regardless of K > 0."""
        tally = TallyResult(
            recall_counts={("E1", "security"): RecallTally(50, 5, 55)},
            precision_counts={},
            rollup_counts={("E1", "security"): PrecisionTally(50, 5, 55)},
            total_coded=55,
            amendments_applied=0,
        )
        cal_off, _ = compute_calibration(
            tally, ["E1"], frame_blind_ids=set(), recall_min_denominator=0
        )
        cal_on, _ = compute_calibration(
            tally, ["E1"], frame_blind_ids=set(), recall_min_denominator=100
        )
        assert cal_off.precision == cal_on.precision

    def test_empirical_precision_prior_fires_on_unit_cells(self) -> None:
        """apply_empirical_precision_prior must still fire on (1.0,1.0) cells
        even when F6 is on (K > 0).

        The prior fires when a non-frame-blind entry has a Beta(1,1) precision
        posterior (TP=0, FP=0 → from_counts(0,0)=Beta(1,1)). It shifts that
        entry's posterior to the mean of measured entries' posteriors.
        """
        # E1: precision-measured (TP=8, FP=2 → Beta(9,3))
        # E2: precision entry with zero observations (TP=0, FP=0 → Beta(1,1))
        #     → empirical prior from E1 must update it to Beta(9,3)
        tally = TallyResult(
            recall_counts={("E1", "security"): RecallTally(10, 1, 11)},
            precision_counts={
                ("E1", "security"): PrecisionTally(8, 2, 10),
                ("E2", "security"): PrecisionTally(0, 0, 0),
            },
            rollup_counts={},
            total_coded=21,
            amendments_applied=0,
        )
        cal_on, _ = compute_calibration(
            tally, ["E1", "E2"],
            frame_blind_ids=set(),
            recall_min_denominator=100,  # F6 fully on
        )
        e2_prec = cal_on.precision.get(("E2", "security"))
        assert e2_prec is not None, "E2 should have a precision posterior"
        # After empirical prior it should not be Beta(1,1) any more
        assert not (math.isclose(e2_prec.alpha, 1.0) and math.isclose(e2_prec.beta, 1.0)), (
            "apply_empirical_precision_prior must have fired on E2's (1,1) cell"
        )
        # It should now match E1's posterior: Beta(9,3)
        assert math.isclose(e2_prec.alpha, 9.0) and math.isclose(e2_prec.beta, 3.0), (
            f"Expected Beta(9,3) from empirical prior, "
            f"got alpha={e2_prec.alpha}, beta={e2_prec.beta}"
        )

    def test_recall_posteriors_changed_by_f6_params_only_if_relevant(self) -> None:
        """Recall posteriors from from_counts are NOT changed by F6 (no widening)."""
        tally = self._multi_entry_tally()
        cal_off, _ = compute_calibration(
            tally, ["E1", "E2"], frame_blind_ids=set(), recall_min_denominator=0
        )
        cal_on, _ = compute_calibration(
            tally, ["E1", "E2"], frame_blind_ids=set(), recall_min_denominator=100
        )
        # Recall posteriors must be byte-identical — F6 does NOT touch them.
        assert cal_off.recall == cal_on.recall, (
            "F6 must not modify recall Beta posteriors (detect+flag only, no regularization)"
        )
