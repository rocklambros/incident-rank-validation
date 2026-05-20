"""Frame-blind release gate -- HANDOFF S6 control 8.

Verify that frame-blind entries are excluded from inference and
cannot receive a falsely low or falsely precise posterior.
"""

from __future__ import annotations

import pytest

from engine.calibrate.beta import BetaPosterior, Calibration
from engine.model.censoring import MeasurabilityVerdict, partition_entries
from engine.schema import EntryDefinition


@pytest.mark.slow
class TestFrameBlindGate:

    def test_frame_blind_excluded_from_inference(self) -> None:
        """Frame-blind entries partition to frame_blind, never measurable."""
        entries = (
            EntryDefinition(entry_id="VIS", name="Visible", frame_blind=False),
            EntryDefinition(entry_id="BLIND", name="Blind", frame_blind=True),
        )
        result = partition_entries(entries, calibration=None)
        assert "BLIND" in result.frame_blind
        assert "BLIND" not in result.measurable
        assert result.verdicts["BLIND"] == MeasurabilityVerdict.FRAME_BLIND_UNMEASURABLE

    def test_frame_blind_cannot_appear_in_measurable_partition(self) -> None:
        """Even with calibration, frame-blind stays frame-blind."""
        entries = (
            EntryDefinition(entry_id="VIS", name="Visible"),
            EntryDefinition(entry_id="BLIND", name="Blind", frame_blind=True),
        )
        # Even if we somehow had calibration for the blind entry, it stays blind
        cal = Calibration(
            recall={
                ("VIS", "s1"): BetaPosterior(80, 20),
                ("BLIND", "s1"): BetaPosterior(80, 20),
            },
            precision={
                ("VIS", "s1"): BetaPosterior(90, 10),
                ("BLIND", "s1"): BetaPosterior(90, 10),
            },
        )
        result = partition_entries(entries, calibration=cal)
        assert "BLIND" in result.frame_blind
        assert "BLIND" not in result.measurable
        assert "BLIND" not in result.classifier_blind

    def test_frame_blind_flag_cannot_exceed_tau(self) -> None:
        """A frame-blind entry's wide posterior cannot exceed tau_flag,
        so it is excluded from flags by construction -- HANDOFF S5.5."""
        # This is a structural test: since frame-blind entries are excluded
        # from run_inference's measurable_entries, they have no posterior
        # and thus cannot be flagged.
        entries = (
            EntryDefinition(entry_id="VIS", name="Visible"),
            EntryDefinition(entry_id="BLIND", name="Blind", frame_blind=True),
        )
        result = partition_entries(entries, calibration=None)
        # The measurable set for inference MUST exclude frame-blind
        assert "BLIND" not in result.measurable
        # Only measurable entries get posteriors; BLIND has no posterior -> no flag possible
