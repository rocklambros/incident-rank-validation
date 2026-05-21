"""End-to-end test: classify -> sample -> code_synthetic -> tally -> calibrate -> cv-stability.

Uses the SyntheticAdapter's incidents with native_labels as ground truth.
Verifies the full pipeline produces valid Calibration and CVResult objects.
"""
from __future__ import annotations

from engine.adapters.synthetic import SyntheticAdapter
from engine.calibrate.batch import code_synthetic_with_ground_truth, generate_batch
from engine.calibrate.calibrate import compute_calibration
from engine.calibrate.cv import cross_validate_calibration
from engine.calibrate.sampler import SampleFrame, SampleRequest
from engine.calibrate.tally import tally_batches
from engine.calibrate.two_frame_sampler import TwoFrameSampler
from engine.classify.stub import classify_stub


class TestCalibrationE2E:
    def test_synthetic_pipeline(self) -> None:
        adapter = SyntheticAdapter(seed=42)
        incidents = tuple(adapter.iter_incidents())
        entries = adapter.entry_definitions()
        entry_ids = tuple(e.entry_id for e in entries)
        non_fb_ids = tuple(e.entry_id for e in entries if not e.frame_blind)
        fb_ids = {e.entry_id for e in entries if e.frame_blind}
        strata = sorted({inc.corpus_stratum for inc in incidents})

        # Stage 1: Classify (stub for synthetic)
        classification = classify_stub(incidents, entry_ids)

        # Stage 2: Sample
        sampler = TwoFrameSampler(classification_result=classification)
        incidents_list = list(incidents)
        incidents_by_id = {inc.id: inc for inc in incidents}

        batches = []

        # Precision-frame: for each non-frame-blind entry x stratum
        for eid in non_fb_ids:
            for s in strata:
                req = SampleRequest(
                    frame=SampleFrame.PRECISION,
                    entry_id=eid,
                    stratum=s,
                    n=40,
                )
                sr = sampler.draw(req, incidents_list, seed=42)
                if sr.actual_n > 0:
                    batch = generate_batch(
                        sample_result=sr,
                        rubric_hash="test",
                        manifest_lock_hash="test",
                        coder_id="synthetic",
                        cycle_id="test",
                    )
                    coded = code_synthetic_with_ground_truth(
                        batch, incidents_by_id=incidents_by_id, valid_entry_ids=set(entry_ids),
                    )
                    batches.append(coded)

        # Recall-frame: per stratum
        for s in strata:
            req = SampleRequest(
                frame=SampleFrame.RECALL,
                entry_id=None,
                stratum=s,
                n=100,
            )
            sr = sampler.draw(req, incidents_list, seed=42)
            batch = generate_batch(
                sample_result=sr,
                rubric_hash="test",
                manifest_lock_hash="test",
                coder_id="synthetic",
                cycle_id="test",
            )
            coded = code_synthetic_with_ground_truth(
                batch, incidents_by_id=incidents_by_id, valid_entry_ids=set(entry_ids),
            )
            batches.append(coded)

        # Stage 4: Tally
        tally = tally_batches(batches)
        assert tally.total_coded > 0

        # Stage 5: Calibrate
        cal, diag = compute_calibration(
            tally,
            all_entry_ids=list(entry_ids),
            strata=strata,
            frame_blind_ids=fb_ids,
        )
        assert len(cal.precision) > 0 or len(cal.recall) > 0
        assert diag.entries_with_both_frames + diag.entries_recall_only + diag.entries_no_data == len(entry_ids)

        # Stage 6: CV stability
        prec_labels: dict[tuple[str, str], list[bool]] = {}
        for batch in batches:
            if batch.header.frame == "precision" and batch.header.entry_id:
                key = (batch.header.entry_id, batch.header.stratum or "unknown")
                prec_labels.setdefault(key, [])
                for inc in batch.incidents:
                    if inc.labels is not None and batch.header.entry_id is not None:
                        prec_labels[key].append(batch.header.entry_id in inc.labels)

        rec_labels: dict[tuple[str, str], list[bool]] = {}
        for batch in batches:
            if batch.header.frame == "recall":
                for inc in batch.incidents:
                    if inc.labels is not None:
                        for eid in entry_ids:
                            key = (eid, batch.header.stratum or "unknown")
                            rec_labels.setdefault(key, [])
                            rec_labels[key].append(eid in inc.labels)

        cv = cross_validate_calibration(prec_labels, rec_labels, n_folds=5)
        assert cv.n_folds == 5
