"""Compute per-entry per-stratum Beta posteriors + calibration-adequacy diagnostic."""
from __future__ import annotations

from dataclasses import dataclass

from scipy import stats as scipy_stats

from engine.calibrate.beta import BetaPosterior, Calibration
from engine.calibrate.tally import TallyResult


@dataclass(frozen=True, slots=True)
class EntryCalibrationReport:
    entry_id: str
    has_precision_data: bool
    has_recall_data: bool
    precision_ci_width: float | None
    recall_ci_width: float | None
    recall_sample_size: int
    precision_sample_size: int
    min_fold_count: int
    flag: str
    reason: str
    # F6 recall-cell diagnostic flags (U2-1).
    # thin_denominator: True only when recall_min_denominator > 0 AND any recall
    #   cell for this entry has total_in_sample < recall_min_denominator.
    # under_detected: True when any recall cell has TP==0 AND total_in_sample > 0
    #   (caught none of N known).  Always computed for disclosure, but only
    #   BLOCKS adequacy when recall_min_denominator > 0 (F6 active).
    thin_denominator: bool
    under_detected: bool


@dataclass(frozen=True, slots=True)
class CalibrationDiagnostic:
    entries_with_both_frames: int
    entries_recall_only: int
    entries_no_data: int
    entry_reports: dict[str, EntryCalibrationReport]


def _ci_width(bp: BetaPosterior) -> float:
    lo, hi = scipy_stats.beta.ppf([0.05, 0.95], bp.alpha, bp.beta)
    return float(hi - lo)


def compute_calibration(
    tally: TallyResult,
    all_entry_ids: list[str],
    frame_blind_ids: set[str],
    n_folds: int = 5,
    classifier_entry_ids: set[str] | None = None,
    recall_min_denominator: int = 0,
) -> tuple[Calibration, CalibrationDiagnostic]:
    """Compute per-entry per-stratum Beta posteriors + calibration-adequacy diagnostic.

    Parameters
    ----------
    tally:
        Aggregated precision/recall counts from the coding batches.
    all_entry_ids:
        All entry IDs to produce reports for.
    frame_blind_ids:
        Entries with no second-frame data (receive empirical precision prior).
    n_folds:
        Number of CV folds for the min-fold-count diagnostic.
    classifier_entry_ids:
        If provided, entries not in this set receive 'no-data: no-classifier-rules'.
    recall_min_denominator:
        F6 thin-cell threshold K (default 0 = off).  When K > 0, a recall cell
        with total_in_sample < K is flagged thin_denominator=True and its
        adequacy label is forced to 'wide' (never 'adequate').
        Setting K=0 (default) preserves byte-identical behaviour with pre-F6
        output: NEITHER thin_denominator NOR under_detected blocks the adequacy
        decision when F6 is off.  Both fields are still computed for disclosure
        when K=0, but they do not alter the flag/reason.
    """
    recall_posteriors: dict[tuple[str, str], BetaPosterior] = {}
    precision_posteriors: dict[tuple[str, str], BetaPosterior] = {}

    for key, pt in tally.precision_counts.items():
        precision_posteriors[key] = BetaPosterior.from_counts(
            pt.true_positives, pt.false_positives,
        )

    for key, rt in tally.recall_counts.items():
        recall_posteriors[key] = BetaPosterior.from_counts(
            rt.true_positives, rt.false_negatives,
        )

    for key, pt_rollup in tally.rollup_counts.items():
        precision_posteriors[key] = BetaPosterior.from_counts(
            pt_rollup.true_positives, pt_rollup.false_positives,
        )

    cal = Calibration(recall=recall_posteriors, precision=precision_posteriors)

    updated_precision = apply_empirical_precision_prior(
        cal.precision, frame_blind_ids,
    )
    cal = Calibration(recall=cal.recall, precision=updated_precision)

    # ------------------------------------------------------------------
    # F6: build per-entry thin/under-detected flags from UN-widened counts.
    # thin: active only when recall_min_denominator > 0.
    # under_detected: ALWAYS computed (for disclosure), but only BLOCKS
    #   adequacy when recall_min_denominator > 0 (F6 active).
    # Default K=0 → _thin_entries empty AND _under_detected_entries does
    # NOT affect the adequacy decision → byte-identical to pre-F6 output.
    # ------------------------------------------------------------------
    _thin_entries: set[str] = set()
    _under_detected_entries: set[str] = set()
    for _key, _rt in tally.recall_counts.items():
        _eid_k = _key[0]
        if recall_min_denominator > 0 and _rt.total_in_sample < recall_min_denominator:
            _thin_entries.add(_eid_k)
        if _rt.true_positives == 0 and _rt.total_in_sample > 0:
            _under_detected_entries.add(_eid_k)

    both = 0
    recall_only = 0
    no_data = 0
    reports: dict[str, EntryCalibrationReport] = {}

    for eid in all_entry_ids:
        has_prec = any(k[0] == eid for k in precision_posteriors)
        has_rec = any(k[0] == eid for k in recall_posteriors)

        prec_size = sum(
            tally.precision_counts[k].total
            for k in tally.precision_counts if k[0] == eid
        )
        rec_size = sum(
            tally.recall_counts[k].total_in_sample
            for k in tally.recall_counts if k[0] == eid
        )

        prec_w: float | None = None
        rec_w: float | None = None
        if has_prec:
            widths = [
                _ci_width(precision_posteriors[k])
                for k in precision_posteriors if k[0] == eid
            ]
            prec_w = max(widths) if widths else None
        if has_rec:
            widths = [
                _ci_width(recall_posteriors[k])
                for k in recall_posteriors if k[0] == eid
            ]
            rec_w = max(widths) if widths else None

        min_count = (
            min(prec_size, rec_size)
            if has_prec and has_rec
            else (prec_size or rec_size)
        )
        min_fold = min_count // n_folds if n_folds > 0 else 0

        if eid in frame_blind_ids:
            flag = "no-data"
            reason = "no-data: frame-blind"
            no_data += 1
        elif classifier_entry_ids is not None and eid not in classifier_entry_ids:
            flag = "no-data"
            reason = "no-data: no-classifier-rules"
            no_data += 1
        elif not has_prec and not has_rec:
            flag = "no-data"
            reason = "no-data: no-classifier-positives"
            no_data += 1
        elif has_prec and has_rec:
            max_width = max(w for w in [prec_w, rec_w] if w is not None)
            # F6: when K>0, thin OR under-detected cells must NEVER be 'adequate'.
            # When K==0 (F6 off), neither thin nor under-detected blocks adequacy
            # → byte-identical to pre-F6 output.
            _f6_flagged = eid in _thin_entries or (
                recall_min_denominator > 0 and eid in _under_detected_entries
            )
            if max_width < 0.30 and not _f6_flagged:
                flag = "adequate"
                reason = "adequate"
            else:
                flag = "wide"
                if _f6_flagged:
                    _parts: list[str] = []
                    if eid in _thin_entries:
                        _parts.append("thin-denominator")
                    if eid in _under_detected_entries:
                        _parts.append("under-detected")
                    reason = f"wide: {'+'.join(_parts)} (n={min(prec_size, rec_size)})"
                else:
                    reason = f"wide: small-sample (n={min(prec_size, rec_size)})"
            both += 1
        else:
            flag = "wide"
            reason = "wide: recall-frame-only"
            recall_only += 1

        reports[eid] = EntryCalibrationReport(
            entry_id=eid,
            has_precision_data=has_prec,
            has_recall_data=has_rec,
            precision_ci_width=prec_w,
            recall_ci_width=rec_w,
            recall_sample_size=rec_size,
            precision_sample_size=prec_size,
            min_fold_count=min_fold,
            flag=flag,
            reason=reason,
            thin_denominator=eid in _thin_entries,
            under_detected=eid in _under_detected_entries,
        )

    diagnostic = CalibrationDiagnostic(
        entries_with_both_frames=both,
        entries_recall_only=recall_only,
        entries_no_data=no_data,
        entry_reports=reports,
    )

    return cal, diagnostic


def apply_empirical_precision_prior(
    precision: dict[tuple[str, str], BetaPosterior],
    frame_blind_ids: set[str],
) -> dict[tuple[str, str], BetaPosterior]:
    measured = {
        k: v for k, v in precision.items()
        if k[0] not in frame_blind_ids and (v.alpha != 1.0 or v.beta != 1.0)
    }
    if not measured:
        return dict(precision)

    mean_alpha = sum(bp.alpha for bp in measured.values()) / len(measured)
    mean_beta = sum(bp.beta for bp in measured.values()) / len(measured)

    result = dict(precision)
    for k, v in result.items():
        if k[0] in frame_blind_ids:
            continue
        if v.alpha == 1.0 and v.beta == 1.0:
            result[k] = BetaPosterior(alpha=mean_alpha, beta=mean_beta)

    return result
