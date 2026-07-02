"""Compute reproducible robustness-validation numbers for the RARR re-analysis cycle.

Produces a JSON summarising four validation dimensions:

1. Bakeoff balanced accuracy (loaded from bakeoff_crosscheck.json).
2. Ranking fidelity: Spearman ρ between each classifier's predicted class
   incidence and the adjudicated truth incidence (all non-OOS classes).
3. Ranking delta bootstrap CIs (B=3000, seed=42) for candidates whose
   point-estimate ρ exceeds the floor — checking they don't significantly beat it.
4. Corpus-reweighted Spearman ρ: post-stratify the goldset to match the
   corpus class mix; compute weighted ρ for floor and best frontier; bootstrap CI.
5. Recall/precision correction neg-L2: 5-fold CV (seed=11) estimates per-class
   recall and precision, applies Rogan-Gladen correction
   (corrected = obs × precision / max(recall, 0.05)), normalises corrected and
   truth to PROPORTION vectors on the same split, and compares them via negative
   L2; bootstrap CI (seed=42) for the ensemble-minus-floor delta (also on
   proportions).

Usage
-----
    python tools/compute_robustness_validation.py \\
        --cycle  projects/owasp-llm/cycles/2026-rarr \\
        --floor-path projects/owasp-llm/cycles/2026/classify/labeled_incidents.json \\
        --out    projects/owasp-llm/cycles/2026-rarr/results/robustness_validation.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.stats import rankdata

from engine.classify.bakeoff import load_bakeoff_truth
from engine.classify.bakeoff_inputs import compute_corpus_class_counts, load_floor_predictions

OOS_CLASS: str = "out-of-scope"
BOOTSTRAP_B: int = 3000
BOOTSTRAP_SEED: int = 42
CV_SEED: int = 11
CV_FOLDS: int = 5
RECALL_FLOOR: float = 0.05
CONFIGS: tuple[str, ...] = (
    "deepseek-v3",
    "llama-405b",
    "mistral-large-2411",
    "qwen3-235b",
)


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------


def _to_idx_array(
    ids: list[str],
    lookup: dict[str, str],
    class_idx: dict[str, int],
) -> npt.NDArray[np.intp]:
    """Map incident ids to class indices; -1 for OOS or any missing/unknown class."""
    return np.array(
        [class_idx.get(lookup.get(inc_id, ""), -1) for inc_id in ids],
        dtype=np.intp,
    )


def _count_vec(idx_arr: npt.NDArray[np.intp], k: int) -> npt.NDArray[np.float64]:
    """Count occurrences of each class index 0..k-1; -1 entries skipped."""
    valid = idx_arr[idx_arr >= 0]
    if len(valid) == 0:
        return np.zeros(k, dtype=np.float64)
    return np.bincount(valid, minlength=k).astype(np.float64)


def _count_mat(
    idx_mat: npt.NDArray[np.intp], k: int
) -> npt.NDArray[np.float64]:
    """Count per class for B bootstrap samples.

    Parameters
    ----------
    idx_mat: (B, n) integer array of class indices; -1 = skip.
    k: number of classes.

    Returns
    -------
    (B, k) float array of per-bootstrap per-class counts.
    """
    counts = np.zeros((idx_mat.shape[0], k), dtype=np.float64)
    for c in range(k):
        counts[:, c] = (idx_mat == c).sum(axis=1)
    return counts


def _proportion(vec: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Normalise a 1-D count vector to a proportion vector (sum→1; all-zero stays zero)."""
    total = float(vec.sum())
    return vec / total if total > 0.0 else vec


def _row_normalize(mat: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Normalise each row of a (B, K) matrix to a proportion (all-zero rows stay zero)."""
    sums = mat.sum(axis=1, keepdims=True)
    result: npt.NDArray[np.float64] = np.divide(
        mat, sums, out=np.zeros_like(mat), where=sums > 0.0
    )
    return result


def _spearman_vec(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> float:
    """Spearman ρ between two 1-D arrays."""
    r_a = rankdata(a, method="average")
    r_b = rankdata(b, method="average")
    da = r_a - r_a.mean()
    db = r_b - r_b.mean()
    denom = float(np.sqrt((da**2).sum() * (db**2).sum()))
    return float((da * db).sum() / denom) if denom > 0.0 else 0.0


def _spearman_batch(
    a_mat: npt.NDArray[np.float64], b_mat: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Vectorised Spearman ρ for B pairs of K-element rows.

    Parameters
    ----------
    a_mat, b_mat: (B, K) float arrays.

    Returns
    -------
    (B,) array of Spearman ρ values.
    """
    r_a: npt.NDArray[np.float64] = rankdata(a_mat, axis=1)
    r_b: npt.NDArray[np.float64] = rankdata(b_mat, axis=1)
    da = r_a - r_a.mean(axis=1, keepdims=True)
    db = r_b - r_b.mean(axis=1, keepdims=True)
    cov = (da * db).sum(axis=1)
    denom = np.sqrt((da**2).sum(axis=1) * (db**2).sum(axis=1))
    result: npt.NDArray[np.float64] = np.where(denom > 0.0, cov / denom, 0.0)
    return result


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def compute_robustness_validation(
    cycle_dir: Path, floor_path: Path
) -> dict[str, Any]:
    """Compute and return the robustness-validation dict.

    Parameters
    ----------
    cycle_dir:
        RARR cycle root (expected sub-paths: calibration/, classify/seq/,
        results/bakeoff_seq/).
    floor_path:
        Path to ``labeled_incidents.json`` (Stage-1 floor predictions and
        corpus class counts).

    Returns
    -------
    dict matching the robustness_validation.json schema.
    """
    cycle_dir = Path(cycle_dir)
    floor_path = Path(floor_path)

    # ------------------------------------------------------------------
    # 1. Load truth from the adjudicated goldset
    # ------------------------------------------------------------------
    goldset_path = cycle_dir / "calibration" / "adjudicated_goldset.jsonl"
    truth: dict[str, frozenset[str]] = load_bakeoff_truth(goldset_path)
    goldset_ids: list[str] = sorted(truth.keys())
    n_goldset = len(goldset_ids)

    # Primary class = alphabetically first label in the truth set; OOS excluded
    primary_class: dict[str, str] = {
        inc_id: sorted(ts)[0] for inc_id, ts in truth.items()
    }
    ranked_classes: list[str] = sorted(
        {pc for pc in primary_class.values() if pc != OOS_CLASS}
    )
    k = len(ranked_classes)
    class_idx: dict[str, int] = {c: i for i, c in enumerate(ranked_classes)}

    # Truth incidence vector (non-OOS primary classes)
    pc_idx_arr: npt.NDArray[np.intp] = np.array(
        [class_idx.get(primary_class[i], -1) for i in goldset_ids], dtype=np.intp
    )
    truth_vec: npt.NDArray[np.float64] = _count_vec(pc_idx_arr, k)

    # ------------------------------------------------------------------
    # 2. Load floor and model predictions
    # ------------------------------------------------------------------
    floor_preds: dict[str, str] = load_floor_predictions(floor_path)
    seq_dir = cycle_dir / "classify" / "seq"
    model_preds: dict[str, dict[str, str]] = {
        cfg: json.loads((seq_dir / f"predictions_{cfg}.json").read_text())
        for cfg in CONFIGS
    }

    # ------------------------------------------------------------------
    # 3. Build ensemble (4-vote majority; ties → deepseek-v3)
    # ------------------------------------------------------------------
    ensemble_preds: dict[str, str] = {}
    for inc_id in goldset_ids:
        votes: dict[str, int] = {}
        for cfg in CONFIGS:
            v = model_preds[cfg].get(inc_id)
            if v:
                votes[v] = votes.get(v, 0) + 1
        if not votes:
            continue
        max_cnt = max(votes.values())
        winners = [v for v, cnt in votes.items() if cnt == max_cnt]
        if len(winners) == 1:
            ensemble_preds[inc_id] = winners[0]
        else:
            dv = model_preds["deepseek-v3"].get(inc_id)
            ensemble_preds[inc_id] = (
                dv if (dv is not None and dv in winners) else winners[0]
            )

    all_classifiers: dict[str, dict[str, str]] = {
        "floor": floor_preds,
        **model_preds,
        "ensemble": ensemble_preds,
    }

    # ------------------------------------------------------------------
    # 4. Pre-compute per-classifier index arrays  (−1 = OOS/missing)
    # ------------------------------------------------------------------
    pred_idx: dict[str, npt.NDArray[np.intp]] = {
        name: _to_idx_array(goldset_ids, preds, class_idx)
        for name, preds in all_classifiers.items()
    }

    # ------------------------------------------------------------------
    # 5. Ranking fidelity: Spearman ρ vs truth for every classifier
    # ------------------------------------------------------------------
    ranking_fidelity: dict[str, float] = {
        name: _spearman_vec(truth_vec, _count_vec(pred_idx[name], k))
        for name in all_classifiers
    }
    floor_rho: float = ranking_fidelity["floor"]

    # ------------------------------------------------------------------
    # 6. Bootstrap ranking delta (B=3000, seed=42)
    #    Candidates = classifiers whose point-estimate ρ exceeds the floor
    # ------------------------------------------------------------------
    candidate_names: list[str] = [
        name
        for name in all_classifiers
        if name != "floor" and ranking_fidelity[name] > floor_rho
    ]

    rng_boot = np.random.default_rng(BOOTSTRAP_SEED)
    boot_idxs: npt.NDArray[np.intp] = rng_boot.integers(
        0, n_goldset, size=(BOOTSTRAP_B, n_goldset), dtype=np.intp
    )

    # Truth and floor count matrices (B, K)
    truth_boot_mat = _count_mat(pc_idx_arr[boot_idxs], k)
    floor_boot_mat = _count_mat(pred_idx["floor"][boot_idxs], k)
    floor_rho_boot = _spearman_batch(truth_boot_mat, floor_boot_mat)

    ranking_delta: dict[str, dict[str, Any]] = {}
    for name in candidate_names:
        cand_boot_mat = _count_mat(pred_idx[name][boot_idxs], k)
        deltas = _spearman_batch(truth_boot_mat, cand_boot_mat) - floor_rho_boot
        ranking_delta[name] = {
            "mean": float(np.mean(deltas)),
            "ci95": [
                float(np.percentile(deltas, 2.5)),
                float(np.percentile(deltas, 97.5)),
            ],
        }

    # ------------------------------------------------------------------
    # 7. Corpus-reweighted Spearman ρ
    #    Post-stratify goldset to corpus class mix, then recompute ρ
    # ------------------------------------------------------------------
    corpus_counts: dict[str, int] = compute_corpus_class_counts(floor_path)
    corpus_total: float = float(sum(corpus_counts.values()))
    goldset_total: float = float(truth_vec.sum())

    # Per-incident importance weight: w(i) = P_corpus(class) / P_goldset(class)
    weight_arr: npt.NDArray[np.float64] = np.zeros(n_goldset, dtype=np.float64)
    for i, inc_id in enumerate(goldset_ids):
        pc = primary_class[inc_id]
        if pc == OOS_CLASS:
            continue
        c_frac = corpus_counts.get(pc, 0) / corpus_total
        g_frac = float(truth_vec[class_idx[pc]]) / goldset_total if goldset_total > 0 else 0.0
        weight_arr[i] = c_frac / g_frac if g_frac > 0.0 else 0.0

    def _weighted_count_vec(name: str) -> npt.NDArray[np.float64]:
        counts = np.zeros(k, dtype=np.float64)
        pidx = pred_idx[name]
        for i in range(n_goldset):
            c = int(pidx[i])
            if c >= 0:
                counts[c] += weight_arr[i]
        return counts

    weighted_truth_vec: npt.NDArray[np.float64] = np.zeros(k, dtype=np.float64)
    for i in range(n_goldset):
        c = int(pc_idx_arr[i])
        if c >= 0:
            weighted_truth_vec[c] += weight_arr[i]

    weighted_rho_floor: float = _spearman_vec(
        weighted_truth_vec, _weighted_count_vec("floor")
    )
    frontier_weighted_rhos: dict[str, float] = {
        name: _spearman_vec(weighted_truth_vec, _weighted_count_vec(name))
        for name in all_classifiers
        if name != "floor"
    }
    best_frontier: str = max(
        frontier_weighted_rhos, key=lambda nm: frontier_weighted_rhos[nm]
    )

    # Bootstrap CI for corpus-reweight delta (re-init seed=42 → independent sequence)
    rng_cw = np.random.default_rng(BOOTSTRAP_SEED)
    boot_idxs_cw: npt.NDArray[np.intp] = rng_cw.integers(
        0, n_goldset, size=(BOOTSTRAP_B, n_goldset), dtype=np.intp
    )
    boot_w = weight_arr[boot_idxs_cw]  # (B, n)

    def _weighted_count_mat_from_boot(name: str) -> npt.NDArray[np.float64]:
        pidx = pred_idx[name]
        counts = np.zeros((BOOTSTRAP_B, k), dtype=np.float64)
        for c in range(k):
            mask = pidx[boot_idxs_cw] == c  # (B, n)
            counts[:, c] = (mask * boot_w).sum(axis=1)
        return counts

    def _weighted_truth_mat() -> npt.NDArray[np.float64]:
        counts = np.zeros((BOOTSTRAP_B, k), dtype=np.float64)
        for c in range(k):
            mask = pc_idx_arr[boot_idxs_cw] == c  # (B, n)
            counts[:, c] = (mask * boot_w).sum(axis=1)
        return counts

    wt_truth_mat = _weighted_truth_mat()
    wt_floor_mat = _weighted_count_mat_from_boot("floor")
    wt_bf_mat = _weighted_count_mat_from_boot(best_frontier)

    wt_floor_rho_boot = _spearman_batch(wt_truth_mat, wt_floor_mat)
    wt_bf_rho_boot = _spearman_batch(wt_truth_mat, wt_bf_mat)
    delta_cw = wt_bf_rho_boot - wt_floor_rho_boot

    corpus_reweight: dict[str, Any] = {
        "floor": weighted_rho_floor,
        "best_frontier": best_frontier,
        "best_frontier_rho": frontier_weighted_rhos[best_frontier],
        "delta_ci95": [
            float(np.percentile(delta_cw, 2.5)),
            float(np.percentile(delta_cw, 97.5)),
        ],
    }

    # ------------------------------------------------------------------
    # 8. Recall/precision correction neg-L2 (5-fold CV, seed=11)
    # ------------------------------------------------------------------
    rng_cv = np.random.default_rng(CV_SEED)

    # Stratified fold assignment: shuffle each class's indices then round-robin
    fold_of: npt.NDArray[np.intp] = np.full(n_goldset, -1, dtype=np.intp)
    by_class: dict[str, list[int]] = {}
    for i, inc_id in enumerate(goldset_ids):
        by_class.setdefault(primary_class[inc_id], []).append(i)
    for cls_indices in by_class.values():
        arr = np.array(cls_indices, dtype=np.intp)
        perm = rng_cv.permutation(len(arr))
        for rank, orig in enumerate(perm):
            fold_of[arr[int(orig)]] = rank % CV_FOLDS

    def _cv_corrected_negl2(name: str) -> float:
        """5-fold CV neg-L2 with per-class recall/precision correction.

        Corrected class counts (obs × precision / max(recall, floor)) and the
        held-out truth counts are each normalised to PROPORTION vectors on the
        test fold before the negative-L2 is taken, so the metric is on the same
        [0, 1] proportion scale as the truth mix (not raw counts).
        """
        pidx = pred_idx[name]
        total: float = 0.0
        for fold in range(CV_FOLDS):
            train_mask: npt.NDArray[np.bool_] = fold_of != fold
            test_mask: npt.NDArray[np.bool_] = fold_of == fold

            recall_cv = np.zeros(k, dtype=np.float64)
            prec_cv = np.zeros(k, dtype=np.float64)
            for c in range(k):
                true_c = pc_idx_arr == c
                pred_c = pidx == c
                tp = int((true_c & pred_c & train_mask).sum())
                fn = int((true_c & ~pred_c & train_mask).sum())
                fp = int((~true_c & pred_c & train_mask).sum())
                recall_cv[c] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                prec_cv[c] = tp / (tp + fp) if (tp + fp) > 0 else 0.0

            recall_clamped = np.maximum(recall_cv, RECALL_FLOOR)
            obs = _count_vec(pidx[test_mask], k)
            corrected_prop = _proportion(obs * prec_cv / recall_clamped)
            truth_prop = _proportion(_count_vec(pc_idx_arr[test_mask], k))
            diff = corrected_prop - truth_prop
            total += -float(np.dot(diff, diff))

        return total / CV_FOLDS

    def _raw_negl2(name: str) -> float:
        """Neg-L2 of raw (uncorrected) class proportions vs truth proportions."""
        obs_prop = _proportion(_count_vec(pred_idx[name], k))
        truth_prop = _proportion(truth_vec)
        diff = obs_prop - truth_prop
        return -float(np.dot(diff, diff))

    floor_raw: float = _raw_negl2("floor")
    ensemble_raw: float = _raw_negl2("ensemble")
    floor_cvcorrected: float = _cv_corrected_negl2("floor")
    ensemble_cvcorrected: float = _cv_corrected_negl2("ensemble")

    # Bootstrap CI for recall-correction delta using global correction factors
    def _global_correction(name: str) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Return (recall_clamped, precision) vectors over full goldset."""
        pidx = pred_idx[name]
        recall_g = np.zeros(k, dtype=np.float64)
        prec_g = np.zeros(k, dtype=np.float64)
        for c in range(k):
            true_c = pc_idx_arr == c
            pred_c = pidx == c
            tp = int((true_c & pred_c).sum())
            fn = int((true_c & ~pred_c).sum())
            fp = int((~true_c & pred_c).sum())
            recall_g[c] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            prec_g[c] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        return np.maximum(recall_g, RECALL_FLOOR), prec_g

    floor_recall, floor_prec = _global_correction("floor")
    ens_recall, ens_prec = _global_correction("ensemble")
    floor_cf: npt.NDArray[np.float64] = floor_prec / floor_recall
    ens_cf: npt.NDArray[np.float64] = ens_prec / ens_recall

    rng_rc = np.random.default_rng(BOOTSTRAP_SEED)
    boot_idxs_rc: npt.NDArray[np.intp] = rng_rc.integers(
        0, n_goldset, size=(BOOTSTRAP_B, n_goldset), dtype=np.intp
    )

    truth_rc_prop = _row_normalize(_count_mat(pc_idx_arr[boot_idxs_rc], k))
    floor_corr_prop = _row_normalize(
        _count_mat(pred_idx["floor"][boot_idxs_rc], k) * floor_cf
    )
    ens_corr_prop = _row_normalize(
        _count_mat(pred_idx["ensemble"][boot_idxs_rc], k) * ens_cf
    )

    floor_negl2_boot = -(((floor_corr_prop - truth_rc_prop) ** 2).sum(axis=1))
    ens_negl2_boot = -(((ens_corr_prop - truth_rc_prop) ** 2).sum(axis=1))
    delta_rc = ens_negl2_boot - floor_negl2_boot

    recall_correction: dict[str, Any] = {
        "floor_raw": floor_raw,
        "ensemble_raw": ensemble_raw,
        "floor_cvcorrected": floor_cvcorrected,
        "ensemble_cvcorrected": ensemble_cvcorrected,
        "delta_ensemble_minus_floor_ci95": [
            float(np.percentile(delta_rc, 2.5)),
            float(np.percentile(delta_rc, 97.5)),
        ],
    }

    # ------------------------------------------------------------------
    # 9. Bakeoff balanced accuracy (from bakeoff_crosscheck.json)
    # ------------------------------------------------------------------
    crosscheck_path = cycle_dir / "results" / "bakeoff_seq" / "bakeoff_crosscheck.json"
    crosscheck: dict[str, Any] = json.loads(crosscheck_path.read_text())
    bakeoff_ba: dict[str, float] = {"floor": float(crosscheck["floor_balanced_accuracy"])}
    for cfg_name, cfg_data in crosscheck["per_config"].items():
        bakeoff_ba[cfg_name] = float(cfg_data["lockbox_balanced_accuracy"])

    # ------------------------------------------------------------------
    # Assemble and return
    # ------------------------------------------------------------------
    return {
        "goldset_n": len(truth),
        "ranked_classes": ranked_classes,
        "bakeoff_balanced_accuracy": bakeoff_ba,
        "ranking_fidelity_spearman_vs_truth": ranking_fidelity,
        "ranking_delta_vs_floor_bootstrap": ranking_delta,
        "corpus_reweight_spearman_vs_truth": corpus_reweight,
        "recall_correction_negL2": recall_correction,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="compute_robustness_validation",
        description="Compute reproducible robustness-validation JSON for the RARR cycle.",
    )
    p.add_argument(
        "--cycle",
        required=True,
        type=Path,
        metavar="DIR",
        help="RARR cycle root (e.g. projects/owasp-llm/cycles/2026-rarr)",
    )
    p.add_argument(
        "--floor-path",
        required=True,
        type=Path,
        metavar="FILE",
        help="Path to labeled_incidents.json for floor predictions",
    )
    p.add_argument(
        "--out",
        required=True,
        type=Path,
        metavar="FILE",
        help="Output path for the robustness_validation.json artifact",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    args = _build_parser().parse_args(argv)
    result = compute_robustness_validation(args.cycle, args.floor_path)
    out_path: Path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {out_path}")
    print(f"  goldset_n                       : {result['goldset_n']}")
    ranked: list[str] = result["ranked_classes"]
    print(f"  ranked_classes                  : {len(ranked)}")
    rff: dict[str, float] = result["ranking_fidelity_spearman_vs_truth"]
    print(f"  floor ranking fidelity rho      : {rff['floor']:.4f}")
    cr: dict[str, Any] = result["corpus_reweight_spearman_vs_truth"]
    print(f"  corpus-reweighted floor rho     : {cr['floor']:.4f}")
    rc: dict[str, Any] = result["recall_correction_negL2"]
    print(f"  recall-correction delta CI95    : {rc['delta_ensemble_minus_floor_ci95']}")


if __name__ == "__main__":
    main()
