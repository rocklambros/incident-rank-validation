"""Oracle orchestration: load a cycle's artifacts, run D1/D2/D3, write report."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from engine.verify.oracle import (
    ORACLE_SIGMA_U_BAND,
    ORACLE_TAU_INCIDENCE,
    ORACLE_TAU_PL,
    OracleDeliverable,
    OracleVerdict,
    compare_ranking,
    compare_sigma_u,
    oracle_incidence_ranking,
    oracle_pl_ranking_mm,
    oracle_sigma_u_surrogate,
)


def _build_strata(
    labeled: list[dict[str, object]],
) -> tuple[dict[str, tuple[str, ...]], dict[str, int]]:
    entry_strata_sets: dict[str, set[str]] = defaultdict(set)
    stratum_counts: dict[str, int] = defaultdict(int)
    for item in labeled:
        eid = str(item.get("entry_id", ""))
        stratum = str(item.get("stratum", "default"))
        entry_strata_sets[eid].add(stratum)
        stratum_counts[stratum] += 1
    entry_strata = {e: tuple(sorted(ss)) for e, ss in entry_strata_sets.items()}
    stratum_sizes = {s: max(c, 1) for s, c in stratum_counts.items()}
    return entry_strata, stratum_sizes


def _hierarchical_sigma_u(spread: dict[str, object]) -> float | None:
    robustness = spread.get("robustness", [])
    if not isinstance(robustness, list):
        return None
    for spec in robustness:
        if isinstance(spec, dict) and spec.get("sigma_u") is not None:
            return float(spec["sigma_u"])
    return None


def _ranking_deliverable(
    name: str,
    engine_ranking: tuple[str, ...],
    oracle_ranking: tuple[str, ...],
    floor: float,
) -> OracleDeliverable:
    """Compare two rankings; FAIL (not crash) on entry-set mismatch.

    A verification oracle must FLAG an entry-set disagreement between the
    engine deliverable and what the oracle can reconstruct, rather than
    raising (kendall_tau rejects mismatched sets) or silently filtering it
    away (which would mask the inconsistency).
    """
    if set(engine_ranking) != set(oracle_ranking):
        return OracleDeliverable(
            name,
            "FAIL",
            "entry-set mismatch",
            f"engine={sorted(engine_ranking)} oracle={sorted(oracle_ranking)}",
        )
    return compare_ranking(name, engine_ranking, oracle_ranking, floor)


def run_oracle(cycle: Path) -> OracleVerdict:
    """Re-derive D1/D2/D3 independently and compare to the engine's output."""
    infer = cycle / "infer"
    results = cycle / "results"
    deliverables: list[OracleDeliverable] = []

    # --- D1: incidence ranking ---
    inc_path = results / "incidence_ranking.json"
    lam_path = infer / "lambda_samples.npy"
    summ_path = infer / "inference_summary.json"
    labeled_path = cycle / "classify" / "labeled_incidents.json"
    if inc_path.exists() and lam_path.exists() and summ_path.exists() and labeled_path.exists():
        engine_ranking = tuple(json.loads(inc_path.read_text())["ranking"])
        lam = np.load(lam_path, allow_pickle=False)
        entry_ids = tuple(json.loads(summ_path.read_text())["entry_ids"])
        labeled = json.loads(labeled_path.read_text())
        entry_strata, stratum_sizes = _build_strata(labeled)
        common = tuple(e for e in entry_ids if e in set(engine_ranking))
        oracle_ranking = oracle_incidence_ranking(
            lam[:, [entry_ids.index(e) for e in common]],
            common,
            entry_strata,
            stratum_sizes,
        )
        deliverables.append(
            _ranking_deliverable("incidence", engine_ranking, oracle_ranking, ORACLE_TAU_INCIDENCE)
        )
    else:
        deliverables.append(
            OracleDeliverable("incidence", "SKIP", "n/a", "missing inputs")
        )

    # --- D2: PL vote ranking ---
    pl_path = results / "vote_plackett_luce.json"
    ballots_path = results / "vote_rankings.npy"
    vote_ids_path = results / "vote_entry_ids.json"
    if pl_path.exists() and ballots_path.exists() and vote_ids_path.exists():
        engine_pl = tuple(json.loads(pl_path.read_text())["ranking"])
        ballots = np.load(ballots_path, allow_pickle=False)
        vote_ids = tuple(json.loads(vote_ids_path.read_text()))
        oracle_pl = oracle_pl_ranking_mm(ballots, vote_ids)
        deliverables.append(
            _ranking_deliverable("plackett_luce", engine_pl, oracle_pl, ORACLE_TAU_PL)
        )
    else:
        deliverables.append(
            OracleDeliverable("plackett_luce", "SKIP", "n/a", "missing inputs")
        )

    # --- D3: sigma_u surrogate ---
    spread_path = results / "robustness_spread.json"
    flat_path = infer / "robustness_poisson_flat_lambda.npy"
    engine_sigma = (
        _hierarchical_sigma_u(json.loads(spread_path.read_text()))
        if spread_path.exists()
        else None
    )
    if engine_sigma is not None and flat_path.exists():
        flat_lam = np.load(flat_path, allow_pickle=False)
        oracle_sigma = oracle_sigma_u_surrogate(flat_lam)
        deliverables.append(
            compare_sigma_u(engine_sigma, oracle_sigma, ORACLE_SIGMA_U_BAND)
        )
    elif engine_sigma is None:
        deliverables.append(
            OracleDeliverable(
                "sigma_u",
                "SKIP",
                "n/a",
                "no hierarchical sigma_u in robustness_spread "
                "(hierarchical_pooling spec not run this cycle)",
            )
        )
    else:
        deliverables.append(
            OracleDeliverable(
                "sigma_u", "SKIP", "n/a", "missing poisson_flat lambda samples"
            )
        )

    verdict = OracleVerdict(deliverables=tuple(deliverables))
    (results / "oracle_report.json").write_text(
        json.dumps(_verdict_to_dict(verdict), indent=2, sort_keys=True)
    )
    return verdict


def _verdict_to_dict(verdict: OracleVerdict) -> dict[str, object]:
    return {
        "provisional": verdict.provisional,
        "deliverables": [
            {"name": d.name, "status": d.status, "metric": d.metric, "detail": d.detail}
            for d in verdict.deliverables
        ],
    }
