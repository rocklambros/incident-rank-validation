"""Integration test for run_oracle over a synthetic cycle dir (Plan 8d Task 6)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from engine.verify.check import run_oracle


def _write_cycle(tmp: Path) -> Path:
    cycle = tmp / "cycle"
    (cycle / "infer").mkdir(parents=True)
    (cycle / "results").mkdir(parents=True)
    (cycle / "classify").mkdir(parents=True)

    entry_ids = ["A", "B", "C", "D"]
    # A>B>C>D incidence: descending lambda, single stratum size 10
    rng = np.random.default_rng(0)
    centers = {"A": 0.8, "B": 0.6, "C": 0.4, "D": 0.2}
    cols = [np.clip(rng.normal(centers[e], 0.01, 400), 1e-6, None) for e in entry_ids]
    lam = np.column_stack(cols)
    np.save(cycle / "infer" / "lambda_samples.npy", lam)
    np.save(cycle / "infer" / "robustness_poisson_flat_lambda.npy", lam)
    (cycle / "infer" / "inference_summary.json").write_text(
        json.dumps({"entry_ids": entry_ids})
    )

    # labeled incidents: every entry in stratum 'security', 10 docs
    labeled = [{"entry_id": e, "stratum": "security"} for e in entry_ids for _ in range(10)]
    (cycle / "classify" / "labeled_incidents.json").write_text(json.dumps(labeled))

    # engine incidence ranking (matches the lambda order)
    (cycle / "results" / "incidence_ranking.json").write_text(
        json.dumps({"ranking": ["A", "B", "C", "D"]})
    )
    # robustness spread carrying a hierarchical sigma_u
    (cycle / "results" / "robustness_spread.json").write_text(
        json.dumps(
            {
                "primary": {"spec_name": "kappa", "sigma_u": None},
                "robustness": [
                    {"spec_name": "hierarchical_pooling", "sigma_u": 0.9},
                ],
            }
        )
    )
    # ballots: all respondents rank A>B>C>D
    rankings = np.tile(np.array([1.0, 2.0, 3.0, 4.0]), (12, 1))
    np.save(cycle / "results" / "vote_rankings.npy", rankings)
    (cycle / "results" / "vote_entry_ids.json").write_text(json.dumps(entry_ids))
    (cycle / "results" / "vote_plackett_luce.json").write_text(
        json.dumps({"ranking": ["A", "B", "C", "D"]})
    )
    return cycle


def test_run_oracle_all_pass(tmp_path: Path) -> None:
    cycle = _write_cycle(tmp_path)
    verdict = run_oracle(cycle)
    names = {d.name: d.status for d in verdict.deliverables}
    assert names["incidence"] == "PASS"
    assert names["plackett_luce"] == "PASS"
    assert names["sigma_u"] == "PASS"
    assert verdict.provisional is False
    # report written
    report = json.loads((cycle / "results" / "oracle_report.json").read_text())
    assert report["provisional"] is False
    assert len(report["deliverables"]) == 3


def test_run_oracle_flags_provisional_on_bad_engine_ranking(tmp_path: Path) -> None:
    cycle = _write_cycle(tmp_path)
    # corrupt the engine incidence ranking -> oracle disagrees -> FAIL
    (cycle / "results" / "incidence_ranking.json").write_text(
        json.dumps({"ranking": ["D", "C", "B", "A"]})
    )
    verdict = run_oracle(cycle)
    names = {d.name: d.status for d in verdict.deliverables}
    assert names["incidence"] == "FAIL"
    assert verdict.provisional is True


def test_run_oracle_skips_missing_sigma_u(tmp_path: Path) -> None:
    cycle = _write_cycle(tmp_path)
    (cycle / "infer" / "robustness_poisson_flat_lambda.npy").unlink()
    verdict = run_oracle(cycle)
    names = {d.name: d.status for d in verdict.deliverables}
    assert names["sigma_u"] == "SKIP"
