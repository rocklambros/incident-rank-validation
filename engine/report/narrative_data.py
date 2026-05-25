"""Data loading for the standalone narrative report generator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def load_narrative_data(cycle_dir: Path) -> dict[str, Any]:
    """Load all data files needed for the narrative report.

    Mirrors notebook Act 0 data loading.
    """
    cycle = cycle_dir.resolve()
    data: dict[str, Any] = {}

    with open(cycle / "prereg" / "rubric.json") as f:
        data["rubric"] = json.load(f)

    with open(cycle / "classify" / "labeled_incidents_multimodel.json") as f:
        data["incidents"] = json.load(f)

    prelabels = []
    with open(cycle / "calibration" / "llm_prelabels.jsonl") as f:
        for line in f:
            prelabels.append(json.loads(line))
    data["prelabels"] = prelabels

    goldset = []
    with open(cycle / "calibration" / "adjudicated_goldset.jsonl") as f:
        for line in f:
            goldset.append(json.loads(line))
    data["goldset"] = goldset

    precision_verif = []
    with open(cycle / "calibration" / "precision_verification.jsonl") as f:
        for line in f:
            precision_verif.append(json.loads(line))
    data["precision_verification"] = precision_verif

    with open(cycle / "calibration" / "posteriors.json") as f:
        data["posteriors"] = json.load(f)

    with open(cycle / "calibration" / "diagnostic.json") as f:
        data["diagnostic"] = json.load(f)

    with open(cycle / "infer" / "inference_summary.json") as f:
        data["inference_summary"] = json.load(f)

    data["lambda_samples"] = np.load(
        cycle / "infer" / "lambda_samples.npy",
        allow_pickle=False,
    )

    with open(cycle / "results" / "concordance.json") as f:
        data["concordance"] = json.load(f)

    with open(cycle / "results" / "selection_bias.json") as f:
        data["selection_bias"] = json.load(f)

    with open(cycle / "results" / "rank_comparison_report.md") as f:
        data["rank_comparison_md"] = f.read()

    report_md_path = cycle / "results" / "report.md"
    data["non_publishable"] = True  # safe default: assume non-publishable if report.md is missing
    if report_md_path.exists():
        report_text = report_md_path.read_text()
        data["non_publishable"] = "NON-PUBLISHABLE" in report_text

    corpus_b_path = cycle / "results" / "corpus_b_corroboration.json"
    if corpus_b_path.exists():
        with open(corpus_b_path) as f:
            data["corpus_b"] = json.load(f)

    data["entry_names"] = {
        e["entry_id"]: e["canonical_name"] for e in data["rubric"]["entries"]
    }
    data["entry_ids"] = data["inference_summary"]["entry_ids"]

    return data
