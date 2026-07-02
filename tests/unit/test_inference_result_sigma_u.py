import json
from pathlib import Path

import numpy as np

from engine.cli.pipeline_executor import write_robustness_artifacts
from engine.model.inference import InferenceResult


def test_inference_result_carries_sigma_u() -> None:
    r = InferenceResult(
        lambda_samples=np.zeros((4, 3)), entry_ids=("A", "B", "C"),
        r_hat={}, ess={}, divergences=0, num_warmup=1, num_samples=4,
        sigma_u=2.19,
    )
    assert r.sigma_u == 2.19


def test_write_robustness_artifacts_persists_sigma_u(tmp_path: Path) -> None:
    r = InferenceResult(
        lambda_samples=np.zeros((4, 3)), entry_ids=("A", "B", "C"),
        r_hat={}, ess={}, divergences=0, num_warmup=1, num_samples=4,
        sigma_u=2.19,
    )
    write_robustness_artifacts(r, tmp_path, "hierarchical_pooling")
    summary = json.loads((tmp_path / "robustness_hierarchical_pooling_summary.json").read_text())
    assert summary["sigma_u"] == 2.19
