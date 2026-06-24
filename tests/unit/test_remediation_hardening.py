"""Tests for Task 4 (F10) hardening fixes:
- manifest rejects hierarchical_pooling without sigma_u_hyperprior_scale
- write_robustness_artifacts rejects path-traversal spec_name
"""
from pathlib import Path

import numpy as np
import pytest

from engine.model.inference import InferenceResult
from tests.unit.test_prereg import _make_manifest


def test_manifest_rejects_hierarchical_without_sigma_u() -> None:
    with pytest.raises(ValueError, match="sigma_u_hyperprior_scale"):
        _make_manifest(
            schema_version=2,
            robustness_specs=("hierarchical_pooling",),
            sigma_u_hyperprior_scale=None,
        )


def test_manifest_accepts_hierarchical_with_sigma_u() -> None:
    _make_manifest(
        schema_version=2, robustness_specs=("hierarchical_pooling",),
        sigma_u_hyperprior_scale=1.0,
    )  # must not raise


def test_write_robustness_rejects_path_traversal(tmp_path: Path) -> None:
    from engine.cli.pipeline_executor import write_robustness_artifacts
    r = InferenceResult(
        lambda_samples=np.zeros((2, 2)), entry_ids=("A", "B"),
        r_hat={}, ess={}, divergences=0, num_warmup=1, num_samples=2,
    )
    with pytest.raises(ValueError, match="spec_name"):
        write_robustness_artifacts(r, tmp_path, "../evil")
