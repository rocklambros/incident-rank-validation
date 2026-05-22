# tests/unit/test_rollup.py
from __future__ import annotations

import numpy as np

from engine.decide.rollup import (
    RollupResult,
    RollupVerdict,
    compute_rollup_subtest,
)


class TestRollupSubTest:
    def test_supported_when_child_has_distinct_cluster(self) -> None:
        parent_lambda = np.array([0.1, 0.08, 0.12, 0.09, 0.11])
        child_lambda = np.array([0.05, 0.04, 0.06, 0.05, 0.055])
        result = compute_rollup_subtest(
            parent_entry_id="LLM06",
            child_entry_id="rollup-mcp-tool",
            parent_lambda_samples=parent_lambda,
            child_lambda_samples=child_lambda,
            threshold=0.01,
        )
        assert isinstance(result, RollupResult)
        assert result.verdict in (RollupVerdict.SUPPORTED, RollupVerdict.CONTRADICTED, RollupVerdict.INDETERMINATE)

    def test_contradicted_when_child_is_negligible(self) -> None:
        parent_lambda = np.full(100, 0.15)
        child_lambda = np.full(100, 0.001)
        result = compute_rollup_subtest(
            parent_entry_id="LLM06",
            child_entry_id="rollup-tiny",
            parent_lambda_samples=parent_lambda,
            child_lambda_samples=child_lambda,
            threshold=0.01,
        )
        assert result.verdict == RollupVerdict.CONTRADICTED

    def test_indeterminate_when_wide_posterior(self) -> None:
        rng = np.random.default_rng(42)
        parent_lambda = rng.exponential(0.1, 100)
        child_lambda = rng.exponential(0.1, 100)
        result = compute_rollup_subtest(
            parent_entry_id="LLM06",
            child_entry_id="rollup-wide",
            parent_lambda_samples=parent_lambda,
            child_lambda_samples=child_lambda,
            threshold=0.01,
        )
        assert isinstance(result, RollupResult)
        assert result.parent_entry_id == "LLM06"
        assert result.child_entry_id == "rollup-wide"

    def test_result_has_probability(self) -> None:
        parent_lambda = np.full(50, 0.1)
        child_lambda = np.full(50, 0.05)
        result = compute_rollup_subtest(
            parent_entry_id="P", child_entry_id="C",
            parent_lambda_samples=parent_lambda,
            child_lambda_samples=child_lambda,
            threshold=0.01,
        )
        assert 0.0 <= result.p_distinct_cluster <= 1.0
