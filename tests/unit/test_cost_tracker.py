from __future__ import annotations

from engine.classify.cost_tracker import CostCeilingExceeded, CostTracker


class TestCostTracker:
    def test_initial_state(self) -> None:
        tracker = CostTracker(ceiling_usd=500.0)
        assert tracker.total_cost_usd == 0.0
        assert not tracker.ceiling_exceeded

    def test_record_cost(self) -> None:
        tracker = CostTracker(ceiling_usd=500.0)
        tracker.record(job_id="j1", cost_usd=10.0, execution_time_ms=5000.0)
        assert tracker.total_cost_usd == 10.0
        assert tracker.job_count == 1

    def test_ceiling_exceeded(self) -> None:
        tracker = CostTracker(ceiling_usd=100.0)
        tracker.record(job_id="j1", cost_usd=80.0, execution_time_ms=5000.0)
        assert not tracker.ceiling_exceeded
        tracker.record(job_id="j2", cost_usd=25.0, execution_time_ms=5000.0)
        assert not tracker.ceiling_exceeded  # 105 < 120 (1.2x ceiling)
        tracker.record(job_id="j3", cost_usd=20.0, execution_time_ms=5000.0)
        assert tracker.ceiling_exceeded  # 125 >= 120

    def test_check_or_abort_raises(self) -> None:
        tracker = CostTracker(ceiling_usd=10.0)
        tracker.record(job_id="j1", cost_usd=13.0, execution_time_ms=1000.0)
        try:
            tracker.check_or_abort()
            raise AssertionError("should raise")
        except CostCeilingExceeded as e:
            assert "13.0" in str(e) or "10.0" in str(e)

    def test_to_json(self) -> None:
        tracker = CostTracker(ceiling_usd=500.0)
        tracker.record(job_id="j1", cost_usd=10.0, execution_time_ms=5000.0)
        j = tracker.to_json()
        assert "ceiling_usd" in j
        assert "total_cost_usd" in j
        assert "jobs" in j

    def test_reconcile_with_billing(self) -> None:
        """R7: post-run billing reconciliation detects discrepancy."""
        tracker = CostTracker(ceiling_usd=500.0)
        tracker.record(job_id="j1", cost_usd=10.0, execution_time_ms=5000.0)
        tracker.record(job_id="j2", cost_usd=15.0, execution_time_ms=3000.0)
        rec = tracker.reconcile(billing_total_usd=28.50)
        assert rec["self_reported_usd"] == 25.0
        assert rec["billing_api_usd"] == 28.50
        assert rec["discrepancy_usd"] == 3.5
        assert rec["discrepancy_pct"] > 0

    def test_reconcile_flags_large_discrepancy(self) -> None:
        """R7: large discrepancy (>10%) is flagged."""
        tracker = CostTracker(ceiling_usd=500.0)
        tracker.record(job_id="j1", cost_usd=10.0, execution_time_ms=5000.0)
        rec = tracker.reconcile(billing_total_usd=20.0)
        assert rec["flagged"] is True
