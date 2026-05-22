"""RunPod cost tracking with auto-abort at ceiling (HANDOFF §7.5, M9)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field


class CostCeilingExceeded(RuntimeError):
    pass


@dataclass
class JobRecord:
    job_id: str
    cost_usd: float
    execution_time_ms: float


@dataclass
class CostTracker:
    ceiling_usd: float
    _abort_factor: float = 1.2
    _jobs: list[JobRecord] = field(default_factory=list)

    @property
    def total_cost_usd(self) -> float:
        return sum(j.cost_usd for j in self._jobs)

    @property
    def job_count(self) -> int:
        return len(self._jobs)

    @property
    def ceiling_exceeded(self) -> bool:
        return self.total_cost_usd >= self.ceiling_usd * self._abort_factor

    @property
    def total_execution_time_ms(self) -> float:
        return sum(j.execution_time_ms for j in self._jobs)

    def record(self, job_id: str, cost_usd: float, execution_time_ms: float) -> None:
        self._jobs.append(JobRecord(
            job_id=job_id,
            cost_usd=cost_usd,
            execution_time_ms=execution_time_ms,
        ))

    def check_or_abort(self) -> None:
        if self.ceiling_exceeded:
            raise CostCeilingExceeded(
                f"RunPod cost ceiling exceeded: ${self.total_cost_usd:.2f} "
                f">= ${self.ceiling_usd * self._abort_factor:.2f} "
                f"(ceiling ${self.ceiling_usd:.2f} x {self._abort_factor}). "
                f"Aborting Stage-2 classification."
            )

    def reconcile(self, billing_total_usd: float) -> dict[str, object]:
        """R7: post-run billing reconciliation against RunPod billing API."""
        self_reported = self.total_cost_usd
        discrepancy = abs(billing_total_usd - self_reported)
        discrepancy_pct = (discrepancy / max(self_reported, 0.01)) * 100.0
        return {
            "self_reported_usd": self_reported,
            "billing_api_usd": billing_total_usd,
            "discrepancy_usd": discrepancy,
            "discrepancy_pct": discrepancy_pct,
            "flagged": discrepancy_pct > 10.0,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "ceiling_usd": self.ceiling_usd,
            "abort_factor": self._abort_factor,
            "total_cost_usd": self.total_cost_usd,
            "job_count": self.job_count,
            "total_execution_time_ms": self.total_execution_time_ms,
            "ceiling_exceeded": self.ceiling_exceeded,
            "jobs": [
                {
                    "job_id": j.job_id,
                    "cost_usd": j.cost_usd,
                    "execution_time_ms": j.execution_time_ms,
                }
                for j in self._jobs
            ],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2) + "\n"
