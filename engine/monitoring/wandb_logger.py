"""Optional WandB logger for NUTS inference monitoring."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class WandBLogger:
    _enabled: bool = False
    _run: object = field(default=None, repr=False)

    @classmethod
    def create(
        cls,
        project: str = "incident-rank-validation",
        enabled: bool = True,
        cycle_id: str = "",
        tags: list[str] | None = None,
    ) -> WandBLogger:
        if not enabled:
            return cls(_enabled=False)
        try:
            import wandb

            run = wandb.init(
                project=project,
                config={"cycle_id": cycle_id},
                tags=tags or [],
                reinit=True,
            )
            return cls(_enabled=True, _run=run)
        except Exception:
            logger.warning("WandB initialization failed; continuing without monitoring")
            return cls(_enabled=False)

    def log_inference_start(
        self,
        num_warmup: int,
        num_samples: int,
        num_chains: int,
    ) -> None:
        if not self._enabled:
            return
        try:
            import wandb
            wandb.log({
                "inference/num_warmup": num_warmup,
                "inference/num_samples": num_samples,
                "inference/num_chains": num_chains,
            })
        except Exception:
            pass

    def log_inference_result(
        self,
        r_hat: dict[str, float],
        ess: dict[str, float],
        divergences: int,
        wall_seconds: float,
    ) -> None:
        if not self._enabled:
            return
        try:
            import wandb
            metrics: dict[str, object] = {
                "inference/divergences": divergences,
                "inference/wall_seconds": wall_seconds,
            }
            if r_hat:
                metrics["inference/max_r_hat"] = max(r_hat.values())
                metrics["inference/mean_r_hat"] = sum(r_hat.values()) / len(r_hat)
            if ess:
                metrics["inference/min_ess"] = min(ess.values())
                metrics["inference/mean_ess"] = sum(ess.values()) / len(ess)
            wandb.log(metrics)
        except Exception:
            pass

    def log_concordance(
        self,
        kappa_median: float | None,
        kappa_ci: tuple[float, float] | None,
        measurable_count: int,
        total_count: int,
    ) -> None:
        if not self._enabled:
            return
        try:
            import wandb
            metrics: dict[str, object] = {
                "concordance/measurable_count": measurable_count,
                "concordance/total_count": total_count,
            }
            if kappa_median is not None:
                metrics["concordance/kappa_median"] = kappa_median
            if kappa_ci is not None:
                metrics["concordance/kappa_ci_low"] = kappa_ci[0]
                metrics["concordance/kappa_ci_high"] = kappa_ci[1]
            wandb.log(metrics)
        except Exception:
            pass

    def log_stage2_cost(
        self,
        total_cost_usd: float,
        job_count: int,
        ceiling_usd: float,
    ) -> None:
        if not self._enabled:
            return
        try:
            import wandb
            wandb.log({
                "stage2/total_cost_usd": total_cost_usd,
                "stage2/job_count": job_count,
                "stage2/ceiling_usd": ceiling_usd,
                "stage2/utilization_pct": (
                    (total_cost_usd / ceiling_usd) * 100 if ceiling_usd > 0 else 0
                ),
            })
        except Exception:
            pass

    def finish(self) -> None:
        if not self._enabled:
            return
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass
