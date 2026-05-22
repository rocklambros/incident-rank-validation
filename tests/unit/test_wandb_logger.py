from __future__ import annotations


def test_wandb_logger_noop_when_disabled() -> None:
    """WandB logger must be no-op when disabled."""
    from engine.monitoring.wandb_logger import WandBLogger

    logger = WandBLogger.create(project="test", enabled=False)
    logger.log_inference_start(num_warmup=100, num_samples=200, num_chains=4)
    logger.log_inference_result(
        r_hat={"lambda[0]": 1.001},
        ess={"lambda[0]": 800.0},
        divergences=0,
        wall_seconds=10.5,
    )
    logger.log_concordance(
        kappa_median=0.65,
        kappa_ci=(0.45, 0.82),
        measurable_count=8,
        total_count=10,
    )
    logger.log_stage2_cost(
        total_cost_usd=42.50,
        job_count=100,
        ceiling_usd=500.0,
    )
    logger.finish()


def test_wandb_logger_create_returns_noop_on_import_error() -> None:
    """If wandb is not installed, create() returns no-op logger."""
    import sys
    from unittest.mock import patch

    with patch.dict(sys.modules, {"wandb": None}):
        # Force reimport
        import importlib
        import engine.monitoring.wandb_logger as mod
        importlib.reload(mod)
        logger = mod.WandBLogger.create(project="test", enabled=True)
        assert not logger._enabled
