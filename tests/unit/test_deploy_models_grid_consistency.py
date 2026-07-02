from pathlib import Path

import tools.deploy_runpod as dep
from engine.prereg.bakeoff_grid import load_bakeoff_grid

GRID = Path("projects/owasp-llm/cycles/2026-rarr/prereg/bakeoff_grid.json")


def test_deploy_models_cover_grid_with_pinned_revisions() -> None:
    grid = {c.name: c for c in load_bakeoff_grid(GRID)}
    deployed = {m["name"]: m for m in dep.MODELS}
    assert set(grid).issubset(set(deployed)), "deploy MODELS missing a grid config"
    for name, cfg in grid.items():
        cmd = str(deployed[name]["vllm_cmd"])
        assert f"--revision {cfg.revision_sha}" in cmd, (
            f"{name}: vllm_cmd not pinned to grid SHA"
        )
        assert cfg.model_id in cmd
