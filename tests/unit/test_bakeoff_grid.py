"""T2: Test the locked RARR bake-off grid loader."""
from pathlib import Path

from engine.classify.bakeoff import BAKEOFF_ALPHA, LOCKBOX_FRACTION, MIN_CELL, ModelConfig
from engine.prereg.bakeoff_grid import load_bakeoff_grid, load_grid_selection

GRID = Path("projects/owasp-llm/cycles/2026-rarr/prereg/bakeoff_grid.json")


def test_grid_parses_to_four_model_configs() -> None:
    configs = load_bakeoff_grid(GRID)
    assert [c.name for c in configs] == [
        "qwen3-235b", "llama-405b", "deepseek-v3", "mistral-large-2411"
    ]
    assert all(isinstance(c, ModelConfig) for c in configs)
    # every revision_sha is a 40-hex-char real commit id (never a placeholder)
    for c in configs:
        assert len(c.revision_sha) == 40 and all(
            ch in "0123456789abcdef" for ch in c.revision_sha
        )


def test_grid_selection_matches_module_constants() -> None:
    sel = load_grid_selection(GRID)
    assert sel["lockbox_fraction"] == LOCKBOX_FRACTION == 0.3
    assert sel["alpha"] == BAKEOFF_ALPHA == 0.05
    assert sel["min_cell"] == MIN_CELL == 5
    assert sel["seed"] == 42
    assert sel["winner_none_rule"] == "empty_eligible_set"
