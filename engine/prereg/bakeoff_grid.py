"""Load the locked RARR bake-off grid (config candidates + selection constants)."""
from __future__ import annotations

import json
from pathlib import Path

from engine.classify.bakeoff import ModelConfig


def _load(path: Path) -> dict[str, object]:
    data: dict[str, object] = json.loads(Path(path).read_text())
    return data


def load_bakeoff_grid(path: Path) -> list[ModelConfig]:
    """Return the 4 candidate ModelConfigs from the locked bakeoff_grid.json."""
    data = _load(path)
    configs = data["configs"]
    assert isinstance(configs, list)
    return [
        ModelConfig(
            name=str(c["name"]),
            model_id=str(c["model_id"]),
            revision_sha=str(c["revision_sha"]),
            gpu_type=str(c["gpu_type"]),
            gpu_count=int(c["gpu_count"]),
        )
        for c in configs
    ]


def load_grid_selection(path: Path) -> dict[str, object]:
    """Return the selection constants dict from the locked bakeoff_grid.json."""
    sel = _load(path)["selection"]
    assert isinstance(sel, dict)
    return sel
