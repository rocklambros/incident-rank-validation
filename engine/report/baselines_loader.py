"""Loader for the committed U3 baselines artifact consumed by U9.

Reads ``projects/owasp-llm/baselines/2026/rankings_baselines.json`` and
returns a typed structure exposing the keys U9 will read.  The loader is
intentionally thin — it just parses the JSON and validates the top-level
keys so that U9 cannot drift from the frozen schema.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PreviousRanking:
    kappa_median: float
    kappa_ci: list[float]
    ranking: list[str]


@dataclass(frozen=True)
class BareLambdaSensitivity:
    method_kappa_delta: float


@dataclass(frozen=True)
class ProspectivePower:
    n_required: int
    disclosure: str
    excludes_zero_at_current_n: bool


@dataclass(frozen=True)
class RankingsBaselines:
    """Typed view of rankings_baselines.json for U9 consumption."""

    previous_ranking: PreviousRanking
    bare_lambda_sensitivity: BareLambdaSensitivity
    measurable_subset_kappa: float
    prospective_power: ProspectivePower
    measurable_entry_ids: list[str]
    entry_ids: list[str]

    # Raw dict preserved so callers can access any additional field.
    _raw: dict[str, Any]


def load_rankings_baselines(
    path: Path | None = None,
) -> RankingsBaselines:
    """Load and validate the rankings_baselines.json artifact.

    Parameters
    ----------
    path:
        Explicit path to the JSON file.  If *None* the default location
        ``<repo_root>/projects/owasp-llm/baselines/2026/rankings_baselines.json``
        is used, resolved relative to this file's location in the repo tree.
    """
    if path is None:
        # engine/report/baselines_loader.py → go up 3 levels to repo root
        repo_root = Path(__file__).resolve().parents[2]
        path = (
            repo_root / "projects" / "owasp-llm" / "baselines" / "2026"
            / "rankings_baselines.json"
        )

    raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))

    # ---- previous_ranking ----
    pr = raw["previous_ranking"]
    previous_ranking = PreviousRanking(
        kappa_median=pr["kappa_median"],
        kappa_ci=pr["kappa_ci"],
        ranking=pr["ranking"],
    )

    # ---- bare_lambda_sensitivity ----
    bls = raw["bare_lambda_sensitivity"]
    bare_lambda_sensitivity = BareLambdaSensitivity(
        method_kappa_delta=bls["method_kappa_delta"],
    )

    # ---- secondary_measurable_subset kappa ----
    measurable_subset_kappa: float = raw["secondary_measurable_subset"]["measurable_kappa_median"]

    # ---- prospective_power ----
    pp = raw["prospective_power"]
    prospective_power = ProspectivePower(
        n_required=pp["n_required"],
        disclosure=pp["disclosure"],
        excludes_zero_at_current_n=pp["excludes_zero_at_current_n"],
    )

    # ---- entry lists ----
    measurable_entry_ids: list[str] = raw["measurable_entry_ids"]
    entry_ids: list[str] = raw["entry_ids"]

    return RankingsBaselines(
        previous_ranking=previous_ranking,
        bare_lambda_sensitivity=bare_lambda_sensitivity,
        measurable_subset_kappa=measurable_subset_kappa,
        prospective_power=prospective_power,
        measurable_entry_ids=measurable_entry_ids,
        entry_ids=entry_ids,
        _raw=raw,
    )
