"""Pure builder + loader for the RARR pre-registration manifest (science-identical to 2026)."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

from engine.prereg.manifest import PreregManifest


def load_manifest(path: Path) -> PreregManifest:
    """Load a PreregManifest from JSON. Round-trips BOTH the schema-1 2026 manifest
    (no schema_version key → defaults to 1) AND the schema-4 RARR manifest.
    Mirrors the pipeline's _load_manifest (pipeline_executor.py) — NO PreregManifest.from_dict
    exists; this is the verified loader. Zero shared-code change (lives in the new U5 module)."""
    data = json.loads(Path(path).read_text())
    field_names = {f.name for f in dataclasses.fields(PreregManifest)}
    filtered = {k: v for k, v in data.items() if k in field_names}
    if isinstance(filtered.get("robustness_specs"), list):
        filtered["robustness_specs"] = tuple(filtered["robustness_specs"])
    return PreregManifest(**filtered)


# Fields whose VALUE may legitimately differ between 2026 and RARR.
GOVERNANCE_OVERRIDE_FIELDS: tuple[str, ...] = (
    "cycle_id", "schema_version", "goldset_hash",
    "recall_min_denominator", "recall_min_denominator_gate",
    "recall_floor_epsilon", "recall_min_denominator_rationale",
    "prospective_power_target_kappa", "prospective_power_confidence_level",
    "prospective_power_1_minus_beta",
)
# Every other manifest field must be byte-identical to 2026 (the science-identity invariant).
SCIENTIFIC_FIELDS: tuple[str, ...] = tuple(
    f.name for f in dataclasses.fields(PreregManifest)
    if f.name not in GOVERNANCE_OVERRIDE_FIELDS
)

_K8_RATIONALE = (
    "K=8 flag-not-widen: recall cells with fewer than 8 in-sample goldset observations "
    "yield Beta-posterior recall estimates too imprecise to support a trustworthy "
    "lambda=observed/recall correction. Such cells are FLAGGED and disclosed (with a "
    "K in {6,8,10} sensitivity grid), NOT gated/dropped and NOT floored — the point "
    "estimate stays honest per the U2 flag-not-widen design; recall_floor_epsilon=0.0 "
    "keeps recall untouched (Bayesian posterior means are strictly positive, so no /0)."
)


def compute_goldset_hash(goldset_path: Path) -> str:
    return hashlib.sha256(Path(goldset_path).read_bytes()).hexdigest()


def build_rarr_manifest(base: PreregManifest, goldset_hash: str) -> PreregManifest:
    return dataclasses.replace(
        base,
        cycle_id="2026-rarr",
        schema_version=4,
        goldset_hash=goldset_hash,
        recall_min_denominator=8,
        recall_min_denominator_gate=False,
        recall_floor_epsilon=0.0,
        recall_min_denominator_rationale=_K8_RATIONALE,
        prospective_power_target_kappa=0.40,
        prospective_power_confidence_level=0.95,
        prospective_power_1_minus_beta=0.80,
    )
