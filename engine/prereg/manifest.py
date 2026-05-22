"""Pre-registration manifest — hash-locked before any concordance number exists.

HANDOFF §6 control 1: the manifest is frozen and hash-locked so that
no parameter can change between pre-registration and analysis.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass

from engine.prereg.rubric_attestation import RubricDraftingAttestation
from engine.prereg.signoff import ReviewerSignoff


def _dc_to_dict(obj: object) -> object:
    """Recursively convert dataclasses, tuples, and other values to JSON-safe types."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {
            k: _dc_to_dict(v)
            for k, v in dataclasses.asdict(obj).items()
        }
    if isinstance(obj, tuple):
        return [_dc_to_dict(item) for item in obj]
    return obj


@dataclass(frozen=True, slots=True)
class PreregManifest:
    """Frozen record of all pre-registered analysis parameters."""

    engine_version: str
    engine_version_range_min: str
    engine_version_range_max: str
    cycle_id: str
    taxonomy_hash: str
    snapshot_hash: str
    primary_spec: str  # e.g., "negative_binomial_per_stratum"
    robustness_specs: tuple[str, ...]
    flag_threshold_tau: float
    statistic: str  # e.g., "weighted_cohens_kappa"
    measurability_minimum: int
    prior_scale: float  # HalfNormal scale for lambda_e (default 0.5)
    concentration_shape: float  # Gamma shape for NB concentration (default 5.0)
    concentration_rate: float  # Gamma rate for NB concentration (default 0.1)
    ess_fraction: float  # ESS check fraction (default 0.4)
    meaningful_kappa_n: int  # minimum measurable entries for kappa (default 4)
    prng_seed: int
    confidence_threshold: float  # Stage-1 classifier confidence threshold (default 0.3)
    rubric_drafting_attestation: RubricDraftingAttestation | None
    rubric_reviewer: ReviewerSignoff | None
    statistical_reviewer: ReviewerSignoff | None
    classifier_rule_hash: str | None  # hash of Stage-1 classifier rules
    rubric_hash: str | None  # hash of frozen rubric (Plan 3)
    post_hoc_register_path: str | None  # path to Merkle-chained register
    rollup_threshold: float = 0.01
    rollup_p_supported: float = 0.8
    rollup_p_contradicted: float = 0.2

    @property
    def non_publishable(self) -> bool:
        """Derive non-publishable status from reviewer presence."""
        if self.rubric_reviewer is None or self.statistical_reviewer is None:
            return True
        if self.rubric_reviewer.viewed_results_before_signoff:
            return True
        return self.statistical_reviewer.viewed_results_before_signoff

    def to_dict(self) -> dict[str, object]:
        """Convert to a dict suitable for JSON serialization and hashing.

        Recursively converts nested dataclasses and tuples.  Use
        ``json.dumps(manifest.to_dict(), sort_keys=True, separators=(",", ":"))``
        for canonical hashing.
        """
        result: dict[str, object] = {}
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            result[field.name] = _dc_to_dict(value)
        return result

    def to_json(self) -> str:
        """Return the canonical JSON string used for hashing."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
