"""Per-stratum bias profiles and quarantine predicates for the genai_agentic corpus.

Corpus A is a mixture (HANDOFF §3 Mixture paragraph, §4 Corpus-A-is-a-mixture row).
The ``corpus`` field defines primary strata: "security" (~7,350) and "ai-harm" (~364).
Each stratum carries a declared BiasProfile with its contamination description and
quarantine rule per HANDOFF §5.1.

Quarantine predicates fire on source-record native labels.  Quarantined records are
still emitted by the adapter as IncidentRecord instances (no ``quarantined`` field on
the schema) — contamination status is determined by calling ``is_bare_llm03_contaminated()``
or ``is_double_default_contaminated()`` on a record's ``native_labels``.  Downstream
pipeline stages (Plans 3–5) MUST call these predicates to route contaminated records
to the out-of-scope sink (HANDOFF §5.2).  (Premortem M8: contamination-status dependency.)

Note on strata vs. source_class (Premortem M7): the ``corpus`` field maps 1:1 to
``corpus_stratum`` on IncidentRecord.  The ``category`` field maps LOSSILY to
``source_class`` via ``_map_source_class()`` — multiple source categories collapse to
fewer engine source classes (e.g., "research" and "threat-report" both → "advisory").
The original category granularity is not preserved on IncidentRecord.  If Plans 3–5
need per-category stratification, the adapter must be extended to expose it.
"""
from __future__ import annotations

from engine.schema import BiasProfile


def _validate_bias_profile(profile: BiasProfile) -> BiasProfile:
    """Validate a BiasProfile at construction time (C2 pattern from Plan 1 M2)."""
    if not profile.stratum:
        raise ValueError("BiasProfile.stratum must be non-empty")
    if not profile.description:
        raise ValueError("BiasProfile.description must be non-empty")
    return profile


def build_bias_profiles() -> tuple[BiasProfile, ...]:
    """Construct per-stratum BiasProfile declarations.

    Construction-time validation (inherited constraint C2 from Plan 1 M2):
    each profile is validated at creation.  Invalid profiles raise ValueError.
    """
    return BIAS_PROFILES


BIAS_PROFILES: tuple[BiasProfile, ...] = (
    _validate_bias_profile(BiasProfile(
        stratum="security",
        description=(
            "Security-focused corpus (~7,350 records).  Built by CVE/GHSA/OSV keyword "
            "crawl plus harm-database ingestion (HANDOFF §3 F-frame).  Over-represents "
            "supply-chain and dependency vulnerabilities.  98.3% machine-labeled with no "
            "human OWASP review (F1).  ~907 bare-LLM03 default-seed entries (F2) contaminate "
            "this stratum."
        ),
        known_blind_spots=(
            "LLM04",
            "LLM08",
            "LLM10",
        ),
        contamination_description=(
            "ingest_cve_nvd_expanded.py seeds every CVE with ['LLM03'] / ['ASI04'] "
            "before refinement.  ~907 entries are bare ['LLM03'] (HANDOFF §3 F2).  "
            "In the actual corpus, ASI04 lives in owasp_asi, not owasp_llm, so the "
            "double-default count in owasp_llm is 0.  Treat CVE-class "
            "single-LLM03 as unknown, not supply chain."
        ),
        quarantine_rule=(
            "Quarantine records where owasp_llm == ['LLM03'] (bare default).  "
            "Double-default detection (sorted == ['ASI04', 'LLM03']) is retained "
            "as a safety net but fires on 0 records in the current corpus.  "
            "Quarantined records are emitted but flagged; downstream stages route "
            "them to the out-of-scope sink per HANDOFF §5.2."
        ),
    )),
    _validate_bias_profile(BiasProfile(
        stratum="ai-harm",
        description=(
            "AI-harm corpus (~364 records).  Drawn from harm-database ingestion, "
            "not CVE/GHSA/OSV.  Under-represents infrastructure and supply-chain "
            "incidents.  Different selection mechanism from the security stratum "
            "(HANDOFF §3 Mixture)."
        ),
        known_blind_spots=(
            "LLM04",
            "LLM08",
        ),
        contamination_description=(
            "Minimal direct contamination — harm reports are not CVE-seeded.  "
            "However, severity is defaulted to 'Medium' when missing in the source "
            "ingest, producing a zero-unknown-severity artifact (HANDOFF §3)."
        ),
        quarantine_rule=(
            "No bare-LLM03 quarantine needed for this stratum (not CVE-seeded).  "
            "Severity-default detection applies: records with source-defaulted "
            "'Medium' severity are emitted with severity=None."
        ),
    )),
)


def is_bare_llm03_contaminated(native_labels: list[str] | tuple[str, ...]) -> bool:
    """Return True if labels indicate bare-LLM03 default contamination (HANDOFF §3 F2)."""
    return list(native_labels) == ["LLM03"]


def is_double_default_contaminated(native_labels: list[str] | tuple[str, ...]) -> bool:
    """Return True if labels indicate the LLM03+ASI04 double-default (HANDOFF §3 F2)."""
    return sorted(native_labels) == ["ASI04", "LLM03"]
