from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Threat:
    threat_id: str
    description: str
    mitigation: str
    residual_risk: str


THREATS: tuple[Threat, ...] = (
    Threat(
        "F1-ingestion-frame",
        "corpus sampling frame is blind to incidents that never become CVE/GHSA/OSV "
        "entries or harm-database rows",
        "declared as frame-blind in measurability map; staged frame-coverage audit extension",
        "entries invisible to the frame stay unmeasurable unless the audit runs",
    ),
    Threat(
        "F2-default-seed-contamination",
        "CVE ingest seeds every entry with LLM03; bare-default labels are contamination",
        "quarantine in adapter bias profile; gold-set contamination stratum measured, "
        "not blanket-dropped",
        "measured leak rate bounds but does not eliminate residual misclassification",
    ),
    Threat(
        "F3-classifier-blind-spots",
        "classifier is a heuristic with measured error, not a source of truth",
        "per-entry, per-stratum recall and precision from gold set; de-biased in posterior",
        "gold-set coverage on rare entries may be insufficient for tight bounds",
    ),
    Threat(
        "F4-structural-blind-spots",
        "owasp_llm distribution reflects classifier ruleset, not threat landscape",
        "frame-blind entries censored; classifier-blind entries bounded",
        "frame-coverage audit needed to upgrade unmeasurable entries",
    ),
    Threat(
        "F5-circular-atlas",
        "MITRE ATLAS labels are backfill from OWASP, not independent",
        "ATLAS not used as cross-check; native labels non-authoritative",
        "residual circularity if future maintainers re-introduce ATLAS as independent",
    ),
    Threat(
        "F-frame",
        "corpus A built by CVE/GHSA/OSV keyword crawl; deployed-app failures outside "
        "these sources never enter the sampling base",
        "frame-blind censoring; staged frame-coverage audit extension",
        "audit may never be built; affected entries stay unmeasurable (honest outcome)",
    ),
    Threat(
        "F-circ",
        "taxonomy-frame circularity: measuring a taxonomy against incidents classified "
        "by that taxonomy",
        "standing caveat in every report; transparency-first publication",
        "cannot be removed, only reframed and disclosed",
    ),
    Threat(
        "F-adversarial-ingestion",
        "public CVE/GHSA/OSV are open submission surfaces; descriptions are "
        "attacker-controlled; infer_attack_vector is pure regex",
        "snapshot + hash for reproducibility; drift detection + signoff; "
        "Stage-2 delimiter fencing",
        "a sufficiently targeted injection campaign could shift classifier output "
        "before detection",
    ),
    Threat(
        "F-defenseindepth",
        "the engine has many integrity controls; this can create false confidence that "
        "all bugs are upstream of the engine when debugging unexpected results",
        "treat the engine as a hypothesis, not a guarantee; SUCCESSOR-PRIMER warns "
        "future maintainers to check assumptions before checking the data",
        "cognitive trap; mitigated by explicit warning + open-source code review "
        "when public",
    ),
    Threat(
        "F-aiharm-precision",
        "ai-harm stratum has no direct precision measurements; ai-harm precision "
        "keys are absent from posteriors.json entirely, so the model falls back to "
        "Beta(1,1) = Uniform(0,1) via the default initialization in inference.py "
        "(apply_empirical_precision_prior cannot reach keys that do not exist)",
        "disclosed in notebook Acts 4, 5, and 10; the uniform prior is maximally "
        "uninformative rather than borrowed — it does not assume high or low "
        "precision for ai-harm entries",
        "ai-harm rankings rely on a flat precision prior (mean 0.5); true ai-harm "
        "precision could be substantially higher or lower, shifting rankings in "
        "either direction; only 3 of 20 ai-harm recall keys have material evidence "
        "(LLM09, LLM04, NEW-MA); NEW-WLA has only 1 observation above the pure prior",
    ),
)


def get_threats_register() -> tuple[Threat, ...]:
    return THREATS
