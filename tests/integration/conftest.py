"""Shared integration-test fixtures (F-C).

The ``real_minimal_cycle`` fixture is a factory that builds a minimal but
real-snapshot cycle tree under ``tmp_path``.  It exercises the REAL
``GenAIAgenticAdapter`` (via the snapshot file) and the REAL production writers
(``write_classify_coverage``, ``snapshot_hash``, ``hash_file``).  It NEVER
reads from or writes under ``projects/owasp-llm/cycles/2026/``.

Fixture is function-scoped (pytest default); every test gets a fresh tmp dir.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import pytest

# ── Incident table ─────────────────────────────────────────────────────────────
#
# id          date        corpus_label  role
# INC-01      2025-12-01  LLM01         clean in-scope
# INC-02      2025-12-02  LLM02         goldset-agree (consensus=LLM02, accept)
# INC-FLIP    2025-12-03  LLM01         recall-flip + FP for W (consensus=LLM02)
# INC-03      2025-12-04  LLM01         filler (≥2 LLM01); truncate_labeled drops this
# INC-OOS     2025-12-05  (absent)      genuine OOS, IN snapshot → n_oos
# INC-FUTURE  2026-05-21  (dropped)     date>pull_date 2026-05-20 → adapter drops
#
# corpus_incident_ids = 6 raw ids; in_scope_incident_ids = 4 labeled ids → n_oos=2

_INCIDENTS: list[dict[str, object]] = [
    {
        "id": "INC-01",
        "date": "2025-12-01",
        "title": "Prompt injection via user input",
        "description": "Attacker used prompt injection to bypass guardrails.",
        "impact": "Unauthorized action taken by the LLM.",
        "corpus": "security",
        "category": "real-world",
        "quality_tier": "reviewed",
        "severity": "High",
        "references": [{"url": "https://example.com/inc-01"}],
        "owasp_llm": ["LLM01"],
    },
    {
        "id": "INC-02",
        "date": "2025-12-02",
        "title": "Sensitive data disclosed via LLM output",
        "description": "An LLM disclosed PII via its completions.",
        "impact": "User PII exposed to third party.",
        "corpus": "security",
        "category": "real-world",
        "quality_tier": "reviewed",
        "severity": "High",
        "references": [{"url": "https://example.com/inc-02"}],
        "owasp_llm": ["LLM02"],
    },
    {
        "id": "INC-FLIP",
        "date": "2025-12-03",
        "title": "Data exfiltration via injected prompt",
        "description": "Classifier labelled as LLM01 but goldset consensus is LLM02.",
        "impact": "Sensitive information exfiltrated via prompt manipulation.",
        "corpus": "security",
        "category": "real-world",
        "quality_tier": "reviewed",
        "severity": "High",
        "references": [{"url": "https://example.com/inc-flip"}],
        "owasp_llm": ["LLM01"],
    },
    {
        "id": "INC-03",
        "date": "2025-12-04",
        "title": "Second prompt injection incident",
        "description": "Additional LLM01 incident for filler coverage.",
        "impact": "Minor unauthorized action via prompt injection.",
        "corpus": "security",
        "category": "real-world",
        "quality_tier": "reviewed",
        "severity": "Medium",
        "references": [{"url": "https://example.com/inc-03"}],
        "owasp_llm": ["LLM01"],
    },
    {
        "id": "INC-OOS",
        "date": "2025-12-05",
        "title": "Out-of-scope incident",
        "description": "An incident that is genuinely out-of-scope for this taxonomy.",
        "impact": "No LLM Top 10 classification applies.",
        "corpus": "security",
        "category": "real-world",
        "quality_tier": "reviewed",
        "severity": "Low",
        "references": [{"url": "https://example.com/inc-oos"}],
        "owasp_llm": [],
    },
    {
        "id": "INC-FUTURE",
        "date": "2026-05-21",
        "title": "Future incident after snapshot date",
        "description": "Incident dated after the snapshot pull date.",
        "impact": "Dropped by adapter date-filter (date > pull_date 2026-05-20).",
        "corpus": "security",
        "category": "real-world",
        "quality_tier": "reviewed",
        "severity": "High",
        "references": [{"url": "https://example.com/inc-future"}],
        "owasp_llm": ["LLM01"],
    },
]

_ALL_IDS: set[str] = {str(inc["id"]) for inc in _INCIDENTS}
_LABELED_IDS: set[str] = {"INC-01", "INC-02", "INC-FLIP", "INC-03"}


def _build_minimal_cycle(
    out_dir: Path,
    *,
    truncate_labeled: bool = False,
    ghost_recall_id: str | None = None,
    with_batches: bool = False,
    with_infer: bool = False,
    with_vote: bool = False,
) -> Path:
    """Write a minimal real-snapshot cycle tree to *out_dir* and return it.

    Parameters
    ----------
    out_dir:
        Absolute directory under which the cycle is created (must be absolute;
        typically ``tmp_path`` from pytest).
    truncate_labeled:
        When True, drop INC-03 from ``labeled_incidents.json`` only.  The
        corpus snapshot and coverage marker counts are UNCHANGED (n_in_scope
        stays 4) — this is the intentional setup for T4's raise test.
    ghost_recall_id:
        When set, append a goldset row for this id (NOT in corpus, NOT in
        labeled) — used by T5 to trigger the infer goldset-guard.
    with_batches:
        Stub flag; fleshed out in T4.
    with_infer:
        Stub flag; fleshed out in T6.
    with_vote:
        Stub flag; fleshed out in T6.
    """
    assert out_dir.is_absolute()

    cycle = out_dir / "cycle"
    cycle.mkdir(parents=True, exist_ok=True)

    # ── 1. incidents.json — written ONCE; hash computed AFTER ─────────────
    data: dict[str, object] = {"incident_count": 6, "incidents": _INCIDENTS}
    incidents_serialized = json.dumps(data, sort_keys=True)
    incidents_bytes = incidents_serialized.encode("utf-8")
    H = hashlib.sha256(incidents_bytes).hexdigest()

    snap_dir = cycle / "corpora" / "genai_agentic" / H
    snap_dir.mkdir(parents=True, exist_ok=True)
    # Write bytes directly so snapshot_hash(path) == H is guaranteed.
    (snap_dir / "incidents.json").write_bytes(incidents_bytes)

    # ── 2. Rubric ─────────────────────────────────────────────────────────
    from engine.calibrate.provenance import hash_file

    prereg = cycle / "prereg"
    prereg.mkdir(parents=True, exist_ok=True)

    rubric_dict: dict[str, object] = {
        "cycle_id": "2026fc",
        "version": "0.1.0",
        "entries": [
            {
                "entry_id": "LLM01",
                "canonical_name": "Prompt Injection",
                "in_scope": "Attacks where LLM input alters model behavior unintentionally.",
                "exclusions": ["Data exfiltration where injection is not primary → LLM02"],
                "boundary_rules": [
                    {
                        "adjacent_entry_id": "LLM02",
                        "rule": "Injection primary → LLM01; exfiltration primary → LLM02.",
                        "is_ambiguous": False,
                    }
                ],
                "positive_indicators": ["prompt injection", "jailbreak"],
                "negative_indicators": ["data exfiltration without prompt manipulation"],
                "co_occurrence_pairs": [["LLM01", "LLM02"]],
                "is_rollup_candidate": False,
                "rolled_into": None,
            },
            {
                "entry_id": "LLM02",
                "canonical_name": "Sensitive Information Disclosure",
                "in_scope": "Sensitive or private information exposed via LLM output.",
                "exclusions": ["Injection as primary mechanism → LLM01"],
                "boundary_rules": [
                    {
                        "adjacent_entry_id": "LLM01",
                        "rule": "Injection primary → LLM01; exfiltration primary → LLM02.",
                        "is_ambiguous": False,
                    }
                ],
                "positive_indicators": ["data exfiltration", "sensitive disclosure"],
                "negative_indicators": ["prompt injection as primary mechanism"],
                "co_occurrence_pairs": [["LLM01", "LLM02"]],
                "is_rollup_candidate": False,
                "rolled_into": None,
            },
        ],
    }
    rubric_path = prereg / "rubric.json"
    rubric_path.write_text(json.dumps(rubric_dict, sort_keys=True, indent=2) + "\n")
    rubric_h = hash_file(rubric_path)

    # ── 3. Manifest (schema_version=2 so overlap_min_fp is in the JSON) ───
    from engine.prereg.manifest import PreregManifest

    manifest_obj = PreregManifest(
        engine_version="0.1.0",
        engine_version_range_min="0.1.0",
        engine_version_range_max="0.2.0",
        cycle_id="2026fc",
        taxonomy_hash="aaa",
        snapshot_hash=H,
        primary_spec="negative_binomial_per_stratum",
        robustness_specs=(),
        flag_threshold_tau=0.8,
        statistic="weighted_cohens_kappa",
        measurability_minimum=10,
        prior_scale=0.5,
        concentration_shape=5.0,
        concentration_rate=0.1,
        ess_fraction=0.4,
        meaningful_kappa_n=4,
        prng_seed=42,
        confidence_threshold=0.3,
        rubric_drafting_attestation=None,
        rubric_reviewer=None,
        statistical_reviewer=None,
        classifier_rule_hash=None,
        rubric_hash=rubric_h,
        post_hoc_register_path=None,
        overlap_min_fp=1,
        schema_version=2,
    )
    # Byte-identical manifest.json and manifest.lock.
    manifest_content = json.dumps(manifest_obj.to_dict(), sort_keys=True, indent=2) + "\n"
    (prereg / "manifest.json").write_text(manifest_content)
    (prereg / "manifest.lock").write_text(manifest_content)

    # ── 4. Calibration posteriors (Beta(1,1) for LLM01 and LLM02) ─────────
    cal_dir = cycle / "calibration"
    cal_dir.mkdir(parents=True, exist_ok=True)

    posteriors: dict[str, object] = {
        "recall": {
            "LLM01::security": {"alpha": 1.0, "beta": 1.0},
            "LLM02::security": {"alpha": 1.0, "beta": 1.0},
        },
        "precision": {
            "LLM01::security": {"alpha": 1.0, "beta": 1.0},
            "LLM02::security": {"alpha": 1.0, "beta": 1.0},
        },
    }
    (cal_dir / "posteriors.json").write_text(json.dumps(posteriors, indent=2) + "\n")

    # ── 5. Adjudicated goldset ─────────────────────────────────────────────
    # INC-02: goldset-agree (classifier=LLM02, consensus=LLM02).
    # INC-FLIP: recall-flip (classifier=LLM01, consensus=LLM02) + FP for W.
    goldset_rows: list[str] = [
        json.dumps({
            "incident_id": "INC-02",
            "llm_consensus": "LLM02",
            "labels": ["LLM02"],
            "adjudicated": "accept",
        }),
        json.dumps({
            "incident_id": "INC-FLIP",
            "llm_consensus": "LLM02",
            "labels": ["LLM02"],
            "adjudicated": "accept",
        }),
    ]
    if ghost_recall_id is not None:
        goldset_rows.append(
            json.dumps({
                "incident_id": ghost_recall_id,
                "llm_consensus": "LLM01",
                "labels": ["LLM01"],
                "adjudicated": "accept",
            })
        )
    (cal_dir / "adjudicated_goldset.jsonl").write_text("\n".join(goldset_rows) + "\n")

    # ── 6. Labeled incidents ───────────────────────────────────────────────
    classify_dir = cycle / "classify"
    classify_dir.mkdir(parents=True, exist_ok=True)

    labeled_rows: list[dict[str, str]] = [
        {"incident_id": "INC-01", "entry_id": "LLM01", "stratum": "security"},
        {"incident_id": "INC-02", "entry_id": "LLM02", "stratum": "security"},
        {"incident_id": "INC-FLIP", "entry_id": "LLM01", "stratum": "security"},
        {"incident_id": "INC-03", "entry_id": "LLM01", "stratum": "security"},
    ]
    if truncate_labeled:
        # Drop INC-03 only from the labeled file; corpus + marker stay at 4/6.
        labeled_rows = [r for r in labeled_rows if r["incident_id"] != "INC-03"]

    (classify_dir / "labeled_incidents.json").write_text(
        json.dumps(labeled_rows, indent=2) + "\n"
    )

    # ── 7. Coverage marker — always 4 in-scope (independent of truncation) ─
    from engine.calibrate.coverage import write_classify_coverage

    write_classify_coverage(
        classify_dir,
        snapshot_hash=H,
        corpus_incident_ids=_ALL_IDS,
        in_scope_incident_ids=_LABELED_IDS,
    )

    # Conditional stubs — fleshed out in T4 (with_batches), T6 (with_infer,
    # with_vote).  Parameters are declared in the signature now so all
    # downstream tasks share the same fixture factory entry point.
    if with_batches:
        pass  # T4 will implement batch chain writing
    if with_infer:
        pass  # T6 will implement lambda_samples.npy + inference_summary.json
    if with_vote:
        pass  # T6 will implement vote/vote.xlsx

    return cycle


@pytest.fixture
def real_minimal_cycle() -> Callable[..., Path]:
    """Factory fixture: ``build(tmp_path, *, truncate_labeled=False, ...) -> Path``.

    Returns the builder callable.  Tests call it with their own ``tmp_path``::

        def test_foo(real_minimal_cycle, tmp_path):
            cycle = real_minimal_cycle(tmp_path)
    """
    return _build_minimal_cycle
