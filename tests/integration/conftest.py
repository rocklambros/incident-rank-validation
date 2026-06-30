"""Shared integration-test fixtures (F-C).

The ``real_minimal_cycle`` fixture is a factory that builds a minimal but
real-snapshot cycle tree under ``tmp_path``.  It exercises the REAL
``GenAIAgenticAdapter`` (via the snapshot file) and the REAL production writers
(``write_classify_coverage``, ``snapshot_hash``, ``hash_file``).  It NEVER
reads from or writes under ``projects/owasp-llm/cycles/2026/``.

Fixture is function-scoped (pytest default); every test gets a fresh tmp dir.
"""
from __future__ import annotations

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
    from engine.snapshot.hashing import snapshot_hash

    data: dict[str, object] = {"incident_count": 6, "incidents": _INCIDENTS}
    incidents_serialized = json.dumps(data, sort_keys=True)
    incidents_bytes = incidents_serialized.encode("utf-8")

    # Write to temp file first, then compute hash using production snapshot_hash function
    corpora_dir = cycle / "corpora" / "genai_agentic"
    corpora_dir.mkdir(parents=True, exist_ok=True)
    temp_incidents_path = corpora_dir / "_incidents.tmp.json"
    temp_incidents_path.write_bytes(incidents_bytes)

    # Compute hash using production snapshot_hash function
    H = snapshot_hash(temp_incidents_path)

    # Create final snapshot directory and move temp file there
    snap_dir = corpora_dir / H
    snap_dir.mkdir(parents=True, exist_ok=True)
    temp_incidents_path.rename(snap_dir / "incidents.json")

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
        # ── T4: write a minimal VALID batch chain so cal_tally reaches the
        # completeness check.  Order matters: rubric.json and manifest.lock are
        # already written above; we hash them HERE so the batch header matches
        # what cal_tally recomputes from the CLI --rubric / --manifest args.
        import uuid
        from datetime import UTC, datetime

        from engine.calibrate.batch import BatchHeader, BatchIncident, CodingBatch
        from engine.calibrate.provenance import (
            StageProvenance,
            hash_json,
            write_provenance,
        )
        from engine.version import __version__ as _engine_version

        # hash_file was already imported in step 2 above (in-scope here).
        lock_h = hash_file(prereg / "manifest.lock")
        # rubric_h was computed in step 2 above.

        # sample_hash: any deterministic string — cal_tally reads it back
        # from the batch header itself, so expected == actual by construction.
        _sample_hash = hash_json(
            {"frame": "recall", "stratum": "security", "fixture": "minimal"}
        )

        _batch_id = str(uuid.uuid4())
        _header = BatchHeader(
            cycle_id="2026fc",
            batch_id=_batch_id,
            frame="recall",
            entry_id=None,
            stratum="security",
            sample_hash=_sample_hash,
            rubric_hash=rubric_h,          # must equal hash_file(rubric.json)
            manifest_lock_hash=lock_h,     # must equal hash_file(manifest.lock)
            coder_id="synthetic",
            generated_at=datetime.now(UTC).isoformat(),
        )

        # Build coded incidents from labeled_rows (possibly truncated).
        # labels must not be None so the incidents count as coded.
        _inc_text: dict[str, str] = {
            str(inc["id"]): f"{inc['title']}: {inc['description']}"
            for inc in _INCIDENTS
        }
        _batch_incidents = [
            BatchIncident(
                incident_id=row["incident_id"],
                text=_inc_text.get(row["incident_id"], row["incident_id"]),
                labels=[row["entry_id"]],
            )
            for row in labeled_rows
        ]

        _batch = CodingBatch(header=_header, incidents=_batch_incidents)
        batch_dir = cal_dir / "batches"
        batch_dir.mkdir(parents=True, exist_ok=True)
        _batch.write(batch_dir / f"{_batch_id}.json")

        # generate_batches_provenance.json — cal_tally reads this (read_provenance);
        # it does NOT verify chained hashes from it, so any valid StageProvenance works.
        _batches_meta = {
            "batch_count": 1,
            "coder_id": "synthetic",
            "rubric_hash": rubric_h,
            "manifest_lock_hash": lock_h,
        }
        _gen_prov = StageProvenance(
            stage_name="generate-batches",
            manifest_lock_hash=lock_h,
            input_hashes={"sample": hash_json({"stub": "minimal-fixture"})},
            output_hash=hash_json(_batches_meta),
            timestamp=datetime.now(UTC).isoformat(),
            engine_version=_engine_version,
        )
        write_provenance(_gen_prov, cal_dir / "generate_batches_provenance.json")

    if with_infer:
        import numpy as np

        infer_dir = cycle / "infer"
        infer_dir.mkdir(parents=True, exist_ok=True)
        # Clear-winner seed: LLM01 ~ Beta(5,1) median≈0.833; LLM02 ~ Beta(1,5)
        # median≈0.167.  Incidence = lambda * stratum_size; both entries are in
        # stratum "security" with size=4, so LLM01 incidence >> LLM02 — no tie,
        # no tiebreak ambiguity → D1 PASS.  Column order aligns with entry_ids.
        rng = np.random.default_rng(42)
        col0 = rng.beta(5, 1, size=(200, 1))   # LLM01
        col1 = rng.beta(1, 5, size=(200, 1))   # LLM02
        lambda_samples = np.concatenate([col0, col1], axis=1)
        np.save(infer_dir / "lambda_samples.npy", lambda_samples)
        inference_summary: dict[str, object] = {
            "entry_ids": ["LLM01", "LLM02"],
            "r_hat": {"LLM01": 1.01, "LLM02": 1.01},
            "ess": {"LLM01": 400.0, "LLM02": 400.0},
            "divergences": 0,
            "num_warmup": 1000,
            "num_samples": 200,
        }
        (infer_dir / "inference_summary.json").write_text(
            json.dumps(inference_summary, indent=2) + "\n"
        )

    if with_vote:
        from openpyxl import Workbook

        vote_dir = cycle / "vote"
        vote_dir.mkdir(parents=True, exist_ok=True)
        wb = Workbook()
        ws = wb.active
        ws.title = "Raw Results (Anonymized)"
        # Simple format (_load_simple_format): header[0] is the label column
        # (ignored by the loader); header[1:] are the entry_ids.  Subsequent
        # rows: [respondent_id, rank_LLM01, rank_LLM02].  All 3 respondents
        # rank LLM01=1 (best), LLM02=2 → unambiguous PL winner → D2 PASS.
        ws.append(["respondent", "LLM01", "LLM02"])
        ws.append(["R1", 1, 2])
        ws.append(["R2", 1, 2])
        ws.append(["R3", 1, 2])
        wb.save(vote_dir / "vote.xlsx")

    return cycle


@pytest.fixture
def real_minimal_cycle() -> Callable[..., Path]:
    """Factory fixture: ``build(tmp_path, *, truncate_labeled=False, ...) -> Path``.

    Returns the builder callable.  Tests call it with their own ``tmp_path``::

        def test_foo(real_minimal_cycle, tmp_path):
            cycle = real_minimal_cycle(tmp_path)
    """
    return _build_minimal_cycle
