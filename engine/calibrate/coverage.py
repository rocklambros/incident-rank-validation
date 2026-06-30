"""Classify-completeness guard against the pinned corpus snapshot (RARR F-B).

Unit A's OOS policy (a) scores a goldset incident with no in-scope classifier
label as a recall MISS, encoding OOS as ABSENCE from labeled_incidents.json.
That makes absence overloaded: a genuinely truncated / partial classify run
(crash) is ALSO absence, and would silently corrupt recall (and the incidence
counts built from labeled_incidents.json) with no error.

This module restores a DISTINCT completeness guard keyed on the pinned corpus
snapshot (manifest.snapshot_hash -> corpora/<corpus>/<snapshot_hash>/
incidents.json) -- an authoritative record INDEPENDENT of the (truncatable)
classifier output, so it cannot be co-truncated.  The classify producer writes
a count-form completion marker (classify_coverage.json) certifying how many
snapshot incidents it issued a verdict for; verify_labeled_completeness()
reconciles that marker to the pinned snapshot and to labeled_incidents.json.

A PROVEN coverage gap RAISES (loud-fail posture, matching _verify_goldset_hash
and the present-but-broken-goldset guard).  A genuinely-OOS incident under a
complete marker does NOT raise -- policy (a) is preserved (it remains a miss).
When no pinned snapshot is resolvable (synthetic cycles use the sentinel
"synthetic-no-snapshot"; minimal fixtures have no corpora dir) this is a no-op:
there is no authoritative universe to reconcile against.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

COVERAGE_FILENAME = "classify_coverage.json"
_SYNTHETIC_SNAPSHOT = "synthetic-no-snapshot"


class LabeledIncidentsIncompleteError(RuntimeError):
    """labeled_incidents.json fails to reconcile to the pinned corpus snapshot."""


@dataclass(frozen=True, slots=True)
class ClassifyCoverage:
    snapshot_hash: str
    n_corpus: int
    n_in_scope: int
    n_oos: int


def write_classify_coverage(
    out_dir: Path,
    *,
    snapshot_hash: str,
    corpus_incident_ids: set[str],
    in_scope_incident_ids: set[str],
) -> ClassifyCoverage:
    """Record, at classify completion, the producer's coverage of the snapshot.

    Call this ONLY after the classify/predict phase has run to completion over
    the full corpus universe (so its absence is itself the primary truncation
    signal).  ``corpus_incident_ids`` is the full snapshot universe the producer
    was given; ``in_scope_incident_ids`` are the incidents it assigned an
    in-scope entry (== the rows it wrote to labeled_incidents.json).  n_oos is
    the remainder (OOS by absence).  Raises if an in-scope label names an
    incident absent from the snapshot (the producer ran against wrong inputs).
    """
    foreign = in_scope_incident_ids - corpus_incident_ids
    if foreign:
        raise LabeledIncidentsIncompleteError(
            f"classify produced {len(foreign)} in-scope label(s) for incident(s) "
            f"absent from the corpus snapshot (e.g. {sorted(foreign)[:3]}); the "
            "labels do not belong to the pinned snapshot."
        )
    n_corpus = len(corpus_incident_ids)
    n_in_scope = len(in_scope_incident_ids)
    cov = ClassifyCoverage(
        snapshot_hash=snapshot_hash,
        n_corpus=n_corpus,
        n_in_scope=n_in_scope,
        n_oos=n_corpus - n_in_scope,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / COVERAGE_FILENAME).write_text(
        json.dumps(
            {
                "snapshot_hash": cov.snapshot_hash,
                "n_corpus": cov.n_corpus,
                "n_in_scope": cov.n_in_scope,
                "n_oos": cov.n_oos,
            },
            indent=2,
        )
        + "\n"
    )
    return cov


def _resolve_snapshot_incidents(cycle: Path, snapshot_hash: str) -> Path | None:
    """Locate corpora/<corpus>/<snapshot_hash>/incidents.json (corpus-agnostic).

    Returns None when no pinned snapshot is resolvable (synthetic sentinel,
    empty hash, or the snapshot dir is absent) -> the verifier no-ops.
    Supports both 2-level layout (corpora/<corpus>/<hash>/incidents.json) and
    1-level layout (corpora/<hash>/incidents.json).
    """
    if not snapshot_hash or snapshot_hash == _SYNTHETIC_SNAPSHOT:
        return None
    matches = sorted((cycle / "corpora").glob(f"*/{snapshot_hash}/incidents.json"))
    if not matches:
        matches = sorted((cycle / "corpora").glob(f"{snapshot_hash}/incidents.json"))
    return matches[0] if matches else None


def read_snapshot_universe_ids(incidents_json: Path) -> set[str]:
    """The pinned snapshot's full incident-id universe (RAW file, UNFILTERED).

    Producer and verifier MUST both source the universe here so their counts
    cannot drift: the producer records n_corpus from this set, the verifier
    reconciles against it.  Deliberately does NOT apply the adapter's
    date/quarantine filtering — the marker's n_oos absorbs any
    adapter-excluded record (it is absent from labeled by construction and is
    never scored for recall, since goldset incidents are always in-scope).
    """
    data = json.loads(incidents_json.read_text(encoding="utf-8"))
    incidents = data["incidents"] if isinstance(data, dict) else data
    return {str(rec["id"]) for rec in incidents}


def _read_coverage(path: Path) -> ClassifyCoverage:
    data = json.loads(path.read_text(encoding="utf-8"))
    return ClassifyCoverage(
        snapshot_hash=str(data["snapshot_hash"]),
        n_corpus=int(data["n_corpus"]),
        n_in_scope=int(data["n_in_scope"]),
        n_oos=int(data["n_oos"]),
    )


def verify_labeled_completeness(
    cycle: Path,
    snapshot_hash: str,
    labeled_incident_ids: set[str],
    *,
    goldset_recall_ids: set[str] | None = None,
) -> None:
    """RAISE LabeledIncidentsIncompleteError on a PROVEN coverage gap.

    ``goldset_recall_ids`` (optional) are the goldset incident ids that will be
    SCORED against this classifier (i.e. recall labels with a non-None
    classifier_entry_id); each must be present in the pinned snapshot, else a
    goldset/snapshot provenance break is being silently scored as OOS-misses.
    """
    incidents_json = _resolve_snapshot_incidents(cycle, snapshot_hash)
    if incidents_json is None:
        return
    corpus_ids = read_snapshot_universe_ids(incidents_json)

    # (1) foreign / stale labeled file: a label for an incident absent from the
    #     pinned snapshot means labeled_incidents.json came from another run.
    foreign = labeled_incident_ids - corpus_ids
    if foreign:
        raise LabeledIncidentsIncompleteError(
            f"labeled_incidents.json references {len(foreign)} incident(s) absent "
            f"from the pinned corpus snapshot {snapshot_hash} "
            f"(e.g. {sorted(foreign)[:3]}); the labeled file does not match the snapshot."
        )

    # (2) the completion marker is REQUIRED for a snapshot-bearing cycle: its
    #     absence is the primary truncation signal (the producer writes it only
    #     after running to completion).
    cov_path = cycle / "classify" / COVERAGE_FILENAME
    if not cov_path.exists():
        raise LabeledIncidentsIncompleteError(
            f"classify coverage marker {cov_path} is missing: the classify phase "
            "did not record completion over the corpus snapshot, so a truncated / "
            "partial run cannot be distinguished from genuine OOS. Re-run classify "
            "to emit the coverage marker."
        )
    cov = _read_coverage(cov_path)

    # (3) marker must bind the SAME pinned snapshot.
    if cov.snapshot_hash != snapshot_hash:
        raise LabeledIncidentsIncompleteError(
            f"coverage marker snapshot_hash {cov.snapshot_hash!r} != manifest "
            f"snapshot_hash {snapshot_hash!r}: classify ran against a different snapshot."
        )

    # (4) anti-circularity: marker n_corpus must equal the INDEPENDENT pinned
    #     universe size (a partial run cannot fake having been given fewer).
    if cov.n_corpus != len(corpus_ids):
        raise LabeledIncidentsIncompleteError(
            f"coverage marker n_corpus={cov.n_corpus} != pinned snapshot size "
            f"{len(corpus_ids)}: classify covered a partial corpus."
        )

    # (5) marker invariant + reconciliation to the labeled file.
    if cov.n_in_scope + cov.n_oos != cov.n_corpus:
        raise LabeledIncidentsIncompleteError(
            f"coverage marker is internally inconsistent: n_in_scope("
            f"{cov.n_in_scope}) + n_oos({cov.n_oos}) != n_corpus({cov.n_corpus})."
        )
    if cov.n_in_scope != len(labeled_incident_ids):
        raise LabeledIncidentsIncompleteError(
            f"coverage marker n_in_scope={cov.n_in_scope} != labeled_incidents.json "
            f"count {len(labeled_incident_ids)}: labeled_incidents.json does not "
            "reconcile to the recorded classify coverage (truncated or altered after "
            "the marker was written)."
        )

    # (6) goldset/snapshot provenance: every scored recall incident must be in
    #     the snapshot the classifier actually ran (else "absent from labeled"
    #     is a provenance break, not genuine OOS).
    if goldset_recall_ids is not None:
        missing = goldset_recall_ids - corpus_ids
        if missing:
            raise LabeledIncidentsIncompleteError(
                f"{len(missing)} goldset recall incident(s) are absent from the "
                f"pinned corpus snapshot {snapshot_hash} (e.g. {sorted(missing)[:3]}); "
                "the goldset does not match the snapshot the classifier ran."
            )
