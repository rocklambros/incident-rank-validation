"""Corpus B corroboration: overlap detection and agreement computation.

HANDOFF §4 Corpus B role: qualitative corroboration of corpus A's curated
head only.  Not a modeled Bayesian channel.  Systematic divergence is a
published finding, never a silent posterior adjustment.

HANDOFF §5.5: agreement between corpus A labels and corpus B on shared
incidents, reported as a declared agree/disagree artifact.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from urllib.parse import urlparse

from engine.schema import IncidentRecord


class OverlapMethod(Enum):
    URL = "url_match"
    CVE = "cve_match"
    TITLE_KEYWORD = "title_keyword_match"


@dataclass(frozen=True, slots=True)
class IncidentOverlap:
    corpus_a_id: str
    corpus_b_id: str
    method: OverlapMethod
    match_detail: str


_CVE_RE = re.compile(r"CVE-\d{4}-\d+")

_TITLE_STOP_WORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "in", "on", "for", "to", "with",
    "by", "via", "is", "was", "are", "from", "at", "as", "its", "that",
    "this", "into", "can", "could", "through", "using",
})


def _normalize_url(url: str) -> str:
    """Normalize a URL for comparison: lowercase, strip trailing slash and www."""
    parsed = urlparse(url.lower().strip())
    host = parsed.netloc.removeprefix("www.")
    path = parsed.path.rstrip("/")
    return f"{host}{path}"


def _extract_cve_ids(text: str) -> set[str]:
    """Extract all CVE IDs from text or URLs."""
    return set(_CVE_RE.findall(text))


def _extract_significant_words(text: str, min_length: int = 4) -> set[str]:
    """Extract significant words from text for title matching."""
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return {
        w for w in words
        if len(w) >= min_length and w not in _TITLE_STOP_WORDS
    }


def detect_overlaps(
    corpus_a: list[IncidentRecord] | tuple[IncidentRecord, ...],
    corpus_b: list[IncidentRecord] | tuple[IncidentRecord, ...],
    title_match_threshold: int = 3,
) -> list[IncidentOverlap]:
    """Detect shared incidents between corpus A and corpus B.

    Strategy priority (highest to lowest confidence):
    1. URL normalization match (exact URL after normalization)
    2. CVE ID match (shared CVE identifier in text or URLs)
    3. Title keyword match (>= threshold significant shared words)

    Each corpus B incident matches at most one corpus A incident (best match).
    """
    url_to_a: dict[str, str] = {}
    cve_to_a: dict[str, str] = {}
    a_words: dict[str, set[str]] = {}
    a_texts: dict[str, str] = {}

    for rec in corpus_a:
        if rec.source_url:
            norm = _normalize_url(rec.source_url)
            if norm:
                url_to_a[norm] = rec.id
        for cve in _extract_cve_ids(rec.text + " " + rec.source_url):
            cve_to_a[cve] = rec.id
        a_words[rec.id] = _extract_significant_words(rec.text)
        a_texts[rec.id] = rec.text

    matched_a_ids: set[str] = set()
    overlaps: list[IncidentOverlap] = []

    for b_rec in corpus_b:
        match: IncidentOverlap | None = None

        b_urls = [b_rec.source_url] if b_rec.source_url else []
        b_all_text = b_rec.text + " " + b_rec.source_url
        for url in b_urls:
            norm = _normalize_url(url)
            if norm in url_to_a:
                a_id = url_to_a[norm]
                if a_id not in matched_a_ids:
                    match = IncidentOverlap(
                        corpus_a_id=a_id,
                        corpus_b_id=b_rec.id,
                        method=OverlapMethod.URL,
                        match_detail=f"URL: {url}",
                    )
                    break

        if match is None:
            b_cves = _extract_cve_ids(b_all_text)
            for cve in b_cves:
                if cve in cve_to_a:
                    a_id = cve_to_a[cve]
                    if a_id not in matched_a_ids:
                        match = IncidentOverlap(
                            corpus_a_id=a_id,
                            corpus_b_id=b_rec.id,
                            method=OverlapMethod.CVE,
                            match_detail=f"CVE: {cve}",
                        )
                        break

        if match is None:
            b_words = _extract_significant_words(b_rec.text)
            best_overlap_count = 0
            best_a_id = ""
            for a_id, a_w in a_words.items():
                if a_id in matched_a_ids:
                    continue
                shared = b_words & a_w
                if len(shared) >= title_match_threshold and len(shared) > best_overlap_count:
                    best_overlap_count = len(shared)
                    best_a_id = a_id
            if best_a_id:
                match = IncidentOverlap(
                    corpus_a_id=best_a_id,
                    corpus_b_id=b_rec.id,
                    method=OverlapMethod.TITLE_KEYWORD,
                    match_detail=f"Shared words: {best_overlap_count}",
                )

        if match is not None:
            matched_a_ids.add(match.corpus_a_id)
            overlaps.append(match)

    return overlaps


@dataclass(frozen=True, slots=True)
class IncidentAgreement:
    corpus_a_id: str
    corpus_b_id: str
    corpus_b_title: str
    match_method: str
    corpus_a_label: str
    corpus_b_label: str
    corpus_b_native_labels: tuple[str, ...]
    agrees: bool


@dataclass(frozen=True, slots=True)
class SystematicDivergence:
    pattern: str
    count: int
    incidents: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CorpusBCorroboration:
    corpus_b_incident_count: int
    corpus_a_incident_count: int
    overlap_count: int
    classification_stages_used: str
    per_incident: tuple[IncidentAgreement, ...]
    agreement_count: int
    disagreement_count: int
    agreement_rate: float
    systematic_divergences: tuple[SystematicDivergence, ...]
    baseline_kappa: float
    overlap_method_limitations: tuple[str, ...] = field(default=(
        "URL matching may miss equivalent URLs with different query parameters or redirects",
        "CVE matching requires both corpora to reference the same CVE ID in text or URLs",
        "Title keyword matching (fallback) may produce false positives from keyword collision on unrelated incidents",
        "Incidents described with different terminology may not match across corpora",
    ))


def compute_agreement(
    overlaps: list[IncidentOverlap],
    corpus_a_labels: dict[str, str],
    corpus_b_labels: dict[str, str],
    corpus_b_records: dict[str, IncidentRecord],
    baseline_kappa: float,
    corpus_a_count: int,
    corpus_b_count: int,
    classification_stages: str = "stage1",
) -> CorpusBCorroboration:
    agreements: list[IncidentAgreement] = []
    for ov in overlaps:
        a_label = corpus_a_labels.get(ov.corpus_a_id, "unclassified")
        b_label = corpus_b_labels.get(ov.corpus_b_id, "unclassified")
        b_rec = corpus_b_records.get(ov.corpus_b_id)
        b_title = b_rec.text.split(" ", 1)[0] if b_rec else ov.corpus_b_id
        if b_rec:
            title_words = b_rec.text.split()
            b_title = " ".join(title_words[:8]) if len(title_words) > 8 else b_rec.text

        agreements.append(IncidentAgreement(
            corpus_a_id=ov.corpus_a_id,
            corpus_b_id=ov.corpus_b_id,
            corpus_b_title=b_title,
            match_method=ov.method.value,
            corpus_a_label=a_label,
            corpus_b_label=b_label,
            corpus_b_native_labels=b_rec.native_labels if b_rec else (),
            agrees=(a_label == b_label),
        ))

    agree_count = sum(1 for a in agreements if a.agrees)
    disagree_count = len(agreements) - agree_count
    rate = agree_count / len(agreements) if agreements else 0.0

    divergence_patterns: dict[str, list[str]] = {}
    for a in agreements:
        if not a.agrees and a.corpus_a_label != "unclassified" and a.corpus_b_label != "unclassified":
            pattern_key = f"{a.corpus_a_label}_vs_{a.corpus_b_label}"
            divergence_patterns.setdefault(pattern_key, []).append(a.corpus_b_id)

    divergences = tuple(
        SystematicDivergence(
            pattern=f"Corpus A labels as {key.split('_vs_')[0]}, corpus B labels as {key.split('_vs_')[1]}",
            count=len(ids),
            incidents=tuple(ids),
        )
        for key, ids in divergence_patterns.items()
        if len(ids) >= 2
    )

    return CorpusBCorroboration(
        corpus_b_incident_count=corpus_b_count,
        corpus_a_incident_count=corpus_a_count,
        overlap_count=len(overlaps),
        classification_stages_used=classification_stages,
        per_incident=tuple(agreements),
        agreement_count=agree_count,
        disagreement_count=disagree_count,
        agreement_rate=rate,
        systematic_divergences=divergences,
        baseline_kappa=baseline_kappa,
    )
