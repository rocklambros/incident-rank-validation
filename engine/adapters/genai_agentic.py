"""Corpus A adapter for the genai_agentic_incidents dataset.

Reads the vendored snapshot (``incidents.json`` in the ``{"incidents": [...]}``
wrapper format) and transforms raw source records into canonical
``IncidentRecord`` instances per HANDOFF §5.1.

Key schema deviations from provisional field names:
- Source URLs live in ``references[0]["url"]``, not a ``source_url`` field.
- Severity defaulting uses ``quality_tier`` alone (no ``severity_source``
  or ``severity_method`` fields exist in the source).
- The ``owasp_asi`` field is separate from ``owasp_llm`` — the double-default
  quarantine fires on 0 records in practice (ASI04 is not in owasp_llm).
- Eight ``category`` values need mapping to coarser ``source_class``.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path

from engine.adapters.base import CorpusAdapter
from engine.adapters.genai_agentic_bias import BIAS_PROFILES
from engine.model.overlap import OverlapWeights
from engine.schema import (
    BiasProfile,
    EntryDefinition,
    IncidentRecord,
    StratumSize,
    make_stratum_size,
)

logger = logging.getLogger(__name__)

_MAX_TEXT_LENGTH = 50_000

# Module-scope taxonomy definition (Premortem: must NOT be inside the class).
_PROVISIONAL_2025_ENTRIES: tuple[EntryDefinition, ...] = (
    EntryDefinition(entry_id="LLM01", name="Prompt Injection"),
    EntryDefinition(entry_id="LLM02", name="Sensitive Information Disclosure"),
    EntryDefinition(entry_id="LLM03", name="Supply Chain Vulnerabilities"),
    EntryDefinition(
        entry_id="LLM04", name="Data and Model Poisoning", frame_blind=True
    ),
    EntryDefinition(entry_id="LLM05", name="Improper Output Handling"),
    EntryDefinition(entry_id="LLM06", name="Excessive Agency"),
    EntryDefinition(entry_id="LLM07", name="System Prompt Leakage"),
    EntryDefinition(
        entry_id="LLM08", name="Vector and Embedding Weaknesses", frame_blind=True
    ),
    EntryDefinition(entry_id="LLM09", name="Misinformation"),
    EntryDefinition(
        entry_id="LLM10", name="Unbounded Consumption", frame_blind=True
    ),
)

# Source category -> engine source_class mapping.
# Multiple source categories collapse to fewer engine classes (Premortem M7).
_CATEGORY_MAP: dict[str, str] = {
    "real-world": "harm-report",
    "vulnerability-disclosure": "cve",
    "research": "advisory",
    "research-demonstrated": "advisory",
    "threat-report": "advisory",
    "red-team": "advisory",
    "report": "advisory",
    "regulatory": "advisory",
}


class GenAIAgenticAdapter(CorpusAdapter):
    """Adapter for the genai_agentic_incidents vendored snapshot.

    Parameters
    ----------
    snapshot_dir:
        Path to the vendored snapshot directory containing ``incidents.json``.
    snapshot_date:
        ISO 8601 date string (``YYYY-MM-DD``).  Records dated after this are
        excluded (future-dated filter).
    """

    def __init__(self, snapshot_dir: Path, snapshot_date: str) -> None:
        if not snapshot_dir.exists():
            raise FileNotFoundError(
                f"Snapshot directory not found: {snapshot_dir}"
            )
        self._incidents_path = snapshot_dir / "incidents.json"
        if not self._incidents_path.exists():
            raise FileNotFoundError(
                f"incidents.json not found in snapshot directory: {snapshot_dir}"
            )
        self._snapshot_date = snapshot_date
        self._records: list[dict[str, object]] | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> list[dict[str, object]]:
        """Load and cache raw incident dicts from the vendored JSON file."""
        if self._records is None:
            raw = self._incidents_path.read_text()
            data = json.loads(raw)
            # Handle {"incidents": [...]} wrapper format
            if isinstance(data, dict) and "incidents" in data:
                incidents = data["incidents"]
            elif isinstance(data, list):
                incidents = data
            else:
                raise TypeError(
                    f"Expected JSON array or dict with 'incidents' key, "
                    f"got {type(data).__name__}"
                )
            if not isinstance(incidents, list):
                raise TypeError(
                    f"Expected incidents to be a list, "
                    f"got {type(incidents).__name__}"
                )
            self._records = incidents
        return self._records

    def _transform(self, raw: dict[str, object]) -> IncidentRecord | None:
        """Transform a raw source dict into a canonical IncidentRecord.

        Returns ``None`` for records that should be excluded (future-dated).
        """
        record_id = str(raw.get("id", ""))
        date_str = str(raw.get("date", ""))

        # Future-dated filter: exclude records after the snapshot date.
        if date_str > self._snapshot_date:
            return None

        # Build text from title + description + impact.
        title = str(raw.get("title", ""))
        description = str(raw.get("description", ""))
        impact = str(raw.get("impact", ""))
        text = " ".join(part for part in [title, description, impact] if part)

        if len(text) > _MAX_TEXT_LENGTH:
            logger.warning(
                "Record %s text truncated from %d to %d chars",
                record_id,
                len(text),
                _MAX_TEXT_LENGTH,
            )
            text = text[:_MAX_TEXT_LENGTH]

        # Severity defaulting (Case 3: quality_tier heuristic only).
        # No severity_source/severity_method fields exist in the source.
        # quality_tier=="curated" means human-confirmed severity — keep it.
        # All other quality tiers with severity=="Medium" are treated as
        # source-ingest default artifacts and mapped to None.
        raw_severity = raw.get("severity")
        severity: str | None
        if raw_severity == "Medium" and self._is_severity_defaulted(raw):
            severity = None
        elif raw_severity is not None:
            severity = str(raw_severity)
        else:
            severity = None

        corpus_stratum = str(raw.get("corpus", "unknown"))
        category = str(raw.get("category", ""))
        source_class = _CATEGORY_MAP.get(category, "advisory")
        quality_tier = str(raw.get("quality_tier", "auto"))
        quality = self._map_quality(quality_tier)

        # Native labels from owasp_llm (non-authoritative metadata).
        owasp_llm = raw.get("owasp_llm")
        native_labels = (
            tuple(str(x) for x in owasp_llm)
            if isinstance(owasp_llm, list)
            else ()
        )

        # Extract URL from references list (no source_url field in source).
        references = raw.get("references")
        source_url = ""
        if isinstance(references, list) and references:
            first_ref = references[0]
            if isinstance(first_ref, dict):
                source_url = str(first_ref.get("url", ""))

        return IncidentRecord(
            id=record_id,
            date=date_str,
            text=text,
            severity=severity,
            source_class=source_class,
            corpus_stratum=corpus_stratum,
            quality=quality,
            native_labels=native_labels,
            source_url=source_url,
        )

    @staticmethod
    def _is_severity_defaulted(raw: dict[str, object]) -> bool:
        """Detect source-ingest severity defaulting to 'Medium'.

        No severity_source/severity_method fields exist in the source.
        quality_tier=='curated' means human-confirmed severity -- keep it.
        All other quality tiers with severity=='Medium' are treated as defaulted.
        """
        quality = raw.get("quality_tier", "")
        return quality != "curated"

    @staticmethod
    def _map_quality(quality_tier: str) -> str:
        """Map source quality_tier to canonical quality label."""
        mapping: dict[str, str] = {
            "curated": "curated",
            "reviewed": "reviewed",
        }
        return mapping.get(quality_tier, "auto")

    # ------------------------------------------------------------------
    # CorpusAdapter interface
    # ------------------------------------------------------------------

    def iter_incidents(self) -> Iterator[IncidentRecord]:
        """Yield canonical incident records from the vendored snapshot."""
        for raw in self._load():
            record = self._transform(raw)
            if record is not None:
                yield record

    def bias_profiles(self) -> tuple[BiasProfile, ...]:
        """Return declared bias profiles, one per stratum."""
        return BIAS_PROFILES

    def stratum_sizes(self) -> dict[str, StratumSize]:
        """Return the exposure term per stratum (computed from data)."""
        counts: dict[str, int] = {}
        for record in self.iter_incidents():
            counts[record.corpus_stratum] = (
                counts.get(record.corpus_stratum, 0) + 1
            )
        return {k: make_stratum_size(v) for k, v in counts.items()}

    def entry_definitions(self) -> tuple[EntryDefinition, ...]:
        """Return the OWASP LLM Top 10 (2025) taxonomy entries."""
        return _PROVISIONAL_2025_ENTRIES

    def overlap_weights(self) -> OverlapWeights:
        """Return the declared FP leakage structure."""
        return OverlapWeights(weights={"LLM05": {"LLM03": 0.2}})
