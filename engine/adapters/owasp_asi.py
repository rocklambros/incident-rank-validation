"""Corpus B adapter for the OWASP ASI Agentic Exploits & Incidents tracker.

Reads the vendored Markdown snapshot and transforms rows into canonical
IncidentRecord instances.  This adapter is for **corroboration only** — the
incidents it emits never enter the Bayesian likelihood (HANDOFF §4, §5.4).

Bias profile is flagged ``qualitative_corroboration_only``.
"""
from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from engine.adapters.base import CorpusAdapter
from engine.model.overlap import OverlapWeights
from engine.schema import (
    BiasProfile,
    EntryDefinition,
    IncidentRecord,
    StratumSize,
    make_stratum_size,
)

_ASI_LABEL_RE = re.compile(r"ASI(\d{2})")
_MD_LINK_RE = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")
_BOLD_RE = re.compile(r"\*\*([^*]+)\*\*")

_MONTH_MAP: dict[str, str] = {
    "Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04",
    "May": "05", "Jun": "06", "Jul": "07", "Aug": "08",
    "Sep": "09", "Oct": "10", "Nov": "11", "Dec": "12",
}

_CORROBORATION_BIAS_PROFILE = BiasProfile(
    stratum="corroboration",
    description=(
        "OWASP ASI Agentic Exploits & Incidents tracker (~46 human-curated "
        "incidents).  Corroboration-only: never enters the Bayesian likelihood.  "
        "Selection is toward high-profile agentic security incidents, heavily "
        "weighted toward coding-agent and MCP-related vulnerabilities (2025)."
    ),
    known_blind_spots=(),
    contamination_description="None — all labels are human-curated.",
    quarantine_rule="None — no contamination quarantine needed.",
)


@dataclass(frozen=True, slots=True)
class ASIIncident:
    """Parsed incident from the ASI Exploits & Incidents table."""
    id: str
    date: str
    title: str
    impact_summary: str
    asi_labels: tuple[str, ...]
    urls: tuple[str, ...]


def _parse_date(raw: str) -> str:
    """Convert '**Dec 2025**' or 'Dec 2025' to '2025-12-01'."""
    clean = _BOLD_RE.sub(r"\1", raw).strip()
    parts = clean.split()
    if len(parts) != 2:
        return "1970-01-01"
    month_str, year_str = parts
    month = _MONTH_MAP.get(month_str[:3], "01")
    return f"{year_str}-{month}-01"


def _extract_asi_labels(raw: str) -> tuple[str, ...]:
    """Extract ASI entry IDs like 'ASI04' from the T&M Mapping column."""
    return tuple(f"ASI{m}" for m in _ASI_LABEL_RE.findall(raw))


def _extract_urls(raw: str) -> tuple[str, ...]:
    """Extract URLs from markdown links in the Links column."""
    return tuple(url for _, url in _MD_LINK_RE.findall(raw))


def _clean_cell(cell: str) -> str:
    """Strip HTML tags, markdown bold, bullet points, and normalize whitespace."""
    text = re.sub(r"<br\s*/?>", " ", cell)
    text = _BOLD_RE.sub(r"\1", text)
    text = re.sub(r"•\s*", "", text)
    return " ".join(text.split()).strip()


def parse_asi_markdown(text: str) -> list[ASIIncident]:
    """Parse the ASI Exploits & Incidents markdown table into ASIIncident list."""
    lines = text.split("\n")
    incidents: list[ASIIncident] = []
    header_found = False
    separator_skipped = False
    idx = 0

    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue

        if "Exploit" in stripped and "Incident" in stripped and "Impact" in stripped:
            header_found = True
            separator_skipped = False
            continue

        if not header_found:
            continue

        if not separator_skipped and re.match(r"^\|[\s\-|]+\|$", stripped):
            separator_skipped = True
            continue

        cells = stripped.split("|")
        cells = [c.strip() for c in cells]
        if cells and cells[0] == "":
            cells = cells[1:]
        if cells and cells[-1] == "":
            cells = cells[:-1]

        if len(cells) < 4:
            continue

        date_cell = cells[0]
        title_cell = cells[1]
        impact_cell = cells[2]
        mapping_cell = cells[3] if len(cells) > 3 else ""
        links_cell = cells[4] if len(cells) > 4 else ""

        title = _clean_cell(title_cell)
        if not title or title == "---":
            continue

        idx += 1
        date_str = _parse_date(date_cell) if date_cell.strip() else "1970-01-01"

        incidents.append(ASIIncident(
            id=f"ASIB-{idx:03d}",
            date=date_str,
            title=title,
            impact_summary=_clean_cell(impact_cell),
            asi_labels=_extract_asi_labels(mapping_cell),
            urls=_extract_urls(links_cell),
        ))

    return incidents


class OWASPASIAdapter(CorpusAdapter):
    """Adapter for the OWASP ASI Agentic Exploits & Incidents tracker.

    Corroboration-only.  Emits canonical IncidentRecord instances from the
    vendored Markdown snapshot.  Never enters the Bayesian model.

    Parameters
    ----------
    snapshot_dir:
        Path to vendored snapshot directory containing
        ``ASI_Agentic_Exploits_Incidents.md``.
    """

    def __init__(self, snapshot_dir: Path) -> None:
        self._md_path = snapshot_dir / "ASI_Agentic_Exploits_Incidents.md"
        if not self._md_path.exists():
            raise FileNotFoundError(
                f"ASI_Agentic_Exploits_Incidents.md not found in {snapshot_dir}"
            )
        self._parsed: list[ASIIncident] | None = None

    def _load(self) -> list[ASIIncident]:
        if self._parsed is None:
            self._parsed = parse_asi_markdown(self._md_path.read_text())
        return self._parsed

    def iter_incidents(self) -> Iterator[IncidentRecord]:
        for inc in self._load():
            source_url = inc.urls[0] if inc.urls else ""
            yield IncidentRecord(
                id=inc.id,
                date=inc.date,
                text=f"{inc.title} {inc.impact_summary}",
                severity=None,
                source_class="advisory",
                corpus_stratum="corroboration",
                quality="curated",
                native_labels=inc.asi_labels,
                source_url=source_url,
            )

    def bias_profiles(self) -> tuple[BiasProfile, ...]:
        return (_CORROBORATION_BIAS_PROFILE,)

    def stratum_sizes(self) -> dict[str, StratumSize]:
        count = sum(1 for _ in self.iter_incidents())
        return {"corroboration": make_stratum_size(max(1, count))}

    def entry_definitions(self) -> tuple[EntryDefinition, ...]:
        return ()

    def overlap_weights(self) -> OverlapWeights:
        return OverlapWeights(weights={})
