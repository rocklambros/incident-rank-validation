# Corpus B Corroboration Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce the declared agree/disagree artifact comparing corpus A and corpus B incident classifications on shared incidents, as a qualitative corroboration artifact — never a posterior input.

**Architecture:** Parse corpus B (the ASI Agentic Exploits & Incidents tracker, ~46 incidents in a Markdown table) into canonical IncidentRecords via a new adapter. Classify corpus B through the same Stage-1 + Stage-2 pipeline used for corpus A. Detect incident overlap via URL/CVE/title matching. Compute per-incident agreement between corpus A and corpus B classifications on shared incidents. Surface systematic divergence as a published finding. Integrate into the cycle report as a declared qualitative section. Enforce via regression test that corpus B never enters `engine/model/inference.py`.

**Tech Stack:** Python 3.11+, existing engine infrastructure (classifier, rubric, report renderer), pytest, mypy, ruff

---

## Inherited constraints from Phases 1-5

1. **Corpus B is corroboration only — NEVER a posterior input.** Corpus B does not enter the Bayesian likelihood (HANDOFF §4 Corpus B role, §5.4 single-channel bullet). A regression test on `engine/model/inference.py` must confirm no corpus B import or artifact reference. Violation of this constraint invalidates the cycle.

2. **N is dozens — this is agreement reporting, not statistical testing.** Corpus B has ~46 incidents (HANDOFF §1: "about 68 lines, dozens of incidents"). No kappa, no hypothesis test, no confidence intervals on the agreement rate. Report raw counts: N shared, N agree, N disagree. Interpretive commentary is qualitative.

3. **Systematic divergence is a published finding per HANDOFF §4, never a silent adjustment.** If corpus B consistently disagrees with corpus A on a particular entry or class of incidents, that divergence is reported as-is in the report. The engine does not adjust posteriors, re-weight entries, or suppress findings based on corpus B disagreement.

4. **Incident-id overlap may be weak — text-match fallback must be defined and its limitations declared in the artifact.** Corpus A uses synthetic IDs (`INC-XXXXX`); corpus B uses parsed titles from the ASI tracker. No shared ID scheme exists. Overlap detection uses a multi-strategy approach: (a) URL normalization matching, (b) CVE ID matching, (c) title keyword matching as fallback. Each strategy's false-positive/false-negative risks are declared in the output artifact.

5. **Corpus B incidents should pass through the same two-stage classification pipeline (Stage-1 + Stage-2) as corpus A for consistency.** Corpus B incidents are classified against the frozen rubric using the same Stage-1 keyword classifier and (if available) Stage-2 LLM classifier. This ensures apples-to-apples comparison of LLM taxonomy labels on shared incidents. If Stage-2 is unavailable for the corpus B batch, the artifact notes "Stage-1 only" as a limitation.

6. **Baseline concordance is weak (kappa 0.275) — corpus B agreement reporting must be interpreted in that context.** The cycle's headline concordance between vote ranking and incident ranking is 0.275 [-0.01, 0.57], computed over 17 of 20 entries (85% coverage). Corpus B agreement/disagreement sits on top of this already-weak baseline signal. The artifact includes the baseline kappa as interpretive context.

7. **3 entries are frame-blind (LLM04, LLM08, LLM10) — corpus B mappings to these entries are reportable but cannot strengthen the posterior.** These entries have `"no-data"` flags in `calibration/diagnostic.json` and are excluded from the measurable subset. If corpus B incidents classify to a frame-blind entry, the per-incident agreement row is still reported, but the report section notes that agreement on frame-blind entries has no bearing on posterior estimates.

---

## File Structure

### New files

| File | Responsibility |
|------|---------------|
| `engine/adapters/owasp_asi.py` | Corpus B adapter: parse ASI Markdown table into canonical `IncidentRecord` instances |
| `engine/decide/corpus_b_corroboration.py` | Overlap detection + per-incident agreement + systematic divergence computation |
| `tests/unit/test_corpus_b_regression.py` | Regression test: `inference.py` never imports corpus B |
| `tests/unit/test_adapter_owasp_asi.py` | Adapter unit tests: parsing, field mapping, edge cases |
| `tests/unit/test_corpus_b_corroboration.py` | Corroboration unit tests: overlap detection, agreement computation |

### Modified files

| File | Change |
|------|--------|
| `engine/report/render.py` | Add `CorpusB Corroboration` section to `render_report()` and `ReportInputs` |
| `engine/cli/pipeline.py` | Add `corroborate` CLI command |
| `docs/METHODOLOGY-CHANGELOG.md` | Add 1.2.0 entry |

### Data artifacts produced

| Artifact | Description |
|----------|-------------|
| `projects/owasp-llm/cycles/2026/corpora/owasp_asi/{hash}/ASI_Agentic_Exploits_Incidents.md` | Vendored snapshot of corpus B source |
| `projects/owasp-llm/cycles/2026/corpora/owasp_asi/{hash}/provenance.json` | Snapshot provenance |
| `projects/owasp-llm/cycles/2026/classify/corpus_b_labeled.json` | Corpus B Stage-1 (+Stage-2) classifications |
| `projects/owasp-llm/cycles/2026/results/corpus_b_corroboration.json` | Agreement artifact |

---

### Task 1: Inference Isolation Regression Test

**Files:**
- Create: `tests/unit/test_corpus_b_regression.py`
- Read: `engine/model/inference.py`

This is the FIRST task per the inherited constraint. It enforces that corpus B artifacts never leak into the Bayesian inference module.

- [ ] **Step 1: Write the regression test**

```python
"""Regression test: corpus B must never enter the inference module.

HANDOFF §4 Corpus B role: 'Not a modeled Bayesian channel.'
HANDOFF §5.4: 'Corpus B is not in the likelihood.'
"""
from __future__ import annotations

import ast
from pathlib import Path


_INFERENCE_PATH = Path(__file__).resolve().parents[2] / "engine" / "model" / "inference.py"

_CORPUS_B_MARKERS = frozenset({
    "owasp_asi",
    "corpus_b",
    "corroboration",
    "asi_agentic",
    "ASI_Agentic",
    "ASIB-",
})


def test_inference_has_no_corpus_b_imports() -> None:
    """Assert inference.py does not import any corpus B module."""
    source = _INFERENCE_PATH.read_text()
    tree = ast.parse(source, filename=str(_INFERENCE_PATH))

    imported_names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_names.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported_names.append(node.module)
            for alias in node.names:
                imported_names.append(alias.name)

    for name in imported_names:
        for marker in _CORPUS_B_MARKERS:
            assert marker not in name, (
                f"inference.py imports '{name}' which contains corpus B marker "
                f"'{marker}'. Corpus B must NEVER enter the likelihood "
                f"(HANDOFF §4, §5.4)."
            )


def test_inference_source_has_no_corpus_b_references() -> None:
    """Assert inference.py source text has no corpus B references."""
    source = _INFERENCE_PATH.read_text()
    for marker in _CORPUS_B_MARKERS:
        assert marker not in source, (
            f"inference.py contains corpus B marker '{marker}'. "
            f"Corpus B is corroboration only — never a posterior input."
        )
```

- [ ] **Step 2: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_corpus_b_regression.py -v`
Expected: PASS (2 tests) — inference.py currently has no corpus B references.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_corpus_b_regression.py
git commit -m "test(regression): assert corpus B never enters inference module (Plan 6)"
```

---

### Task 2: Corpus B Markdown Adapter

**Files:**
- Create: `tests/unit/test_adapter_owasp_asi.py`
- Create: `engine/adapters/owasp_asi.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Unit tests for the OWASP ASI corpus B adapter."""
from __future__ import annotations

import textwrap

import pytest

from engine.adapters.owasp_asi import OWASPASIAdapter, parse_asi_markdown


_SAMPLE_MD = textwrap.dedent("""\
    # ASI Agentic Exploits & Incidents Tracker

    ## Exploits & Incidents Table

    | Date | Exploit / Incident | Impact Summary | ASI T&M Mapping | Links to further analysis<br>(Vendor / CVE / Discoverer) |
    |------------|----------------------------|-------------------------------|------------------------------|---------------------------|
    |**Dec 2025**| **Claude Skills Ransomware Deployment** | Cato Networks demonstrated ransomware deployment via Claude Skills. | • ASI04 (Agentic Supply Chain Vulnerabilities)<br> • ASI05 (Unexpected Code Execution (RCE)) | • —<br> • —<br> • [Cato CTRL](https://www.catonetworks.com/blog/cato-ctrl-weaponizing-claude-skills-with-medusalocker/) |
    |**Nov 2025**| **ShadowRay 2.0 Botnet** | Attackers exploited Ray AI framework flaw. | • ASI05 (Unexpected Code Execution (RCE))<br> • ASI01 (Agent Goal Hijack) | • —<br> • [NVD](https://nvd.nist.gov/vuln/detail/CVE-2023-48022)<br> • [Oligo Security](https://www.oligo.security/blog/shadowray-2-0-attackers-turn-ai-against-itself-in-global-campaign-that-hijacks-ai-into-self-propagating-botnet) |
    |**Feb 2025**| **OpenAI ChatGPT Operator Vulnerability** | Prompt injection caused Operator to follow attacker instructions. | • ASI01 (Agent Goal Hijack)<br> • ASI02 (Tool Misuse & Exploitation) | • —<br> • —<br> • [Embrace The Red](https://embracethered.com/blog/posts/2025/chatgpt-operator-prompt-injection-exploits/) |
    ---
""")


class TestParseASIMarkdown:
    def test_parses_correct_count(self) -> None:
        incidents = parse_asi_markdown(_SAMPLE_MD)
        assert len(incidents) == 3

    def test_first_incident_fields(self) -> None:
        incidents = parse_asi_markdown(_SAMPLE_MD)
        first = incidents[0]
        assert first.id == "ASIB-001"
        assert first.date == "2025-12-01"
        assert "Claude Skills Ransomware Deployment" in first.title
        assert "Cato Networks" in first.impact_summary
        assert first.asi_labels == ("ASI04", "ASI05")
        assert any("catonetworks.com" in u for u in first.urls)

    def test_cve_url_extracted(self) -> None:
        incidents = parse_asi_markdown(_SAMPLE_MD)
        second = incidents[1]
        assert any("CVE-2023-48022" in u for u in second.urls)
        assert second.asi_labels == ("ASI05", "ASI01")

    def test_date_parsing(self) -> None:
        incidents = parse_asi_markdown(_SAMPLE_MD)
        assert incidents[0].date == "2025-12-01"
        assert incidents[1].date == "2025-11-01"
        assert incidents[2].date == "2025-02-01"

    def test_empty_table_returns_empty(self) -> None:
        md = "# No table here\n\nJust text."
        assert parse_asi_markdown(md) == []


class TestOWASPASIAdapter:
    def test_iter_incidents_yields_incident_records(self, tmp_path: Path) -> None:
        md_file = tmp_path / "ASI_Agentic_Exploits_Incidents.md"
        md_file.write_text(_SAMPLE_MD)
        adapter = OWASPASIAdapter(tmp_path)

        records = list(adapter.iter_incidents())
        assert len(records) == 3
        assert records[0].id == "ASIB-001"
        assert records[0].corpus_stratum == "corroboration"
        assert records[0].quality == "curated"
        assert records[0].source_class == "advisory"

    def test_bias_profile_is_corroboration_only(self, tmp_path: Path) -> None:
        md_file = tmp_path / "ASI_Agentic_Exploits_Incidents.md"
        md_file.write_text(_SAMPLE_MD)
        adapter = OWASPASIAdapter(tmp_path)

        profiles = adapter.bias_profiles()
        assert len(profiles) == 1
        assert profiles[0].stratum == "corroboration"

    def test_native_labels_carry_asi_ids(self, tmp_path: Path) -> None:
        md_file = tmp_path / "ASI_Agentic_Exploits_Incidents.md"
        md_file.write_text(_SAMPLE_MD)
        adapter = OWASPASIAdapter(tmp_path)

        records = list(adapter.iter_incidents())
        assert "ASI04" in records[0].native_labels
        assert "ASI05" in records[0].native_labels


from pathlib import Path
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_adapter_owasp_asi.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'engine.adapters.owasp_asi'`

- [ ] **Step 3: Implement the adapter**

```python
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

        if not separator_skipped:
            if re.match(r"^\|[\s\-|]+\|$", stripped):
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_adapter_owasp_asi.py -v`
Expected: PASS (all 8 tests)

- [ ] **Step 5: Run full suite to check for regressions**

Run: `uv run pytest -v && uv run mypy engine tests && uv run ruff check .`
Expected: All green

- [ ] **Step 6: Commit**

```bash
git add engine/adapters/owasp_asi.py tests/unit/test_adapter_owasp_asi.py
git commit -m "feat(adapters): OWASP ASI corpus B adapter for corroboration (Plan 6)"
```

---

### Task 3: Corpus B Snapshot Vendoring

**Files:**
- Produce: `projects/owasp-llm/cycles/2026/corpora/owasp_asi/{hash}/ASI_Agentic_Exploits_Incidents.md`
- Produce: `projects/owasp-llm/cycles/2026/corpora/owasp_asi/{hash}/provenance.json`

- [ ] **Step 1: Compute content hash of the source file**

```bash
CORPUS_B_SRC="$HOME/github_projects/www-project-top-10-for-large-language-model-applications/initiatives/agent_security_initiative/ASI Agentic Exploits & Incidents/ASI_Agentic_Exploits_Incidents.md"
HASH=$(sha256sum "$CORPUS_B_SRC" | cut -c1-12)
echo "Content hash: $HASH"
```

- [ ] **Step 2: Create vendored snapshot directory and copy source**

```bash
DEST="projects/owasp-llm/cycles/2026/corpora/owasp_asi/$HASH"
mkdir -p "$DEST"
cp "$CORPUS_B_SRC" "$DEST/ASI_Agentic_Exploits_Incidents.md"
echo "Vendored to $DEST"
```

- [ ] **Step 3: Write provenance.json**

```bash
cat > "$DEST/provenance.json" << 'PROVENANCE'
{
  "source_repo": "www-project-top-10-for-large-language-model-applications",
  "source_path": "initiatives/agent_security_initiative/ASI Agentic Exploits & Incidents/ASI_Agentic_Exploits_Incidents.md",
  "content_hash_algorithm": "sha256",
  "content_hash": "HASH_PLACEHOLDER",
  "pull_date": "2026-05-22",
  "adapter_version": "owasp_asi-1.0.0",
  "engine_version": "1.2.0",
  "corpus_role": "qualitative_corroboration_only"
}
PROVENANCE
```

Replace `HASH_PLACEHOLDER` with the actual hash computed in Step 1.

- [ ] **Step 4: Verify the adapter loads the vendored snapshot**

```bash
uv run python -c "
from pathlib import Path
from engine.adapters.owasp_asi import OWASPASIAdapter
import glob

dirs = glob.glob('projects/owasp-llm/cycles/2026/corpora/owasp_asi/*/')
adapter = OWASPASIAdapter(Path(dirs[0]))
records = list(adapter.iter_incidents())
print(f'Loaded {len(records)} corpus B incidents from vendored snapshot')
for r in records[:3]:
    print(f'  {r.id}: {r.text[:80]}...')
"
```

Expected: ~46 incidents loaded successfully.

- [ ] **Step 5: Commit**

```bash
git add "projects/owasp-llm/cycles/2026/corpora/owasp_asi/"
git commit -m "data(corpora): vendor OWASP ASI corpus B snapshot for corroboration (Plan 6)"
```

---

### Task 4: Overlap Detection Module

**Files:**
- Create: `tests/unit/test_corpus_b_corroboration.py` (first part: overlap tests)
- Create: `engine/decide/corpus_b_corroboration.py` (overlap detection)

- [ ] **Step 1: Write the failing overlap detection tests**

```python
"""Unit tests for corpus B corroboration: overlap detection and agreement."""
from __future__ import annotations

import pytest

from engine.decide.corpus_b_corroboration import (
    CorpusBCorroboration,
    IncidentOverlap,
    OverlapMethod,
    detect_overlaps,
)
from engine.schema import IncidentRecord


def _make_record(
    record_id: str,
    text: str = "test",
    source_url: str = "",
    native_labels: tuple[str, ...] = (),
) -> IncidentRecord:
    return IncidentRecord(
        id=record_id,
        date="2025-01-01",
        text=text,
        severity=None,
        source_class="advisory",
        corpus_stratum="security",
        quality="auto",
        native_labels=native_labels,
        source_url=source_url,
    )


class TestOverlapDetection:
    def test_url_match(self) -> None:
        corpus_a = [
            _make_record("INC-001", source_url="https://nvd.nist.gov/vuln/detail/CVE-2025-64110"),
        ]
        corpus_b = [
            _make_record("ASIB-001", source_url="https://nvd.nist.gov/vuln/detail/CVE-2025-64110"),
        ]
        overlaps = detect_overlaps(corpus_a, corpus_b)
        assert len(overlaps) == 1
        assert overlaps[0].corpus_a_id == "INC-001"
        assert overlaps[0].corpus_b_id == "ASIB-001"
        assert overlaps[0].method == OverlapMethod.URL

    def test_cve_match_from_url(self) -> None:
        corpus_a = [
            _make_record(
                "INC-002",
                text="Vulnerability CVE-2023-48022 exploited",
                source_url="https://example.com/some-page",
            ),
        ]
        corpus_b = [
            _make_record(
                "ASIB-002",
                source_url="https://nvd.nist.gov/vuln/detail/CVE-2023-48022",
            ),
        ]
        overlaps = detect_overlaps(corpus_a, corpus_b)
        assert len(overlaps) == 1
        assert overlaps[0].method == OverlapMethod.CVE

    def test_no_overlap_when_different(self) -> None:
        corpus_a = [_make_record("INC-001", text="Something unrelated")]
        corpus_b = [_make_record("ASIB-001", text="Completely different")]
        overlaps = detect_overlaps(corpus_a, corpus_b)
        assert len(overlaps) == 0

    def test_title_keyword_match(self) -> None:
        corpus_a = [
            _make_record(
                "INC-003",
                text="Claude Skills Ransomware Deployment demonstrated by Cato Networks",
            ),
        ]
        corpus_b = [
            _make_record(
                "ASIB-003",
                text="Claude Skills Ransomware Deployment via MedusaLocker",
            ),
        ]
        overlaps = detect_overlaps(corpus_a, corpus_b)
        assert len(overlaps) == 1
        assert overlaps[0].method == OverlapMethod.TITLE_KEYWORD

    def test_deduplicates_matches(self) -> None:
        url = "https://nvd.nist.gov/vuln/detail/CVE-2025-64110"
        corpus_a = [
            _make_record("INC-001", text="CVE-2025-64110 details", source_url=url),
        ]
        corpus_b = [
            _make_record("ASIB-001", text="CVE-2025-64110 info", source_url=url),
        ]
        overlaps = detect_overlaps(corpus_a, corpus_b)
        assert len(overlaps) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_corpus_b_corroboration.py::TestOverlapDetection -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'engine.decide.corpus_b_corroboration'`

- [ ] **Step 3: Implement overlap detection**

```python
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
    3. Title keyword match (≥ threshold significant shared words)

    Each corpus B incident matches at most one corpus A incident (best match).
    """
    # Build corpus A indexes
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

        # Strategy 1: URL match
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

        # Strategy 2: CVE ID match
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

        # Strategy 3: Title keyword match
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
    """Compute per-incident agreement and systematic divergence.

    Parameters
    ----------
    overlaps
        Detected incident overlaps between corpora.
    corpus_a_labels
        Mapping of corpus A incident_id → highest-confidence entry_id.
    corpus_b_labels
        Mapping of corpus B incident_id → highest-confidence entry_id.
    corpus_b_records
        Mapping of corpus B incident_id → IncidentRecord (for title).
    baseline_kappa
        Cycle headline kappa (for interpretive context).
    corpus_a_count
        Total corpus A incident count.
    corpus_b_count
        Total corpus B incident count.
    classification_stages
        Which stages were used: "stage1" or "stage1+stage2".
    """
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

    # Detect systematic divergence: entries where disagreement is consistent
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_corpus_b_corroboration.py::TestOverlapDetection -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add engine/decide/corpus_b_corroboration.py tests/unit/test_corpus_b_corroboration.py
git commit -m "feat(decide): corpus B overlap detection + agreement computation (Plan 6)"
```

---

### Task 5: Corroboration Agreement Tests

**Files:**
- Modify: `tests/unit/test_corpus_b_corroboration.py` (add agreement tests)

- [ ] **Step 1: Add agreement computation tests**

Append to `tests/unit/test_corpus_b_corroboration.py`:

```python
from engine.decide.corpus_b_corroboration import compute_agreement


class TestAgreementComputation:
    def test_full_agreement(self) -> None:
        overlaps = [
            IncidentOverlap("INC-001", "ASIB-001", OverlapMethod.URL, "url match"),
            IncidentOverlap("INC-002", "ASIB-002", OverlapMethod.CVE, "cve match"),
        ]
        a_labels = {"INC-001": "LLM01", "INC-002": "LLM05"}
        b_labels = {"ASIB-001": "LLM01", "ASIB-002": "LLM05"}
        b_records = {
            "ASIB-001": _make_record("ASIB-001", text="Incident Alpha"),
            "ASIB-002": _make_record("ASIB-002", text="Incident Beta"),
        }

        result = compute_agreement(
            overlaps, a_labels, b_labels, b_records,
            baseline_kappa=0.275, corpus_a_count=100, corpus_b_count=10,
        )
        assert result.agreement_count == 2
        assert result.disagreement_count == 0
        assert result.agreement_rate == 1.0

    def test_partial_agreement(self) -> None:
        overlaps = [
            IncidentOverlap("INC-001", "ASIB-001", OverlapMethod.URL, "match"),
            IncidentOverlap("INC-002", "ASIB-002", OverlapMethod.CVE, "match"),
        ]
        a_labels = {"INC-001": "LLM01", "INC-002": "LLM05"}
        b_labels = {"ASIB-001": "LLM01", "ASIB-002": "LLM03"}
        b_records = {
            "ASIB-001": _make_record("ASIB-001", text="Incident Alpha"),
            "ASIB-002": _make_record("ASIB-002", text="Incident Beta"),
        }

        result = compute_agreement(
            overlaps, a_labels, b_labels, b_records,
            baseline_kappa=0.275, corpus_a_count=100, corpus_b_count=10,
        )
        assert result.agreement_count == 1
        assert result.disagreement_count == 1
        assert result.agreement_rate == 0.5

    def test_systematic_divergence_detected(self) -> None:
        overlaps = [
            IncidentOverlap("INC-001", "ASIB-001", OverlapMethod.URL, "m"),
            IncidentOverlap("INC-002", "ASIB-002", OverlapMethod.URL, "m"),
            IncidentOverlap("INC-003", "ASIB-003", OverlapMethod.URL, "m"),
        ]
        a_labels = {"INC-001": "LLM05", "INC-002": "LLM05", "INC-003": "LLM01"}
        b_labels = {"ASIB-001": "LLM03", "ASIB-002": "LLM03", "ASIB-003": "LLM01"}
        b_records = {
            f"ASIB-{i:03d}": _make_record(f"ASIB-{i:03d}", text=f"Inc {i}")
            for i in range(1, 4)
        }

        result = compute_agreement(
            overlaps, a_labels, b_labels, b_records,
            baseline_kappa=0.275, corpus_a_count=100, corpus_b_count=10,
        )
        assert len(result.systematic_divergences) >= 1
        assert any(d.count >= 2 for d in result.systematic_divergences)

    def test_empty_overlap_produces_zero_rate(self) -> None:
        result = compute_agreement(
            [], {}, {}, {},
            baseline_kappa=0.275, corpus_a_count=100, corpus_b_count=10,
        )
        assert result.overlap_count == 0
        assert result.agreement_rate == 0.0

    def test_baseline_kappa_propagated(self) -> None:
        result = compute_agreement(
            [], {}, {}, {},
            baseline_kappa=0.275, corpus_a_count=6674, corpus_b_count=46,
        )
        assert result.baseline_kappa == 0.275
        assert result.corpus_a_incident_count == 6674
        assert result.corpus_b_incident_count == 46
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_corpus_b_corroboration.py -v`
Expected: PASS (10 tests total: 5 overlap + 5 agreement)

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_corpus_b_corroboration.py
git commit -m "test(decide): corpus B agreement computation tests (Plan 6)"
```

---

### Task 6: CLI Corroborate Command

**Files:**
- Modify: `engine/cli/pipeline.py` (add `corroborate` command)
- Modify: `engine/cli/main.py` (register command)

- [ ] **Step 1: Write the CLI corroborate command test**

Add to `tests/unit/test_pipeline_cli.py`:

```python
from click.testing import CliRunner


def test_corroborate_requires_cycle(cli_runner: CliRunner) -> None:
    from engine.cli.pipeline import corroborate
    result = cli_runner.invoke(corroborate, [])
    assert result.exit_code != 0
    assert "Missing" in result.output or "required" in result.output.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_pipeline_cli.py::test_corroborate_requires_cycle -v`
Expected: FAIL — `ImportError: cannot import name 'corroborate'`

- [ ] **Step 3: Implement the corroborate CLI command**

Add to `engine/cli/pipeline.py`, after the `repro_bundle_cmd` function:

```python
@click.command(name="corroborate")
@click.option("--cycle", required=True, type=click.Path(path_type=Path, exists=True))
@click.option("--corpus-b-dir", required=True, type=click.Path(path_type=Path, exists=True),
              help="Path to vendored corpus B snapshot directory")
@click.option("--execute", is_flag=True, default=False,
              help="Execute corroboration (without flag, validates prerequisites only)")
def corroborate(cycle: Path, corpus_b_dir: Path, execute: bool) -> None:
    """Run corpus B corroboration cross-check (Plan 6).

    Classifies corpus B through Stage-1 (+ Stage-2 if available),
    detects incident overlap with corpus A, computes agreement,
    and writes the corroboration artifact.

    Corpus B is qualitative corroboration only — NEVER a posterior input.
    """
    prereg = cycle / "prereg"
    if not (prereg / "rubric.json").exists():
        raise click.ClickException("prereg/rubric.json not found — rubric must be frozen")

    classify_dir = cycle / "classify"
    corpus_a_labels_path = classify_dir / "labeled_incidents.json"
    if not corpus_a_labels_path.exists():
        raise click.ClickException(
            "classify/labeled_incidents.json not found — run classify first"
        )

    results_dir = cycle / "results"
    conc_path = results_dir / "concordance.json"
    if not conc_path.exists():
        raise click.ClickException(
            "results/concordance.json not found — run decide first"
        )

    click.echo(f"Corpus B corroboration: loading from {corpus_b_dir}")

    if not execute:
        click.echo(
            "Corroborate: prerequisites satisfied. "
            "Run with --execute to compute corroboration."
        )
        return

    click.echo("Executing corpus B corroboration...")
    try:
        import glob

        from engine.adapters.owasp_asi import OWASPASIAdapter
        from engine.classify.classifier import build_rules_from_rubric, classify_real
        from engine.cli.pipeline_executor import _load_manifest
        from engine.decide.corpus_b_corroboration import (
            compute_agreement,
            detect_overlaps,
        )
        from engine.prereg.rubric_io import read_rubric

        # Load frozen rubric and manifest
        rubric = read_rubric(prereg / "rubric.json")
        manifest = _load_manifest(prereg / "manifest.json")
        confidence_threshold = manifest.confidence_threshold

        # Load corpus B via adapter
        adapter = OWASPASIAdapter(corpus_b_dir)
        corpus_b_incidents = list(adapter.iter_incidents())
        click.echo(f"Loaded {len(corpus_b_incidents)} corpus B incidents")

        # Stage-1 classification of corpus B
        rules = build_rules_from_rubric(rubric, confidence_threshold=confidence_threshold)
        b_result = classify_real(tuple(corpus_b_incidents), rules)
        click.echo(f"Stage-1 classified corpus B: {len(b_result.classifications)} classifications")

        classification_stages = "stage1"

        # Stage-2 routing (if available)
        stage2_config = prereg / "stage2_manifest.json"
        if stage2_config.exists():
            from engine.cli.pipeline_executor import merge_classifications, route_to_stage2

            all_b_ids = {inc.id for inc in corpus_b_incidents}
            low_conf_ids = route_to_stage2(
                b_result.classifications, all_b_ids,
                confidence_threshold=confidence_threshold,
            )
            click.echo(f"Stage-2 candidates: {len(low_conf_ids)} corpus B incidents")

            if low_conf_ids:
                try:
                    import os

                    from engine.classify.cost_tracker import CostTracker
                    from engine.classify.runpod_client import HttpRunPodClient
                    from engine.classify.stage2 import Stage2Classifier
                    from engine.classify.stage2_manifest import Stage2Manifest
                    from engine.cli.secrets import load_secret

                    s2_manifest = Stage2Manifest.read(stage2_config)
                    api_key = load_secret("runpod/api-key", env_var="RUNPOD_API_KEY")
                    endpoint_id = os.environ.get("RUNPOD_ENDPOINT_ID", "")

                    client = HttpRunPodClient(
                        api_key=api_key,
                        endpoint_id=endpoint_id,
                        model_name=s2_manifest.model_identity,
                    )
                    tracker = CostTracker(ceiling_usd=s2_manifest.cost_ceiling_usd)
                    classifier = Stage2Classifier(
                        client=client,
                        cost_tracker=tracker,
                        rubric_json=(prereg / "rubric.json").read_text(),
                        model_identity=s2_manifest.model_identity,
                        weight_provenance_hash=s2_manifest.weight_provenance_hash,
                        prng_seed=s2_manifest.prng_seed,
                    )

                    s2_incidents = tuple(i for i in corpus_b_incidents if i.id in low_conf_ids)
                    rubric_hash = manifest.rubric_hash or ""
                    click.echo(f"Stage-2: classifying {len(s2_incidents)} corpus B incidents...")

                    s2_results = tuple(
                        classifier.classify(inc, rubric_hash) for inc in s2_incidents
                    )
                    client.close()

                    merged = merge_classifications(
                        b_result.classifications, s2_results, confidence_threshold,
                    )
                    from engine.classify.stub import ClassificationResult
                    b_result = ClassificationResult(
                        classifications=merged,
                        classifier_version=b_result.classifier_version,
                        classifier_rule_hash=b_result.classifier_rule_hash,
                    )
                    classification_stages = "stage1+stage2"
                    click.echo(f"Stage-2 complete for corpus B ({len(s2_results)} results)")
                except (RuntimeError, OSError) as exc:
                    click.echo(
                        f"Stage-2 unavailable for corpus B ({exc}); "
                        f"proceeding with Stage-1 only"
                    )

        # Write corpus B classification artifact
        b_labeled = [
            {
                "incident_id": c.incident_id,
                "entry_id": c.entry_id,
                "confidence": c.confidence,
                "stage": c.stage,
                "rationale": c.rationale,
                "stratum": "corroboration",
            }
            for c in b_result.classifications
        ]
        b_labeled_path = classify_dir / "corpus_b_labeled.json"
        b_labeled_path.write_text(json.dumps(b_labeled, indent=2) + "\n")
        click.echo(f"Corpus B classifications written to {b_labeled_path}")

        # Build label maps: incident_id → highest-confidence entry_id
        a_labels_raw = json.loads(corpus_a_labels_path.read_text())
        a_label_map: dict[str, str] = {}
        a_label_conf: dict[str, float] = {}
        for rec in a_labels_raw:
            iid = rec["incident_id"]
            conf = rec["confidence"]
            if iid not in a_label_map or conf > a_label_conf.get(iid, -1.0):
                a_label_map[iid] = rec["entry_id"]
                a_label_conf[iid] = conf

        b_label_map: dict[str, str] = {}
        b_label_conf: dict[str, float] = {}
        for c in b_result.classifications:
            if c.incident_id not in b_label_map or c.confidence > b_label_conf.get(c.incident_id, -1.0):
                b_label_map[c.incident_id] = c.entry_id
                b_label_conf[c.incident_id] = c.confidence

        # Load corpus A incidents for overlap detection
        snapshot_dirs = list((cycle / "corpora" / "genai_agentic").iterdir())
        if not snapshot_dirs:
            raise click.ClickException("No corpus A snapshot found")
        from engine.adapters.genai_agentic import GenAIAgenticAdapter
        corpus_a_adapter = GenAIAgenticAdapter(snapshot_dirs[0], "2099-12-31")
        corpus_a_incidents = list(corpus_a_adapter.iter_incidents())

        # Detect overlaps
        overlaps = detect_overlaps(corpus_a_incidents, corpus_b_incidents)
        click.echo(f"Detected {len(overlaps)} incident overlaps between corpora")

        # Get baseline kappa
        conc_data = json.loads(conc_path.read_text())
        baseline_kappa = conc_data.get("weighted_kappa_median", 0.0) or 0.0

        # Compute agreement
        b_records_map = {inc.id: inc for inc in corpus_b_incidents}
        corroboration = compute_agreement(
            overlaps=overlaps,
            corpus_a_labels=a_label_map,
            corpus_b_labels=b_label_map,
            corpus_b_records=b_records_map,
            baseline_kappa=baseline_kappa,
            corpus_a_count=len(corpus_a_incidents),
            corpus_b_count=len(corpus_b_incidents),
            classification_stages=classification_stages,
        )

        # Write corroboration artifact
        results_dir.mkdir(parents=True, exist_ok=True)
        artifact = {
            "corpus_b_incident_count": corroboration.corpus_b_incident_count,
            "corpus_a_incident_count": corroboration.corpus_a_incident_count,
            "overlap_count": corroboration.overlap_count,
            "classification_stages_used": corroboration.classification_stages_used,
            "agreement_count": corroboration.agreement_count,
            "disagreement_count": corroboration.disagreement_count,
            "agreement_rate": corroboration.agreement_rate,
            "baseline_kappa": corroboration.baseline_kappa,
            "overlap_method_limitations": list(corroboration.overlap_method_limitations),
            "per_incident": [
                {
                    "corpus_a_id": a.corpus_a_id,
                    "corpus_b_id": a.corpus_b_id,
                    "corpus_b_title": a.corpus_b_title,
                    "match_method": a.match_method,
                    "corpus_a_label": a.corpus_a_label,
                    "corpus_b_label": a.corpus_b_label,
                    "corpus_b_native_labels": list(a.corpus_b_native_labels),
                    "agrees": a.agrees,
                }
                for a in corroboration.per_incident
            ],
            "systematic_divergences": [
                {
                    "pattern": d.pattern,
                    "count": d.count,
                    "incidents": list(d.incidents),
                }
                for d in corroboration.systematic_divergences
            ],
        }
        artifact_path = results_dir / "corpus_b_corroboration.json"
        artifact_path.write_text(json.dumps(artifact, indent=2) + "\n")
        click.echo(f"Corroboration artifact written to {artifact_path}")
        click.echo(
            f"Result: {corroboration.overlap_count} shared incidents, "
            f"{corroboration.agreement_count} agree, "
            f"{corroboration.disagreement_count} disagree "
            f"(rate={corroboration.agreement_rate:.2f})"
        )
        if corroboration.systematic_divergences:
            click.echo("Systematic divergences detected:")
            for d in corroboration.systematic_divergences:
                click.echo(f"  {d.pattern} ({d.count} incidents)")

    except Exception as e:
        raise click.ClickException(f"Corroboration failed: {e}") from e
```

- [ ] **Step 4: Register the command in main.py**

In `engine/cli/main.py`, add:

```python
from engine.cli.pipeline import corroborate

cli.add_command(corroborate)
```

Follow the existing pattern for how other commands (e.g., `decide_real`, `report_cmd`) are registered.

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_pipeline_cli.py::test_corroborate_requires_cycle -v`
Expected: PASS

- [ ] **Step 6: Run full suite**

Run: `uv run pytest -v && uv run mypy engine tests && uv run ruff check .`
Expected: All green

- [ ] **Step 7: Commit**

```bash
git add engine/cli/pipeline.py engine/cli/main.py tests/unit/test_pipeline_cli.py
git commit -m "feat(cli): add corroborate command for corpus B cross-check (Plan 6)"
```

---

### Task 7: Report Integration

**Files:**
- Modify: `engine/report/render.py`
- Modify: `tests/unit/test_report.py`

- [ ] **Step 1: Write failing test for corpus B report section**

Add to `tests/unit/test_report.py`:

```python
def test_render_report_with_corpus_b_section() -> None:
    from engine.report.render import ReportInputs, render_report

    inputs = _make_minimal_report_inputs()
    inputs_with_cb = ReportInputs(
        cycle_id=inputs.cycle_id,
        engine_version=inputs.engine_version,
        measurability_map=inputs.measurability_map,
        concordance=inputs.concordance,
        selection_bias=inputs.selection_bias,
        robustness=inputs.robustness,
        twin_agreement=inputs.twin_agreement,
        non_publishable=inputs.non_publishable,
        corpus_b_corroboration={
            "overlap_count": 5,
            "agreement_count": 3,
            "disagreement_count": 2,
            "agreement_rate": 0.6,
            "corpus_b_incident_count": 46,
            "baseline_kappa": 0.275,
            "systematic_divergences": [],
        },
    )
    report = render_report(inputs_with_cb)
    assert "## Corpus B Corroboration" in report
    assert "5 shared incidents" in report
    assert "3 agree" in report
    assert "NOT a posterior input" in report or "not a posterior input" in report.lower()
    assert "frame-blind" in report
```

Use the existing `_make_minimal_report_inputs` helper from the test file (or create a minimal fixture matching the existing test patterns).

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_report.py::test_render_report_with_corpus_b_section -v`
Expected: FAIL — `ReportInputs` does not accept `corpus_b_corroboration`

- [ ] **Step 3: Add corpus_b_corroboration field to ReportInputs**

In `engine/report/render.py`, modify the `ReportInputs` dataclass:

```python
@dataclass(frozen=True, slots=True)
class ReportInputs:
    cycle_id: str
    engine_version: str
    measurability_map: MeasurabilityMap
    concordance: ConcordanceResult
    selection_bias: SelectionBiasDisclosure
    robustness: RobustnessSpread | None
    twin_agreement: TwinAgreement | None
    non_publishable: bool
    rollup_results: tuple[RollupResult, ...] = ()
    prereg_diff: PreregDiff | None = None
    runpod_cost_usd: float | None = None
    cost_ceiling_usd: float | None = None
    corpus_b_corroboration: dict[str, object] | None = None
```

- [ ] **Step 4: Add corpus B section to render_report()**

In `engine/report/render.py`, add the following section to `render_report()` — insert after the RunPod cost section and before the Threats to Validity section:

```python
    # Corpus B corroboration (Plan 6, HANDOFF §4, §5.5)
    if inputs.corpus_b_corroboration is not None:
        cb = inputs.corpus_b_corroboration
        lines.append("\n## Corpus B Corroboration\n")
        lines.append(
            "Declared qualitative artifact — NOT a posterior input "
            "(HANDOFF §4, §5.4).\n\n"
        )
        overlap = cb.get("overlap_count", 0)
        agree = cb.get("agreement_count", 0)
        disagree = cb.get("disagreement_count", 0)
        rate = cb.get("agreement_rate", 0.0)
        b_count = cb.get("corpus_b_incident_count", 0)
        lines.append(
            f"Corpus B incidents: {b_count}. "
            f"Shared with corpus A: {overlap}.\n"
        )
        if overlap > 0:
            lines.append(
                f"Label agreement on shared incidents: "
                f"{agree} agree, {disagree} disagree "
                f"(rate = {rate:.0%}).\n"
            )
        else:
            lines.append("No shared incidents detected.\n")

        baseline = cb.get("baseline_kappa", 0.0)
        lines.append(
            f"\nContext: cycle headline kappa = {baseline:.3f}. "
            f"Agreement reporting at N = {overlap} is qualitative, "
            f"not statistical.\n"
        )

        lines.append(
            "\nNote: 3 entries are frame-blind (LLM04, LLM08, LLM10). "
            "Agreement on incidents classified to these entries is reported "
            "but has no bearing on posterior estimates.\n"
        )

        divergences = cb.get("systematic_divergences", [])
        if divergences:
            lines.append("\nSystematic divergences (published finding):\n")
            for d in divergences:
                if isinstance(d, dict):
                    lines.append(
                        f"- {d.get('pattern', 'unknown')} "
                        f"({d.get('count', 0)} incidents)\n"
                    )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_report.py -v`
Expected: PASS (all report tests)

- [ ] **Step 6: Run full suite**

Run: `uv run pytest -v && uv run mypy engine tests && uv run ruff check .`
Expected: All green

- [ ] **Step 7: Commit**

```bash
git add engine/report/render.py tests/unit/test_report.py
git commit -m "feat(report): add corpus B corroboration section to report renderer (Plan 6)"
```

---

### Task 8: End-to-End Execution and Artifact Production

**Files:**
- Produce: `projects/owasp-llm/cycles/2026/classify/corpus_b_labeled.json`
- Produce: `projects/owasp-llm/cycles/2026/results/corpus_b_corroboration.json`
- Modify: `projects/owasp-llm/cycles/2026/results/report.md` (regenerated)

- [ ] **Step 1: Run the corroborate command**

First, find the vendored corpus B snapshot directory:

```bash
CORPUS_B_DIR=$(ls -d projects/owasp-llm/cycles/2026/corpora/owasp_asi/*/ | head -1)
echo "Corpus B dir: $CORPUS_B_DIR"
```

Then execute:

```bash
uv run python -m engine.cli.main corroborate \
  --cycle projects/owasp-llm/cycles/2026 \
  --corpus-b-dir "$CORPUS_B_DIR" \
  --execute
```

Expected output: incident count, classification count, overlap count, agreement summary.

- [ ] **Step 2: Verify the corroboration artifact**

```bash
uv run python -c "
import json
from pathlib import Path
data = json.loads(Path('projects/owasp-llm/cycles/2026/results/corpus_b_corroboration.json').read_text())
print(f'Corpus B incidents: {data[\"corpus_b_incident_count\"]}')
print(f'Overlaps found: {data[\"overlap_count\"]}')
print(f'Agreement: {data[\"agreement_count\"]}/{data[\"overlap_count\"]} ({data[\"agreement_rate\"]:.0%})')
print(f'Classification stages: {data[\"classification_stages_used\"]}')
print(f'Systematic divergences: {len(data[\"systematic_divergences\"])}')
print(f'Baseline kappa context: {data[\"baseline_kappa\"]}')
for inc in data['per_incident']:
    agree_str = '✓' if inc['agrees'] else '✗'
    print(f'  {agree_str} {inc[\"corpus_b_id\"]}: A={inc[\"corpus_a_label\"]} B={inc[\"corpus_b_label\"]} ({inc[\"match_method\"]})')
"
```

- [ ] **Step 3: Regenerate the report with corpus B section**

Modify the `report_cmd` in `engine/cli/pipeline.py` to load the corroboration artifact if it exists. In the `report_cmd` function, before the `inputs = ReportInputs(...)` line, add:

```python
        corpus_b_corr = None
        cb_path = results_dir / "corpus_b_corroboration.json"
        if cb_path.exists():
            corpus_b_corr = json.loads(cb_path.read_text())
```

And add `corpus_b_corroboration=corpus_b_corr,` to the `ReportInputs(...)` constructor call.

Then regenerate:

```bash
uv run python -m engine.cli.main report --cycle projects/owasp-llm/cycles/2026
```

- [ ] **Step 4: Verify report contains corpus B section**

```bash
grep -A 10 "## Corpus B Corroboration" projects/owasp-llm/cycles/2026/results/report.md
```

Expected: the Corpus B Corroboration section with overlap count, agreement rate, and "NOT a posterior input" text.

- [ ] **Step 5: Run regression test to confirm inference isolation**

```bash
uv run pytest tests/unit/test_corpus_b_regression.py -v
```

Expected: PASS — inference.py still clean.

- [ ] **Step 6: Run full test suite**

```bash
uv run pytest -v && uv run mypy engine tests && uv run ruff check .
```

Expected: All green

- [ ] **Step 7: Commit artifacts and report integration**

```bash
git add engine/cli/pipeline.py
git add projects/owasp-llm/cycles/2026/classify/corpus_b_labeled.json
git add projects/owasp-llm/cycles/2026/results/corpus_b_corroboration.json
git add projects/owasp-llm/cycles/2026/results/report.md
git commit -m "data(cycle): corpus B corroboration artifact + report integration (Plan 6)"
```

---

### Task 9: Methodology Changelog, CI Verification, and Tag

**Files:**
- Modify: `docs/METHODOLOGY-CHANGELOG.md`
- Modify: `engine/version.py`

- [ ] **Step 1: Bump engine version**

In `engine/version.py`, change:

```python
__version__ = "1.2.0"
```

- [ ] **Step 2: Add methodology changelog entry**

Prepend to `docs/METHODOLOGY-CHANGELOG.md`, before the existing `## 1.0.0` entry:

```markdown
## 1.2.0 (Plan 6, 2026-05-22): Corpus B corroboration artifact

- OWASP ASI Agentic Exploits & Incidents tracker adapter (`engine/adapters/owasp_asi.py`): parses ~46 human-curated incidents from vendored Markdown snapshot into canonical IncidentRecords.
- Corpus B incidents classified through the same Stage-1 (+ Stage-2 if available) pipeline as corpus A for consistency.
- Incident overlap detection via URL/CVE/title matching between corpus A and corpus B.
- Per-incident agreement computation with systematic divergence detection (HANDOFF §4: divergence is a published finding, never a silent adjustment).
- Report section added: declared qualitative artifact, NOT a posterior input.
- Regression test enforces that `engine/model/inference.py` never imports corpus B artifacts.

Methodology decision: corpus B has ~46 incidents (N too small for statistical testing). Agreement is reported as raw counts (N shared, N agree, N disagree) with qualitative interpretation against the baseline kappa. No kappa, no hypothesis test, no confidence intervals on the agreement rate. Overlap detection limitations are declared in the artifact.

```

- [ ] **Step 3: Commit**

```bash
git add engine/version.py docs/METHODOLOGY-CHANGELOG.md
git commit -m "docs: methodology changelog 1.2.0 + version bump (Plan 6)"
```

- [ ] **Step 4: Run full CI-equivalent checks locally**

```bash
uv run ruff check . && uv run mypy engine tests && uv run pytest -v
```

Expected: All green

- [ ] **Step 5: Verify CI actually executes** (Plan 1 erratum lesson)

Push and confirm CI runs in the GitHub Actions tab. Do NOT trust local-only verification — the Plan 1 erratum showed that a workflow file's presence does not prove execution.

```bash
git push origin main
```

Then verify: `gh run list --limit 1` shows a green check.

- [ ] **Step 6: Tag**

```bash
git tag -a v1.2.0-plan6 -m "Plan 6: corpus B corroboration artifact"
git push origin v1.2.0-plan6
```

---

## Self-Review Checklist

### 1. Spec coverage

| PRD §7 Requirement | Task |
|---|---|
| §7.4.1 — Adapter emitting canonical records | Task 2 |
| §7.4.2 — Cross-check tool computing overlap and agreement | Tasks 4, 5 |
| §7.4.3 — Output artifact `corpus_b_corroboration.json` | Task 8 |
| §7.4.4 — Report integration | Task 7, 8 |
| §7.4.5 — Regression test on inference.py | Task 1 |
| §7.5.1 — Corroboration artifact computed and reported | Task 8 |
| §7.5.2 — Inference module unchanged | Task 1 |
| §7.5.3 — Report reads as declared artifact | Task 7 |
| §7.5.4 — Tag v1.2.0-plan6 | Task 9 |

### 2. Placeholder scan

No TBD, TODO, or "implement later" found. All code steps contain complete implementations.

### 3. Type consistency

- `ASIIncident` — defined in Task 2, consumed in Task 2 only (internal to adapter).
- `IncidentOverlap`, `OverlapMethod` — defined in Task 4, consumed in Tasks 5, 6.
- `CorpusBCorroboration`, `IncidentAgreement`, `SystematicDivergence` — defined in Task 4, consumed in Task 6.
- `compute_agreement()` signature matches Task 5 tests and Task 6 CLI calls.
- `detect_overlaps()` signature matches Task 4 tests and Task 6 CLI calls.
- `ReportInputs.corpus_b_corroboration` — `dict[str, object] | None`, matches Task 7 test and Task 8 CLI.

### 4. Inherited constraint coverage

| Constraint | Enforcement |
|---|---|
| (i) Never a posterior input | Task 1 regression test |
| (ii) N is dozens, agreement reporting | Task 5 `compute_agreement` returns raw counts, no kappa |
| (iii) Systematic divergence is published | Task 4 `SystematicDivergence` dataclass, Task 7 report section |
| (iv) Text-match fallback with declared limitations | Task 4 multi-strategy overlap detection, `overlap_method_limitations` in artifact |
| (v) Same two-stage pipeline | Task 6 CLI runs Stage-1 + Stage-2 on corpus B |
| (vi) Baseline kappa context | Task 4 `CorpusBCorroboration.baseline_kappa`, Task 7 report context line |
| (vii) Frame-blind entries reportable but no posterior impact | Task 7 report section notes frame-blind caveat; Task 1 regression test prevents posterior contamination |
