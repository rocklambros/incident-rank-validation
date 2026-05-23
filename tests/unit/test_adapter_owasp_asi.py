"""Unit tests for the OWASP ASI corpus B adapter."""
# ruff: noqa: E501
from __future__ import annotations

import textwrap
from pathlib import Path

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
