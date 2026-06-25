"""Tests for oracle render section + PROVISIONAL banner (Plan 8d Task 7)."""
from __future__ import annotations

from engine.report.render import _render_oracle_lines


def test_render_oracle_none_is_empty() -> None:
    assert _render_oracle_lines(None) == []


def test_render_oracle_pass_shows_section_no_banner() -> None:
    oracle: dict[str, object] = {
        "provisional": False,
        "deliverables": [
            {"name": "incidence", "status": "PASS", "metric": "kendall_tau=1.000", "detail": ""},
            {"name": "sigma_u", "status": "PASS", "metric": "|delta|=0.10", "detail": ""},
        ],
    }
    text = "".join(_render_oracle_lines(oracle))
    assert "Oracle Consistency Check" in text
    assert "incidence" in text
    assert "PASS" in text
    assert "PROVISIONAL" not in text


def test_render_oracle_fail_shows_provisional_banner() -> None:
    oracle: dict[str, object] = {
        "provisional": True,
        "deliverables": [
            {"name": "incidence", "status": "FAIL", "metric": "kendall_tau=-1.0", "detail": ""},
            {"name": "sigma_u", "status": "SKIP", "metric": "n/a", "detail": "missing"},
        ],
    }
    text = "".join(_render_oracle_lines(oracle))
    assert "PROVISIONAL" in text
    assert "incidence" in text
    assert "FAIL" in text
