"""Tests for system/user role split in Stage-2 prompt."""
from __future__ import annotations

from engine.classify.stage2_prompt import build_messages
from engine.schema import IncidentRecord

_RUBRIC_JSON = '{"entries": [{"entry_id": "LLM01", "canonical_name": "Prompt Injection", "in_scope": "test"}]}'


def _make_incident(text: str = "Test incident") -> IncidentRecord:
    return IncidentRecord(
        id="TEST-001",
        date="2025-01-01",
        text=text,
        severity="High",
        source_class="advisory",
        corpus_stratum="security",
        quality="curated",
        native_labels=(),
        source_url="https://example.com",
    )


class TestBuildMessages:
    def test_returns_two_messages(self) -> None:
        msgs = build_messages(_make_incident(), _RUBRIC_JSON)
        assert len(msgs) == 2

    def test_first_message_is_system_role(self) -> None:
        msgs = build_messages(_make_incident(), _RUBRIC_JSON)
        assert msgs[0]["role"] == "system"

    def test_second_message_is_user_role(self) -> None:
        msgs = build_messages(_make_incident(), _RUBRIC_JSON)
        assert msgs[1]["role"] == "user"

    def test_incident_text_only_in_user_message(self) -> None:
        msgs = build_messages(_make_incident("unique_test_text_xyz"), _RUBRIC_JSON)
        assert "unique_test_text_xyz" not in msgs[0]["content"]
        assert "unique_test_text_xyz" in msgs[1]["content"]

    def test_rubric_in_system_message(self) -> None:
        msgs = build_messages(_make_incident(), _RUBRIC_JSON)
        assert "LLM01" in msgs[0]["content"]

    def test_safety_rule_in_system_message(self) -> None:
        msgs = build_messages(_make_incident(), _RUBRIC_JSON)
        assert "CRITICAL SAFETY RULE" in msgs[0]["content"]

    def test_delimiters_in_user_message(self) -> None:
        msgs = build_messages(_make_incident(), _RUBRIC_JSON)
        assert "<<<INCIDENT_TEXT_BEGIN>>>" in msgs[1]["content"]
        assert "<<<INCIDENT_TEXT_END>>>" in msgs[1]["content"]

    def test_braces_in_incident_text_preserved(self) -> None:
        msgs = build_messages(_make_incident('cmd {arg} and {"key": "val"}'), _RUBRIC_JSON)
        assert "{arg}" in msgs[1]["content"]
        assert '{"key": "val"}' in msgs[1]["content"]

    def test_build_prompt_still_works(self) -> None:
        """Backward compat: build_prompt returns single string."""
        from engine.classify.stage2_prompt import build_prompt
        result = build_prompt(_make_incident(), _RUBRIC_JSON)
        assert isinstance(result, str)
        assert "LLM01" in result
