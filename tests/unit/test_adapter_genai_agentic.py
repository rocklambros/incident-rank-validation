"""Tests for the genai_agentic corpus A adapter.

All test counts and field references are derived from the audit in
HANDOFF §3 (owasp-mapping-quality-audit.md, N=7,714) and confirmed
against the vendored snapshot in Task 0 Step 7.
"""
from __future__ import annotations

import pytest

from engine.adapters.genai_agentic_bias import (
    build_bias_profiles,
    is_bare_llm03_contaminated,
    is_double_default_contaminated,
)
from engine.schema import BiasProfile


class TestBiasProfiles:
    """Per-stratum bias profile declarations (HANDOFF §3 Mixture, §4 row, §5.1)."""

    def test_one_profile_per_stratum(self) -> None:
        profiles = build_bias_profiles()
        strata = {p.stratum for p in profiles}
        assert "security" in strata
        assert "ai-harm" in strata
        assert len(profiles) >= 2

    def test_profiles_are_biasprofile_instances(self) -> None:
        for p in build_bias_profiles():
            assert isinstance(p, BiasProfile)

    def test_security_stratum_declares_contamination(self) -> None:
        sec = next(p for p in build_bias_profiles() if p.stratum == "security")
        assert "LLM03" in sec.contamination_description
        assert sec.quarantine_rule != ""

    def test_ai_harm_stratum_declares_known_blind_spots(self) -> None:
        ah = next(p for p in build_bias_profiles() if p.stratum == "ai-harm")
        assert len(ah.known_blind_spots) > 0

    def test_construction_time_validation_rejects_empty_stratum(self) -> None:
        """C2 pattern: invalid input fails at construction, not at use time."""
        from engine.adapters.genai_agentic_bias import _validate_bias_profile
        with pytest.raises(ValueError, match="stratum"):
            _validate_bias_profile(BiasProfile(
                stratum="",
                description="empty",
                known_blind_spots=(),
                contamination_description="none",
                quarantine_rule="none",
            ))


class TestQuarantinePredicates:
    """Contamination quarantine rules (HANDOFF §3 F2, §5.2 out-of-scope sink)."""

    def test_bare_llm03_detected(self) -> None:
        assert is_bare_llm03_contaminated(["LLM03"]) is True

    def test_bare_llm03_not_triggered_on_multi_label(self) -> None:
        assert is_bare_llm03_contaminated(["LLM03", "LLM05"]) is False

    def test_double_default_detected(self) -> None:
        assert is_double_default_contaminated(["LLM03", "ASI04"]) is True
        assert is_double_default_contaminated(["ASI04", "LLM03"]) is True

    def test_double_default_not_triggered_on_triple(self) -> None:
        assert is_double_default_contaminated(["LLM03", "ASI04", "LLM05"]) is False

    def test_empty_labels_not_contaminated(self) -> None:
        assert is_bare_llm03_contaminated([]) is False
        assert is_double_default_contaminated([]) is False
