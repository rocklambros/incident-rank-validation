"""Unit tests for engine.safety.corpus_mode."""

from __future__ import annotations

import pytest

from engine.safety.corpus_mode import CorpusModeViolation, verify_corpus_mode


class TestVerifyCorpusMode:
    def test_synthetic_with_synthetic_adapter_passes(self) -> None:
        verify_corpus_mode("synthetic", "synthetic")

    def test_synthetic_with_synthetic_stress_adapter_passes(self) -> None:
        verify_corpus_mode("synthetic", "synthetic_stress")

    def test_real_with_real_adapter_passes(self) -> None:
        verify_corpus_mode("real", "cve_ghsa")

    def test_synthetic_with_real_adapter_raises(self) -> None:
        with pytest.raises(CorpusModeViolation, match="corpus-mode=synthetic"):
            verify_corpus_mode("synthetic", "cve_ghsa")

    def test_real_with_synthetic_adapter_raises(self) -> None:
        with pytest.raises(CorpusModeViolation, match="corpus-mode=real"):
            verify_corpus_mode("real", "synthetic")

    def test_real_with_synthetic_stress_adapter_raises(self) -> None:
        with pytest.raises(CorpusModeViolation, match="corpus-mode=real"):
            verify_corpus_mode("real", "synthetic_stress")

    def test_violation_is_runtime_error(self) -> None:
        with pytest.raises(RuntimeError):
            verify_corpus_mode("synthetic", "cve_ghsa")
