from __future__ import annotations


class CorpusModeViolation(RuntimeError):
    pass


def verify_corpus_mode(declared_mode: str, provenance_adapter: str) -> None:
    """Verify corpus mode matches provenance."""
    synthetic_adapters = {"synthetic", "synthetic_stress"}
    if declared_mode == "synthetic" and provenance_adapter not in synthetic_adapters:
        raise CorpusModeViolation(
            f"corpus-mode=synthetic but provenance adapter is {provenance_adapter!r}"
        )
    if declared_mode == "real" and provenance_adapter in synthetic_adapters:
        raise CorpusModeViolation(
            f"corpus-mode=real but provenance adapter is {provenance_adapter!r} (synthetic)"
        )
