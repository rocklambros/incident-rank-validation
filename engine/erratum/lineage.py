"""Cross-cycle lineage check (HANDOFF §5.1)."""

from __future__ import annotations


class CrossCycleComparisonError(RuntimeError):
    """Raised when comparing cycles that are not lineage-compatible."""


def verify_lineage(
    cycle_id_a: str,
    cycle_id_b: str,
    taxonomy_hash_a: str,
    taxonomy_hash_b: str,
) -> None:
    """Refuse to compare cycles with different IDs or taxonomy hashes."""
    if cycle_id_a != cycle_id_b:
        raise CrossCycleComparisonError(
            f"Cannot compare cycles with different IDs: {cycle_id_a!r} vs {cycle_id_b!r}"
        )
    if taxonomy_hash_a != taxonomy_hash_b:
        raise CrossCycleComparisonError(
            f"Cannot compare cycles with different taxonomy hashes: "
            f"{taxonomy_hash_a!r} vs {taxonomy_hash_b!r}"
        )
