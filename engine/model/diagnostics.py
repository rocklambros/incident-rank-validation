"""Lightweight diagnostics summary for NUTS inference results."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DiagnosticsSummary:
    """Compact summary of NUTS diagnostic checks."""

    max_r_hat: float
    min_ess_fraction: float
    divergences: int
    passed: bool
