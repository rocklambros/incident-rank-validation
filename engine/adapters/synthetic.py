"""SyntheticAdapter: deterministic synthetic corpus for testing.

Generates a fixed taxonomy of 6 entries across 2 strata with known
prevalence patterns, suitable for exercising the full Plan 1 synthetic cycle.
"""

from __future__ import annotations

import random
from collections.abc import Iterator

from engine.adapters.base import CorpusAdapter
from engine.model.overlap import OverlapWeights
from engine.schema import (
    BiasProfile,
    EntryDefinition,
    IncidentRecord,
    StratumSize,
    make_stratum_size,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ENTRIES: tuple[EntryDefinition, ...] = (
    EntryDefinition(entry_id="E01", name="Entry One", frame_blind=False),
    EntryDefinition(entry_id="E02", name="Entry Two", frame_blind=False),
    EntryDefinition(entry_id="E03", name="Entry Three", frame_blind=False),
    EntryDefinition(entry_id="E04", name="Entry Four", frame_blind=False),
    EntryDefinition(entry_id="E05", name="Entry Five", frame_blind=False),
    EntryDefinition(entry_id="E06", name="Entry Six", frame_blind=True),
)

_STRATA = ("stratum_a", "stratum_b")

# Ground-truth incident counts per (stratum, entry_id).
# E06 is frame_blind=True → zero incidents generated for it.
# stratum_a size=500, stratum_b size=300.
_GROUND_TRUTH: dict[str, dict[str, int]] = {
    "stratum_a": {
        "E01": 120,  # high prevalence
        "E02": 75,   # moderate
        "E03": 60,   # moderate
        "E04": 25,   # low
        "E05": 5,    # very low / classifier-blind territory
        "E06": 0,    # frame_blind — invisible
    },
    "stratum_b": {
        "E01": 80,   # high prevalence
        "E02": 45,   # moderate
        "E03": 35,   # moderate
        "E04": 12,   # low
        "E05": 2,    # very low
        "E06": 0,    # frame_blind — invisible
    },
}

_STRATUM_SIZES: dict[str, StratumSize] = {
    "stratum_a": make_stratum_size(500),
    "stratum_b": make_stratum_size(300),
}

_BIAS_PROFILES: tuple[BiasProfile, ...] = (
    BiasProfile(
        stratum="stratum_a",
        description="Security-focused corpus; over-represents infrastructure incidents.",
        known_blind_spots=("E05", "E06"),
        contamination_description="Some bare-label singletons present.",
        quarantine_rule="drop records with only a single native_label matching E02",
    ),
    BiasProfile(
        stratum="stratum_b",
        description="AI-harm corpus; under-represents low-severity entries.",
        known_blind_spots=("E04", "E05", "E06"),
        contamination_description="LLM default seed entries present.",
        quarantine_rule="drop bare E03 singletons from automated sources",
    ),
)

# E02's FPs leak into E01 at weight 0.3 (column-stochastic, no self-loop).
_OVERLAP_WEIGHTS = OverlapWeights(
    weights={
        "E01": {"E02": 0.3},
    }
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_incident(
    rng: random.Random,
    incident_id: str,
    stratum: str,
    entry_id: str,
) -> IncidentRecord:
    """Generate one deterministic synthetic incident."""
    year = rng.randint(2020, 2024)
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)
    date = f"{year}-{month:02d}-{day:02d}"
    severity = rng.choice(["Critical", "High", "Medium", "Low"])
    source_class = rng.choice(["advisory", "cve", "harm-report"])
    quality = rng.choice(["curated", "reviewed", "auto"])
    text = (
        f"Synthetic incident {incident_id} for entry {entry_id} in {stratum}. "
        f"Severity: {severity}. Generated for testing purposes."
    )
    return IncidentRecord(
        id=incident_id,
        date=date,
        text=text,
        severity=severity,
        source_class=source_class,
        corpus_stratum=stratum,
        quality=quality,
        native_labels=(entry_id,),
        source_url=f"https://synthetic.example/incidents/{incident_id}",
    )


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class SyntheticAdapter(CorpusAdapter):
    """Deterministic synthetic corpus adapter for unit and integration testing.

    Parameters
    ----------
    seed:
        Integer seed for Python's :mod:`random` module.  The same seed
        always produces the same sequence of incidents.
    """

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed

    # ------------------------------------------------------------------
    # CorpusAdapter interface
    # ------------------------------------------------------------------

    def iter_incidents(self) -> Iterator[IncidentRecord]:
        """Yield all synthetic incidents in deterministic order."""
        rng = random.Random(self._seed)
        counter = 0
        for stratum in _STRATA:
            for entry in _ENTRIES:
                count = _GROUND_TRUTH[stratum][entry.entry_id]
                for _ in range(count):
                    counter += 1
                    incident_id = f"SYN-{counter:05d}"
                    yield _make_incident(rng, incident_id, stratum, entry.entry_id)

    def bias_profiles(self) -> tuple[BiasProfile, ...]:
        """Return the declared bias profiles for each stratum."""
        return _BIAS_PROFILES

    def stratum_sizes(self) -> dict[str, StratumSize]:
        """Return the exposure size per stratum."""
        return dict(_STRATUM_SIZES)

    def entry_definitions(self) -> tuple[EntryDefinition, ...]:
        """Return all 6 entry definitions (E01–E06, E06 is frame_blind)."""
        return _ENTRIES

    def overlap_weights(self) -> OverlapWeights:
        """Return the FP leakage structure: E02 → E01 at 0.3."""
        return _OVERLAP_WEIGHTS
