"""SyntheticStressAdapter: stress corpus for multi-tier, N/A-kappa, and multi-target overlap.

12 entries across 2 strata:
  - STR01-STR03: measurable (3 < meaningful_kappa_n=4, exercises N/A branch)
  - STR04-STR09: frame_blind — invisible to corpus sampling frame
  - STR10-STR12: classifier-blind — in scope but zero incidents (classifier cannot detect them)

Multi-target leakage: STR07's FPs split 60% to STR01 and 40% to STR02.
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
    EntryDefinition(entry_id="STR01", name="Stress Entry One", frame_blind=False),
    EntryDefinition(entry_id="STR02", name="Stress Entry Two", frame_blind=False),
    EntryDefinition(entry_id="STR03", name="Stress Entry Three", frame_blind=False),
    EntryDefinition(entry_id="STR04", name="Stress Entry Four", frame_blind=True),
    EntryDefinition(entry_id="STR05", name="Stress Entry Five", frame_blind=True),
    EntryDefinition(entry_id="STR06", name="Stress Entry Six", frame_blind=True),
    EntryDefinition(entry_id="STR07", name="Stress Entry Seven", frame_blind=True),
    EntryDefinition(entry_id="STR08", name="Stress Entry Eight", frame_blind=True),
    EntryDefinition(entry_id="STR09", name="Stress Entry Nine", frame_blind=True),
    EntryDefinition(entry_id="STR10", name="Stress Entry Ten", frame_blind=False),
    EntryDefinition(entry_id="STR11", name="Stress Entry Eleven", frame_blind=False),
    EntryDefinition(entry_id="STR12", name="Stress Entry Twelve", frame_blind=False),
)

_STRATA = ("stratum_a", "stratum_b")

# Ground-truth incident counts per (stratum, entry_id).
# STR04-STR09 are frame_blind=True → zero incidents.
# STR10-STR12 are classifier-blind (frame_blind=False, zero incidents).
# STR01-STR03 are measurable but only 3 entries < meaningful_kappa_n=4 → N/A kappa.
_GROUND_TRUTH: dict[str, dict[str, int]] = {
    "stratum_a": {
        "STR01": 90,   # measurable — meaningful count
        "STR02": 55,   # measurable — moderate count
        "STR03": 30,   # measurable — lower count
        "STR04": 0,    # frame_blind
        "STR05": 0,    # frame_blind
        "STR06": 0,    # frame_blind
        "STR07": 0,    # frame_blind — FPs split to STR01/STR02 via overlap weights
        "STR08": 0,    # frame_blind
        "STR09": 0,    # frame_blind
        "STR10": 0,    # classifier-blind
        "STR11": 0,    # classifier-blind
        "STR12": 0,    # classifier-blind
    },
    "stratum_b": {
        "STR01": 50,   # measurable
        "STR02": 30,   # measurable
        "STR03": 15,   # measurable
        "STR04": 0,    # frame_blind
        "STR05": 0,    # frame_blind
        "STR06": 0,    # frame_blind
        "STR07": 0,    # frame_blind
        "STR08": 0,    # frame_blind
        "STR09": 0,    # frame_blind
        "STR10": 0,    # classifier-blind
        "STR11": 0,    # classifier-blind
        "STR12": 0,    # classifier-blind
    },
}

_STRATUM_SIZES: dict[str, StratumSize] = {
    "stratum_a": make_stratum_size(400),
    "stratum_b": make_stratum_size(200),
}

_BIAS_PROFILES: tuple[BiasProfile, ...] = (
    BiasProfile(
        stratum="stratum_a",
        description="Stress corpus stratum A; over-represents measurable entries.",
        known_blind_spots=("STR04", "STR05", "STR06", "STR07", "STR08", "STR09",
                           "STR10", "STR11", "STR12"),
        contamination_description="Stress fixture — no real contamination.",
        quarantine_rule="no quarantine rule for stress fixture",
    ),
    BiasProfile(
        stratum="stratum_b",
        description="Stress corpus stratum B; classifier cannot detect STR10-STR12.",
        known_blind_spots=("STR04", "STR05", "STR06", "STR07", "STR08", "STR09",
                           "STR10", "STR11", "STR12"),
        contamination_description="Stress fixture — no real contamination.",
        quarantine_rule="no quarantine rule for stress fixture",
    ),
)

# Multi-target leakage (M1): STR07's FPs split 60% to STR01 and 40% to STR02.
# Column sum for STR07: 0.6 + 0.4 = 1.0 (fully allocated, column-stochastic).
_OVERLAP_WEIGHTS = OverlapWeights(
    weights={
        "STR01": {"STR07": 0.6},
        "STR02": {"STR07": 0.4},
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
    """Generate one deterministic synthetic stress incident."""
    year = rng.randint(2020, 2024)
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)
    date = f"{year}-{month:02d}-{day:02d}"
    severity = rng.choice(["Critical", "High", "Medium", "Low"])
    source_class = rng.choice(["advisory", "cve", "harm-report"])
    quality = rng.choice(["curated", "reviewed", "auto"])
    text = (
        f"Stress incident {incident_id} for entry {entry_id} in {stratum}. "
        f"Severity: {severity}. Generated for stress testing."
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
        source_url=f"https://synthetic-stress.example/incidents/{incident_id}",
    )


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class SyntheticStressAdapter(CorpusAdapter):
    """Stress corpus adapter exercising multi-tier, N/A-kappa, and multi-target overlap.

    Parameters
    ----------
    seed:
        Integer seed for Python's :mod:`random` module.  The same seed
        always produces the same sequence of incidents.
    """

    def __init__(self, seed: int = 99) -> None:
        self._seed = seed

    # ------------------------------------------------------------------
    # CorpusAdapter interface
    # ------------------------------------------------------------------

    def iter_incidents(self) -> Iterator[IncidentRecord]:
        """Yield all stress incidents in deterministic order."""
        rng = random.Random(self._seed)
        counter = 0
        for stratum in _STRATA:
            for entry in _ENTRIES:
                count = _GROUND_TRUTH[stratum][entry.entry_id]
                for _ in range(count):
                    counter += 1
                    incident_id = f"STR-{counter:05d}"
                    yield _make_incident(rng, incident_id, stratum, entry.entry_id)

    def bias_profiles(self) -> tuple[BiasProfile, ...]:
        """Return the declared bias profiles for each stratum."""
        return _BIAS_PROFILES

    def stratum_sizes(self) -> dict[str, StratumSize]:
        """Return the exposure size per stratum."""
        return dict(_STRATUM_SIZES)

    def entry_definitions(self) -> tuple[EntryDefinition, ...]:
        """Return all 12 entry definitions (STR01-STR12)."""
        return _ENTRIES

    def overlap_weights(self) -> OverlapWeights:
        """Return multi-target leakage: STR07 FPs split 60% to STR01, 40% to STR02."""
        return _OVERLAP_WEIGHTS
