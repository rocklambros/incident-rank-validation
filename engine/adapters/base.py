"""Abstract base class for corpus adapters.

All corpus data flows through a CorpusAdapter.  See HANDOFF §5.1.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

from engine.model.overlap import OverlapWeights
from engine.schema import BiasProfile, EntryDefinition, IncidentRecord, StratumSize


class CorpusAdapter(ABC):
    """Interface that every corpus adapter must implement."""

    @abstractmethod
    def iter_incidents(self) -> Iterator[IncidentRecord]:
        """Yield canonical incident records."""
        ...

    @abstractmethod
    def bias_profiles(self) -> tuple[BiasProfile, ...]:
        """Return declared bias profiles, one per stratum."""
        ...

    @abstractmethod
    def stratum_sizes(self) -> dict[str, StratumSize]:
        """Return the exposure term per stratum."""
        ...

    @abstractmethod
    def entry_definitions(self) -> tuple[EntryDefinition, ...]:
        """Return the taxonomy entries for this adapter."""
        ...

    @abstractmethod
    def overlap_weights(self) -> OverlapWeights:
        """Return the declared FP leakage structure."""
        ...
