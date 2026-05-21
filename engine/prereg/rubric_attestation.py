"""Rubric-drafting attestation record.

Records whether the rubric drafter viewed any corpus samples before
drafting, and if so, which ones.  This is a pre-registration integrity
field: viewing corpus data before drafting a rubric compromises blinding.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RubricDraftingAttestation:
    """Attestation about corpus and vote-data exposure during rubric drafting."""

    viewed_corpus_before_drafting: bool
    viewed_corpus_details: str  # which samples, if any — empty string if none
    viewed_vote_data_before_drafting: bool  # HANDOFF §6 control 2
    viewed_vote_data_details: str  # which vote data, if any — empty string if none
