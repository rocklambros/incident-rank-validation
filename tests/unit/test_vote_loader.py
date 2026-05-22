from __future__ import annotations

from pathlib import Path

import numpy as np

from engine.vote.loader import load_vote_data, VoteData


FIXTURE = Path(__file__).parent.parent / "fixtures" / "vote_fixture.xlsx"


class TestVoteLoader:
    def test_load_shape(self) -> None:
        vd = load_vote_data(FIXTURE, sheet_name="Raw Results (Anonymized)")
        assert isinstance(vd, VoteData)
        assert vd.rankings.shape == (10, 5)
        assert len(vd.entry_ids) == 5

    def test_entry_ids(self) -> None:
        vd = load_vote_data(FIXTURE, sheet_name="Raw Results (Anonymized)")
        assert vd.entry_ids == ("LLM01", "LLM02", "LLM03", "LLM04", "LLM05")

    def test_rankings_are_ranks(self) -> None:
        vd = load_vote_data(FIXTURE, sheet_name="Raw Results (Anonymized)")
        for row in range(vd.rankings.shape[0]):
            sorted_row = sorted(vd.rankings[row])
            assert sorted_row == [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_n_respondents(self) -> None:
        vd = load_vote_data(FIXTURE, sheet_name="Raw Results (Anonymized)")
        assert vd.n_respondents == 10

    def test_missing_sheet_raises(self) -> None:
        try:
            load_vote_data(FIXTURE, sheet_name="Nonexistent")
            assert False, "should raise"
        except ValueError:
            pass

    def test_column_id_mapping(self) -> None:
        """R4: XLSX may use human-readable headers; mapping normalizes to canonical IDs."""
        mapping = {"LLM01": "LLM01", "Prompt Injection": "LLM01", "LLM02": "LLM02"}
        vd = load_vote_data(
            FIXTURE,
            sheet_name="Raw Results (Anonymized)",
            column_id_mapping=mapping,
        )
        assert all(eid.startswith("LLM") for eid in vd.entry_ids)
