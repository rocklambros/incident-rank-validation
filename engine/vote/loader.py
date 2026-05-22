"""Vote data XLSX loader (HANDOFF §5.4, §6 control 2)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class VoteData:
    rankings: npt.NDArray[np.float64]
    entry_ids: tuple[str, ...]
    n_respondents: int


def load_vote_data(
    xlsx_path: Path,
    sheet_name: str = "Raw Results (Anonymized)",
    column_id_mapping: dict[str, str] | None = None,
) -> VoteData:
    from openpyxl import load_workbook

    wb = load_workbook(xlsx_path, read_only=True, data_only=True)
    if sheet_name not in wb.sheetnames:
        wb.close()
        raise ValueError(
            f"Sheet '{sheet_name}' not found in {xlsx_path}. "
            f"Available: {wb.sheetnames}"
        )
    ws = wb[sheet_name]

    rows = list(ws.iter_rows(values_only=True))
    wb.close()

    if not rows:
        raise ValueError(f"Sheet '{sheet_name}' is empty")

    header = rows[0]
    raw_ids = tuple(str(h) for h in header[1:] if h is not None)
    if column_id_mapping:
        entry_ids = tuple(column_id_mapping.get(h, h) for h in raw_ids)
    else:
        entry_ids = raw_ids
    n_entries = len(entry_ids)

    data_rows: list[list[float]] = []
    for row in rows[1:]:
        vals = row[1:n_entries + 1]
        if all(v is not None for v in vals):
            data_rows.append([float(v) for v in vals])

    if not data_rows:
        raise ValueError("No respondent data found")

    rankings = np.array(data_rows, dtype=np.float64)
    return VoteData(
        rankings=rankings,
        entry_ids=entry_ids,
        n_respondents=rankings.shape[0],
    )
