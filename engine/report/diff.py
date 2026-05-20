from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PreregDiff:
    deviations: tuple[str, ...]  # list of deviation descriptions

    @property
    def has_deviations(self) -> bool:
        return len(self.deviations) > 0

    def to_markdown(self) -> str:
        if not self.deviations:
            return "No deviations from pre-registration.\n"
        lines = ["## Pre-registration Deviations\n"]
        for d in self.deviations:
            lines.append(f"- {d}\n")
        return "".join(lines)
