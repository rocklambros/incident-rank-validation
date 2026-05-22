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


def compute_prereg_diff(
    prereg_primary_spec: str,
    actual_primary_spec: str,
    prereg_flag_tau: float,
    actual_flag_tau: float,
    prereg_measurability_min: int,
    actual_measurability_min: int,
    additional_deviations: tuple[str, ...] = (),
) -> PreregDiff:
    devs: list[str] = []
    if prereg_primary_spec != actual_primary_spec:
        devs.append(
            f"primary_spec changed: pre-registered '{prereg_primary_spec}', "
            f"actual '{actual_primary_spec}'"
        )
    if prereg_flag_tau != actual_flag_tau:
        devs.append(
            f"flag_threshold_tau changed: pre-registered {prereg_flag_tau}, "
            f"actual {actual_flag_tau}"
        )
    if prereg_measurability_min != actual_measurability_min:
        devs.append(
            f"measurability_minimum changed: pre-registered {prereg_measurability_min}, "
            f"actual {actual_measurability_min}"
        )
    devs.extend(additional_deviations)
    return PreregDiff(deviations=tuple(devs))
