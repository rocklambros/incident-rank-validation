from __future__ import annotations

import math
from dataclasses import dataclass

from engine.decide.concordance import STANDING_CAVEAT, ConcordanceResult
from engine.decide.measurability import MeasurabilityMap
from engine.decide.robustness_multiplicity import RobustnessSpread
from engine.decide.selection_bias import SelectionBiasDisclosure
from engine.decide.twin_agreement import TwinAgreement
from engine.threats.register import get_threats_register


@dataclass(frozen=True, slots=True)
class ReportInputs:
    cycle_id: str
    engine_version: str
    measurability_map: MeasurabilityMap
    concordance: ConcordanceResult
    selection_bias: SelectionBiasDisclosure
    robustness: RobustnessSpread | None
    twin_agreement: TwinAgreement | None
    non_publishable: bool


def render_report(inputs: ReportInputs) -> str:
    lines: list[str] = []
    lines.append(f"# Cycle Report: {inputs.cycle_id}\n")
    lines.append(f"Engine version: {inputs.engine_version}\n")
    if inputs.non_publishable:
        lines.append(
            "**STATUS: NON-PUBLISHABLE** (single-author rubric, uncontrolled)\n"
        )
    lines.append("\n## Measurability Map\n")
    lines.append(f"Coverage ratio: {inputs.measurability_map.coverage_ratio:.2%}\n")
    lines.append(
        f"Measurable: {', '.join(inputs.measurability_map.measurable) or 'none'}\n"
    )
    lines.append(
        f"Classifier-blind: {', '.join(inputs.measurability_map.classifier_blind) or 'none'}\n"
    )
    lines.append(
        f"Frame-blind: {', '.join(inputs.measurability_map.frame_blind) or 'none'}\n"
    )
    if inputs.measurability_map.below_prereg_minimum:
        lines.append("**Below pre-registered measurability minimum.**\n")

    lines.append("\n## Concordance\n")
    if inputs.concordance.weighted_kappa_median is not None:
        ci = inputs.concordance.weighted_kappa_ci
        ci_str = f"[{ci[0]:.2f}, {ci[1]:.2f}]" if ci else "N/A"
        lines.append(
            f"Weighted kappa: {inputs.concordance.weighted_kappa_median:.2f} {ci_str}\n"
        )
        lines.append(
            f"Computed over {inputs.concordance.measurable_count} of "
            f"{inputs.concordance.total_count} entries "
            f"({inputs.concordance.coverage_ratio:.0%} coverage)\n"
        )
    else:
        lines.append(
            f"N/A: insufficient measurable subset "
            f"(n={inputs.concordance.measurable_count}, "
            f"minimum={inputs.concordance.meaningful_kappa_n})\n"
        )
    lines.append(f"\n> {STANDING_CAVEAT}\n")

    lines.append("\n## Selection Bias\n")
    lines.append(f"Statistic: {inputs.selection_bias.statistic_name}\n")
    if not math.isnan(inputs.selection_bias.statistic_value):
        lines.append(
            f"H = {inputs.selection_bias.statistic_value:.4f}, "
            f"p = {inputs.selection_bias.p_value:.4f}\n"
        )
    lines.append(f"Severity: {inputs.selection_bias.severity}\n")

    if inputs.concordance.flags:
        lines.append("\n## Flags\n")
        for f in inputs.concordance.flags:
            lines.append(
                f"- {f.entry_id}: P(tier mismatch) = {f.probability:.2f}, "
                f"direction = {f.direction.value}\n"
            )

    lines.append("\n## Threats to Validity\n")
    for t in get_threats_register():
        lines.append(f"- **{t.threat_id}**: {t.description}\n")

    # M6: PRE-PUBLISH CHECKLIST footer
    lines.append("\n---\n")
    lines.append(
        "Before publishing externally, verify against `docs/REVIEWERS.md` "
        "PRE-PUBLISH CHECKLIST. This report is internal-only unless the checklist passes.\n"
    )
    return "".join(lines)
