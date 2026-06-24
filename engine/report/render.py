from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, cast

from engine.decide.concordance import STANDING_CAVEAT, ConcordanceResult
from engine.decide.measurability import MeasurabilityMap
from engine.decide.robustness_multiplicity import RobustnessSpread
from engine.decide.rollup import RollupResult
from engine.decide.selection_bias import SelectionBiasDisclosure
from engine.decide.twin_agreement import TwinAgreement
from engine.report.diff import PreregDiff
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
    rollup_results: tuple[RollupResult, ...] = ()
    prereg_diff: PreregDiff | None = None
    runpod_cost_usd: float | None = None
    cost_ceiling_usd: float | None = None
    corpus_b_corroboration: dict[str, object] | None = None
    vote_plackett_luce: dict[str, object] | None = None


def _render_robustness_lines(spread: RobustnessSpread) -> list[str]:
    lines: list[str] = ["\n## Robustness\n"]
    all_specs = [spread.primary, *spread.robustness]
    for sr in all_specs:
        if sr.weighted_kappa_median is not None and sr.weighted_kappa_ci is not None:
            line = (
                f"- {sr.spec_name}: kappa = {sr.weighted_kappa_median:.2f} "
                f"[{sr.weighted_kappa_ci[0]:.2f}, {sr.weighted_kappa_ci[1]:.2f}]"
            )
        else:
            line = f"- {sr.spec_name}: kappa = N/A"
        if sr.sigma_u is not None:
            line += f"  (σ_u = {sr.sigma_u:.2f})"
        lines.append(line + "\n")
        if sr.extra_rankings:
            for name, ranking in sorted(sr.extra_rankings.items()):
                top = " > ".join(ranking[:5])
                suffix = " > ..." if len(ranking) > 5 else ""
                lines.append(f"  - {name} ranking: {top}{suffix}\n")
    spread_val = spread.spread
    if spread_val is not None:
        lines.append(f"Spread: {spread_val:.3f}\n")
    if not spread.is_consistent_in_direction():
        lines.append("**WARNING: Specs disagree on flag direction.**\n")
    return lines


def _render_vote_pl_lines(vote_pl: dict[str, object] | None) -> list[str]:
    """Render the Davidson tie-aware vote-aggregation robustness section."""
    if vote_pl is None:
        return []
    lines: list[str] = ["\n## Vote Aggregation (Plackett-Luce / Davidson)\n"]
    ranking = vote_pl.get("ranking", [])
    if isinstance(ranking, list) and ranking:
        top = " > ".join(str(e) for e in ranking[:5])
        suffix = " > ..." if len(ranking) > 5 else ""
        lines.append(f"- Worth ranking: {top}{suffix}\n")
    tie_param = vote_pl.get("tie_param")
    if isinstance(tie_param, int | float):
        lines.append(f"- Tie parameter (nu): {tie_param:.2f}\n")
    tau_mean = vote_pl.get("kendall_tau_vs_meanrank")
    if isinstance(tau_mean, int | float):
        lines.append(f"- Kendall tau vs mean-rank vote ranking: {tau_mean:.2f}\n")
    tau_drop = vote_pl.get("kendall_tau_withties_vs_dropties")
    if isinstance(tau_drop, int | float):
        lines.append(f"- Kendall tau with-ties vs drop-ties: {tau_drop:.2f}\n")
    boot = vote_pl.get("mean_kendall_tau_vs_point")
    if isinstance(boot, int | float):
        lines.append(f"- Bootstrap stability (mean Kendall tau vs point): {boot:.2f}\n")
    return lines


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

    # Robustness spread (R2: HANDOFF §6 control 11(g))
    if inputs.robustness is not None:
        lines.extend(_render_robustness_lines(inputs.robustness))
    lines.extend(_render_vote_pl_lines(inputs.vote_plackett_luce))

    # Rollup sub-test
    if inputs.rollup_results:
        lines.append("\n## Rollup Sub-Test\n")
        for r in inputs.rollup_results:
            lines.append(
                f"- {r.child_entry_id} (parent: {r.parent_entry_id}): "
                f"{r.verdict.value}, P(distinct cluster) = {r.p_distinct_cluster:.2f}, "
                f"ratio = {r.ratio_median:.2f}\n"
            )

    # Pre-registration diff
    if inputs.prereg_diff is not None:
        lines.append("\n" + inputs.prereg_diff.to_markdown())

    # RunPod cost
    if inputs.runpod_cost_usd is not None:
        lines.append("\n## Stage-2 Cost\n")
        lines.append(
            f"RunPod actual: ${inputs.runpod_cost_usd:.2f} / "
            f"${inputs.cost_ceiling_usd:.2f} ceiling\n"
        )

    # Corpus B corroboration
    if inputs.corpus_b_corroboration is not None:
        cb = inputs.corpus_b_corroboration
        lines.append("\n## Corpus B Corroboration\n")
        lines.append(
            "Declared qualitative artifact — NOT a posterior input "
            "(HANDOFF §4, §5.4).\n\n"
        )
        overlap = cast(int, cb.get("overlap_count", 0))
        agree = cb.get("agreement_count", 0)
        disagree = cb.get("disagreement_count", 0)
        rate = cb.get("agreement_rate", 0.0)
        b_count = cb.get("corpus_b_incident_count", 0)
        lines.append(
            f"Corpus B incidents: {b_count}. "
            f"Shared with corpus A: {overlap}.\n"
        )
        if overlap > 0:
            lines.append(
                f"Label agreement on shared incidents: "
                f"{agree} agree, {disagree} disagree "
                f"(rate = {rate:.0%}).\n"
            )
        else:
            lines.append("No shared incidents detected.\n")

        baseline = cb.get("baseline_kappa", 0.0)
        lines.append(
            f"\nContext: cycle headline kappa = {baseline:.3f}. "
            f"Agreement reporting at N = {overlap} is qualitative, "
            f"not statistical.\n"
        )

        lines.append(
            "\nNote: 3 entries are frame-blind (LLM04, LLM08, LLM10). "
            "Agreement on incidents classified to these entries is reported "
            "but has no bearing on posterior estimates.\n"
        )

        divergences = cast("list[Any]", cb.get("systematic_divergences", []))
        if divergences:
            lines.append("\nSystematic divergences (published finding):\n")
            for d in divergences:
                if isinstance(d, dict):
                    lines.append(
                        f"- {d.get('pattern', 'unknown')} "
                        f"({d.get('count', 0)} incidents)\n"
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
