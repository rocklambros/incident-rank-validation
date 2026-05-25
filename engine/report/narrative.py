"""Standalone narrative report generator.

Reads the same data files as the notebook, generates charts as static PNGs,
and writes a self-contained markdown report for redistribution.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from engine.threats.register import get_threats_register


def _render_markdown(data: dict[str, Any], figures_dir: Path) -> str:
    """Generate the full narrative markdown report."""
    lines: list[str] = []

    if data.get("non_publishable"):
        lines.append(
            "**STATUS: NON-PUBLISHABLE** (single-author rubric, uncontrolled)\n\n"
        )

    entry_names = data["entry_names"]
    entry_ids = data["entry_ids"]
    conc = data["concordance"]
    sel_bias = data["selection_bias"]

    # Act 1: The Question
    lines.append("# What the Data Says About the 2026 Top 10\n\n")
    lines.append("## Act 1: The Question\n\n")
    lines.append("| Entry | Name | Incident Rank |\n")
    lines.append("|-------|------|---------------|\n")
    for eid in entry_ids:
        name = entry_names.get(eid, eid)
        lines.append(f"| {eid} | {name} | — |\n")
    lines.append("\n")

    # Act 2: The Corpus
    lines.append("## Act 2: The Corpus\n\n")
    total_incidents = len(data["incidents"])
    from collections import Counter
    stratum_counts = Counter(inc.get("stratum", "unknown") for inc in data["incidents"])
    for stratum, count in stratum_counts.most_common():
        lines.append(f"- **{stratum}**: {count} incidents\n")
    lines.append(f"\nTotal: {total_incidents} incidents.\n\n")
    lines.append(f"![Stratum breakdown](figures/stratum_bar.png)\n\n")

    # Act 3: Classification
    lines.append("## Act 3: Classification\n\n")
    tier_counts = Counter(
        p.get("triage_tier", "unknown")
        for p in data["prelabels"]
        if p.get("consensus") != "out-of-scope"
    )
    lines.append(
        f"Consensus tiers: {tier_counts.get('agree', 0)} agree, "
        f"{tier_counts.get('split', 0)} split, "
        f"{tier_counts.get('disagree', 0)} disagree.\n\n"
    )
    lines.append(f"![Tier distribution](figures/tier_donut.png)\n\n")
    lines.append(f"![Confusion heatmap](figures/confusion_heatmap.png)\n\n")

    # Act 4: How Good Is the Classifier?
    lines.append("## Act 4: How Good Is the Classifier?\n\n")
    prec_keys = list(data["posteriors"].get("precision", {}).keys())
    security_prec = [k for k in prec_keys if "::security" in k]
    aiharm_prec = [k for k in prec_keys if "::ai-harm" in k or "::ai_harm" in k]
    verified_entries = {r.get("claimed_entry_id") for r in data["precision_verification"]}
    lines.append(
        f"Precision verifications: {len(data['precision_verification'])} records "
        f"across {len(verified_entries)} verified entries. "
        f"Posterior keys: {len(security_prec)} security-stratum, "
        f"{len(aiharm_prec)} ai-harm.\n\n"
    )
    if not aiharm_prec:
        lines.append(
            "**Note:** The ai-harm stratum has zero direct precision measurements. "
            "The model falls back to a flat Beta(1,1) = Uniform(0,1) prior for "
            "ai-harm precision.\n\n"
        )
    lines.append(f"![Precision bars](figures/precision_bars.png)\n\n")
    lines.append(f"![Precision posteriors](figures/precision_posteriors.png)\n\n")

    # Act 5: From Counts to Rankings
    lines.append("## Act 5: From Counts to Rankings\n\n")
    lambda_samples = data["lambda_samples"]
    lines.append(
        f"MCMC: {lambda_samples.shape[0]} posterior draws, "
        f"{len(entry_ids)} entries.\n\n"
    )
    inf_summary = data["inference_summary"]
    r_hat = inf_summary.get("r_hat", {})
    ess = inf_summary.get("ess", {})
    if r_hat:
        max_r_hat = max(r_hat.values()) if isinstance(r_hat, dict) else r_hat
        lines.append(f"Max R̂: {max_r_hat:.4f}. ")
    if ess:
        min_ess = min(ess.values()) if isinstance(ess, dict) else ess
        lines.append(f"Min ESS: {min_ess:.0f}.\n\n")
    lines.append(f"![Ridge plot](figures/ridge_plot.png)\n\n")

    # Act 6: The Incident-Derived Rankings
    lines.append("## Act 6: The Incident-Derived Rankings\n\n")
    lines.append(f"![Dumbbell chart](figures/dumbbell_chart.png)\n\n")
    lines.append(f"![Rankings](figures/plotly_rankings.png)\n\n")

    # Corpus B corroboration
    if "corpus_b" in data:
        cb = data["corpus_b"]
        lines.append("### Corpus B Corroboration\n\n")
        overlap = cb.get("overlap_count", 0)
        agree = cb.get("agreement_count", 0)
        rate = cb.get("agreement_rate", 0.0)
        lines.append(
            f"Corpus B (GenAI agentic): {cb.get('corpus_b_incident_count', 0)} incidents. "
            f"Shared with corpus A: {overlap}. "
            f"Label agreement: {agree}/{overlap} ({rate:.0%}).\n\n"
        )

    # Act 7: The Confrontation
    lines.append("## Act 7: The Confrontation\n\n")
    if conc.get("weighted_kappa_median") is not None:
        ci = conc.get("weighted_kappa_ci", [])
        if "ci_method" not in conc:
            raise ValueError(
                "concordance.json missing ci_method field; "
                "run Phase 1 (Tasks 1-3) before generating the narrative report"
            )
        ci_method = conc["ci_method"]
        lines.append(
            f"Weighted Cohen's kappa: {conc['weighted_kappa_median']:.4f} "
            f"[{ci[0]:.4f}, {ci[1]:.4f}] (95% interval, method: {ci_method}).\n\n"
        )
    lines.append(
        f"Selection bias: H = {sel_bias.get('statistic_value', 0):.4f}, "
        f"p = {sel_bias.get('p_value', 0):.4f}.\n\n"
    )
    lines.append(f"![Bump chart](figures/bump_chart.png)\n\n")
    lines.append(f"![CI overlap](figures/ci_overlap.png)\n\n")

    # Act 8: Where Experts and Incidents Disagree
    lines.append("## Act 8: Where Experts and Incidents Disagree\n\n")
    flags = conc.get("flags", [])
    if flags:
        lines.append(f"{len(flags)} entries flagged (P(tier mismatch) > τ):\n\n")
        for f in flags:
            lines.append(
                f"- **{f['entry_id']}**: P = {f['probability']:.2f}, "
                f"direction = {f['direction']}\n"
            )
        lines.append("\n")
    lines.append(f"![Paired dots](figures/paired_dots.png)\n\n")
    lines.append(f"![Theme bars LLM09](figures/theme_bars_llm09.png)\n\n")
    lines.append(f"![Theme bars NEW-WLA](figures/theme_bars_new_wla.png)\n\n")

    # Act 9: What the Data Cannot See
    lines.append("## Act 9: What the Data Cannot See\n\n")
    lines.append(f"![OOS treemap](figures/oos_treemap.png)\n\n")
    lines.append(f"![Sankey confusion](figures/sankey_confusion.png)\n\n")
    lines.append(f"![Confusion matrix](figures/confusion_matrix_3x3.png)\n\n")

    # Act 10: What This Means
    lines.append("## Act 10: What This Means\n\n")
    lines.append(
        "The current kappa of "
        f"{conc.get('weighted_kappa_median', 0):.2f} "
    )
    if conc.get("weighted_kappa_ci"):
        ci = conc["weighted_kappa_ci"]
        lines.append(f"[{ci[0]:.2f}, {ci[1]:.2f}] ")
    lines.append(
        "is consistent with weak-to-moderate agreement, but the confidence interval "
        "is too wide to draw firm conclusions.\n\n"
    )

    # Accepted limitation: ai-harm precision
    lines.append(
        "**Accepted limitation: ai-harm precision.** The 323 precision verifications "
        "were drawn entirely from the security stratum. The ai-harm stratum (92 "
        "in-scope incidents across 8 entry assignments, of which only 3 received "
        "recall posteriors with material evidence — LLM09, LLM04, NEW-MA; NEW-WLA "
        "has only 1 observation above the pure prior) has no direct precision "
        "measurements — "
        "ai-harm precision keys are absent from the calibration data entirely. "
        "The model falls back to a flat Beta(1,1) = Uniform(0,1) prior for ai-harm "
        "precision, meaning it assumes no prior knowledge about how precise the "
        "classifier is on ai-harm incidents (prior mean 0.5).\n\n"
    )

    # Threats to Validity — programmatic from register
    lines.append("## Threats to Validity\n\n")
    for t in get_threats_register():
        lines.append(f"- **{t.threat_id}**: {t.description}\n")
    lines.append("\n")

    lines.append("---\n")
    lines.append(
        "Before publishing externally, verify against `docs/REVIEWERS.md` "
        "PRE-PUBLISH CHECKLIST. This report is internal-only unless the "
        "checklist passes.\n"
    )

    return "".join(lines)


def generate_narrative_report(cycle_dir: Path, output_dir: Path) -> Path:
    """Generate a standalone narrative report with embedded figures.

    Returns the path to the generated report.md.
    """
    from engine.report.narrative_charts import (
        generate_all_matplotlib_charts,
        generate_all_plotly_charts,
    )
    from engine.report.narrative_data import load_narrative_data

    data = load_narrative_data(cycle_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    generate_all_matplotlib_charts(data, figures_dir)
    generate_all_plotly_charts(data, figures_dir)

    report_md = _render_markdown(data, figures_dir)
    report_path = output_dir / "report.md"
    report_path.write_text(report_md)

    return report_path
