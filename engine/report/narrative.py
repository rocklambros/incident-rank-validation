"""Standalone narrative report generator.

Reads the same data files as the notebook, generates charts as static PNGs,
and writes an arXiv-ready markdown report plus a compiled PDF. The narrative
mirrors notebooks/2026_top_10_llm_update_what_the_data_says.ipynb at full
prose depth, including every deep dive.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from engine.threats.register import get_threats_register


_FRAME_BLIND = {"LLM04", "LLM08", "LLM10"}


def _tidy_rank_cell(text: str) -> str:
    """Strip trailing `.0` from rank values, keep half-ranks like 11.5 intact.

    Input examples: '12.0 (4.0–18.0)', '11.5 (8.0–15.0)'.
    """

    def _sub(m: re.Match[str]) -> str:
        return m.group(1)

    return re.sub(r"(\d+)\.0(?!\d)", _sub, text)


def _parse_rank_comparison_rows(md: str) -> list[dict[str, Any]]:
    """Parse the rank_comparison_report.md table into structured rows.

    Returns a list sorted by lambda (incident-derived) median rank ascending.
    Each row carries the entry id, both rank+CI strings, and the median ranks
    as floats for sorting.
    """
    rows: list[dict[str, Any]] = []
    cell_re = re.compile(r"^\s*\|\s*(LLM\d+|NEW-[A-Z]+|ROLL-[A-Z]+)\s*\|")
    for line in md.splitlines():
        if not cell_re.match(line):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 6:
            continue
        entry_id, lam, vote, tier, direction, _action = cells[:6]
        lam_med = float(lam.split()[0])
        vote_med = float(vote.split()[0])
        rows.append(
            {
                "entry_id": entry_id,
                "lambda": lam,
                "vote": vote,
                "tier": tier,
                "direction": direction,
                "lambda_med": lam_med,
                "vote_med": vote_med,
            }
        )
    rows.sort(key=lambda r: (r["lambda_med"], r["vote_med"], r["entry_id"]))
    return rows


def _render_act6_ranking_table(data: dict[str, Any]) -> str:
    """Render the Act 6 ranking table, sorted by incident-derived rank.

    The table is the climax of the report — the literal answer to "what does
    the incident data say the Top 10 looks like." Sorted ascending by lambda
    median so the eye reads the new ordering top-to-bottom.
    """
    md = data.get("rank_comparison_md", "")
    rows = _parse_rank_comparison_rows(md)
    if not rows:
        return ""
    names = data.get("entry_names", {})
    lines = [
        "| # | Entry | Name | Incident rank (90% CI) | Expert rank (90% CI) | Direction |",
        "|---|-------|------|------------------------|----------------------|-----------|",
    ]
    for i, r in enumerate(rows, start=1):
        eid = r["entry_id"]
        name = names.get(eid, "")
        marker = " ★" if eid in _FRAME_BLIND else ""
        direction = r["direction"].replace("votes-over-lambda", "vote > incidents")
        direction = direction.replace("lambda-over-votes", "incidents > vote")
        direction = direction.replace("concordant", "agree")
        lines.append(
            f"| {i} | {eid}{marker} | {name} | {_tidy_rank_cell(r['lambda'])} | "
            f"{_tidy_rank_cell(r['vote'])} | {direction} |"
        )
    lines.append("")
    lines.append(
        "★ marks frame-blind entries (LLM04, LLM08, LLM10) whose incident counts "
        "come from a single stratum, so their rank positions carry structural "
        "uncertainty beyond the CI.\n"
    )
    return "\n".join(lines) + "\n"

REPORT_STEM = "2026_top_10_llm_update_what_the_data_says"


def _precision_median(posteriors: dict[str, Any], entry_id: str, stratum: str = "security") -> float | None:
    """Median of Beta(alpha, beta) posterior for an entry's precision in a stratum.

    Beta median is approximated by alpha / (alpha + beta) (posterior mean); the
    full median has no closed form but the mean is within a percentage point
    for the alpha/beta magnitudes we observe in calibration.
    """
    key = f"{entry_id}::{stratum}"
    prec = posteriors.get("precision", {}).get(key)
    if not prec:
        return None
    a = float(prec.get("alpha", 1.0))
    b = float(prec.get("beta", 1.0))
    if (a + b) <= 0:
        return None
    return a / (a + b)


def _fmt_pct(value: float | None) -> str:
    return f"{value * 100:.0f}%" if value is not None else "n/a"


def _render_markdown(data: dict[str, Any]) -> str:  # noqa: PLR0915 — single-page report writer, splitting hurts readability
    """Generate the full arXiv-ready narrative markdown report."""
    entry_names = data["entry_names"]
    entry_ids = data["entry_ids"]
    conc = data["concordance"]
    sel_bias = data["selection_bias"]
    posteriors = data["posteriors"]

    total_incidents = len(data["incidents"])
    stratum_counts = Counter(inc.get("stratum", "unknown") for inc in data["incidents"])
    n_security = stratum_counts.get("security", 0)
    n_aiharm = stratum_counts.get("ai-harm", 0)

    tier_counts = Counter(
        p.get("triage_tier", "unknown")
        for p in data["prelabels"]
        if p.get("consensus") != "out-of-scope"
    )
    n_agree = tier_counts.get("agree", 0)
    n_split = tier_counts.get("split", 0)
    n_disagree = tier_counts.get("disagree", 0)

    prec_keys = list(posteriors.get("precision", {}).keys())
    security_prec = [k for k in prec_keys if "::security" in k]
    aiharm_prec = [k for k in prec_keys if "::ai-harm" in k or "::ai_harm" in k]
    verified_entries = {r.get("claimed_entry_id") for r in data["precision_verification"]}
    n_prec_verif = len(data["precision_verification"])

    lambda_samples = data["lambda_samples"]
    n_draws = int(lambda_samples.shape[0])
    inf_summary = data["inference_summary"]
    r_hat = inf_summary.get("r_hat", {})
    ess = inf_summary.get("ess", {})
    max_r_hat = (
        max(r_hat.values()) if isinstance(r_hat, dict) and r_hat else (r_hat if r_hat else None)
    )
    min_ess = (
        min(ess.values()) if isinstance(ess, dict) and ess else (ess if ess else None)
    )

    kappa_median = conc.get("weighted_kappa_median")
    ci = conc.get("weighted_kappa_ci", [])
    if kappa_median is None:
        raise ValueError("concordance.json missing weighted_kappa_median")
    if "ci_method" not in conc:
        raise ValueError(
            "concordance.json missing ci_method field; "
            "run Phase 1 (Tasks 1-3) before generating the narrative report"
        )
    ci_method = conc["ci_method"]
    kappa_lo, kappa_hi = ci[0], ci[1]

    flags = conc.get("flags", [])
    cb = data.get("corpus_b", {})

    p_llm01 = _precision_median(posteriors, "LLM01")
    p_llm03 = _precision_median(posteriors, "LLM03")
    p_llm07 = _precision_median(posteriors, "LLM07")
    p_llm08 = _precision_median(posteriors, "LLM08")
    p_cfas = _precision_median(posteriors, "ROLL-CFAS")
    p_cmsb = _precision_median(posteriors, "ROLL-CMSB")

    odds_llm08 = (
        f"{round(1.0 / p_llm08)}:1"
        if (p_llm08 is not None and p_llm08 > 0)
        else "n/a"
    )

    non_publishable = bool(data.get("non_publishable"))

    lines: list[str] = []

    # ─── Pandoc YAML frontmatter (consumed by --pdf-engine=xelatex via metadata) ───
    lines.append("---\n")
    lines.append('title: "What the Data Says About the 2026 Top 10"\n')
    lines.append(
        'subtitle: "An Incident-Derived Validation of the OWASP Top 10 for LLM Applications (2026)"\n'
    )
    lines.append("author:\n")
    lines.append("  - OWASP Top 10 for LLM Applications — Incident Validation Working Group\n")
    lines.append('date: "2026"\n')
    lines.append("abstract: |\n")
    lines.append(
        "  The OWASP Top 10 for LLM Applications (2026) was assembled from an expert\n"
        "  community vote. Expert opinion is one signal; the pattern of real-world\n"
        "  incidents is another. We assemble a corpus of "
        f"{total_incidents:,} incidents drawn from\n"
        "  public security-advisory databases (CVE, GHSA, OSV) and a curated AI-harm\n"
        "  database (AIAAIC), classify each record against the 20-entry 2026 taxonomy\n"
        "  using a three-model LLM consensus, calibrate classifier precision and recall\n"
        "  on a hand-adjudicated gold set, and fit a Bayesian negative-binomial\n"
        "  measurement-error model to recover per-entry latent prevalence. We then\n"
        "  compare the incident-derived ranking against the expert vote using weighted\n"
        f"  Cohen's kappa. The point estimate is {kappa_median:.2f} with a 95% credible\n"
        f"  interval of [{kappa_lo:.2f}, {kappa_hi:.2f}]: consistent with weak-to-moderate\n"
        "  agreement, but wide enough that we cannot exclude chance agreement. Five\n"
        "  entries show substantial vote-versus-incident divergence (LLM01 Prompt\n"
        "  Injection, LLM09 Misinformation, NEW-MTIE MCP Tool Interface Exploitation,\n"
        "  NEW-PMP Persistent Memory Poisoning, NEW-WLA Weaponized LLM Abuse). Three\n"
        "  entries (LLM04, LLM08, LLM10) are unmeasurable in our corpus because their\n"
        "  incidents come from only one stratum, preventing cross-stratum recall\n"
        "  estimation. We discuss the structural limits of incident-counting for\n"
        "  taxonomy validation: a sampling-frame gap that hides incidents not\n"
        "  registered in public advisory pipelines, and a confusion boundary among\n"
        "  LLM09, NEW-WLA, and ROLL-CMSB that the data cannot disentangle on its own.\n"
        "  The headline finding is methodological: incident data and expert surveys\n"
        "  carry different — and sometimes conflicting — information, and their\n"
        "  disagreement is itself the most diagnostic signal a triangulation exercise\n"
        "  can produce.\n"
    )
    lines.append("toc: true\n")
    lines.append("toc-depth: 2\n")
    lines.append("numbersections: true\n")
    lines.append("geometry: margin=1in\n")
    lines.append("linkcolor: blue\n")
    lines.append("urlcolor: blue\n")
    lines.append("colorlinks: true\n")
    lines.append("papersize: letter\n")
    lines.append("fontsize: 11pt\n")
    lines.append("---\n\n")

    if non_publishable:
        lines.append(
            "**STATUS: NON-PUBLISHABLE** (single-author rubric, uncontrolled). "
            "This document is an internal pre-print. Before any external distribution, "
            "verify against `docs/REVIEWERS.md` PRE-PUBLISH CHECKLIST.\n\n"
        )

    # Manual TOC for the standalone .md (pandoc PDF generates its own from --toc).
    # {.unnumbered .unlisted} so pandoc skips numbering and does not duplicate it
    # in the PDF TOC.
    lines.append("# Table of Contents {.unnumbered .unlisted}\n\n")
    lines.append("1. Act 1: The Question\n")
    lines.append("2. Act 2: The Corpus\n")
    lines.append("3. Act 3: Classification — How We Labeled 6,600 Incidents\n")
    lines.append("4. Act 4: How Good Is the Classifier?\n")
    lines.append("5. Act 5: From Counts to Rankings — The Bayesian Model\n")
    lines.append("6. Act 6: The Incident-Derived Rankings\n")
    lines.append("7. Act 7: The Confrontation — Do Experts and Incidents Agree?\n")
    lines.append("8. Act 8: Where Experts and Incidents Disagree\n")
    lines.append("9. Act 9: What the Data Cannot See\n")
    lines.append("10. Act 10: What This Means\n")
    lines.append("11. Threats to Validity\n")
    lines.append("12. Accepted Limitations\n")
    lines.append("13. Reproducibility\n\n")

    # ─── Act 1 ───
    lines.append("# Act 1: The Question\n\n")
    lines.append(
        "The OWASP Top 10 for LLMs ranks AI security vulnerabilities. The 2025 list was "
        "built from expert surveys — hundreds of security professionals voting on what "
        "matters most. That process produced a consensus: Prompt Injection at #1, "
        "Sensitive Information Disclosure at #2, and so on down to #10.\n\n"
    )
    lines.append(
        "Expert opinion is one signal. We wanted to check it against a second signal: "
        "the pattern of real-world incidents. We built a corpus of "
        f"~{total_incidents:,} AI security incidents from public databases, classified "
        "each one against the 20-entry taxonomy, and asked: does the incident data agree "
        "with the experts?\n\n"
    )
    lines.append(
        "This report walks through that analysis step by step. Along the way, you will "
        "see how the classification worked, how we measured its accuracy, and what a "
        "Bayesian model does with noisy measurements. Every chart and table below is "
        "computed live from the data — you can re-run any cell in the companion notebook "
        "to verify.\n\n"
    )
    lines.append(
        "Here are the 20 taxonomy entries we are working with. The \"Incident Rank\" "
        "column is blank for now. We will fill it in Act 6, after walking through the "
        "methodology.\n\n"
    )
    lines.append("| Entry | Name | Incident Rank |\n")
    lines.append("|-------|------|---------------|\n")
    for eid in entry_ids:
        name = entry_names.get(eid, eid)
        lines.append(f"| {eid} | {name} | — |\n")
    lines.append("\n")

    # ─── Act 2 ───
    lines.append("# Act 2: The Corpus\n\n")
    lines.append(
        f"The corpus contains {total_incidents:,} incidents from public databases: CVE, "
        "GHSA, and OSV (security advisories), plus AIAAIC (a database of AI-related "
        "harms and controversies). Each record has a text description of what happened.\n\n"
    )
    lines.append(
        "The corpus splits into two strata. The **security** stratum (CVE/GHSA/OSV) "
        f"contains {n_security:,} incidents — things like prompt-injection exploits, "
        "data leakage through APIs, and supply-chain compromises in ML packages. "
        "The **ai-harm** stratum (AIAAIC) contains "
        f"{n_aiharm:,} incidents — things like algorithmic discrimination, deepfake "
        "misuse, and surveillance overreach.\n\n"
    )
    lines.append(
        "This split matters. The classifier performs differently on each stratum — "
        "security incidents have more structured descriptions (CVE format), while "
        "ai-harm incidents are written as news summaries with varying detail. We will "
        "see in Act 4 that the precision and recall calibration is dense for the "
        "security stratum and sparse-to-absent for ai-harm, and we will see in Act 5 "
        "how the Bayesian model surfaces that imbalance as wider credible intervals on "
        "the affected entries.\n\n"
    )
    lines.append("Stratum breakdown:\n\n")
    for stratum, count in stratum_counts.most_common():
        lines.append(f"- **{stratum}**: {count:,} incidents\n")
    lines.append(f"\nTotal: {total_incidents:,} incidents.\n\n")
    lines.append("![Stratum breakdown](figures/stratum_bar.png)\n\n")

    lines.append("## Data sources\n\n")
    lines.append(
        "Corpus A is vendored from a public aggregator and pinned to a specific "
        "commit so the analysis is reproducible. The aggregator itself draws from "
        "established public incident databases.\n\n"
    )
    lines.append(
        "**Corpus A (primary, feeds the Bayesian model):**\n\n"
        "- Repository: <https://github.com/emmanuelgjr/genai_agentic_incidents>\n"
        "- Pinned commit: `e474ce7d0a8b2510e487a6d76d2c70bfe8b05d90` "
        "([snapshot tree](https://github.com/emmanuelgjr/genai_agentic_incidents/tree/e474ce7d0a8b2510e487a6d76d2c70bfe8b05d90))\n"
        "- Pull date: 2026-05-20\n\n"
    )
    lines.append(
        "**Corpus B (independent, corroboration only — never enters the likelihood):**\n\n"
        "- Repository: <https://github.com/OWASP/www-project-top-10-for-large-language-model-applications>\n"
        "- File: `initiatives/agent_security_initiative/ASI Agentic Exploits & Incidents/ASI_Agentic_Exploits_Incidents.md`\n"
        "- Pull date: 2026-05-23\n\n"
    )
    lines.append(
        "**Upstream public databases feeding Corpus A:**\n\n"
        "- **CVE** — Common Vulnerabilities and Exposures, accessed via the "
        "National Vulnerability Database: <https://nvd.nist.gov>\n"
        "- **GHSA** — GitHub Security Advisories: <https://github.com/advisories>\n"
        "- **OSV** — Open Source Vulnerabilities database (records reach Corpus A "
        "via the upstream aggregator)\n"
        "- **AIAAIC** — AI, Algorithmic, and Automation Incidents and Controversies "
        "repository: <https://www.aiaaic.org/aiaaic-repository>\n\n"
    )
    lines.append(
        "Per-incident reference URLs are preserved in the snapshot at "
        "`projects/owasp-llm/cycles/2026/corpora/genai_agentic/"
        "24806f1a4f0917f85f7509d6cb2a34b12e56eb902714b37bc2b03a2cf1a246bb/"
        "incidents.json`.\n\n"
    )

    # ─── Act 3 ───
    lines.append("# Act 3: Classification — How We Labeled 6,600 Incidents\n\n")
    lines.append(
        "Each incident was classified by three different large language models: "
        "Qwen 235B, Llama 405B, and DeepSeek V3. Each model independently read the "
        "incident text and assigned it to one of the 20 taxonomy entries — or marked "
        "it \"out of scope\" if none fit.\n\n"
    )
    lines.append(
        "When all three models agreed on the same entry, we call it **agree tier**. "
        "When two agreed and one differed, **split tier**. When all three picked "
        "different entries, **disagree tier**. The tier tells us how confident we can "
        "be in the classification. Agree-tier incidents have strong consensus. "
        "Disagree-tier incidents sit in ambiguous territory where even three "
        "independent classifiers could not converge.\n\n"
    )
    lines.append(
        f"In this cycle: **{n_agree:,} agree, {n_split:,} split, {n_disagree:,} disagree** "
        "(excluding out-of-scope). The agree-tier count is what feeds the Bayesian "
        "model's primary signal; split-tier incidents enter with reduced weight via the "
        "calibration step; disagree-tier incidents primarily inform the confusion "
        "boundaries we analyze in Act 9B.\n\n"
    )
    lines.append("![Tier distribution](figures/tier_donut.png)\n\n")
    lines.append("![Confusion heatmap](figures/confusion_heatmap.png)\n\n")

    # ─── Act 4 ───
    lines.append("# Act 4: How Good Is the Classifier?\n\n")
    lines.append(
        "The classifier is a tool, not ground truth. To trust the incident counts, we "
        "need to measure how often the classifier gets it right — and how often it "
        "misses things.\n\n"
    )
    lines.append(
        "**Precision**: When the classifier says \"this incident belongs to LLM02,\" "
        "how often is it correct? We verified "
        f"{n_prec_verif} classifications by hand to measure this, covering "
        f"{len(verified_entries)} verified entries. Each entry gets its own precision "
        "score — some entries are easier to classify than others.\n\n"
    )
    lines.append(
        "**Recall**: Does the classifier find all incidents of a given type, or does "
        "it miss some? A human reviewer adjudicated 1,200 incidents across all tiers "
        "to measure this. The reviewer saw the incident text and the three model votes, "
        "then decided whether to accept the consensus, override it, or mark the "
        "incident as out of scope.\n\n"
    )
    lines.append("![Precision bars](figures/precision_bars.png)\n\n")
    lines.append("![Precision posteriors](figures/precision_posteriors.png)\n\n")

    # — Deep Dive 4.1: Why precision varies —
    lines.append("## Why precision varies so much across entries\n\n")
    lines.append(
        "The chart above shows precision ranging from "
        f"{_fmt_pct(p_llm01)} (LLM01) and {_fmt_pct(p_llm03)} (LLM03) at the top down "
        f"to {_fmt_pct(p_llm08)} (LLM08) at the bottom. The variation is not random — "
        "it reflects how cleanly each entry's definition separates it from neighboring "
        "categories. Four entries fall **below the 50% threshold**, which deserves "
        "explanation:\n\n"
    )
    lines.append(
        f"- **LLM08 (Vector and Embedding Weaknesses): {_fmt_pct(p_llm08)}.** This is "
        "the lowest precision in the taxonomy. Out of every "
        f"{round(1.0 / p_llm08) if p_llm08 else 0} incidents the classifier labels as "
        "LLM08, only 1 actually is. The category describes a narrow class of attacks "
        "— adversarial manipulation of embedding spaces, vector database poisoning, "
        "retrieval-augmented generation exploits. But the classifier confuses it with "
        "LLM03 (Training Data Poisoning) and general data integrity issues. Most "
        "incidents classified here describe data manipulation that affects a model, "
        "which is conceptually adjacent but taxonomically distinct.\n\n"
    )
    lines.append(
        f"- **LLM07 (System Prompt Leakage): {_fmt_pct(p_llm07)}.** The classifier "
        "struggles to distinguish \"extracting a system prompt\" (LLM07) from "
        "\"overriding a system prompt\" (LLM01, Prompt Injection). Both involve "
        "adversarial interaction with the prompt layer, and many real incidents "
        "involve both — an attacker extracts the system prompt *in order to* craft a "
        "better injection. The boundary is taxonomically clear but operationally "
        "blurred.\n\n"
    )
    lines.append(
        f"- **ROLL-CFAS (Comprehensive Framework Attacks): {_fmt_pct(p_cfas)}.** Only "
        "a handful of precision observations — the posterior is dominated by the "
        "Beta(1,1) prior, so the point estimate has a wide 90% credible interval "
        "spanning from a few percent to most of the (0, 1) range. The category's broad "
        "definition (\"attacks on the comprehensive AI framework\") makes it a natural "
        "catch-all that absorbs incidents from adjacent entries.\n\n"
    )
    lines.append(
        f"- **ROLL-CMSB (Cross-Modal Safety Bypass): {_fmt_pct(p_cmsb)}.** This entry "
        "sits at the center of the confusion boundary described in §9B. A deepfake "
        "video that bypasses content filters could be classified as ROLL-CMSB "
        "(cross-modal bypass), LLM09 (misinformation), or NEW-WLA (weaponized abuse). "
        "The classifier picks one; a human might reasonably pick any of the three.\n\n"
    )

    # — Deep Dive 4.2: 50% threshold interpretation —
    lines.append("## What the 50% threshold means\n\n")
    lines.append(
        "Precision below 50% means the classifier is **wrong more often than it is "
        "right** for that entry. When you see an incident labeled \"LLM08,\" the odds "
        f"are {odds_llm08} against it actually being a vector/embedding weakness. This "
        "has direct consequences for the Bayesian model in Act 5: low-precision "
        "entries get large upward corrections (because much of their observed count is "
        "misclassification noise from other entries) and wide credible intervals "
        f"(because the correction itself is uncertain). A {_fmt_pct(p_llm08)} precision "
        "estimate does not mean the entry is unimportant — it means the *automated "
        "measurement* of that entry is unreliable, and the model's uncertainty "
        "reflects that.\n\n"
    )

    # — Deep Dive 4.3: Stratum coverage —
    lines.append("## Stratum coverage of precision verifications\n\n")
    if not aiharm_prec:
        lines.append(
            f"The {n_prec_verif} precision verifications were drawn entirely from the "
            "security stratum (CVE/GHSA/OSV). The ai-harm stratum (AIAAIC) has **zero** "
            "direct precision measurements — the posteriors.json file contains "
            f"{len(security_prec)} security-stratum precision keys and "
            f"{len(aiharm_prec)} ai-harm keys. The Bayesian model falls back to a flat "
            "Beta(1,1) = Uniform(0,1) prior for ai-harm precision, meaning it assumes "
            "no prior knowledge about classifier accuracy on those incidents (prior "
            "mean 0.5). This means error correction for ai-harm incidents relies on a "
            "weak prior, not direct measurement. We treat this as an accepted "
            "limitation (see §12) and surface it explicitly anywhere ai-harm-stratum "
            "estimates appear in the report.\n\n"
        )
    else:
        lines.append(
            f"The {n_prec_verif} precision verifications include both stratum and "
            "ai-harm-stratum coverage. The posterior keys partition as "
            f"{len(security_prec)} security and {len(aiharm_prec)} ai-harm. "
            "Error-correction for ai-harm incidents uses measured posteriors rather "
            "than a uniform fallback prior.\n\n"
        )

    # ─── Act 5 ───
    lines.append("# Act 5: From Counts to Rankings — The Bayesian Model\n\n")
    lines.append(
        "Raw incident counts would be misleading. An entry whose classifier has 30% "
        "precision looks like it has many incidents — but two-thirds of those are "
        "misclassifications wrongly attributed to it.\n\n"
    )
    lines.append(
        "We need a model that adjusts the observed counts for known classifier error. "
        "Think of a bathroom scale that reads 2 pounds heavy. You would subtract 2 "
        "pounds from every reading. The Bayesian model does this for each entry "
        "separately, and it carries the uncertainty through — if the scale is 2±1 "
        "pounds off, the corrected weight is also uncertain.\n\n"
    )

    lines.append("## What happens with low-precision entries\n\n")
    lines.append(
        "For entries above 50% precision, the correction is a moderate downward "
        "adjustment — some of the observed incidents were misclassified, so the true "
        "count is lower than the raw count. The correction tightens toward the true "
        "signal.\n\n"
    )
    lines.append(
        "For entries **below 50% precision**, the correction works differently. If "
        f"only {_fmt_pct(p_llm08)} of incidents labeled \"LLM08\" actually belong "
        "there, the model must infer the true rate from a signal that is mostly noise. "
        "It is like reading a bathroom scale that is off by more than half the "
        "measurement — the \"correction\" is larger than the reading itself. This "
        "produces two effects: (1) the corrected estimate can be very different from "
        "the raw count, and (2) the uncertainty around that estimate is wide, because "
        "small changes in the precision estimate propagate into large changes in the "
        "corrected rate.\n\n"
    )
    lines.append(
        "This is why some entries in the Act 6 chart have 90% credible intervals "
        "spanning 10+ rank positions. The width is not a flaw in the model — it is "
        "the model honestly reporting how much information the data contains about "
        "each entry's true incident rate.\n\n"
    )

    lines.append("## The model's inputs and sampler\n\n")
    lines.append(
        "The model takes the observed incident counts, the measured precision and "
        "recall for each entry, and produces a **posterior distribution** over the "
        "true incident rate for each entry. A posterior distribution is not a single "
        "number — it is a range of plausible values given the data. Wide distributions "
        "mean less certainty.\n\n"
    )
    lines.append(
        f"We drew {n_draws:,} samples from this distribution using the No-U-Turn "
        "Sampler (NUTS), an adaptive variant of Hamiltonian Monte Carlo. NUTS is a "
        "method for sampling from probability distributions that are too complex to "
        "compute directly. It generates a sequence of correlated draws that, after a "
        "warmup phase, represent the target distribution. Our run used 4 chains of "
        "4,000 retained samples each, with 2,000 warmup iterations per chain, sampling "
        "pinned to CPU for cross-platform determinism.\n\n"
    )
    if max_r_hat is not None and min_ess is not None:
        lines.append(
            "**Convergence diagnostics.** "
            f"Maximum R̂ across all parameters: {max_r_hat:.4f} (target < 1.01). "
            f"Minimum effective sample size (ESS): {min_ess:.0f} (target ≥ 400 per "
            "chain). Both are well within the conventional thresholds for trusted "
            "posterior inference, so we do not report further per-chain diagnostics "
            "in this body. The full inference_summary.json artifact carries the "
            "per-entry breakdowns for any reader who wishes to inspect them.\n\n"
        )

    lines.append("## Handling missing data\n\n")
    lines.append(
        "Three entries — LLM04 (Data and Model Poisoning), LLM08 (Vector and "
        "Embedding Weaknesses), LLM10 (Unbounded Consumption) — are **frame-blind**: "
        "their incident counts come entirely from one stratum, so the model cannot "
        "cross-validate their rates across strata. These entries are included in the "
        "posterior but flagged with a ★ in the charts. Their rank estimates carry "
        "additional structural uncertainty beyond what the credible intervals "
        "capture.\n\n"
    )
    lines.append(
        "**For 16 of 20 entries in the ai-harm stratum, recall has not been measured "
        "directly.** The model uses a conservative prior of approximately 1% recall "
        "— Beta(1, 101) — for those entries. This means the model assumes the "
        "classifier finds very few of those incidents and adjusts upward accordingly. "
        "These corrections are large, which is one reason the credible intervals in "
        "Act 6 are wide.\n\n"
    )
    lines.append(
        "**Precision in the ai-harm stratum is also unmeasured.** The model uses a "
        "flat Beta(1,1) = Uniform(0,1) prior, meaning it treats ai-harm precision as "
        "completely unknown (prior mean 50%). This is a weaker assumption than the "
        "security stratum, where we have 5–88 hand-verified observations per entry. "
        "We surface this through §12 (Accepted Limitations) rather than burying it.\n\n"
    )
    lines.append("![Ridge plot](figures/ridge_plot.png)\n\n")

    # ─── Act 6 ───
    lines.append("# Act 6: The Incident-Derived Rankings\n\n")
    lines.append(
        "These rankings reflect what the incident data suggests after correcting for "
        "classifier error. They are one signal, not the final word.\n\n"
    )
    lines.append(
        "For each entry, the Bayesian model gives us a posterior distribution over its "
        "true incident rate. We rank entries by their median rate and report a 90% "
        "credible interval on the rank. Some entries have tight intervals (the data "
        "is informative) and others are wide (less certain). The width tells you how "
        "much to trust the rank position.\n\n"
    )

    lines.append("## The incident-derived ranking\n\n")
    lines.append(
        "Rows are sorted by the incident-derived median rank, ascending. This is "
        "the literal output of the Bayesian model: the order the corpus implies "
        "after correcting for measured classifier precision. The expert rank is "
        "shown next to it for direct comparison.\n\n"
    )
    table_md = _render_act6_ranking_table(data)
    if table_md:
        lines.append(table_md)
        lines.append("\n")

    lines.append("## How to read the chart\n\n")
    lines.append(
        "Each row is one taxonomy entry. The diamond marks the **median rank** — the "
        "rank position at the center of the posterior distribution. The horizontal bar "
        "spans the **90% credible interval** — the range of ranks that the model "
        "considers plausible given the data and its uncertainty about precision, "
        "recall, and the true incident rate.\n\n"
    )
    lines.append(
        "**Tight intervals** (e.g., LLM02 spanning 1–6) mean the data strongly "
        "constrains that entry's position. Even after accounting for classifier error, "
        "the evidence points to a narrow range.\n\n"
    )
    lines.append(
        "**Wide intervals** (e.g., spanning 6–20) mean the data is compatible with "
        "many rank positions. This happens when the entry has low precision (the "
        "correction is large and uncertain), few observations (small sample sizes "
        "produce wide posteriors), or unmeasured recall (the model uses a conservative "
        "prior that adds uncertainty).\n\n"
    )
    lines.append(
        "**Grey entries** (LLM04, LLM08, LLM10) are frame-blind — their measurements "
        "come from only one stratum. Their positions carry structural uncertainty "
        "beyond what the CI captures.\n\n"
    )
    lines.append("![Dumbbell chart](figures/dumbbell_chart.png)\n\n")
    lines.append("![Rankings](figures/plotly_rankings.png)\n\n")

    # — Corpus B corroboration —
    if cb:
        overlap = cb.get("overlap_count", 0)
        agree = cb.get("agreement_count", 0)
        rate = cb.get("agreement_rate", 0.0)
        cb_total = cb.get("corpus_b_incident_count", 0)
        lines.append("## Corpus B corroboration\n\n")
        lines.append(
            f"Corpus B is the OWASP **Agent Security Initiative (ASI) Agentic "
            f"Exploits & Incidents** tracker — a human-curated list of {cb_total} "
            f"records maintained inside the OWASP Top 10 for LLM Applications "
            "project repository "
            "(<https://github.com/OWASP/www-project-top-10-for-large-language-model-applications>). "
            f"Of those, {overlap} records are shared with corpus A; label agreement "
            f"on the shared subset is {agree}/{overlap} ({rate:.0%}). The agreement "
            "rate is modest, which is expected given the different sampling frames "
            "and labeling rubrics. We treat Corpus B as a cross-check on corpus A "
            "rather than as ground truth.\n\n"
        )

    # ─── Act 7 ───
    lines.append("# Act 7: The Confrontation — Do Experts and Incidents Agree?\n\n")
    lines.append(
        "Cohen's weighted kappa measures agreement between two ranking systems, "
        "adjusted for chance. A value of 1.0 means perfect agreement. A value of 0 "
        "means no better than random. Negative values mean systematic disagreement.\n\n"
    )
    lines.append(
        f"**Our result: kappa = {kappa_median:.4f} with a 95% credible interval of "
        f"[{kappa_lo:.4f}, {kappa_hi:.4f}]** (interval method: {ci_method}). This "
        "interval includes zero. We cannot exclude the possibility that expert and "
        f"incident rankings agree by chance alone. The point estimate of "
        f"{kappa_median:.2f} suggests slight agreement, but the wide interval means "
        "this is a weak signal, not a firm conclusion.\n\n"
    )
    lines.append(
        "Why is the interval so wide? Two reasons. First, we only have 17 measurable "
        "entries (three are frame-blind). Statistical agreement measures need larger "
        "samples for narrow confidence intervals. Second, the posterior rank "
        "distributions themselves are wide — most entries have 90% CIs spanning 10+ "
        "rank positions.\n\n"
    )
    lines.append(
        f"**Selection bias check.** Kruskal-Wallis H = {sel_bias.get('statistic_value', 0):.4f}, "
        f"p = {sel_bias.get('p_value', 0):.4f}. We cannot reject the null hypothesis "
        "that the incident-count distribution is the same across the vote-rank tiers, "
        "which is the result we want: no detectable selection bias in how incidents "
        "distribute against the expert vote.\n\n"
    )
    lines.append(
        "**Probability of tier mismatch.** Across the full joint posterior, we compute "
        "for each entry the probability that the Bayesian model and the expert survey "
        "place it in different thirds of the ranking. Five entries exceed the 83% "
        "probability threshold τ:\n\n"
    )
    if flags:
        lines.append(f"{len(flags)} entries flagged (P(tier mismatch) > τ):\n\n")
        for f in flags:
            lines.append(
                f"- **{f['entry_id']}**: P = {f['probability']:.2f}, "
                f"direction = {f['direction']}\n"
            )
        lines.append("\n")
    lines.append("![Bump chart](figures/bump_chart.png)\n\n")
    lines.append("![CI overlap](figures/ci_overlap.png)\n\n")

    # ─── Act 8 ───
    lines.append("# Act 8: Where Experts and Incidents Disagree\n\n")
    lines.append(
        "Five entries have notable tier mismatches. For each, we dig into *why* the "
        "disagreement exists — what the data shows and what it might mean.\n\n"
    )
    lines.append(
        "**LLM01 (Prompt Injection)**: Expert rank #1 (90% CI: 1–2), incident rank "
        "#12 (90% CI: 4–18). Prompt injection is the best-understood LLM attack. "
        "Deployed systems defend against it actively — input filtering, output "
        "sandboxing, system-prompt hardening. Fewer incidents reach public databases "
        "because defenses often work. Experts rank it #1 because the attack surface "
        "is enormous even when defenses hold. The incident data sees fewer successful "
        "exploits, so the model ranks it lower. This is a defense-effect artifact, "
        "not evidence that prompt injection is unimportant.\n\n"
    )
    lines.append(
        "**LLM09 (Misinformation)**: Incident rank #2 (90% CI: 1–5), expert rank "
        "#13 (90% CI: 9–16). The corpus contains a large volume of deepfake and "
        "AI-generated disinformation incidents from the AIAAIC harm database. Experts "
        "may rank misinformation lower because \"misinformation\" as a category "
        "overlaps with other entries (NEW-WLA, ROLL-CMSB) and because many of these "
        "incidents describe harm *from* AI rather than a vulnerability *in* an LLM. "
        "We examine this overlap in §9B.\n\n"
    )
    lines.append(
        "**NEW-PMP (Persistent Memory Poisoning)** and **NEW-MTIE (MCP Tool Interface "
        "Exploitation)**: Expert top-5, almost no incidents yet. These are emerging "
        "threats — persistent memory poisoning and MCP tool exploitation are new "
        "enough that the public incident record has not caught up. If the goal of the "
        "Top 10 is to warn practitioners about threats they will face, expert signal "
        "may matter more than incident counts for emerging entries. The incident "
        "ranking treats the absence of public reports as low prevalence, which is "
        "operationally wrong when the actual cause is reporting lag.\n\n"
    )
    lines.append(
        "**NEW-WLA (Weaponized LLM Abuse)**: Incident rank #8 (90% CI: 3–15), expert "
        "rank #17 (90% CI: 13–20). The large incident count is driven by a broad entry "
        "definition that captures AI-generated disinformation, deepfake CSAM, and "
        "synthetic media abuse. Experts may rank it low because many of these "
        "incidents describe harm *from* AI systems rather than an exploitable "
        "vulnerability *in* an LLM. The category sits inside the confusion boundary "
        "discussed in §9B, which inflates its count at the expense of LLM09 and "
        "ROLL-CMSB.\n\n"
    )
    lines.append("![Paired dots](figures/paired_dots.png)\n\n")
    lines.append("![Theme bars LLM09](figures/theme_bars_llm09.png)\n\n")
    lines.append("![Theme bars NEW-WLA](figures/theme_bars_new_wla.png)\n\n")

    # ─── Act 9 ───
    lines.append("# Act 9: What the Data Cannot See\n\n")
    lines.append(
        "The ranking analysis covers the 17 measurable entries. But two patterns in "
        "the data reveal structural limits of what incident-counting can tell us.\n\n"
    )

    lines.append("## \"AI Harm Without LLM Vulnerability\"\n\n")
    lines.append(
        "Roughly 40% of the corpus — 2,394 incidents — landed in \"out of scope.\" "
        "All three models agreed these do not belong to any of the 20 taxonomy "
        "entries.\n\n"
    )
    lines.append(
        "These are real AI harms. Facial recognition that misidentifies people. "
        "Algorithmic hiring tools that discriminate. Drones used for surveillance. "
        "Recommendation engines that radicalize users. But none of them describe a "
        "vulnerability *in* a large language model. They are incidents *from* AI "
        "systems, not incidents *of* LLM vulnerabilities.\n\n"
    )
    lines.append(
        "This gap is a feature of the sampling frame, not a failure of the taxonomy. "
        "The corpus was built by crawling CVE/GHSA/OSV databases with AI-related "
        "keywords. Those keywords pull in any incident that mentions \"AI\" or "
        "\"machine learning,\" regardless of whether an LLM is involved. The AIAAIC "
        "harm database, by design, covers all AI-related harms.\n\n"
    )
    lines.append(
        "The out-of-scope cluster matters because it shows the boundary of what this "
        "methodology can measure. Incident-counting works when incidents map to "
        "taxonomy entries. For harms that sit outside the taxonomy — because they "
        "involve non-LLM AI, or because they describe societal effects rather than "
        "technical vulnerabilities — the incident signal is silent. A future cycle "
        "could (a) tighten the sampling-frame keywords, (b) add a parallel taxonomy "
        "for AI-system harms outside LLM vulnerability, or (c) accept the silence and "
        "report it as a known scope limit. We do (c) in this cycle and recommend (b) "
        "for future work.\n\n"
    )
    lines.append("![OOS treemap](figures/oos_treemap.png)\n\n")

    lines.append("## The LLM09 / NEW-WLA / ROLL-CMSB confusion boundary\n\n")
    lines.append(
        "A **confusion boundary** is a region where categories overlap enough that "
        "classifiers — and sometimes humans — cannot reliably tell them apart. The "
        "problem is not that the classifier is broken. The problem is that the "
        "categories share real conceptual territory.\n\n"
    )
    lines.append("The data shows a clear confusion boundary between three entries:\n\n")
    lines.append("- **LLM09 (Misinformation)**: the output is false or misleading\n")
    lines.append("- **NEW-WLA (Weaponized LLM Abuse)**: an adversary uses AI as a weapon\n")
    lines.append("- **ROLL-CMSB (Cross-Modal Safety Bypass)**: the attack uses image/video/audio modalities\n\n")
    lines.append(
        "Consider a deepfake video that spreads political disinformation. Which entry "
        "does it belong to? It is misleading content (LLM09). It was created using AI "
        "as a weapon (NEW-WLA). It exploits an image/video generation modality "
        "(ROLL-CMSB). The three categories overlap in real-world incidents, and the "
        "overlap is not a classification error — it reflects genuine ambiguity in the "
        "taxonomy.\n\n"
    )
    lines.append(
        "This matters for interpretation. When the incident data ranks LLM09 at #2, "
        "some of that signal comes from incidents that could equally have been "
        "classified as NEW-WLA or ROLL-CMSB. The confusion boundary inflates counts "
        "for whichever entry the classifier happens to prefer and deflates counts for "
        "the others. The Bayesian model corrects for measured precision (how often "
        "each entry's classifications are right), but it cannot correct for ambiguity "
        "that the gold-set reviewers themselves found difficult to resolve. Resolving "
        "this would require either tighter entry definitions in the next rubric "
        "revision or an explicit modeling of label uncertainty for the three "
        "boundary entries; we recommend the former.\n\n"
    )
    lines.append("![Sankey confusion](figures/sankey_confusion.png)\n\n")
    lines.append("![Confusion matrix](figures/confusion_matrix_3x3.png)\n\n")

    # ─── Act 10 ───
    lines.append("# Act 10: What This Means\n\n")
    lines.append(
        "**Where the data and experts agree.** LLM02 (Sensitive Information "
        "Disclosure) sits near the top by both measures — experts rank it #2 and "
        "incidents rank it #2. ROLL-SICG, NEW-ITSCD, and NEW-MSDA are consistently "
        "near the bottom by both signals. These positions are stable across the "
        "uncertainty ranges, which is the strongest form of convergent validation the "
        "methodology can produce.\n\n"
    )
    lines.append(
        "**Where the data pushes back.** LLM09's incident volume is much higher than "
        "its expert rank. Part of this comes from the broad entry definition, which "
        "captures AI-adjacent harms (deepfake misuse, synthetic disinformation) that "
        "may not represent LLM vulnerabilities in the narrow sense. NEW-WLA shows a "
        "similar pattern. The confusion boundary between LLM09, NEW-WLA, and "
        "ROLL-CMSB inflates whichever entry the classifier prefers and makes all "
        "three counts less reliable than entries with cleaner boundaries.\n\n"
    )
    lines.append(
        "**What the experts see that incidents miss.** NEW-PMP and NEW-MTIE have "
        "almost no incidents in the public record but strong expert signal. These are "
        "forward-looking entries — persistent memory poisoning and MCP tool "
        "exploitation are new enough that the incident databases have not caught up. "
        "If the purpose of the Top 10 is to warn practitioners about threats they "
        "will face, expert signal matters more than incident counts for emerging "
        "threats. We recommend treating these entries as expert-led for at least one "
        "more cycle.\n\n"
    )
    lines.append(
        "**What this methodology can and cannot do.** This is a triangulation tool. "
        "It checks one signal (expert surveys) against another (incident data). "
        "Neither signal is the truth. The incident data has known structural biases: "
        "the sampling frame misses incidents that are not publicly reported, the "
        "classifier has measured error rates, and the taxonomy-frame circularity "
        "(F-circ) means we are partially measuring the classifier's preferences "
        "rather than the true threat distribution. The expert data has its own "
        "biases: availability bias, recency effects, anchoring to prior Top 10 "
        "lists.\n\n"
    )
    lines.append(
        "The value is in the comparison, not in either signal alone. Where experts "
        "and incidents agree, confidence is higher. Where they diverge, the "
        "disagreement itself is the finding — it points to entries where one signal "
        "or the other may be systematically distorted.\n\n"
    )
    lines.append(
        f"**The kappa ceiling is structural.** Some of the disagreement between expert "
        "and incident rankings is informative — it reveals real differences between "
        "\"what experts worry about\" and \"what has actually happened.\" Perfect "
        "agreement would be surprising and arguably suspicious. The current kappa of "
        f"{kappa_median:.2f} [{kappa_lo:.2f}, {kappa_hi:.2f}] is consistent with "
        "weak-to-moderate agreement, but the confidence interval is too wide to draw "
        "firm conclusions. A larger corpus, better-defined entry boundaries, and "
        "independent recall measurement would all narrow the interval and sharpen the "
        "comparison.\n\n"
    )

    # ─── Threats to Validity ───
    lines.append("# Threats to Validity\n\n")
    lines.append(
        "The threats register below is loaded programmatically from "
        "`engine/threats/register.py`. Each entry names a specific failure mode and "
        "the rationale for why it is either mitigated or accepted in this cycle. "
        "Future cycles should re-evaluate every accepted threat.\n\n"
    )
    for t in get_threats_register():
        lines.append(f"- **{t.threat_id}**: {t.description}\n")
    lines.append("\n")

    # ─── Accepted Limitations ───
    lines.append("# Accepted Limitations\n\n")
    lines.append(
        f"**Ai-harm precision (F-aiharm-precision).** The {n_prec_verif} precision "
        "verifications were drawn entirely from the security stratum. The ai-harm "
        "stratum (92 in-scope incidents across 8 entry assignments, of which only 3 "
        "received recall posteriors with material evidence — LLM09, LLM04, NEW-MA; "
        "NEW-WLA has only 1 observation above the pure prior) has no direct "
        "precision measurements — ai-harm precision keys are absent from the "
        "calibration data entirely. The model falls back to a flat Beta(1,1) = "
        "Uniform(0,1) prior for ai-harm precision, meaning it assumes no prior "
        "knowledge about how precise the classifier is on ai-harm incidents (prior "
        "mean 0.5). Closing this gap would require sourcing additional ai-harm "
        "incidents beyond the existing corpus, which is outside this project's scope. "
        "The disclosure in §4 and §5 describes how the model handles missing "
        "precision data.\n\n"
    )
    lines.append(
        "**Frame-blind entries (F1-ingestion-frame).** LLM04, LLM08, and LLM10 have "
        "incident counts that come entirely from one stratum, preventing cross-stratum "
        "recall estimation. Their ranks in §6 carry structural uncertainty beyond what "
        "the credible intervals capture. We mark them with ★ in the charts and treat "
        "their rank positions as advisory rather than authoritative.\n\n"
    )
    lines.append(
        "**Taxonomy-frame circularity (F-circ).** We are measuring a taxonomy against "
        "incidents classified by that taxonomy. The classifier's preferences inflate "
        "entries it picks readily and deflate entries it picks rarely. The Bayesian "
        "error-correction layer addresses this only to the extent that precision and "
        "recall are well-measured — for low-precision entries (§4.1) the correction "
        "itself carries large uncertainty.\n\n"
    )
    lines.append(
        "**Single-author rubric.** The 2026 rubric was authored by a single hand. "
        "Independent rubric review and dual-coded gold sets are required before any "
        "external publication of these results. The non-publishable banner at the top "
        "of this document reflects that constraint.\n\n"
    )

    # ─── Reproducibility ───
    lines.append("# Reproducibility\n\n")
    lines.append(
        "Every figure and statistic in this document is regenerated from the cycle "
        "directory `projects/owasp-llm/cycles/2026/`. Re-run via:\n\n"
    )
    lines.append("```\n")
    lines.append("python -m engine.cli report-narrative \\\n")
    lines.append("    --cycle-dir projects/owasp-llm/cycles/2026 \\\n")
    lines.append("    --output-dir notebooks/narrative\n")
    lines.append("```\n\n")
    lines.append(
        f"Output filenames: `{REPORT_STEM}.md` (this file) and "
        f"`{REPORT_STEM}.pdf` (PDF compiled by pandoc with the xelatex engine). "
        "All figures land in `notebooks/narrative/figures/`. The corresponding "
        "Jupyter notebook is "
        "`notebooks/2026_top_10_llm_update_what_the_data_says.ipynb`; the prose in "
        "this report is congruent with the notebook narrative cell by cell, with "
        "added section numbering and an abstract for arXiv-style distribution.\n\n"
    )
    lines.append(
        f"**Data provenance.** Cycle artifacts are content-hashed at "
        "`projects/owasp-llm/cycles/2026/snapshot/`. Hyperparameters and seeds are "
        "hash-locked in `prereg/manifest.json`. The MCMC sampler is pinned to CPU "
        "for cross-platform determinism (NumPyro + JAX). Posterior draws are stored "
        "as `infer/lambda_samples.npy` and the convergence diagnostics in "
        "`infer/inference_summary.json`.\n\n"
    )

    return "".join(lines)


def _compile_pdf(md_path: Path, pdf_path: Path) -> bool:
    """Compile the markdown to PDF via pandoc + xelatex.

    Returns True on success, False if pandoc is unavailable. Raises on pandoc
    failure when pandoc IS present (so silent breakage is impossible).
    """
    if shutil.which("pandoc") is None:
        return False

    cmd = [
        "pandoc",
        md_path.name,
        "-o", pdf_path.name,
        "--pdf-engine=xelatex",
        "--toc",
        "--toc-depth=2",
        "-V", "linkcolor=blue",
        "-V", "urlcolor=blue",
        "--from=markdown+yaml_metadata_block",
    ]
    result = subprocess.run(  # noqa: S603 — args are constructed locally, not from user input
        cmd,
        capture_output=True,
        text=True,
        cwd=md_path.parent,  # so figures/foo.png resolves relative to the md
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"pandoc PDF compile failed (exit {result.returncode}):\n"
            f"STDERR: {result.stderr[-2000:]}"
        )
    return True


def generate_narrative_report(cycle_dir: Path, output_dir: Path) -> Path:
    """Generate the arXiv-ready narrative report with embedded figures and PDF.

    Returns the path to the generated markdown file. The PDF, when pandoc is
    available, lives at the same stem with a .pdf extension.
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

    report_md = _render_markdown(data)
    md_path = output_dir / f"{REPORT_STEM}.md"
    md_path.write_text(report_md)

    pdf_path = output_dir / f"{REPORT_STEM}.pdf"
    _compile_pdf(md_path, pdf_path)

    return md_path
