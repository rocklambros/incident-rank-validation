"""Chart generators for the standalone narrative report.

Each function renders one chart and saves it as a PNG.
All matplotlib charts use the Agg backend for headless rendering.
"""

from __future__ import annotations

import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
import seaborn as sns  # noqa: E402

ENTRY_IDS = [
    "LLM01", "LLM02", "LLM03", "LLM04", "LLM05",
    "LLM06", "LLM07", "LLM08", "LLM09", "LLM10",
    "NEW-ITSCD", "NEW-MA", "NEW-MSDA", "NEW-MTIE", "NEW-PMP", "NEW-WLA",
    "ROLL-CFAS", "ROLL-CMSB", "ROLL-LAPTF", "ROLL-SICG",
]
FRAME_BLIND = {"LLM04", "LLM08", "LLM10"}

sns.set_theme(style="whitegrid", font_scale=1.1)


def _setup_colors() -> dict[str, str]:
    ib = sns.color_palette("mako", n_colors=12)
    no = sns.color_palette("flare", n_colors=8)
    rp = sns.color_palette("crest", n_colors=6)
    colors: dict[str, str] = {}
    ii, ni, ri = 0, 0, 0
    for eid in ENTRY_IDS:
        if eid in FRAME_BLIND:
            colors[eid] = "#999999"
        elif eid.startswith("LLM"):
            c = ib[ii % len(ib)]
            colors[eid] = f"#{int(c[0]*255):02x}{int(c[1]*255):02x}{int(c[2]*255):02x}"
            ii += 1
        elif eid.startswith("NEW"):
            c = no[ni % len(no)]
            colors[eid] = f"#{int(c[0]*255):02x}{int(c[1]*255):02x}{int(c[2]*255):02x}"
            ni += 1
        else:
            c = rp[ri % len(rp)]
            colors[eid] = f"#{int(c[0]*255):02x}{int(c[1]*255):02x}{int(c[2]*255):02x}"
            ri += 1
    return colors


ENTRY_COLORS = _setup_colors()


def render_stratum_bar(data: dict[str, Any], figures_dir: Path) -> None:
    """Act 2: Stratum breakdown bar chart."""
    import pandas as pd

    inc_df = pd.DataFrame(data["incidents"])
    stratum_counts = inc_df["stratum"].value_counts()

    fig, ax = plt.subplots(figsize=(8, 4))
    stratum_counts.plot(kind="bar", ax=ax, color=["#2196F3", "#FF9800"])
    ax.set_title("Incidents by Stratum")
    ax.set_xlabel("Stratum")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(figures_dir / "stratum_bar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_tier_donut(data: dict[str, Any], figures_dir: Path) -> None:
    """Act 3: Tier distribution donut chart."""
    tier_counts = Counter(
        p["triage_tier"]
        for p in data["prelabels"]
        if p.get("consensus") != "out-of-scope"
    )
    labels = ["agree", "split", "disagree"]
    values = [tier_counts.get(t, 0) for t in labels]
    colors_list = ["#4CAF50", "#FFC107", "#F44336"]

    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        values, labels=labels, autopct="%1.0f%%", startangle=90,
        colors=colors_list, pctdistance=0.75,
    )
    centre_circle = plt.Circle((0, 0), 0.50, fc="white")
    ax.add_patch(centre_circle)
    ax.set_title("Consensus Tier Distribution")
    fig.tight_layout()
    fig.savefig(figures_dir / "tier_donut.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_confusion_heatmap(data: dict[str, Any], figures_dir: Path) -> None:
    """Act 3: Entry-pair disagreement heatmap."""
    in_scope_entries = [e for e in ENTRY_IDS if e not in FRAME_BLIND]
    pair_counts: dict[tuple[str, str], int] = defaultdict(int)

    for p in data["prelabels"]:
        if p.get("triage_tier") in ("split", "disagree") and p.get("consensus") != "out-of-scope":
            votes = p.get("model_votes", [])
            unique_entries = {v["entry_id"] for v in votes if isinstance(v, dict) and v.get("entry_id") in set(in_scope_entries)}
            entries_list = sorted(unique_entries)
            for i_idx in range(len(entries_list)):
                for j_idx in range(i_idx + 1, len(entries_list)):
                    pair_counts[(entries_list[i_idx], entries_list[j_idx])] += 1

    n = len(in_scope_entries)
    matrix = np.zeros((n, n), dtype=int)
    for (a, b), count in pair_counts.items():
        if a in in_scope_entries and b in in_scope_entries:
            i = in_scope_entries.index(a)
            j = in_scope_entries.index(b)
            matrix[i, j] = count
            matrix[j, i] = count

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        matrix, xticklabels=in_scope_entries, yticklabels=in_scope_entries,
        cmap="YlOrRd", ax=ax, annot=True, fmt="d",
    )
    ax.set_title("Entry-Pair Disagreement Frequency")
    fig.tight_layout()
    fig.savefig(figures_dir / "confusion_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_precision_bars(data: dict[str, Any], figures_dir: Path) -> None:
    """Act 4: Precision posterior mean bars."""
    precision = data["posteriors"]["precision"]
    entries_with_prec = []
    means = []
    for key, params in sorted(precision.items()):
        eid = key.split("::")[0]
        alpha = params["alpha"]
        beta = params["beta"]
        mean = alpha / (alpha + beta) if (alpha + beta) > 0 else 0
        entries_with_prec.append(eid)
        means.append(mean)

    fig, ax = plt.subplots(figsize=(12, 6))
    y_pos = range(len(entries_with_prec))
    ax.barh(y_pos, means, color=[ENTRY_COLORS.get(e, "#999999") for e in entries_with_prec])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(entries_with_prec, fontsize=9)
    ax.set_xlabel("Precision Posterior Mean")
    ax.set_title("Precision Posteriors (security stratum only)")
    ax.axvline(x=0.5, color="red", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(figures_dir / "precision_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_precision_posteriors(data: dict[str, Any], figures_dir: Path) -> None:
    """Act 4: Beta posterior distributions for key entries."""
    from scipy import stats as sp_stats

    precision = data["posteriors"]["precision"]
    key_entries = ["LLM03", "LLM09", "LLM02", "out-of-scope"]
    x = np.linspace(0, 1, 200)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, eid in zip(axes.flat, key_entries):
        key = f"{eid}::security"
        if key in precision:
            alpha = precision[key]["alpha"]
            beta = precision[key]["beta"]
            y = sp_stats.beta.pdf(x, alpha, beta)
            ax.plot(x, y, linewidth=2)
            ax.fill_between(x, y, alpha=0.3)
            ax.set_title(f"{eid} (α={alpha:.0f}, β={beta:.0f})")
            ax.set_xlabel("Precision")
            ax.set_ylabel("Density")
        else:
            ax.text(0.5, 0.5, f"No data for {eid}", ha="center", va="center")
    fig.suptitle("Precision Posterior Distributions")
    fig.tight_layout()
    fig.savefig(figures_dir / "precision_posteriors.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_ridge_plot(data: dict[str, Any], figures_dir: Path) -> None:
    """Act 5: Ridge plot of posterior lambda for all 20 entries."""
    lambda_samples = data["lambda_samples"]
    entry_ids = data["entry_ids"]
    medians = {eid: float(np.median(lambda_samples[:, i])) for i, eid in enumerate(entry_ids)}
    sorted_entries = sorted(entry_ids, key=lambda e: medians[e], reverse=True)

    fig, axes = plt.subplots(len(sorted_entries), 1, figsize=(10, 16), sharex=True)
    for ax, eid in zip(axes, sorted_entries):
        idx = entry_ids.index(eid)
        vals = lambda_samples[:, idx]
        color = ENTRY_COLORS.get(eid, "#999999")
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(vals, bw_method=0.2)
        x_grid = np.linspace(vals.min(), vals.max(), 200)
        density = kde(x_grid)
        ax.fill_between(x_grid, density, alpha=0.6, color=color)
        ax.set_ylabel(eid, rotation=0, ha="right", fontsize=9)
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[-1].set_xlabel("λ (incident rate)")
    fig.suptitle("Posterior λ Distributions (sorted by median)")
    fig.tight_layout()
    fig.savefig(figures_dir / "ridge_plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_dumbbell_chart(data: dict[str, Any], figures_dir: Path) -> None:
    """Act 6: Dumbbell chart of rank distributions."""
    lambda_samples = data["lambda_samples"]
    entry_ids = data["entry_ids"]

    rank_medians = {}
    rank_cis = {}
    for i, eid in enumerate(entry_ids):
        ranks = np.zeros(lambda_samples.shape[0])
        for s in range(lambda_samples.shape[0]):
            draw = lambda_samples[s]
            order = np.argsort(-draw)
            rank_arr = np.empty_like(order, dtype=float)
            rank_arr[order] = np.arange(1, len(draw) + 1, dtype=float)
            ranks[s] = rank_arr[i]
        rank_medians[eid] = float(np.median(ranks))
        rank_cis[eid] = (float(np.percentile(ranks, 5)), float(np.percentile(ranks, 95)))

    sorted_entries = sorted(entry_ids, key=lambda e: rank_medians[e])

    fig, ax = plt.subplots(figsize=(10, 12))
    y_pos = range(len(sorted_entries))
    for y, eid in zip(y_pos, sorted_entries):
        ci = rank_cis[eid]
        color = ENTRY_COLORS.get(eid, "#999999")
        ax.plot([ci[0], ci[1]], [y, y], color=color, linewidth=2, alpha=0.6)
        ax.scatter([rank_medians[eid]], [y], color=color, s=80, zorder=5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{e} ({data['entry_names'].get(e, e)})" for e in sorted_entries], fontsize=9)
    ax.set_xlabel("Rank (90% CI)")
    ax.set_title("Incident-Derived Rankings with Uncertainty")
    ax.invert_xaxis()
    fig.tight_layout()
    fig.savefig(figures_dir / "dumbbell_chart.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_bump_chart(data: dict[str, Any], figures_dir: Path) -> None:
    """Act 7: Bump chart comparing expert and incident ranks."""
    conc = data["concordance"]
    rank_md = data["rank_comparison_md"]

    vote_ranks: dict[str, float] = {}
    lambda_ranks: dict[str, float] = {}
    for line in rank_md.split("\n"):
        if "|" in line and not line.startswith("|--") and "Entry" not in line:
            parts = [p.strip() for p in line.split("|") if p.strip()]
            if len(parts) >= 3:
                eid = parts[0]
                try:
                    lam_med = float(parts[1].split("(")[0].strip())
                    vote_med = float(parts[2].split("(")[0].strip())
                    lambda_ranks[eid] = lam_med
                    vote_ranks[eid] = vote_med
                except (ValueError, IndexError):
                    continue

    common = sorted(set(lambda_ranks) & set(vote_ranks))
    if not common:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No rank data available", ha="center", va="center")
        fig.savefig(figures_dir / "bump_chart.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=(10, 12))
    for eid in common:
        color = ENTRY_COLORS.get(eid, "#999999")
        ax.plot(
            [0, 1], [lambda_ranks[eid], vote_ranks[eid]],
            marker="o", color=color, linewidth=2, markersize=8,
        )
        ax.annotate(eid, (0, lambda_ranks[eid]), textcoords="offset points",
                     xytext=(-50, 0), fontsize=9, ha="right")
        ax.annotate(eid, (1, vote_ranks[eid]), textcoords="offset points",
                     xytext=(10, 0), fontsize=9, ha="left")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Incident Rank", "Expert Rank"])
    ax.set_ylabel("Rank")
    ax.invert_yaxis()
    ax.set_title("Expert vs Incident Rankings")
    fig.tight_layout()
    fig.savefig(figures_dir / "bump_chart.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_ci_overlap(data: dict[str, Any], figures_dir: Path) -> None:
    """Act 7: CI overlap between lambda and vote rank CIs."""
    rank_md = data["rank_comparison_md"]
    entries_data: list[dict[str, Any]] = []

    for line in rank_md.split("\n"):
        if "|" in line and not line.startswith("|--") and "Entry" not in line:
            parts = [p.strip() for p in line.split("|") if p.strip()]
            if len(parts) >= 3:
                eid = parts[0]
                try:
                    lam_part = parts[1]
                    vote_part = parts[2]
                    lam_med = float(lam_part.split("(")[0].strip())
                    lam_ci_str = lam_part.split("(")[1].rstrip(")")
                    lam_lo, lam_hi = [float(x) for x in lam_ci_str.replace("–", "-").split("-") if x]
                    vote_med = float(vote_part.split("(")[0].strip())
                    vote_ci_str = vote_part.split("(")[1].rstrip(")")
                    vote_lo, vote_hi = [float(x) for x in vote_ci_str.replace("–", "-").split("-") if x]
                    entries_data.append({
                        "entry_id": eid, "lam_med": lam_med, "lam_lo": lam_lo, "lam_hi": lam_hi,
                        "vote_med": vote_med, "vote_lo": vote_lo, "vote_hi": vote_hi,
                    })
                except (ValueError, IndexError):
                    continue

    if not entries_data:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No CI data", ha="center")
        fig.savefig(figures_dir / "ci_overlap.png", dpi=150)
        plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=(12, 10))
    for i, ed in enumerate(entries_data):
        color = ENTRY_COLORS.get(ed["entry_id"], "#999999")
        ax.plot([ed["lam_lo"], ed["lam_hi"]], [i - 0.1, i - 0.1], color=color, linewidth=3, alpha=0.7)
        ax.plot([ed["vote_lo"], ed["vote_hi"]], [i + 0.1, i + 0.1], color=color, linewidth=3, alpha=0.4, linestyle="--")
        ax.scatter([ed["lam_med"]], [i - 0.1], color=color, s=60, zorder=5)
        ax.scatter([ed["vote_med"]], [i + 0.1], color=color, s=60, zorder=5, marker="^")
    ax.set_yticks(range(len(entries_data)))
    ax.set_yticklabels([ed["entry_id"] for ed in entries_data], fontsize=9)
    ax.set_xlabel("Rank")
    ax.set_title("CI Overlap: Incident (solid) vs Expert (dashed)")
    fig.tight_layout()
    fig.savefig(figures_dir / "ci_overlap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_paired_dots(data: dict[str, Any], figures_dir: Path) -> None:
    """Act 8: Paired dot plots for flagged entries."""
    flags = data["concordance"].get("flags", [])
    if not flags:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No flagged entries", ha="center")
        fig.savefig(figures_dir / "paired_dots.png", dpi=150)
        plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, f in enumerate(flags):
        eid = f["entry_id"]
        prob = f["probability"]
        direction = f["direction"]
        color = ENTRY_COLORS.get(eid, "#999999")
        ax.barh(i, prob, color=color, alpha=0.7)
        label = "↑ expert" if direction == "vote_over_ranks" else "↓ expert"
        ax.text(prob + 0.01, i, f"{eid} ({label})", va="center", fontsize=10)
    ax.set_xlabel("P(tier mismatch)")
    ax.set_title("Flagged Entries: Expert vs Incident Rank Divergence")
    ax.set_yticks([])
    ax.axvline(x=0.8, color="red", linestyle="--", alpha=0.5, label="τ = 0.80")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "paired_dots.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_theme_bars(data: dict[str, Any], figures_dir: Path, entry_id: str, filename: str) -> None:
    """Act 8: Theme keyword bars for a specific entry."""
    theme_keywords = {
        "deepfake": ["deepfake", "synthetic media", "face swap"],
        "misinfo": ["disinformation", "misinformation", "fake news"],
        "voice_clone": ["voice clone", "voice synthesis", "audio deepfake"],
        "code_gen": ["code generation", "copilot", "code completion"],
        "data_leak": ["data leak", "data exposure", "information disclosure"],
        "prompt_inject": ["prompt injection", "jailbreak", "prompt attack"],
        "supply_chain": ["supply chain", "dependency", "package"],
        "agent_abuse": ["agent", "autonomous", "tool use", "mcp"],
    }

    entry_prelabels = [
        p for p in data["prelabels"]
        if p.get("consensus") == entry_id
    ]

    theme_counts: dict[str, int] = defaultdict(int)
    for p in entry_prelabels:
        text = (p.get("text", "") or "").lower()
        for theme, keywords in theme_keywords.items():
            if any(kw in text for kw in keywords):
                theme_counts[theme] += 1

    if not theme_counts:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, f"No theme keywords found for {entry_id}", ha="center")
        fig.savefig(figures_dir / filename, dpi=150)
        plt.close(fig)
        return

    sorted_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)
    themes, counts = zip(*sorted_themes)

    fig, ax = plt.subplots(figsize=(8, 4))
    color = ENTRY_COLORS.get(entry_id, "#999999")
    ax.barh(range(len(themes)), counts, color=color, alpha=0.8)
    ax.set_yticks(range(len(themes)))
    ax.set_yticklabels(themes, fontsize=10)
    ax.set_xlabel("Keyword Frequency")
    ax.set_title(f"Incident Themes: {entry_id}")
    fig.tight_layout()
    fig.savefig(figures_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_confusion_matrix_3x3(data: dict[str, Any], figures_dir: Path) -> None:
    """Act 9B: 3x3 confusion matrix for boundary entries."""
    boundary_entries = ["LLM09", "NEW-WLA", "ROLL-CMSB"]

    pair_disagree: dict[tuple[str, str], int] = defaultdict(int)
    for p in data["prelabels"]:
        votes = p.get("model_votes", [])
        unique_votes = {v["entry_id"] for v in votes if isinstance(v, dict) and v.get("entry_id") in set(boundary_entries)}
        if len(unique_votes) >= 2:
            vote_list = sorted(unique_votes)
            for i_idx in range(len(vote_list)):
                for j_idx in range(i_idx + 1, len(vote_list)):
                    pair_disagree[(vote_list[i_idx], vote_list[j_idx])] += 1

    n = len(boundary_entries)
    matrix = np.zeros((n, n), dtype=int)
    for (a, b), count in pair_disagree.items():
        if a in boundary_entries and b in boundary_entries:
            i = boundary_entries.index(a)
            j = boundary_entries.index(b)
            matrix[i, j] = count
            matrix[j, i] = count

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        matrix, xticklabels=boundary_entries, yticklabels=boundary_entries,
        cmap="YlOrRd", ax=ax, annot=True, fmt="d",
    )
    ax.set_title("Confusion Boundary: Model Disagreement")
    fig.tight_layout()
    fig.savefig(figures_dir / "confusion_matrix_3x3.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_all_matplotlib_charts(data: dict[str, Any], figures_dir: Path) -> None:
    """Generate all matplotlib-based charts."""
    render_stratum_bar(data, figures_dir)
    render_tier_donut(data, figures_dir)
    render_confusion_heatmap(data, figures_dir)
    render_precision_bars(data, figures_dir)
    render_precision_posteriors(data, figures_dir)
    render_ridge_plot(data, figures_dir)
    render_dumbbell_chart(data, figures_dir)
    render_bump_chart(data, figures_dir)
    render_ci_overlap(data, figures_dir)
    render_paired_dots(data, figures_dir)
    render_theme_bars(data, figures_dir, "LLM09", "theme_bars_llm09.png")
    render_theme_bars(data, figures_dir, "NEW-WLA", "theme_bars_new_wla.png")
    render_confusion_matrix_3x3(data, figures_dir)


def render_plotly_rankings(data: dict[str, Any], figures_dir: Path) -> None:
    """Act 6: Interactive-style rankings as static PNG."""
    import plotly.express as px

    lambda_samples = data["lambda_samples"]
    entry_ids = data["entry_ids"]
    entry_names = data["entry_names"]

    rows = []
    for i, eid in enumerate(entry_ids):
        med = float(np.median(lambda_samples[:, i]))
        lo = float(np.percentile(lambda_samples[:, i], 5))
        hi = float(np.percentile(lambda_samples[:, i], 95))
        rows.append({"entry_id": eid, "name": entry_names.get(eid, eid), "median": med, "lo": lo, "hi": hi})

    import pandas as pd
    df = pd.DataFrame(rows).sort_values("median", ascending=False)

    fig = px.bar(
        df, x="median", y="entry_id", orientation="h",
        error_x_minus=df["median"] - df["lo"],
        error_x=df["hi"] - df["median"],
        title="Incident-Derived Rankings (λ posteriors)",
        labels={"median": "λ (posterior median)", "entry_id": "Entry"},
    )
    fig.update_layout(height=600, width=1000, yaxis={"categoryorder": "total ascending"})
    fig.write_image(str(figures_dir / "plotly_rankings.png"), width=1000, height=600)


def render_oos_treemap(data: dict[str, Any], figures_dir: Path) -> None:
    """Act 9A: Out-of-scope incident treemap."""
    import plotly.express as px

    oos_prelabels = [
        p for p in data["prelabels"]
        if p.get("consensus") == "out-of-scope"
    ]

    theme_keywords = {
        "Surveillance / Facial Recognition": ["surveillance", "facial recognition", "biometric", "monitoring"],
        "Algorithmic Discrimination": ["discrimination", "bias", "discriminat", "racial", "gender bias", "hiring"],
        "Deepfake / Synthetic Media": ["deepfake", "deep fake", "synthetic media", "face swap", "generated image"],
        "Autonomous Vehicles": ["self-driving", "autonomous vehicle", "autopilot", "tesla"],
        "AI Labor & Employment": ["worker", "labor", "employment", "automation", "job loss"],
        "Copyright & IP": ["copyright", "intellectual property", "plagiarism", "training data"],
        "CSAM / NCII": ["csam", "child sexual", "ncii", "non-consensual intimate"],
        "Healthcare AI": ["healthcare", "medical", "diagnosis", "patient", "clinical"],
        "Military / Weapons": ["military", "weapon", "drone strike", "lethal autonomous", "warfare"],
        "Other": [],
    }

    cluster_counts: dict[str, int] = defaultdict(int)
    for p in oos_prelabels:
        text = (p.get("text", "") or "").lower()
        matched = False
        for cluster, keywords in theme_keywords.items():
            if cluster == "Other":
                continue
            if any(kw in text for kw in keywords):
                cluster_counts[cluster] += 1
                matched = True
                break
        if not matched:
            cluster_counts["Other"] += 1

    if not cluster_counts:
        # Write a placeholder
        fig = px.treemap(
            names=["No OOS data"], parents=[""], values=[1],
            title="Out-of-Scope Incidents (no data)",
        )
        fig.write_image(str(figures_dir / "oos_treemap.png"), width=1000, height=600)
        return

    import pandas as pd
    df = pd.DataFrame([
        {"cluster": k, "count": v, "parent": "Out-of-Scope"}
        for k, v in cluster_counts.items()
    ])

    fig = px.treemap(
        df, path=["parent", "cluster"], values="count",
        title=f"Out-of-Scope Incidents by Theme ({sum(cluster_counts.values())} total)",
    )
    fig.update_layout(width=1000, height=600)
    fig.write_image(str(figures_dir / "oos_treemap.png"), width=1000, height=600)


def render_sankey_confusion(data: dict[str, Any], figures_dir: Path) -> None:
    """Act 9B: Sankey diagram for confusion boundary entries."""
    import plotly.graph_objects as go_plotly

    boundary = ["LLM09", "NEW-WLA", "ROLL-CMSB"]

    flows: dict[tuple[str, str], int] = defaultdict(int)
    for p in data["prelabels"]:
        votes = p.get("model_votes", [])
        consensus = p.get("consensus", "")
        if consensus not in boundary:
            continue
        for v in votes:
            if not isinstance(v, dict):
                continue
            model_id = v.get("model_id", "")
            model_short = model_id.split("/")[-1].split("-")[0].lower() if "/" in model_id else model_id.lower()
            vote_entry = v.get("entry_id", "")
            if vote_entry in boundary:
                flows[(f"{model_short}: {vote_entry}", consensus)] += 1

    if not flows:
        fig = go_plotly.Figure()
        fig.add_annotation(text="No confusion boundary data", x=0.5, y=0.5, showarrow=False)
        fig.write_image(str(figures_dir / "sankey_confusion.png"), width=1000, height=600)
        return

    all_labels = sorted(set(
        [k[0] for k in flows] + [k[1] for k in flows]
    ))
    label_idx = {lb: i for i, lb in enumerate(all_labels)}

    source = [label_idx[k[0]] for k in flows]
    target = [label_idx[k[1]] for k in flows]
    value = list(flows.values())

    fig = go_plotly.Figure(data=[go_plotly.Sankey(
        node=dict(label=all_labels, pad=15, thickness=20),
        link=dict(source=source, target=target, value=value),
    )])
    fig.update_layout(title="Model Votes → Consensus (Confusion Boundary)", width=1000, height=600)
    fig.write_image(str(figures_dir / "sankey_confusion.png"), width=1000, height=600)


def generate_all_plotly_charts(data: dict[str, Any], figures_dir: Path) -> None:
    """Generate all plotly-based charts (rendered as static PNG via kaleido)."""
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="setDaemon", category=DeprecationWarning)
        render_plotly_rankings(data, figures_dir)
        render_oos_treemap(data, figures_dir)
        render_sankey_confusion(data, figures_dir)
