"""Chart generators for the standalone narrative report.

Each function renders one chart and saves it as a PNG.
All matplotlib charts use the Agg backend for headless rendering.
"""

from __future__ import annotations

import os
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")


def _plotly_write_image(fig: Any, path: str, **kw: Any) -> None:
    """Wrapper around fig.write_image that scopes a narrow warning filter.

    Plotly 6.x emits an internal DeprecationWarning whenever write_image() is
    called — the message tells callers to upgrade past Sep-2025, but the
    warning originates inside plotly's own code path (not ours) and we are
    already on kaleido 1.x which is the post-deprecation API. We suppress only
    that specific DeprecationWarning by category + message, scoped to the call.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            # (?s) = DOTALL — plotly's message starts with '\n', so '.' must match newlines
            message=r"(?s).*Support for the 'engine' argument is deprecated.*",
            category=DeprecationWarning,
        )
        fig.write_image(path, **kw)

import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

ENTRY_IDS = [
    "LLM01", "LLM02", "LLM03", "LLM04", "LLM05",
    "LLM06", "LLM07", "LLM08", "LLM09", "LLM10",
    "NEW-ITSCD", "NEW-MA", "NEW-MSDA", "NEW-MTIE", "NEW-PMP", "NEW-WLA",
    "ROLL-CFAS", "ROLL-CMSB", "ROLL-LAPTF", "ROLL-SICG",
]
FRAME_BLIND = {"LLM04", "LLM08", "LLM10"}

# Official 2025 OWASP LLM Top-10 published names (fixed historical facts) —
# used on the left axis of the 2025->2026 slopegraph.
PUBLISHED_2025_NAMES = {
    "LLM01": "Prompt Injection",
    "LLM02": "Sensitive Information Disclosure",
    "LLM03": "Supply Chain",
    "LLM04": "Data and Model Poisoning",
    "LLM05": "Improper Output Handling",
    "LLM06": "Excessive Agency",
    "LLM07": "System Prompt Leakage",
    "LLM08": "Vector and Embedding Weaknesses",
    "LLM09": "Misinformation",
    "LLM10": "Unbounded Consumption",
}
# Incumbents whose 2026 canonical name is a true rename (not a cosmetic diff).
RENAMED_2026 = {"LLM07"}


def _rank_change_rows(
    blended: list[dict[str, Any]], entry_names: dict[str, str]
) -> list[dict[str, Any]]:
    """Row model for the 2025->2026 incumbent slopegraph.

    published rank = int(LLMkk); move = published - blend_rank (positive = moved
    up toward #1). Style band: nc (move 0) / hold (|move|==1) / mover (|move|>=2).
    New right-side code = LLM<blend_rank:02d>.
    """
    rows: list[dict[str, Any]] = []
    for item in blended:
        eid = item["entry_id"]
        if not (eid.startswith("LLM") and eid[3:].isdigit()):
            continue  # slopegraph is incumbents-only
        pub = int(eid[3:])
        blend_rank = int(item["blend_rank"])
        move = pub - blend_rank
        if move == 0:
            style = "nc"
        elif abs(move) == 1:
            style = "hold"
        else:
            style = "mover"
        rows.append({
            "left_num": pub,
            "left_code": eid,
            "left_name": PUBLISHED_2025_NAMES.get(eid, eid),
            "right_code": f"LLM{blend_rank:02d}",
            "right_name": entry_names.get(eid, eid),
            "move": move,
            "renamed": eid in RENAMED_2026,
            "style": style,
        })
    return rows


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
    fig.savefig(figures_dir / "stratum_bar.png", dpi=300, bbox_inches="tight")
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
    centre_circle = mpatches.Circle((0, 0), 0.50, fc="white")
    ax.add_patch(centre_circle)
    ax.set_title("Consensus Tier Distribution")
    fig.tight_layout()
    fig.savefig(figures_dir / "tier_donut.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def render_confusion_heatmap(data: dict[str, Any], figures_dir: Path) -> None:
    """Act 3: Entry-pair disagreement heatmap."""
    in_scope_entries = [e for e in ENTRY_IDS if e not in FRAME_BLIND]
    pair_counts: dict[tuple[str, str], int] = defaultdict(int)

    for p in data["prelabels"]:
        if p.get("triage_tier") in ("split", "disagree") and p.get("consensus") != "out-of-scope":
            votes = p.get("model_votes", [])
            unique_entries = {
                v["entry_id"]
                for v in votes
                if isinstance(v, dict) and v.get("entry_id") in set(in_scope_entries)
            }
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
    fig.savefig(figures_dir / "confusion_heatmap.png", dpi=300, bbox_inches="tight")
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
    fig.savefig(figures_dir / "precision_bars.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def render_precision_posteriors(data: dict[str, Any], figures_dir: Path) -> None:
    """Act 4: Beta posterior distributions for key entries."""
    from scipy import stats as sp_stats

    precision = data["posteriors"]["precision"]
    key_entries = ["LLM03", "LLM09", "LLM02", "out-of-scope"]
    x = np.linspace(0, 1, 200)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, eid in zip(axes.flat, key_entries, strict=False):
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
    fig.savefig(figures_dir / "precision_posteriors.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def render_ridge_plot(data: dict[str, Any], figures_dir: Path) -> None:
    """Act 5: overlapping ridgeline (joyplot) of posterior lambda for 20 entries."""
    from scipy.stats import gaussian_kde
    lambda_samples = data["lambda_samples"]
    entry_ids = data["entry_ids"]
    medians = {e: float(np.median(lambda_samples[:, i])) for i, e in enumerate(entry_ids)}
    order = sorted(entry_ids, key=lambda e: medians[e])  # low at bottom, high at top

    n = len(order)
    fig, ax = plt.subplots(figsize=(10, 6.8))
    x_lo = float(lambda_samples.min())
    x_hi = float(np.percentile(lambda_samples, 99.5))
    xs = np.linspace(x_lo, x_hi, 400)
    pitch = 1.0            # vertical spacing between baselines
    scale = 2.1 * pitch    # KDE height (>pitch => overlap)
    for row, eid in enumerate(order):
        idx = entry_ids.index(eid)
        vals = lambda_samples[:, idx]
        # A near-degenerate column (near-zero variance) can make gaussian_kde's
        # internal covariance matrix singular, raising LinAlgError from the
        # constructor -- that genuine failure mode is still guarded below.
        # But the more common degenerate case does NOT raise at all: kde(xs)
        # silently returns an all-zero density array. The old guard stopped
        # there and let `dens / dens.max()` run unguarded, which is a 0/0
        # division -> RuntimeWarning that this repo's filterwarnings=["error"]
        # policy turns into a hard crash. So we must check dens for
        # finiteness/positivity BEFORE normalizing, not just check dens is
        # not None.
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("error", category=RuntimeWarning)
                kde = gaussian_kde(vals, bw_method=0.25)
                dens = kde(xs)
        except (np.linalg.LinAlgError, RuntimeWarning, ValueError):
            dens = None
        base = row * pitch
        color = ENTRY_COLORS.get(eid, "#999999")
        if dens is not None and np.isfinite(dens).all() and dens.max() > 0:
            dens = dens / dens.max() * scale
            ax.fill_between(xs, base, base + dens, color=color, alpha=0.85,
                            zorder=n - row, lw=0)
            ax.plot(xs, base + dens, color="white", lw=0.6, zorder=n - row)
        # else: degenerate (near-)zero-variance column -- no curve to draw,
        # but the row label below still renders so the entry is accounted for.
        ax.text(x_lo, base + 0.15, eid, ha="right", va="bottom", fontsize=9,
                color=color, fontweight="bold")
    ax.set_yticks([])
    ax.set_xlabel("λ  (latent incidence)", fontsize=12)
    for side in ("left", "top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xlim(x_lo - (x_hi - x_lo) * 0.12, x_hi)
    ax.set_title("Posterior λ by entry (sorted by median)", fontsize=13)
    fig.tight_layout()
    fig.savefig(figures_dir / "ridge_plot.png", dpi=300, bbox_inches="tight")
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

    fig, ax = plt.subplots(figsize=(11, 7.2))
    y_pos = range(len(sorted_entries))
    for y, eid in zip(y_pos, sorted_entries, strict=False):
        lo, hi = rank_cis[eid]
        color = ENTRY_COLORS.get(eid, "#999999")
        ax.plot([lo, hi], [y, y], color=color, linewidth=3.0, alpha=0.55,
                solid_capstyle="round")
        ax.scatter([rank_medians[eid]], [y], color=color, s=70, zorder=5)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(
        [f"{e}  {data['entry_names'].get(e, e)}" for e in sorted_entries], fontsize=10
    )
    ax.set_xlabel("Incident-derived rank (median · 90% CI)", fontsize=12)
    ax.invert_xaxis()
    ax.margins(y=0.02)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(figures_dir / "dumbbell_chart.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def render_bump_chart(data: dict[str, Any], figures_dir: Path) -> None:
    """Act 7: expert-vs-incident slopegraph; only flagged mismatches highlighted."""
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
                except (ValueError, IndexError):
                    continue
                lambda_ranks[eid] = lam_med
                vote_ranks[eid] = vote_med

    common = sorted(set(lambda_ranks) & set(vote_ranks))
    if not common:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No rank data available", ha="center", va="center")
        fig.savefig(figures_dir / "bump_chart.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    flagged = {f["entry_id"] for f in data.get("concordance", {}).get("flags", [])}

    def _decollide(entries: list[str], ycol: dict[str, float]) -> dict[str, float]:
        # greedy: sort by y, push apart to >= min_gap in label space
        order = sorted(entries, key=lambda e: ycol[e])
        min_gap = 0.9
        placed: dict[str, float] = {}
        last = -1e9
        for e in order:
            y = max(ycol[e], last + min_gap)
            placed[e] = y
            last = y
        return placed

    # Only flagged (drawn) entries get labels, so only they should consume
    # the de-collision spacing budget — otherwise undrawn entries can push
    # flagged labels further apart than necessary, or leave co-located
    # flagged labels (e.g. two entries sharing the same incident rank)
    # nearly overlapping.
    flagged_common = [e for e in common if e in flagged]
    left_lab = _decollide(flagged_common, lambda_ranks)
    right_lab = _decollide(flagged_common, vote_ranks)

    fig, ax = plt.subplots(figsize=(11, 7.2))
    for eid in common:
        hot = eid in flagged
        color = ENTRY_COLORS.get(eid, "#666666") if hot else "#D4D7DC"
        ax.plot([0, 1], [lambda_ranks[eid], vote_ranks[eid]],
                color=color, lw=2.6 if hot else 1.1, alpha=0.95 if hot else 0.6,
                zorder=3 if hot else 1, solid_capstyle="round")
        if hot:
            ax.scatter([0, 1], [lambda_ranks[eid], vote_ranks[eid]],
                       color=color, s=42, zorder=4)
            ax.annotate(eid, (0, left_lab[eid]), textcoords="offset points",
                        xytext=(-10, 0), ha="right", va="center",
                        fontsize=11, color=color, fontweight="bold")
            ax.annotate(eid, (1, right_lab[eid]), textcoords="offset points",
                        xytext=(10, 0), ha="left", va="center",
                        fontsize=11, color=color, fontweight="bold")
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(max(list(vote_ranks.values()) + list(lambda_ranks.values())) + 0.6, 0.4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Incident rank", "Expert rank"], fontsize=13, fontweight="bold")
    ax.set_ylabel("Rank", fontsize=12)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title(
        f"Expert vs incident rank — the {len(flagged)} flagged disagreements highlighted",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(figures_dir / "bump_chart.png", dpi=300, bbox_inches="tight")
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
                    lam_lo, lam_hi = [
                        float(x) for x in lam_ci_str.replace("–", "-").split("-") if x
                    ]
                    vote_med = float(vote_part.split("(")[0].strip())
                    vote_ci_str = vote_part.split("(")[1].rstrip(")")
                    vote_lo, vote_hi = [
                        float(x) for x in vote_ci_str.replace("–", "-").split("-") if x
                    ]
                    entries_data.append({
                        "entry_id": eid, "lam_med": lam_med, "lam_lo": lam_lo, "lam_hi": lam_hi,
                        "vote_med": vote_med, "vote_lo": vote_lo, "vote_hi": vote_hi,
                    })
                except (ValueError, IndexError):
                    continue

    if not entries_data:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No CI data", ha="center")
        fig.savefig(figures_dir / "ci_overlap.png", dpi=300)
        plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=(12, 10))
    for i, ed in enumerate(entries_data):
        color = ENTRY_COLORS.get(ed["entry_id"], "#999999")
        ax.plot(
            [ed["lam_lo"], ed["lam_hi"]], [i - 0.1, i - 0.1], color=color, linewidth=3, alpha=0.7
        )
        ax.plot(
            [ed["vote_lo"], ed["vote_hi"]],
            [i + 0.1, i + 0.1],
            color=color,
            linewidth=3,
            alpha=0.4,
            linestyle="--",
        )
        ax.scatter([ed["lam_med"]], [i - 0.1], color=color, s=60, zorder=5)
        ax.scatter([ed["vote_med"]], [i + 0.1], color=color, s=60, zorder=5, marker="^")
    ax.set_yticks(range(len(entries_data)))
    ax.set_yticklabels([ed["entry_id"] for ed in entries_data], fontsize=9)
    ax.set_xlabel("Rank")
    ax.set_title("CI Overlap: Incident (solid) vs Expert (dashed)")
    fig.tight_layout()
    fig.savefig(figures_dir / "ci_overlap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def render_paired_dots(data: dict[str, Any], figures_dir: Path) -> None:
    """Act 8: Paired dot plots for flagged entries."""
    flags = data["concordance"].get("flags", [])
    if not flags:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No flagged entries", ha="center")
        fig.savefig(figures_dir / "paired_dots.png", dpi=300)
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
    fig.savefig(figures_dir / "paired_dots.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def render_theme_bars(
    data: dict[str, Any], figures_dir: Path, entry_id: str, filename: str
) -> None:
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
        fig.savefig(figures_dir / filename, dpi=300)
        plt.close(fig)
        return

    sorted_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)
    themes, counts = zip(*sorted_themes, strict=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    color = ENTRY_COLORS.get(entry_id, "#999999")
    ax.barh(range(len(themes)), counts, color=color, alpha=0.8)
    ax.set_yticks(range(len(themes)))
    ax.set_yticklabels(themes, fontsize=10)
    ax.set_xlabel("Keyword Frequency")
    ax.set_title(f"Incident Themes: {entry_id}")
    fig.tight_layout()
    fig.savefig(figures_dir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def render_confusion_matrix_3x3(data: dict[str, Any], figures_dir: Path) -> None:
    """Act 9B: 3x3 confusion matrix for boundary entries."""
    boundary_entries = ["LLM09", "NEW-WLA", "ROLL-CMSB"]

    pair_disagree: dict[tuple[str, str], int] = defaultdict(int)
    for p in data["prelabels"]:
        votes = p.get("model_votes", [])
        unique_votes = {
            v["entry_id"]
            for v in votes
            if isinstance(v, dict) and v.get("entry_id") in set(boundary_entries)
        }
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
    fig.savefig(figures_dir / "confusion_matrix_3x3.png", dpi=300, bbox_inches="tight")
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
        rows.append(
            {"entry_id": eid, "name": entry_names.get(eid, eid), "median": med, "lo": lo, "hi": hi}
        )

    import pandas as pd
    df = pd.DataFrame(rows).sort_values("median", ascending=False)

    fig = px.bar(
        df, x="median", y="entry_id", orientation="h",
        error_x_minus=df["median"] - df["lo"],
        error_x=df["hi"] - df["median"],
        title="Incident-Derived Rankings (λ posteriors)",
        labels={"median": "λ (posterior median)", "entry_id": "Entry"},
    )
    fig.update_layout(height=1200, width=2000, yaxis={"categoryorder": "total ascending"})
    _plotly_write_image(fig, str(figures_dir / "plotly_rankings.png"), width=2000, height=1200)


def render_oos_treemap(data: dict[str, Any], figures_dir: Path) -> None:
    """Act 9A: Out-of-scope incident treemap."""
    import plotly.express as px

    oos_prelabels = [
        p for p in data["prelabels"]
        if p.get("consensus") == "out-of-scope"
    ]

    theme_keywords = {
        "Surveillance / Facial Recognition": [
            "surveillance",
            "facial recognition",
            "biometric",
            "monitoring",
        ],
        "Algorithmic Discrimination": [
            "discrimination",
            "bias",
            "discriminat",
            "racial",
            "gender bias",
            "hiring",
        ],
        "Deepfake / Synthetic Media": [
            "deepfake",
            "deep fake",
            "synthetic media",
            "face swap",
            "generated image",
        ],
        "Autonomous Vehicles": ["self-driving", "autonomous vehicle", "autopilot", "tesla"],
        "AI Labor & Employment": ["worker", "labor", "employment", "automation", "job loss"],
        "Copyright & IP": ["copyright", "intellectual property", "plagiarism", "training data"],
        "CSAM / NCII": ["csam", "child sexual", "ncii", "non-consensual intimate"],
        "Healthcare AI": ["healthcare", "medical", "diagnosis", "patient", "clinical"],
        "Military / Weapons": [
            "military", "weapon", "drone strike", "lethal autonomous", "warfare",
        ],
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
        _plotly_write_image(fig, str(figures_dir / "oos_treemap.png"), width=1500, height=1050)
        return

    import pandas as pd
    df = pd.DataFrame([
        {"cluster": k, "count": v, "parent": "Out-of-Scope"}
        for k, v in cluster_counts.items()
    ])

    fig = px.treemap(
        df, path=["parent", "cluster"], values="count",
        title=f"Out-of-scope incidents by theme ({sum(cluster_counts.values())} total)",
    )
    fig.update_traces(
        textinfo="label+value+percent parent",
        textfont_size=22,
        insidetextfont=dict(size=22, color="white"),
        marker=dict(line=dict(width=2, color="white")),
    )
    fig.update_layout(
        width=1500, height=1050, margin=dict(t=90, l=10, r=10, b=10),
        title_font_size=30, uniformtext=dict(minsize=16, mode="hide"),
    )
    _plotly_write_image(fig, str(figures_dir / "oos_treemap.png"), width=1500, height=1050)


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
            model_short = (
                model_id.split("/")[-1].split("-")[0].lower()
                if "/" in model_id
                else model_id.lower()
            )
            vote_entry = v.get("entry_id", "")
            if vote_entry in boundary:
                flows[(f"{model_short}: {vote_entry}", consensus)] += 1

    if not flows:
        fig = go_plotly.Figure()
        fig.add_annotation(text="No confusion boundary data", x=0.5, y=0.5, showarrow=False)
        _plotly_write_image(fig, str(figures_dir / "sankey_confusion.png"), width=1600, height=1000)
        return

    all_labels = sorted(set(
        [k[0] for k in flows] + [k[1] for k in flows]
    ))
    label_idx = {lb: i for i, lb in enumerate(all_labels)}

    source = [label_idx[k[0]] for k in flows]
    target = [label_idx[k[1]] for k in flows]
    value = list(flows.values())

    node_colors = [
        ENTRY_COLORS.get(lb.split(": ")[-1], "#888888") for lb in all_labels
    ]

    # Colour each link by its source node, semi-transparent so overlapping
    # flows stay legible.
    def _rgba(hex_c: str, a: float = 0.45) -> str:
        h = hex_c.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{a})"

    link_colors = [_rgba(node_colors[s]) for s in source]

    # Node totals: every node in this sankey is either a "model: entry" vote
    # node (always a link *source*, never a target — labels contain ": ") or
    # a bare consensus-entry node (always a link *target*, never a source —
    # labels have no ": "). That label-format invariant is verified against
    # real cycle data (see task-6 report) so the two sums below never both
    # apply to the same node; summing per-direction directly (rather than
    # rescanning source/target/value once per node) is O(edges) instead of
    # O(nodes * edges) and doesn't rely on a truthy-zero fallback to pick the
    # right side.
    outbound: dict[int, int] = defaultdict(int)
    inbound: dict[int, int] = defaultdict(int)
    for s, t, v in zip(source, target, value, strict=False):
        outbound[s] += v
        inbound[t] += v
    node_totals = [outbound.get(i, 0) or inbound.get(i, 0) for i in range(len(all_labels))]
    node_labels = [f"{lb}  ({v})" for lb, v in zip(all_labels, node_totals, strict=False)]

    fig = go_plotly.Figure(data=[go_plotly.Sankey(
        node=dict(label=node_labels, color=node_colors, pad=22, thickness=26,
                  line=dict(color="white", width=1)),
        link=dict(source=source, target=target, value=value, color=link_colors),
    )])
    fig.update_layout(
        title="Model votes → consensus at the confusion boundary",
        font=dict(size=18), title_font_size=26,
        width=1600, height=1000, margin=dict(t=80, l=10, r=10, b=60),
    )
    _plotly_write_image(fig, str(figures_dir / "sankey_confusion.png"), width=1600, height=1000)


def generate_all_plotly_charts(data: dict[str, Any], figures_dir: Path) -> None:
    """Generate all plotly-based charts (rendered as static PNG via kaleido)."""
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="setDaemon", category=DeprecationWarning)
        render_plotly_rankings(data, figures_dir)
        render_oos_treemap(data, figures_dir)
        render_sankey_confusion(data, figures_dir)


# ---------------------------------------------------------------------------
# Preprint charts (Task 3)
# ---------------------------------------------------------------------------


def render_rank_change_2025_2026(
    blended: list[dict[str, Any]],
    entry_names: dict[str, str],
    figures_dir: Path,
) -> Path:
    """Reference-matched 2025->2026 incumbent slopegraph (replaces the §4.2 table)."""
    out = figures_dir / "rank_change_2025_2026.png"
    rows = _rank_change_rows(blended, entry_names)

    MOVER = "#E8590C"   # orange
    GREY = "#9AA0A6"
    style_kw: dict[str, dict[str, Any]] = {
        "mover": dict(color=MOVER, lw=3.2, ls="-", alpha=0.95),
        "hold": dict(color=GREY, lw=2.0, ls="-", alpha=0.75),
        "nc": dict(color=GREY, lw=1.6, ls=(0, (4, 3)), alpha=0.7),
    }

    fig, ax = plt.subplots(figsize=(12, 6.6))
    for r in rows:
        y0 = r["left_num"]                 # 2025 published rank
        y1 = r["left_num"] - r["move"]     # blended rank (= published - move)
        ax.plot([0, 1], [y0, y1], solid_capstyle="round", **style_kw[r["style"]])
        ax.scatter([0], [y0], s=34, color=style_kw[r["style"]]["color"], zorder=4)
        ax.scatter([1], [y1], s=34, color=style_kw[r["style"]]["color"], zorder=4)
        ax.annotate(
            f'{r["left_num"]}  {r["left_code"]}  {r["left_name"]}',
            (0, y0), textcoords="offset points", xytext=(-10, 0),
            ha="right", va="center", fontsize=12, color="#202124",
        )
        move_txt = "nc" if r["move"] == 0 else f'{r["move"]:+d}'
        rn = " [renamed]" if r["renamed"] else ""
        ax.annotate(
            f'{r["right_code"]}  {r["right_name"]}{rn}  ({move_txt})',
            (1, y1), textcoords="offset points", xytext=(10, 0),
            ha="left", va="center", fontsize=12,
            color=(MOVER if r["style"] == "mover" else "#202124"),
        )
    ax.set_xlim(-0.55, 1.75)
    ax.set_ylim(10.6, 0.4)  # rank 1 at top
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["2025 Published", "2026 Blended"], fontsize=14, fontweight="bold")
    ax.set_yticks([])
    for side in ("left", "right", "top", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def render_entry_expansion_map(
    entries: list[dict[str, Any]],
    figures_dir: Path,
) -> Path:
    """Preprint chart: grouped map of the 20-entry candidate ballot.

    Three column panels — incumbent (LLM*), new (NEW-*), rollup (ROLL-*) —
    with horizontal arrows from each ROLL-* box to the incumbent it rolls into.
    """
    out = figures_dir / "entry_expansion_map.png"

    groups: dict[str, list[dict[str, Any]]] = {"incumbent": [], "new": [], "rollup": []}
    rolled_into: dict[str, str] = {}
    for e in entries:
        g = e.get("group", "")
        if g in groups:
            groups[g].append(e)
        parent = e.get("rolled_into")
        if parent:
            rolled_into[e["entry_id"]] = parent

    n_rows = max(len(v) for v in groups.values()) if any(groups.values()) else 1
    fig, ax = plt.subplots(figsize=(14, max(6, n_rows * 0.8 + 2)))
    ax.set_xlim(0, 3)
    ax.set_ylim(-1, n_rows + 0.5)
    ax.axis("off")

    col_x = {"incumbent": 0.35, "new": 1.5, "rollup": 2.65}
    col_labels = {
        "incumbent": "Incumbent (LLM*)",
        "new": "New Entries (NEW-*)",
        "rollup": "Roll-ups (ROLL-*)",
    }

    entry_pos: dict[str, tuple[float, float]] = {}

    for grp, col_xv in col_x.items():
        ax.text(
            col_xv, -0.6, col_labels[grp],
            ha="center", va="center", fontsize=11, fontweight="bold",
        )
        for i, e in enumerate(groups[grp]):
            eid = e["entry_id"]
            name = e.get("canonical_name", eid)
            y = float(i)
            color = ENTRY_COLORS.get(eid, "#999999")
            rect = mpatches.FancyBboxPatch(
                (col_xv - 0.42, y - 0.22),
                0.84,
                0.44,
                boxstyle="round,pad=0.04",
                facecolor=color,
                edgecolor="white",
                alpha=0.88,
                linewidth=1,
            )
            ax.add_patch(rect)
            ax.text(
                col_xv, y,
                f"{eid}\n{name[:18]}",
                ha="center", va="center", fontsize=7,
                color="white", fontweight="bold",
            )
            entry_pos[eid] = (col_xv, y)

    # Arrows from ROLL-* to their incumbent parent
    for eid, parent_id in rolled_into.items():
        if eid in entry_pos and parent_id in entry_pos:
            sx, sy = entry_pos[eid]
            tx, ty = entry_pos[parent_id]
            ax.annotate(
                "",
                xy=(tx + 0.42, ty),
                xytext=(sx - 0.42, sy),
                arrowprops=dict(
                    arrowstyle="->",
                    color="#555555",
                    lw=1.5,
                    connectionstyle="arc3,rad=0.15",
                ),
            )

    ax.set_title(
        "2026 Entry Expansion Map: 20 Candidate Entries",
        fontsize=13, pad=20,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def render_rarr_robustness(robustness: dict[str, Any], figures_dir: Path) -> Path:
    """Preprint chart: RARR ranking-fidelity bar chart.

    Draws one horizontal bar per model / ensemble from
    ``robustness["ranking_fidelity_spearman_vs_truth"]`` and adds a vertical
    reference line at the ``floor`` value.
    """
    out = figures_dir / "rarr_robustness.png"
    fidelity: dict[str, float] = robustness["ranking_fidelity_spearman_vs_truth"]
    floor_val: float = fidelity["floor"]

    model_keys = [k for k in fidelity if k != "floor"]
    values = [fidelity[k] for k in model_keys]

    incumbent_color = ENTRY_COLORS.get("LLM01", "#2196F3")
    bar_colors = [
        "#E53935" if v < floor_val else incumbent_color
        for v in values
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    y_pos = range(len(model_keys))
    bars = ax.barh(list(y_pos), values, color=bar_colors, alpha=0.85, edgecolor="white")
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(model_keys, fontsize=10)
    ax.set_xlabel("Spearman ρ vs Ground Truth", fontsize=11)
    ax.set_title("RARR Ranking Fidelity: Spearman ρ vs Held-Out Truth", fontsize=12)

    ax.axvline(
        x=floor_val,
        color="#FF6F00",
        linestyle="--",
        linewidth=2,
        label=f"Floor ρ = {floor_val:.3f}",
    )
    ax.legend(fontsize=9)

    x_max = max(list(values) + [floor_val])
    for bar, v in zip(bars, values, strict=False):
        ax.text(
            v + x_max * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{v:.3f}",
            va="center",
            fontsize=9,
        )

    ax.set_xlim(0, x_max * 1.15)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out
