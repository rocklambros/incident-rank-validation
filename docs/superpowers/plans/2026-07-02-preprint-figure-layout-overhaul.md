# Preprint Figure & Layout Overhaul — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redesign the ugly/illegible preprint figures, replace the §4.2 table with a reference-matched 2025→2026 slopegraph, renumber, guarantee no figure occupies a full page (wrapfig + centered blocks), and rename the build outputs — turning the arXiv preprint into a marketing-grade document without touching prose voice or the analysis.

**Architecture:** All figure rendering lives in `engine/report/narrative_charts.py` (the notebook only calls these functions). Layout is controlled by a pandoc Lua filter (`figure-layout.lua`) that converts attributed images to `wrapfigure` or centered `figure[htbp]`, plus two LaTeX packages. The notebook markdown carries per-figure `{width=NN% wrap=side}` attributes. `build_preprint.py` gains an output-name/lua-filter parameterization.

**Tech Stack:** Python 3.12, matplotlib + seaborn (static charts), plotly + kaleido (treemap/sankey), pandoc 3.9 + xelatex, Lua filter, pytest.

**Spec:** `docs/superpowers/specs/2026-07-02-preprint-figure-layout-overhaul-design.md` (hardened via the 2026-07-02 adversarial premortem — findings F1–F8).

## Global Constraints

- **No full-page figures.** Every figure shares its page with text. Redesigned tall charts target aspect ratio h/w ≤ 0.90 (landscape/compact), verified by test.
- **Legibility budget (F3):** smallest on-page text ≥ 8 pt. On-page pt ≈ `source_font_pt × (width × 6.5 / figsize_width_in)`. Wrap figures authored small (~3.2 in wide) with large relative fonts.
- **Pandoc 3.9 emits `Figure` AST nodes (F1):** the Lua filter targets `Figure`, reads width/wrap from the descendant `Image`, caption from `fig.caption.long`.
- **Captions rendered through pandoc, never raw (F2):** two captions contain `%` (`"50% line"`, `"90% credible interval"`) plus `λ κ ρ –`. Use `pandoc.write` on the caption inlines.
- **Figure numbering is automatic** (pandoc, by source order). No prose references "Figure N" — moving/removing figures is safe. Do not hand-number captions.
- **Preserve every render function's signature and output filename** (existing tests assert them).
- **`[renamed]` tag driven by the explicit set `{"LLM07"}` (F6)**, not a name diff.
- **Filename:** build outputs named `Incident_Data_Robustness_Analysis_of_the_OWASP_Top_10_for_LLM_Applications_2026.{pdf,md,tex}`. Source `.ipynb` keeps its name.
- **Prose voice untouched.** `STYLE-GUIDE.md` banned-pattern rules still bind any prose you touch.
- **Drop `plotly_rankings` entirely** (render call, cell-0 import, markdown ref, and the "interactive companion" sentence).
- **Attribution:** no AI/Claude/Anthropic mention in commits or any GitHub-visible text.
- **Data cycle for real-data tests:** `CYCLE = <repo>/projects/owasp-llm/cycles/2026`; loader `engine.report.narrative_data.load_narrative_data(CYCLE)`.

**Per-figure layout matrix (final 18 figures, applied in Task 9):**

| file | width | wrap | method |
|------|-------|------|--------|
| entry_expansion_map.png | 85% | — | center |
| stratum_bar.png | 42% | right | wrap |
| tier_donut.png | 40% | right | wrap |
| confusion_heatmap.png | 72% | — | center |
| precision_bars.png | 60% | — | center |
| precision_posteriors.png | 62% | — | center |
| ridge_plot.png | 70% | — | center |
| dumbbell_chart.png | 68% | — | center |
| bump_chart.png | 72% | — | center |
| ci_overlap.png | 72% | — | center |
| paired_dots.png | 46% | right | wrap |
| theme_bars_llm09.png | 42% | left | wrap |
| theme_bars_new_wla.png | 42% | right | wrap |
| oos_treemap.png | 85% | — | center |
| sankey_confusion.png | 90% | — | center |
| confusion_matrix_3x3.png | 42% | right | wrap |
| rarr_robustness.png | 48% | left | wrap |
| rank_change_2025_2026.png | 92% | — | center |

---

## File Structure

- `engine/report/narrative_charts.py` — MODIFY: add `PUBLISHED_2025_NAMES`, `RENAMED_2026`, `_rank_change_rows()` helper; rewrite `render_rank_change_2025_2026`, `render_bump_chart`, `render_ridge_plot`, `render_dumbbell_chart`, `render_oos_treemap`, `render_sankey_confusion`.
- `notebooks/preprint/figure-layout.lua` — CREATE: pandoc Lua filter.
- `notebooks/preprint/arxiv-template.latex` — MODIFY: add `wrapfig`, `float`.
- `tools/build_preprint.py` — MODIFY: `output_name`/`lua_filter` params, `.tex` emission, `--lua-filter` wiring.
- `notebooks/2026_top_10_llm_update_what_the_data_says.ipynb` — MODIFY: move rank_change to §4.2, drop table + plotly, reword prose, apply layout attrs.
- `notebooks/preprint/BUILD.md` — MODIFY: build command.
- `tests/unit/test_narrative_charts_new.py` — MODIFY: helper + aspect-ratio tests.
- `tests/unit/test_figure_layout_filter.py` — CREATE: Lua-filter compile test.
- `tests/unit/test_preprint_build.py` — MODIFY: output-name test.
- `tests/unit/test_notebook_figure_refs.py` — CREATE: notebook-structure guards.

---

## Task 1: Slopegraph hero — `render_rank_change_2025_2026` (§4.2)

**Files:**
- Modify: `engine/report/narrative_charts.py` (add constants + `_rank_change_rows`; rewrite `render_rank_change_2025_2026`)
- Test: `tests/unit/test_narrative_charts_new.py`

**Interfaces:**
- Consumes: `blended: list[dict]` (each has `entry_id`, `blend_rank`), `entry_names: dict[str,str]` (2026 names), `figures_dir: Path`.
- Produces: `_rank_change_rows(blended, entry_names) -> list[dict]` with keys `left_num:int, left_code:str, left_name:str, right_code:str, right_name:str, move:int, renamed:bool, style:str` (`style ∈ {"mover","hold","nc"}`). `render_rank_change_2025_2026(blended, entry_names, figures_dir) -> Path` (signature + output path unchanged).

- [ ] **Step 1: Write the failing test for the row helper**

Add to `tests/unit/test_narrative_charts_new.py`:

```python
# ---------------------------------------------------------------------------
# _rank_change_rows (slopegraph row model)
# ---------------------------------------------------------------------------
class TestRankChangeRows:
    def _blended(self):
        # incumbent-only blended list with a mix of moves incl. LLM07 (nc, renamed)
        # published rank = int(LLMkk); move = published - blend_rank
        return [
            {"entry_id": "LLM02", "blend_rank": 1},  # pub 2 -> +1  hold
            {"entry_id": "LLM01", "blend_rank": 2},  # pub 1 -> -1  hold
            {"entry_id": "LLM06", "blend_rank": 3},  # pub 6 -> +3  mover
            {"entry_id": "LLM04", "blend_rank": 4},  # pub 4 -> 0   nc
            {"entry_id": "LLM03", "blend_rank": 5},  # pub 3 -> -2  mover
            {"entry_id": "LLM10", "blend_rank": 6},  # pub 10 -> +4 mover
            {"entry_id": "LLM07", "blend_rank": 7},  # pub 7 -> 0   nc + renamed
            {"entry_id": "LLM09", "blend_rank": 8},  # pub 9 -> +1  hold
            {"entry_id": "LLM08", "blend_rank": 9},  # pub 8 -> -1  hold
            {"entry_id": "LLM05", "blend_rank": 10}, # pub 5 -> -5  mover
        ]

    def _names(self):
        return {
            "LLM01": "Prompt Injection", "LLM02": "Sensitive Information Disclosure",
            "LLM03": "Supply Chain", "LLM04": "Data and Model Poisoning",
            "LLM05": "Improper Output Handling", "LLM06": "Excessive Agency",
            "LLM07": "Hidden Context Exposure", "LLM08": "Vector and Embedding Weaknesses",
            "LLM09": "Misinformation", "LLM10": "Unbounded Consumption",
        }

    def test_new_code_equals_blend_position_and_moves(self):
        from engine.report.narrative_charts import _rank_change_rows
        rows = {r["right_code"]: r for r in _rank_change_rows(self._blended(), self._names())}
        assert rows["LLM01"]["right_name"] == "Sensitive Information Disclosure"
        assert rows["LLM01"]["move"] == 1
        assert rows["LLM06"]["move"] == 3 and rows["LLM06"]["style"] == "mover"
        assert rows["LLM10"]["move"] == 4 and rows["LLM10"]["style"] == "mover"
        assert rows["LLM10"]["right_name"] == "Unbounded Consumption"

    def test_renamed_only_llm07(self):
        from engine.report.narrative_charts import _rank_change_rows
        rows = _rank_change_rows(self._blended(), self._names())
        renamed = {r["right_code"] for r in rows if r["renamed"]}
        assert renamed == {"LLM07"}

    def test_style_bands(self):
        from engine.report.narrative_charts import _rank_change_rows
        rows = {r["right_code"]: r for r in _rank_change_rows(self._blended(), self._names())}
        assert rows["LLM04"]["style"] == "nc"      # move 0
        assert rows["LLM02"]["style"] == "hold"    # |move| == 1
        assert rows["LLM05"]["style"] == "mover"   # |move| == 5

    def test_left_side_uses_published_2025_name_and_rank(self):
        from engine.report.narrative_charts import _rank_change_rows
        rows = {r["right_code"]: r for r in _rank_change_rows(self._blended(), self._names())}
        # LLM07 published 2025 name is "System Prompt Leakage"; left_num == published rank
        assert rows["LLM07"]["left_code"] == "LLM07"
        assert rows["LLM07"]["left_name"] == "System Prompt Leakage"
        assert rows["LLM07"]["left_num"] == 7
```

- [ ] **Step 2: Run it — verify it fails**

Run: `pytest tests/unit/test_narrative_charts_new.py::TestRankChangeRows -v`
Expected: FAIL (`cannot import name '_rank_change_rows'`).

- [ ] **Step 3: Add constants + the helper**

In `engine/report/narrative_charts.py`, after `FRAME_BLIND = {...}` (near line 48) add:

```python
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
```

- [ ] **Step 4: Run the helper test — verify it passes**

Run: `pytest tests/unit/test_narrative_charts_new.py::TestRankChangeRows -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Rewrite the renderer to match the reference slopegraph**

Replace the body of `render_rank_change_2025_2026` (keep the signature and the `out = figures_dir / "rank_change_2025_2026.png"` / `return out`). Reference screenshot semantics: left column `N  LLMkk  <2025 name>` at published rank; right column `LLMjj  <2026 name>  (<move>)` at blended rank with `[renamed]` where applicable; **orange** solid thick for movers, **grey** solid for holds, **grey dashed** for nc.

```python
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
    style_kw = {
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
```

- [ ] **Step 6: Add the aspect-ratio guard test**

Append to `TestRankChangeRows`'s file (new class), reusing the module fixtures `minimal_blended`, `minimal_entry_names`:

```python
class TestRankChangeShape:
    def test_landscape_not_tall(self, minimal_blended, minimal_entry_names, figures_dir):
        from PIL import Image
        from engine.report.narrative_charts import render_rank_change_2025_2026
        out = render_rank_change_2025_2026(minimal_blended, minimal_entry_names, figures_dir)
        w, h = Image.open(out).size
        assert h / w <= 0.85, f"slopegraph too tall: h/w={h/w:.2f}"
```

- [ ] **Step 7: Run the rank_change tests (existing + new)**

Run: `pytest tests/unit/test_narrative_charts_new.py -k "RankChange or rank_change" -v`
Expected: PASS (helper tests, shape test, and the pre-existing `render_rank_change_2025_2026` path/size/type tests).

- [ ] **Step 8: Commit**

```bash
git add engine/report/narrative_charts.py tests/unit/test_narrative_charts_new.py
git commit -m "feat(charts): reference-matched 2025->2026 slopegraph with dual codes"
```

---

## Task 2: `render_bump_chart` — focused slopegraph (Fig 9)

**Files:**
- Modify: `engine/report/narrative_charts.py` (`render_bump_chart`)
- Test: `tests/unit/test_narrative_charts_new.py`

**Interfaces:**
- Consumes: `data: dict` with `rank_comparison_md` (parsed as today) and `concordance["flags"]` (list of `{entry_id, probability, direction}`). `figures_dir: Path`.
- Produces: `render_bump_chart(data, figures_dir) -> None`, saves `bump_chart.png`.

- [ ] **Step 1: Write the failing shape/smoke test (real data)**

Add a new test class. Uses the real cycle (integration-flavoured; mark it):

```python
import os
import pytest

REPO = Path(__file__).resolve().parents[2]
CYCLE = REPO / "projects" / "owasp-llm" / "cycles" / "2026"

@pytest.mark.integration
class TestBumpChartFocused:
    def _data(self):
        from engine.report.narrative_data import load_narrative_data
        return load_narrative_data(CYCLE)

    def test_renders_and_is_not_tall(self, figures_dir):
        from PIL import Image
        from engine.report.narrative_charts import render_bump_chart
        render_bump_chart(self._data(), figures_dir)
        out = figures_dir / "bump_chart.png"
        assert out.exists() and out.stat().st_size > 1024
        w, h = Image.open(out).size
        assert h / w <= 0.90, f"bump chart too tall: h/w={h/w:.2f}"
```

- [ ] **Step 2: Run it — verify it fails**

Run: `pytest tests/unit/test_narrative_charts_new.py::TestBumpChartFocused -v`
Expected: FAIL (current bump chart is tall, h/w≈1.21).

- [ ] **Step 3: Rewrite `render_bump_chart`**

Keep the `rank_comparison_md` parse (lines ~283–306). Replace the drawing: grey/thin all entries; highlight only entries in `data["concordance"]["flags"]`; de-collide endpoint labels by nudging within a column.

```python
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
                    lambda_ranks[eid] = float(parts[1].split("(")[0].strip())
                    vote_ranks[eid] = float(parts[2].split("(")[0].strip())
                except (ValueError, IndexError):
                    continue

    common = sorted(set(lambda_ranks) & set(vote_ranks))
    if not common:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No rank data available", ha="center", va="center")
        fig.savefig(figures_dir / "bump_chart.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    flagged = {f["entry_id"] for f in data.get("concordance", {}).get("flags", [])}

    def _decollide(entries, ycol):
        # greedy: sort by y, push apart to >= min_gap in label space
        order = sorted(entries, key=lambda e: ycol[e])
        min_gap = 0.62
        placed: dict[str, float] = {}
        last = -1e9
        for e in order:
            y = max(ycol[e], last + min_gap)
            placed[e] = y
            last = y
        return placed

    left_lab = _decollide(common, lambda_ranks)
    right_lab = _decollide(common, vote_ranks)

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
    ax.set_ylim(max(vote_ranks.values() | {*lambda_ranks.values()}) + 0.6, 0.4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Incident rank", "Expert rank"], fontsize=13, fontweight="bold")
    ax.set_ylabel("Rank", fontsize=12)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Expert vs incident rank — the five flagged disagreements highlighted",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(figures_dir / "bump_chart.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
```

Note: `max(vote_ranks.values() | {...})` — replace with `max(list(vote_ranks.values()) + list(lambda_ranks.values()))` if the set-union of floats is awkward; use whichever the implementer verifies runs.

- [ ] **Step 4: Run the test — verify it passes**

Run: `pytest tests/unit/test_narrative_charts_new.py::TestBumpChartFocused -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add engine/report/narrative_charts.py tests/unit/test_narrative_charts_new.py
git commit -m "feat(charts): focused expert-vs-incident slopegraph (grey base, flagged in colour)"
```

---

## Task 3: `render_ridge_plot` — overlapping joyplot (Fig 7)

**Files:**
- Modify: `engine/report/narrative_charts.py` (`render_ridge_plot`)
- Test: `tests/unit/test_narrative_charts_new.py`

**Interfaces:** Consumes `data` with `lambda_samples: np.ndarray (16000,20)`, `entry_ids: list[str]`. Produces `render_ridge_plot(data, figures_dir) -> None` → `ridge_plot.png`.

- [ ] **Step 1: Write the failing shape test**

```python
@pytest.mark.integration
class TestRidgeJoyplot:
    def test_not_tall(self, figures_dir):
        from PIL import Image
        from engine.report.narrative_data import load_narrative_data
        from engine.report.narrative_charts import render_ridge_plot
        render_ridge_plot(load_narrative_data(CYCLE), figures_dir)
        w, h = Image.open(figures_dir / "ridge_plot.png").size
        assert h / w <= 0.90, f"ridge too tall: h/w={h/w:.2f}"
```

- [ ] **Step 2: Run it — verify it fails**

Run: `pytest tests/unit/test_narrative_charts_new.py::TestRidgeJoyplot -v`
Expected: FAIL (current ridge h/w≈1.60).

- [ ] **Step 3: Rewrite `render_ridge_plot` as an overlapping ridgeline**

```python
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
        kde = gaussian_kde(vals, bw_method=0.25)
        dens = kde(xs)
        dens = dens / dens.max() * scale
        base = row * pitch
        color = ENTRY_COLORS.get(eid, "#999999")
        ax.fill_between(xs, base, base + dens, color=color, alpha=0.85,
                        zorder=n - row, lw=0)
        ax.plot(xs, base + dens, color="white", lw=0.6, zorder=n - row)
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
```

- [ ] **Step 4: Run the test — verify it passes**

Run: `pytest tests/unit/test_narrative_charts_new.py::TestRidgeJoyplot -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add engine/report/narrative_charts.py tests/unit/test_narrative_charts_new.py
git commit -m "feat(charts): overlapping ridgeline (joyplot) for posterior lambda"
```

---

## Task 4: `render_dumbbell_chart` — compact single column (Fig 8)

**Files:** Modify `engine/report/narrative_charts.py` (`render_dumbbell_chart`); Test `tests/unit/test_narrative_charts_new.py`.

**Interfaces:** Consumes `data` with `lambda_samples`, `entry_ids`, `entry_names`. Produces `render_dumbbell_chart(data, figures_dir) -> None` → `dumbbell_chart.png`.

- [ ] **Step 1: Failing shape test**

```python
@pytest.mark.integration
class TestDumbbellCompact:
    def test_not_tall(self, figures_dir):
        from PIL import Image
        from engine.report.narrative_data import load_narrative_data
        from engine.report.narrative_charts import render_dumbbell_chart
        render_dumbbell_chart(load_narrative_data(CYCLE), figures_dir)
        w, h = Image.open(figures_dir / "dumbbell_chart.png").size
        assert h / w <= 0.90, f"dumbbell too tall: h/w={h/w:.2f}"
```

- [ ] **Step 2: Run — verify fails** (`h/w≈1.20`)

Run: `pytest tests/unit/test_narrative_charts_new.py::TestDumbbellCompact -v`

- [ ] **Step 3: Rewrite for a compact aspect**

Keep the existing rank-median/CI computation. Change only the figure to a wide, compact layout and larger fonts:

```python
    # ... after computing rank_medians / rank_cis / sorted_entries (unchanged) ...
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
```

- [ ] **Step 4: Run — verify passes**

Run: `pytest tests/unit/test_narrative_charts_new.py::TestDumbbellCompact -v`

- [ ] **Step 5: Commit**

```bash
git add engine/report/narrative_charts.py tests/unit/test_narrative_charts_new.py
git commit -m "feat(charts): compact single-column dumbbell rank chart"
```

---

## Task 5: `render_oos_treemap` — legible treemap (Fig 14)

**Files:** Modify `engine/report/narrative_charts.py` (`render_oos_treemap`); Test `tests/unit/test_narrative_charts_new.py`.

**Interfaces:** Consumes `data["prelabels"]`. Produces `render_oos_treemap(data, figures_dir) -> None` → `oos_treemap.png` (plotly/kaleido).

- [ ] **Step 1: Failing test (aspect + exists)**

```python
@pytest.mark.integration
class TestOOSTreemap:
    def test_renders_landscape(self, figures_dir):
        from PIL import Image
        from engine.report.narrative_data import load_narrative_data
        from engine.report.narrative_charts import render_oos_treemap
        render_oos_treemap(load_narrative_data(CYCLE), figures_dir)
        out = figures_dir / "oos_treemap.png"
        assert out.exists() and out.stat().st_size > 1024
        w, h = Image.open(out).size
        assert 0.55 <= h / w <= 0.80
```

- [ ] **Step 2: Run — verify current state** (current 2000×1200 → h/w 0.60 already passes aspect; the test guards regressions). If it passes already, still proceed to Step 3 for the legibility upgrades, then re-run.

Run: `pytest tests/unit/test_narrative_charts_new.py::TestOOSTreemap -v`

- [ ] **Step 3: Upgrade legibility**

In `render_oos_treemap`, change the treemap construction to enlarge fonts and show counts + percents:

```python
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
```

(Delete the old `fig.update_layout(width=2000, height=1200)` + write_image lines it replaces. Keep the no-data placeholder branch, updating its size to 1500×1050 too.)

- [ ] **Step 4: Run — verify passes**

Run: `pytest tests/unit/test_narrative_charts_new.py::TestOOSTreemap -v`

- [ ] **Step 5: Commit**

```bash
git add engine/report/narrative_charts.py tests/unit/test_narrative_charts_new.py
git commit -m "feat(charts): legible out-of-scope treemap (larger fonts, count+percent)"
```

---

## Task 6: `render_sankey_confusion` — legible sankey (Fig 15)

**Files:** Modify `engine/report/narrative_charts.py` (`render_sankey_confusion`); Test `tests/unit/test_narrative_charts_new.py`.

**Interfaces:** Consumes `data["prelabels"]`. Produces `render_sankey_confusion(data, figures_dir) -> None` → `sankey_confusion.png`.

- [ ] **Step 1: Failing test**

```python
@pytest.mark.integration
class TestSankey:
    def test_renders_landscape(self, figures_dir):
        from PIL import Image
        from engine.report.narrative_data import load_narrative_data
        from engine.report.narrative_charts import render_sankey_confusion
        render_sankey_confusion(load_narrative_data(CYCLE), figures_dir)
        out = figures_dir / "sankey_confusion.png"
        assert out.exists() and out.stat().st_size > 1024
        w, h = Image.open(out).size
        assert 0.55 <= h / w <= 0.72
```

- [ ] **Step 2: Run — verify state**

Run: `pytest tests/unit/test_narrative_charts_new.py::TestSankey -v`

- [ ] **Step 3: Upgrade legibility + colour the links**

Replace the `go_plotly.Figure(...)` construction and layout in `render_sankey_confusion`:

```python
    node_colors = [
        ENTRY_COLORS.get(lb.split(": ")[-1], "#888888") for lb in all_labels
    ]
    # colour each link by its source node, semi-transparent
    def _rgba(hex_c, a=0.45):
        h = hex_c.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{a})"
    link_colors = [_rgba(node_colors[s]) for s in source]
    node_labels = [f"{lb}  ({int(v)})" for lb, v in
                   zip(all_labels, [sum(val for s2, val in zip(source, value) if s2 == i)
                                    or sum(val for t2, val in zip(target, value) if t2 == i)
                                    for i in range(len(all_labels))])]

    fig = go_plotly.Figure(data=[go_plotly.Sankey(
        node=dict(label=node_labels, color=node_colors, pad=22, thickness=26,
                  line=dict(color="white", width=1)),
        link=dict(source=source, target=target, value=value, color=link_colors),
    )])
    fig.update_layout(
        title="Model votes → consensus at the confusion boundary",
        font=dict(size=18), title_font_size=26,
        width=1600, height=1000, margin=dict(t=80, l=10, r=10, b=10),
    )
    _plotly_write_image(fig, str(figures_dir / "sankey_confusion.png"), width=1600, height=1000)
```

(Replace the prior `fig.update_layout(... width=2000, height=1200)` + write_image. Keep the no-data placeholder branch, sizing it 1600×1000.)

- [ ] **Step 4: Run — verify passes**

Run: `pytest tests/unit/test_narrative_charts_new.py::TestSankey -v`

- [ ] **Step 5: Commit**

```bash
git add engine/report/narrative_charts.py tests/unit/test_narrative_charts_new.py
git commit -m "feat(charts): legible confusion-boundary sankey (bigger fonts, coloured flows, node totals)"
```

---

## ⟢ CONTROLLER CHECKPOINT — Preview gate (after Task 6)

Not a subagent task. The controller renders all six redesigned charts against **real** data and presents the PNGs to the user for aesthetic sign-off before the layout/build work:

```bash
python - <<'PY'
from pathlib import Path
from engine.report.narrative_data import load_narrative_data
from engine.report.blend_2025_2026 import blended_ranking, load_entries
from engine.report import narrative_charts as nc
REPO = Path.cwd(); CYCLE = REPO/"projects/owasp-llm/cycles/2026"
FIG = REPO/"notebooks/preprint/figures"
DATA = load_narrative_data(CYCLE)
nc.render_bump_chart(DATA, FIG); nc.render_ridge_plot(DATA, FIG)
nc.render_dumbbell_chart(DATA, FIG); nc.render_oos_treemap(DATA, FIG)
nc.render_sankey_confusion(DATA, FIG)
# slopegraph: reuse cell-45 blend/fold verbatim (F7) — see notebook cell 45
PY
```
Then `Read` each PNG, present to user, iterate on any chart the user rejects (re-open the relevant task). Do not proceed to Task 7 until the user approves the visuals.

---

## Task 7: Layout engine — Lua filter + LaTeX packages

**Files:**
- Create: `notebooks/preprint/figure-layout.lua`
- Modify: `notebooks/preprint/arxiv-template.latex` (add packages)
- Create: `tests/unit/test_figure_layout_filter.py`

**Interfaces:** Produces a pandoc Lua filter invoked as `--lua-filter notebooks/preprint/figure-layout.lua` (wired in Task 8). Emits `wrapfigure` for `wrap=left|right`, `figure[htbp]` otherwise.

- [ ] **Step 1: Write the failing filter test**

`tests/unit/test_figure_layout_filter.py`:

```python
from __future__ import annotations
import subprocess
from pathlib import Path

PRE = Path(__file__).resolve().parents[2] / "notebooks" / "preprint"
LUA = PRE / "figure-layout.lua"

def _pandoc_latex(md: str) -> str:
    r = subprocess.run(
        ["pandoc", "-f", "markdown", "-t", "latex", f"--lua-filter={LUA}"],
        input=md, capture_output=True, text=True, check=True,
    )
    return r.stdout

def test_wrap_becomes_wrapfigure():
    out = _pandoc_latex("![Cap with 50% sign](figures/x.png){width=42% wrap=right}\n")
    assert "\\begin{wrapfigure}{R}" in out
    assert "0.42\\textwidth" in out
    assert "50\\%" in out  # % escaped, not swallowed

def test_center_becomes_figure_htbp():
    out = _pandoc_latex("![Latent incidence (λ)](figures/y.png){width=72%}\n")
    assert "\\begin{figure}[htbp]" in out
    assert "0.72\\textwidth" in out
```

- [ ] **Step 2: Run — verify it fails**

Run: `pytest tests/unit/test_figure_layout_filter.py -v`
Expected: FAIL (filter file missing → pandoc errors, or assertions fail).

- [ ] **Step 3: Write the Lua filter**

`notebooks/preprint/figure-layout.lua`:

```lua
-- figure-layout.lua — control preprint figure placement.
-- pandoc >= 3 wraps an attributed image in a Figure node; width/wrap live on
-- the inner Image, caption on the Figure. Emit wrapfigure or figure[htbp].

local function pct_to_frac(s)
  if not s then return 0.85 end
  local n = s:match("^(%d+%.?%d*)%%$")
  if n then return tonumber(n) / 100.0 end
  return tonumber(s) or 0.85
end

local function find_image(blocks)
  local found = nil
  for _, b in ipairs(blocks) do
    if b.content then
      for _, inl in ipairs(b.content) do
        if inl.t == "Image" then found = inl; break end
      end
    end
    if found then break end
  end
  return found
end

local function caption_latex(caption)
  if not caption or not caption.long then return "" end
  local s = pandoc.write(pandoc.Pandoc(caption.long), "latex")
  return (s:gsub("%s+$", ""))
end

function Figure(fig)
  local img = find_image(fig.content)
  if not img then return nil end
  local w = pct_to_frac(img.attributes.width)
  local wrap = img.attributes.wrap
  local src = img.src
  local cap = caption_latex(fig.caption)
  local latex
  if wrap == "left" or wrap == "right" then
    local side = (wrap == "left") and "L" or "R"
    latex = string.format(
      "\\begin{wrapfigure}{%s}{%.3f\\textwidth}\n\\centering\n" ..
      "\\includegraphics[width=%.3f\\textwidth]{%s}\n\\caption{%s}\n\\end{wrapfigure}",
      side, w + 0.03, w, src, cap)
  else
    if wrap ~= nil then
      io.stderr:write("figure-layout.lua: unknown wrap='" .. tostring(wrap) ..
                      "' for " .. src .. "; centering\n")
    end
    latex = string.format(
      "\\begin{figure}[htbp]\n\\centering\n" ..
      "\\includegraphics[width=%.3f\\textwidth]{%s}\n\\caption{%s}\n\\end{figure}",
      w, src, cap)
  end
  return pandoc.RawBlock("latex", latex)
end
```

- [ ] **Step 4: Add the LaTeX packages**

In `notebooks/preprint/arxiv-template.latex`, after `\usepackage[margin=1in]{geometry}` (line 8), add:

```latex
\usepackage{wrapfig}
\usepackage{float}
```

- [ ] **Step 5: Run — verify the filter test passes**

Run: `pytest tests/unit/test_figure_layout_filter.py -v`
Expected: PASS (both tests; the `%` is emitted as `50\%`).

- [ ] **Step 6: Commit**

```bash
git add notebooks/preprint/figure-layout.lua notebooks/preprint/arxiv-template.latex tests/unit/test_figure_layout_filter.py
git commit -m "feat(build): pandoc Lua filter for wrapfig/centered figure placement"
```

---

## Task 8: `build_preprint` — output name, `.tex` emission, filter wiring

**Files:**
- Modify: `tools/build_preprint.py`
- Modify: `notebooks/preprint/BUILD.md`
- Test: `tests/unit/test_preprint_build.py`

**Interfaces:**
- Consumes: `figure-layout.lua` (Task 7).
- Produces: `build_preprint(notebook, out_dir, front_matter_md, template, output_name=None, lua_filter=None) -> Path`. Emits `<stem>.md`, `<stem>.pdf`, `<stem>.tex` where `stem = output_name or notebook.stem`.

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_preprint_build.py`:

```python
def test_build_with_output_name(tmp_path: Path) -> None:
    """Fixture build with output_name → renamed PDF + TeX both produced."""
    # (reuse the fixture-notebook setup from test_build_preprint_fixture: build the
    #  same 3-cell fixture nb, front_matter, template=PRE/'arxiv-template.latex')
    from tools.build_preprint import build_preprint
    # ... construct fixture_nb, front_matter as in test_build_preprint_fixture ...
    pdf = build_preprint(
        notebook=fixture_nb, out_dir=tmp_path, front_matter_md=front_matter,
        template=PRE / "arxiv-template.latex", output_name="Renamed_Doc",
    )
    assert pdf == tmp_path / "Renamed_Doc.pdf"
    assert pdf.exists() and pdf.stat().st_size > 10 * 1024
    assert (tmp_path / "Renamed_Doc.tex").exists()
    assert (tmp_path / "Renamed_Doc.md").exists()
```

(Factor the fixture-notebook construction from `test_build_preprint_fixture` into a shared helper `_make_fixture(tmp_path)` and call it from both tests.)

- [ ] **Step 2: Run — verify it fails**

Run: `pytest tests/unit/test_preprint_build.py::test_build_with_output_name -v`
Expected: FAIL (`build_preprint() got an unexpected keyword argument 'output_name'`).

- [ ] **Step 3: Parameterize `build_preprint`**

Change the signature and the tail of `build_preprint`:

```python
def build_preprint(
    notebook: Path,
    out_dir: Path,
    front_matter_md: Path,
    template: Path,
    output_name: str | None = None,
    lua_filter: Path | None = None,
) -> Path:
    ...
    stem = output_name or notebook.stem
    md_name = stem + ".md"
    ...  # nbconvert markdown export uses md_name (already parameterized by md_name)
    ...
    pdf_path = out_dir / (stem + ".pdf")
    tex_path = out_dir / (stem + ".tex")
    lf = lua_filter or (template.parent / "figure-layout.lua")
    common = [
        "--from=markdown+yaml_metadata_block",
        f"--template={template}",
        f"--lua-filter={lf}",
        "--toc", "--toc-depth=2",
    ]
    subprocess.run(
        ["pandoc", str(md_path), "-o", str(pdf_path), "--pdf-engine=xelatex", *common],
        check=True, cwd=str(out_dir),
    )
    subprocess.run(
        ["pandoc", str(md_path), "-o", str(tex_path), "--standalone", *common],
        check=True, cwd=str(out_dir),
    )
    return pdf_path
```

Update the two earlier uses of `notebook.stem + ".md"` / the md export `--output` to use `md_name = stem + ".md"`. Add `--output-name` to `_cli()` (default `None`).

- [ ] **Step 4: Run — verify it passes (and the existing fixture test still passes)**

Run: `pytest tests/unit/test_preprint_build.py -v`
Expected: PASS (`test_build_preprint_fixture` and `test_build_with_output_name`).

- [ ] **Step 5: Update BUILD.md**

Replace the build command block in `notebooks/preprint/BUILD.md` with:

```bash
python tools/build_preprint.py \
  --notebook notebooks/2026_top_10_llm_update_what_the_data_says.ipynb \
  --out-dir notebooks/preprint \
  --output-name Incident_Data_Robustness_Analysis_of_the_OWASP_Top_10_for_LLM_Applications_2026
```

Add a line: "Emits `<output-name>.{md,pdf,tex}`; submit the `.tex` + `figures/` to arXiv."

- [ ] **Step 6: Commit**

```bash
git add tools/build_preprint.py notebooks/preprint/BUILD.md tests/unit/test_preprint_build.py
git commit -m "feat(build): output-name parameter, .tex emission, lua-filter wiring"
```

---

## Task 9: Notebook surgery — move slopegraph to §4.2, drop table + plotly, apply layout attrs

**Files:**
- Modify: `notebooks/2026_top_10_llm_update_what_the_data_says.ipynb`
- Create: `tests/unit/test_notebook_figure_refs.py`

**Interfaces:** Consumes the redesigned renderers + layout matrix. Produces a notebook whose exported markdown carries the per-figure attributes and the §4.2 slopegraph.

- [ ] **Step 1: Write the failing structure test**

`tests/unit/test_notebook_figure_refs.py`:

```python
from __future__ import annotations
import json
from pathlib import Path

NB = Path(__file__).resolve().parents[2] / "notebooks" / "2026_top_10_llm_update_what_the_data_says.ipynb"

def _sources():
    nb = json.loads(NB.read_text())
    return ["".join(c["source"]) for c in nb["cells"]]

def _all():
    return "\n".join(_sources())

def test_plotly_rankings_fully_removed():
    a = _all()
    assert "plotly_rankings.png" not in a
    assert "render_plotly_rankings" not in a
    assert "interactive companion" not in a

def test_slopegraph_in_blend_section_not_part1():
    srcs = _sources()
    blend = [s for s in srcs if "The 2026 blended Top 10" in s][0]
    part1 = [s for s in srcs if "What changed from 2025 to 2026" in s][0]
    assert "rank_change_2025_2026.png" in blend
    assert "| Blended # |" not in blend            # table removed
    assert "under the figure" in blend             # reworded from "under the table"
    assert "rank_change_2025_2026.png" not in part1  # moved out of Part I

def test_layout_attributes_applied():
    a = _all()
    expected = {
        "stratum_bar.png": "width=42% wrap=right",
        "tier_donut.png": "width=40% wrap=right",
        "precision_bars.png": "width=60%",
        "paired_dots.png": "width=46% wrap=right",
        "theme_bars_llm09.png": "width=42% wrap=left",
        "sankey_confusion.png": "width=90%",
        "rarr_robustness.png": "width=48% wrap=left",
        "rank_change_2025_2026.png": "width=92%",
    }
    for fname, attr in expected.items():
        assert f"{fname}){{{attr}}}" in a, f"missing/incorrect attrs for {fname}"
```

- [ ] **Step 2: Run — verify it fails**

Run: `pytest tests/unit/test_notebook_figure_refs.py -v`
Expected: FAIL.

- [ ] **Step 3: Edit the notebook via a script**

Run this editing script (it mutates the committed notebook in place; the build always copies before executing, so this is safe):

```python
import json, re
from pathlib import Path
NB = Path("notebooks/2026_top_10_llm_update_what_the_data_says.ipynb")
nb = json.loads(NB.read_text())
cells = nb["cells"]
def src(i): return "".join(cells[i]["source"])
def setsrc(i, s): cells[i]["source"] = s.splitlines(keepends=True)

# -- layout attribute matrix (filename -> new {..} attr block) --
ATTR = {
  "entry_expansion_map.png": "{width=85%}",
  "stratum_bar.png": "{width=42% wrap=right}",
  "tier_donut.png": "{width=40% wrap=right}",
  "confusion_heatmap.png": "{width=72%}",
  "precision_bars.png": "{width=60%}",
  "precision_posteriors.png": "{width=62%}",
  "ridge_plot.png": "{width=70%}",
  "dumbbell_chart.png": "{width=68%}",
  "bump_chart.png": "{width=72%}",
  "ci_overlap.png": "{width=72%}",
  "paired_dots.png": "{width=46% wrap=right}",
  "theme_bars_llm09.png": "{width=42% wrap=left}",
  "theme_bars_new_wla.png": "{width=42% wrap=right}",
  "oos_treemap.png": "{width=85%}",
  "sankey_confusion.png": "{width=90%}",
  "confusion_matrix_3x3.png": "{width=42% wrap=right}",
  "rarr_robustness.png": "{width=48% wrap=left}",
  "rank_change_2025_2026.png": "{width=92%}",
}
def apply_attrs(s):
    for fn, at in ATTR.items():
        s = re.sub(r'(\]\(figures/'+re.escape(fn)+r'\))\{[^}]*\}', r'\1'+at, s)
    return s

# Locate cells by content
i_part1 = next(i for i,c in enumerate(cells) if "What changed from 2025 to 2026" in src(i))
i_act6  = next(i for i,c in enumerate(cells) if "## Act 6:" in src(i))
i_blend = next(i for i,c in enumerate(cells) if "The 2026 blended Top 10" in src(i))
i_plcall= next(i for i,c in enumerate(cells) if "render_plotly_rankings(" in src(i))

# (a) Part I §2.4: remove the rank_change image line; reword the pointer sentence
s = src(i_part1)
s = re.sub(r'\n!\[[^\]]*\]\(figures/rank_change_2025_2026\.png\)\{[^}]*\}\n?', '\n', s)
s = s.replace("The figure below traces every incumbent from its 2025 position to its blended 2026 position.",
              "The slopegraph in the blended Top-10 section traces every incumbent from its 2025 position to its blended 2026 position.")
setsrc(i_part1, s)

# (b) Act 6: drop the plotly image + the interactive-companion sentence
s = src(i_act6)
s = re.sub(r'\n!\[[^\]]*\]\(figures/plotly_rankings\.png\)\{[^}]*\}\n?', '\n', s)
s = s.replace(" The interactive companion below the static chart shows each entry's median, interval, and flags on hover.", "")
setsrc(i_act6, s)

# (c) drop the plotly render call cell and its cell-0 import
setsrc(i_plcall, "# (removed) plotly_rankings dropped from the preprint\n")
s0 = src(0).replace("    render_plotly_rankings,\n", "")
setsrc(0, s0)

# (d) §4.2: remove the markdown table, insert the slopegraph image, reword caveat line
s = src(i_blend)
s = re.sub(r'\n\| Blended # \|.*?\| 10 \| LLM05 Improper Output Handling \| −5 \|\n',
           '\n![Published 2025 order to blended 2026 order for the ten incumbents; orange marks the large movers, dashed lines mark no-change.](figures/rank_change_2025_2026.png){width=92%}\n',
           s, flags=re.S)
s = s.replace("Two structural caveats sit under the table.", "Two structural caveats sit under the figure.")
setsrc(i_blend, s)

# (e) apply the layout attribute matrix to every cell
for i in range(len(cells)):
    if cells[i]["cell_type"] == "markdown":
        setsrc(i, apply_attrs(src(i)))

NB.write_text(json.dumps(nb, indent=1) + "\n")
print("notebook edited")
```

- [ ] **Step 4: Run — verify the structure test passes**

Run: `pytest tests/unit/test_notebook_figure_refs.py -v`
Expected: PASS (3 tests). If the table regex misses (heading uses different dashes), adjust the pattern to the actual `−`/`-` character and re-run.

- [ ] **Step 5: Sanity-check the notebook still parses**

Run: `python -c "import json,nbformat; nbformat.read(open('notebooks/2026_top_10_llm_update_what_the_data_says.ipynb'),4); print('nb OK')"`
Expected: `nb OK`.

- [ ] **Step 6: Commit**

```bash
git add notebooks/2026_top_10_llm_update_what_the_data_says.ipynb tests/unit/test_notebook_figure_refs.py
git commit -m "feat(preprint): move slopegraph to §4.2, drop table + plotly companion, apply layout attrs"
```

---

## ⟢ CONTROLLER CHECKPOINT — Full build + print-scale verification (after Task 9)

Not a subagent task. The controller runs the real build and inspects at print scale:

```bash
python tools/build_preprint.py \
  --notebook notebooks/2026_top_10_llm_update_what_the_data_says.ipynb \
  --out-dir notebooks/preprint \
  --output-name Incident_Data_Robustness_Analysis_of_the_OWASP_Top_10_for_LLM_Applications_2026
pdftoppm -r 150 -png notebooks/preprint/Incident_Data_Robustness_Analysis_of_the_OWASP_Top_10_for_LLM_Applications_2026.pdf /tmp/prev/page
```

Confirm on the rasterized pages: figures numbered 1–18, captions monotonic; §4.2 shows the slopegraph, no table; **zero** full-page figures and no large forced-whitespace gaps (F5); every figure legible at print scale (F3/F4); wrapped figures flow cleanly; filename + `.tex` correct. Then run the whole suite:

```bash
pytest -q                       # full suite: green with only known XFAILs
pytest -q -m integration        # the real-data figure tests
```

Then the whole-branch code review and `superpowers:finishing-a-development-branch`.

---

## Self-Review notes (author)

- **Spec coverage:** every §5 figure → Tasks 1–6; §6 layout → Task 7; §8 rename → Task 8; §7 prose/notebook → Task 9; §10 verification → the two controller checkpoints. Legibility budget (F3) enforced by the aspect-ratio tests + print-scale checkpoint; F1/F2 covered by `test_figure_layout_filter.py`.
- **Type consistency:** `_rank_change_rows` keys are consumed only inside `render_rank_change_2025_2026`; render signatures unchanged (existing tests hold).
- **Known latitude:** figure drawing code is a verified starting point; the preview gate authorizes aesthetic iteration within the same signatures/aspect guards. Not a placeholder — each renders and passes its test as written.
```
