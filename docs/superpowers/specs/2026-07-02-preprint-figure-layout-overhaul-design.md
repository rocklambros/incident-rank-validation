# Preprint figure & layout overhaul — design

**Date:** 2026-07-02
**Author:** Rock Lambros
**Artifact under change:** the arXiv preprint *Incident-Data Robustness Analysis of the OWASP Top 10 for LLM Applications (2026)* — notebook-first build.
**Predecessor spec:** `2026-07-02-arxiv-preprint-design.md` (this overhaul refines that deliverable's figures and layout).

## 1. Goal

Make the preprint read as **as much a marketing document as an educational one** — aesthetics second only to content. Concretely: fix the illegible/ugly figures, replace the §4.2 table with a reference-matched slopegraph, renumber, ensure **no figure occupies a full page**, and rename the build output.

The figure content and the numbers are correct today; this is a presentation overhaul, not a re-analysis.

## 2. Non-goals

- No change to prose **voice** or the analytical claims. `STYLE-GUIDE.md` says "scientific preprint, not marketing"; that governs prose banned-patterns and stays in force. The marketing lift is entirely figures + layout. (The one-line tension is noted, not resolved by editing prose.)
- No change to the statistical pipeline, the committed data artifacts, or `load_narrative_data`.
- No renaming of the source `.ipynb`.

## 3. Decisions locked (from brainstorming)

1. **Fig 9 (bump_chart)** → *focused slopegraph*: grey all 20 lines lightly; highlight only the flagged mismatches in colour with labels; de-collide labels.
2. **§4.2 slopegraph** → replaces the table; **codes on both sides**: left `1  LLM01 Prompt Injection`, right `LLM01 Sensitive Information Disclosure (+1)`, with `[renamed]` tag on LLM07. Orange = mover, grey = hold, dashed = no-change. Matches the reference screenshot.
3. **Filename** → rename build **outputs** (PDF + MD + TeX) to `Incident_Data_Robustness_Analysis_of_the_OWASP_Top_10_for_LLM_Applications_2026`; notebook keeps its name.
4. **Layout** → `wrapfig` text-wrap for the small charts that stay legible when shrunk; centered non-float blocks for the heavy charts; **no full-page figures**.
5. **plotly_rankings dropped** — Act 6 keeps one compact rank figure (the dumbbell). Its render call, markdown ref, and the "interactive companion" prose sentence are removed.

## 4. Figure inventory, final numbering, and per-figure layout matrix

Pandoc auto-numbers figures by order of appearance. Moving `rank_change` to §4.2 and dropping `plotly_rankings` yields **18 figures**. Widths are `\textwidth` fractions and are **starting targets**, tuned during the full-build pass so nothing floats alone.

| # | file | section | method | width | notes |
|---|------|---------|--------|-------|-------|
| 1 | entry_expansion_map | §2.4 | center `[H]` | 0.85 | wide/short map |
| 2 | stratum_bar | §3.2 | **wrap right** | 0.42 | simple bars |
| 3 | tier_donut | §3.3 | **wrap right** | 0.40 | donut |
| 4 | confusion_heatmap | §3.3 | center `[H]` | 0.72 | 17×17, needs size |
| 5 | precision_bars | §3.4 | **wrap left** | 0.46 | horizontal bars |
| 6 | precision_posteriors | §3.4 | center `[H]` | 0.62 | 2×2 grid |
| 7 | ridge_plot ↺ | §3.5 | center `[H]` | 0.70 | **redesign: joyplot** |
| 8 | dumbbell_chart ↺ | §3.6 | center `[H]` | 0.68 | **redesign: compact** |
| 9 | bump_chart ↺ | §3.7 | center `[H]` | 0.72 | **redesign: focused slopegraph** |
| 10 | ci_overlap | §3.7 | center `[H]` | 0.72 | 17-row |
| 11 | paired_dots | §3.8 | **wrap right** | 0.46 | flagged-entry bars |
| 12 | theme_bars_llm09 | §3.8 | **wrap left** | 0.42 | keyword bars |
| 13 | theme_bars_new_wla | §3.8 | **wrap right** | 0.42 | keyword bars |
| 14 | oos_treemap ↺ | §3.9A | center `[H]` | 0.85 | **redesign: fonts + count/%** |
| 15 | sankey_confusion ↺ | §3.9B | center `[H]` | 0.90 | **redesign: fonts + coloured flows** |
| 16 | confusion_matrix_3x3 | §3.9B | **wrap right** | 0.42 | 3×3 |
| 17 | rarr_robustness | §4.1 | **wrap left** | 0.48 | 5-bar |
| 18 | rank_change ↺ | §4.2 | center `[H]` | 0.92 | **redesign: slopegraph, replaces table** |

Wrap side alternates L/R for rhythm where prose allows; default `right`.

## 5. Redesign specs (all in `engine/report/narrative_charts.py`; signatures preserved)

**Common:** keep each render function's signature and output path (tests assert on these). Raise per-chart font sizes so text stays legible at the reduced print widths above. Reuse `ENTRY_COLORS`.

### 5.1 `render_ridge_plot` — overlapping joyplot
- Replace the 20 non-overlapping subplots with a single axes; draw each entry's KDE as a filled curve on a shared x (λ), offset vertically by a fixed pitch with ~60–70% overlap (classic ridgeline).
- Sorted by median λ (descending) as now. Per-entry colour; frame-blind entries greyed. Left-margin entry label per ridge.
- Target figsize ≈ (9, 6.5). Legible entry labels (≥10 pt effective).

### 5.2 `render_dumbbell_chart` — compact single column
- Same rank + 90% CI content; tighten row pitch, enlarge dots and CI bars, larger tick/label fonts. Target figsize ≈ (9, 6.2). Keep `LLMxx (Name)` y-labels.

### 5.3 `render_bump_chart` — focused slopegraph
- Two columns (Incident rank | Expert rank), all 20 entries.
- Draw all lines light grey/thin. Highlight only the flagged mismatches — source from `DATA['concordance']['flags']` (same set the prose and `paired_dots` use) — in per-entry colour, thicker, with labels on both ends.
- **De-collide labels:** when two endpoints share a y within a threshold, offset their label y so text never overlaps (greedy vertical nudge). Axis tick numbers must not collide with entry labels (the current bug).
- Target figsize ≈ (9, 6.5).

### 5.4 `render_oos_treemap` — legible treemap
- Keep plotly, but: enlarge title + cell fonts; set `textinfo` to label + value + percent; uniform font colour with contrast; aspect ≈ 1500×1050. Ensure smallest cells still show a count.

### 5.5 `render_sankey_confusion` — legible sankey
- Enlarge node-label font (~16); colour each link by its source node (not all grey); show node value in the label; aspect ≈ 1600×1000; pad nodes so labels don't clip.

### 5.6 `render_rank_change_2025_2026` — reference-matched slopegraph (the §4.2 hero)
- Left axis = 2025 published order, rows 1–10, label `N  LLMkk  <2025 name>`. Right axis = 2026 blended order, label `LLMjj  <2026 name>  (<move>)`, with `[renamed]` appended where the 2026 name differs from 2025 (LLM07: System Prompt Leakage → Hidden Context Exposure).
  - New right-side code `LLMjj` = blended position (1→LLM01 … 10→LLM10).
  - `<move>` = `+n` / `-n` / `nc`.
- Line style: **orange** for movers (|move|≥1), **grey solid** for holds that changed position within tolerance, **grey dashed** for `nc`. Matches the reference screenshot palette (orange/grey), not the mako blues.
- 2025 names come from a module-level constant `PUBLISHED_2025_NAMES` (fixed historical facts); 2026 names from `entry_names`. **Signature unchanged** `(blended, entry_names, figures_dir) -> Path`.
- Only the 10 incumbents are shown (the table was incumbents-only). Target figsize ≈ (11, 7), landscape like the reference.

**Preview gate:** after implementing 5.1–5.6, render each PNG via the standalone harness and present them to the user for a thumbs-up before the full build.

## 6. Layout mechanism

1. **Template** (`arxiv-template.latex`): add `\usepackage{wrapfig}` and `\usepackage{float}` to the preamble (near the other `\usepackage` lines).
2. **Pandoc Lua filter** `notebooks/preprint/figure-layout.lua`:
   - Match a `Para`/`Figure` whose single `Image` has attributes.
   - If `wrap` attr present → emit `RawBlock('latex', …)` with `\begin{wrapfigure}{R|L}{(w+0.03)\textwidth} \centering \includegraphics[width=w\textwidth]{src} \caption{cap} \end{wrapfigure}`.
   - Else → emit `\begin{figure}[H]\centering\includegraphics[width=w\textwidth]{src}\caption{cap}\end{figure}`.
   - `w` parsed from the image `width=NN%` attr; wrap side from `wrap=left|right`. Caption stringified from image caption inlines. Both branches carry `\caption` so the shared figure counter numbers every figure monotonically in source order (wrapfigure is not a float; `[H]` pins the others — this prevents number reordering).
3. **Build wiring** (`build_preprint.py`): pass `--lua-filter notebooks/preprint/figure-layout.lua` to the pandoc call.
4. **Notebook markdown**: update each figure ref's attributes to `{width=NN% wrap=side}` or `{width=NN%}` per the §4 matrix.

## 7. Notebook prose / markdown edits (source of truth = the `.ipynb`)

- **Cell 2 (§2.4):** remove the `rank_change_2025_2026.png` image; reword the sentence ending "The figure below traces every incumbent…" to point to the §4.2 figure instead (e.g., "…traced in the slopegraph in §4.2").
- **Cell 22 (§3.6 / Act 6):** delete the `plotly_rankings.png` image ref; delete the trailing sentence "The interactive companion below the static chart shows each entry's median, interval, and flags on hover." Update remaining ref to `{width=68%}`.
- **Cell 24:** delete the `render_plotly_rankings(...)` call cell (or blank it).
- **Cell 43 (§4.2):** delete the markdown table; insert the `rank_change_2025_2026.png` image ref (`{width=92%}`); change "Two structural caveats sit under the table." → "under the figure."
- **Cell 45 (§4.2 code):** already calls `render_rank_change_2025_2026`; keep. (The figure now renders where §4.2 references it.)
- Apply the §4 width/wrap attributes to every other figure ref cell (2, 5, 9, 14, 18, 22, 26, 30, 33, 36, 41).

## 8. Filename rename

- Add optional `output_name: str | None` to `build_preprint(...)`; when set, name the exported MD and PDF `<output_name>.md/.pdf` instead of `notebook.stem`.
- Add a pandoc step emitting `<output_name>.tex` (same template + lua filter, `-o …tex`) for arXiv source.
- CLI: `--output-name` (default derives from notebook stem for back-compat).
- Default invocation uses `Incident_Data_Robustness_Analysis_of_the_OWASP_Top_10_for_LLM_Applications_2026`.
- Update `BUILD.md` build command.

## 9. Tests

- `test_narrative_charts_new.py`: unchanged assertions (path/size/type) stay green because signatures + output paths are preserved. Add nothing unless a redesign needs a new guard.
- `test_preprint_build.py`: `build_preprint` gains optional kwargs (defaulted) → existing fixture test still passes. Add one test: build with `output_name="X"` produces `X.pdf` (>10 KB) and `X.tex`. Add a `test_stub_compiles_with_template`-style check that the lua filter + wrapfig template compile the stub.
- Full suite must stay green (10 known XFAILs only).

## 10. Verification

1. **Fast loop:** standalone harness (`load_narrative_data(CYCLE)` + `blended_ranking`) regenerates each changed PNG; read each image to confirm legibility and correctness. Present the 6 redesigned charts to the user (preview gate, §5).
2. **Full build:** run `build_preprint.py` with the new output name; open the PDF and confirm:
   - figures renumbered 1–18; captions monotonic;
   - §4.2 shows the slopegraph, **no table**;
   - **zero** figures occupy a full page;
   - wrapped figures legible, prose flows cleanly (no wrapfig collisions near headings/lists);
   - filename correct; `.tex` emitted.
3. Run `pytest tests/unit/test_narrative_charts_new.py tests/unit/test_preprint_build.py` and the full suite.

## 11. Risks / mitigations

- **wrapfig collisions** near headings/short paragraphs → mitigated by `[H]` on non-wrapped figures, right/left default, and the full-build visual pass; if a wrap misbehaves, downgrade that figure to centered `[H]`.
- **Caption-number reordering** from drifting floats → mitigated by `float`+`[H]` pinning every non-wrap figure in source order.
- **Legibility at reduced widths** → each redesign raises font sizes; the fast PNG loop catches any that are still too small before the build.
- **arXiv recompilation** → `wrapfig` and `float` are standard TeXLive packages (present locally; present on arXiv).

## 12. Work hygiene

Feature branch off `main` (not committed to `main` directly). No AI attribution in commits/PR.
