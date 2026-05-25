# Residual Risks + Narrative Report Design

**Goal:** Close two premortem residual risks (kappa CI method field, ai-harm precision
disclosure), update the notebook to reflect the changes, then generate a standalone
narrative report for redistribution to non-Jupyter audiences.

**Three phases, in dependency order:**

1. Risk 2: `ci_method` field in concordance engine + regenerate artifact
2. Risk 1: `F-aiharm-precision` threat in register + notebook Act 10 update
3. Narrative report generator: standalone markdown + PNG export of the notebook story

---

## Phase 1: `ci_method` Field (Risk 2)

### Problem

`concordance.json` documents the kappa point estimate and CI bounds but not *how* the
CI was computed. The CI is posterior-propagated: for each of 16,000 draws, kappa is
evaluated on the paired (lambda sample, vote bootstrap sample), and the 2.5th/97.5th
percentiles of those kappa draws form the CI. This is not an analytical normal
approximation. Without the `ci_method` field, a reader might assume it is.

### Changes

**`engine/decide/concordance.py`:**
- Add field to `ConcordanceResult` dataclass:
  ```python
  ci_method: str = "posterior_propagated_percentile"
  ```
- No changes to `compute_concordance()` — the field is metadata about the existing
  computation, not a new computation.

**`engine/cli/pipeline_executor.py`:**
- Add `"ci_method": concordance.ci_method` to the `conc_dict` serialization block
  (around line 349-367).

**`projects/owasp-llm/cycles/2026/results/concordance.json`:**
- Regenerate by re-running `decide-real` or by patching the JSON directly. The latter
  is simpler since no data has changed — only metadata is added.
- Direct patch approach: read the existing JSON, add the field, write it back.

**Tests:**
- Unit test: `ConcordanceResult` includes `ci_method` with correct default.
- Serialization test: verify the field appears in the output dict.

### Acceptance

`concordance.json` contains `"ci_method": "posterior_propagated_percentile"` and the
notebook reads it without error.

---

## Phase 2: ai-harm Precision Disclosure (Risk 1)

### Problem

All 323 precision verifications are from the security stratum. The ai-harm stratum has
zero direct precision measurements. The Bayesian model handles this correctly via
`apply_empirical_precision_prior()` — entries without measured precision get the average
alpha/beta from measured entries. But the report's Threats to Validity section does not
disclose this, and the notebook's Act 10 does not mention why the gap is accepted.

### Accepted limitation rationale

The ai-harm stratum has only 92 in-scope incidents across 8 entries (LLM09: 34,
LLM04: 30, NEW-MA: 17, then single digits). Finding additional ai-harm incidents to
close the precision gap would require sourcing new data outside the existing corpus,
which is outside this project's scope.

### Changes

**`engine/threats/register.py`:**
- Add new threat after `F-defenseindepth`:
  ```python
  Threat(
      "F-aiharm-precision",
      "ai-harm stratum has no direct precision measurements; precision posteriors "
      "for ai-harm entries borrow the empirical mean from security-stratum "
      "verifications via apply_empirical_precision_prior()",
      "disclosed in notebook Acts 4 and 5; empirical prior is conservative "
      "(averages across all measured entries rather than assuming high precision)",
      "ai-harm error correction relies on borrowed estimates; true ai-harm "
      "precision could differ from security-stratum precision in either direction",
  )
  ```

**Notebook `what_the_data_says_2026.ipynb` — Act 10 markdown cell:**
- Add one paragraph after the "kappa ceiling" paragraph:
  ```
  **Accepted limitation: ai-harm precision.** The 323 precision verifications were
  drawn entirely from the security stratum. The ai-harm stratum (92 in-scope
  incidents across 8 entries) has no direct precision measurements — the model
  borrows estimates from the security stratum average. Closing this gap would
  require sourcing additional ai-harm incidents beyond the existing corpus, which
  is outside this project's scope. The disclosure in Acts 4 and 5 describes how
  the model handles this.
  ```

**Regenerate `report.md`:**
- Re-run the report renderer (or patch directly) so the new threat appears in
  Threats to Validity.

### Acceptance

- `F-aiharm-precision` appears in `report.md` Threats to Validity.
- Notebook Act 10 contains the accepted-limitation paragraph.
- Notebook executes end-to-end without error.

---

## Phase 3: Narrative Report Generator

### Problem

The notebook is the primary interactive artifact, but many stakeholders do not have
Jupyter environments. The internal `render_report()` produces a 53-line summary for
engine use. There is no redistribution-ready document that tells the full story with
visualizations.

### Approach

A standalone Python module that reads the same data files as the notebook, generates
all charts as static PNGs, and writes a narrative markdown report. The output is a
self-contained folder that can be shared with anyone who has a markdown viewer.

### Output structure

```
projects/owasp-llm/cycles/2026/results/narrative/
├── report.md              # Full narrative markdown
└── figures/
    ├── stratum_bar.png
    ├── tier_donut.png
    ├── confusion_heatmap.png
    ├── precision_bars.png
    ├── precision_posteriors.png
    ├── ridge_plot.png
    ├── dumbbell_chart.png
    ├── plotly_rankings.png
    ├── bump_chart.png
    ├── ci_overlap.png
    ├── paired_dots.png
    ├── theme_bars_llm09.png
    ├── theme_bars_new_wla.png
    ├── oos_treemap.png
    ├── sankey_confusion.png
    └── confusion_matrix_3x3.png
```

### Module: `engine/report/narrative.py`

**Single public function:**
```python
def generate_narrative_report(cycle_dir: Path, output_dir: Path) -> Path:
    """Generate a standalone narrative report with embedded figures.

    Returns the path to the generated report.md.
    """
```

**Internals:**
- `_load_data(cycle_dir)` — same data loading as notebook Act 0
- `_generate_figures(data, figures_dir)` — renders all charts as PNG using
  matplotlib/seaborn + plotly/kaleido. Each chart is a private function that returns
  a matplotlib Figure or plotly Figure, then saves it.
- `_render_markdown(data, figures_dir)` — generates the narrative markdown with
  `![caption](figures/filename.png)` image references.

**Narrative structure** — mirrors the notebook's 10 acts:
1. The Question — 20-entry table, context paragraph
2. The Corpus — stratum breakdown, F-circ callout, example incidents
3. Classification — tier distribution, confusion heatmap, pipeline description
4. How Good Is the Classifier? — precision bars, calibration posteriors, gold-set
   description, security-only precision disclosure
5. From Counts to Rankings — ridge plot, MCMC diagnostics, model description
6. The Incident-Derived Rankings — dumbbell chart, rankings table with CIs
7. The Confrontation — bump chart, CI overlap, kappa interpretation, selection bias
8. Where Experts and Incidents Disagree — paired dots, theme bars, flagged entry
   explanations
9. What the Data Cannot See — OOS treemap, confusion boundary Sankey + matrix,
   boundary examples
10. What This Means — conclusions, accepted limitations, kappa ceiling

**Key differences from notebook:**
- All plotly charts rendered as static PNG via kaleido (no interactivity)
- No collapsible `<details>` sidebars — deep-dive content is inline or in footnotes
- No code cells shown — pure narrative + images
- Includes `ci_method` disclosure and `F-aiharm-precision` disclosure from Phases 1-2
- Entry table includes both expert and incident ranks (not the "fill in later" pattern)

### CLI command

New click command in `engine/cli/pipeline_executor.py` alongside the existing
`report` command:

```python
@click.command("report-narrative")
@click.option("--cycle-dir", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--output-dir", type=click.Path(path_type=Path), default=None)
def report_narrative_cmd(cycle_dir: Path, output_dir: Path | None) -> None:
    if output_dir is None:
        output_dir = cycle_dir / "results" / "narrative"
    result_path = generate_narrative_report(cycle_dir, output_dir)
    click.echo(f"Narrative report written to {result_path}")
```

### Chart generation approach

Each chart function follows the same pattern as the notebook code but saves to file
instead of `plt.show()`:

```python
def _render_bump_chart(data: dict, figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 12))
    # ... same plotting code as notebook ...
    fig.savefig(figures_dir / "bump_chart.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
```

For plotly charts:
```python
def _render_oos_treemap(data: dict, figures_dir: Path) -> None:
    fig = px.treemap(...)
    fig.write_image(figures_dir / "oos_treemap.png", width=1000, height=600)
```

### Tests

- Integration test: call `generate_narrative_report()` with the real cycle data,
  verify it produces `report.md` + all 16 PNG files.
- Verify `report.md` contains all 10 act headings.
- Verify no banned AI slop patterns in the generated markdown.
- Verify all `![...](figures/...)` references point to files that exist.

### Acceptance

- `results/narrative/report.md` exists and is readable as standalone markdown.
- All 16 figure PNGs are present and non-empty.
- The narrative is congruent with the notebook — same numbers, same disclosures,
  same conclusions.
- The report includes both Risk 1 and Risk 2 disclosures.

---

## Dependency Order

```
Phase 1 (ci_method)
    ↓
Phase 2 (ai-harm disclosure + notebook update)
    ↓
Phase 3 (narrative report — reads final notebook content + all data)
```

Phase 3 depends on Phases 1 and 2 because the narrative report must include the
disclosures added in those phases.

---

## What Does NOT Change

- `render_report()` in `engine/report/render.py` — stays as internal summary
- The Bayesian model and inference pipeline — no data changes
- The notebook's existing Acts 0-9 — only Act 10 gets one added paragraph
- posteriors.json — no new precision data to add
- No LLM reruns, no RunPod, no human review batches
