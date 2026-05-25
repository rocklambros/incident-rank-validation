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
CI was computed. The CI is a paired-draw percentile interval: for each of 5,000 paired
draws (min of 16,000 lambda posterior samples and 5,000 vote bootstrap samples), kappa
is evaluated on the paired (lambda sample, vote bootstrap sample), and the 2.5th/97.5th
percentiles of those kappa draws form the CI. This is not an analytical normal
approximation, nor a proper Bayesian posterior credible interval — it is an empirical
percentile of a composite variability measure mixing MCMC uncertainty with bootstrap
sampling uncertainty. Without the `ci_method` field, a reader might assume it is one of
these.

### Changes

**`engine/decide/concordance.py`:**
- Add field to `ConcordanceResult` dataclass:
  ```python
  ci_method: str = "paired_draw_percentile"
  ```
  Place `ci_method` after `standing_caveat` and before `entry_comparisons` (both have
  defaults, so field ordering is valid). The `_na_result()` and `compute_concordance()`
  callers use keyword args with defaults — no changes to call sites needed.
- No changes to `compute_concordance()` — the field is metadata about the existing
  computation, not a new computation.

**`engine/cli/pipeline_executor.py`:**
- Add `"ci_method": concordance.ci_method` to the `conc_dict` serialization block
  (around line 349-367).
- Update the deserialization block in `pipeline.py:519-532` to read `ci_method` from
  JSON with fallback to the default value, ensuring round-trip fidelity.

**`projects/owasp-llm/cycles/2026/results/concordance.json`:**
- Regenerate by re-running the serialization path (code change + pipeline run). Do NOT
  use a direct JSON patch — hand-patched artifacts diverge silently from the pipeline
  and are overwritten on the next `decide-real` run without warning.

**Tests:**
- Unit test: `ConcordanceResult` includes `ci_method` with correct default
  (`"paired_draw_percentile"`).
- Serialization test: verify the field appears in the output dict.
- Round-trip test: serialize `ConcordanceResult` → JSON → deserialize back; assert all
  fields match including `ci_method`.

### Acceptance

`concordance.json` contains `"ci_method": "paired_draw_percentile"` and the
notebook reads it without error. The round-trip test (serialize → deserialize → compare)
passes.

---

## Phase 2: ai-harm Precision Disclosure (Risk 1)

### Problem

All 323 precision verifications are from the security stratum. The ai-harm stratum has
zero direct precision measurements — no `::ai-harm` precision keys exist in
`posteriors.json` at all. The Bayesian model's actual fallback for these missing keys
is `Beta(1,1)` = Uniform(0,1) in `inference.py:69-70` (precision_alpha/beta initialized
to `np.ones(...)`). `apply_empirical_precision_prior()` in `calibrate.py:160-181` only
updates keys that **exist** with uninformative priors (alpha==1 and beta==1); since
ai-harm precision keys are entirely absent from the dict, the function never reaches
them. The report's Threats to Validity section does not disclose this, and the
notebook's Act 10 does not mention why the gap is accepted.

**Note:** The `apply_empirical_precision_prior()` empirical mean is also contaminated by
`out-of-scope::security` (alpha=1, beta=125, precision ~0.008), which depresses the
mean by ~7.8pp (0.6439 with out-of-scope vs 0.7216 without). This affects
security-stratum entries that DO receive the borrowed prior but not ai-harm entries
(which don't receive it at all). This contamination is disclosed but not fixed by this
spec — fixing requires a model change outside scope.

### Accepted limitation rationale

The ai-harm stratum has 92 in-scope incidents across 8 entry assignments (LLM09: 34,
LLM04: 30, NEW-MA: 17, then single digits). Of these 8, only 3 entries (LLM09, LLM04,
NEW-MA) received recall posteriors with material evidence; NEW-WLA has only 1
observation above the pure prior (alpha=2 vs prior alpha=1). The remaining 16 ai-harm
recall keys carry the pure prior Beta(1,101). Finding additional ai-harm incidents to
close the precision gap would require sourcing new data outside the existing corpus,
which is outside this project's scope.

### Changes

**`engine/threats/register.py`:**
- Add new threat after `F-defenseindepth`:
  ```python
  Threat(
      "F-aiharm-precision",
      "ai-harm stratum has no direct precision measurements; ai-harm precision "
      "keys are absent from posteriors.json entirely, so the model falls back to "
      "Beta(1,1) = Uniform(0,1) via the default initialization in inference.py "
      "(apply_empirical_precision_prior cannot reach keys that do not exist)",
      "disclosed in notebook Acts 4, 5, and 10; the uniform prior is maximally "
      "uninformative rather than borrowed — it does not assume high or low "
      "precision for ai-harm entries",
      "ai-harm rankings rely on a flat precision prior (mean 0.5); true ai-harm "
      "precision could be substantially higher or lower, shifting rankings in "
      "either direction; only 3 of 20 ai-harm recall keys have material evidence "
      "(LLM09, LLM04, NEW-MA); NEW-WLA has only 1 observation above the pure prior",
  )
  ```

**Notebook `what_the_data_says_2026.ipynb` — Act 10 markdown cell:**
- Add one paragraph after the "kappa ceiling" paragraph:
  ```
  **Accepted limitation: ai-harm precision.** The 323 precision verifications were
  drawn entirely from the security stratum. The ai-harm stratum (92 in-scope
  incidents across 8 entry assignments, of which only 4 received non-trivial
  recall posteriors) has no direct precision measurements — ai-harm precision
  keys are absent from the calibration data entirely. The model falls back to
  a flat Beta(1,1) = Uniform(0,1) prior for ai-harm precision, meaning it
  assumes no prior knowledge about how precise the classifier is on ai-harm
  incidents (prior mean 0.5). Closing this gap would require sourcing additional
  ai-harm incidents beyond the existing corpus, which is outside this project's
  scope. The disclosure in Acts 4 and 5 describes how the model handles missing
  precision data.
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

**Status header:**
- If the cycle is non-publishable (check via `report.md` banner or `ReportInputs`),
  emit `**STATUS: NON-PUBLISHABLE** (single-author rubric, uncontrolled)` as the first
  line of the narrative report, before any act content. This mirrors `render.py:37-40`.

**Narrative structure** — mirrors the notebook's 10 acts:
1. The Question — 20-entry table, context paragraph
2. The Corpus — stratum breakdown, F-circ callout, example incidents
3. Classification — tier distribution, confusion heatmap, pipeline description
4. How Good Is the Classifier? — precision bars, calibration posteriors, gold-set
   description, security-only precision disclosure
5. From Counts to Rankings — ridge plot, MCMC diagnostics, model description
6. The Incident-Derived Rankings — dumbbell chart, rankings table with CIs, Corpus B
   (GenAI agentic) corroboration
7. The Confrontation — bump chart, CI overlap, kappa interpretation, selection bias
8. Where Experts and Incidents Disagree — paired dots, theme bars, flagged entry
   explanations
9. What the Data Cannot See — OOS treemap, confusion boundary Sankey + matrix,
   boundary examples
10. What This Means — conclusions, accepted limitations, kappa ceiling, Threats to
    Validity (rendered programmatically from `get_threats_register()` in
    `engine/threats/register.py`, same as `render.py:181-183`)

**Key differences from notebook:**
- All plotly charts rendered as static PNG via kaleido (no interactivity)
- No collapsible `<details>` sidebars — deep-dive content is inline or in footnotes
- No code cells shown — pure narrative + images
- Includes `ci_method` disclosure and `F-aiharm-precision` disclosure from Phases 1-2
- Entry table includes both expert and incident ranks (not the "fill in later" pattern)
- Includes NON-PUBLISHABLE status header (matching `render.py` behavior)
- Threats to Validity section reads from `get_threats_register()` programmatically
- Set `MPLBACKEND=Agg` in `_generate_figures()` for headless environments

### CLI command

New click command in `engine/cli/pipeline_executor.py` alongside the existing
`report` command:

```python
@click.command("report-narrative")
@click.option("--cycle-dir", type=click.Path(exists=True, path_type=Path, resolve_path=True), required=True)
@click.option("--output-dir", type=click.Path(path_type=Path, resolve_path=True), default=None)
def report_narrative_cmd(cycle_dir: Path, output_dir: Path | None) -> None:
    if output_dir is None:
        output_dir = cycle_dir / "results" / "narrative"
    result_path = generate_narrative_report(cycle_dir, output_dir)
    click.echo(f"Narrative report written to {result_path}")
```

Register this command in the existing click group using the same pattern as the
`report` command (e.g., `cli.add_command(report_narrative_cmd)` or the decorator
pattern used in the codebase). Use a lazy import of `generate_narrative_report`
inside the function body to avoid breaking the CLI if narrative dependencies are not
installed.

**Dependencies (`pyproject.toml`):**
- Add optional dependency group: `[project.optional-dependencies] narrative =
  ["matplotlib>=3.8", "seaborn>=0.13", "plotly>=5.15", "kaleido>=0.2.1"]`
- Pin kaleido ≥ 0.2.1 (fixes headless Chromium subprocess issue on Linux).
- Usage: `pip install -e ".[narrative]"` to install narrative dependencies.

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

- Integration test: call `generate_narrative_report()` with the 2026 cycle data at
  the known path; mark with `pytest.mark.skipif` if the cycle data path is absent
  (allows CI to skip gracefully on fresh clones). Verify it produces `report.md` +
  all 16 PNG files. Verify all PNG files are > 1 KB (catches silent kaleido failures
  that produce empty or 1x1 error images).
- Verify `report.md` contains all 10 act headings.
- Verify no banned AI slop patterns in the generated markdown.
- Verify all `![...](figures/...)` references point to files that exist.
- Numerical congruence spot-check: parse `report.md` for the kappa value and compare
  against `concordance.json`; verify they match within formatting tolerance.
- Verify NON-PUBLISHABLE banner is present (for the 2026 cycle, which is
  non-publishable).

### Acceptance

- `results/narrative/report.md` exists and is readable as standalone markdown.
- All 16 figure PNGs are present, non-empty, and > 1 KB.
- The narrative is congruent with the notebook — same numbers, same disclosures,
  same conclusions. Kappa value spot-checked against `concordance.json`.
- The report includes both Risk 1 and Risk 2 disclosures.
- NON-PUBLISHABLE status header is present (for non-publishable cycles).
- Threats to Validity section is rendered from `get_threats_register()` and includes
  `F-aiharm-precision`.

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
