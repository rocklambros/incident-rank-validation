# "What the Data Says About the 2026 Top 10" — Notebook Design Spec

**Goal**: Build a single Jupyter notebook that tells the story of the incident-rank-validation
analysis for the 2026 OWASP Top 10 for LLMs. Targets AI security experts who are data science
novices. Serves as both a public companion to the published Top 10 list and a standalone
artifact for conference/workshop use.

**Architecture**: Linear narrative ("The Investigation") with deep-dive sidebars via HTML
`<details>` blocks. 10 narrative acts. Hybrid visualization: matplotlib/seaborn by default,
plotly for 2-3 key interactive charts with static PNG fallback.

**Runtime deps** (no JAX, no PyTorch): `numpy>=2.0`, `pandas>=2.0`, `matplotlib>=3.8`,
`seaborn>=0.13`, `plotly>=5.15`, `scipy>=1.10`. Specified in `notebooks/requirements.txt`.

---

## File layout

```
notebooks/
  what_the_data_says_2026.ipynb   # the notebook
  requirements.txt                # notebook-only deps (no JAX/PyTorch)
```

## Data files consumed (all pre-computed, read-only)

All paths are relative to the repository root. The setup cell resolves the repo root
dynamically (see Act 0 spec below).

| File | Used in | Purpose |
|------|---------|---------|
| `prereg/rubric.json` | Act 1, 6 | Entry IDs and canonical names |
| `classify/labeled_incidents_multimodel.json` | Act 2 | Corpus overview, source/stratum breakdown |
| `calibration/llm_prelabels.jsonl` | Acts 3, 8, 9 | 3-model votes, consensus, tier, incident text |
| `calibration/adjudicated_goldset.jsonl` | Acts 4, 9 | 1,200 human adjudications |
| `calibration/precision_verification.jsonl` | Act 4 | 323 precision labels |
| `calibration/posteriors.json` | Acts 4, 5 | Beta posteriors for recall and precision |
| `calibration/diagnostic.json` | Acts 4, 6, 7 | Per-entry adequacy flags (adequate/wide/no-data) |
| `infer/inference_summary.json` | Act 5 sidebar | MCMC diagnostics (R-hat, ESS, divergences) |
| `infer/lambda_samples.npy` | Acts 5, 6 | 16,000 posterior draws, shape (16000, 20) |
| `results/concordance.json` | Act 7 | Kappa, CI, flags |
| `results/selection_bias.json` | Act 7 sidebar | Kruskal-Wallis H, p-value |
| `results/rank_comparison_report.md` | Acts 7, 8 | Per-entry lambda vs vote ranks with CIs (parsed as text; rank data also derivable from lambda_samples.npy + concordance.json) |
| `results/corpus_b_corroboration.json` | NOT USED | 100% overlap with corpus A; excluded to avoid false-independence framing |

**corpus_b_corroboration.json is excluded by design.** All 46 corpus B incidents overlap
with corpus A. Presenting this as "corroboration" would imply independence that does not
exist. The file is loaded nowhere and referenced nowhere in the narrative.

## Writing constraints

These patterns are banned from all markdown cells. The implementation must grep for them
before finalizing:

**Banned words/phrases**: delve, landscape (abstract), tapestry, pivotal, testament,
robust (unless describing a statistical method by name), groundbreaking, vibrant, nestled,
meticulous, intricate, interplay, fosters/fostering, showcases/showcasing, underscores,
bolsters, garner, boasts, serves as, stands as, not just X but also Y, it is worth noting,
it is important to note, a comprehensive look, this section explores, in this section we
will, let us now turn to.

**Banned structural patterns**: Rule-of-three adjective lists. Separate "Challenges and
Future Prospects" sections. Promotional tone. Weasel attributions ("experts contend,"
"researchers suggest" without citation). Outline-like conclusions
("Despite X, Y faces challenges...").

**Voice**: Direct, specific, occasionally dry. Short sentences. Active voice. The reader
is a colleague being briefed, not a student being lectured. Use "we" for the analysis team,
"you" for the reader. Concrete examples over abstractions.

---

## Story Arc

### Act 0: Setup (hidden cell, collapsed)

**Cell type**: Code (hidden via metadata `"jupyter": {"source_hidden": true}`)

Contents:
1. **Warning suppression**: `warnings.filterwarnings('ignore')` for matplotlib deprecation
   and pandas future warnings. No JAX imports.
2. **Path resolution**: Find repo root dynamically:
   ```python
   from pathlib import Path
   # Try notebook dir first, then CWD
   _here = Path('.').resolve()
   if (_here / 'projects').exists():
       REPO_ROOT = _here
   elif (_here.parent / 'projects').exists():
       REPO_ROOT = _here.parent
   else:
       raise FileNotFoundError(
           "Cannot find 'projects/' directory. "
           "Run this notebook from the repository root or the notebooks/ directory."
       )
   CYCLE = REPO_ROOT / 'projects' / 'owasp-llm' / 'cycles' / '2026'
   ```
3. **Imports**: numpy, pandas, matplotlib, seaborn, plotly, json, pathlib, scipy.stats.
   Set seaborn theme. Define color palette (muted, 20 colors mapped to entry IDs).
4. **Plotly renderer**: `import plotly.io as pio; pio.renderers.default = "notebook+png"`
   (renders interactive in Jupyter, falls back to static PNG in export/nbviewer).
5. **Helper: deep-dive sidebar**:
   ```python
   from IPython.display import HTML
   def sidebar(title, content):
       return HTML(f'<details><summary><strong>{title}</strong></summary>{content}</details>')
   ```
6. **Load all data** into a dict-of-dicts. Validate shapes and key fields on load. Print
   a one-line summary: "Loaded N data files. Ready."

### Act 1: The Question

**Markdown cell** (~4 paragraphs):
- The OWASP Top 10 for LLMs ranks AI security vulnerabilities. The 2025 list was built from
  expert surveys — hundreds of security professionals voting on what matters most.
- Expert opinion is one signal. What if we checked it against a second signal: the pattern of
  real-world incidents? We built a corpus of ~6,600 real AI security incidents, classified each
  one, and asked: does the incident data agree with the experts?
- This notebook walks through that analysis. Along the way, it will show you how the
  classification worked, how we measured its accuracy, and what a Bayesian model does with
  noisy measurements. Every chart and table is computed live from the data — you can re-run
  any cell to verify.

**Code cell**: Load rubric, display a table of 20 entries with canonical names. The
"Incident-Derived Rank" column shows "—" for all entries. A footnote says: "We will fill
this column in Act 6, after walking through the methodology."

### Act 2: The Corpus

**Markdown cell** (~3 paragraphs):
- The corpus contains 6,639 incidents from public databases: CVE, GHSA, OSV (security
  advisories), and AIAAIC (AI harm reports). Each record has a description of what happened.
- Two strata: "security" (CVE/GHSA/OSV — things like prompt injection, data leakage,
  supply chain compromise) and "ai-harm" (AIAAIC — things like bias, misuse,
  misinformation). This split matters because the classifier performs differently on each.

**Code cell 1**: Bar chart of incident counts by stratum.

**Code cell 2**: Show 2-3 real incident examples (redacted if needed) — one from each
stratum. Display as formatted cards with incident ID, source, and text snippet.

**Sidebar: What the corpus can't see (F-frame)**:
Markdown explaining that the corpus is built from a keyword crawl of public databases.
Incidents that never became CVEs, GHSAs, or harm-database entries are invisible. This
creates a structural bias toward vulnerability types that get reported in those channels.

**Sidebar: Taxonomy-frame circularity (F-circ)** [PREMORTEM R9]:
**This must be an always-visible boxed callout, NOT a collapsed sidebar.**
Text: "A structural limitation: we classified these incidents using the same taxonomy we
are trying to validate. If the classifier systematically favors certain entries, the incident
counts will appear to confirm the expert rankings even if the true pattern is different.
This is taxonomy-frame circularity. It means the concordance we measure later is an upper
bound on true agreement, not a precise estimate of it."

### Act 3: Classification — How We Labeled 6,600 Incidents

**Markdown cell** (~3 paragraphs):
- Each incident was classified by three different large language models (Qwen 235B, Llama
  405B, DeepSeek V3), each independently reading the incident text and assigning it to one of
  the 20 taxonomy entries or marking it "out of scope."
- When all three agreed, we called it "agree" tier. When two agreed, "split" tier. When all
  three picked different entries, "disagree" tier.
- Walk through one concrete example: show a real incident, the three model votes, and the
  consensus outcome.

**Code cell 1**: Donut chart of tier distribution: 2,568 agree / 2,973 split / 431
disagree. Label percentages. (Donut, not pie — the center hole avoids the angle-comparison
problem that makes pie charts hard to read.)

**Code cell 2**: Heatmap of entry-pair disagreements (confusion matrix). Which pairs of
entries do models most often confuse? This sets up Act 9's confusion boundary analysis.

**Sidebar: Why three models?** Explanation of multi-model consensus. Single-model
classification had lower precision. Three models with majority vote reduces noise.

**Sidebar: The two-stage pipeline.** Stage 1 uses regex/heuristic indicators for fast
initial assignment. Stage 2 sends the full text to each LLM for detailed classification.

### Act 4: How Good Is the Classifier?

**Markdown cell** (~3 paragraphs):
- The classifier is a tool, not a source of truth. We measured its accuracy two ways.
- **Precision**: When the classifier says "this incident belongs to LLM02," how often is
  it right? We verified 323 classifications by hand.
- **Recall**: Does the classifier find all incidents of a given type, or does it miss some?
  We had a human reviewer adjudicate 1,200 incidents across all tiers.

**Code cell 1**: Bar chart of precision by entry, from posteriors.json. **Each bar annotated
with sample size (n=X)** [PREMORTEM R10]. Entries with n<5 shown with hatched/transparent
bars and footnote: "Entries with fewer than 5 precision observations are dominated by the
prior assumption, not the data."

**Code cell 2**: Calibration-style plot showing precision Beta posterior distributions for
4-5 key entries (LLM03 at ~93%, LLM02 at ~69%, LLM06 at ~30%, LLM07 at ~31%). Show the
distribution width to communicate uncertainty.

**Sidebar: The gold-set process.** 1,200 human adjudications using a blind-first protocol.
The reviewer sees the incident text and the three model votes, then decides: accept the
consensus, override to a different entry, or mark as out of scope. Explain what
"adjudication" means and why it's different from voting.

**Sidebar: Precision posteriors table.** Full table of alpha, beta, mean, 90% CI for all
entries with precision data. Flag entries where the posterior is prior-dominated (n<5).

### Act 5: From Counts to Rankings — The Bayesian Model

**Markdown cell** (~4 paragraphs):
- Raw incident counts would be misleading. An entry with a 30% precision classifier looks
  like it has many incidents — but two-thirds of those are misclassifications.
- We need a model that adjusts the observed counts for known classifier error. Think of a
  bathroom scale that reads 2 pounds heavy. You would subtract 2 pounds from every reading.
  The Bayesian model does this, but for each entry separately, and it carries the uncertainty
  through — if the scale is 2±1 pounds off, the corrected weight is also uncertain.
- The model takes the observed incident counts, the measured precision and recall for each
  entry, and produces a posterior distribution over the true incident rate for each entry.
  We drew 16,000 samples from this distribution using Markov chain Monte Carlo (MCMC), which
  is a method for sampling from probability distributions that are too complex to compute
  directly.
- **For 16 of 20 entries in one data stratum, we have not measured recall directly.**
  [PREMORTEM R8] The model uses a conservative prior estimate of ~1% recall for those
  entries. This means the model assumes the classifier finds very few of those incidents
  and adjusts upward accordingly. These corrections are large, which is one reason the
  credible intervals in Act 6 are wide.

**Code cell 1**: Ridge plot of posterior distributions for all 20 entries
(lambda_samples.npy). **Three frame-blind entries (LLM04, LLM08, LLM10) shown in gray with
a "prior only" label** [PREMORTEM R2]. The remaining 17 entries shown in color.

**Code cell 2**: Summary statistics table — median lambda, 90% CI width, diagnostic flag
(adequate/wide/no-data) from diagnostic.json.

**Sidebar: The NumPyro model specification.** Static markdown code block (NOT executed)
showing the model. Preceded by: "This is the actual model we used, written in NumPyro
(a probabilistic programming library for JAX). You do not need to install NumPyro to run
this notebook — the results below are pre-computed."

**Sidebar: MCMC convergence diagnostics.** R-hat values (all should be ~1.0), effective
sample sizes, zero divergences, trace plots for 2-3 representative parameters.
Explain what each diagnostic means in plain language.

### Act 6: The Incident-Derived Rankings

**Markdown cell** (~2 paragraphs):
- Now we can fill in the table from Act 1. For each entry, the Bayesian model gives us a
  posterior distribution over its true incident rate. We rank entries by their median rate
  and report a 90% credible interval on the rank.
- Some entries have tight intervals (the data is informative) and others are wide (less
  certain). The width tells you how much to trust the rank position.

**Title: "The Incident-Derived Rankings"** (NOT "The Answer") [PREMORTEM R6].

Opening qualifier: "These rankings reflect what the incident data suggests after correcting
for classifier error. They are one signal, not the final word."

**Code cell 1**: Horizontal lollipop/dumbbell chart with 90% credible intervals for each
entry's rank. **Frame-blind entries (LLM04, LLM08, LLM10) grayed out with asterisk and
footnote: "No incident signal — rank reflects prior assumptions only."** [PREMORTEM R2].

**Code cell 2** (interactive plotly): Same data as an interactive chart with hover showing
exact median, CI bounds, and diagnostic flag. Static PNG fallback via
`pio.renderers.default = "notebook+png"` [PREMORTEM R7].

**Code cell 3**: Fill in the Act 1 table with the incident-derived ranks. Show the updated
table side by side with the still-hidden vote ranks.

### Act 7: The Confrontation — Do Experts and Incidents Agree?

**Markdown cell** (~4 paragraphs):
- Cohen's weighted kappa measures agreement between two ranking systems, adjusted for
  chance. A value of 1.0 means perfect agreement. A value of 0 means no better than random.
  Negative values mean systematic disagreement.
- **Our result: kappa = 0.20, with a 90% credible interval of [−0.16, 0.57].**
  [PREMORTEM R3] This interval includes zero. The data cannot exclude the possibility
  that expert and incident rankings agree by chance alone. The point estimate of 0.20
  suggests slight agreement, but the wide interval means this is a weak signal, not a firm
  conclusion.
- Five entries have statistically significant tier mismatches (probability > 0.83 that
  expert and incident tiers disagree):
  - LLM01 Prompt Injection: experts rank it #1, incidents rank it #12
  - LLM09 Misinformation: incidents rank it #2, experts rank it #13
  - NEW-MTIE MCP Tool Interface Exploitation: experts rank it #7, incidents #16
  - NEW-PMP Persistent Memory Poisoning: experts rank it #4, incidents #16
  - NEW-WLA Weaponized LLM Abuse: incidents rank it #8, experts #17

**Code cell 1**: Bump chart / slope chart showing expert rank vs incident rank for all 20
entries. Color-coded: green for same tier, yellow for ±1 tier, red for ±2+ tiers. The five
flagged entries labeled by name. **Frame-blind entries shown as dashed lines.**

**Code cell 2**: Show the CI overlap for each entry. **Add shaded regions where lambda and
vote CIs overlap** [PREMORTEM R5]. Annotation: entries with non-overlapping CIs (only
LLM09, possibly LLM01) are marked as "statistically distinguishable."

**Sidebar: How weighted kappa works.** Walk through the calculation with a small concrete
example. Explain why sample size (N=17 measurable entries) drives the wide CI.

**Sidebar: Selection bias.** Kruskal-Wallis test: H=0.55, p=0.46, severity=low.
Explain what the test checked (whether incident rates differ systematically between data
strata) and what "low severity" means.

### Act 8: Where Experts and Incidents Disagree

**Markdown cell** (~5 paragraphs, one per flagged entry):
For each of the five flagged entries, explain WHY the disagreement exists. Ground each
explanation in specific data:

- **LLM01 (Prompt Injection)**: Expert #1, incident #12. Prompt injection is
  well-understood and well-defended in deployed systems. Fewer incidents make it to public
  databases because defenses work. Experts rank it high because they know the attack surface
  is large even when defenses hold.
- **LLM09 (Misinformation)**: Incident #2, expert #13. The corpus is full of deepfake and
  misinformation incidents from the AIAAIC harm database. Experts may rank it lower because
  "misinformation" spans a broad category that overlaps with other entries (see Act 9B).
- **NEW-PMP / NEW-MTIE**: Expert top-tier, almost no incidents yet. These are emerging
  threats — MCP tool exploitation and persistent memory poisoning are new enough that the
  public incident record hasn't caught up.
- **NEW-WLA (Weaponized LLM Abuse)**: 863 incidents, expert rank 17. Large incident count
  driven by a broad entry definition that captures AI-generated disinformation, deepfake
  CSAM, and synthetic media abuse. Experts may rank it low because many of these incidents
  describe harm FROM AI rather than a vulnerability IN an LLM.

**Code cell 1**: Paired dot plots for each flagged entry showing lambda rank vs vote rank
with CIs. Compact 5-panel figure.

**Code cell 2**: For LLM09 and NEW-WLA, mini stacked bars showing incident themes
(deepfake, CSAM, political disinformation, etc.) from the prelabels text mining.

### Act 9: What the Data Cannot See

**Markdown cell** (intro, ~2 paragraphs):
- The ranking analysis covers the 17 measurable entries. But two patterns in the data
  reveal structural limits of what incident-counting can tell us.

#### Act 9A: "AI Harm Without LLM Vulnerability"

**Markdown cell** (~3 paragraphs):
- 2,394 incidents — 40% of the corpus — landed in "out of scope." All three models agreed
  these don't belong to any of the 20 taxonomy entries. Another 272 disagree-tier incidents
  were confirmed as out of scope by the human reviewer, who noted: "No entry fit after rubric
  indicator review."
- These are real AI harms — facial recognition failures, algorithmic discrimination, drone
  surveillance, recommendation engine manipulation. But they don't describe a vulnerability
  in a large language model. They're incidents FROM AI systems, not incidents OF LLM
  vulnerabilities.
- This gap is a feature of the sampling frame, not a bug in the taxonomy. The corpus was
  built by crawling CVE/GHSA/OSV databases with AI-related keywords. Those keywords pull in
  any incident that mentions "AI" or "machine learning," regardless of whether an LLM is
  involved.

**Code cell 1**: Treemap (plotly, interactive, with PNG fallback) of OOS themes:
bias/discrimination (251), surveillance/facial recognition (527), algorithmic harm (350),
deepfake/synthetic media (123), autonomous vehicles (127), AI labor (196), copyright/IP (68),
CSAM/NCII (46), governance gap (189), etc. Computed from keyword scan of OOS incident text.

**Code cell 2**: What the human reviewer overrode to out-of-scope in the disagree tier.
Bar chart of the top entry triples that got sent to OOS: LLM04/NEW-MA/OOS (32),
LLM09/NEW-WLA/OOS (29), etc.

#### Act 9B: The LLM09 / NEW-WLA / ROLL-CMSB Confusion Boundary

**Markdown cell** (~5 paragraphs):
- Explain what a confusion boundary is: a region where categories overlap enough that
  classifiers — and sometimes humans — can't reliably tell them apart. In a confusion
  boundary, the problem isn't that the classifier is broken. The problem is that the
  categories share real conceptual territory.
- The data shows a clear confusion boundary between three entries:
  - LLM09 (Misinformation): the output is false or misleading
  - NEW-WLA (Weaponized LLM Abuse): an adversary uses AI as a weapon
  - ROLL-CMSB (Cross-Modal Safety Bypass): the attack uses image/video/audio modalities
- When a deepfake video spreads political disinformation, which entry does it belong to?
  It's misleading content (LLM09). It was created using AI as a weapon (NEW-WLA). It
  exploits an image/video generation modality (ROLL-CMSB). The three categories overlap
  in real-world incidents.
- Show the data: 52 disagree-tier incidents where all three models picked different entries
  from this cluster. 130 split-tier incidents where LLM09 and NEW-WLA split. 70 where
  NEW-WLA and ROLL-CMSB split.
- What the human reviewer decided: of the LLM09/NEW-WLA/OOS cluster, 43 went to out of
  scope ("not an LLM vulnerability at all"), 31 went to LLM09, and only 2 went to NEW-WLA.

**Code cell 1**: Sankey diagram (plotly, interactive) showing how model votes flow between
LLM09, NEW-WLA, ROLL-CMSB, and out-of-scope in the disagree tier. The width of each flow
shows the number of incidents.

**Code cell 2**: 3x3 confusion matrix heatmap for just these three entries, showing how
often each model pair disagrees.

**Code cell 3**: Show 2-3 real example incidents from this cluster with the three model
votes and the human decision. These make the confusion tangible.

### Act 10: What This Means

**Markdown cell** (~5 paragraphs, no code):
- **Where the data and experts agree**: LLM02 (Sensitive Information Disclosure) is
  consistently near the top by both measures. ROLL-SICG, NEW-ITSCD, and NEW-MSDA are
  consistently low. These positions are stable.
- **Where the data pushes back**: LLM09's incident volume is much higher than its expert
  rank. This may reflect real prevalence or a broad entry definition that captures
  AI-adjacent harm. NEW-WLA shows the same pattern. The confusion boundary between these
  entries and ROLL-CMSB suggests the taxonomy could benefit from clearer boundary
  definitions.
- **What's genuinely new**: NEW-PMP and NEW-MTIE have almost no incidents in the public
  record but strong expert signal. These are forward-looking entries — the kind of threat
  that experts recognize before the incident database catches up. If the goal is to warn
  practitioners, expert signal matters more than incident counts for emerging threats.
- **What this methodology can and can't do**: This is a triangulation tool. It checks one
  signal (expert surveys) against another (incident data). Neither signal is the truth.
  The incident data has known structural biases (sampling frame, classifier error,
  taxonomy circularity). The expert data has known structural biases (availability bias,
  recency effects, anchoring to prior lists). The value is in the comparison, not in either
  signal alone.
- **The kappa ceiling is structural.** Some of the disagreement between expert and incident
  rankings is informative — it reveals real differences between "what experts worry about"
  and "what has actually happened." Improving kappa is not necessarily the goal. Understanding
  the disagreements is.

---

## Visualization specifications

### Color palette
A 20-color palette with semantic grouping:
- Incumbent entries (LLM01-LLM10): blues and teals
- NEW- entries: oranges and reds
- ROLL- entries: purples
- Frame-blind entries (LLM04, LLM08, LLM10): gray in all charts

### Chart defaults
```python
plt.rcParams.update({
    'figure.figsize': (12, 6),
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.dpi': 150,
})
sns.set_theme(style='whitegrid', font_scale=1.1)
```

### Plotly configuration
```python
import plotly.io as pio
pio.renderers.default = "notebook+png"
pio.templates.default = "plotly_white"
```

This ensures interactive charts in Jupyter and static PNG fallback in nbviewer/export.

### Annotation strategy
- Annotate directly on data points rather than relying on legends for key findings
- Frame-blind entries always marked with ★ or gray fill
- Small-sample entries (n<5) shown with hatched bars
- CI overlap shown as shaded regions, not just error bars

---

## Premortem remediations incorporated

| ID | Finding | Where addressed |
|----|---------|-----------------|
| R1 | Missing runtime deps | `notebooks/requirements.txt`, Act 0 |
| R2 | Frame-blind entries unmarked | Acts 5, 6, 7 — gray + "prior only" label |
| R3 | Kappa CI in main text | Act 7 — CI and zero-crossing in main narrative |
| R4 | GPU detection contradicts deps | Removed — no GPU detection, no JAX |
| R5 | Dumbbell chart CI overlap | Act 7 — shaded overlap regions |
| R6 | Act 6 title "The Answer" | Renamed to "The Incident-Derived Rankings" |
| R7 | Plotly renderer unspecified | Act 0 — `pio.renderers.default = "notebook+png"` |
| R8 | Recall prior dominance hidden | Act 5 — main text discloses 16/20 prior-dominated |
| R9 | F-circ buried | Act 2 — always-visible boxed callout |
| R10 | Precision chart sample sizes | Act 4 — annotated bars, hatched for n<5 |
| R11 | Path resolution | Act 0 — dynamic repo root detection |

---

## Implementation notes

- **No re-running classification or inference.** The notebook only reads pre-computed files.
- **Incident text display**: When showing real incident examples (Acts 2, 3, 9B), truncate
  to ~200 characters with "..." and display as formatted cards.
- **Deep-dive sidebars**: Use `<details><summary>` HTML in markdown cells. Collapsed by
  default. Title clearly states what's inside.
- **Act 0 data validation**: Check that all expected files exist on load. If any are missing,
  print a clear error message with the expected path rather than silently failing.
- **Corpus B exclusion**: `corpus_b_corroboration.json` is deliberately not loaded. Do not
  reference it in any narrative cell.
- **AI slop check**: Before finalizing, grep all markdown cells for the banned patterns
  listed in the writing constraints section above.
