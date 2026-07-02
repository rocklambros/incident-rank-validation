# Preprint style guide (binding for all prose)

Audience: cybersecurity professionals, DS-novice to DS-expert. Voice: first-person plural ("we") for our methods; plain, precise, technical; explain every data-science term in a boxed sidebar on first use. This is a scientific preprint, not marketing.

## Calibrated framing (do not deviate)
- Title: *Incident-Data Robustness Analysis of the OWASP Top 10 for LLM Applications (2026): How a Community-Expert Ranking Holds Up Against a Large-Scale LLM Incident Corpus.*
- Thesis, stated the same way everywhere (abstract, Part I, conclusion): the incident data agrees only **weakly** with the expert ranking (Cohen's κ ≈ 0.20, interval crossing zero), and the expert ranking is **robust** — four frontier classifiers plus a ground-truth check do not move it.
- Never write "validates/validated" (the data did not confirm the list) or "first-ever" (write "large-scale"). Say what is true: weak agreement, robust ranking.
- Numbers come from the committed artifacts / the notebook's own computation, never typed by hand.

## Banned AI-writing patterns (per Wikipedia:Signs_of_AI_writing — a reviewer will reject these)
- **Puffery**: rich, vibrant, robust (except the technical sense here), comprehensive, seamless, cutting-edge, powerful, vital/crucial/pivotal role, "stands as", "a testament to", boasts, "in today's ... landscape".
- **Editorializing filler**: "it is important/worth noting", "notably", "importantly", "indeed", "clearly", "of course", "needless to say".
- **Connective filler**: moreover, furthermore, additionally, "in conclusion", "overall", "that said", "when it comes to".
- **Buzzwords**: delve, leverage, utilize (use "use"), showcase, underscore, foster, harness, unlock, elevate, navigate (the complexities of), realm, tapestry, journey.
- **"Not only X but also Y"**, "From X to Y" openers, and rule-of-three padding (three adjectives/nouns where one is exact).
- **Definitional openers**: "X is a Y that…" as a paragraph start — lead with the point instead.
- **Section-summary sentences** that restate the heading; **conclusion paragraphs** that repeat the intro.
- **Vague attribution**: "studies show", "experts agree" — cite the specific artifact or number.
- **Trailing participles**: "…, highlighting the importance of…", "…, reflecting a broader trend…".
- **Em-dash spam** and **over-bolding**. Bold a term at most once (its definition).

## Positive rules
- Lead each paragraph with its claim; support with the number. Prefer concrete counts (6,639; κ=0.20; ρ=0.918) to adjectives.
- Short declarative sentences. Active voice. Define jargon in a `> **Sidebar — <term>.**` blockquote box on first use, and again in the glossary.
- Hedge honestly where the evidence is weak; do not both-sides every point.

## Sidebar/glossary terms to define (novice-first)
incident corpus; classifier; precision; recall; gold set; blind labeling; Cohen's κ (and why an interval crossing 0 means weak agreement); credible interval; prior/posterior; MCMC; balanced accuracy; bootstrap; Spearman ρ; out-of-scope; negative-binomial measurement-error model; latent incidence (λ); the 0.75/0.25 blend.

## Figure filenames (prose `![caption](figures/<name>.png){width=85%}` MUST match what the code cells save, into `notebooks/preprint/figures/`)
Existing (re-saved at 300 dpi): stratum_bar, tier_donut, confusion_heatmap, precision_bars, precision_posteriors, ridge_plot, dumbbell_chart, plotly_rankings, bump_chart, ci_overlap, paired_dots, theme_bars_llm09, theme_bars_new_wla, oos_treemap, sankey_confusion, confusion_matrix_3x3.
New: rank_change_2025_2026, entry_expansion_map, rarr_robustness.

## Key numbers (cite from the artifacts; do not memorize/transcribe — the consistency cell guards them)
- Corpus: 7,714 incidents snapshotted; 6,639 labeled (security 6,297 / ai-harm 342); Corpus B (OWASP ASI) 46 incidents, 26% label agreement.
- Cohen's κ = 0.2029, 90% interval [-0.159, 0.565] (from `baselines/2026/rankings_baselines.json` `previous_ranking`).
- 20 taxonomy entries = 10 incumbents (LLM01–10) + 6 NEW-* + 4 ROLL-*.
- Blend: 0.75·vote_rank + 0.25·λ_rank; biggest movers: Improper Output Handling −5, Unbounded Consumption +4, Excessive Agency +3.
- Robustness (from `cycles/2026-rarr/results/robustness_validation.json`): floor Spearman ρ=0.918 vs truth; no frontier model beats it (deltas' intervals cross 0); corpus-reweighted floor ρ=0.971. Bake-off winner = None. deepseek-v3 balanced accuracy 0.711 < floor 0.863.
- Recall-correction: cite the JSON's values; disclose the nuance — the correction removes the small specific-class distributional gap (interval crosses 0), but it CANNOT recover the floor's 0% out-of-scope recall (a classifier can't be corrected for a class it never predicts), so frontier models are genuinely better at out-of-scope detection. This affects magnitudes, not the rank order.
