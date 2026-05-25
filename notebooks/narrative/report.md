**STATUS: NON-PUBLISHABLE** (single-author rubric, uncontrolled)

# What the Data Says About the 2026 Top 10

## Act 1: The Question

| Entry | Name | Incident Rank |
|-------|------|---------------|
| LLM01 | Prompt Injection | — |
| LLM02 | Sensitive Information Disclosure | — |
| LLM03 | Supply Chain Vulnerabilities | — |
| LLM04 | Data and Model Poisoning | — |
| LLM05 | Improper Output Handling | — |
| LLM06 | Excessive Agency | — |
| LLM07 | Hidden Context Exposure | — |
| LLM08 | Vector and Embedding Weaknesses | — |
| LLM09 | Misinformation | — |
| LLM10 | Unbounded Consumption | — |
| NEW-ITSCD | Inference-Time Side-Channel Disclosure | — |
| NEW-MA | Model Misalignment | — |
| NEW-MSDA | Model Scheming and Deceptive Alignment | — |
| NEW-MTIE | MCP Tool Interface Exploitation | — |
| NEW-PMP | Persistent Memory Poisoning | — |
| NEW-WLA | Weaponized LLM Abuse | — |
| ROLL-CFAS | Compositional Fine-tuning Alignment Subversion | — |
| ROLL-CMSB | Cross-Modal Safety Bypass | — |
| ROLL-LAPTF | LLM Artifact Promotion Trust Failure | — |
| ROLL-SICG | Systemic Insecure Code Generation | — |

## Act 2: The Corpus

- **security**: 6297 incidents
- **ai-harm**: 342 incidents

Total: 6639 incidents.

![Stratum breakdown](figures/stratum_bar.png)

## Act 3: Classification

Consensus tiers: 1772 agree, 1375 split, 431 disagree.

![Tier distribution](figures/tier_donut.png)

![Confusion heatmap](figures/confusion_heatmap.png)

## Act 4: How Good Is the Classifier?

Precision verifications: 323 records across 16 verified entries. Posterior keys: 21 security-stratum, 0 ai-harm.

**Note:** The ai-harm stratum has zero direct precision measurements. The model falls back to a flat Beta(1,1) = Uniform(0,1) prior for ai-harm precision.

![Precision bars](figures/precision_bars.png)

![Precision posteriors](figures/precision_posteriors.png)

## Act 5: From Counts to Rankings

MCMC: 16000 posterior draws, 20 entries.

Max R̂: 1.0001. Min ESS: 15708.

![Ridge plot](figures/ridge_plot.png)

## Act 6: The Incident-Derived Rankings

![Dumbbell chart](figures/dumbbell_chart.png)

![Rankings](figures/plotly_rankings.png)

### Corpus B Corroboration

Corpus B (GenAI agentic): 46 incidents. Shared with corpus A: 46. Label agreement: 12/46 (26%).

## Act 7: The Confrontation

Weighted Cohen's kappa: 0.2029 [-0.1594, 0.5652] (95% interval, method: paired_draw_percentile).

Selection bias: H = 0.5507, p = 0.4580.

![Bump chart](figures/bump_chart.png)

![CI overlap](figures/ci_overlap.png)

## Act 8: Where Experts and Incidents Disagree

5 entries flagged (P(tier mismatch) > τ):

- **LLM01**: P = 0.87, direction = vote_over_ranks
- **LLM09**: P = 0.99, direction = vote_under_ranks
- **NEW-MTIE**: P = 0.83, direction = vote_over_ranks
- **NEW-PMP**: P = 0.92, direction = vote_over_ranks
- **NEW-WLA**: P = 0.84, direction = vote_under_ranks

![Paired dots](figures/paired_dots.png)

![Theme bars LLM09](figures/theme_bars_llm09.png)

![Theme bars NEW-WLA](figures/theme_bars_new_wla.png)

## Act 9: What the Data Cannot See

![OOS treemap](figures/oos_treemap.png)

![Sankey confusion](figures/sankey_confusion.png)

![Confusion matrix](figures/confusion_matrix_3x3.png)

## Act 10: What This Means

The current kappa of 0.20 [-0.16, 0.57] is consistent with weak-to-moderate agreement, but the confidence interval is too wide to draw firm conclusions.

**Accepted limitation: ai-harm precision.** The 323 precision verifications were drawn entirely from the security stratum. The ai-harm stratum (92 in-scope incidents across 8 entry assignments, of which only 3 received recall posteriors with material evidence — LLM09, LLM04, NEW-MA; NEW-WLA has only 1 observation above the pure prior) has no direct precision measurements — ai-harm precision keys are absent from the calibration data entirely. The model falls back to a flat Beta(1,1) = Uniform(0,1) prior for ai-harm precision, meaning it assumes no prior knowledge about how precise the classifier is on ai-harm incidents (prior mean 0.5).

## Threats to Validity

- **F1-ingestion-frame**: corpus sampling frame is blind to incidents that never become CVE/GHSA/OSV entries or harm-database rows
- **F2-default-seed-contamination**: CVE ingest seeds every entry with LLM03; bare-default labels are contamination
- **F3-classifier-blind-spots**: classifier is a heuristic with measured error, not a source of truth
- **F4-structural-blind-spots**: owasp_llm distribution reflects classifier ruleset, not threat landscape
- **F5-circular-atlas**: MITRE ATLAS labels are backfill from OWASP, not independent
- **F-frame**: corpus A built by CVE/GHSA/OSV keyword crawl; deployed-app failures outside these sources never enter the sampling base
- **F-circ**: taxonomy-frame circularity: measuring a taxonomy against incidents classified by that taxonomy
- **F-adversarial-ingestion**: public CVE/GHSA/OSV are open submission surfaces; descriptions are attacker-controlled; infer_attack_vector is pure regex
- **F-defenseindepth**: the engine has many integrity controls; this can create false confidence that all bugs are upstream of the engine when debugging unexpected results
- **F-aiharm-precision**: ai-harm stratum has no direct precision measurements; ai-harm precision keys are absent from posteriors.json entirely, so the model falls back to Beta(1,1) = Uniform(0,1) via the default initialization in inference.py (apply_empirical_precision_prior cannot reach keys that do not exist)

---
Before publishing externally, verify against `docs/REVIEWERS.md` PRE-PUBLISH CHECKLIST. This report is internal-only unless the checklist passes.
