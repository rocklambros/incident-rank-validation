# Blended GenAI/OWASP LLM Top 10 (2026): Methodology, Justification, and Ranking

Owner: Rock Lambros (rock@rockcyber.com)
Version: 1.0 (probabilistic blend, adopted method), 2026-07-05
Status: Adopted analytical method for this exploratory report (authors' adoption, not an OWASP endorsement). Standing disclosures unchanged: single-author gold set, interim reviewers, non-publishable status, weak agreement (Cohen's kappa about 0.20, interval crossing zero). This is a documented amendment from the 0.1 interim rank-space blend. See Section 5 for the ranking and Section 7 for residual and tail risk.

---

## 1. The question

Produce one ordered Top 10 for the 2026 GenAI/OWASP LLM risk list that combines two independent signals: what practitioners believe matters (the community vote) and what the incident record shows (data-derived prevalence). The community vote carries weight 0.75. The data carries weight 0.25.

The 0.75/0.25 split is a deliberate stance. The community vote leads because the list is a practitioner-consensus artifact and because the incident data, while useful, is noisy and under-detects unevenly. The data does not overrule the crowd. At a quarter weight it corrects and pressure-tests the crowd.

## 2. Inputs

Human vote. About 29 respondents scored each candidate risk on a 1-to-5 importance scale. The loader converts each ballot to a within-ballot rank, then aggregates to a median vote rank per entry with a 90 percent interval. Source: `engine/vote/`, the survey workbook.

Data vote. A two-stage classifier labels each incident in a corpus of 7,714 GenAI agentic incidents. Stage 1 applies deterministic indicator rules. Stage 2 runs an LLM adjudication pass on RunPod. A dual-coded, adjudicated gold set calibrates per-category precision and recall as Beta posteriors. A Bayesian model infers each category's latent prevalence (lambda) from the calibrated counts. MCMC converged with R-hat below 1.001 and zero divergences. The data vote is the rank of each entry by inferred prevalence. Source: `engine/classify/`, `engine/calibrate/`, `engine/model/`.

## 3. Taxonomy and rollups

The 2026 cycle tracks 20 candidates: 10 published incumbents (LLM01 to LLM10), 6 new candidates held off the published list, and 4 rolled-up children that the working group merged into a parent entry.

| Rolled-up child | Folds into |
|---|---|
| ROLL-CMSB, Cross-Modal Safety Bypass | LLM01 Prompt Injection |
| ROLL-LAPTF, Artifact Promotion Trust Failure | LLM03 Supply Chain |
| ROLL-CFAS, Compositional Fine-tuning Subversion | LLM04 Data and Model Poisoning |
| ROLL-SICG, Systemic Insecure Code Generation | LLM05 Improper Output Handling |

"Rolled up" means the working group folded a narrower child risk into a broader parent rather than listing it on its own. The incidents that belong to the child still happened. They now count toward the parent.

## 4. Methodology decisions

### 4.1 Scope: how to treat the rollups

Three options were compared on real data.

- Scope 1, fold. Rank the 10 incumbents. Fold each child's signal into its parent on both axes. The 6 new candidates stay off the list.
- Scope 2, re-select. Let the blend decide which 10 of the 20 candidates make the cut, so a new candidate could displace an incumbent.
- Scope 3, parent-only. Rank the 10 on each parent's own signal and report the children separately as a sensitivity check.

Decision: Scope 1 is the headline ranking. Scope 2 is rejected. Scope 3 is the rollup-audit method described in the reasons below. This document does not report its results as a separate table.

Reasons:

- Scope 1 credits each incident to the entry that absorbed it. That is the honest accounting given the published list, and it matches the current state of `main`.
- Scope 3 tests whether a child carries a distinct incident cluster the parent does not absorb, using the distinct-cluster test already implemented in `engine/decide/rollup.py`.
- Scope 2 would change which risks appear on the list. That is a membership claim, not an ordering claim, and the incident data is too weak to carry it. Weighted Cohen's kappa is 0.203 with a 90 percent interval of -0.16 to 0.57, which crosses zero. Re-selecting the list on that evidence would overreach. For the record, under a rank-space blend Scope 2 would have promoted two new candidates (Persistent Memory Poisoning, MCP Tool Interface Exploitation) and re-admitted one rolled-up child (Cross-Modal Safety Bypass), ejecting Misinformation, Vector and Embedding Weaknesses, and Improper Output Handling. The working group already weighed those membership calls. The blend respects them.

### 4.2 Blend weights

Human 0.75, data 0.25. The list is a consensus product, so practitioner judgment anchors it. The data enters as a structured corrective at one quarter weight, strong enough to move an entry a tier when the gap is large, not strong enough to overturn the consensus on a single noisy corpus.

### 4.3 Blend scale: the adopted probabilistic method

The ranking in Section 5 uses the probabilistic blend as the adopted analytical method. `engine/decide/blend.py` pairs 16,000 draws from the lambda posterior (the recall- and precision-corrected incidence rates) with 16,000 draws from the vote-rank posterior, blends each pair in score space at the 0.75/0.25 weights, and reports a distribution over each entry's final position. The reconstruction is recorded in `docs/provenance/2026-07-05-probabilistic-blend-reconstruction.md`, and the executive decision to adopt it is recorded in `docs/decisions/2026-07-05-probabilistic-blend-adoption.md`.

The data axis is z-scored on its native, linear scale: the corrected incidence rates are standardized directly, so the 0.25 data tug reflects absolute prevalence differences. This is not a purity claim. Linear is the transform that reproduces the recorded order from the committed posterior samples, and the original computation that first produced this order ran in a session never committed to version control. Choosing linear is a defended reconstruction, disclosed as such. It does not prove that linear is the uniquely correct scale. The committed knob-sweep in `engine/decide/blend_prototype_reference.py` shows a log transform swaps two entries, both already inside the unordered tail (Section 5), so no ordered claim in this report depends on the transform choice.

A separate robustness campaign, RARR, tested whether the incidence ranking behind these lambda posteriors depends on which classifier labeled the corpus (`projects/owasp-llm/cycles/2026-rarr/`). Four independent methods agree the ranking is robust to classifier choice, at Spearman rho 0.918 against a held-out adjudicated ground truth, rising to 0.971 once the gold set is reweighted to match the corpus's class mix. That result establishes ordinal robustness to classifier choice. It does not establish that the lambda magnitudes this blend consumes are stable under a future retrain with a recall-corrected engine, and this document does not claim that it does. Proceeding on the current committed lambda posteriors is the scope owner's decision, recorded in the adoption note above. The retrain-sensitivity of the magnitudes is the parked tail risk carried in Section 7.

### 4.4 Fold rule

The adopted method uses a sum-prevalence fold on the data axis. A rolled-up child's corrected incidence rate adds to its parent's, since incidents accumulate (`lambda_folded[parent] += lambda[child]`, applied per draw before the z-score). This is the production rule `engine/decide/blend.py` applies. The 0.1 interim rank-space blend used a max-severity placeholder instead. The vote axis still folds by minimum rank. The merged entry inherits the stronger (lower) rank of parent or child. Under the committed posterior samples, summing prevalence lifts LLM01 (which absorbs Cross-Modal Safety Bypass) and LLM03 (which absorbs Artifact Promotion Trust Failure) further up the data axis than max-severity would. The 0.25 data weight and the vote's three-quarters weight damp how far that lift moves either entry in the final tiers (Section 5).

### 4.5 Frame-blind entries: the data term is dropped

Three entries are frame-blind. The corpus cannot estimate their recall, so their raw data-axis signal has no reliable measurement behind it: LLM04 Data and Model Poisoning, LLM08 Vector and Embedding Weaknesses, and LLM10 Unbounded Consumption. The adopted method drops the data term for these three entries and places them by the vote alone, shifting their weights from 0.75/0.25 to 1.0/0.0. The 0.1 interim blend instead carried a soft data rank forward for these entries.

Dropping the data term changes LLM04's placement: under the drop rule it lands at position 5, one place lower than the position 4 it would reach if its data term were kept at 0.25 weight. The alternative is shown here for comparison. The drop rule is the one this document's ranking (Section 5) uses. The frame-blind entries' corrected rates still enter the all-ten z-score normalization for the measurable entries even after their own data term drops, and that carries a small, disclosed effect: P(top-k) shifts by about 0.03 under an alternative population that excludes them from the normalization entirely.

## 5. The adopted ranking

The adopted ranking comes from `engine/decide/blend.py`, applied to the committed posterior samples. 16,000 lambda draws from the corrected-incidence model are paired, seed `20260520`, with 16,000 vote-rank draws, folded to the ten incumbents (Section 4.4), z-scored per axis (Section 4.3), and blended at 0.75 vote and 0.25 data per draw, with the data term dropped for the three frame-blind entries (Section 4.5). Sorting each draw and aggregating over all 16,000 gives a mean position, a 90 percent position interval, and the probability each entry lands in the top 3 or top 5, per entry. The result is a distribution over positions for each entry, and the ten incumbents fall into three tiers.

| Tier | Entry | Risk | Mean position | 90% interval | P(top 3) | P(top 5) |
|---|---|---|---|---|---|---|
| Co-leading pair | LLM01 | Prompt Injection | 1.6 | 1-3 | 0.99 | 1.00 |
| Co-leading pair | LLM02 | Sensitive Information Disclosure | 1.8 | 1-4 | 0.95 | 1.00 |
| Tied band | LLM06 | Excessive Agency | 3.6 | 2-5 | 0.48 | 0.97 |
| Tied band | LLM03 | Supply Chain | 4.0 | 2-6 | 0.35 | 0.92 |
| Tied band | LLM04 | Data and Model Poisoning | 4.6 | 2-6 | 0.20 | 0.76 |
| Unordered tail | LLM10 | Unbounded Consumption | 5.7 | 4-8 | 0.05 | 0.33 |
| Unordered tail | LLM09 | Misinformation | 8.1 | 6-10 | <0.01 | 0.02 |
| Unordered tail | LLM07 | Hidden Context Exposure | 8.3 | 6-10 | <0.01 | 0.01 |
| Unordered tail | LLM08 | Vector and Embedding Weaknesses | 8.4 | 6-10 | <0.01 | <0.01 |
| Unordered tail | LLM05 | Improper Output Handling | 8.9 | 7-10 | <0.01 | <0.01 |

**The co-leading pair.** Prompt Injection and Sensitive Information Disclosure hold the top of the field, each a near-certain top-three entry. The two blend scales disagree on which one leads. The probabilistic method orders Prompt Injection first. The rank-space lens below orders Sensitive Information Disclosure first. Their position intervals overlap across the full range. This document reports the pair as co-leading and does not assign a method-independent first place.

**The tied band.** Excessive Agency, Supply Chain, and Data and Model Poisoning form a middle band with overlapping position intervals. Excessive Agency rises three places from its published position, the one entry in this ranking whose movement is stated as a numbered mover. Data and Model Poisoning is frame-blind. The drop rule (Section 4.5) places it at position 5 within this band on the vote alone.

**The unordered tail.** Unbounded Consumption tops the tail. It is frame-blind and vote-placed (Section 4.5), and it reaches the top five in about a third of the draws, the closest any tail entry comes to the band above it. Misinformation, Hidden Context Exposure, Vector and Embedding Weaknesses, and Improper Output Handling each reach the top five in fewer than one draw in twenty. No table or sentence in this document assigns a numbered rank or a "+N" move to any of these five entries. They are reported as a group, in the order the point estimate happens to sort them. That order does not indicate a claim about their relative ranking.

A simpler rank-space blend, which uses only the order of each witness and discards the posterior magnitudes, gives nearly the same bulk order. Kendall's tau between the two methods across the ten incumbents is about 0.87. The one place the two methods disagree is the top. This document reports the pair above as co-leading and does not assign either one a crisp first or second place.

Reproducibility: the blend computation is `engine/decide/blend.py`. The committed golden output (order, tiers, position statistics, seed) is `projects/owasp-llm/cycles/2026/blend/blend_golden.json`. The input manifest is `projects/owasp-llm/cycles/2026/blend/blend_manifest.json`.

## 6. What the ranking shows

Three patterns carry the result.

Agreement anchors the top tier. Prompt Injection and Sensitive Information Disclosure co-lead because both witnesses place them at the top of the field, and their position intervals overlap through the full range of the 16,000 draws. Where the vote and the incident data agree this closely, confidence is highest, and the co-leading pair is the safest position in this ranking to act on.

The vote leads where the data sits mid-pack. Prompt Injection is the clearest case. The expert vote ranks it first among 20 candidates. The incident record ranks it twelfth. The three-quarters vote weight keeps it inside the co-leading pair. Without that weight, it would drop into the tied band. The 0.75/0.25 split is doing what it was set up to do. The data tugs, and the vote holds the position.

The data flags what the vote underweights. Misinformation is the widest disagreement on the board. The incident record ranks it near the top of the field. The expert vote ranks it near the bottom. The engine's concordance flag puts the probability the two signals disagree at 99 percent. That number measures disagreement between the two witnesses. It does not measure how underrated the risk is, and the incident signal behind it rests on the ai-harm stratum, where the corpus has no direct precision measurement (Section 4.5, Section 7). Misinformation sits in the unordered tail alongside four other entries, each reaching the top five in fewer than one draw in twenty. The tail is reported as a group, so this disagreement is a data-witness finding about Misinformation's incident signal. It does not make a claim about Misinformation's position inside the tail.

Two structural caveats sit under the tiers. Three entries are frame-blind, LLM04, LLM08, LLM10 (Section 4.5). The adopted method drops their data term and places each by the vote alone. The headline agreement statistic, weighted Cohen's kappa 0.20 with a 90 percent interval of -0.16 to 0.57 that crosses zero, says the vote and the data are only in fair, uncertain agreement overall. The tiers reconcile two imperfect witnesses. They are not a verdict from either one.

## 7. Residual and tail risk

This section names what remains uncertain under the adopted method.

**Transform provenance.** The session that first produced this order was never committed to version control. The linear data-axis z-score (Section 4.3) reproduces that order under a defended, disclosed assumption; it does not recover ground truth for the original computation. The committed knob-sweep in `engine/decide/blend_prototype_reference.py` shows a log transform swaps two entries, both inside the unordered tail (Section 5), so no ordered claim in this report depends on the choice. Disclosed and accepted.

**Manifest and golden self-certification.** The integrity checks, an input-hash manifest, a golden output, and a cross-implementation anchor against the reconstruction prototype, close the accidental-corruption and naive-tamper classes. They do not defend against a commit-access adversary, which sits outside the threat model of an internal research tool. This gap is disclosed here, without additional mitigation.

**The parked retrain tail risk.** A future recall-corrected engine retrain will change the lambda posteriors this blend consumes, and a large enough magnitude shift could move entries between tiers, most plausibly inside the tied band and the tail where the position intervals already overlap. This risk is rated critical in impact and currently unlikely in probability, on the strength of the RARR result recorded in `projects/owasp-llm/cycles/2026-rarr/results/RARR-conclusion.md`. Four independent tests show the incidence ranking is robust to classifier choice, at Spearman rho 0.918 against held-out adjudicated truth, rising to 0.971 once the gold set is reweighted to the corpus class mix. That result is ordinal robustness to which classifier labels the corpus. This document does not read it as a demonstration that the magnitudes are stable under a future retrain with corrected recall. Proceeding on the current committed lambda posteriors is the scope owner's decision, recorded in `docs/decisions/2026-07-05-probabilistic-blend-adoption.md`. The retrain-sensitivity of the magnitudes stays parked here, for re-evaluation before the next engine-upgrade cycle runs.

**Classifier recall variance.** Measured recall runs from roughly 2 to 49 percent across categories. The Bayesian model corrects each entry's count for its own measured precision and recall, and the correction carries the classifier's uncertainty forward. Low-recall entries receive a larger, wider correction, which is part of why the tied band and the tail overlap as much as they do.

**Frame-blind entries.** LLM04, LLM08, and LLM10 draw their incident signal from a single stratum the corpus cannot cross-check. The adopted method's production rule drops their data term and places them by the vote alone (Section 4.5). The keep-at-0.25 alternative is recorded there for comparison. This document does not carry it forward here as an open question.

**Agreement is weak.** The vote and the data agree only weakly overall: weighted Cohen's kappa 0.20, 90 percent interval -0.16 to 0.57, crossing zero. Section 6 reads this alongside the tiers. This section carries it forward as a standing limitation and does not treat it as resolved.

## 8. Reproducibility

- Both-axis ranks with intervals: `projects/owasp-llm/cycles/2026/results/rank_comparison_report.md`
- Agreement statistic and flags: `projects/owasp-llm/cycles/2026/results/concordance.json`
- Taxonomy and rollup crosswalk: `projects/owasp-llm/cycles/2026/taxonomy/taxonomy.json`
- Inference diagnostics: `projects/owasp-llm/cycles/2026/infer/inference_summary.json`
- Adopted blend computation: `engine/decide/blend.py`. Golden output and input manifest: `projects/owasp-llm/cycles/2026/blend/blend_golden.json`, `projects/owasp-llm/cycles/2026/blend/blend_manifest.json`
- Adoption decision and reconstruction provenance: `docs/decisions/2026-07-05-probabilistic-blend-adoption.md`, `docs/provenance/2026-07-05-probabilistic-blend-reconstruction.md`
- RARR classifier-robustness result (Section 7): `projects/owasp-llm/cycles/2026-rarr/results/RARR-conclusion.md`
- Planned engine upgrade whose recall correction is the parked retrain tail risk (Section 7): `claudedocs/engine-upgrade-runpod.md`

---

## 9. Gamma.app prompt: TED-style presentation of the ranking and method

Paste the block below into Gamma.app (Create with AI, "Paste in text" or "Generate"). It is written to produce a TED-style talk arc that teaches the data-driven method as a primer, then reveals the ranking and its surprises with intellectual honesty.

> **Build a TED-style presentation deck. Title it "What the Incidents Say: Stress-Testing the 2026 LLM Top 10." Audience: security leaders, AI engineers, and risk owners who know the OWASP/GenAI LLM Top 10 by name but have never seen it checked against real incident data. Tone: a single-argument TED talk. One throughline, a clear tension, an honest reveal, no jargon walls, no vendor pitch. Confident and plain-spoken, with a researcher's honesty about what is not yet known. Length: 15 cards. Each card has a short, spoken-style headline and 2 to 4 tight supporting lines or a simple visual, never a wall of text. Use a clean editorial style, high contrast, one accent color, simple diagrams and ranking tables over stock imagery. Every number below is real and must appear exactly as given.**
>
> **Card 1, Title.** "What the Incidents Say: Stress-Testing the 2026 LLM Top 10." Subtitle: "When expert belief meets the incident record, the surprises are not where you'd expect." Presenter line: Rock Lambros.
>
> **Card 2, The hook.** Open on a contradiction. Practitioners rank Prompt Injection the number one LLM risk. A corpus of 7,714 real incidents ranks it twelfth. Big question on the card: "Who is right, the experts or the evidence?"
>
> **Card 3, Why this matters.** These lists steer security budgets, audits, and controls. If the ranking is only a vote, we defend against what we fear, not against what is actually happening. The fix is not to replace the experts. It is to weigh them against the record.
>
> **Card 4, Two witnesses.** Introduce the method as a courtroom of two witnesses. Witness one: the crowd, about 29 practitioners scoring each risk 1 to 5 on importance. Witness two: the corpus, 7,714 GenAI agentic incidents. Neither is perfect. Both get heard.
>
> **Card 5, Primer part 1, the crowd.** Explain the vote simply. Each expert ranks the risks by importance. We aggregate to a median rank per risk with an uncertainty band. This is consensus, and consensus is valuable, but consensus can be an echo of last year's headlines.
>
> **Card 6, Primer part 2, the data pipeline.** A simple four-step diagram. 1) Label every incident with a two-stage classifier (rules, then an LLM adjudicator). 2) Calibrate the classifier against a hand-adjudicated gold set to learn its precision and recall. 3) Infer each risk's true underlying rate with a Bayesian model. 4) Rank by that rate. Note the honesty checks: the model converged cleanly (R-hat under 1.001, zero divergences).
>
> **Card 7, The honest problem.** The data witness has blind spots, and we say so out loud. The classifier under-detects unevenly, catching as little as 2 percent and at most 49 percent of cases depending on the risk. So the raw counts undercount, and three risks cannot be measured at all yet. A witness with blind spots still testifies. We just weight the testimony.
>
> **Card 8, The reconciliation.** Show the rule on one card: each risk's final position blends 0.75 of the expert vote with 0.25 of the incident data, in score space, drawn many times over to produce a range of likely positions. The crowd leads at three-quarters weight because the list is a consensus product. The data enters at a quarter weight as a corrective, strong enough to move a risk a tier, not strong enough to overturn the consensus on one noisy corpus. Add one line on rollups: where the group merged a narrow risk into a broader one, the merged risk's incident count sums both.
>
> **Card 9, The reveal, three tiers.** Show the ten risks grouped into three tiers: a co-leading pair at the top, a tied band in the middle, and an unordered tail. Render it as a position chart: each risk as a dot at its mean position across 16,000 posterior draws, with a bar spanning its 5th-to-95th-percentile position, color-grouped by tier. Use this exact data.
>
> | Tier | Risk | Mean position | 90% interval |
> |---|---|---|---|
> | Co-leading pair | Prompt Injection | 1.6 | 1-3 |
> | Co-leading pair | Sensitive Information Disclosure | 1.8 | 1-4 |
> | Tied band | Excessive Agency | 3.6 | 2-5 |
> | Tied band | Supply Chain | 4.0 | 2-6 |
> | Tied band | Data and Model Poisoning | 4.6 | 2-6 |
> | Unordered tail | Unbounded Consumption | 5.7 | 4-8 |
> | Unordered tail | Misinformation | 8.1 | 6-10 |
> | Unordered tail | Hidden Context Exposure | 8.3 | 6-10 |
> | Unordered tail | Vector and Embedding Weaknesses | 8.4 | 6-10 |
> | Unordered tail | Improper Output Handling | 8.9 | 7-10 |
>
> Caption: "Belief and evidence, blended into a distribution across many draws. Excessive Agency rises three places into the tied band. The tail moves as a group." Visual direction: shade the three tiers as distinct bands on the chart, and do not print a rank number or a move value beside any tail entry.
>
> **Card 10, Surprise one, the underrated risk.** Misinformation. Voters rank it near the bottom, thirteenth of twenty. The incident record ranks it near the top, second. The engine's concordance flag marks this as the sharpest disagreement between the two witnesses on the board. That flag measures how far apart the two signals sit; it does not measure how underrated the risk is. It sits inside the unordered tail on the blend, one of five entries reported together as a group, because the data carries only a quarter weight. The evidence still says this is the risk to watch most closely as the incident record grows.
>
> **Card 11, Surprise two, the overweighted fear.** Prompt Injection. The crowd's number one. Mid-pack in the incident record at twelfth. It co-leads the blend with Sensitive Information Disclosure, carried by the three-quarters vote weight and a folded-in risk that lifts its data signal. The gap is a reminder that fear and frequency measure different things.
>
> **Card 12, Where they agree.** Agreement is signal too. Prompt Injection and Sensitive Information Disclosure both sit at the top of the field, though the two blend scales split on which leads: the probabilistic method used here orders Prompt Injection first, a rank-space alternative orders Sensitive Information Disclosure first, and their position intervals overlap enough that the deck calls them a co-leading pair. Improper Output Handling and Vector and Embedding Weaknesses sit low on both. Where the expert and the record agree, confidence is highest and action is easiest to justify.
>
> **Card 13, What we cannot see yet.** Name the blind spots plainly. Three risks are frame-blind: Data and Model Poisoning, Vector and Embedding Weaknesses, and Unbounded Consumption. The corpus cannot measure their data signal, so the method places each by the expert vote alone. Naming what you cannot measure is itself part of the method.
>
> **Card 14, An uncertainty-aware ranking.** Be honest about certainty. The overall agreement between vote and data is only fair, and its confidence interval crosses zero. The ranking reports a distribution over each risk's position, a co-leading pair, a tied band, and an unordered tail, so a coin-flip placement reads as a coin flip on the page. A future engine retrain that corrects the classifier's recall could still move these positions. That risk is tracked openly in the methodology record, alongside the ranking delivered today.
>
> **Card 15, The takeaway.** One closing idea: build defenses on the blend of belief and evidence, and be loud about the gaps. Final line on the card: "Rank what you fear against what actually happened. Then act on both." End with a quiet call to action for the audience to check their own risk lists against their own incident data.
>
> **Design notes for the whole deck:** keep one accent color across all cards. Use real ranked tables and the four-step pipeline diagram as the two anchor visuals. Avoid clip art and stock photos of hackers in hoodies. Prefer whitespace and one strong idea per card. Headlines should sound like spoken lines, not slide labels.
