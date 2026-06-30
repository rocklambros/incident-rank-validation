# Blended GenAI/OWASP LLM Top 10 (2026): Methodology, Justification, and Ranking

Owner: Rock Lambros (rock@rockcyber.com)
Version: 0.1 interim (rank-space blend), 2026-06-30
Status: Provisional. The ranking in Section 5 uses an interim rank-space blend. The final blend scale waits on the recall-corrected engine upgrade. See Section 7.

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

Decision: Scope 1 is the headline ranking. Scope 3 is the rollup audit reported beside it. Scope 2 is rejected.

Reasons:

- Scope 1 credits each incident to the entry that absorbed it. That is the honest accounting given the published list, and it matches the current state of `main`.
- Scope 3 reported beside it shows whether a child carries a distinct incident cluster the parent does not absorb. `engine/decide/rollup.py` already runs this distinct-cluster test.
- Scope 2 would change which risks appear on the list. That is a membership claim, not an ordering claim, and the incident data is too weak to carry it. Weighted Cohen's kappa is 0.203 with a 90 percent interval of -0.16 to 0.57, which crosses zero. Re-selecting the list on that evidence would overreach. For the record, under a rank-space blend Scope 2 would have promoted two new candidates (Persistent Memory Poisoning, MCP Tool Interface Exploitation) and re-admitted one rolled-up child (Cross-Modal Safety Bypass), ejecting Misinformation, Vector and Embedding Weaknesses, and Improper Output Handling. The working group already weighed those membership calls. The blend respects them.

### 4.2 Blend weights

Human 0.75, data 0.25. The list is a consensus product, so practitioner judgment anchors it. The data enters as a structured corrective at one quarter weight, strong enough to move an entry a tier when the gap is large, not strong enough to overturn the consensus on a single noisy corpus.

### 4.3 Blend scale: interim and deferred

The headline ranking below uses a rank-space blend: convert each axis to a rank, compute `0.75 x vote_rank + 0.25 x lambda_rank` (lower is better), and sort. Rank space is the most assumption-light blend and needs only the published median ranks.

Two richer blend scales are deferred until the recall-corrected engine upgrade lands (`claudedocs/engine-upgrade-runpod.md`):

- Normalized-score space. Z-score each axis on its native scale (vote worth and recall-corrected incidence) and average. This keeps magnitude and gap, which rank space discards.
- Probabilistic. A weighted Plackett-Luce or Bayesian blend that propagates both sides' uncertainty into a distribution over each entry's final position, so a coin-flip placement reads as a coin flip rather than a hard number.

Both richer scales operate on lambda magnitudes and their uncertainty. The recall-corrected retrain will change those magnitudes. Running score-space or probabilistic blends now would anchor on numbers set to move. The rank-space result here is a checkpoint, not the final order.

### 4.4 Fold rule

The interim fold uses a max-severity rule: the merged entry inherits the stronger (lower) rank of parent or child on each axis. Under current data this lifts only LLM01, whose data rank improves from 12 to 9 once Cross-Modal Safety Bypass folds in. The other three parents already out-rank their children on both axes, so their positions hold.

The production fold should sum prevalence on the data axis, since incidents add. A sum-prevalence fold pushes LLM01 and LLM03 up the data axis harder than max-severity does. The 0.25 data weight damps that effect, which is why the interim order is stable, but the rule choice is recorded as open and will be settled with the upgrade.

### 4.5 Frame-blind entries: open

Three entries are frame-blind. The corpus cannot estimate their recall, so their data rank is soft: LLM04 Data and Model Poisoning, LLM08 Vector and Embedding Weaknesses, LLM10 Unbounded Consumption. Two handling options remain open:

- Use the soft data rank as-is (current interim behavior).
- Drop the data vote for these three and renormalize their weight to 100 percent human, so a blind axis never moves a placement.

This choice changes mid-list and lower-list positions and will be settled with the upgrade.

## 5. The interim ranking

Scope 1 fold, rank-space blend, weights 0.75 human and 0.25 data. Vote rank and data rank are positions in the full 20-candidate field, taken from `results/rank_comparison_report.md`. The blend orders the 10 incumbents by those positions. Lower blend score is higher priority.

| # | Entry | Risk | Vote rank | Data rank | Blend | Read |
|---|---|---|---|---|---|---|
| 1 | LLM02 | Sensitive Information Disclosure | 2 | 2 | 2.00 | Crowd and data agree. Top tier on both. |
| 2 | LLM01 | Prompt Injection | 1 | 12 (9 folded) | 3.00 | The crowd's top fear. Data ranks it mid-pack. Stays second because the vote leads and the Cross-Modal Safety Bypass fold lifts its data signal. |
| 3 | LLM06 | Excessive Agency | 4 | 7 | 4.75 | Both signals place it upper tier. |
| 4 | LLM04 | Data and Model Poisoning | 6 | 4 | 5.50 | Data leans higher than the vote. Frame-blind, so the data rank is soft. |
| 5 | LLM03 | Supply Chain | 5 | 9 | 6.00 | Absorbs Artifact Promotion Trust Failure. Vote and data broadly agree. |
| 6 | LLM10 | Unbounded Consumption | 8 | 15 | 9.75 | Vote ranks it above the data. Frame-blind. |
| 7 | LLM07 | Hidden Context Exposure | 11.5 | 6 | 10.12 | New for 2026. The data sees more incidents here than voters expect. |
| 8 | LLM09 | Misinformation | 13 | 2 | 10.25 | The widest gap on the board. The data says it is badly underrated. The vote weight holds it down to eighth. |
| 9 | LLM08 | Vector and Embedding Weaknesses | 11 | 12 | 11.25 | Both signals low. Frame-blind. |
| 10 | LLM05 | Improper Output Handling | 13 | 10 | 12.25 | Both signals low. Absorbs Systemic Insecure Code Generation. |

Scope 3 audit: the parent-only order matches this list. The only difference is LLM01, whose blend reads 3.75 parent-only versus 3.00 folded. It holds second place either way. Folding does not reorder the list under current weights.

### 5.1 Movement from the published order

The previous ranking is the published 2026 order, the LLM01-to-LLM10 numbering on `main`. The blend reorders it as shown. Move is the change from published position to blended position.

| Blended # | Risk | Published # | Move |
|---|---|---|---|
| 1 | LLM02 Sensitive Information Disclosure | 2 | up 1 |
| 2 | LLM01 Prompt Injection | 1 | down 1 |
| 3 | LLM06 Excessive Agency | 6 | up 3 |
| 4 | LLM04 Data and Model Poisoning | 4 | no change |
| 5 | LLM03 Supply Chain | 3 | down 2 |
| 6 | LLM10 Unbounded Consumption | 10 | up 4 |
| 7 | LLM07 Hidden Context Exposure | 7 | no change |
| 8 | LLM09 Misinformation | 9 | up 1 |
| 9 | LLM08 Vector and Embedding Weaknesses | 8 | down 1 |
| 10 | LLM05 Improper Output Handling | 5 | down 5 |

Biggest movers: Improper Output Handling falls five places, Unbounded Consumption rises four, Excessive Agency rises three. The published numbering is treated as the prior importance rank. If the intended baseline is the human-vote-only order instead, the move column changes and this table needs regenerating.

## 6. What the ranking shows

Three patterns carry the result.

Agreement is signal. LLM02 sits first because both witnesses rank it top. Where the crowd and the corpus agree, confidence is highest, and four of the top five entries are agreements or near-agreements.

The crowd leads where the data is mid. Prompt Injection is the clearest case. Practitioners rank it first. The incident record ranks it twelfth of twenty. At 0.75 weight the vote keeps it second. This is the 0.75/0.25 split doing its job: the data tugs, the consensus holds.

The data flags what the crowd underweights. Misinformation is the headline disagreement. Voters place it near the bottom. The corpus places it near the top, and the engine flags this gap at 99 percent probability. The blend still seats it eighth because the data carries only a quarter weight, but the flag is the point. This is the entry to watch, and it is the entry the recall-corrected upgrade is most likely to move.

Two structural caveats sit underneath the numbers. Three entries are frame-blind, so their data rank is provisional by construction. The headline agreement statistic, weighted kappa 0.203 with an interval crossing zero, says the vote and the data are only in fair, uncertain agreement overall. The blend is a reconciliation of two imperfect witnesses, not a verdict from one authority.

## 7. Provisional status and limitations

- The blend scale is interim. Rank space discards magnitude. The score-space and probabilistic blends are deferred to the recall-corrected upgrade and may reorder the mid and lower list.
- The classifier under-detects unevenly. Recall runs from roughly 2 to 49 percent across categories, so the raw counts are a biased undercount and the data rank is skewed until the measurement-error correction lands.
- Frame-blind handling is unsettled (Section 4.5).
- The fold rule is interim max-severity, not production sum-prevalence (Section 4.4).
- The agreement signal is weak and uncertain (kappa interval crosses zero). Treat the blended order as a defensible working ranking, not a settled finding.

## 8. Reproducibility

- Both-axis ranks with intervals: `projects/owasp-llm/cycles/2026/results/rank_comparison_report.md`
- Agreement statistic and flags: `projects/owasp-llm/cycles/2026/results/concordance.json`
- Taxonomy and rollup crosswalk: `projects/owasp-llm/cycles/2026/taxonomy/taxonomy.json`
- Inference diagnostics: `projects/owasp-llm/cycles/2026/infer/inference_summary.json`
- Planned upgrade that supersedes the interim blend scale: `claudedocs/engine-upgrade-runpod.md`

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
> **Card 8, The reconciliation.** Show the rule on one card: final rank = 0.75 times the expert vote plus 0.25 times the data. The crowd leads at three-quarters weight because the list is a consensus product. The data enters at a quarter weight as a corrective, strong enough to move a risk a tier, not strong enough to overturn the consensus on one noisy corpus. Add one line on rollups: where the group merged a narrow risk into a broader one, the merged risk inherits the incidents of both.
>
> **Card 9, The reveal, before and after.** Show the blended Top 10 beside the published 2026 order so the movement is visible. Render it as a before-to-after comparison: published order on the left, blended order on the right, with up and down arrows on the movers and a muted dash for no change. Use this exact data.
>
> | Blended order (new) | Risk | Published order (was) | Move |
> |---|---|---|---|
> | 1 | Sensitive Information Disclosure | 2 | up 1 |
> | 2 | Prompt Injection | 1 | down 1 |
> | 3 | Excessive Agency | 6 | up 3 |
> | 4 | Data and Model Poisoning | 4 | no change |
> | 5 | Supply Chain | 3 | down 2 |
> | 6 | Unbounded Consumption | 10 | up 4 |
> | 7 | Hidden Context Exposure | 7 | no change |
> | 8 | Misinformation | 9 | up 1 |
> | 9 | Vector and Embedding Weaknesses | 8 | down 1 |
> | 10 | Improper Output Handling | 5 | down 5 |
>
> Caption: "Belief and evidence, weighted together. Improper Output Handling falls five places, Unbounded Consumption rises four, Excessive Agency rises three." Visual direction: draw connector lines between the two columns so the audience can trace each risk's move at a glance, color rises in the accent color and falls in a muted gray.
>
> **Card 10, Surprise one, the underrated risk.** Misinformation. Voters rank it near the bottom, thirteenth of twenty. The incident record ranks it near the top, second. The engine flags this gap at 99 percent confidence. It still lands eighth on the blend because the data carries only a quarter weight, but this is the risk the evidence says we are sleeping on.
>
> **Card 11, Surprise two, the overweighted fear.** Prompt Injection. The crowd's number one. Mid-pack in the incident record at twelfth. It holds second place on the blend because the vote leads and a folded-in risk lifts its data signal. The lesson is not that the crowd is wrong. It is that fear and frequency are not the same measurement.
>
> **Card 12, Where they agree.** Agreement is signal too. Sensitive Information Disclosure tops both lists. Improper Output Handling and Vector and Embedding Weaknesses sit low on both. Where the expert and the record agree, confidence is highest and action is easiest to justify.
>
> **Card 13, What we cannot see yet.** Name the blind spots plainly. Three risks are frame-blind: Data and Model Poisoning, Vector and Embedding Weaknesses, and Unbounded Consumption. The corpus cannot yet estimate how often we miss them, so their rank is provisional by construction. Naming what you cannot measure is part of the method, not a footnote.
>
> **Card 14, This is a checkpoint, not a verdict.** Be honest about certainty. The overall agreement between vote and data is only fair, and its confidence interval crosses zero. A recall-corrected upgrade is in progress to fix the undercount and to add uncertainty-aware ranking. Expect Misinformation to move. This ranking is the best current read, and it is built to be revised.
>
> **Card 15, The takeaway.** One closing idea: build defenses on the blend of belief and evidence, and be loud about the gaps. Final line on the card: "Rank what you fear against what actually happened. Then act on both." End with a quiet call to action for the audience to check their own risk lists against their own incident data.
>
> **Design notes for the whole deck:** keep one accent color across all cards. Use real ranked tables and the four-step pipeline diagram as the two anchor visuals. Avoid clip art and stock photos of hackers in hoodies. Prefer whitespace and one strong idea per card. Headlines should sound like spoken lines, not slide labels.
