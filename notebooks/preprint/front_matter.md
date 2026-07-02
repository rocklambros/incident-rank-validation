---
title: "Incident-Data Robustness Analysis of the OWASP Top 10 for LLM Applications (2026): How a Community-Expert Ranking Holds Up Against a Large-Scale LLM Incident Corpus"
shorttitle: "Incident-Data Robustness of the OWASP LLM Top 10 (2026)"
date: "2026"
author:
  - name: "Kyriakos \"Rock\" Lambros"
    affiliation: "OWASP GenAI Security Project — Top 10 for LLM Applications, Co-Lead"
  - name: "Steve Wilson"
    affiliation: "OWASP GenAI Security Project — Top 10 for LLM Applications, Founder & Co-Lead"
numbersections: true
abstract: |
  The OWASP Top 10 for LLM Applications ranks the risks that a community of
  security practitioners judges most important. We ask a narrower question: checked
  against the record of real incidents, does that expert ranking agree with the data?
  We assembled a large-scale corpus of LLM-security incidents — 7,714 snapshotted and
  6,639 labeled against the 20-entry taxonomy — drawn from CVE, GHSA, OSV, and AIAAIC,
  and derived an incident-based ranking with a Bayesian measurement-error model that
  corrects each category's count for classifier precision and recall. The 2026
  candidate list blends the two signals at fixed weights, 0.75 on the expert vote and
  0.25 on the data, so the corpus corrects the consensus without overturning it. The
  agreement between the two rankings is weak: Cohen's κ ≈ 0.20, with a 90% interval
  that crosses zero. The expert ranking is nonetheless robust. A pre-registered
  bake-off of four frontier classifiers returns no winner — none beats the incidence
  floor's balanced accuracy of 0.863 — and a ground-truth check leaves the floor's
  ordering (Spearman ρ = 0.918 against held-out truth) in place. This is an
  exploratory analysis by two working-group members, not the official OWASP release,
  and it does not supersede the official list or process.
---
