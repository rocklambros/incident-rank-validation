# Decision: adopt the probabilistic blend (2026-07-05)

**Decision.** The project leads adopt the probabilistic blend as the analytical method for the 2026 blended Top-10 LLM ranking. Rock Lambros, as Co-Lead, made this an executive decision on 2026-07-05 to finalize the exploratory analysis.

**Scope.** This is an authorial method choice for an exploratory report. It is NOT an OWASP institutional endorsement of the method or the ranking. The working group's standing pre-publication review applies. Every standing disclosure holds: single-author gold set, interim reviewers, non-publishable status, weak agreement (Cohen's kappa about 0.20, interval crossing zero). The public preprint describes the method as the authors' adoption and does not claim working-group sign-off.

**Basis.** The method reproduces the recorded order from the committed posterior samples (engine/decide/blend.py, cross-checked against the committed reconstruction). Proceeding on the current committed lambda is this decision; retrain-sensitivity of the magnitudes is a parked tail risk recorded in the methodology doc.
