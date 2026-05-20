---
name: correctness-over-speed
description: When facing a tradeoff between cheapest/easiest and most-correct, choose most-correct. Rock prefers methodological rigor over shipping speed and rejects framings that bias toward the cheaper option.
metadata:
  type: feedback
---

When presented with a binary "fast/cheap" vs "rigorous/expensive" implementation choice, default to the rigorous path. Do not anchor on "what I already wrote" as a reason to keep weaker work.

**Why:** Stated 2026-05-19 during Plan 1 v2 design for `incident-rank-validation`. Rock said: "I want what will be the most correct, not the easiest or cheapest." Context: I had presented two paths (patch existing NUTS plan vs rescope per R25); Rock rejected the framing as biased toward the cheaper option and asked for the most-correct path regardless of effort. Consistent with HANDOFF §1 working posture (evidence over assertion, surface the strongest counterargument before being asked) and with how he runs methodology work generally.

**How to apply:** When sketching options for Rock, lead with the most-correct option *first*, even if it's the largest or slowest. If a cheaper option is materially worse on methodology or correctness, say so explicitly and name the gap. Do not let prior commitment to a plan version bias the comparison — sunk-cost rationalization is the failure mode to avoid. If "most correct" requires re-spec or re-architecture, propose that openly rather than patching around the gap. When two approaches conflict but the spec calls for both (e.g., HANDOFF §5.4 NUTS-as-primary AND §5.5 robustness twin), build both. Related: [[decision-style-recommendation-first]].
