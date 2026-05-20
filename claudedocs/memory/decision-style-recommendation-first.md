---
name: decision-style-recommendation-first
description: How Rock wants decision points presented — option menu WITH a marked recommendation
metadata:
  node_type: memory
  type: feedback
  originSessionId: b8a2bb68-f524-48d7-b5f9-542b3a5013f2
---

At decision points, present the multiple-choice option menu (AskUserQuestion) AND mark the option you would choose with "(Recommended)" as the first option, accompanied by evidence-grounded reasoning and the strongest counterargument against your own pick.

**Why:** During the HANDOFF.md spec revision (2026-05-19) Rock first rejected bare option menus (no recommendation), then after a prose-only recommendation explicitly said: "I still want the multiple choice options you were presenting before just label (Recommended) next to the one you would chose." So neither extreme works — not a bare menu, not recommendation-only prose. He wants both: the structured choices plus a clearly flagged recommendation he can accept or override.

**How to apply:** Use AskUserQuestion. Put the recommended option first with "(Recommended)" appended to its label. In the surrounding message give the reasoning and the strongest counterargument (matches the global CLAUDE.md posture: brutally honest, evidence over assertion, counterargument before being asked). One decision at a time. He frequently just says "go with recommendation," so make the recommended option genuinely the one you'd defend.
