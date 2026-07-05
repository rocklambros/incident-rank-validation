# Provenance: probabilistic-blend reconstruction (2026-07-05)

The accepted probabilistic blend was first computed in a session whose scratchpad was
never committed. `engine/decide/blend_prototype_reference.py` reconstructs it. Under the
linear data-axis z-score it reproduces the recorded order
(LLM01, LLM02, LLM06, LLM03, LLM04, LLM10, LLM09, LLM07, LLM08, LLM05); under a log
transform it swaps positions 7 and 8 (LLM09 and LLM07), both inside the unordered tail,
so no ordered claim in the report depends on the transform. Linear is a defended
reconstruction on the corrected incidence rates' native additive scale.

Integrity scope: the input manifest and the golden output are commit-anchored, not
signed. This closes the accidental-corruption and naive-tamper classes; a commit-access
adversary is out of scope for this internal tool. The reference module is hash-frozen in
the manifest so its `APPROVED` anchor cannot be edited silently. There is no independent
external oracle: the golden is a regression pin against the reconstruction, not a proof of
correctness.
