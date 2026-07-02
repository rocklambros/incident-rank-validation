"""F8 strata-disjoint guard: prevents incidence double-count (U2-6).

The real incidence double-count vectors this guard targets:
  (a) A single incident_id appearing in two different strata' populations.
  (b) A stratum listed twice in entry_strata[e] (doubles the Σsize exposure term).
  (c) Overlapping stratum populations at the population level.

An entry SPANNING multiple strata is LEGITIMATE AND COMMON — 9/20 real OWASP-LLM
entries span both 'security' and 'ai-harm'.  This guard NEVER raises just because
len(entry_strata[e]) > 1.
"""
from __future__ import annotations

__all__ = ["StrataOverlapError", "check_strata_disjoint"]


class StrataOverlapError(ValueError):
    """Raised when strata populations overlap, creating an incidence double-count risk.

    This is a named error (not a plain ValueError) so callers can catch it
    specifically — e.g. to surface it as a pipeline failure rather than a crash.
    """


def check_strata_disjoint(
    labeled_incidents: list[dict[str, object]],
    entry_strata: dict[str, tuple[str, ...]],
) -> None:
    """Assert no incidence double-count across strata (F8 guard, U2-6).

    Four assertions (all must hold):

    1. **Global incident-disjointness**: no incident_id appears in two different
       strata' populations.  Incident ids are canonicalized (stripped) before
       building the sets.
    2. **No stratum repeats within entry_strata[e]** for any entry.  A stratum
       listed twice would double the Σsize exposure term at concordance.py:84.
    3. **Stratum populations pairwise disjoint** — this is assertion 1 restated at
       the population level; enforced by the same per-incident check.
    4. **Empty/absent stratum population for a multi-stratum entry = FAILURE**.
       If entry_strata[e] declares a stratum but that stratum has zero incidents,
       the Σsize term is malformed (a zero-population stratum with a declared
       exposure).  Single-stratum entries are exempt (no Σ involved).

    Critical anti-false-positive property: an entry spanning two strata with
    fully *disjoint* incident populations passes all four assertions and does NOT
    raise.

    Args:
        labeled_incidents: rows from labeled_incidents.json (or an equivalent
            in-memory list), each containing at minimum ``incident_id`` and
            ``stratum`` keys.  An empty list is accepted (no incidents → no
            violations detectable; guard is a no-op for assertions 1/3 and 4
            only triggers for multi-stratum entries that exist in entry_strata).
        entry_strata: mapping from entry_id to a sorted tuple of stratum names,
            as built by the pipeline builders.

    Raises:
        StrataOverlapError: on any of the four violation conditions above.
    """
    # --- Pass 1: build stratum → incident_set and check global disjointness ---
    stratum_incident_sets: dict[str, set[str]] = {}
    incident_to_stratum: dict[str, str] = {}

    for item in labeled_incidents:
        # Canonicalize identically to _build_counts_from_labeled and _build_strata:
        # strip incident_id whitespace (for dedup correctness) but do NOT strip the
        # stratum field (so the key space matches the pipeline's unstripped strata).
        raw_iid = str(item.get("incident_id", "")).strip()
        stratum = str(item.get("stratum", "default"))  # no .strip() — match pipeline

        # Register the stratum even for blank-id rows so assertion 4 sees the same
        # stratum set as the pipeline's exposure term (_build_counts_from_labeled
        # counts blank-id rows toward stratum_doc_counts, so those strata appear in
        # the strata tuple and therefore in entry_strata tuples passed to this guard).
        if stratum not in stratum_incident_sets:
            stratum_incident_sets[stratum] = set()

        if not raw_iid:
            # Blank incident_id: the pipeline counts this row toward the stratum
            # population but there is no unique id to check for disjointness.
            # Stratum already registered above; skip per-incident assertions.
            continue

        stratum_incident_sets[stratum].add(raw_iid)

        # Assertion 1 / 3: each incident_id must map to exactly one stratum.
        prior = incident_to_stratum.get(raw_iid)
        if prior is not None and prior != stratum:
            raise StrataOverlapError(
                f"incident {raw_iid!r} appears in stratum {prior!r} AND stratum "
                f"{stratum!r}: strata populations are not disjoint — this creates "
                "an incidence double-count in the concordance Σsize computation "
                "(concordance.py line 84)."
            )
        incident_to_stratum[raw_iid] = stratum

    # --- Pass 2: per-entry checks on entry_strata ---
    for eid, strata_tuple in entry_strata.items():
        # Assertion 2: no stratum may appear more than once in entry_strata[e].
        seen: set[str] = set()
        for s in strata_tuple:
            if s in seen:
                raise StrataOverlapError(
                    f"entry {eid!r}: stratum {s!r} appears more than once in "
                    f"entry_strata = {strata_tuple!r}; a repeated stratum would "
                    "double-count the Σsize exposure term."
                )
            seen.add(s)

        # Assertion 4: multi-stratum entries must have non-empty populations.
        # Single-stratum entries are exempt (the Σsize just equals that one size).
        # Use key-existence (s in stratum_incident_sets) rather than set-emptiness
        # (not stratum_incident_sets.get(s)) so that strata whose only rows have
        # blank incident_ids still pass — those rows are counted by the pipeline's
        # stratum_doc_counts / Σsize term and must not false-positive here (U2 fix #3).
        if len(strata_tuple) > 1:
            for s in strata_tuple:
                if s not in stratum_incident_sets:
                    raise StrataOverlapError(
                        f"entry {eid!r} spans strata {strata_tuple!r} but stratum "
                        f"{s!r} has no incidents in labeled_incidents: the Σsize "
                        "term for this stratum is malformed (declared exposure "
                        "with zero population)."
                    )
