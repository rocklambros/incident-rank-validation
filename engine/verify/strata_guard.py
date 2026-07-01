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
        # Canonicalize: strip whitespace from both fields.
        raw_iid = str(item.get("incident_id", "")).strip()
        stratum = str(item.get("stratum", "default")).strip()
        if not raw_iid:
            # Skip rows without an incident_id (defensive; should not occur).
            continue

        if stratum not in stratum_incident_sets:
            stratum_incident_sets[stratum] = set()
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
        if len(strata_tuple) > 1:
            for s in strata_tuple:
                if not stratum_incident_sets.get(s):
                    raise StrataOverlapError(
                        f"entry {eid!r} spans strata {strata_tuple!r} but stratum "
                        f"{s!r} has no incidents in labeled_incidents: the Σsize "
                        "term for this stratum is malformed (declared exposure "
                        "with zero population)."
                    )
