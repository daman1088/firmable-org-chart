"""Company-level context pass.

Some title qualifiers only make sense inside one company: a division or
product line ("Payments", "Aged Care", "SMB") repeated across several people's
titles is an organisational scope, but no global lexicon can list every
company's units. A qualifier that the title parser could not explain becomes a
`unit` scope when at least MIN_SUPPORT distinct people in the same company
carry it.
"""

from __future__ import annotations

import dataclasses
from collections import Counter

from orgchart.normalization.parser import ParsedTitle

MIN_SUPPORT = 3


def apply_company_context(parsed: list[ParsedTitle]) -> list[ParsedTitle]:
    """Return copies of `parsed` with recurring unexplained qualifiers promoted to unit scopes."""
    support = Counter()
    for p in parsed:
        for q in {q.strip().lower() for q in p.unresolved_qualifiers}:
            support[q] += 1
    recurring = {q for q, n in support.items() if n >= MIN_SUPPORT}

    out = []
    for p in parsed:
        if p.scope is None and recurring:
            hit = next((q for q in p.unresolved_qualifiers if q.strip().lower() in recurring), None)
            if hit:
                n = support[hit.strip().lower()]
                p = dataclasses.replace(
                    p,
                    scope=hit.strip(),
                    scope_type="unit",
                    explanation=[*p.explanation, f"scope: unit '{hit.strip()}' (qualifier shared by {n} people "
                                                 f"in this company)"],
                    rules_fired=[*p.rules_fired, "company.recurring_qualifier"],
                )
        out.append(p)
    return out
