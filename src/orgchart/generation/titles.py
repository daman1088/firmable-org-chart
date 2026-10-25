"""Canonical (clean) job titles for tree nodes.

The canonical title is what the person's title "really" is; the observed title
is derived from it by orgchart.generation.noise. Rendering at a different level
(`level=`) is used to simulate stale and inflated titles.
"""

from __future__ import annotations

import re

import numpy as np

from orgchart.generation.catalog import (
    BOARD_TITLES,
    CHIEF_OF_STAFF_TITLES,
    DEPARTMENTS,
    EA_TITLES_CEO,
    EA_TITLES_EXEC,
    Archetype,
)
from orgchart.generation.rand import chance, pick, weighted_pick
from orgchart.generation.structure import CompanyTree, Node
from orgchart.ladder import Level

_PLACEHOLDER = re.compile(r"\{(role|spec|func)\}")


def exec_abbreviation(title: str) -> str:
    """'Chief Financial Officer' -> 'CFO'; 'Co-Founder & CTO' -> 'CTO'."""
    match = re.search(r"\b(C[A-Z]{1,3}O)\b", title)
    if match:
        return match.group(1)
    words = [w for w in re.findall(r"[A-Za-z]+", title) if w[0].isupper()]
    if words and words[0] == "Chief":
        return "".join(w[0] for w in words)
    return title


def _fill(template: str, role: str, spec: str | None, func: str) -> str:
    if spec is None:  # no specialty: drop "..., {spec}" qualifiers, otherwise use the function name
        template = re.sub(r"(, | - )\{spec\}", "", template)
    values = {"role": role, "spec": spec or func, "func": func}
    text = _PLACEHOLDER.sub(lambda m: values[m.group(1)], template)
    # Collapse accidental duplications like "Engineering Engineering Manager".
    text = re.sub(r"\b(\w+) \1\b", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def _with_scope(rng: np.random.Generator, title: str, scope: str | None, p: float) -> str:
    if not scope or not chance(rng, p) or scope in title:
        return title
    return f"{title} - {scope}" if ", " in title else f"{title}, {scope}"


def canonical_title(
    rng: np.random.Generator,
    node: Node,
    nodes: list[Node],
    archetype: Archetype,
    *,
    level: int | None = None,
    allow_decorations: bool = True,
) -> str:
    """Render a title for `node`. Parents must already have titles (BFS order)."""
    rt = node.role_type
    if rt == "ceo":
        return weighted_pick(rng, dict(archetype.ceo_titles))
    if rt == "board":
        return pick(rng, BOARD_TITLES)
    if rt == "chief_of_staff":
        return pick(rng, CHIEF_OF_STAFF_TITLES)
    if rt == "assistant":
        parent = nodes[node.parent]
        if parent.role_type == "ceo":
            return pick(rng, EA_TITLES_CEO)
        return pick(rng, EA_TITLES_EXEC).format(exec=exec_abbreviation(parent.title))

    spec = DEPARTMENTS[node.dept]
    lvl = node.level if level is None else level
    specialty = node.specialty
    if node.role_noun is None:
        roles = spec.specialties.get(specialty) if specialty else None
        node.role_noun = pick(rng, roles or next(iter(spec.specialties.values())))
    role = node.role_noun

    if rt == "head":
        lvl = max(Level.LEAD, min(lvl, Level.C_LEVEL))
        title = _fill(pick(rng, spec.head_templates(lvl)), role, spec.func, spec.func)
        if allow_decorations and lvl == Level.C_LEVEL and node.dept == "engineering":
            ceo_title = nodes[0].title
            if "Founder" in ceo_title and chance(rng, 0.4):
                title = "Co-Founder & CTO"
    elif rt == "manager" and lvl >= Level.LEAD:
        lvl = min(lvl, Level.SVP)
        title = _fill(pick(rng, spec.mgr_templates(lvl, specialty)), role, specialty, spec.func)
        if allow_decorations:
            title = _with_scope(rng, title, node.scope, 0.7)
    else:  # IC, or a manager rendered below the manager track (stale title)
        lvl = max(Level.INTERN, min(lvl, Level.LEAD))
        title = _fill(pick(rng, spec.ic_templates(lvl, specialty)), role, specialty, spec.func)
        if allow_decorations:
            title = _with_scope(rng, title, node.scope, 0.15)

    if allow_decorations and rt in ("head", "manager") and (node.level or 0) >= Level.DIRECTOR and chance(rng, 0.02):
        title = f"Interim {title}"
    return title


def assign_canonical_titles(rng: np.random.Generator, tree: CompanyTree, archetype: Archetype) -> None:
    """Title every node: employees top-down (EA titles depend on their exec), then board profiles."""
    for node in tree.bfs_order():
        node.title = canonical_title(rng, node, tree.nodes, archetype)
    for node in tree.nodes:
        if node.role_type == "board":
            node.title = canonical_title(rng, node, tree.nodes, archetype)
