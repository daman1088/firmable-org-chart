"""Observed-view simulation: missing employees and noisy profile fields.

Every transform that fires is recorded as a tag in truth.noise_tags so the
evaluation can break errors down by noise type. Transforms listed under
`holdout_transforms` are only enabled for test-split companies, so test metrics
measure robustness to title styles never seen during training/calibration.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from orgchart.config import CoverageConfig, NoiseConfig
from orgchart.generation.catalog import BOARD_LABELS, DEPARTMENTS, EXECUTIVE_LABELS, VAGUE_TITLES
from orgchart.generation.rand import chance, pick
from orgchart.generation.structure import BOARD, EXECUTIVE, CompanyTree, Node
from orgchart.ladder import Level

# ----------------------------------------------------------------------------- vocabulary
TERM_ABBREVIATIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Senior Vice President", ("SVP",)),
    ("Executive Vice President", ("EVP",)),
    ("Vice President", ("VP",)),
    ("Chief Executive Officer", ("CEO",)),
    ("Chief Technology Officer", ("CTO",)),
    ("Chief Financial Officer", ("CFO",)),
    ("Chief Operating Officer", ("COO",)),
    ("Chief Marketing Officer", ("CMO",)),
    ("Chief Revenue Officer", ("CRO",)),
    ("Chief Product Officer", ("CPO",)),
    ("Chief People Officer", ("CPO",)),
    ("Chief Information Officer", ("CIO",)),
    ("Chief Human Resources Officer", ("CHRO",)),
    ("Chief Risk Officer", ("CRO",)),
    ("Chief Medical Officer", ("CMO",)),
    ("Chief Data Officer", ("CDO",)),
    ("Chief Customer Officer", ("CCO",)),
    ("Chief Commercial Officer", ("CCO",)),
    ("Chief Legal Officer", ("CLO",)),
    ("Sales Development Representative", ("SDR",)),
    ("Business Development Representative", ("BDR",)),
    ("Business Development Manager", ("BDM",)),
    ("Account Executive", ("AE",)),
    ("Customer Success Manager", ("CSM",)),
    ("Executive Assistant", ("EA",)),
    ("Human Resources", ("HR",)),
    ("Quality Assurance", ("QA",)),
    ("General Manager", ("GM",)),
    ("Managing Director", ("MD",)),
    ("Information Technology", ("IT",)),
    ("Operations", ("Ops",)),
    ("Development", ("Dev",)),
    ("Engineering", ("Eng",)),
    ("Representative", ("Rep",)),
    ("Assistant", ("Asst.",)),
    ("Director", ("Dir.",)),
    ("and", ("&",)),
    ("&", ("and",)),
)
SENIORITY_ABBREVIATIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Senior", ("Sr.", "Sr")),
    ("Junior", ("Jr.", "Jr")),
)
HOLDOUT_ABBREVIATIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Software Engineer", ("SWE",)),
    ("Senior", ("Snr", "Snr.")),
    ("Junior", ("Jnr",)),
    ("Manager", ("Mgr", "Mgr.")),
    ("Engineer", ("Engr",)),
    ("Coordinator", ("Coord.",)),
    ("Management", ("Mgmt",)),
    ("Principal", ("Princ.",)),
)
EXPANSIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("SVP", ("Senior Vice President",)),
    ("VP", ("Vice President",)),
    ("CEO", ("Chief Executive Officer",)),
    ("CTO", ("Chief Technology Officer",)),
    ("CFO", ("Chief Financial Officer",)),
    ("HR", ("Human Resources",)),
)
SEPARATORS = (" - ", " | ", " – ", ", ")
HOLDOUT_SEPARATORS = (" / ", " :: ")
_QUALIFIER = re.compile(r"^(?P<main>.+?)(?:, | - )(?P<qual>.+)$")
_DIRECTOR_OF = re.compile(r"^(?P<role>Director|Head) of (?P<func>[^,|]+)$")


# ----------------------------------------------------------------------------- coverage
def simulate_coverage(rng: np.random.Generator, tree: CompanyTree, coverage: float, cfg: CoverageConfig,
                      min_observed: int) -> set[int]:
    """Choose which employees are observed. Guarantees >= min_observed observed employees."""
    employees = tree.employees
    ceo = tree.root
    others = [n for n in employees if n is not ceo]
    mult = np.asarray([cfg.level_missing_multiplier[n.level] for n in others], dtype=float)
    expected_missing = (1.0 - coverage) * len(employees)
    scale = expected_missing / mult.sum() if mult.sum() > 0 else 0.0
    p_missing = np.clip(mult * scale, 0.0, 0.95)
    draws = rng.random(len(others))
    observed = {n.nid for n, d, p in zip(others, draws, p_missing, strict=True) if d >= p}
    if chance(rng, cfg.ceo_observed_probability):
        observed.add(ceo.nid)
    if len(observed) < min_observed:
        dropped = [n.nid for n in employees if n.nid not in observed]
        rng.shuffle(dropped)
        observed.update(dropped[: min_observed - len(observed)])
    for node in tree.nodes:  # board / advisor profiles exist only because they are observed
        if node.role_type == BOARD:
            observed.add(node.nid)
    return observed


def nearest_observed_manager(tree: CompanyTree, observed: set[int]) -> dict[int, int | None]:
    result: dict[int, int | None] = {}
    for node in tree.bfs_order():
        parent = node.parent
        while parent is not None and parent not in observed:
            parent = tree.nodes[parent].parent
        result[node.nid] = parent
    return result


# ----------------------------------------------------------------------------- titles
@dataclass(frozen=True)
class TitleContext:
    cfg: NoiseConfig
    holdout: bool
    small_company: bool
    company_short: str
    render_at_level: Callable[[Node, int], str]


def _apply_rules(rng: np.random.Generator, text: str, rules, p_each: float = 0.6) -> str:
    matching = [(src, dst) for src, dst in rules if re.search(rf"(?<!\w){re.escape(src)}(?!\w)", text)]
    if not matching:
        return text
    chosen = [r for r in matching if chance(rng, p_each)] or [matching[int(rng.integers(len(matching)))]]
    for src, dst in chosen:
        text = re.sub(rf"(?<!\w){re.escape(src)}(?!\w)", pick(rng, dst), text, count=1)
    return text


def _drop_qualifier(rng, text):
    m = _QUALIFIER.match(text)
    return m.group("main") if m else text


def _reorder_qualifier(rng, text):
    m = _QUALIFIER.match(text)
    if m and len(m.group("qual").split()) <= 3:
        return f"{m.group('qual')} {m.group('main')}"
    m = _DIRECTOR_OF.match(text)
    if m:
        return f"{m.group('func')} {'Director' if m.group('role') == 'Director' else 'Lead'}"
    return text


def _separator(choices):
    def transform(rng, text):
        m = _QUALIFIER.match(text)
        return f"{m.group('main')}{pick(rng, choices)}{m.group('qual')}" if m else text

    return transform


def _parenthesise(rng, text):
    m = _QUALIFIER.match(text)
    return f"{m.group('main')} ({m.group('qual')})" if m else text


def _typo(rng, text):
    words = text.split(" ")
    long_words = [i for i, w in enumerate(words) if len(w) >= 5 and w.isalpha()]
    if not long_words:
        return text
    i = pick(rng, long_words)
    w = words[i]
    j = int(rng.integers(1, len(w) - 2))
    words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
    return " ".join(words)


def _case(rng, text):
    return text.lower() if chance(rng, 0.6) else text.upper()


def observe_title(rng: np.random.Generator, node: Node, ctx: TitleContext) -> tuple[str | None, list[str]]:
    cfg = ctx.cfg
    tags: list[str] = []
    title = node.title
    is_hierarchy_role = node.role_type in ("ic", "manager", "head")

    if is_hierarchy_role and node.level is not None:
        if node.level >= Level.JUNIOR and chance(rng, cfg.stale_title):
            title = ctx.render_at_level(node, node.level - 1)
            tags.append("stale_title")
        elif (ctx.small_company and node.role_type in ("manager", "head") and node.level < Level.C_LEVEL
              and chance(rng, cfg.title_inflation_small_company)):
            title = ctx.render_at_level(node, node.level + 1)
            tags.append("inflated_title")

    if node.role_type != "ceo":
        if chance(rng, cfg.empty_title):
            return None, tags + ["empty_title"]
        if chance(rng, cfg.vague_title):
            return pick(rng, VAGUE_TITLES), tags + ["vague_title"]

    t = cfg.transforms
    h = cfg.holdout_transforms if ctx.holdout else {}
    steps: list[tuple[str, float, Callable]] = [
        ("drop_qualifier", t.get("drop_qualifier", 0), _drop_qualifier),
        ("holdout:reorder_qualifier", h.get("reorder_qualifier", 0), _reorder_qualifier),
        ("abbreviate_terms", t.get("abbreviate_terms", 0), lambda r, s: _apply_rules(r, s, TERM_ABBREVIATIONS)),
        ("abbreviate_seniority", t.get("abbreviate_seniority", 0), lambda r, s: _apply_rules(r, s, SENIORITY_ABBREVIATIONS, 1.0)),
        ("holdout:abbreviations", h.get("holdout_abbreviations", 0), lambda r, s: _apply_rules(r, s, HOLDOUT_ABBREVIATIONS)),
        ("expand_terms", t.get("expand_terms", 0), lambda r, s: _apply_rules(r, s, EXPANSIONS, 1.0)),
        ("parenthesise_qualifier", t.get("parenthesise_qualifier", 0), _parenthesise),
        ("separator", t.get("separator", 0), _separator(SEPARATORS)),
        ("holdout:separator", h.get("holdout_separator", 0), _separator(HOLDOUT_SEPARATORS)),
    ]
    for tag, p, fn in steps:
        if p and chance(rng, p):
            new = fn(rng, title)
            if new != title:
                title = new
                tags.append(tag)

    if node.employment_type == "contractor" and chance(rng, cfg.contractor_marker):
        title = pick(rng, (f"{title} (Contract)", f"Contract {title}", f"{title} - Contractor", f"{title} (Contractor)"))
        tags.append("contractor_marker")
    if chance(rng, t.get("decoration", 0)):
        co = ctx.company_short
        title = pick(rng, (f"{title} @ {co}", f"{title} at {co}", f"{title} | {co}", f"{title} | Hiring", f"{title} (Hiring!)"))
        tags.append("decoration")
    if h.get("typo") and chance(rng, h["typo"]):
        new = _typo(rng, title)
        if new != title:
            title = new
            tags.append("holdout:typo")
    if chance(rng, t.get("case", 0)):
        title = _case(rng, title)
        tags.append("case")
    return title, tags


# ----------------------------------------------------------------------------- department
def observe_department(rng: np.random.Generator, node: Node, cfg: NoiseConfig,
                       company_depts: list[str]) -> tuple[str | None, list[str]]:
    if node.dept == BOARD:
        return (None, []) if chance(rng, 0.7) else (pick(rng, BOARD_LABELS), [])
    if chance(rng, cfg.missing_department):
        return None, ["dept_missing"]
    if node.dept == EXECUTIVE:
        return pick(rng, EXECUTIVE_LABELS), []
    others = [d for d in company_depts if d != node.dept]
    if others and chance(rng, cfg.mislabeled_department):
        return DEPARTMENTS[pick(rng, others)].labels[0], ["dept_mislabeled"]
    labels = DEPARTMENTS[node.dept].labels
    label = labels[0] if chance(rng, 0.55) or len(labels) == 1 else pick(rng, labels[1:])
    return label, []
