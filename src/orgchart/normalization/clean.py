"""Title cleaning and tokenisation.

Turns a raw profile title into a main token sequence plus qualifier token
sequences, records what was stripped or rewritten, and sets flags
(contractor / interim / founder). Every rewrite is reported in `notes` so the
final result stays explainable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

from orgchart.normalization import lexicon as lx

LEGAL_SUFFIXES = re.compile(r"\s+(pty ltd|pte ltd|pvt ltd|limited|ltd|inc\.?|llc|plc)$", re.IGNORECASE)
_SEPARATORS = re.compile(r"\s+(?:-|–|—|\||/|::|•)\s+|\s*,\s*")
_PAREN = re.compile(r"\s*\(([^)]*)\)\s*")
_TOKEN = re.compile(r"[a-z0-9&+]+")
MANAGEMENT_WORDS = {"manager", "director", "lead", "head", "leader", "supervisor", "president", "officer", "vice"}


@dataclass
class CleanTitle:
    raw: str
    display: str  # title after removing decorations / markers, original casing
    main: list[str]
    qualifiers: list[list[str]]
    qualifier_text: list[str]  # qualifiers in original casing (for scope labels)
    contractor: bool = False
    interim: bool = False
    founder: bool = False
    typo_corrected: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def main_text(self) -> str:
        return " ".join(self.main)

    @property
    def all_tokens(self) -> list[str]:
        return self.main + [t for q in self.qualifiers for t in q]


def company_short_name(company_name: str | None) -> str | None:
    if not company_name:
        return None
    return LEGAL_SUFFIXES.sub("", company_name.strip()).strip() or None


@lru_cache(maxsize=1)
def vocabulary() -> frozenset[str]:
    """Every word the lexicon knows; used to repair adjacent-letter transpositions."""
    words: set[str] = set(MANAGEMENT_WORDS)
    phrases = (
        list(lx.IC_ROLES) + list(lx.DEPT_KEYWORDS) + list(lx.SPECIALTIES) + list(lx.REGIONS) + list(lx.DEPT_LABELS)
        + list(lx.TOKEN_EXPANSIONS.values()) + list(lx.CHIEF_WORDS) + list(lx.VAGUE_TITLES)
    )
    for phrase in phrases:
        words.update(_TOKEN.findall(phrase))
    words.update({"senior", "junior", "graduate", "intern", "principal", "staff", "chief", "executive", "group",
                  "regional", "national", "associate", "assistant", "deputy", "general", "managing", "partner",
                  "team", "store", "area", "district", "plant", "site", "trainee", "apprentice", "student"})
    return frozenset(w for w in words if len(w) >= 4)


def _transposition_fix(token: str) -> str | None:
    vocab = vocabulary()
    if len(token) < 5 or not token.isalpha() or token in vocab:
        return None
    fixes = {token[:i] + token[i + 1] + token[i] + token[i + 2:] for i in range(1, len(token) - 1)}
    hits = sorted(fixes & vocab)
    return hits[0] if len(hits) == 1 else None


def _expand(tokens: list[str], industry: str, notes: list[str]) -> list[str]:
    out: list[str] = []
    for i, tok in enumerate(tokens):
        last = i == len(tokens) - 1
        nxt = tokens[i + 1] if not last else None
        if tok == "and":
            out.append("&")
            continue
        if tok in lx.TOKEN_EXPANSIONS:
            expansion = lx.TOKEN_EXPANSIONS[tok]
            out.extend(expansion.split())
            notes.append(f"expanded '{tok}' -> '{expansion}'")
            continue
        if tok == "eng":
            word = "engineer" if last else "engineering"
        elif tok == "dev":
            word = "developer" if last else "development"
        elif tok == "admin":
            word = "administrator" if last else "administration"
        elif tok == "pm" and nxt is None:
            word = "product manager" if industry in ("saas", "fintech") else "project manager"
        elif tok == "cnc" and industry == "healthcare":
            word = "clinical nurse consultant"
        else:
            out.append(tok)
            continue
        out.extend(word.split())
        notes.append(f"expanded '{tok}' -> '{word}' (context: {'end of title' if last else 'before ' + str(nxt)}"
                     f"{', industry ' + industry if tok in ('pm', 'cnc') else ''})")
    return out


def _tokens(segment: str, industry: str, notes: list[str], state: dict) -> list[str]:
    seg = segment.lower().replace("’", "'")
    seg = re.sub(r"(?<=\w)-(?=\w)", " ", seg)  # co-founder, multi-site, mid-market
    raw_tokens = _TOKEN.findall(seg)
    fixed = []
    for tok in raw_tokens:
        repaired = _transposition_fix(tok)
        if repaired:
            notes.append(f"repaired typo '{tok}' -> '{repaired}'")
            state["typo"] = True
            tok = repaired
        fixed.append(tok)
    expanded = _expand(fixed, industry, notes)
    # Canonical short forms so 'Human Resources Business Partner' == 'HR Business Partner'.
    joined = " ".join(expanded).replace("human resources", "hr").replace("information technology", "it")
    return joined.split()


def clean_title(raw: str, company_name: str | None = None, industry: str = "general") -> CleanTitle:
    notes: list[str] = []
    text = re.sub(r"\s+", " ", raw).strip()
    flags = {"contractor": False, "interim": False, "founder": False}

    def strip(pattern: str, label: str, flag: str | None = None) -> None:
        nonlocal text
        new = re.sub(pattern, "", text, flags=re.IGNORECASE).strip(" ,-|")
        if new != text and new:
            notes.append(f"removed {label}")
            text = new
            if flag:
                flags[flag] = True

    short = company_short_name(company_name)
    strip(r"\s*\(hiring!?\)\s*$|\s*\|\s*(we'?re )?hiring!?\s*$", "hiring decoration")
    strip(r"\s+@\s+.*$", "'@ company' decoration")
    if short:
        names = "|".join(re.escape(n) for n in {company_name.strip(), short})
        strip(rf"\s+(at|\|)\s+({names})\s*$", f"company name '{short}'")
    strip(r"\s*\(contract(or)?\)\s*$|\s+-\s+contract(or)?\s*$|^contract(or)?\s+|\bfreelance\b", "contractor marker",
          "contractor")
    strip(r"^(interim|acting)\s+", "interim marker", "interim")
    founder_match = re.match(r"^(co[- ]?founder|founder)\b\s*(?:&|and|/|,|-)?\s*", text, flags=re.IGNORECASE)
    if founder_match:
        flags["founder"] = True
        rest = text[founder_match.end():].strip()
        notes.append("founder marker")
        text = rest or "founder"

    # Parenthesised parts become qualifiers; then split on separators.
    paren_parts = _PAREN.findall(text)
    base = _PAREN.sub(" ", text).strip()
    segments = [s for s in _SEPARATORS.split(base) if s and s.strip()] + [p for p in paren_parts if p.strip()]
    state = {"typo": False}
    token_segments = [_tokens(s, industry, notes, state) for s in segments]
    keep = [(s.strip(), t) for s, t in zip(segments, token_segments, strict=True) if t]
    if not keep:
        return CleanTitle(raw, text, [], [], [], notes=notes, **flags)
    main = keep[0][1]
    return CleanTitle(
        raw=raw,
        display=text,
        main=main,
        qualifiers=[t for _, t in keep[1:]],
        qualifier_text=[s for s, _ in keep[1:]],
        typo_corrected=state["typo"],
        notes=notes,
        **flags,
    )
