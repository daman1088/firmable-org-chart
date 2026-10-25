"""Rule-based, explainable job-title parser.

`parse_title` maps one observed profile (raw title, raw department, company
industry/name) to a `ParsedTitle`. Parsing runs in fixed stages, each of which
records what it decided and why:

1. clean        decorations, markers, separators, abbreviations, typo repair
2. frame        what kind of title it is: board / CEO / support / C-suite /
                people-manager noun / individual-contributor role; this sets
                the seniority level and management authority
3. department    title evidence (role noun, function keywords) reconciled with
                the raw department field and industry conventions
4. specialty    team / discipline, only when the title states or implies it
5. scope        region from qualifiers (product/division scopes are added by
                the company-level pass in company_context.py)

Seniority (rank on the ladder) and management authority are separate outputs:
"Principal Engineer" is level 4 *and* an individual contributor.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

from orgchart.ladder import LEVEL_NAMES, Level
from orgchart.normalization import lexicon as lx
from orgchart.normalization.clean import CleanTitle, clean_title

NORMALIZER_VERSION = "1.0.0"


@dataclass
class ParsedTitle:
    title_clean: str | None = None
    level: int | None = None
    level_confidence: float = 0.0
    track: str = "unknown"  # executive | manager | ic | support | board | unknown
    manages_people: bool | None = None
    management_confidence: float = 0.0
    department: str | None = None
    department_confidence: float = 0.0
    department_source: str = "none"  # title | field | title+field | industry | none
    specialty: str | None = None
    specialty_confidence: float = 0.0
    scope: str | None = None
    scope_type: str | None = None  # region | unit
    is_contractor: bool = False
    is_intern: bool = False
    is_interim: bool = False
    is_founder: bool = False
    is_non_employee: bool = False
    parse_confidence: float = 0.0
    reason_codes: list[str] = field(default_factory=list)
    rules_fired: list[str] = field(default_factory=list)
    explanation: list[str] = field(default_factory=list)
    unresolved_qualifiers: list[str] = field(default_factory=list)  # used by the company pass

    @property
    def level_name(self) -> str | None:
        return None if self.level is None else LEVEL_NAMES[self.level]


@dataclass
class Frame:
    kind: str  # board | ceo | support | csuite | manager | ic | vague | unknown
    level: int | None
    level_conf: float
    manages: bool | None
    mgmt_conf: float
    rule: str
    why: str
    dept: str | None = None  # department implied by the frame itself (role noun / C-suite)
    dept_conf: float = 0.0
    specialty: str | None = None
    ambiguous_dept: tuple[str, ...] = ()


def industry_key(label: str | None) -> str:
    return lx.INDUSTRY_LABELS.get((label or "").strip().lower(), "general")


# ----------------------------------------------------------------------------- phrase matching
def _find(tokens: list[str], phrase: str) -> int:
    """Index of `phrase` (space separated) as a whole-token run in `tokens`, else -1."""
    words = phrase.split()
    n = len(words)
    for i in range(len(tokens) - n + 1):
        if tokens[i:i + n] == words:
            return i
    return -1


@lru_cache(maxsize=1)
def _roles_longest_first() -> list[str]:
    return sorted(lx.IC_ROLES, key=lambda p: (-len(p.split()), -len(p)))


@lru_cache(maxsize=1)
def _keywords_longest_first() -> list[str]:
    return sorted(lx.DEPT_KEYWORDS, key=lambda p: (-len(p.split()), -len(p)))


@lru_cache(maxsize=1)
def _specialties_longest_first() -> list[str]:
    return sorted(lx.SPECIALTIES, key=lambda p: (-len(p.split()), -len(p)))


def _longest_role(tokens: list[str]) -> tuple[str, int] | None:
    best: tuple[str, int] | None = None
    for phrase in _roles_longest_first():
        idx = _find(tokens, phrase)
        if idx >= 0:
            # Prefer the longest phrase; among equal length, the right-most (head noun is last in English titles).
            if best is None or len(phrase.split()) > len(best[0].split()) or (
                len(phrase.split()) == len(best[0].split()) and idx > best[1]
            ):
                best = (phrase, idx)
            if best and len(phrase.split()) < len(best[0].split()):
                break
    return best


def _scan(tokens: list[str], phrases: list[str]) -> list[tuple[str, int]]:
    """Non-overlapping longest-first phrase matches, in title order."""
    taken = [False] * len(tokens)
    hits = []
    for phrase in phrases:
        words = phrase.split()
        n = len(words)
        for i in range(len(tokens) - n + 1):
            if tokens[i:i + n] == words and not any(taken[i:i + n]):
                hits.append((phrase, i))
                for j in range(i, i + n):
                    taken[j] = True
    return sorted(hits, key=lambda h: h[1])


# ----------------------------------------------------------------------------- frames
def _ic_frame(tokens: list[str], phrase: str, idx: int, industry: str, rule: str) -> Frame:
    role = lx.IC_ROLES[phrase]
    level = lx.INDUSTRY_ROLE_LEVELS.get(industry, {}).get(phrase, role.base_level)
    why = [f"role '{phrase}' (individual contributor, base level {level})"]
    if role.note:
        why.append(role.note)
    conf = 0.75
    before = tokens[:idx]
    after = tokens[idx + len(phrase.split()):]
    if any(t in tokens for t in ("intern", "internship", "student", "apprentice", "placement")):
        level, conf = 0, 0.9
        why.append("intern/student marker -> intern")
    elif "casual" in before:
        level, conf = 0, 0.7
        why.append("'casual' marker -> entry (intern-equivalent) level")
    elif any(t in before for t in ("graduate", "junior", "trainee", "entry")):
        level, conf = 1, 0.85
        why.append("junior/graduate marker -> junior")
    elif "strategic" in before and role.dept == "sales":
        level, conf = 4, 0.75
        why.append("'strategic' account role -> most senior sales IC (level 4)")
    elif any(t in before for t in ("staff", "principal", "lead")):
        level, conf = 4, 0.8
        why.append("technical seniority ('staff'/'principal'/'lead' prefix) -> level 4; not a people-management signal")
    elif "senior" in before:
        level, conf = min(max(level + 1, 3 if level >= 2 else level + 1), 4), 0.85
        why.append(f"'senior' modifier -> level {level}")
    elif any(t in before for t in ("associate", "assistant")) and level >= 2:
        level, conf = 1, 0.75
        why.append("associate/assistant prefix -> junior")
    if after and after[-1] in lx.ROMAN_LEVELS and level != Level.INTERN:
        graded = lx.ROMAN_LEVELS[after[-1]]
        level, conf = (max(level, graded), 0.8) if "senior" in before else (graded, 0.85)
        why.append(f"grade suffix '{after[-1].upper()}' -> level {level}")
    dept = lx.resolve(role.dept, industry) if role.dept is not None else None
    weak = phrase in lx.WEAK_ROLE_DEPARTMENTS and not (industry == "consulting" and dept == "consulting")
    dept_conf = (0.55 if weak else 0.9) if dept else 0.0
    return Frame("ic", level, conf, False, 0.85 if role.note.startswith("IC role") else 0.9, rule, "; ".join(why),
                 dept=dept, dept_conf=dept_conf, specialty=role.specialty)


_MANAGER_RULES: list[tuple[str, str, int, float, float, str]] = [
    # (rule id, regex over main text, level, level_conf, mgmt_conf, explanation)
    ("vp.executive", r"\bexecutive vice president\b", Level.SVP, 0.85, 0.95, "EVP -> SVP level"),
    ("vp.senior", r"\bsenior vice president\b", Level.SVP, 0.9, 0.95, "SVP"),
    ("vp.assistant", r"\bassistant vice president\b", Level.SENIOR_DIRECTOR, 0.7, 0.9, "AVP sits below VP"),
    ("vp", r"\bvice president\b", Level.VP, 0.9, 0.95, "VP"),
    ("president", r"\bpresident\b", Level.SVP, 0.6, 0.95, "President of a unit (not the company)"),
    ("partner.senior", r"\b(senior|lead) partner\b", Level.SVP, 0.75, 0.9, "senior/lead partner"),
    ("partner.associate", r"\bassociate partner\b", Level.SENIOR_DIRECTOR, 0.8, 0.9, "associate partner"),
    ("partner.practice", r"\bpractice lead\b", Level.VP, 0.7, 0.9, "practice lead = partner level"),
    ("gm", r"\bgeneral manager\b", Level.VP, 0.6, 0.9, "General Manager of a function (can also mean top executive)"),
    ("gc.deputy", r"\bdeputy general counsel\b", Level.SENIOR_DIRECTOR, 0.6, 0.85, "deputy GC"),
    ("gc", r"\bgeneral counsel\b", Level.VP, 0.6, 0.85, "General Counsel heads legal"),
    ("director.senior", r"\bsenior director\b", Level.SENIOR_DIRECTOR, 0.85, 0.9, "senior director"),
    ("director.executive", r"\bexecutive director\b", Level.VP, 0.55, 0.9, "executive director"),
    ("director.associate", r"\b(associate|assistant|deputy) director\b", Level.DIRECTOR, 0.65, 0.85,
     "associate/assistant/deputy director (director band)"),
    ("director.finance", r"^finance director\b", Level.VP, 0.55, 0.9, "'Finance Director' (AU/UK usage) heads finance"),
    ("director", r"\bdirector\b", Level.DIRECTOR, 0.75, 0.9, "director"),
    ("head", r"\bhead\b", Level.DIRECTOR, 0.5, 0.9, "'Head of' spans levels 5-9; director band assumed"),
    ("controller.group", r"\bgroup financial controller\b", Level.SENIOR_DIRECTOR, 0.7, 0.85, "group financial controller"),
    ("controller", r"\bfinancial controller\b", Level.SENIOR_MANAGER, 0.55, 0.85,
     "financial controller (senior manager band; often heads finance in smaller firms)"),
    ("manager.group", r"\bgroup\b.*\bmanager\b", Level.MANAGER, 0.75, 0.85, "'Group ... Manager' manages a group of ICs"),
]
_LOCATION_MANAGER_RULES: list[tuple[str, str, int, float, str]] = [
    ("manager.national", r"\bnational\b.*\bmanager\b", Level.SENIOR_DIRECTOR, 0.7, "national-scope manager"),
    ("manager.regional", r"\b(regional|state)\b.*\bmanager\b", Level.DIRECTOR, 0.7, "regional/state manager"),
    ("manager.area", r"\b(area|district|multi site)\b manager\b", Level.SENIOR_MANAGER, 0.75, "multi-store manager"),
    ("manager.plant", r"\b(plant|site)\b manager\b", Level.DIRECTOR, 0.7, "plant/site manager runs a facility"),
    ("manager.dc", r"\bdistribution (centre|center)\b manager\b", Level.SENIOR_MANAGER, 0.7, "distribution centre manager"),
    ("manager.department", r"\bdepartment\b manager\b", Level.SENIOR_MANAGER, 0.55, "department manager"),
]
_LEAD_RULES = r"\b(team lead|team leader|tech lead|technical lead|supervisor|2ic|engagement lead|delivery lead)\b"


def _manager_frame(tokens: list[str], industry: str) -> Frame | None:
    text = " ".join(tokens)
    senior = "senior" in tokens

    for rule, pattern, level, lconf, mconf, why in _MANAGER_RULES:
        if re.search(pattern, text):
            if rule == "director.executive" and industry == "consulting":
                level, why = Level.SENIOR_DIRECTOR, "executive director (consulting partner track)"
            if rule == "manager.group" and senior:
                level, why = Level.SENIOR_MANAGER, why + " + 'senior'"
            if rule in ("head", "gm") and senior:
                level = min(level + 1, Level.SVP)
            return Frame("manager", int(level), lconf, True, mconf, f"mgr.{rule}", why)

    # 'Partner' that is not an HR/alliance IC role.
    if re.search(r"\bpartner\b", text) and not _longest_role_contains(tokens, "partner"):
        lconf = 0.75 if industry == "consulting" else 0.5
        return Frame("manager", Level.VP, lconf, True, 0.85, "mgr.partner", "partner (equity/partner-track leader)")

    # IC roles that contain 'manager' (Product Manager, Account Manager, ...) win over the generic manager rule.
    ic = _longest_role(tokens)
    if ic and "manager" in ic[0].split():
        return None

    for rule, pattern, level, lconf, why in _LOCATION_MANAGER_RULES:
        if re.search(pattern, text):
            if senior:
                level, why = level + 1, why + " + 'senior'"
            return Frame("manager", int(level), lconf, True, 0.9, f"mgr.{rule}", why)
    if re.search(r"\b(associate|assistant|deputy)\b.*\bmanager\b", text):
        return Frame("manager", Level.LEAD, 0.7, True, 0.8, "mgr.manager.assistant",
                     "assistant/associate/deputy manager (second-in-charge)")
    if re.search(r"\bmanager\b", text):
        conf = 0.75 if len(tokens) > 1 or industry == "consulting" else 0.4
        if senior:
            return Frame("manager", Level.SENIOR_MANAGER, conf, True, 0.85, "mgr.manager.senior", "senior manager")
        return Frame("manager", Level.MANAGER, conf, True, 0.85, "mgr.manager", "manager")
    if re.search(_LEAD_RULES, text):
        conf = 0.55 if "tech lead" in text or "technical lead" in text else 0.8
        return Frame("manager", Level.LEAD, 0.8, True, conf, "mgr.lead", "team lead / supervisor (frontline manager)")
    if tokens and tokens[-1] == "lead" and len(tokens) > 1 and tokens[0] != "lead":
        return Frame("manager", Level.LEAD, 0.65, True, 0.65, "mgr.lead.suffix",
                     "'<function> Lead' (suffix) leads a team; 'Lead <role>' (prefix) would be an IC")
    if text == "principal":
        if industry == "consulting":
            return Frame("manager", Level.SENIOR_DIRECTOR, 0.65, True, 0.85, "mgr.principal",
                         "'Principal' alone is a partner-track grade in consulting")
        return Frame("ic", Level.LEAD, 0.4, False, 0.5, "ic.principal", "'Principal' alone (non-consulting)")
    return None


def _longest_role_contains(tokens: list[str], word: str) -> bool:
    ic = _longest_role(tokens)
    return bool(ic and word in ic[0].split() and len(ic[0].split()) > 1)


def _csuite_dept(words: list[str], industry: str) -> tuple[str | None, tuple[str, ...]]:
    for n in (2, 1):
        for i in range(len(words) - n + 1):
            key = " ".join(words[i:i + n])
            if key in lx.CHIEF_WORDS:
                return lx.resolve(lx.CHIEF_WORDS[key], industry), ()
    return None, ()


def _special_frame(ct: CleanTitle, industry: str) -> Frame | None:
    text = ct.main_text
    tokens = ct.main
    if any(re.match(p, text) for p in lx.BOARD_PATTERNS):
        return Frame("board", None, 0.9, None, 0.9, "board", "board / advisor profile: not part of the reporting line")

    if re.search(lx.CHIEF_OF_STAFF, text):
        return Frame("support", Level.DIRECTOR, 0.7, False, 0.75, "support.chief_of_staff",
                     "Chief of Staff is a senior advisor role, not a C-suite executive", dept="executive", dept_conf=0.9)
    if re.search(lx.EXEC_ASSISTANT, text):
        return Frame("support", Level.INTERMEDIATE, 0.8, False, 0.95, "support.assistant",
                     "executive/personal assistant supports an executive", dept="executive", dept_conf=0.9)

    ceo_titles = {"chief executive officer", "ceo", "chief executive", "managing director", "founder", "president",
                  "managing partner", "ceo & managing partner", "managing director & ceo", "ceo & founder",
                  "ceo & managing director"}
    if text in ceo_titles or "ceo" in tokens or (ct.founder and text == "founder"):
        if ct.qualifiers and text in ("managing director", "president"):
            return Frame("manager", Level.SVP, 0.6, True, 0.95, "exec.md.scoped",
                         f"'{text}' of a region/unit, not the company", dept="executive", dept_conf=0.6)
        if ct.qualifiers and text == "managing partner":
            return Frame("csuite", Level.C_LEVEL, 0.6, True, 0.95, "exec.managing_partner.scoped",
                         "managing partner of a practice")
        why = "chief executive title" + (" (founder)" if ct.founder else "")
        if text in ("managing director", "managing partner"):
            why = f"'{text}' is the top executive title (AU/UK and partnership usage)"
        return Frame("ceo", Level.CEO, 0.9, True, 0.95, "exec.ceo", why, dept="executive", dept_conf=0.95)

    acronyms = [t for t in tokens if t in lx.C_SUITE]
    if acronyms:
        acr = acronyms[0]
        options = tuple(lx.resolve(o, industry) for o in lx.C_SUITE[acr])
        ambiguous = acr in lx.AMBIGUOUS_ACRONYMS and len(set(options)) > 1
        return Frame("csuite", Level.C_LEVEL, 0.9, True, 0.95, f"exec.csuite.{acr}",
                     f"C-suite acronym '{acr.upper()}'" + (f" (ambiguous: {', '.join(sorted(set(options)))})" if ambiguous else ""),
                     dept=options[0], dept_conf=0.5 if ambiguous else 0.9,
                     ambiguous_dept=tuple(dict.fromkeys(options)) if ambiguous else ())
    m = re.search(r"\bchief (.+?) officer\b", text)
    if m:
        dept, _ = _csuite_dept(m.group(1).split(), industry)
        return Frame("csuite", Level.C_LEVEL, 0.9, True, 0.95, "exec.csuite.chief_officer",
                     f"'Chief {m.group(1).title()} Officer'", dept=dept, dept_conf=0.9 if dept else 0.0)
    m = re.search(r"\bchief (\w+)\b", text)
    if m and m.group(1) not in ("of",):
        dept, _ = _csuite_dept([m.group(1)], industry)
        return Frame("csuite", Level.C_LEVEL, 0.6, True, 0.85, "exec.chief_other",
                     f"'Chief {m.group(1).title()}' (top of function, not a standard C-suite title)",
                     dept=dept, dept_conf=0.7 if dept else 0.0)
    return None


def _frame(ct: CleanTitle, industry: str) -> Frame:
    special = _special_frame(ct, industry)
    if special:
        return special
    tokens = ct.main
    if ct.main_text in lx.VAGUE_TITLES and not ct.qualifiers:
        return Frame("vague", None, 0.0, None, 0.0, "vague", f"'{ct.main_text}' does not identify a role")
    manager = _manager_frame(tokens, industry)
    if manager:
        return manager
    ic = _longest_role(tokens)
    if ic:
        return _ic_frame(tokens, ic[0], ic[1], industry, f"ic.role.{ic[0].replace(' ', '_')}")
    if "intern" in tokens or "internship" in tokens or "vacationer" in tokens:
        return Frame("ic", Level.INTERN, 0.85, False, 0.95, "ic.intern", "intern")
    # Main part has no role noun: try qualifiers ("Intern, Backend" handled above; "Lead, Payroll" etc.)
    for q in ct.qualifiers:
        ic = _longest_role(q)
        if ic and ic[0] not in ("specialist", "associate", "assistant", "officer"):
            frame = _ic_frame(q, ic[0], ic[1], industry, f"ic.role_in_qualifier.{ic[0].replace(' ', '_')}")
            frame.level_conf *= 0.8
            return frame
    if ct.main_text in lx.VAGUE_TITLES:
        return Frame("vague", None, 0.0, None, 0.0, "vague", f"'{ct.main_text}' does not identify a role")
    return Frame("unknown", None, 0.0, None, 0.0, "unknown", "no known role or management noun")


def _vague_frame(text: str, industry: str) -> Frame:
    """Best-effort reading of a vague title; always low confidence."""
    if industry == "consulting" and text in ("manager", "consultant", "associate"):
        level = {"manager": Level.MANAGER, "consultant": Level.INTERMEDIATE, "associate": Level.INTERMEDIATE}[text]
        return Frame("manager" if text == "manager" else "ic", level, 0.55, text == "manager", 0.6,
                     f"consulting.grade.{text}", f"'{text.title()}' is a consulting grade", dept="consulting",
                     dept_conf=0.6)
    if text in ("manager", "leader"):
        return Frame("manager", Level.MANAGER, 0.3, True, 0.4, "vague.manager", f"vague '{text}'")
    if text in ("team member", "staff member", "employee", "staff", "worker", "team"):
        return Frame("ic", Level.INTERMEDIATE, 0.2, False, 0.5, "vague.member", f"vague '{text}'")
    return Frame("ic", Level.INTERMEDIATE, 0.25, False, 0.4, "vague.generic", f"vague '{text}'")


# ----------------------------------------------------------------------------- department / specialty / scope
def department_from_field(raw: str | None, industry: str) -> tuple[str | None, bool]:
    """(department, is_board_label)."""
    if not raw:
        return None, False
    key = re.sub(r"\s+", " ", raw.strip().lower()).replace(" and ", " & ")
    if key in lx.BOARD_DEPT_LABELS:
        return None, True
    value = lx.DEPT_LABELS.get(key)
    return (lx.resolve(value, industry) if value else None), False


def _keyword_departments(ct: CleanTitle, industry: str) -> list[tuple[str, str, float]]:
    """(keyword, department, weight) for function keywords in main (weight 2) and qualifiers (weight 1)."""
    out = []
    for tokens, weight in [(ct.main, 2.0)] + [(q, 1.0) for q in ct.qualifiers]:
        for phrase, _ in _scan(tokens, _keywords_longest_first()):
            out.append((phrase, lx.resolve(lx.DEPT_KEYWORDS[phrase], industry), weight))
    return out


def _decide_department(frame: Frame, ct: CleanTitle, industry: str, field_dept: str | None,
                       field_raw: str | None, why: list[str], reasons: list[str]) -> tuple[str | None, float, str]:
    title_dept, title_conf, source_note = frame.dept, frame.dept_conf, "role/frame"
    if frame.ambiguous_dept and field_dept in frame.ambiguous_dept:
        why.append(f"department: ambiguous title resolved by department field '{field_raw}' -> {field_dept}")
        return field_dept, 0.85, "title+field"
    if title_dept is None and industry == "consulting" and frame.kind in ("manager", "ic"):
        # 'Manager, Technology' in a consulting firm: the qualifier names a practice, not a support function.
        for q in ct.qualifiers:
            practice = next((p for p, _ in _scan(q, _specialties_longest_first()) if lx.SPECIALTIES[p][0] == "consulting"),
                            None)
            if practice and not _keyword_departments(CleanTitle(ct.raw, ct.display, ct.main, [], []), industry):
                title_dept, title_conf, source_note = "consulting", 0.8, f"consulting practice '{practice}'"
                break
    if title_dept is None:
        hits = _keyword_departments(ct, industry)
        if hits:
            scores: dict[str, float] = {}
            for _, dept, weight in hits:
                scores[dept] = scores.get(dept, 0.0) + weight
            ranked = sorted(scores.items(), key=lambda kv: -kv[1])
            top = ranked[0]
            tied = len(ranked) > 1 and ranked[1][1] == top[1]
            # Ties: the right-most function word is the head of the noun phrase ("Infrastructure Engineering").
            title_dept = top[0] if not tied else next(d for _, d, _ in reversed(hits) if scores[d] == top[1])
            title_conf = 0.8 if not tied else 0.55
            source_note = "keywords " + ", ".join(f"'{k}'" for k, d, _ in hits if d == title_dept)
    if title_dept and field_dept:
        if title_dept == field_dept:
            why.append(f"department: {title_dept} (title {source_note} agrees with field '{field_raw}')")
            return title_dept, min(0.98, title_conf + 0.1), "title+field"
        if frame.kind in ("vague", "unknown") or title_conf < 0.6:
            why.append(f"department: field '{field_raw}' -> {field_dept} preferred over weak title evidence "
                       f"({title_dept})")
            reasons.append("DEPT_CONFLICT")
            return field_dept, 0.6, "field"
        why.append(f"department: title {source_note} -> {title_dept} overrides field '{field_raw}' ({field_dept})")
        reasons.append("DEPT_CONFLICT")
        return title_dept, title_conf - 0.15, "title"
    if title_dept:
        why.append(f"department: {title_dept} from title {source_note} (no usable department field)")
        return title_dept, title_conf, "title"
    if field_dept:
        why.append(f"department: {field_dept} from field '{field_raw}' (title gives no function)")
        return field_dept, 0.8, "field"
    if industry == "consulting" and frame.kind in ("manager", "ic", "vague"):
        why.append("department: consulting firm, generic grade title -> consulting (industry default)")
        return "consulting", 0.5, "industry"
    reasons.append("DEPT_UNRESOLVED")
    why.append("department: unresolved (no function in title, no department field)")
    return None, 0.0, "none"


def _specialty(ct: CleanTitle, dept: str | None, frame: Frame, why: list[str]) -> tuple[str | None, float, set[int]]:
    """Specialty for the resolved department; returns qualifier indexes consumed."""
    if dept is None:
        return None, 0.0, set()
    used: set[int] = set()
    for qi, q in enumerate(ct.qualifiers):
        for phrase, _ in _scan(q, _specialties_longest_first()):
            spec = _specialty_for(phrase, dept)
            if spec:
                used.add(qi)
                why.append(f"specialty: '{spec}' from qualifier")
                return spec, 0.9, used
    for phrase, _ in _scan(ct.main, _specialties_longest_first()):
        spec = _specialty_for(phrase, dept)
        if spec:
            why.append(f"specialty: '{spec}' from title words")
            return spec, 0.8, used
    if frame.specialty:
        why.append(f"specialty: '{frame.specialty}' implied by role")
        return frame.specialty, 0.75, used
    return None, 0.0, used


def _specialty_for(phrase: str, dept: str) -> str | None:
    if (dept, phrase) in lx.SPECIALTY_BY_DEPT_OVERRIDES:
        return lx.SPECIALTY_BY_DEPT_OVERRIDES[(dept, phrase)]
    spec_dept, spec = lx.SPECIALTIES[phrase]
    return spec if spec_dept == dept else None


def _scope(ct: CleanTitle, used: set[int], why: list[str]) -> tuple[str | None, str | None, list[str]]:
    unresolved = []
    found: str | None = None
    for qi, (q, text) in enumerate(zip(ct.qualifiers, ct.qualifier_text, strict=True)):
        if qi in used:
            continue
        joined = " ".join(q)
        if joined in lx.REGIONS and found is None:
            found = lx.REGIONS[joined]
            why.append(f"scope: region '{found}' from qualifier")
            continue
        if not _scan(q, _keywords_longest_first()) and not _longest_role(q) and len(q) <= 3:
            unresolved.append(text)
    if found is None:
        for region, label in lx.REGIONS.items():
            if region not in lx.QUALIFIER_ONLY_REGIONS and len(region) > 2 and _find(ct.main, region) >= 0:
                found = label
                why.append(f"scope: region '{label}' in title")
                break
    return found, ("region" if found else None), unresolved


# ----------------------------------------------------------------------------- entry point
@lru_cache(maxsize=200_000)
def parse_title(raw_title: str | None, raw_department: str | None, industry_label: str | None,
                company_name: str | None = None) -> ParsedTitle:
    industry = industry_key(industry_label)
    out = ParsedTitle()
    why, reasons, rules = out.explanation, out.reason_codes, out.rules_fired
    field_dept, board_label = department_from_field(raw_department, industry)

    if raw_title is None or not raw_title.strip():
        reasons += ["EMPTY_TITLE", "LEVEL_UNRESOLVED"]
        why.append("title is empty")
        if field_dept:
            out.department, out.department_confidence, out.department_source = field_dept, 0.8, "field"
            why.append(f"department: {field_dept} from field '{raw_department}'")
        out.is_non_employee = board_label
        return out

    ct = clean_title(raw_title, company_name, industry)
    out.title_clean = ct.main_text + "".join(f", {' '.join(q)}" for q in ct.qualifiers)
    out.is_contractor, out.is_interim, out.is_founder = ct.contractor, ct.interim, ct.founder
    why.extend(f"clean: {n}" for n in ct.notes)

    frame = _frame(ct, industry)
    if frame.kind == "vague":
        why.append(f"frame: {frame.why}")
        frame = _vague_frame(ct.main_text, industry)
        if not frame.rule.startswith("consulting.grade"):
            reasons.append("VAGUE_TITLE")
    elif frame.kind == "unknown":
        reasons.append("UNKNOWN_ROLE")
    if board_label and frame.kind in ("manager", "vague", "unknown") and frame.level in (None, Level.DIRECTOR):
        frame = Frame("board", None, 0.8, None, 0.8, "board.by_field", "director/advisor title in a Board department")
    rules.append(frame.rule)
    why.append(f"frame: {frame.why}")

    if frame.kind == "board":
        out.is_non_employee, out.track = True, "board"
        out.department, out.department_source, out.department_confidence = None, "none", 0.0
        out.parse_confidence = frame.level_conf
        return out

    out.level = frame.level
    out.level_confidence = frame.level_conf
    out.manages_people = frame.manages
    out.management_confidence = frame.mgmt_conf
    out.is_intern = frame.level == Level.INTERN
    if frame.level is None:
        reasons.append("LEVEL_UNRESOLVED")
    else:
        why.append(f"seniority: level {frame.level} ({LEVEL_NAMES[frame.level]})")

    if frame.kind in ("ceo", "csuite"):
        out.track = "executive"
    elif frame.kind == "support":
        out.track = "support"
    elif frame.manages is True:
        out.track = "manager"
    elif frame.manages is False:
        out.track = "ic"
    if out.track in ("manager", "executive"):
        why.append("management: title carries people-management authority")
    elif out.track == "ic":
        why.append("management: individual contributor title")

    dept, dept_conf, source = _decide_department(frame, ct, industry, field_dept, raw_department, why, reasons)
    if frame.ambiguous_dept and source != "title+field":
        reasons.append("AMBIGUOUS_ACRONYM")
    out.department, out.department_confidence, out.department_source = dept, dept_conf, source
    spec, spec_conf, used = _specialty(ct, dept, frame, why)
    out.specialty, out.specialty_confidence = spec, spec_conf
    out.scope, out.scope_type, out.unresolved_qualifiers = _scope(ct, used, why)

    if ct.typo_corrected:
        reasons.append("TYPO_CORRECTED")
        out.level_confidence *= 0.9
        out.department_confidence *= 0.9
    if ct.contractor:
        why.append("employment: contractor marker")
    if ct.interim:
        why.append("employment: interim/acting appointment")
    out.parse_confidence = round(min(out.level_confidence, out.department_confidence, out.management_confidence), 4)
    out.level_confidence = round(out.level_confidence, 4)
    out.department_confidence = round(out.department_confidence, 4)
    return out
