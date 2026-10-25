"""Ask the Org Chart: answers to three fixed kinds of question, from predictions and normalised titles.

Deterministic: questions are matched with regular expressions, people and departments by plain text
matching. No language model, no external service and no new data. Every reporting statement is read from
the decisions in the current view (the same claims the chart draws), so nothing is said that the chart
does not show:

* "Who does X report to?"  DIRECT names the manager. AMBIGUOUS gives the candidate set. ANCESTOR says the
  direct manager is not identified and names the person somewhere above. Anything else: not identified.
* "Who is above X?"        follows DIRECT and ANCESTOR claims upwards, one claim per step.
* "Who runs the X team?"   the most senior person in the department by normalised job title. This is an
  inference from titles, not a reporting claim, and is labelled as one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from orgchart.normalization.lexicon import DEPT_LABELS, INDUSTRY_LABELS

DEPARTMENT, MANAGER, ABOVE = "department", "manager", "above"

# Outcomes, from most to least definite. The UI picks a callout style from these.
ANSWER = "answer"  # a reporting claim from the model (DIRECT, or a chain of claims)
INFERRED = "inferred"  # a department leader inferred from job titles; not a reporting claim
NOT_IDENTIFIED = "not_identified"  # the person was found, but the model makes no direct-manager claim
AMBIGUOUS_MATCH = "ambiguous_match"  # several people or departments match the text
NOT_FOUND = "not_found"
UNSUPPORTED = "unsupported"

MAX_STEPS = 25
MAX_LISTED = 8
_TAIL = r"(?:\s+(?:team|department|dept|function|group|org))?"
PATTERNS = (  # tried in order; the subject is group "x"
    (DEPARTMENT, rf"who\s+(?:runs|leads|heads)\s+(?:the\s+)?(?P<x>.+?){_TAIL}"),
    (DEPARTMENT, rf"who\s+is\s+(?:the\s+)?(?:head|leader|lead)\s+of\s+(?:the\s+)?(?P<x>.+?){_TAIL}"),
    (DEPARTMENT, r"who\s+manages\s+the\s+(?P<x>.+?)\s+(?:team|department|dept|function|group|org)"),
    (MANAGER, r"who\s+does\s+(?P<x>.+?)\s+report\s+to"),
    (MANAGER, r"who\s+is\s+(?P<x>.+?)'s?\s+(?:direct\s+)?(?:manager|boss|line\s+manager)"),
    (MANAGER, r"who\s+manages\s+(?P<x>.+?)"),
    (ABOVE, r"who\s+(?:is|are)\s+above\s+(?P<x>.+?)"),
    (ABOVE, r"who\s+is\s+in\s+(?P<x>.+?)'s?\s+(?:reporting\s+)?chain"),
    (ABOVE, r"(?:show\s+)?(?P<x>.+?)'s?\s+reporting\s+chain"),
)
HELP = ("I can answer three kinds of question about the selected company: “Who runs the Sales team?”, "
        "“Who does Jane Smith report to?” and “Who is above Jane Smith?”. Use a full name or an employee ID.")


@dataclass
class Parsed:
    kind: str
    subject: str


@dataclass
class Answer:
    kind: str | None
    outcome: str
    headline: str
    rows: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def parse(question: str) -> Parsed | None:
    """The kind of question and its subject (a name, ID or department), or None if it is not supported."""
    q = re.sub(r"\s+", " ", question.replace("’", "'")).strip().rstrip("?.! ")
    q = re.sub(r"(?i)\bwho's\b", "who is", q)
    for kind, pattern in PATTERNS:
        m = re.fullmatch(pattern, q, flags=re.IGNORECASE)
        if m and m.group("x").strip(" \"'"):
            return Parsed(kind, m.group("x").strip(" \"'"))
    return None


def ask(question: str, people: pd.DataFrame, view: pd.DataFrame, everyone: pd.DataFrame | None = None) -> Answer:
    """Answer a question about one company. `people` and `view` are that company's profiles and decisions;
    `everyone` (all companies' profiles) is only used to say where an unknown name can be found."""
    parsed = parse(question)
    if parsed is None:
        return Answer(None, UNSUPPORTED, "Sorry, I can't answer that question.", notes=[HELP])
    if parsed.kind == DEPARTMENT:
        return ask_department(people, view, parsed.subject)
    found = _resolve_person(people, parsed.kind, parsed.subject, everyone)
    if isinstance(found, Answer):
        return found
    return (ask_manager if parsed.kind == MANAGER else ask_above)(people, view, found)


# --- people and departments --------------------------------------------------------------------------

def _norm(text: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def find_people(people: pd.DataFrame, text: str) -> pd.DataFrame:
    """Profiles matching an employee ID, else a full name, else every word of the text as whole words."""
    by_id = people[people["employee_id"].str.lower() == text.strip().lower()]
    if len(by_id):
        return by_id
    q = _norm(text)
    if not q:
        return people.iloc[0:0]
    names = people["full_name"].map(_norm)
    exact = people[names == q]
    if len(exact):
        return exact
    words = q.split()
    return people[names.map(lambda n: all(w in n.split() for w in words))]


def _person(people: pd.DataFrame, eid: str) -> tuple[str, str]:
    row = people.loc[people["employee_id"] == eid]
    if row.empty:
        return eid, ""
    title = row["title_raw"].iloc[0]
    return row["full_name"].iloc[0], title if isinstance(title, str) and title else "(no title)"


def _row(people: pd.DataFrame, eid: str, relationship: str, p: object = None, **extra: object) -> dict:
    name, title = _person(people, eid)
    return {"name": name, "title": title, "relationship": relationship,
            "confidence": round(float(p), 2) if isinstance(p, (int, float)) and pd.notna(p) else None,
            **extra, "employee_id": eid}


def _resolve_person(people: pd.DataFrame, kind: str, text: str, everyone: pd.DataFrame | None) -> str | Answer:
    hits = find_people(people, text)
    if len(hits) == 1:
        return hits["employee_id"].iloc[0]
    if len(hits) > 1:
        rows = [_row(people, e, "matches the name") for e in hits["employee_id"].head(MAX_LISTED)]
        more = f" (showing {MAX_LISTED})" if len(hits) > MAX_LISTED else ""
        return Answer(kind, AMBIGUOUS_MATCH, f"{len(hits)} people match “{text}”{more}. Ask again with a full name "
                      "or an employee ID.", rows)
    notes = []
    if everyone is not None:
        elsewhere = find_people(everyone, text).head(5)
        if len(elsewhere):
            notes.append("Found in other companies: " + "; ".join(
                f"{r.full_name} ({r.title_raw}) at {r.company_name}" for r in elsewhere.itertuples())
                + ". Choose that company in the sidebar.")
    return Answer(kind, NOT_FOUND, f"No one called “{text}” in this company.", notes=notes or [HELP])


def _industry_key(people: pd.DataFrame) -> str:
    raw = people["industry"].iloc[0] if "industry" in people and len(people) else ""
    return INDUSTRY_LABELS.get(str(raw).lower(), "general")


def find_departments(people: pd.DataFrame, text: str) -> list[str]:
    """Departments of this company matching the text: by name, by a common label ("HR", "customer
    support"), or by partial name."""
    present = sorted(people["department"].dropna().unique())
    q = re.sub(r"^the ", "", _norm(text))
    q = re.sub(r" (team|department|dept|function|group|org)$", "", q)
    exact = [d for d in present if _norm(d) == q]
    if exact:
        return exact
    for label, dept in DEPT_LABELS.items():
        if _norm(label) == q:
            dept = dept if isinstance(dept, str) else dept.get(_industry_key(people), dept["*"])
            if dept in present:
                return [dept]
    return [d for d in present if q and (q in _norm(d) or _norm(d) in q)]


# --- answers ------------------------------------------------------------------------------------------

def _decisions(view: pd.DataFrame) -> pd.DataFrame:
    return view.set_index("employee_id")


def ask_manager(people: pd.DataFrame, view: pd.DataFrame, eid: str) -> Answer:
    name, _ = _person(people, eid)
    v = _decisions(view)
    if eid not in v.index:
        return Answer(MANAGER, NOT_IDENTIFIED, f"{name}'s direct manager is **not identified**: there is no "
                      "prediction for this profile.")
    r = v.loc[eid]
    s = r["view_status"]
    if s == "DIRECT":
        boss, title = _person(people, r["view_manager"])
        return Answer(MANAGER, ANSWER, f"{name} reports to **{boss}** ({title}).",
                      [_row(people, r["view_manager"], "DIRECT: direct manager", r["view_confidence"])],
                      [f"DIRECT claim: calibrated probability {r['view_confidence']:.2f} that {boss} is "
                       f"{name}'s direct manager."])
    if s == "ANCESTOR":
        above, _ = _person(people, r["ancestor_id"])
        return Answer(MANAGER, NOT_IDENTIFIED, f"{name}'s direct manager is **not confidently identified**.",
                      [_row(people, r["ancestor_id"], "ANCESTOR: in the reporting chain above "
                            "(may be the direct manager or someone higher)", r["confidence"])],
                      [f"The model is confident that {above} is somewhere above {name} in the reporting chain "
                       f"(p = {r['confidence']:.2f}), but not whether {above} is the direct manager or someone "
                       "higher, so no direct manager is claimed."])
    if s == "AMBIGUOUS":
        cands = list(r["candidate_ids"])
        return Answer(MANAGER, NOT_IDENTIFIED, f"{name}'s direct manager is **not identified as one person**. "
                      f"It is one of {len(cands)} candidates.",
                      [_row(people, c, "AMBIGUOUS: one of the candidates") for c in cands],
                      [f"p = {r['confidence']:.2f} that the direct manager is one of these {len(cands)} people. "
                       "No single manager is claimed."])
    notes = [f"Reasons: {', '.join(r['reason_codes'])}."] if len(r["reason_codes"]) else []
    if s == "BELOW THRESHOLD":
        notes.insert(0, "The model's DIRECT claim is hidden because its probability is below the custom threshold.")
    return Answer(MANAGER, NOT_IDENTIFIED, f"{name}'s direct manager is **not identified**: no claim reached the "
                  "threshold.", notes=notes)


def ask_above(people: pd.DataFrame, view: pd.DataFrame, eid: str) -> Answer:
    """The chain of claims above an employee: each step is the DIRECT manager or an ANCESTOR of the previous
    person. Stops where the next step is not claimed."""
    name, _ = _person(people, eid)
    v = _decisions(view)
    rows: list[dict] = []
    cur, seen, stop = eid, {eid}, ""
    while len(rows) < MAX_STEPS:
        if cur not in v.index:
            stop = f"There is no prediction for {_person(people, cur)[0]}."
            break
        r = v.loc[cur]
        below = _person(people, cur)[0]
        if r["view_status"] == "DIRECT":
            nxt, rel, p = r["view_manager"], f"DIRECT: direct manager of {below}", r["view_confidence"]
        elif r["view_status"] == "ANCESTOR":
            nxt, p = r["ancestor_id"], r["confidence"]
            rel = f"ANCESTOR: somewhere above {below} (direct manager of {below} not identified)"
        else:
            if r["view_status"] == "AMBIGUOUS":
                options = ", ".join(_person(people, c)[0] for c in r["candidate_ids"])
                stop = f"{below}'s direct manager is one of {options}, so the chain is not followed further."
            else:
                stop = f"No confident claim above {below}."
            break
        if not isinstance(nxt, str) or nxt in seen:
            break
        rows.append(_row(people, nxt, rel, p, step=len(rows) + 1))
        seen.add(nxt)
        cur = nxt
    if not rows:
        return Answer(ABOVE, NOT_IDENTIFIED, f"No one above {name} is identified with confidence.",
                      notes=[stop] if stop else [])
    chain = " → ".join(row["name"] for row in rows)
    notes = ["Each step is a separate model claim with its own probability. After an ANCESTOR step there may be "
             "people in between who are not listed."]
    if rows[0]["relationship"].startswith("ANCESTOR"):
        notes.insert(0, f"{name}'s direct manager is not identified. {rows[0]['name']} is somewhere above "
                        f"{name}, possibly the direct manager, possibly higher.")
    if stop:
        notes.append(stop)
    people_word = "person is" if len(rows) == 1 else "people are"
    return Answer(ABOVE, ANSWER, f"{len(rows)} {people_word} identified above {name}: {chain}.", rows, notes)


def ask_department(people: pd.DataFrame, view: pd.DataFrame, text: str) -> Answer:
    depts = find_departments(people, text)
    present = sorted(people["department"].dropna().unique())
    if not depts:
        return Answer(DEPARTMENT, NOT_FOUND, f"No department matching “{text}” in this company.",
                      notes=["Departments here: " + ", ".join(_label(d) for d in present) + "."])
    if len(depts) > 1:
        return Answer(DEPARTMENT, AMBIGUOUS_MATCH, f"“{text}” matches {len(depts)} departments: "
                      + ", ".join(_label(d) for d in depts) + ". Ask about one of them.")
    dept = depts[0]
    staff = people[(people["department"] == dept) & ~people["is_non_employee"].fillna(False).astype(bool)]
    staff = staff[staff["level"].notna()]
    if staff.empty:
        return Answer(DEPARTMENT, NOT_IDENTIFIED, f"No leader can be inferred for {_label(dept)}: no job title "
                      "there has a known seniority level.")
    top = staff["level"].max()
    leaders = (staff[staff["level"] == top]
               .sort_values(["manages_people", "full_name"], ascending=[False, True]))
    direct = view.loc[view["view_status"] == "DIRECT", "view_manager"].value_counts()
    linked = view.loc[view["view_status"] == "ANCESTOR", "ancestor_id"].value_counts()
    rows = [_row(people, e, "Inferred leader: most senior job title in the department", None,
                 level=int(top), **{"DIRECT claims naming them as manager": int(direct.get(e, 0)),
                                    "ANCESTOR claims (in chain above)": int(linked.get(e, 0))})
            for e in leaders["employee_id"].head(MAX_LISTED)]
    basis = (f"Inferred from normalised job titles (seniority level {int(top)}, the highest among the {len(staff)} "
             f"people in {_label(dept)}). This is not a reporting-line claim: the model predicts managers, not department "
             "heads, and someone outside the department (an executive, say) may oversee it.")
    if len(leaders) > 1:
        return Answer(DEPARTMENT, INFERRED, f"No single leader of {_label(dept)} can be inferred: {len(leaders)} "
                      "people share the most senior job-title level.", rows, [basis])
    lead = rows[0]
    claims = lead["DIRECT claims naming them as manager"]
    support = (f"Separately, the model claims {lead['name']} is the DIRECT manager of {claims} "
               f"{'person' if claims == 1 else 'people'}." if claims else
               f"No DIRECT claim names {lead['name']} as anyone's manager at this precision target.")
    return Answer(DEPARTMENT, INFERRED, f"**{lead['name']}** ({lead['title']}) is the **inferred** leader of "
                  f"{_label(dept)}: the most senior job title there.", rows, [basis, support])


def _label(dept: str) -> str:
    return "IT" if dept == "it" else dept.replace("_", " ").title()


def examples(people: pd.DataFrame, view: pd.DataFrame) -> list[str]:
    """One example of each question type, using this company's own departments and people."""
    unique = people[~people["full_name"].duplicated(keep=False)]
    depts = people.loc[people["department"] != "executive", "department"].value_counts()
    dept = _label(depts.index[0]) if len(depts) else "Sales"

    def someone(status: str) -> str | None:
        ids = view.loc[view["view_status"] == status, "employee_id"]
        hit = unique[unique["employee_id"].isin(ids)]
        return hit["full_name"].iloc[0] if len(hit) else None

    first = someone("DIRECT") or someone("ANCESTOR") or (unique["full_name"].iloc[0] if len(unique) else "Jane Smith")
    second = someone("ANCESTOR") or first
    return [f"Who runs the {dept} team?", f"Who does {first} report to?", f"Who is above {second}?"]
