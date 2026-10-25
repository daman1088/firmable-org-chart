"""Ask the Org Chart: question parsing and answers on a small hand-built company."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from ask import (ABOVE, AMBIGUOUS_MATCH, ANSWER, DEPARTMENT, INFERRED, MANAGER, NOT_FOUND,  # noqa: E402
                 NOT_IDENTIFIED, UNSUPPORTED, ask, examples, parse)

PEOPLE = pd.DataFrame([
    # employee_id, full_name, title_raw, department, level, manages_people, is_non_employee
    ("C", "Ada King", "Chief Executive Officer", "executive", 9, True, False),
    ("V", "Ben Ortiz", "VP Sales", "sales", 8, True, False),
    ("S1", "Cara Diaz", "Sales Manager", "sales", 6, True, False),
    ("S2", "Dan Webb", "Account Executive", "sales", 3, False, False),
    ("S3", "Eve Webb", "Account Executive", "sales", 3, False, False),
    ("S4", "Dan Moss", "Sales Development Rep", "sales", 2, False, False),
    ("E1", "Finn Lee", "Head of Engineering", "engineering", 8, True, False),
    ("E2", "Gia Park", "Director of Engineering", "engineering", 8, True, False),
    ("X", "Hal Fox", "Board Member", "executive", np.nan, False, True),
], columns=["employee_id", "full_name", "title_raw", "department", "level", "manages_people", "is_non_employee"]
).assign(industry="Software Development", company_name="Acme")

VIEW = pd.DataFrame([
    # employee_id, view_status, view_manager, view_confidence, ancestor_id, confidence, candidate_ids, reasons
    ("C", "UNRESOLVED", None, np.nan, None, np.nan, [], ["TOP_OF_COMPANY"]),
    ("V", "DIRECT", "C", 0.93, None, 0.93, [], []),
    ("S1", "ANCESTOR", None, 0.97, "C", 0.97, [], ["DIRECT_UNCLEAR"]),
    ("S2", "DIRECT", "S1", 0.88, None, 0.88, [], []),
    ("S3", "AMBIGUOUS", None, 0.91, None, 0.91, ["S1", "V"], []),
    ("S4", "UNRESOLVED", None, np.nan, None, np.nan, [], ["NO_CANDIDATE"]),
    ("E1", "ANCESTOR", None, 0.96, "C", 0.96, [], []),
    ("E2", "UNRESOLVED", None, np.nan, None, np.nan, [], []),
    ("X", "UNRESOLVED", None, np.nan, None, np.nan, [], ["NON_EMPLOYEE"]),
], columns=["employee_id", "view_status", "view_manager", "view_confidence", "ancestor_id", "confidence",
            "candidate_ids", "reason_codes"])


def q(question: str):
    return ask(question, PEOPLE, VIEW)


@pytest.mark.parametrize(("question", "kind", "subject"), [
    ("Who runs the Sales team?", DEPARTMENT, "Sales"),
    ("who leads engineering", DEPARTMENT, "engineering"),
    ("Who heads the customer success department?", DEPARTMENT, "customer success"),
    ("Who is the head of Finance?", DEPARTMENT, "Finance"),
    ("Who manages the supply chain team?", DEPARTMENT, "supply chain"),
    ("Who does Dan Webb report to?", MANAGER, "Dan Webb"),
    ("who does  dan webb report to", MANAGER, "dan webb"),
    ("Who is Dan Webb's manager?", MANAGER, "Dan Webb"),
    ("Who's Chris’ boss?", MANAGER, "Chris"),
    ("Who manages Dan Webb?", MANAGER, "Dan Webb"),
    ("Who is above Dan Webb?", ABOVE, "Dan Webb"),
    ("Who's above S2", ABOVE, "S2"),
    ("Who is in Dan Webb's reporting chain?", ABOVE, "Dan Webb"),
])
def test_parse_supported(question, kind, subject):
    p = parse(question)
    assert p is not None and (p.kind, p.subject) == (kind, subject)


@pytest.mark.parametrize("question", ["", "What is the weather?", "Who is Dan Webb?", "List everyone",
                                      "Who does report to?"])
def test_unsupported_questions_get_help(question):
    a = q(question)
    assert a.outcome == UNSUPPORTED and "three kinds of question" in a.notes[0]


def test_direct_manager():
    a = q("Who does Dan Webb report to?")
    assert a.outcome == ANSWER and "reports to **Cara Diaz**" in a.headline
    assert [(r["name"], r["confidence"]) for r in a.rows] == [("Cara Diaz", 0.88)]
    assert a.rows[0]["relationship"].startswith("DIRECT")


def test_ancestor_is_not_called_the_manager():
    a = q("Who does Cara Diaz report to?")
    assert a.outcome == NOT_IDENTIFIED and "not confidently identified" in a.headline
    assert "reports to" not in a.headline and "Ada King" not in a.headline
    (row,) = a.rows
    assert row["name"] == "Ada King" and row["relationship"].startswith("ANCESTOR")
    assert "may be the direct manager or someone higher" in row["relationship"]
    assert "no direct manager is claimed" in a.notes[0]


def test_ambiguous_manager_lists_the_set_without_picking_one():
    a = q("Who is Eve Webb's manager?")
    assert a.outcome == NOT_IDENTIFIED and "not identified as one person" in a.headline
    assert {r["name"] for r in a.rows} == {"Cara Diaz", "Ben Ortiz"}
    assert all(r["confidence"] is None and r["relationship"].startswith("AMBIGUOUS") for r in a.rows)
    assert "0.91" in a.notes[0]


def test_unresolved_manager_names_nobody():
    a = q("Who does Dan Moss report to?")
    assert a.outcome == NOT_IDENTIFIED and "not identified" in a.headline and not a.rows
    assert "NO_CANDIDATE" in a.notes[0]


def test_below_threshold_is_not_a_claim():
    view = VIEW.assign(view_status=VIEW["view_status"].replace({"DIRECT": "BELOW THRESHOLD"}))
    a = ask("Who does Dan Webb report to?", PEOPLE, view)
    assert a.outcome == NOT_IDENTIFIED and not a.rows and "custom threshold" in a.notes[0]


def test_chain_above_keeps_direct_and_ancestor_steps_apart():
    a = q("Who is above Dan Webb?")
    assert a.outcome == ANSWER and a.headline == "2 people are identified above Dan Webb: Cara Diaz → Ada King."
    assert [(r["step"], r["name"], r["relationship"][:8]) for r in a.rows] == [(1, "Cara Diaz", "DIRECT: "),
                                                                                (2, "Ada King", "ANCESTOR")]
    assert any("people in between" in n for n in a.notes)
    assert any("No confident claim above Ada King" in n for n in a.notes)


def test_chain_starting_with_an_ancestor_says_the_manager_is_unknown():
    a = q("Who is above Cara Diaz?")
    assert [r["name"] for r in a.rows] == ["Ada King"]
    assert "direct manager is not identified" in a.notes[0] and "possibly the direct manager" in a.notes[0]


def test_chain_stops_at_an_ambiguous_step_and_at_the_top():
    a = q("Who is above Eve Webb?")
    assert a.outcome == NOT_IDENTIFIED and not a.rows and "one of Cara Diaz, Ben Ortiz" in a.notes[0]
    assert q("Who is above Ada King?").outcome == NOT_IDENTIFIED


def test_people_are_found_by_id_name_or_words():
    assert q("Who does s2 report to?").rows[0]["name"] == "Cara Diaz"
    assert q("Who does dan WEBB report to?").outcome == ANSWER
    a = q("Who does Webb report to?")
    assert a.outcome == AMBIGUOUS_MATCH and {r["name"] for r in a.rows} == {"Dan Webb", "Eve Webb"}
    assert "full name or an employee ID" in a.headline
    assert {r["name"] for r in q("Who is above Dan?").rows} == {"Dan Webb", "Dan Moss"}


def test_unknown_person_points_to_other_companies():
    other = pd.DataFrame({"employee_id": ["Z1"], "full_name": ["Zoe Nash"], "title_raw": ["CFO"],
                          "company_name": ["Globex"]})
    a = ask("Who does Zoe Nash report to?", PEOPLE, VIEW, everyone=pd.concat([PEOPLE, other]))
    assert a.outcome == NOT_FOUND and "Zoe Nash" in a.headline and "Globex" in a.notes[0]
    assert q("Who does Zoe Nash report to?").outcome == NOT_FOUND


def test_department_leader_is_inferred_not_claimed():
    a = q("Who runs the Sales team?")
    assert a.outcome == INFERRED and a.headline.startswith("**Ben Ortiz** (VP Sales) is the **inferred** leader")
    (row,) = a.rows
    assert row["relationship"].startswith("Inferred leader") and row["confidence"] is None and row["level"] == 8
    assert row["DIRECT claims naming them as manager"] == 0
    assert "not a reporting-line claim" in a.notes[0]
    assert "No DIRECT claim names Ben Ortiz" in a.notes[1]


def test_department_tie_names_no_single_leader():
    a = q("Who leads engineering?")
    assert a.outcome == INFERRED and "No single leader" in a.headline
    assert {r["name"] for r in a.rows} == {"Finn Lee", "Gia Park"}


def test_department_aliases_and_unknown_departments():
    assert q("Who runs the revenue team?").rows[0]["name"] == "Ben Ortiz"  # lexicon label: revenue -> sales
    a = q("Who runs the Legal team?")
    assert a.outcome == NOT_FOUND and "Engineering" in a.notes[0] and "Sales" in a.notes[0]


def test_department_leader_ignores_non_employees():
    a = q("Who runs the executive team?")
    assert [r["name"] for r in a.rows] == ["Ada King"]


def test_manager_answers_only_repeat_existing_claims():
    v = VIEW.set_index("employee_id")
    for eid in VIEW["employee_id"]:
        a = q(f"Who does {eid} report to?")
        if v.at[eid, "view_status"] == "DIRECT":
            assert a.outcome == ANSWER and [r["employee_id"] for r in a.rows] == [v.at[eid, "view_manager"]]
        else:
            assert a.outcome != ANSWER and "reports to" not in a.headline


def test_examples_are_answerable():
    ex = examples(PEOPLE, VIEW)
    assert [parse(e).kind for e in ex] == [DEPARTMENT, MANAGER, ABOVE]
    assert all(q(e).outcome not in (NOT_FOUND, UNSUPPORTED, AMBIGUOUS_MATCH) for e in ex)
