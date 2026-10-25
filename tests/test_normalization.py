"""Unit tests for title normalisation (observed inputs only; no dataset needed)."""

import dataclasses

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from orgchart.normalization.clean import clean_title, company_short_name
from orgchart.normalization.company_context import MIN_SUPPORT, apply_company_context
from orgchart.normalization.parser import industry_key, parse_title

SAAS, FIN, CONS, HEALTH, RETAIL, MFG = (
    "Software Development", "Financial Services", "Management Consulting", "Hospitals and Health Care",
    "Retail", "Manufacturing",
)

# (title, department field, industry, level, manages_people, department, track)
CASES = [
    # Technical seniority is not management authority.
    ("Staff Software Engineer", None, SAAS, 4, False, "engineering", "ic"),
    ("Principal Engineer", "Engineering", SAAS, 4, False, "engineering", "ic"),
    ("Software Engineer", None, SAAS, 2, False, "engineering", "ic"),
    ("Engineering Manager", None, SAAS, 5, True, "engineering", "manager"),
    ("Senior Engineering Manager", None, SAAS, 6, True, "engineering", "manager"),
    # 'Manager' in the noun of an IC role.
    ("Product Manager", None, SAAS, 2, False, "product", "ic"),
    ("Customer Success Manager", None, SAAS, 2, False, "customer_success", "ic"),
    ("Group Product Manager", None, SAAS, 5, True, "product", "manager"),
    # Abbreviations.
    ("Sr. Software Engineer", None, SAAS, 3, False, "engineering", "ic"),
    ("Snr Software Engr II", None, SAAS, 3, False, "engineering", "ic"),
    ("SWE", None, SAAS, 2, False, "engineering", "ic"),
    ("VP Engineering", None, SAAS, 9, True, "engineering", "manager"),
    ("SVP, Sales", None, SAAS, 10, True, "sales", "manager"),
    ("Mgr, Finance", None, RETAIL, 5, True, "finance", "manager"),
    # Executives.
    ("Director of Marketing", None, SAAS, 7, True, "marketing", "manager"),
    ("CEO", None, SAAS, 12, True, "executive", "executive"),
    ("CFO", None, SAAS, 11, True, "finance", "executive"),
    # Fintech: CRO is risk or revenue; the department field decides.
    ("CRO", "Risk", FIN, 11, True, "risk_compliance", "executive"),
    ("CRO", "Sales", FIN, 11, True, "sales", "executive"),
    # Consulting grades.
    ("Analyst", None, CONS, 1, False, "consulting", "ic"),
    ("Consultant", None, CONS, 2, False, "consulting", "ic"),
    ("Senior Consultant", None, CONS, 3, False, "consulting", "ic"),
    ("Manager", None, CONS, 5, True, "consulting", "manager"),
    ("Principal", None, CONS, 8, True, "consulting", "manager"),
    ("Partner", None, CONS, 9, True, "consulting", "manager"),
    # Healthcare nursing ladder.
    ("Registered Nurse", None, HEALTH, 2, False, "clinical", "ic"),
    ("Clinical Nurse Specialist", None, HEALTH, 4, False, "clinical", "ic"),
    ("Nurse Unit Manager", None, HEALTH, 5, True, "clinical", "manager"),
    ("Director of Nursing", None, HEALTH, 7, True, "clinical", "manager"),
    # Retail store hierarchy.
    ("Sales Assistant", None, RETAIL, 1, False, "retail_operations", "ic"),
    ("Assistant Store Manager", None, RETAIL, 4, True, "retail_operations", "manager"),
    ("Store Manager", None, RETAIL, 5, True, "retail_operations", "manager"),
    ("Area Manager", None, RETAIL, 6, True, "retail_operations", "manager"),
    # Manufacturing.
    ("Machine Operator", None, MFG, 2, False, "production", "ic"),
    ("Production Supervisor", None, MFG, 4, True, "production", "manager"),
    ("Plant Manager", None, MFG, 7, True, "production", "manager"),
    ("Graduate Engineer", None, MFG, 1, False, "hardware_engineering", "ic"),
    # Support roles near executives carry no reports.
    ("Executive Assistant to the CEO", None, SAAS, 2, False, "executive", "support"),
    ("Chief of Staff", None, SAAS, 7, False, "executive", "support"),
]


@pytest.mark.parametrize(("title", "dept", "industry", "level", "manages", "department", "track"), CASES,
                         ids=[f"{c[0]}|{c[1]}|{c[2]}" for c in CASES])
def test_parse_table(title, dept, industry, level, manages, department, track):
    p = parse_title(title, dept, industry)
    assert (p.level, p.manages_people, p.department, p.track) == (level, manages, department, track), p.explanation


@pytest.mark.parametrize("title", ["Staff Software Engineer", "Principal Engineer", "Staff Data Scientist",
                                   "Principal Product Designer", "Senior Staff Engineer"])
def test_staff_and_principal_are_not_managers(title):
    p = parse_title(title, None, SAAS)
    assert p.manages_people is False and p.level == 4
    assert any("not a people-management signal" in line for line in p.explanation)


def test_ambiguous_acronym_without_field_is_flagged_and_low_confidence():
    p = parse_title("CRO", None, FIN)
    assert "AMBIGUOUS_ACRONYM" in p.reason_codes
    assert p.department_confidence <= 0.5
    resolved = parse_title("CRO", "Risk", FIN)
    assert "AMBIGUOUS_ACRONYM" not in resolved.reason_codes
    assert resolved.department_confidence > p.department_confidence


def test_industry_changes_meaning_of_pm():
    assert parse_title("PM", None, SAAS).department == "product"
    assert parse_title("PM", None, "Construction").department != "product"


def test_industry_key_mapping():
    assert industry_key(SAAS) == "saas"
    assert industry_key(FIN) == "fintech"
    assert industry_key(CONS) == "consulting"
    assert industry_key(None) == "general"
    assert industry_key("Something Unheard Of") == "general"


def test_department_field_used_when_title_has_no_function():
    p = parse_title("Team Lead", "Marketing", SAAS)
    assert p.department == "marketing" and p.department_source == "field"


def test_missing_department_falls_back_to_title():
    p = parse_title("Payroll Officer", None, RETAIL)
    assert p.department == "finance" and p.department_source == "title"
    assert p.specialty == "Payroll"


def test_title_and_field_agreement_raises_confidence():
    alone = parse_title("Software Engineer", None, SAAS)
    agreed = parse_title("Software Engineer", "Engineering", SAAS)
    assert agreed.department_source == "title+field"
    assert agreed.department_confidence > alone.department_confidence


def test_conflicting_field_is_reported():
    p = parse_title("Recruiter", "Engineering", SAAS)
    assert p.department == "people" and "DEPT_CONFLICT" in p.reason_codes


@pytest.mark.parametrize("title", ["Manager", "Team Member", "Associate"])
def test_vague_titles_flagged(title):
    p = parse_title(title, None, SAAS)
    assert "VAGUE_TITLE" in p.reason_codes
    assert p.parse_confidence <= 0.5


def test_empty_title_unresolved():
    p = parse_title("", None, RETAIL)
    assert p.level is None and "EMPTY_TITLE" in p.reason_codes and p.parse_confidence == 0.0
    assert parse_title(None, None, RETAIL).level is None


@pytest.mark.parametrize("title", ["Board Member", "Non-Executive Director", "Advisor", "Chairman of the Board"])
def test_board_and_advisors_are_non_employees(title):
    p = parse_title(title, None, SAAS)
    assert p.is_non_employee and p.track == "board" and p.manages_people is None


def test_flags_contractor_interim_founder_intern():
    assert parse_title("Senior Software Engineer (Contract)", None, SAAS).is_contractor
    interim = parse_title("Interim CFO", None, SAAS)
    assert interim.is_interim and interim.level == 11
    founder = parse_title("Co-Founder & CTO", None, SAAS)
    assert founder.is_founder and founder.level == 11 and founder.department == "engineering"
    intern = parse_title("Software Engineering Intern", None, SAAS)
    assert intern.is_intern and intern.level == 0


def test_typo_repair_lowers_confidence_and_is_explained():
    clean = parse_title("Software Engineer", None, SAAS)
    typo = parse_title("Sofwtare Engineer", None, SAAS)
    assert typo.level == clean.level and typo.department == clean.department
    assert "TYPO_CORRECTED" in typo.reason_codes
    assert typo.parse_confidence < clean.parse_confidence
    assert any("repaired typo" in line for line in typo.explanation)


def test_region_scope_from_qualifier():
    assert parse_title("Sales Manager - APAC", None, SAAS).scope == "APAC"
    p = parse_title("Regional Sales Director, EMEA", None, SAAS)
    assert (p.scope, p.scope_type, p.level) == ("EMEA", "region", 7)


def test_decorations_stripped():
    p = parse_title("Senior Data Engineer @ Acme", None, SAAS)
    assert (p.level, p.department) == (3, "engineering")
    ct = clean_title("Product Designer at Acme Pty Ltd", "Acme Pty Ltd", "saas")
    assert ct.main == ["product", "designer"]
    assert company_short_name("Acme Pty Ltd") == "Acme"


def test_clean_splits_qualifiers_and_expands():
    ct = clean_title("Sr Mgr :: Payments (Contract)", None, "fintech")
    assert ct.main == ["senior", "manager"]
    assert ct.qualifiers == [["payments"]]
    assert ct.contractor
    assert any("expanded" in n for n in ct.notes)
    assert clean_title("Human Resources Business Partner").main == ["hr", "business", "partner"]


def test_hr_business_partner_is_not_a_partner_grade():
    p = parse_title("Human Resources Business Partner", None, CONS)
    assert p.manages_people is False and p.department == "people"


def test_company_context_promotes_recurring_qualifier():
    titles = [f"{role}, Aged Care" for role in ("Registered Nurse", "Care Coordinator", "Nurse Unit Manager")]
    parsed = [parse_title(t, None, HEALTH) for t in titles] + [parse_title("Registered Nurse, Oncology Ward X", None,
                                                                           HEALTH)]
    assert all(p.scope is None for p in parsed[:MIN_SUPPORT])
    out = apply_company_context(parsed)
    assert [p.scope_type for p in out[:MIN_SUPPORT]] == ["unit"] * MIN_SUPPORT
    assert out[0].scope == "Aged Care" and "company.recurring_qualifier" in out[0].rules_fired
    assert out[3].scope_type is None  # qualifier seen once: not promoted
    # The cached parse objects are not mutated.
    assert parse_title(titles[0], None, HEALTH).scope is None


def test_company_context_needs_min_support():
    parsed = [parse_title("Registered Nurse, Aged Care", None, HEALTH)] * (MIN_SUPPORT - 1)
    assert all(p.scope is None for p in apply_company_context(parsed))


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_every_result_is_explained_and_bounded(case):
    p = parse_title(case[0], case[1], case[2])
    assert p.explanation and all(isinstance(line, str) and line for line in p.explanation)
    assert p.rules_fired
    for conf in (p.level_confidence, p.management_confidence, p.department_confidence, p.specialty_confidence,
                 p.parse_confidence):
        assert 0.0 <= conf <= 1.0
    assert p.parse_confidence <= min(p.level_confidence, p.department_confidence, p.management_confidence) + 1e-9


def test_parse_is_deterministic_and_cached_result_is_safe_to_copy():
    a = parse_title("Senior Product Manager - Payments", "Product", FIN)
    b = parse_title("Senior Product Manager - Payments", "Product", FIN)
    assert dataclasses.asdict(a) == dataclasses.asdict(b)


@settings(max_examples=300, deadline=None)
@given(title=st.text(max_size=60), dept=st.one_of(st.none(), st.text(max_size=20)),
       industry=st.sampled_from([SAAS, FIN, CONS, HEALTH, RETAIL, MFG, None]))
def test_parser_never_crashes_and_stays_bounded(title, dept, industry):
    p = parse_title(title, dept, industry)
    assert p.level is None or 0 <= p.level <= 12
    assert 0.0 <= p.parse_confidence <= 1.0
    assert p.explanation
