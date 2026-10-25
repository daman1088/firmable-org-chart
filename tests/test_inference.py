"""Unit tests for reporting-line inference on a hand-built company (no dataset, no ground truth)."""

import numpy as np
import pandas as pd
import pytest

from orgchart.inference.decide import decide
from orgchart.inference.features import EMPLOYEE_LEVEL_FEATURES, FEATURES, build_candidates, company_people
from orgchart.inference.location import parse_location
from orgchart.inference.naive import naive_predictions
from orgchart.inference.scoring import Calibration, Model, Thresholds, rules_model
from orgchart.inference.validate import validate_predictions
from orgchart.learning.estimators import fit_conditional_logit, fit_isotonic, fit_logistic

SYD, MEL = "Sydney, New South Wales, Australia", "Melbourne, Victoria, Australia"

# id, title, level, track, manages, department, specialty, location, non_employee
ROWS = [
    ("ceo", "Chief Executive Officer", 12, "executive", True, "executive", None, SYD, False),
    ("vp", "VP Engineering", 9, "manager", True, "engineering", None, SYD, False),
    ("em_be", "Engineering Manager, Backend", 5, "manager", True, "engineering", "Backend", SYD, False),
    ("em_fe", "Engineering Manager, Frontend", 5, "manager", True, "engineering", "Frontend", MEL, False),
    ("be1", "Backend Engineer", 2, "ic", False, "engineering", "Backend", SYD, False),
    ("be2", "Senior Backend Engineer", 3, "ic", False, "engineering", "Backend", SYD, False),
    ("fe1", "Frontend Engineer", 2, "ic", False, "engineering", "Frontend", MEL, False),
    ("cfo", "CFO", 11, "executive", True, "finance", None, SYD, False),
    ("acct", "Accountant", 2, "ic", False, "finance", None, SYD, False),
    ("board", "Board Member", None, "board", None, None, None, None, True),
    ("blank", None, None, "unknown", None, None, None, SYD, False),
]


@pytest.fixture()
def people() -> pd.DataFrame:
    df = pd.DataFrame(ROWS, columns=["employee_id", "title_raw", "level", "track", "manages_people", "department",
                                     "specialty", "location", "is_non_employee"])
    df["company_id"] = "C9999"
    df["title_clean"] = df["title_raw"].str.lower()
    df["scope"] = None
    df["start_date"] = pd.NaT
    df["level"] = df["level"].astype("Int64")
    df["management_confidence"] = 0.8
    df["level_confidence"] = 0.8
    df["manages_people"] = df["manages_people"].astype("boolean")
    return company_people(df)


def _pairs(people):
    return build_candidates(people)


def test_candidates_respect_hard_constraints(people):
    pairs = _pairs(people)
    level = people.set_index("employee_id")["lvl"]
    assert (pairs["employee_id"] != pairs["candidate_id"]).all()
    assert (level[pairs["candidate_id"]].to_numpy() > level[pairs["employee_id"]].to_numpy()).all()
    assert not pairs["employee_id"].isin(["board", "blank", "ceo"]).any()
    assert not pairs["candidate_id"].isin(["board", "blank", "be1", "acct"]).any()
    assert set(FEATURES) <= set(pairs.columns)


def test_pair_features_capture_team_signals(people):
    p = _pairs(people).set_index(["employee_id", "candidate_id"])
    be_em = p.loc[("be1", "em_be")]
    assert be_em["spec_match"] == 1 and be_em["same_city"] == 1 and be_em["dept_match"] == 1
    assert p.loc[("be1", "em_fe")]["spec_mismatch"] == 1
    assert p.loc[("be1", "vp")]["log_between_dept"] > 0  # two engineering managers sit between
    assert p.loc[("be1", "em_be")]["log_between_dept"] == 0
    assert p.loc[("vp", "ceo")]["e_top_x_m_exec"] == 1
    assert p.loc[("acct", "vp")]["dept_mismatch"] == 1


def test_rules_model_ranks_the_obvious_manager_first(people):
    ranked = rules_model().score_pairs(_pairs(people))
    top = ranked[ranked["rank"] == 0].set_index("employee_id")["candidate_id"]
    assert top["be1"] == "em_be"
    assert top["fe1"] == "em_fe"
    assert top["em_be"] == "vp"
    assert top["vp"] == "ceo"
    assert top["acct"] == "cfo"
    q = ranked.groupby("employee_id")["q_direct"].sum()
    assert (q <= 1 + 1e-9).all()  # listwise probabilities never exceed 1 per employee


def _decide(people, thresholds):
    model = rules_model()
    model.thresholds = thresholds
    return decide(people, model.score_pairs(_pairs(people)), model)


def test_decide_statuses_and_reasons(people):
    pred, cands = _decide(people, Thresholds(direct=0.3, ambiguous=0.3, ancestor=0.3))
    by = pred.set_index("employee_id")
    assert len(pred) == len(people)
    assert by.loc["board", "status"] == "UNRESOLVED" and "NON_EMPLOYEE" in by.loc["board", "reason_codes"]
    assert "LEVEL_UNRESOLVED" in by.loc["blank", "reason_codes"]
    assert "TOP_OF_ORG" in by.loc["ceo", "reason_codes"]
    assert by.loc["be1", "status"] == "DIRECT" and by.loc["be1", "reports_to_id"] == "em_be"
    assert all(by.loc["be1", "explanation"])
    assert set(cands["employee_id"]) <= set(people["employee_id"])
    assert all(c["passed"] for c in validate_predictions(pred, people))


def test_never_forces_when_thresholds_unreachable(people):
    pred, _ = _decide(people, Thresholds(direct=1.01, ambiguous=1.01, ancestor=1.01))
    assert set(pred["status"]) == {"UNRESOLVED"}
    assert "INSUFFICIENT_EVIDENCE" in pred.set_index("employee_id").loc["be1", "reason_codes"]


def test_ancestor_fallback_excludes_ceo(people):
    pred, _ = _decide(people, Thresholds(direct=1.01, ambiguous=1.01, ancestor=0.5))
    anc = pred[pred["status"] == "ANCESTOR"]
    assert len(anc) and not (anc["ancestor_id"] == "ceo").any()
    assert all(c["passed"] for c in validate_predictions(pred, people))


def test_validator_detects_violations(people):
    pred, _ = _decide(people, Thresholds(direct=0.3, ambiguous=0.3, ancestor=0.3))

    def failed(mutate):
        bad = pred.copy()
        mutate(bad)
        return {c["name"] for c in validate_predictions(bad, people) if not c["passed"]}

    def self_loop(df):
        df.loc[df["employee_id"] == "be1", "reports_to_id"] = "be1"

    def cycle(df):
        row = df["employee_id"] == "ceo"
        df.loc[row, "status"] = "DIRECT"
        df.loc[row, "reports_to_id"] = "be1"

    def board_assigned(df):
        row = df["employee_id"] == "board"
        df.loc[row, "status"] = "DIRECT"
        df.loc[row, "reports_to_id"] = "ceo"

    def duplicate(df):
        df.loc[len(df)] = df.iloc[0]

    assert "no_self_reporting" in failed(self_loop)
    assert "manager_strictly_more_senior" in failed(cycle)
    assert "non_employees_unassigned" in failed(board_assigned)
    assert "one_row_per_observed_profile" in failed(duplicate)


def test_naive_baseline_always_guesses(people):
    pred = naive_predictions(people).set_index("employee_id")
    eligible = people.set_index("employee_id")["eligible_employee"]
    assert (pred.loc[eligible[eligible].index.drop("ceo"), "status"] == "DIRECT").all()
    assert pred.loc["be1", "reports_to_id"] == "em_be"
    assert pred.loc["board", "status"] == "UNRESOLVED"


def test_model_json_round_trip_and_feature_guard(tmp_path):
    model = rules_model()
    model.direct_calibration = Calibration([0.0, 0.5, 1.0], [0.0, 0.4, 0.9])
    path = tmp_path / "m.json"
    model.save(path)
    loaded = Model.load(path)
    assert loaded.to_json() == model.to_json()
    data = model.to_json()
    data["features"] = data["features"][:-1]
    with pytest.raises(ValueError):
        Model.from_json(data)


def test_explanations_skip_employee_level_features():
    head = rules_model().direct
    row = dict.fromkeys(FEATURES, 1.0)
    head.weights.update({f: 9.0 for f in EMPLOYEE_LEVEL_FEATURES})
    labels = [label for label, _ in head.contributions(row, top=50)]
    assert not any("employee heads their function" == lbl for lbl in labels)


@pytest.mark.parametrize(("raw", "city", "country", "remote"), [
    ("Sydney, New South Wales, Australia", "sydney", "australia", False),
    ("Greater Sydney Area", "sydney", None, False),
    ("Singapore", "singapore", "singapore", False),
    ("Australia", None, "australia", False),
    ("Remote", None, None, True),
    ("United States (Remote)", None, "united states", True),
    (None, None, None, False),
    (float("nan"), None, None, False),
])
def test_parse_location(raw, city, country, remote):
    loc = parse_location(raw)
    assert (loc.city, loc.country, loc.remote) == (city, country, remote)


def test_estimators_on_toy_data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4000, 2))
    y = (rng.random(4000) < 1 / (1 + np.exp(-(0.5 + 2 * X[:, 0] - X[:, 1])))).astype(float)
    b, w = fit_logistic(X, y, l2=0.1)
    assert abs(b - 0.5) < 0.25 and abs(w[0] - 2) < 0.3 and abs(w[1] + 1) < 0.3

    xs, ys = fit_isotonic(rng.random(1000), rng.random(1000) < 0.5)
    assert xs == sorted(xs) and ys == sorted(ys) and all(0 <= v <= 1 for v in ys)

    # Two candidates per employee; the true manager has feature 1.
    X = np.array([[1.0], [0.0]] * 300)
    y = np.array([1.0, 0.0] * 300)
    starts = np.arange(0, 600, 2)
    b, w = fit_conditional_logit(X, y, starts, l2=0.1)
    assert w[0] > 1.0
