"""Headless dashboard test (Streamlit AppTest) on the small generated dataset."""

import sys
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from orgchart.inference.pipeline import decision_view, infer_dataset
from orgchart.learning.fit import fit_models
from orgchart.evaluation.inference_eval import evaluate_inference
from orgchart.normalization.pipeline import normalize_dataset

APP_DIR = Path(__file__).resolve().parents[1] / "app"
APP = APP_DIR / "streamlit_app.py"
sys.path.insert(0, str(APP_DIR))

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module", autouse=True)
def _restore_main_module():
    # AppTest leaves sys.modules["__main__"] pointing at streamlit_app.py. Spawned worker processes
    # re-run the parent's __main__, so later ProcessPoolExecutor users (dataset generation) would
    # execute the whole app outside a Streamlit session and die with BrokenProcessPool.
    main = sys.modules["__main__"]
    yield
    sys.modules["__main__"] = main


@pytest.fixture(scope="module")
def built(small_dataset):
    normalize_dataset(small_dataset)
    fit_models(small_dataset)
    infer_dataset(small_dataset)
    evaluate_inference(small_dataset)
    return small_dataset


@pytest.fixture(scope="module")
def built_app(built):
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("ORGCHART_DATASET", str(built))
        at = AppTest.from_file(str(APP), default_timeout=120)
        at.run()
        yield at


def _count(at, status: str) -> int:
    return int(next(m.value for m in at.metric if m.label.startswith(status)))


def _target_radio(at):
    return next(r for r in at.sidebar.radio if r.label == "Operating point")


def test_explore_views_render(built_app):
    at = built_app
    assert not at.exception, [e.value for e in at.exception]
    assert [t.label for t in at.tabs][:6] == ["Org chart", "Ask the Org Chart", "Employees", "Review queue",
                                              "Precision targets", "Model"]
    labels = {m.label for m in at.metric}
    assert "DIRECT · exact manager" in labels and "ANCESTOR · in chain above" in labels
    for option in ("Around an employee", "One department", "Whole company", "Managers and leaders"):
        next(s for s in at.selectbox if s.label == "View").set_value(option).run()  # re-find: widgets change per view
        assert not at.exception, (option, [e.value for e in at.exception])


def test_precision_targets_switch(built_app):
    at = built_app
    radio = _target_radio(at)
    assert radio.options[-1] == "Custom DIRECT threshold" and len(radio.options) == 4
    claims = {}
    for value in ("0.95", "0.90", "0.80"):
        _target_radio(at).set_value(value).run()
        assert not at.exception, (value, [e.value for e in at.exception])
        assert not at.sidebar.slider  # the slider exists only in custom mode
        claims[value] = _count(at, "DIRECT") + _count(at, "ANCESTOR") + _count(at, "AMBIGUOUS")
    assert claims["0.80"] >= claims["0.95"]
    assert any("DIRECT and ANCESTOR are different claims" in e.value for e in at.error)
    _target_radio(at).set_value("0.95").run()


def test_threshold_slider_changes_edges(built_app):
    at = built_app
    _target_radio(at).set_value("custom").run()
    at.sidebar.slider[0].set_value(0.99).run()
    high = _count(at, "DIRECT")
    at.sidebar.slider[0].set_value(0.30).run()
    low = _count(at, "DIRECT")
    assert not at.exception and low >= high
    assert any(m.label == "BELOW THRESHOLD" for m in at.metric)
    _target_radio(at).set_value("0.95").run()


def test_company_switch_and_evaluation_mode(built_app):
    at = built_app
    company = next(s for s in at.sidebar.selectbox if s.label.startswith("Company"))
    company.set_value(company.options[-1].split(" · ")[0]).run()
    assert not at.exception, [e.value for e in at.exception]
    at.sidebar.radio[0].set_value("Evaluation (ground truth)").run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("ground truth" in w.value for w in at.warning)
    _target_radio(at).set_value("0.80").run()
    assert not at.exception, [e.value for e in at.exception]
    at.sidebar.radio[0].set_value("Explore predictions").run()


def test_default_target_reproduces_stored_predictions(built):
    from app_data import company_predictions, default_target, load, load_model

    _, pred, companies = load(str(built))
    model = load_model(str(built))
    for cid in companies["company_id"].head(3):
        stored = decision_view(pred[pred["company_id"] == cid])
        assert decision_view(company_predictions(str(built), cid, default_target(model))).equals(stored)
        assert decision_view(company_predictions.__wrapped__(str(built), cid, "0.80")).shape == stored.shape


def _company_views(built, targets=("0.95", "0.80")):
    from app_data import company_predictions, load, model_view

    people, _, companies = load(str(built))
    for target in targets:
        for cid in companies["company_id"]:
            comp = people[people["company_id"] == cid].reset_index(drop=True)
            yield target, cid, comp, model_view(company_predictions(str(built), cid, target))


def test_ancestor_is_never_drawn_as_a_reporting_line(built):
    from chart import ANCESTOR, DIRECT, components, select, split, to_dot

    checked = 0
    for _, _, comp, view in _company_views(built):
        anc = view[view["view_status"] == "ANCESTOR"]
        assert anc["view_manager"].isna().all()  # no manager for an ANCESTOR row
        graph = select(comp, view, "Whole company")
        drawn = {(p, c): k for p, c, k, _ in graph.edges}
        group_of = {eid: g for g in graph.groups for eid, _ in g.members}
        for _, r in anc.iterrows():
            if pd.isna(r["ancestor_id"]):
                continue
            if r["employee_id"] in graph.people:
                assert drawn.get((r["ancestor_id"], r["employee_id"])) == ANCESTOR
            else:
                g = group_of[r["employee_id"]]
                assert g.kind == ANCESTOR and g.parents == (r["ancestor_id"],)
            checked += 1
        assert not any(k == DIRECT for (p, c), k in drawn.items() if c in anc["employee_id"].values)
        for card in (c for b in components(graph) for c in split(b)):
            dot = to_dot(card, comp, view)
            for line in dot.splitlines():
                if "->" in line and "style=dashed" in line:
                    assert 'label="ancestor relationship"' in line and "minlen=2" in line
                if "->" in line and "style=dashed" not in line and "style=dotted" not in line:
                    assert "ancestor" not in line  # solid lines are DIRECT only
            n_groups = sum(g.kind == ANCESTOR for g in card.groups)
            assert dot.count('employees linked through hierarchy<BR ALIGN="LEFT"/>direct manager not identified') == n_groups
            assert not any(s in dot for s in ("in chain above", "below in chain", "not direct", "NOT direct"))
    assert checked, "no ANCESTOR predictions in the small dataset"


def test_default_view_is_compact_and_readable(built):
    from chart import DEFAULT_LEADERS, MAX_WIDTH_PX, components, direction_for, select, split

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("ORGCHART_DATASET", str(built))
        fresh = AppTest.from_file(str(APP), default_timeout=120)
        fresh.run()
    assert not fresh.exception, [e.value for e in fresh.exception]
    assert next(s for s in fresh.selectbox if s.label == "View").value == "Managers and leaders"
    assert any(s.label == "Boxes shown" and s.value == DEFAULT_LEADERS for s in fresh.slider)
    for _, _, comp, view in _company_views(built):
        graph = select(comp, view, "Managers and leaders")
        assert graph.node_count() <= DEFAULT_LEADERS + 10
        for mode in ("Managers and leaders", "Whole company"):
            for card in (c for b in components(select(comp, view, mode)) for c in split(b)):
                d = direction_for(card, "Auto")
                assert d == "TB" or card.width("TB") > MAX_WIDTH_PX  # top-down unless it cannot fit


def test_split_ends_when_a_candidate_set_includes_the_top_person():
    from chart import AMBIGUOUS, DIRECT, Graph, Group, split

    # top -> head (DIRECT); one employee's manager is "head or top" (AMBIGUOUS), so head's team contains top again
    g = Graph({"top", "head"}, [Group("g", AMBIGUOUS, ("head", "top"), [("e", 0.9)])],
              [("top", "head", DIRECT, 0.9), ("head", "g", AMBIGUOUS, 0.9), ("top", "g", AMBIGUOUS, 0.9)],
              ranks={"top": 0, "head": 1, "g": 2}, widths={"top": 200, "head": 200, "g": 2000})
    cards = split(g)
    assert cards and all(c.people <= g.people for c in cards)


def test_ask_tab_examples_and_questions(built_app):
    at = built_app
    for i in range(3):  # each example question fills the box and gets an answer
        at.button(key=f"ask_example_{i}").click().run()
        assert not at.exception, [e.value for e in at.exception]
        question = at.text_input(key="ask_question").value
        assert question == at.button(key=f"ask_example_{i}").label
        callouts = [c.value for c in (*at.success, *at.info, *at.warning)]
        assert any(s in c for c in callouts for s in ("reports to", "inferred", "identified", "No single leader"))
    at.text_input(key="ask_question").set_value("What is the weather?").run()
    assert any("can't answer" in e.value for e in at.error)
    at.text_input(key="ask_question").set_value("Who does Nobody Atall report to?").run()
    assert any("No one called" in e.value for e in at.error)
    at.text_input(key="ask_question").set_value("").run()
    assert not at.exception


def test_ask_never_invents_a_reporting_line(built):
    """On every employee of the small dataset: a manager is named only for a DIRECT claim, and every step of a
    chain is an existing DIRECT or ANCESTOR claim."""
    from ask import ANSWER, ask

    checked = 0
    for _, _, comp, view in _company_views(built, ("0.80",)):
        v = view.set_index("employee_id")
        for eid in view["employee_id"]:
            a = ask(f"Who does {eid} report to?", comp, view)
            if v.at[eid, "view_status"] == "DIRECT":
                assert a.outcome == ANSWER and a.rows[0]["employee_id"] == v.at[eid, "view_manager"]
            else:
                assert a.outcome != ANSWER and "reports to" not in a.headline
            below = eid
            for row in ask(f"Who is above {eid}?", comp, view).rows:
                r = v.loc[below]
                claimed = r["view_manager"] if r["view_status"] == "DIRECT" else r["ancestor_id"]
                assert r["view_status"] in ("DIRECT", "ANCESTOR") and row["employee_id"] == claimed
                assert row["relationship"].startswith(r["view_status"])
                below = row["employee_id"]
            checked += 1
    assert checked


def test_around_an_employee_shows_chain_and_neighbours(built):
    from chart import ANCESTOR, DIRECT, nearby, select

    for _, _, comp, view in _company_views(built, ("0.80",)):
        v = view.set_index("employee_id")
        for eid in view.loc[view["view_status"].isin([DIRECT, ANCESTOR]), "employee_id"].head(5):
            graph = select(comp, view, "Around an employee", focus=eid)
            up = v.at[eid, "view_manager"] if v.at[eid, "view_status"] == DIRECT else v.at[eid, "ancestor_id"]
            assert graph.focus == eid and {eid, up} <= graph.people
            peers = set(view.loc[(view["view_manager"] == up) | (view["ancestor_id"] == up), "employee_id"])
            assert peers <= set(nearby(graph)) | {eid}
        return
