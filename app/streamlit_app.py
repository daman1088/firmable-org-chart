"""Org chart inference explorer.

    streamlit run app/streamlit_app.py

Explore mode uses observed profiles, normalised titles, model files and
predictions only. Its "Precision targets" tab shows the aggregate held-out
metrics from reports/inference_eval.json; it never sees per-employee labels.
Evaluation mode is a separate module (`evaluation_view.py`), imported only when
selected. It is the only part of the app that reads ground truth.

DIRECT (exact manager) and ANCESTOR (someone in the chain above, direct or higher) are kept
visually and verbally distinct everywhere: separate metric cards, separate
table columns, separately styled chart edges and separate result tables.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ask_view  # noqa: E402
from app_data import (STATUS_COLORS, STATUS_HELP, STATUS_LABELS, apply_threshold, candidate_table,  # noqa: E402
                      company_predictions, contributions_long, default_target, load, load_candidates, load_model,
                      model_view, ready, targets)
from chart import (DEFAULT_LEADERS, LAYOUTS, VIEWS, components, direction_for, nearby, needs_full_width,  # noqa: E402
                   roots, select, split, to_dot)
from results_view import (fmt_threshold, held_out_summary, load_report, render_operating_points,  # noqa: E402
                          target_label)

st.set_page_config(page_title="Org chart inference", page_icon="🗂️", layout="wide")

# --- sidebar ---------------------------------------------------------------------------------------
st.sidebar.title("Org chart inference")
root = st.sidebar.text_input("Dataset directory", os.environ.get("ORGCHART_DATASET", "data/full"))
# Hosted deployment: download precomputed synthetic data instead of retraining.
# The public release contains only reproducible synthetic data and model outputs.
if ready(Path(root)) and os.environ.get("ORGCHART_AUTOBUILD") == "1":
    import shutil
    import tempfile
    import urllib.request
    import zipfile

    release_url = ("https://github.com/daman1088/firmable-org-chart/"
                   "releases/download/demo-data-v1/demo-data.zip")
    with st.spinner("Loading precomputed 600-company demo dataset…"):
        try:
            # Download to disk instead of buffering the entire archive in RAM.
            with tempfile.TemporaryDirectory() as tempdir:
                zip_path = Path(tempdir) / "demo-data.zip"
                with urllib.request.urlopen(release_url, timeout=120) as response, zip_path.open("wb") as output:
                    shutil.copyfileobj(response, output, length=1024 * 1024)
                destination = Path.cwd().resolve()
                with zipfile.ZipFile(zip_path) as archive:
                    for member in archive.infolist():
                        resolved = (destination / member.filename).resolve()
                        if not resolved.is_relative_to(destination):
                            raise ValueError("Invalid path in dataset archive")
                    archive.extractall(destination)
        except Exception as exc:
            st.error(f"Could not download the demo dataset: {exc}")
            st.stop()

missing = ready(Path(root))
if missing:
    st.error("Dataset is not built yet. Run `make pipeline GENERATE=1` (or `orgchart pipeline data/full --generate`). "
             "Missing: " + ", ".join(missing))
    st.stop()

people, pred, companies = load(root)
model = load_model(root)
report = load_report(root)
mode = st.sidebar.radio("Mode", ("Explore predictions", "Evaluation (ground truth)"),
                        help="Explore never reads ground truth. Evaluation compares predictions with the hidden truth.")

industries = sorted(companies["industry"].unique())
chosen_ind = st.sidebar.multiselect("Industry", industries)
pool = companies[companies["industry"].isin(chosen_ind)] if chosen_ind else companies

search = st.sidebar.text_input("Find employee (name or ID, all companies)")
if search:
    hits = people[people["full_name"].str.contains(search, case=False, na=False) | (people["employee_id"] == search)]
    if hits.empty:
        st.sidebar.caption("No match.")
    else:
        pick = st.sidebar.selectbox("Matches", hits.head(50).index,
                                    format_func=lambda i: f"{hits.at[i, 'full_name']} · {hits.at[i, 'title_raw']} · "
                                                          f"{hits.at[i, 'company_name']}")
        if st.sidebar.button("Open employee"):
            st.session_state["company_id"] = hits.at[pick, "company_id"]
            st.session_state["focus"] = hits.at[pick, "employee_id"]
            st.session_state["chart_view"] = "Around an employee"
            pool = companies

ids = list(pool["company_id"])
default = st.session_state.get("company_id")
idx = ids.index(default) if default in ids else 0
labels = pool.set_index("company_id")
company_id = st.sidebar.selectbox(
    f"Company ({len(ids)} of {len(companies)})", ids, index=idx,
    format_func=lambda c: f"{c} · {labels.at[c, 'company_name']} · {labels.at[c, 'industry']} · {labels.at[c, 'profiles']}")
st.session_state["company_id"] = company_id

st.sidebar.subheader("Precision target")
model_target = default_target(model)
CUSTOM = "custom"
target_options = [*targets(model), CUSTOM]
target = st.sidebar.radio(
    "Operating point", target_options, index=0,
    format_func=lambda t: "Custom DIRECT threshold" if t == CUSTOM else
    f"{target_label(t)} precision" + (" (model default)" if t == model_target else ""),
    help="Each target has its own DIRECT, AMBIGUOUS and ANCESTOR thresholds, chosen on calibration companies. "
         "Lower targets make more claims at lower precision.")
active = model.metadata.get("operating_points", {}).get(target) or vars(model.thresholds)
if target == CUSTOM:
    model_t = float(model.thresholds.direct)
    threshold = st.sidebar.slider("DIRECT confidence threshold", 0.30, 0.99, min(model_t, 0.99), 0.01,
                                  help="Show an employee's top candidate as a direct manager when its calibrated "
                                       "probability is at least this value. Other statuses use the model's "
                                       f"{target_label(model_target)} thresholds.")
    st.sidebar.caption(f"The model's {target_label(model_target)} threshold is **{model_t:.2f}**. A DIRECT edge's "
                       "probability is calibrated, so its expected precision is about that probability.")
    base = company_predictions(root, company_id, model_target)
    view = apply_threshold(base, threshold)
    direct_t = threshold
else:
    view = model_view(company_predictions(root, company_id, target))
    direct_t = float(active["direct"])
    st.sidebar.markdown(held_out_summary(report, target))
st.sidebar.caption(f"Model: {model.variant}")

# --- company data ----------------------------------------------------------------------------------
comp_people = people[people["company_id"] == company_id].reset_index(drop=True)
titles = comp_people.set_index("employee_id")["title_raw"].fillna("(no title)")
names = comp_people.set_index("employee_id")["full_name"]
info = labels.loc[company_id]
target_name = "custom DIRECT threshold" if target == CUSTOM else f"{target_label(target)} precision target"

st.title(f"{info['company_name']}")
st.caption(f"{company_id} · {info['industry']} · {info['profiles']} observed profiles · split: {info['split']} · "
           f"showing the {target_name}")

counts = view["view_status"].value_counts()
shown_statuses = ["DIRECT", "ANCESTOR", "AMBIGUOUS", "UNRESOLVED"] + (["BELOW THRESHOLD"] if target == CUSTOM else [])
for col, status in zip(st.columns(len(shown_statuses)), shown_statuses, strict=True):
    col.metric(STATUS_LABELS[status], int(counts.get(status, 0)), help=STATUS_HELP[status])
st.caption("Only **DIRECT** names a reporting line. **ANCESTOR** names someone in the reporting chain above the "
           "person when the direct manager can't be identified. That person may be the direct manager or someone "
           "higher; the model does not establish which.")

if mode.startswith("Evaluation"):
    import evaluation_view  # noqa: PLC0415  (ground truth is only loaded in this mode)

    decisions = f"{target_label(model_target if target == CUSTOM else target)} precision target"
    evaluation_view.render(root, company_id, comp_people, view, decisions)
    st.stop()


def person(eid: object) -> str:
    if not isinstance(eid, str):
        return ""
    return f"{names.get(eid, eid)} — {titles.get(eid, '')}"


def decision_callout(r: pd.Series) -> None:
    """State the decision in words, so an ANCESTOR is never read as a manager."""
    s = r["view_status"]
    if s == "DIRECT":
        st.success(f"**DIRECT: exact direct manager.** {person(r['view_manager'])} "
                   f"(p = {r['view_confidence']:.2f} that this is the direct manager).")
        if r["status"] != "DIRECT":
            st.caption(f"Shown as DIRECT only because of the custom threshold. The model's own decision is "
                       f"{r['status']}.")
    elif s == "ANCESTOR":
        st.info(f"**ANCESTOR: in the reporting chain; direct manager not identified.** "
                f"{person(r['ancestor_id'])} is somewhere *above* this person in the reporting chain "
                f"(p = {r['confidence']:.2f} that they are above). They may be the direct manager or someone higher; "
                "the evidence was not strong enough to say which, so no direct manager is claimed.")
        if isinstance(r["top_candidate_id"], str):
            st.caption(f"Best direct-manager candidate: {person(r['top_candidate_id'])} at p = "
                       f"{r['direct_confidence']:.2f}, below the DIRECT threshold of {direct_t:.2f}. Not claimed.")
    elif s == "AMBIGUOUS":
        options = "; ".join(person(c) for c in r["candidate_ids"])
        st.warning(f"**AMBIGUOUS: the direct manager is one of {len(r['candidate_ids'])}.** {options} "
                   f"(combined p = {r['confidence']:.2f}). No single manager is claimed.")
    elif s == "BELOW THRESHOLD":
        st.info(f"**Hidden by the custom threshold.** The model's DIRECT claim, {person(r['reports_to_id'])} "
                f"(p = {r['direct_confidence']:.2f}), is below {direct_t:.2f}.")
    else:
        st.info("**UNRESOLVED: no claim.** The evidence did not reach any threshold. "
                f"Reasons: `{', '.join(r['reason_codes'])}`.")


tab_chart, tab_ask, tab_people, tab_review, tab_targets, tab_model = st.tabs(
    ["Org chart", "Ask the Org Chart", "Employees", "Review queue", "Precision targets", "Model"])

with tab_ask:
    ask_view.render(comp_people, view, people, target_name)

with tab_chart:
    c1, c2, c3 = st.columns([1.1, 1.6, 1.0])
    chart_mode = c1.selectbox("View", VIEWS, key="chart_view")  # default: Managers and leaders
    focus = department = None
    n_leaders = DEFAULT_LEADERS
    if chart_mode == "Managers and leaders":
        n_leaders = c2.slider("Boxes shown", 10, 60, DEFAULT_LEADERS, 5,
                              help="About this many boxes: the leaders who anchor the most predicted structure, "
                                   "their chain upwards, and one list per leader and kind of claim.")
    elif chart_mode == "Around an employee":
        emp_ids = list(comp_people.loc[~comp_people["is_non_employee"], "employee_id"])
        f0 = st.session_state.get("focus")
        focus = c2.selectbox("Employee", emp_ids, index=emp_ids.index(f0) if f0 in emp_ids else 0, format_func=person)
        st.session_state["focus"] = focus
    elif chart_mode == "One department":
        depts = sorted(comp_people["department"].dropna().unique())
        department = c2.selectbox("Department", depts)
    layout = c3.selectbox("Layout", LAYOUTS, help="Auto draws each branch top-down, and switches a branch to "
                                                   "left-to-right only when one of its levels would be too wide to "
                                                   "read.")
    graph = select(comp_people, view, chart_mode, focus, department, n_leaders)
    branches = [card for b in components(graph) for card in split(b)]

    if chart_mode == "Around an employee" and focus:
        def _go(key: str) -> None:
            if st.session_state.get(key):
                st.session_state["focus"] = st.session_state[key]

        others = [e for e in nearby(graph) if e != focus]
        go_key = f"goto_{focus}"
        st.selectbox("Go to a nearby person", [""] + others, key=go_key, on_change=_go, args=(go_key,),
                     format_func=lambda e: "— choose someone shown below —" if e == "" else person(e))

    st.markdown(
        "**Legend.** :green[**━━ solid green**]: DIRECT, the exact manager (label = probability; thicker and "
        "darker = more confident).  \n"
        ":blue[**╍╍ dashed blue, \"ancestor relationship\"**]: ANCESTOR. The upper person is somewhere above the lower one "
        "in the reporting chain. They may be the direct manager or someone higher; that is **not established**, so it "
        "is not drawn as a reporting line. It skips a level because there may be people in between, and lists under "
        "it are headed \"N employees linked through hierarchy: direct manager not identified\".  \n"
        ":orange[**┈┈ dotted orange, \"?\"**]: AMBIGUOUS. One of the linked candidates is the manager.  \n"
        "Box colour shows each person's own status. Lists group people with the same claim; each person's own "
        "probability is shown next to their name.")
    shown_people = len(graph.people) + sum(len(g.members) for g in graph.groups)
    no_claim = int(view["view_status"].isin(["UNRESOLVED", "BELOW THRESHOLD"]).sum())
    notes = [f"{len(graph.people)} people drawn individually"
             + (f", {shown_people - len(graph.people)} more listed in {len(graph.groups)} groups" if graph.groups
                else "")]
    if chart_mode == "Managers and leaders":
        notes.append("leaders without any predicted link are not drawn")
    if chart_mode != "Around an employee":
        notes.append(f"{no_claim} people with no claim are in the Review queue")
    if graph.truncated:
        notes.append(f"{graph.truncated} smaller teams left out to keep the chart readable; use 'One department' or "
                     "'Around an employee'")
    if graph.not_drawn:
        notes.append(f"{graph.not_drawn} people whose claimed manager is not drawn are left out")
    if len(branches) > 1:
        notes.append(f"{len(branches)} separate branches: no confident claim links their top people, so they are "
                     "drawn apart rather than under an assumed common manager")
    st.caption(" · ".join(notes) + ".")

    def draw(branch, box, direction: str) -> None:
        tops = [names.get(r, r) for r in roots(branch)]
        title = "Around " + names.get(focus, "") if branch.focus else (
            f"Team of {names.get(branch.head, branch.head)}" if branch.head else
            "Branch under " + ", ".join(tops[:3]) + (" …" if len(tops) > 3 else "") if tops else "Branch")
        with box.container(border=True):
            st.markdown(f"**{title}**")
            if branch.continued_from:
                st.caption(f"Continued from {names.get(branch.continued_from, '')}'s card above.")
            if branch.continues_in:
                st.caption("Teams of " + ", ".join(names.get(h, h) for h in branch.continues_in)
                           + " continue on their own cards below.")
            st.graphviz_chart(to_dot(branch, comp_people, view, direction,
                                     max_rows=4 if chart_mode == "Managers and leaders" else 6), width="content")

    if not graph.node_count():
        st.info("No predicted links to draw at this precision target. Try a lower target, or 'Around an employee'.")
    directions = [direction_for(b, layout) for b in branches]
    if layout == "Auto" and "LR" in directions:
        notes_lr = sum(d == "LR" for d in directions)
        st.caption(f"{notes_lr} branch{'es are' if notes_lr > 1 else ' is'} drawn left-to-right because top-down "
                   "would be too wide to read. Choose Layout → Top-down to override.")
    wide = [(b, d) for b, d in zip(branches, directions, strict=True) if needs_full_width(b, d)]
    small = [(b, d) for b, d in zip(branches, directions, strict=True) if not needs_full_width(b, d)]
    for b, d in wide:
        draw(b, st, d)
    for i in range(0, len(small), 3):
        for col, (b, d) in zip(st.columns(3), small[i:i + 3], strict=False):
            draw(b, col, d)

with tab_people:
    q = st.text_input("Search name or title in this company")
    table = view.merge(comp_people[["employee_id", "full_name", "title_raw", "department", "level"]], on="employee_id")
    if q:
        table = table[table["full_name"].str.contains(q, case=False, na=False)
                      | table["title_raw"].str.contains(q, case=False, na=False)]
    s = table["view_status"]
    table = table.assign(
        exact_manager=table["view_manager"].where(s == "DIRECT").map(person),
        p_exact=table["view_confidence"].where(s == "DIRECT"),
        above=table["ancestor_id"].where(s == "ANCESTOR").map(person),
        p_above=table["confidence"].where(s == "ANCESTOR"),
        one_of=[" | ".join(person(x) for x in c) if v == "AMBIGUOUS" else ""
                for c, v in zip(table["candidate_ids"], s, strict=True)],
        p_set=table["confidence"].where(s == "AMBIGUOUS"),
        reasons=table["reason_codes"].map(", ".join),
    )
    shown = table[["employee_id", "full_name", "title_raw", "department", "level", "view_status", "exact_manager",
                   "p_exact", "above", "p_above", "one_of", "p_set", "reasons"]].rename(columns={
        "full_name": "name", "title_raw": "title", "view_status": "status",
        "exact_manager": "DIRECT: exact manager", "p_exact": "p(exact manager)",
        "above": "ANCESTOR: in chain above (direct manager not identified)", "p_above": "p(in chain above)",
        "one_of": "AMBIGUOUS: manager is one of", "p_set": "p(manager in set)"})
    prob = {c: st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f")
            for c in ("p(exact manager)", "p(in chain above)", "p(manager in set)")}
    sel = st.dataframe(shown, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row",
                       column_config=prob)
    st.caption("Each kind of claim has its own column. An ANCESTOR is never listed as the direct manager, because "
               "that is not established.")
    rows = sel.selection.rows if sel and sel.selection else []
    if rows:
        eid = shown.iloc[rows[0]]["employee_id"]
        st.session_state["focus"] = eid
        r = view.set_index("employee_id").loc[eid]
        st.subheader(person(eid))
        decision_callout(r)
        st.markdown(f"**Model explanation** (reasons: `{', '.join(r['reason_codes'])}`)")
        for line in r["explanation"]:
            st.markdown(f"- {line}")
        cands = load_candidates(root, company_id)
        cands = cands[cands["employee_id"] == eid]
        if not cands.empty:
            st.markdown("**Ranked direct-manager candidates.** p(direct manager) and p(above in chain) answer "
                        "different questions. The second is high for anyone senior in the same line.")
            st.dataframe(candidate_table(cands, model, titles), hide_index=True, width="stretch")
            contrib = contributions_long(cands, model)
            if not contrib.empty:
                st.altair_chart(alt.Chart(contrib).mark_bar().encode(
                    x=alt.X("contribution:Q", title="contribution to direct-manager score (log-odds)"),
                    y=alt.Y("feature:N", sort="-x", title=None),
                    color=alt.condition("datum.contribution > 0", alt.value("#2e7d32"), alt.value("#c62828")),
                    row=alt.Row("candidate:N", title=None)).properties(height=120), width="stretch")
        st.caption("Select another row to inspect it; 'Org chart → Around an employee' shows this person in context.")
    else:
        st.caption("Select a row to see the decision, explanation and ranked candidates.")

with tab_review:
    review_statuses = ["AMBIGUOUS", "UNRESOLVED", "ANCESTOR"] + (["BELOW THRESHOLD"] if target == CUSTOM else [])
    st.markdown("Employees **without an exact (DIRECT) manager**, with the reason. ANCESTOR rows have a confident "
                "person in the chain above them, but no identified direct manager. These are the cases a human reviewer or an extra data source "
                "would resolve.")
    status_pick = st.multiselect("Status", review_statuses, default=["AMBIGUOUS", "UNRESOLVED"])
    rev = view[view["view_status"].isin(status_pick)].copy()
    reasons = rev["reason_codes"].explode().value_counts().rename_axis("reason").reset_index(name="employees")
    left, right = st.columns([1, 2])
    left.dataframe(reasons, hide_index=True, width="stretch")
    rev = rev.assign(employee=rev["employee_id"].map(person),
                     candidates=rev["candidate_ids"].map(lambda c: " | ".join(person(x) for x in c)),
                     above=rev["ancestor_id"].map(person),
                     best_candidate=rev["top_candidate_id"].map(person), reasons=rev["reason_codes"].map(", ".join),
                     why=rev["explanation"].map(lambda e: e[0] if len(e) else ""))
    right.dataframe(rev[["employee", "view_status", "best_candidate", "direct_confidence", "candidates", "above",
                         "reasons", "why"]].rename(columns={
                             "view_status": "status", "best_candidate": "best direct candidate (not claimed)",
                             "direct_confidence": "best p", "candidates": "AMBIGUOUS: one of",
                             "above": "ANCESTOR: in chain above"}),
                    hide_index=True, width="stretch")

with tab_targets:
    render_operating_points(report, current=None if target == CUSTOM else target)
    st.caption("Use the sidebar's **Precision target** to apply any of these operating points to the current "
               "company. Per-company accuracy against truth is in Evaluation mode.")

with tab_model:
    st.markdown(f"**Variant:** `{model.variant}` · thresholds in use ({target_name}, fixed on calibration "
                f"companies): DIRECT ≥ {direct_t:.2f}, AMBIGUOUS set mass ≥ {fmt_threshold(active['ambiguous'])} "
                f"(≤ {active['max_ambiguous']} candidates), ANCESTOR ≥ {fmt_threshold(active['ancestor'])}.")
    if model.metadata.get("selection"):
        st.markdown(f"**Selection:** {model.metadata['selection']['rule']}.")
    from orgchart.inference.features import FEATURE_LABELS

    w = pd.DataFrame({"feature": list(FEATURE_LABELS.values()),
                      "direct weight": [model.direct.weights.get(f, 0.0) for f in FEATURE_LABELS],
                      "ancestor weight": [model.ancestor.weights.get(f, 0.0) for f in FEATURE_LABELS]})
    st.markdown("Every score is a sum of these weights × feature values, so each prediction decomposes into the "
                "contributions shown in the Employees tab. The direct and ancestor heads are separate models with "
                "separate weights.")
    st.dataframe(w.sort_values("direct weight", key=abs, ascending=False), hide_index=True, width="stretch")
    dist = pred["status"].value_counts().rename_axis("status").reset_index(name="profiles")
    dist["status"] = dist["status"].map(STATUS_LABELS)
    st.altair_chart(alt.Chart(dist).mark_bar().encode(
        x="profiles:Q", y=alt.Y("status:N", sort="-x", title=None),
        color=alt.Color("status:N", scale=alt.Scale(domain=[STATUS_LABELS[k] for k in STATUS_COLORS],
                                                    range=list(STATUS_COLORS.values())), legend=None))
        .properties(title=f"Model decisions across all {len(companies)} companies ({target_label(model_target)} target)"),
        width="stretch")
