"""Evaluation mode: compare predictions with ground truth. Reads truth; never used by explore mode."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from orgchart.evaluation.inference_eval import label_predictions, status_metrics
from orgchart.evaluation.labels import employee_truth, true_ancestors
from results_view import claim_columns, load_report, render_baselines, render_operating_points, report_path


@st.cache_data(show_spinner="Loading ground truth…")
def _truth(root: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    return employee_truth(root), true_ancestors(root)


def render(root: str, company_id: str, comp_people: pd.DataFrame, view: pd.DataFrame,
           target_name: str = "model default") -> None:
    st.warning("Evaluation mode: this view reads the hidden ground truth. Inference never does.")
    truth, ancestors = _truth(root)
    report = load_report(root)
    tab_company, tab_global = st.tabs(["This company vs truth", "Held-out test results"])

    with tab_company:
        model_pred = view.drop(columns=["view_status", "view_manager", "view_confidence"])
        labelled = label_predictions(model_pred, truth, ancestors)
        m = status_metrics(labelled)
        st.markdown(f"**Model decisions for this company** at the {target_name} (thresholds fixed on calibration "
                    "companies). DIRECT is correct only for the exact manager; ANCESTOR is correct for anyone above.")
        st.dataframe(pd.DataFrame([claim_columns(m)]), hide_index=True, width="stretch")
        # Threshold view: is the top candidate the true manager?
        t = view.merge(truth[["employee_id", "true_manager_id", "manager_observed", "role_type"]], on="employee_id")
        shown = t[t["view_status"] == "DIRECT"]
        correct = (shown["view_manager"] == shown["true_manager_id"]).mean() if len(shown) else None
        st.markdown(f"At the sidebar threshold: **{len(shown)}** DIRECT edges, precision "
                    f"**{'—' if correct is None else f'{100 * correct:.1f}%'}** in this company. "
                    f"{int(t['manager_observed'].sum())} of {int((t['role_type'] != 'board').sum())} employees have "
                    "their true manager observed, which is the ceiling for DIRECT recall.")
        title = comp_people.set_index("employee_id")["title_raw"].fillna("(no title)")
        out = labelled.assign(
            employee=labelled["employee_id"].map(title),
            direct_claim=labelled["reports_to_id"].map(title),
            ancestor_claim=labelled["ancestor_id"].map(title),
            ancestor_levels_up=labelled["ancestor_hops"],
            true_manager=labelled["true_manager_id"].map(title),
            correct=labelled["claim_correct"].where(labelled["status"] != "UNRESOLVED"))
        pick = st.radio("Show", ("Errors", "All claims", "Unresolved"), horizontal=True)
        if pick == "Errors":
            out = out[(out["status"] != "UNRESOLVED") & ~out["claim_correct"]]
        elif pick == "Unresolved":
            out = out[out["status"] == "UNRESOLVED"]
        else:
            out = out[out["status"] != "UNRESOLVED"]
        st.dataframe(out[["employee", "status", "direct_claim", "ancestor_claim", "ancestor_levels_up",
                          "candidate_ids", "true_manager", "manager_observed", "confidence", "correct"]].rename(columns={
                              "direct_claim": "DIRECT: predicted manager",
                              "ancestor_claim": "ANCESTOR: predicted in chain above",
                              "ancestor_levels_up": "ANCESTOR: true levels above",
                              "candidate_ids": "AMBIGUOUS: candidates"}),
                     hide_index=True, width="stretch")
        st.caption("'correct' applies each status's own rule: exact manager for DIRECT, anyone above for ANCESTOR, "
                   "manager in the set for AMBIGUOUS.")

    with tab_global:
        if report is None:
            st.info("Run `orgchart evaluate data/full` to produce reports/inference_eval.json.")
            return
        st.markdown(f"Selected model **{report['selected_variant']}**. Test companies were never used for fitting, "
                    "calibration, thresholds or model selection.")
        render_baselines(report)
        st.markdown("### Precision targets (95 / 90 / 80%)")
        render_operating_points(report)
        curve = pd.DataFrame([{**pt, "variant": v} for v, pts in report["test_precision_coverage"].items() for pt in pts])
        rel = pd.DataFrame(report["test_calibration"]["top_candidate_direct_probability"]["bins"])
        left, right = st.columns(2)
        left.altair_chart(alt.Chart(curve.dropna()).mark_line(point=True).encode(
            x=alt.X("coverage_pct:Q", title="DIRECT (exact manager) coverage % (test)"),
            y=alt.Y("precision_pct:Q", title="precision %", scale=alt.Scale(domain=[0, 100])),
            color="variant:N", tooltip=["threshold", "coverage_pct", "precision_pct"]).properties(
            title="DIRECT precision vs coverage"), width="stretch")
        diag = pd.DataFrame({"x": [0, 1], "y": [0, 1]})
        right.altair_chart((alt.Chart(rel).mark_line(point=True).encode(
            x=alt.X("mean_confidence:Q", title="predicted probability", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("observed_accuracy:Q", title="observed accuracy", scale=alt.Scale(domain=[0, 1])),
            tooltip=["bin", "n"]) + alt.Chart(diag).mark_line(strokeDash=[4, 4], color="gray").encode(x="x", y="y"))
            .properties(title=f"Calibration (ECE {report['test_calibration']['top_candidate_direct_probability']['ece_pct']}%)"),
            width="stretch")
        for key, rows in report["test_breakdowns"].items():
            with st.expander(f"Breakdown by {key.replace('_', ' ')}"):
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.markdown(f"Full report: `{report_path(root)}`")
