"""Held-out results panels. Aggregate metrics from reports/inference_eval.json only; no per-employee labels.

DIRECT and ANCESTOR are different claims and are always reported in separate
tables: a DIRECT claim names the exact manager, an ANCESTOR claim names someone
anywhere above the employee. Their precisions are not comparable, and the
combined "any claim" figure is labelled as a mix.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from orgchart.paths import DatasetPaths

OFF = 1.0  # thresholds above 1 mean "no threshold met the target on calibration": the status is never used


@st.cache_data(show_spinner=False)
def load_report(root: str) -> dict | None:
    path = DatasetPaths.of(root).reports_dir / "inference_eval.json"
    return json.loads(path.read_text()) if path.exists() else None


def target_label(target: str) -> str:
    return f"{float(target):.0%}"


def fmt_threshold(value: float) -> str:
    return "off" if value > OFF else f"{value:.2f}"


def claim_columns(m: dict) -> dict:
    """One metrics dict (status_metrics shape) as separately named DIRECT / ANCESTOR / AMBIGUOUS / any-claim columns."""
    return {
        "DIRECT (exact manager) coverage %": m["direct"]["coverage_pct"],
        "DIRECT precision %": m["direct"]["precision_pct"],
        "ANCESTOR (in chain above) coverage %": m["ancestor"]["coverage_pct"],
        "ANCESTOR precision %": m["ancestor"]["precision_pct"],
        "AMBIGUOUS coverage %": m["ambiguous"]["coverage_pct"],
        "AMBIGUOUS contains manager %": m["ambiguous"]["containment_precision_pct"],
        "any claim (mixed) coverage %": m["resolved"]["coverage_pct"],
        "any claim (mixed) precision %": m["resolved"]["precision_pct"],
    }


def held_out_summary(report: dict | None, target: str) -> str:
    """One-line measured test result for a precision target, DIRECT and ANCESTOR kept apart."""
    op = (report or {}).get("test_operating_points", {}).get(target)
    if op is None:
        return "Held-out results not available (run `orgchart evaluate`)."
    d, a = op["direct"], op["ancestor"]
    return (f"Measured on held-out test companies at this target:  \n"
            f"**DIRECT** (exact manager): {d['coverage_pct']}% of employees, **{d['precision_pct']}%** precision  \n"
            f"**ANCESTOR** (in chain above; direct manager not identified): {a['coverage_pct']}% of employees, "
            f"**{a['precision_pct']}%** precision")


def render_operating_points(report: dict | None, current: str | None = None) -> None:
    """The 95 / 90 / 80% precision-target comparison, one table per kind of claim."""
    if report is None:
        st.info("Run `orgchart evaluate data/full` to produce reports/inference_eval.json.")
        return
    ops = report["test_operating_points"]
    n = report["variants"]["selected"]["test"]["employees"]

    def mark(t: str) -> str:
        return target_label(t) + ("  ◀ in use" if t == current else "")

    st.markdown(
        f"Each precision target has its own thresholds. They were chosen on **calibration** companies and then "
        f"measured once on the held-out **test** companies ({n:,} employees). Coverage is the share of those "
        "employees that receive the claim.")
    st.error("**DIRECT and ANCESTOR are different claims and are scored differently.** A DIRECT claim is correct "
             "only if it names the exact manager. An ANCESTOR claim is correct if the named person is *anywhere* "
             "above the employee. ANCESTOR precision is not manager accuracy.", icon="⚠️")

    st.subheader("DIRECT: exact direct manager")
    st.dataframe(pd.DataFrame([{
        "precision target": mark(t), "DIRECT threshold": fmt_threshold(op["thresholds"]["direct"]),
        "coverage %": op["direct"]["coverage_pct"], "precision % (exact manager)": op["direct"]["precision_pct"],
        "recall % (manager observed)": op["direct"]["recall_resolvable_pct"],
        "claims": op["direct"]["predictions"], "skip-level errors": op["direct"]["skip_level_errors"],
    } for t, op in ops.items()]), hide_index=True, width="stretch")
    st.caption("Recall is over employees whose true manager appears in the data. A skip-level error names the "
               "manager's own manager and counts as wrong.")

    st.subheader("ANCESTOR: someone in the chain above (direct manager not identified)")
    st.dataframe(pd.DataFrame([{
        "precision target": mark(t), "ANCESTOR threshold": fmt_threshold(op["thresholds"]["ancestor"]),
        "coverage %": op["ancestor"]["coverage_pct"], "precision % (truly above)": op["ancestor"]["precision_pct"],
        "claims": op["ancestor"]["predictions"],
        "named person is in fact the direct manager %": op["ancestor"]["is_actually_direct_manager_pct"],
        "median levels above (when correct)": op["ancestor"]["median_hops_when_correct"],
    } for t, op in ops.items()]), hide_index=True, width="stretch")
    st.caption("ANCESTOR is reported when the direct manager cannot be pinned down. The named person may be the "
               "direct manager or someone higher; the claim does not say which. The CEO is never used, because "
               "everyone sits under the CEO.")

    st.subheader("AMBIGUOUS: the direct manager is one of 2–3 candidates")
    st.dataframe(pd.DataFrame([{
        "precision target": mark(t), "set probability threshold": fmt_threshold(op["thresholds"]["ambiguous"]),
        "coverage %": op["ambiguous"]["coverage_pct"],
        "true manager in set %": op["ambiguous"]["containment_precision_pct"],
        "claims": op["ambiguous"]["predictions"], "mean set size": op["ambiguous"]["mean_set_size"],
    } for t, op in ops.items()]), hide_index=True, width="stretch")
    st.caption("'off' means no threshold reached the target on calibration companies, so the status is not used.")

    with st.expander("Any claim combined (DIRECT + AMBIGUOUS + ANCESTOR)"):
        st.markdown("This mixes exact-manager, candidate-set and ancestor claims, so it is **not** a "
                    "direct-manager accuracy. It shows how many employees get *some* correct, usable statement.")
        st.dataframe(pd.DataFrame([{
            "precision target": mark(t), "coverage %": op["resolved"]["coverage_pct"],
            "precision %": op["resolved"]["precision_pct"],
        } for t, op in ops.items()]), hide_index=True, width="stretch")


def render_baselines(report: dict) -> None:
    st.markdown("**Baseline comparison on test companies** (each variant at its own 95% thresholds; "
                "`naive` always predicts DIRECT)")
    rows = [{"variant": v, **claim_columns(splits["test"])} for v, splits in report["variants"].items()]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def report_path(root: str) -> Path:
    return Path(root) / "reports" / "inference_eval.md"
