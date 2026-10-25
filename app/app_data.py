"""Data access for the explorer. Observed data, normalised titles, model files and predictions only."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pandas as pd
import streamlit as st

from orgchart.inference.data import load_people
from orgchart.inference.decide import top_contributions
from orgchart.inference.pipeline import predict_people
from orgchart.inference.scoring import Model, Thresholds
from orgchart.paths import DatasetPaths

STATUS_COLORS = {"DIRECT": "#2e7d32", "AMBIGUOUS": "#ef6c00", "ANCESTOR": "#1565c0", "UNRESOLVED": "#9e9e9e",
                 "BELOW THRESHOLD": "#bdbdbd"}
STATUS_LABELS = {"DIRECT": "DIRECT · exact manager", "AMBIGUOUS": "AMBIGUOUS · one of 2–3",
                 "ANCESTOR": "ANCESTOR · in chain above", "UNRESOLVED": "UNRESOLVED · no claim",
                 "BELOW THRESHOLD": "BELOW THRESHOLD"}
STATUS_HELP = {
    "DIRECT": "The named person is claimed to be this employee's exact direct manager.",
    "AMBIGUOUS": "The direct manager is claimed to be one of 2–3 named candidates. No single manager is claimed.",
    "ANCESTOR": "The named person is claimed to be somewhere above this employee in the reporting chain. They may be "
                "the direct manager or someone higher; the model does not establish which, so this is not a "
                "direct-manager claim.",
    "UNRESOLVED": "No claim. The evidence did not reach any threshold.",
    "BELOW THRESHOLD": "The model made a DIRECT claim, but it is hidden because its probability is below the custom "
                       "threshold.",
}


def ready(root: Path) -> list[str]:
    """Missing build outputs, if any."""
    paths = DatasetPaths.of(root)
    needed = [paths.observed_employees, paths.normalized_titles, paths.predictions(), paths.candidates, paths.model()]
    return [str(p) for p in needed if not p.exists()]


@st.cache_data(show_spinner="Loading observed profiles and predictions…")
def load(root: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = DatasetPaths.of(root)
    people = load_people(root)
    pred = pd.read_parquet(paths.predictions())
    companies = (people.groupby(["company_id", "company_name", "industry", "split"], as_index=False)
                 .size().rename(columns={"size": "profiles"}))
    status = pred.groupby("company_id")["status"].value_counts().unstack(fill_value=0)
    companies = companies.merge(status, left_on="company_id", right_index=True, how="left")
    return people, pred, companies


@st.cache_data(show_spinner=False)
def load_candidates(root: str, company_id: str) -> pd.DataFrame:
    return pd.read_parquet(DatasetPaths.of(root).candidates, filters=[("company_id", "==", company_id)])


@st.cache_resource(show_spinner=False)
def load_model(root: str) -> Model:
    return Model.load(DatasetPaths.of(root).model())


def default_target(model: Model) -> str:
    return f"{model.metadata.get('target_precision', 0.95):.2f}"


def targets(model: Model) -> list[str]:
    """Stored precision targets (e.g. 0.95, 0.90, 0.80), strictest first."""
    ops = model.metadata.get("operating_points") or {default_target(model): None}
    return sorted(ops, key=float, reverse=True)


@st.cache_data(show_spinner="Applying precision target…")
def company_predictions(root: str, company_id: str, target: str) -> pd.DataFrame:
    """Model decisions for one company at a stored precision target.

    The model's own target is read from predictions.parquet. Other targets re-run the same inference code
    with that target's thresholds from the model file. Nothing is refitted or recalibrated, and at the model's
    own thresholds this reproduces predictions.parquet exactly.
    """
    people, pred, _ = load(root)
    model = load_model(root)
    if target == default_target(model):
        return pred[pred["company_id"] == company_id].reset_index(drop=True)
    at_target = dataclasses.replace(model, thresholds=Thresholds(**model.metadata["operating_points"][target]))
    out = predict_people(people[people["company_id"] == company_id], at_target)
    return out.assign(model_variant=model.variant)


def model_view(pred: pd.DataFrame) -> pd.DataFrame:
    """View columns equal to the model's own decisions (no custom threshold)."""
    return pred.assign(view_status=pred["status"], view_manager=pred["reports_to_id"],
                       view_confidence=pred["confidence"])


def apply_threshold(pred: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """View-level statuses: a top candidate with p_direct >= threshold is shown as a DIRECT edge.

    The model's own decision is kept in `status`. `view_status` and `view_manager` reflect the slider.
    Former DIRECT edges under the slider become "BELOW THRESHOLD". Edges always point to a strictly more
    senior candidate, so the view stays acyclic at any threshold.
    """
    df = pred.copy()
    direct = df["top_candidate_id"].notna() & (df["direct_confidence"].fillna(-1.0) >= threshold)
    df["view_status"] = df["status"]
    df.loc[(df["status"] == "DIRECT") & ~direct, "view_status"] = "BELOW THRESHOLD"
    df.loc[direct, "view_status"] = "DIRECT"
    df["view_manager"] = df["top_candidate_id"].where(direct)
    df["view_confidence"] = df["direct_confidence"].where(direct, df["confidence"])
    return df


def candidate_table(cands: pd.DataFrame, model: Model, titles: pd.Series) -> pd.DataFrame:
    if cands.empty:
        return pd.DataFrame()
    c = cands.sort_values("rank").reset_index(drop=True)
    why = top_contributions(c, model.direct.vector(), k=4)
    return pd.DataFrame({
        "rank": c["rank"] + 1,
        "candidate": [titles.get(x, x) for x in c["candidate_id"]],
        "candidate_id": c["candidate_id"],
        "level": c["m_level"],
        "p(direct manager)": c["p_direct"].round(3),
        "p(above in chain)": c["p_ancestor"].round(3),
        "why (top feature contributions)": ["; ".join(w) for w in why],
    })


def contributions_long(cands: pd.DataFrame, model: Model, k: int = 6) -> pd.DataFrame:
    """Per-feature contributions of the top candidates, long format for charting."""
    from orgchart.inference.features import EMPLOYEE_LEVEL_FEATURES, FEATURE_LABELS, FEATURES

    rows = []
    for _, c in cands.sort_values("rank").head(3).iterrows():
        items = [(FEATURE_LABELS[f], model.direct.weights.get(f, 0.0) * float(c[f])) for f in FEATURES
                 if f not in EMPLOYEE_LEVEL_FEATURES]
        for label, value in sorted(items, key=lambda x: -abs(x[1]))[:k]:
            if abs(value) > 1e-9:
                rows.append({"candidate": f"#{int(c['rank']) + 1}", "feature": label, "contribution": round(value, 3)})
    return pd.DataFrame(rows)
