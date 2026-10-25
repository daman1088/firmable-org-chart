"""Run reporting-line inference over a dataset (observed data + model files only).

Writes, under <dataset>/derived/:

* predictions_<variant>.parquet: one row per observed profile, for each model variant;
* predictions.parquet: the selected variant (a copy of models/model_selected.json's variant);
* predictions_naive.parquet: the never-abstain comparison baseline;
* candidates.parquet: the top candidates per employee for the selected
  variant, with feature values and scores (used for explanations in the app).
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pyarrow as pa

from orgchart.inference.data import load_people
from orgchart.inference.decide import decide
from orgchart.inference.features import FEATURES, CandidateSettings, build_candidates, company_people
from orgchart.inference.naive import naive_predictions
from orgchart.inference.scoring import Model
from orgchart.paths import DatasetPaths
from orgchart.schemas import PREDICTIONS
from orgchart.storage.parquet_io import write_table

CANDIDATE_META = (
    ("company_id", pa.string()), ("employee_id", pa.string()), ("candidate_id", pa.string()), ("rank", pa.int16()),
    ("p_direct", pa.float64()), ("q_direct", pa.float64()), ("p_ancestor", pa.float64()),
    ("score_direct", pa.float64()), ("score_ancestor", pa.float64()),
    ("e_level", pa.int8()), ("m_level", pa.int8()), ("m_is_ceo", pa.bool_()),
    ("n_between_dept", pa.int16()), ("n_between_strict", pa.int16()), ("n_equiv", pa.int16()),
)
CANDIDATES = pa.schema([pa.field(n, t) for n, t in CANDIDATE_META] + [pa.field(f, pa.float32()) for f in FEATURES])


def iter_company_candidates(people: pd.DataFrame, settings: CandidateSettings = CandidateSettings(),
                            ) -> Iterator[tuple[str, pd.DataFrame, pd.DataFrame]]:
    """Yield (company_id, people frame, candidate pairs) per company, in company order."""
    for company_id, group in people.groupby("company_id", sort=True):
        frame = company_people(group)
        pairs = build_candidates(frame, settings)
        pairs["company_id"] = company_id
        yield company_id, frame, pairs


def predict_people(people: pd.DataFrame, model: Model) -> pd.DataFrame:
    """Predictions for the given profiles (any subset of whole companies) without writing anything."""
    parts = [decide(frame, model.score_pairs(pairs), model)[0] for _, frame, pairs in iter_company_candidates(people)]
    return pd.concat(parts, ignore_index=True)


DECISION_COLUMNS = ("employee_id", "status", "reports_to_id", "ancestor_id", "confidence")


def decision_view(predictions: pd.DataFrame) -> pd.DataFrame:
    """Canonical string view of the decision columns, sorted by employee (for determinism checks)."""
    df = predictions[list(DECISION_COLUMNS)].sort_values("employee_id").reset_index(drop=True).astype(object)
    return df.where(df.notna(), "").astype(str)


def _to_table(df: pd.DataFrame, schema: pa.Schema) -> pa.Table:
    return pa.Table.from_pandas(df[[f.name for f in schema]], schema=schema, preserve_index=False)


def infer_dataset(root: str | Path, variants: tuple[str, ...] = ("rules", "learned"), selected: str | None = None,
                  ) -> dict:
    """Score every company with each available model variant and write predictions."""
    paths = DatasetPaths.of(root)
    started = time.perf_counter()
    models = {v: Model.load(paths.model(v)) for v in variants if paths.model(v).exists()}
    if not models:
        raise FileNotFoundError(f"no model files in {paths.models_dir}; run `orgchart fit` first")
    if selected is None and paths.model("selected").exists():
        selected = Model.load(paths.model("selected")).variant
    people = load_people(root)
    preds: dict[str, list[pd.DataFrame]] = {v: [] for v in [*models, "naive"]}
    cands: list[pd.DataFrame] = []
    for _, frame, pairs in iter_company_candidates(people):
        preds["naive"].append(naive_predictions(frame))
        for variant, model in models.items():
            p, top = decide(frame, model.score_pairs(pairs), model)
            preds[variant].append(p)
            if variant == selected:
                cands.append(top)

    summary: dict = {"companies": int(people["company_id"].nunique()), "profiles": len(people), "variants": {}}
    for variant, parts in preds.items():
        df = pd.concat(parts, ignore_index=True)
        df["model_variant"] = variant
        for col, default in (("candidate_ids", None), ("confidence", None), ("direct_confidence", None),
                             ("top_candidate_id", None),
                             ("runner_up_id", None), ("runner_up_confidence", None), ("ancestor_id", None),
                             ("explanation", None), ("n_candidates", 0)):
            if col not in df:
                df[col] = [[] for _ in range(len(df))] if col in ("candidate_ids", "explanation") else default
        write_table(_to_table(df, PREDICTIONS), paths.predictions(variant), PREDICTIONS)
        summary["variants"][variant] = df["status"].value_counts().to_dict()
        if variant == selected:
            write_table(_to_table(df, PREDICTIONS), paths.predictions("selected"), PREDICTIONS)
    if cands:
        write_table(_to_table(pd.concat(cands, ignore_index=True), CANDIDATES), paths.candidates, CANDIDATES)
    summary["selected"] = selected
    summary["seconds"] = round(time.perf_counter() - started, 1)
    return summary
