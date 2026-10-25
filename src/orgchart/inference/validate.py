"""Structural checks on predicted reporting lines (observed data only).

These checks guard the inference output contract:

* exactly one row per observed profile, with a valid status;
* status-specific fields are set consistently;
* no self-reporting and no cross-company references;
* DIRECT managers are strictly more senior and hold management authority;
* the DIRECT graph is acyclic, and board/advisor profiles get no manager;
* confidences are probabilities.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from orgchart.inference.data import load_people
from orgchart.paths import DatasetPaths
from orgchart.schemas import PREDICTION_STATUSES


def _check(name: str, bad: int, detail: str = "") -> dict:
    return {"name": name, "passed": bad == 0, "violations": int(bad), "detail": detail}


def _has_cycle(edges: dict[str, str]) -> int:
    """Number of employees whose DIRECT chain revisits a node."""
    state: dict[str, int] = {}
    bad = 0
    for start in edges:
        path, node = [], start
        while node in edges and state.get(node) is None:
            state[node] = 1
            path.append(node)
            node = edges[node]
        if state.get(node) == 1 and node in path:
            bad += 1
        for n in path:
            state[n] = 2
    return bad


def validate_predictions(predictions: pd.DataFrame, people: pd.DataFrame) -> list[dict]:
    pred = predictions.merge(people[["employee_id", "company_id", "level", "manages_people", "track",
                                     "is_non_employee"]], on=["employee_id", "company_id"], how="left",
                             indicator=True)
    info = people.set_index("employee_id")
    company = info["company_id"].to_dict()
    level = info["level"].to_dict()
    authority = ((info["manages_people"].fillna(False)) | (info["track"] == "executive")).to_dict()
    checks = [
        _check("one_row_per_observed_profile",
               abs(len(predictions) - len(people)) + int(predictions["employee_id"].duplicated().sum())
               + int((pred["_merge"] != "both").sum())),
        _check("valid_status", int((~predictions["status"].isin(PREDICTION_STATUSES)).sum())),
    ]
    direct = pred[pred["status"] == "DIRECT"]
    amb = pred[pred["status"] == "AMBIGUOUS"]
    anc = pred[pred["status"] == "ANCESTOR"]
    others = pred[pred["status"] != "DIRECT"]
    checks.append(_check("status_fields_consistent",
                         int(direct["reports_to_id"].isna().sum()) + int(others["reports_to_id"].notna().sum())
                         + int(anc["ancestor_id"].isna().sum())
                         + int(pred[pred["status"] != "ANCESTOR"]["ancestor_id"].notna().sum())
                         + int((amb["candidate_ids"].map(len) < 2).sum())
                         + int((pred[pred["status"] != "AMBIGUOUS"]["candidate_ids"].map(len) > 0).sum())))
    refs = [(r.employee_id, r.reports_to_id) for r in direct.itertuples()]
    refs += [(r.employee_id, r.ancestor_id) for r in anc.itertuples()]
    refs += [(r.employee_id, c) for r in amb.itertuples() for c in r.candidate_ids]
    checks.append(_check("no_self_reporting", sum(e == m for e, m in refs)))
    checks.append(_check("same_company_references", sum(company.get(m) != company[e] for e, m in refs)))
    def more_senior(e: str, m: str) -> bool:
        le, lm = level.get(e), level.get(m)
        return le is not None and lm is not None and not pd.isna(le) and not pd.isna(lm) and lm > le

    checks.append(_check("manager_strictly_more_senior", sum(not more_senior(e, m) for e, m in refs)))
    checks.append(_check("manager_has_authority", sum(not authority.get(m, False) for e, m in refs)))
    checks.append(_check("acyclic_direct_graph", _has_cycle(dict(zip(direct["employee_id"], direct["reports_to_id"],
                                                                    strict=True)))))
    checks.append(_check("non_employees_unassigned", int((pred["is_non_employee"].fillna(False).astype(bool)
                                                           & (pred["status"] != "UNRESOLVED")).sum())))
    conf = pred["confidence"].dropna().to_numpy(np.float64)
    checks.append(_check("confidence_in_unit_interval", int(((conf < 0) | (conf > 1)).sum())))
    return checks


def validate_dataset_predictions(root: str | Path, variant: str = "selected") -> list[dict]:
    paths = DatasetPaths.of(root)
    predictions = pd.read_parquet(paths.predictions(variant))
    return validate_predictions(predictions, load_people(root))
