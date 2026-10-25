"""Ground-truth labels for reporting-line evaluation and model fitting (truth side only)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from orgchart.storage.truth import connect_with_truth


def employee_truth(root: str | Path) -> pd.DataFrame:
    """One row per observed profile: true manager, nearest observed manager, and evaluation attributes."""
    con = connect_with_truth(root)
    return con.execute("""
        SELECT e.employee_id, e.company_id, c.split, t.manager_id AS true_manager_id,
               t.nearest_observed_manager_id AS true_nearest_observed_id,
               t.manager_id IS NOT NULL AND t.manager_id = t.nearest_observed_manager_id AS manager_observed,
               t.level AS true_level, t.role_type, t.department AS true_department,
               tc.archetype, tc.scope_kind, tc.observed_employee_count
        FROM employees e
        JOIN truth_employees t USING (employee_id)
        JOIN companies c ON c.company_id = e.company_id
        JOIN truth_companies tc ON tc.company_id = e.company_id""").df()


def true_ancestors(root: str | Path) -> pd.DataFrame:
    """(employee_id, ancestor_id) for every observed employee and every true ancestor (observed or not)."""
    con = connect_with_truth(root)
    return con.execute("""
        WITH RECURSIVE chain(employee_id, ancestor_id, hops) AS (
            SELECT t.employee_id, t.manager_id, 1 FROM truth_employees t
            JOIN employees e USING (employee_id) WHERE t.manager_id IS NOT NULL
            UNION ALL
            SELECT c.employee_id, t.manager_id, c.hops + 1 FROM chain c
            JOIN truth_employees t ON t.employee_id = c.ancestor_id
            WHERE t.manager_id IS NOT NULL AND c.hops < 30)
        SELECT employee_id, ancestor_id, hops FROM chain""").df()


def label_pairs(pairs: pd.DataFrame, truth: pd.DataFrame, ancestors: pd.DataFrame) -> pd.DataFrame:
    """Add `is_direct` and `is_ancestor` to (employee_id, candidate_id) pairs."""
    out = pairs.merge(truth[["employee_id", "true_manager_id"]], on="employee_id", how="left")
    out["is_direct"] = out["candidate_id"].to_numpy() == out["true_manager_id"].to_numpy()
    anc = ancestors[["employee_id", "ancestor_id"]].rename(columns={"ancestor_id": "candidate_id"})
    anc["is_ancestor"] = True
    out = out.merge(anc, on=["employee_id", "candidate_id"], how="left")
    out["is_ancestor"] = out["is_ancestor"].fillna(False).astype(bool)
    return out.drop(columns=["true_manager_id"])
