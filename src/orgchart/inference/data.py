"""Load inference inputs: observed employees joined with their normalised titles."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from orgchart.paths import DatasetPaths
from orgchart.storage.observed import connect_observed

PEOPLE_COLUMNS = (
    "employee_id", "company_id", "full_name", "title_raw", "title_clean", "department_raw", "location", "start_date",
    "level", "level_name", "level_confidence", "track", "manages_people", "management_confidence",
    "department", "department_confidence", "specialty", "scope", "scope_type", "is_contractor", "is_intern",
    "is_non_employee", "parse_confidence", "reason_codes",
)


def load_people(root: str | Path) -> pd.DataFrame:
    """One row per observed profile with the fields inference may use, sorted by company and employee."""
    paths = DatasetPaths.of(root)
    if not paths.normalized_titles.exists():
        raise FileNotFoundError(f"{paths.normalized_titles} not found; run `orgchart normalize` first")
    con = connect_observed(root)
    cols = ", ".join(f"n.{c}" if c not in ("full_name", "location", "start_date") else f"e.{c}"
                     for c in PEOPLE_COLUMNS)
    return con.execute(f"""
        SELECT {cols}, c.split, c.industry, c.company_name
        FROM read_parquet('{paths.normalized_titles}') n
        JOIN employees e USING (employee_id)
        JOIN companies c ON c.company_id = n.company_id
        ORDER BY n.company_id, n.employee_id""").df()
