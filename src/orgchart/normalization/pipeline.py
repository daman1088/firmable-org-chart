"""Normalise every observed employee title in a dataset.

Reads observed tables only (orgchart.storage.observed) and writes
<dataset>/derived/normalized_titles.parquet.
"""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path

import pyarrow as pa

from orgchart.normalization.company_context import apply_company_context
from orgchart.normalization.parser import NORMALIZER_VERSION, ParsedTitle, parse_title
from orgchart.paths import DatasetPaths
from orgchart.schemas import NORMALIZED_TITLES
from orgchart.storage.observed import read_observed_companies, read_observed_employees
from orgchart.storage.parquet_io import write_table


def _row(emp: dict, p: ParsedTitle) -> dict:
    return {
        "employee_id": emp["employee_id"],
        "company_id": emp["company_id"],
        "title_raw": emp["title"],
        "department_raw": emp["department"],
        "title_clean": p.title_clean,
        "level": p.level,
        "level_name": p.level_name,
        "level_confidence": p.level_confidence,
        "track": p.track,
        "manages_people": p.manages_people,
        "management_confidence": p.management_confidence,
        "department": p.department,
        "department_confidence": p.department_confidence,
        "department_source": p.department_source,
        "specialty": p.specialty,
        "specialty_confidence": p.specialty_confidence,
        "scope": p.scope,
        "scope_type": p.scope_type,
        "is_contractor": p.is_contractor,
        "is_intern": p.is_intern,
        "is_interim": p.is_interim,
        "is_founder": p.is_founder,
        "is_non_employee": p.is_non_employee,
        "parse_confidence": p.parse_confidence,
        "reason_codes": list(p.reason_codes),
        "rules_fired": list(p.rules_fired),
        "explanation": list(p.explanation),
        "normalizer_version": NORMALIZER_VERSION,
    }


def normalize_employees(employees: list[dict], companies: dict[str, dict]) -> list[dict]:
    by_company: dict[str, list[dict]] = defaultdict(list)
    for emp in employees:
        by_company[emp["company_id"]].append(emp)
    rows = []
    for company_id in sorted(by_company):
        company = companies[company_id]
        emps = by_company[company_id]
        parsed = [parse_title(e["title"], e["department"], company["industry"], company["company_name"]) for e in emps]
        parsed = apply_company_context(parsed)
        rows.extend(_row(e, p) for e, p in zip(emps, parsed, strict=True))
    return rows


def normalize_dataset(root: str | Path) -> dict:
    paths = DatasetPaths.of(root)
    started = time.perf_counter()
    companies = {c["company_id"]: c for c in read_observed_companies(root).to_pylist()}
    employees = read_observed_employees(root).to_pylist()
    rows = normalize_employees(employees, companies)
    table = pa.Table.from_pylist(rows, schema=NORMALIZED_TITLES)
    write_table(table, paths.normalized_titles, NORMALIZED_TITLES)
    n = len(rows)
    return {
        "rows": n,
        "seconds": round(time.perf_counter() - started, 1),
        "level_resolved_pct": round(100 * sum(r["level"] is not None for r in rows) / n, 2),
        "department_resolved_pct": round(100 * sum(r["department"] is not None for r in rows) / n, 2),
        "non_employee_profiles": sum(r["is_non_employee"] for r in rows),
        "output": str(paths.normalized_titles),
    }
