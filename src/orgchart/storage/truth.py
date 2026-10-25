"""Ground-truth access. EVALUATION / VALIDATION ONLY.

Never import this module from orgchart.normalization, orgchart.inference or the
Streamlit app's inference path; tests/test_isolation.py fails the build if you do.
"""

from pathlib import Path

import duckdb
import pyarrow as pa

from orgchart.paths import DatasetPaths
from orgchart.schemas import TRUTH_COMPANIES, TRUTH_EMPLOYEES
from orgchart.storage.observed import connect_observed
from orgchart.storage.parquet_io import read_table


def connect_with_truth(root: str | Path) -> duckdb.DuckDBPyConnection:
    """Observed views plus `truth_companies` and `truth_employees`."""
    paths = DatasetPaths.of(root)
    con = connect_observed(root)
    con.execute(f"CREATE VIEW truth_companies AS SELECT * FROM read_parquet('{paths.truth_companies}')")
    con.execute(f"CREATE VIEW truth_employees AS SELECT * FROM read_parquet('{paths.truth_employees}')")
    return con


def read_truth_companies(root: str | Path) -> pa.Table:
    return read_table(DatasetPaths.of(root).truth_companies, TRUTH_COMPANIES)


def read_truth_employees(root: str | Path) -> pa.Table:
    return read_table(DatasetPaths.of(root).truth_employees, TRUTH_EMPLOYEES)
