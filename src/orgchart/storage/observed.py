"""Read access to OBSERVED data only. This is the only storage module inference may use."""

from pathlib import Path

import duckdb
import pyarrow as pa

from orgchart.paths import DatasetPaths
from orgchart.schemas import OBSERVED_COMPANIES, OBSERVED_EMPLOYEES
from orgchart.storage.parquet_io import read_table


def connect_observed(root: str | Path) -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB connection with views `companies` and `employees` (observed only)."""
    paths = DatasetPaths.of(root)
    con = duckdb.connect()
    con.execute(f"CREATE VIEW companies AS SELECT * FROM read_parquet('{paths.observed_companies}')")
    con.execute(f"CREATE VIEW employees AS SELECT * FROM read_parquet('{paths.observed_employees}')")
    return con


def read_observed_companies(root: str | Path) -> pa.Table:
    return read_table(DatasetPaths.of(root).observed_companies, OBSERVED_COMPANIES)


def read_observed_employees(root: str | Path) -> pa.Table:
    return read_table(DatasetPaths.of(root).observed_employees, OBSERVED_EMPLOYEES)
