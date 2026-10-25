"""Integration tests: normalise a generated dataset end to end, evaluate it, drive the CLI."""

import json

import duckdb
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from orgchart.cli import app
from orgchart.evaluation.normalization_eval import evaluate_normalization
from orgchart.normalization.pipeline import normalize_dataset
from orgchart.paths import DatasetPaths
from orgchart.schemas import NORMALIZED_TITLES
from orgchart.storage.observed import read_observed_employees

pytestmark = pytest.mark.slow
runner = CliRunner()


@pytest.fixture(scope="module")
def normalized(small_dataset):
    summary = normalize_dataset(small_dataset)
    return small_dataset, summary


def test_one_row_per_observed_employee_with_schema(normalized):
    root, summary = normalized
    table = pq.read_table(DatasetPaths.of(root).normalized_titles)
    assert table.schema.equals(NORMALIZED_TITLES, check_metadata=False)
    observed = read_observed_employees(root)
    assert table.num_rows == observed.num_rows == summary["rows"]
    assert sorted(table.column("employee_id").to_pylist()) == sorted(observed.column("employee_id").to_pylist())


def test_output_is_bounded_and_explained(normalized):
    root, _ = normalized
    path = DatasetPaths.of(root).normalized_titles
    row = duckdb.sql(f"""
        SELECT min(parse_confidence), max(parse_confidence), min(len(explanation)),
               count(*) FILTER (WHERE level NOT BETWEEN 0 AND 12),
               count(*) FILTER (WHERE level IS NULL AND NOT is_non_employee
                                AND NOT list_contains(reason_codes, 'LEVEL_UNRESOLVED')
                                AND NOT list_contains(reason_codes, 'EMPTY_TITLE'))
        FROM '{path}'""").fetchall()[0]
    assert 0.0 <= row[0] <= row[1] <= 1.0
    assert row[2] >= 1
    assert row[3] == 0
    assert row[4] == 0  # an unresolved level always carries a reason code


def test_normalization_is_deterministic(normalized):
    root, _ = normalized
    path = DatasetPaths.of(root).normalized_titles
    first = pq.read_table(path)
    normalize_dataset(root)
    assert pq.read_table(path).equals(first)


def test_evaluation_meets_floors(normalized):
    root, _ = normalized
    report = evaluate_normalization(root)
    m = report["splits"]["all"]
    assert m["seniority"]["coverage_pct"] >= 98
    assert m["seniority"]["within_one_pct"] >= 95
    assert m["seniority"]["exact_pct"] >= 85
    assert m["department"]["accuracy_pct"] >= 97
    assert m["management"]["accuracy_pct"] >= 96
    assert m["management"]["staff_principal_title_false_manager_pct"] in (0.0, None)
    assert (DatasetPaths.of(root).reports_dir / "normalization_eval.md").exists()
    saved = json.loads((DatasetPaths.of(root).reports_dir / "normalization_eval.json").read_text())
    assert set(saved["splits"]) == {"train", "calibration", "test", "all"}


def test_cli_normalize_and_parse_title(small_dataset):
    result = runner.invoke(app, ["normalize", str(small_dataset)])
    assert result.exit_code == 0, result.output
    assert "Normalised" in result.output and "seniority exact" in result.output

    result = runner.invoke(app, ["parse-title", "Staff Software Engineer", "--department", "Engineering"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output[: result.output.rindex("}") + 1])
    assert payload["level"] == 4 and payload["manages_people"] is False
    assert "•" in result.output
