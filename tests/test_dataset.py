"""End-to-end tests on a small generated dataset."""

import json
import re

import pyarrow.parquet as pq
import pytest

from orgchart.generation.generator import _size_band, assign_splits, generate_dataset, plan_companies
from orgchart.generation.validation import validate_dataset
from orgchart.paths import DatasetPaths
from orgchart.schemas import FORBIDDEN_OBSERVED_FRAGMENTS, OBSERVED_EMPLOYEES, TRUTH_EMPLOYEES
from orgchart.storage.observed import connect_observed, read_observed_employees
from orgchart.storage.truth import connect_with_truth, read_truth_employees

pytestmark = pytest.mark.slow


def test_all_structural_checks_pass(small_dataset):
    report = validate_dataset(small_dataset)
    failed = [c for c in report["checks"] if not c["passed"]]
    assert not failed, failed
    assert (DatasetPaths.of(small_dataset).reports_dir / "realism.md").exists()


def test_every_company_has_100_observed_employees(small_dataset):
    con = connect_with_truth(small_dataset)
    counts = con.execute("""
        SELECT company_id, count(*) FROM truth_employees
        WHERE is_observed AND role_type <> 'board' GROUP BY 1""").fetchall()
    assert len(counts) == 10
    assert min(n for _, n in counts) >= 100
    # Even counting only the observed table (what inference sees), every company has >= 100 profiles.
    obs = connect_observed(small_dataset).execute("SELECT min(n) FROM (SELECT count(*) n FROM employees GROUP BY company_id)")
    assert obs.fetchone()[0] >= 100


def test_observed_tables_contain_no_truth(small_dataset):
    table = read_observed_employees(small_dataset)
    assert table.schema.equals(OBSERVED_EMPLOYEES)
    for name in table.column_names:
        assert not any(f in name for f in FORBIDDEN_OBSERVED_FRAGMENTS), name
    paths = DatasetPaths.of(small_dataset)
    assert {p.name for p in paths.observed_dir.iterdir()} == {"companies.parquet", "employees.parquet"}


def test_ids_are_opaque_and_order_does_not_leak_hierarchy(small_dataset):
    truth = read_truth_employees(small_dataset).to_pylist()
    assert all(re.fullmatch(r"E[0-9a-f]{12}", r["employee_id"]) for r in truth)
    obs = read_observed_employees(small_dataset).to_pylist()
    level = {r["employee_id"]: r["level"] for r in truth}
    # The first row of each company in the observed file should not systematically be the CEO.
    first_rows = {}
    for r in obs:
        first_rows.setdefault(r["company_id"], r["employee_id"])
    assert sum(level[e] == 12 for e in first_rows.values()) <= 2


def test_truth_schema_and_holdout_flag(small_dataset):
    assert read_truth_employees(small_dataset).schema.equals(TRUTH_EMPLOYEES)
    con = connect_with_truth(small_dataset)
    mismatch = con.execute("""
        SELECT count(*) FROM truth_companies t JOIN companies c USING (company_id)
        WHERE t.holdout_noise <> (c.split = 'test')""").fetchone()[0]
    assert mismatch == 0


def test_generation_is_reproducible_across_runs_and_worker_counts(tmp_path, small_config, small_dataset):
    a, b = tmp_path / "a", tmp_path / "b"
    generate_dataset(small_config, a, workers=1)
    generate_dataset(small_config, b, workers=3)
    ref = json.loads(DatasetPaths.of(small_dataset).manifest.read_text())["files"]
    for out in (a, b):
        files = json.loads(DatasetPaths.of(out).manifest.read_text())["files"]
        assert files == ref
    for rel in ref:
        assert pq.read_table(a / rel).equals(pq.read_table(b / rel))


def test_different_seed_changes_data(tmp_path, small_config, small_dataset):
    other = small_config.model_copy(update={"dataset": small_config.dataset.model_copy(update={"seed": 999})})
    generate_dataset(other, tmp_path / "c", workers=1)
    ref = json.loads(DatasetPaths.of(small_dataset).manifest.read_text())["files"]
    files = json.loads(DatasetPaths.of(tmp_path / "c").manifest.read_text())["files"]
    assert files != ref


def test_refuses_to_overwrite_without_force(small_config, small_dataset):
    with pytest.raises(FileExistsError):
        generate_dataset(small_config, small_dataset, workers=1)


def test_split_assignment_is_stratified_and_complete(config):
    plans = plan_companies(config)
    splits = [p.split for p in plans]
    assert set(splits) == {"train", "calibration", "test"}
    n = len(plans)
    ratios = config.splits.model_dump()
    for split, ratio in ratios.items():
        assert abs(splits.count(split) / n - ratio) < 0.01, split
    # Diversity is preserved inside every split: all archetypes, size bands and most countries appear.
    for split in ratios:
        members = [p for p in plans if p.split == split]
        assert {p.archetype for p in members} == set(config.industries), split
        assert {_size_band(p.observed_target) for p in members} == {0, 1, 2, 3}, split
        assert len({p.country for p in members}) >= 5, split
    # Every company has a distinct seed and plans are reproducible.
    assert len({p.seed for p in plans}) == n
    assert plan_companies(config) == plans


def test_assign_splits_tracks_ratios():
    import numpy as np

    strata = [("a", 0)] * 100 + [("b", 1)] * 100
    out = assign_splits(strata, {"train": 0.70, "calibration": 0.15, "test": 0.15}, np.random.default_rng(0))
    for key in (("a", 0), ("b", 1)):
        members = [s for s, k in zip(out, strata, strict=True) if k == key]
        assert abs(members.count("train") - 70) <= 1
        assert abs(members.count("calibration") - 15) <= 1
        assert abs(members.count("test") - 15) <= 1
