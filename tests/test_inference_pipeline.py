"""End to end on a small generated dataset: normalise -> fit -> infer -> check -> evaluate, plus the CLI."""

import ast
import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from orgchart.cli import app
from orgchart.evaluation.inference_eval import evaluate_inference
from orgchart.inference.data import load_people
from orgchart.inference.pipeline import decision_view, infer_dataset, predict_people
from orgchart.inference.scoring import Model
from orgchart.inference.validate import validate_dataset_predictions
from orgchart.learning.fit import fit_models
from orgchart.normalization.pipeline import normalize_dataset
from orgchart.paths import DatasetPaths
from orgchart.schemas import PREDICTIONS

REPO = Path(__file__).resolve().parents[1]
runner = CliRunner()


def test_learning_never_names_the_test_split():
    """Model fitting may use train and calibration companies only."""
    for path in (REPO / "src/orgchart/learning").rglob("*.py"):
        tree = ast.parse(path.read_text())
        consts = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
        assert "test" not in consts, f"{path} references the test split"


@pytest.fixture(scope="module")
def built(small_dataset):
    normalize_dataset(small_dataset)
    fit = fit_models(small_dataset)
    inferred = infer_dataset(small_dataset)
    return small_dataset, fit, inferred


@pytest.mark.slow
def test_models_written_and_selected(built):
    root, fit, _ = built
    paths = DatasetPaths.of(root)
    for variant in ("rules", "learned", "selected"):
        model = Model.load(paths.model(variant))
        assert 0 < model.thresholds.direct <= 1.01
    assert fit["selected"] in ("rules", "learned")
    assert Model.load(paths.model("selected")).variant == fit["selected"]
    meta = Model.load(paths.model("learned")).metadata
    assert meta["training"]["split"] == "train" and meta["training"]["positives_direct"] > 0


@pytest.mark.slow
def test_predictions_cover_every_profile_and_pass_invariants(built):
    root, _, inferred = built
    paths = DatasetPaths.of(root)
    people = load_people(root)
    for variant in ("selected", "naive", "rules", "learned"):
        pred = pd.read_parquet(paths.predictions(variant))
        assert len(pred) == len(people) and set(pred["employee_id"]) == set(people["employee_id"])
        assert list(pred.columns) == [f.name for f in PREDICTIONS]
        assert all(c["passed"] for c in validate_dataset_predictions(root, variant)), variant
    assert paths.candidates.exists()
    assert inferred["variants"]["naive"].get("DIRECT", 0) > 0.9 * len(people)


@pytest.mark.slow
def test_inference_is_deterministic(built):
    root, _, _ = built
    paths = DatasetPaths.of(root)
    stored = pd.read_parquet(paths.predictions())
    again = predict_people(load_people(root), Model.load(paths.model()))
    assert decision_view(again).equals(decision_view(stored))


@pytest.mark.slow
def test_evaluation_report(built):
    root, _, _ = built
    report = evaluate_inference(root)
    test = report["variants"]["selected"]["test"]
    naive = report["variants"]["naive"]["test"]
    assert naive["direct"]["coverage_pct"] > 90
    assert test["resolved"]["coverage_pct"] is not None
    if test["resolved"]["precision_pct"] is not None:
        assert test["resolved"]["precision_pct"] > naive["direct"]["precision_pct"]
    assert all(all(v.values()) for v in report["invariants"].values())
    assert (DatasetPaths.of(root).reports_dir / "inference_eval.md").exists()
    assert set(report["test_breakdowns"]) == {"archetype", "company_size", "seniority", "scope_kind",
                                              "manager_is_observed"}


@pytest.mark.slow
def test_cli_infer_check_evaluate_explain_verify(built):
    root, _, _ = built
    result = runner.invoke(app, ["infer", str(root)])
    assert result.exit_code == 0, result.output
    result = runner.invoke(app, ["check-predictions", str(root)])
    assert result.exit_code == 0 and "FAIL" not in result.output, result.output
    result = runner.invoke(app, ["evaluate", str(root)])
    assert result.exit_code == 0 and "TEST" in result.output, result.output
    pred = pd.read_parquet(DatasetPaths.of(root).predictions())
    some = pred[pred["n_candidates"] > 0]["employee_id"].iloc[0]
    result = runner.invoke(app, ["explain", some, "--dataset", str(root)])
    assert result.exit_code == 0 and "Ranked candidates" in result.output, result.output
    result = runner.invoke(app, ["verify", str(root), "--min-companies", "10"])
    assert result.exit_code == 0, result.output
    result = runner.invoke(app, ["verify", str(root), "--min-companies", "600"])
    assert result.exit_code == 1  # the small fixture is not the delivery dataset


@pytest.mark.slow
def test_cli_pipeline_end_to_end(tmp_path):
    out = tmp_path / "pipe"
    result = runner.invoke(app, ["generate", "--out", str(out), "--n-companies", "8", "--seed", "11",
                                 "--set", "company_size.observed_median=120", "--workers", "1"])
    assert result.exit_code == 0, result.output
    result = runner.invoke(app, ["pipeline", str(out)])
    assert result.exit_code == 0, result.output
    timings = json.loads((out / "reports" / "pipeline_runtime.json").read_text())
    assert {"normalize", "fit", "infer", "check-predictions", "evaluate", "total"} <= set(timings)
