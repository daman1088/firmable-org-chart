import json

import pytest
from typer.testing import CliRunner

from orgchart.cli import app
from tests.conftest import CONFIG_PATH

pytestmark = pytest.mark.slow
runner = CliRunner()


def test_generate_validate_and_query(tmp_path):
    out = tmp_path / "cli"
    result = runner.invoke(app, [
        "generate", "--out", str(out), "--config", str(CONFIG_PATH), "--n-companies", "8", "--seed", "5",
        "--set", "company_size.observed_median=120", "--workers", "1",
    ])
    assert result.exit_code == 0, result.output
    assert "PASS" in result.output and "FAIL" not in result.output
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["config"]["company_size"]["observed_median"] == 120
    assert manifest["counts"]["companies"] == 8

    result = runner.invoke(app, ["sql", "SELECT count(*) AS n FROM companies", "--dataset", str(out)])
    assert result.exit_code == 0 and "8" in result.output

    result = runner.invoke(app, ["sql", "SELECT count(*) FROM truth_employees", "--dataset", str(out)])
    assert result.exit_code != 0  # truth is not exposed without --with-truth

    result = runner.invoke(app, ["show-company", "C0000", "--dataset", str(out), "--max-depth", "1"])
    assert result.exit_code == 0 and "L12" in result.output

    result = runner.invoke(app, ["validate", str(out)])
    assert result.exit_code == 0
