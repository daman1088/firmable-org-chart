import pytest
from pydantic import ValidationError

from orgchart.config import load_config, parse_override
from tests.conftest import CONFIG_PATH


def test_default_config_meets_assessment_requirements(config):
    assert config.dataset.n_companies >= 600
    assert config.dataset.min_observed_employees >= 100
    assert (config.splits.train, config.splits.calibration, config.splits.test) == (0.70, 0.15, 0.15)


def test_overrides_apply():
    cfg = load_config(CONFIG_PATH, {"dataset.seed": 7, "noise.missing_department": 0.5})
    assert cfg.dataset.seed == 7
    assert cfg.noise.missing_department == 0.5


def test_unknown_override_path_rejected():
    with pytest.raises(KeyError):
        load_config(CONFIG_PATH, {"dataset.nope": 1})


@pytest.mark.parametrize(
    "overrides",
    [
        {"splits.train": 0.9},  # ratios no longer sum to 1
        {"noise.missing_department": 1.5},  # probability out of range
        {"coverage.min": 0.99},  # min > mean
        {"industries": {"crypto": 1.0}},  # unknown archetype
        {"company_size.max_observed": 50},  # below the 100-employee guarantee
        {"dataset.n_companies": 4},  # a 15% split would get zero companies
    ],
)
def test_invalid_configs_rejected(overrides):
    with pytest.raises(ValidationError):
        load_config(CONFIG_PATH, overrides)


def test_fingerprint_is_stable_and_sensitive():
    a = load_config(CONFIG_PATH)
    b = load_config(CONFIG_PATH)
    c = load_config(CONFIG_PATH, {"dataset.seed": 99})
    assert a.fingerprint() == b.fingerprint() != c.fingerprint()


def test_parse_override_types_values():
    assert parse_override("dataset.seed=42") == ("dataset.seed", 42)
    assert parse_override("noise.case=0.1") == ("noise.case", 0.1)
    assert parse_override("dataset.name=demo") == ("dataset.name", "demo")
    with pytest.raises(ValueError):
        parse_override("no-equals-sign")
