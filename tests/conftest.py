from pathlib import Path

import pytest

from orgchart.config import GenerationConfig, load_config
from orgchart.generation.generator import generate_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "generation.yaml"

# Small but representative: every archetype is reachable, sizes stay test-friendly.
SMALL_OVERRIDES = {
    "dataset.name": "pytest",
    "dataset.n_companies": 10,
    "dataset.seed": 1234,
    "company_size.observed_median": 160,
    "company_size.max_observed": 700,
}


@pytest.fixture(scope="session")
def config() -> GenerationConfig:
    return load_config(CONFIG_PATH)


@pytest.fixture(scope="session")
def small_config() -> GenerationConfig:
    return load_config(CONFIG_PATH, SMALL_OVERRIDES)


@pytest.fixture(scope="session")
def small_dataset(tmp_path_factory, small_config) -> Path:
    out = tmp_path_factory.mktemp("datasets") / "small"
    generate_dataset(small_config, out, workers=1)
    return out
