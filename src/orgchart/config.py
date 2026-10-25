"""Typed generation config, loaded from YAML with optional dotted-path overrides."""

from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Probability = Field(ge=0.0, le=1.0)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DatasetConfig(_Strict):
    name: str
    n_companies: int = Field(ge=1)
    seed: int
    snapshot_date: dt.date
    min_observed_employees: int = Field(ge=1)


class SplitConfig(_Strict):
    train: float = Probability
    calibration: float = Probability
    test: float = Probability

    @model_validator(mode="after")
    def _sums_to_one(self) -> SplitConfig:
        total = self.train + self.calibration + self.test
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"split ratios must sum to 1, got {total}")
        return self


class CompanySizeConfig(_Strict):
    observed_median: int = Field(ge=1)
    sigma: float = Field(ge=0.0)
    max_observed: int = Field(ge=1)


class StructureConfig(_Strict):
    span: dict[int, tuple[int, int]]
    skip_level_probability: float = Probability
    regional_probability: float = Probability
    divisional_probability: float = Probability
    divisional_min_size: int
    scope_min_department_size: int
    executive_assistant_min_size: int
    executive_assistant_probability: float = Probability
    chief_of_staff_min_size: int
    chief_of_staff_probability: float = Probability
    exec_assistant_for_c_level_min_size: int
    exec_assistant_for_c_level_probability: float = Probability
    contractor_rate: float = Probability
    board_profiles_mean: float = Field(ge=0.0)
    board_profiles_max: int = Field(ge=0)

    @field_validator("span")
    @classmethod
    def _valid_spans(cls, v: dict[int, tuple[int, int]]) -> dict[int, tuple[int, int]]:
        for level, (lo, hi) in v.items():
            if not 2 <= lo <= hi:
                raise ValueError(f"span for level {level} must satisfy 2 <= min <= max")
        missing = set(range(5, 13)) - set(v)
        if missing:
            raise ValueError(f"span missing levels {sorted(missing)}")
        return v

    @model_validator(mode="after")
    def _scope_probabilities(self) -> StructureConfig:
        if self.regional_probability + self.divisional_probability > 1.0:
            raise ValueError("regional_probability + divisional_probability must be <= 1")
        return self


class CoverageConfig(_Strict):
    mean: float = Probability
    concentration: float = Field(gt=0.0)
    min: float = Probability
    max: float = Probability
    ceo_observed_probability: float = Probability
    level_missing_multiplier: dict[int, float]

    @model_validator(mode="after")
    def _bounds(self) -> CoverageConfig:
        if not 0 < self.min <= self.mean <= self.max <= 1:
            raise ValueError("coverage must satisfy 0 < min <= mean <= max <= 1")
        missing = set(range(0, 13)) - set(self.level_missing_multiplier)
        if missing:
            raise ValueError(f"level_missing_multiplier missing levels {sorted(missing)}")
        return self


class NoiseConfig(_Strict):
    missing_department: float = Probability
    mislabeled_department: float = Probability
    missing_location: float = Probability
    greater_area_location: float = Probability
    missing_start_date: float = Probability
    stale_title: float = Probability
    title_inflation_small_company: float = Probability
    small_company_size: int
    vague_title: float = Probability
    empty_title: float = Probability
    contractor_marker: float = Probability
    transforms: dict[str, float]
    holdout_transforms: dict[str, float]

    @field_validator("transforms", "holdout_transforms")
    @classmethod
    def _probabilities(cls, v: dict[str, float]) -> dict[str, float]:
        for name, p in v.items():
            if not 0.0 <= p <= 1.0:
                raise ValueError(f"transform {name} probability {p} not in [0, 1]")
        return v


class GenerationConfig(_Strict):
    dataset: DatasetConfig
    splits: SplitConfig
    company_size: CompanySizeConfig
    industries: dict[str, float]
    countries: dict[str, float]
    structure: StructureConfig
    coverage: CoverageConfig
    noise: NoiseConfig
    realism_targets: dict[str, tuple[float, float]]

    @model_validator(mode="after")
    def _cross_checks(self) -> GenerationConfig:
        from orgchart.generation.catalog import ARCHETYPES, COUNTRIES

        if unknown := set(self.industries) - set(ARCHETYPES):
            raise ValueError(f"unknown industries {sorted(unknown)}")
        if unknown := set(self.countries) - set(COUNTRIES):
            raise ValueError(f"unknown countries {sorted(unknown)}")
        if self.company_size.max_observed < self.dataset.min_observed_employees:
            raise ValueError("company_size.max_observed must be >= dataset.min_observed_employees")
        smallest = min(r for r in self.splits.model_dump().values() if r > 0)
        if self.dataset.n_companies * smallest < 1:
            raise ValueError(
                f"n_companies={self.dataset.n_companies} is too small to give every split a company "
                f"(need >= {math.ceil(1 / smallest)} with a {smallest:.0%} split)"
            )
        return self

    def to_jsonable(self) -> dict[str, Any]:
        return json.loads(self.model_dump_json())

    def fingerprint(self) -> str:
        payload = json.dumps(self.to_jsonable(), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:16]


def _set_dotted(data: dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = data
    for key in keys[:-1]:
        if key not in node or not isinstance(node[key], dict):
            raise KeyError(f"config override path not found: {dotted}")
        node = node[key]
    if keys[-1] not in node:
        raise KeyError(f"config override path not found: {dotted}")
    node[keys[-1]] = value


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> GenerationConfig:
    """Load YAML config and apply dotted-path overrides, e.g. {"dataset.seed": 7}."""
    raw = yaml.safe_load(Path(path).read_text())
    data = copy.deepcopy(raw)
    for dotted, value in (overrides or {}).items():
        _set_dotted(data, dotted, value)
    return GenerationConfig.model_validate(data)


def parse_override(expr: str) -> tuple[str, Any]:
    """Parse a CLI override of the form key.path=value (value is YAML-typed)."""
    if "=" not in expr:
        raise ValueError(f"override must look like key.path=value, got {expr!r}")
    key, raw_value = expr.split("=", 1)
    return key.strip(), yaml.safe_load(raw_value)
