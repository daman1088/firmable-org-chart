"""On-disk dataset layout.

    <root>/
      manifest.json
      observed/companies.parquet   observed/employees.parquet    <- inference inputs
      truth/companies.parquet      truth/employees.parquet       <- evaluation only
      derived/normalized_titles.parquet                          <- normalisation output (Phase 2)
      derived/candidates.parquet   derived/predictions*.parquet  <- inference output (Phase 3+)
      models/model_<variant>.json                                <- fitted scorers, calibration, thresholds
      reports/validation.json      reports/realism.md      reports/normalization_eval.{json,md}
      reports/inference_eval.{json,md}
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetPaths:
    root: Path

    @classmethod
    def of(cls, root: str | Path) -> "DatasetPaths":
        return cls(Path(root))

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    @property
    def observed_dir(self) -> Path:
        return self.root / "observed"

    @property
    def truth_dir(self) -> Path:
        return self.root / "truth"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    @property
    def derived_dir(self) -> Path:
        return self.root / "derived"

    @property
    def normalized_titles(self) -> Path:
        return self.derived_dir / "normalized_titles.parquet"

    @property
    def candidates(self) -> Path:
        return self.derived_dir / "candidates.parquet"

    def predictions(self, variant: str = "selected") -> Path:
        return self.derived_dir / ("predictions.parquet" if variant == "selected" else f"predictions_{variant}.parquet")

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    def model(self, variant: str = "selected") -> Path:
        return self.models_dir / f"model_{variant}.json"

    @property
    def observed_companies(self) -> Path:
        return self.observed_dir / "companies.parquet"

    @property
    def observed_employees(self) -> Path:
        return self.observed_dir / "employees.parquet"

    @property
    def truth_companies(self) -> Path:
        return self.truth_dir / "companies.parquet"

    @property
    def truth_employees(self) -> Path:
        return self.truth_dir / "employees.parquet"

    def data_files(self) -> list[Path]:
        return [self.observed_companies, self.observed_employees, self.truth_companies, self.truth_employees]
