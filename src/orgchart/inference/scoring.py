"""Pair scorers, calibration maps and the model file format.

A scorer is a linear model over the interpretable features in
`features.FEATURES`: raw score = bias + sum(weight * feature). That makes every
score decomposable into per-feature contributions for explanations.

Two variants share this format:

* `rules`: hand-set weights (below), written from domain reasoning before any
  fitting. This is the deterministic, explainable baseline.
* `learned`: weights fitted by logistic regression on train companies (see
  `orgchart.learning`). The fitting code lives outside the inference package;
  inference only reads the resulting JSON.

Each scorer has two heads:

* `direct`: which candidate, if any, is e's direct manager? The direct
  scores of one employee's candidates are combined *listwise*:
  q_m = exp(s_m) / (1 + sum_k exp(s_k)). The "1" is the outside option: the
  manager is not among the observed candidates. So q sums to at most 1, and
  equally plausible candidates share probability.
* `ancestor`: is m somewhere above e in the chain? Scored pairwise, because
  several candidates can be true ancestors at once.

Isotonic maps fitted on calibration companies turn q (direct) and the raw
ancestor score into calibrated probabilities. Before calibration, q is used
as is and the logistic function is applied to the ancestor score.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from orgchart.inference.features import EMPLOYEE_LEVEL_FEATURES, FEATURE_LABELS, FEATURES

MODEL_FORMAT = 1

RULE_DIRECT: dict[str, float] = {
    "ic_gap1": 1.0, "ic_gap2": 1.0, "ic_gap3": 0.8, "ic_gap4p": -1.0,
    "mg_gap1": 1.0, "mg_gap2": 0.8, "mg_gap3p": -0.3,
    "log_between_dept": -1.2, "log_between_strict": -2.0,
    "dept_match": 1.5, "dept_mismatch": -2.5,
    "e_top_x_m_exec": 2.5, "e_top_x_dept_mismatch": 1.0, "m_ceo": 1.0,
    "spec_match": 1.0, "spec_mismatch": -1.5,
    "scope_match": 1.0, "scope_mismatch": -2.5,
    "same_city": 0.8, "diff_city": -0.5, "diff_country": -1.0,
    "log_equiv": -1.2,
    "m_mgmt_conf_low": -0.5, "e_level_conf_low": -0.5,
    "start_before": 0.2, "start_after": -0.2,
    "e_support_x_m_exec": 2.0,
}
RULE_DIRECT_BIAS = -2.0

RULE_ANCESTOR: dict[str, float] = {
    "dept_match": 2.5, "dept_mismatch": -3.0,
    "e_top_x_m_exec": 3.0, "e_top_x_dept_mismatch": 2.0, "m_exec": 1.0,
    "scope_match": 1.0, "scope_mismatch": -3.0,
    "spec_match": 0.5, "spec_mismatch": -0.8,
    "diff_country": -0.5,
    "ic_gap4p": 0.5, "mg_gap3p": 0.5,
    "m_mgmt_conf_low": -0.5, "e_level_conf_low": -0.5,
}
RULE_ANCESTOR_BIAS = 0.5


@dataclass
class LinearHead:
    bias: float
    weights: dict[str, float]

    def vector(self) -> np.ndarray:
        return np.array([self.weights.get(f, 0.0) for f in FEATURES], dtype=np.float64)

    def score(self, pairs: pd.DataFrame) -> np.ndarray:
        if len(pairs) == 0:
            return np.zeros(0)
        return self.bias + pairs[list(FEATURES)].to_numpy(np.float64) @ self.vector()

    def contributions(self, row: pd.Series | dict, top: int = 5) -> list[tuple[str, float]]:
        """Largest absolute per-feature contributions for one pair, as (label, contribution)."""
        items = [(FEATURE_LABELS[f], self.weights.get(f, 0.0) * float(row[f])) for f in FEATURES
                 if f not in EMPLOYEE_LEVEL_FEATURES]
        items = [(label, c) for label, c in items if abs(c) > 1e-9]
        return sorted(items, key=lambda x: -abs(x[1]))[:top]


@dataclass
class Calibration:
    """Monotone map raw score -> probability (isotonic fit, linear interpolation between knots)."""

    x: list[float] = field(default_factory=list)
    y: list[float] = field(default_factory=list)

    def apply(self, raw: np.ndarray, identity: bool = False) -> np.ndarray:
        raw = np.asarray(raw, dtype=np.float64)
        if not self.x:
            return raw if identity else 1.0 / (1.0 + np.exp(-raw))
        return np.interp(raw, self.x, self.y)


@dataclass
class Thresholds:
    direct: float = 0.9
    ambiguous: float = 0.9
    ancestor: float = 0.9
    max_ambiguous: int = 3


@dataclass
class Model:
    variant: str
    direct: LinearHead
    ancestor: LinearHead
    direct_calibration: Calibration = field(default_factory=Calibration)
    ancestor_calibration: Calibration = field(default_factory=Calibration)
    thresholds: Thresholds = field(default_factory=Thresholds)
    metadata: dict = field(default_factory=dict)

    def score_pairs(self, pairs: pd.DataFrame) -> pd.DataFrame:
        """Add raw scores, listwise q_direct, calibrated p_direct / p_ancestor and per-employee rank (0 = best)."""
        out = pairs.copy()
        out["score_direct"] = self.direct.score(pairs)
        out["score_ancestor"] = self.ancestor.score(pairs)
        out = out.sort_values(["employee_id", "score_direct", "candidate_id"], ascending=[True, False, True],
                              kind="mergesort").reset_index(drop=True)
        if len(out):
            s = out["score_direct"].to_numpy()
            gmax = np.maximum(out.groupby("employee_id", sort=False)["score_direct"].transform("max").to_numpy(), 0.0)
            e = np.exp(s - gmax)
            denom = np.exp(-gmax) + pd.Series(e).groupby(out["employee_id"].to_numpy(), sort=False).transform("sum").to_numpy()
            out["q_direct"] = e / denom
        else:
            out["q_direct"] = np.zeros(0)
        out["p_direct"] = self.direct_calibration.apply(out["q_direct"].to_numpy(), identity=True)
        out["p_ancestor"] = self.ancestor_calibration.apply(out["score_ancestor"].to_numpy())
        out["rank"] = out.groupby("employee_id", sort=False).cumcount()
        return out

    def to_json(self) -> dict:
        return {
            "format": MODEL_FORMAT,
            "variant": self.variant,
            "features": list(FEATURES),
            "direct": {"bias": self.direct.bias, "weights": self.direct.weights,
                       "calibration": {"x": self.direct_calibration.x, "y": self.direct_calibration.y}},
            "ancestor": {"bias": self.ancestor.bias, "weights": self.ancestor.weights,
                         "calibration": {"x": self.ancestor_calibration.x, "y": self.ancestor_calibration.y}},
            "thresholds": vars(self.thresholds),
            "metadata": self.metadata,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=2) + "\n")

    @classmethod
    def from_json(cls, data: dict) -> "Model":
        if data.get("format") != MODEL_FORMAT or data.get("features") != list(FEATURES):
            raise ValueError("model file was built for a different feature set; re-run `orgchart fit`")
        return cls(
            variant=data["variant"],
            direct=LinearHead(data["direct"]["bias"], data["direct"]["weights"]),
            ancestor=LinearHead(data["ancestor"]["bias"], data["ancestor"]["weights"]),
            direct_calibration=Calibration(**data["direct"]["calibration"]),
            ancestor_calibration=Calibration(**data["ancestor"]["calibration"]),
            thresholds=Thresholds(**data["thresholds"]),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def load(cls, path: Path) -> "Model":
        return cls.from_json(json.loads(path.read_text()))


def rules_model() -> Model:
    """The deterministic baseline before calibration."""
    return Model("rules", LinearHead(RULE_DIRECT_BIAS, dict(RULE_DIRECT)),
                 LinearHead(RULE_ANCESTOR_BIAS, dict(RULE_ANCESTOR)))
