"""Fit, calibrate and select reporting-line models.

1. **Learned weights** (train companies only).
   * Direct head: a listwise conditional logit over each employee's
     shortlisted candidates, with an outside option for "manager not observed".
   * Ancestor head: pairwise L2 logistic regression.
2. **Calibration** (calibration companies only). For each variant (`rules`,
   `learned`), two isotonic maps:
   * the top candidate's listwise q to P(correct direct manager);
   * the raw ancestor score to P(true ancestor).
3. **Thresholds** (calibration companies only). The decision-cascade
   thresholds are chosen in order (direct, then ambiguous, then ancestor). Each
   is the lowest threshold whose precision meets `target_precision` at that
   threshold and at every higher one with enough support.
4. **Selection** (calibration companies only). The learned variant is adopted
   only if, at the target precision, it:
   * raises resolved coverage (DIRECT + AMBIGUOUS + ANCESTOR) by at least
     `min_gain_pp` percentage points over the rules baseline; and
   * does not lower DIRECT coverage.

   Otherwise the rules baseline is kept. Thresholds for alternative
   precision targets (`ALT_TARGETS`) are also stored, so the precision /
   coverage trade-off can be reported and explored.

Test companies are never read here.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from orgchart.evaluation.labels import employee_truth, label_pairs, true_ancestors
from orgchart.inference.data import load_people
from orgchart.inference.features import FEATURES, CandidateSettings
from orgchart.inference.pipeline import iter_company_candidates
from orgchart.inference.scoring import Calibration, LinearHead, Model, Thresholds, rules_model
from orgchart.learning.estimators import fit_conditional_logit, fit_isotonic, fit_logistic
from orgchart.paths import DatasetPaths

GRID = tuple(round(x, 3) for x in [*np.arange(0.30, 0.99, 0.01), 0.99, 0.995, 0.999])
NEVER = 1.01  # threshold that disables a status
ALT_TARGETS = (0.90, 0.80)


def _pairs_for(people: pd.DataFrame, split: str, truth: pd.DataFrame, ancestors: pd.DataFrame) -> pd.DataFrame:
    subset = people[people["split"] == split]
    parts = [pairs for _, _, pairs in iter_company_candidates(subset, CandidateSettings())]
    return label_pairs(pd.concat(parts, ignore_index=True), truth, ancestors)


def _head(bias: float, w: np.ndarray) -> LinearHead:
    return LinearHead(round(bias, 4), {f: round(float(v), 4) for f, v in zip(FEATURES, w, strict=True)})


def fit_direct_head(pairs: pd.DataFrame, l2: float) -> LinearHead:
    pairs = pairs.sort_values(["employee_id", "candidate_id"], kind="mergesort")
    emp = pairs["employee_id"].to_numpy()
    starts = np.flatnonzero(np.r_[True, emp[1:] != emp[:-1]])
    return _head(*fit_conditional_logit(pairs[list(FEATURES)].to_numpy(np.float32), pairs["is_direct"].to_numpy(),
                                        starts, l2=l2))


def fit_ancestor_head(pairs: pd.DataFrame, l2: float) -> LinearHead:
    return _head(*fit_logistic(pairs[list(FEATURES)].to_numpy(np.float32), pairs["is_ancestor"].to_numpy(), l2=l2))


def _stable_threshold(values: np.ndarray, correct: np.ndarray, target: float, min_support: int) -> tuple[float, dict]:
    """Lowest grid threshold t such that precision >= target at t and at every higher supported threshold."""
    stats = []
    for t in GRID:
        sel = values >= t
        n = int(sel.sum())
        stats.append((t, n, float(correct[sel].mean()) if n else None))
    chosen = NEVER
    for t, n, prec in reversed(stats):
        if n < min_support:
            continue
        if prec is None or prec < target:
            break
        chosen = t
    chosen = float(chosen)
    sel = values >= chosen
    return chosen, {"threshold": chosen, "predictions": int(sel.sum()),
                    "precision": round(float(correct[sel].mean()), 4) if sel.any() else None}


def select_thresholds(ranked: pd.DataFrame, truth: pd.DataFrame, target: float, max_ambiguous: int = 3,
                      min_support: int = 100) -> tuple[Thresholds, dict]:
    """Choose cascade thresholds on (calibration) ranked pairs that carry is_direct / is_ancestor labels."""
    top = ranked[ranked["rank"] < max_ambiguous]
    wide = {c: top.pivot(index="employee_id", columns="rank", values=c).reindex(columns=range(max_ambiguous))
            for c in ("q_direct", "p_direct", "is_direct")}
    q, p_top = wide["q_direct"].fillna(0.0).to_numpy(), wide["p_direct"].fillna(0.0).to_numpy()[:, 0]
    hit = wide["is_direct"].fillna(False).to_numpy(bool)
    employees = wide["q_direct"].index.to_numpy()
    report: dict = {"employees_with_candidates": len(employees)}

    t_direct, report["direct"] = _stable_threshold(p_top, hit[:, 0], target, min_support)
    rest = p_top < t_direct

    cum = np.cumsum(q, axis=1)
    def amb(t: float) -> tuple[np.ndarray, np.ndarray]:
        size = np.where(cum[:, 1] >= t, 2, np.where(cum[:, max_ambiguous - 1] >= t, max_ambiguous, 0))
        contains = np.array([hit[i, :s].any() if s else False for i, s in enumerate(size)])
        return size > 0, contains
    best_amb, amb_stats = NEVER, []
    for t in reversed(GRID):
        made, contains = amb(t)
        sel = made & rest
        n = int(sel.sum())
        amb_stats.append((t, n, float(contains[sel].mean()) if n else None))
    for t, n, prec in amb_stats:  # descending thresholds
        if n < min_support:
            continue
        if prec is None or prec < target:
            break
        best_amb = float(t)
    made, contains = amb(best_amb) if best_amb <= 1 else (np.zeros(len(q), bool), np.zeros(len(q), bool))
    sel = made & rest
    report["ambiguous"] = {"threshold": best_amb, "predictions": int(sel.sum()),
                           "precision": round(float(contains[sel].mean()), 4) if sel.any() else None}
    rest &= ~made

    remaining = set(employees[rest])
    pool = ranked[ranked["employee_id"].isin(remaining) & ~ranked["m_is_ceo"]]
    best_anc, anc_stats = NEVER, []
    for t in reversed(GRID):
        pick = pool[pool["p_ancestor"] >= t].sort_values(["employee_id", "m_level", "p_ancestor"],
                                                         ascending=[True, True, False]).drop_duplicates("employee_id")
        anc_stats.append((t, len(pick), float(pick["is_ancestor"].mean()) if len(pick) else None))
    for t, n, prec in anc_stats:
        if n < min_support:
            continue
        if prec is None or prec < target:
            break
        best_anc = float(t)
    n_anc, p_anc = next(((n, p) for t, n, p in anc_stats if t == best_anc), (0, None))
    report["ancestor"] = {"threshold": best_anc, "predictions": n_anc,
                          "precision": round(p_anc, 4) if p_anc is not None else None}
    n = max(1, len(employees))
    report["direct_coverage"] = round(report["direct"]["predictions"] / n, 4)
    report["resolved_coverage"] = round(sum(report[k]["predictions"] for k in ("direct", "ambiguous", "ancestor")) / n, 4)
    return Thresholds(direct=t_direct, ambiguous=best_amb, ancestor=best_anc, max_ambiguous=max_ambiguous), report


def calibrate(model: Model, pairs: pd.DataFrame, truth: pd.DataFrame, target: float) -> Model:
    model.direct_calibration, model.ancestor_calibration = Calibration(), Calibration()
    ranked = model.score_pairs(pairs)
    top = ranked[ranked["rank"] == 0]
    model.direct_calibration = Calibration(*fit_isotonic(top["q_direct"].to_numpy(), top["is_direct"].to_numpy()))
    model.ancestor_calibration = Calibration(*fit_isotonic(ranked["score_ancestor"].to_numpy(),
                                                           ranked["is_ancestor"].to_numpy()))
    ranked = model.score_pairs(pairs)
    model.thresholds, report = select_thresholds(ranked, truth, target)
    model.metadata["calibration_split"] = report
    model.metadata["operating_points"] = {f"{target:.2f}": asdict(model.thresholds)}
    for alt in ALT_TARGETS:
        th, _ = select_thresholds(ranked, truth, alt)
        model.metadata["operating_points"][f"{alt:.2f}"] = asdict(th)
    return model


def fit_models(root: str | Path, target_precision: float = 0.95, l2: float = 1.0, min_gain_pp: float = 1.0) -> dict:
    paths = DatasetPaths.of(root)
    started = time.perf_counter()
    people = load_people(root)
    truth = employee_truth(root)
    ancestors = true_ancestors(root)

    train = _pairs_for(people, "train", truth, ancestors)
    learned = Model("learned", fit_direct_head(train, l2), fit_ancestor_head(train, l2))
    learned.metadata["training"] = {"split": "train", "pairs": len(train), "positives_direct": int(train["is_direct"].sum()),
                                    "positives_ancestor": int(train["is_ancestor"].sum()), "l2": l2}
    del train

    cal = _pairs_for(people, "calibration", truth, ancestors)
    cal_truth = truth[truth["split"] == "calibration"]
    results = {}
    for model in (rules_model(), learned):
        model.metadata.update(target_precision=target_precision, settings=asdict(CandidateSettings()))
        calibrate(model, cal, cal_truth, target_precision)
        model.save(paths.model(model.variant))
        results[model.variant] = model

    cal_l, cal_r = (results[v].metadata["calibration_split"] for v in ("learned", "rules"))
    gain = 100 * (cal_l["resolved_coverage"] - cal_r["resolved_coverage"])
    direct_gain = 100 * (cal_l["direct_coverage"] - cal_r["direct_coverage"])
    chosen = results["learned"] if gain >= min_gain_pp and direct_gain >= 0 else results["rules"]
    chosen.metadata["selection"] = {
        "rule": f"adopt learned only if resolved coverage at {target_precision:.0%} precision improves by >= "
                f"{min_gain_pp} pp and DIRECT coverage does not fall (calibration companies)",
        "learned_minus_rules_resolved_coverage_pp": round(gain, 2),
        "learned_minus_rules_direct_coverage_pp": round(direct_gain, 2),
        "selected": chosen.variant,
    }
    chosen.save(paths.model("selected"))
    summary = {
        "selected": chosen.variant,
        "selection": chosen.metadata["selection"],
        "calibration": {v: m.metadata["calibration_split"] for v, m in results.items()},
        "thresholds": {v: asdict(m.thresholds) for v, m in results.items()},
        "seconds": round(time.perf_counter() - started, 1),
    }
    (paths.models_dir / "fit_summary.json").write_text(
        json.dumps({k: v for k, v in summary.items() if k != "seconds"}, indent=2) + "\n")  # deterministic artifact
    return summary
