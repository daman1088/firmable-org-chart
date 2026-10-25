"""Evaluate predicted reporting lines against ground truth.

Scoring rules (from docs/architecture.md):

* DIRECT is correct when `reports_to_id` is the true direct manager. Naming a
  skip-level manager counts as an error; it is tracked separately as
  "skip-level".
* ANCESTOR is correct when `ancestor_id` is a true ancestor.
* AMBIGUOUS is correct when the true manager is in `candidate_ids`.
* UNRESOLVED makes no claim.

Definitions:

* **Coverage** = predictions of a status / observed employees (board
  profiles excluded).
* **Precision** = correct / predictions of that status.
* **DIRECT recall** = correct DIRECT / *resolvable* employees, i.e. those whose
  true manager is observed (the best achievable). Recall against all
  employees with a manager is reported as well.

Headline numbers are on test companies. Thresholds and model selection were
fixed on calibration companies (see orgchart.learning.fit) before this runs.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from orgchart.evaluation.labels import employee_truth, true_ancestors
from orgchart.inference.data import load_people
from orgchart.inference.pipeline import predict_people
from orgchart.inference.scoring import Model, Thresholds
from orgchart.inference.validate import validate_predictions
from orgchart.paths import DatasetPaths
from orgchart.schemas import SPLITS

VARIANTS = ("naive", "rules", "learned", "selected")
SIZE_BANDS = [(0, 199, "100-199"), (200, 499, "200-499"), (500, 999, "500-999"), (1000, 10**9, "1000+")]


def _pct(x: float | None) -> float | None:
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(100.0 * float(x), 2)


def _rate(num: float, den: float) -> float | None:
    return None if den == 0 else num / den


def label_predictions(pred: pd.DataFrame, truth: pd.DataFrame, ancestors: pd.DataFrame) -> pd.DataFrame:
    """Join predictions with truth and mark each claim correct or not."""
    df = pred.merge(truth, on=["employee_id", "company_id"], how="left")
    anc = ancestors.rename(columns={"ancestor_id": "_aid"})
    hops = anc.set_index(["employee_id", "_aid"])["hops"]

    def lookup(col: str) -> pd.Series:
        keys = pd.MultiIndex.from_arrays([df["employee_id"], df[col].fillna("")])
        return pd.Series(hops.reindex(keys).to_numpy(), index=df.index)

    df["direct_correct"] = (df["status"] == "DIRECT") & (df["reports_to_id"] == df["true_manager_id"])
    df["direct_hops"] = lookup("reports_to_id")
    df["ancestor_hops"] = lookup("ancestor_id")
    df["ancestor_correct"] = (df["status"] == "ANCESTOR") & df["ancestor_hops"].notna()
    contains = np.array([t in c for t, c in zip(df["true_manager_id"], df["candidate_ids"], strict=True)], dtype=bool)
    df["ambiguous_correct"] = (df["status"] == "AMBIGUOUS").to_numpy() & contains
    df["top1_correct"] = df["top_candidate_id"].notna() & (df["top_candidate_id"] == df["true_manager_id"])
    df["claim_correct"] = df["direct_correct"] | df["ancestor_correct"] | df["ambiguous_correct"]
    df["is_board"] = df["role_type"] == "board"
    return df


def status_metrics(df: pd.DataFrame) -> dict[str, Any]:
    emp = df[~df["is_board"]]
    n = len(emp)
    resolvable = int(emp["manager_observed"].sum())
    with_manager = int(emp["true_manager_id"].notna().sum())
    d = df[df["status"] == "DIRECT"]
    a = df[df["status"] == "ANCESTOR"]
    m = df[df["status"] == "AMBIGUOUS"]
    resolved = df[df["status"] != "UNRESOLVED"]
    skip = d[~d["direct_correct"] & d["direct_hops"].notna()]
    board = df[df["is_board"]]
    return {
        "employees": n,
        "resolvable_employees": resolvable,
        "status_share_pct": {s: _pct(_rate(int((emp["status"] == s).sum()), n))
                             for s in ("DIRECT", "AMBIGUOUS", "ANCESTOR", "UNRESOLVED")},
        "direct": {
            "predictions": len(d), "coverage_pct": _pct(_rate(len(d), n)),
            "precision_pct": _pct(_rate(int(d["direct_correct"].sum()), len(d))),
            "recall_resolvable_pct": _pct(_rate(int(d["direct_correct"].sum()), resolvable)),
            "recall_all_pct": _pct(_rate(int(d["direct_correct"].sum()), with_manager)),
            "skip_level_errors": len(skip),
            "errors_on_top_of_org": int((d["true_manager_id"].isna()).sum()),
            "mean_confidence": None if d["confidence"].isna().all() else round(float(d["confidence"].mean()), 4),
        },
        "ambiguous": {
            "predictions": len(m), "coverage_pct": _pct(_rate(len(m), n)),
            "containment_precision_pct": _pct(_rate(int(m["ambiguous_correct"].sum()), len(m))),
            "mean_set_size": round(float(m["candidate_ids"].map(len).mean()), 2) if len(m) else None,
        },
        "ancestor": {
            "predictions": len(a), "coverage_pct": _pct(_rate(len(a), n)),
            "precision_pct": _pct(_rate(int(a["ancestor_correct"].sum()), len(a))),
            "is_actually_direct_manager_pct": _pct(_rate(int((a["ancestor_hops"] == 1).sum()), len(a))),
            "median_hops_when_correct": None if not a["ancestor_correct"].any()
            else float(a.loc[a["ancestor_correct"], "ancestor_hops"].median()),
        },
        "resolved": {
            "coverage_pct": _pct(_rate(int((emp["status"] != "UNRESOLVED").sum()), n)),
            "precision_pct": _pct(_rate(int(resolved["claim_correct"].sum()), len(resolved))),
        },
        "forced_top1_accuracy_pct": _pct(_rate(int(emp["top1_correct"].sum()), int(emp["top_candidate_id"].notna().sum()))),
        "board_profiles_left_unassigned_pct": _pct(_rate(int((board["status"] == "UNRESOLVED").sum()), len(board))),
    }


def _reliability(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> dict[str, Any]:
    if len(conf) == 0:
        return {"n": 0, "ece_pct": None, "brier": None, "bins": []}
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    rows, ece = [], 0.0
    for b in range(bins):
        sel = idx == b
        if not sel.any():
            continue
        acc, mean_conf = float(correct[sel].mean()), float(conf[sel].mean())
        ece += sel.sum() / len(conf) * abs(acc - mean_conf)
        rows.append({"bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}", "n": int(sel.sum()),
                     "mean_confidence": round(mean_conf, 4), "observed_accuracy": round(acc, 4)})
    return {"n": len(conf), "ece_pct": _pct(ece), "brier": round(float(np.mean((conf - correct) ** 2)), 4), "bins": rows}


def calibration_report(df: pd.DataFrame) -> dict[str, Any]:
    top = df[df["direct_confidence"].notna() & ~df["is_board"]]
    anc = df[df["status"] == "ANCESTOR"]
    return {
        "top_candidate_direct_probability": _reliability(top["direct_confidence"].to_numpy(float),
                                                         top["top1_correct"].to_numpy(float)),
        "direct_predictions": _reliability(df.loc[df["status"] == "DIRECT", "confidence"].to_numpy(float),
                                           df.loc[df["status"] == "DIRECT", "direct_correct"].to_numpy(float)),
        "ancestor_predictions": _reliability(anc["confidence"].to_numpy(float), anc["ancestor_correct"].to_numpy(float)),
    }


def precision_coverage_curve(df: pd.DataFrame, thresholds=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95)
                             ) -> list[dict]:
    emp = df[~df["is_board"]]
    out = []
    for t in thresholds:
        sel = emp["direct_confidence"].fillna(-1) >= t
        out.append({"threshold": t, "coverage_pct": _pct(_rate(int(sel.sum()), len(emp))),
                    "precision_pct": _pct(_rate(int(emp.loc[sel, "top1_correct"].sum()), int(sel.sum())))})
    return out


def _band(level: float) -> str:
    if pd.isna(level):
        return "board"
    for lo, hi, name in ((0, 3, "IC (L0-3)"), (4, 4, "lead (L4)"), (5, 6, "manager (L5-6)"), (7, 8, "director (L7-8)"),
                         (9, 12, "VP+ (L9-12)")):
        if lo <= level <= hi:
            return name
    return "other"


def _size(n: int) -> str:
    return next(name for lo, hi, name in SIZE_BANDS if lo <= n <= hi)


def breakdowns(df: pd.DataFrame) -> dict[str, list[dict]]:
    emp = df[~df["is_board"]].copy()
    emp["seniority"] = emp["true_level"].map(_band)
    emp["company_size"] = emp["observed_employee_count"].map(_size)
    emp["manager_is_observed"] = emp["manager_observed"].map({True: "observed", False: "not observed"})
    out = {}
    for key in ("archetype", "company_size", "seniority", "scope_kind", "manager_is_observed"):
        rows = []
        for value, g in emp.groupby(key, sort=True):
            m = status_metrics(g)
            rows.append({key: value, "employees": m["employees"], "direct_coverage_pct": m["direct"]["coverage_pct"],
                         "direct_precision_pct": m["direct"]["precision_pct"],
                         "ancestor_coverage_pct": m["ancestor"]["coverage_pct"],
                         "ancestor_precision_pct": m["ancestor"]["precision_pct"],
                         "resolved_coverage_pct": m["resolved"]["coverage_pct"],
                         "resolved_precision_pct": m["resolved"]["precision_pct"],
                         "forced_top1_accuracy_pct": m["forced_top1_accuracy_pct"]})
        out[key] = rows
    return out


def error_analysis(df: pd.DataFrame, people: pd.DataFrame, limit: int = 12) -> dict[str, Any]:
    title = people.set_index("employee_id")["title_raw"]
    d = df[(df["status"] == "DIRECT") & ~df["direct_correct"]].copy()
    d["kind"] = np.where(d["true_manager_id"].isna(), "employee has no manager (top of org / board)",
                         np.where(d["direct_hops"].notna(), "skip-level (named a higher ancestor)",
                                  np.where(~d["manager_observed"], "true manager not observed; picked another person",
                                           "wrong branch (sibling / other team)")))
    kinds = d["kind"].value_counts().to_dict()
    sample = d.sort_values("confidence", ascending=False).head(limit)
    examples = [{"employee": title.get(r.employee_id), "predicted": title.get(r.reports_to_id),
                 "true_manager": title.get(r.true_manager_id) if isinstance(r.true_manager_id, str) else None,
                 "confidence": r.confidence, "kind": r.kind} for r in sample.itertuples()]
    reasons = df[df["status"] == "UNRESOLVED"]["reason_codes"].explode().value_counts().to_dict()
    return {"direct_error_kinds": kinds, "highest_confidence_direct_errors": examples, "unresolved_reasons": reasons}


def evaluate_inference(root: str | Path) -> dict[str, Any]:
    paths = DatasetPaths.of(root)
    started = time.perf_counter()
    truth = employee_truth(root)
    ancestors = true_ancestors(root)
    people = load_people(root)
    selected = Model.load(paths.model("selected"))
    report: dict[str, Any] = {"selected_variant": selected.variant, "thresholds": vars(selected.thresholds),
                              "selection": selected.metadata.get("selection"), "variants": {}, "invariants": {}}
    labelled: dict[str, pd.DataFrame] = {}
    for variant in VARIANTS:
        path = paths.predictions(variant)
        if not path.exists():
            continue
        pred = pd.read_parquet(path)
        report["invariants"][variant] = {c["name"]: c["passed"] for c in validate_predictions(pred, people)}
        df = label_predictions(pred, truth, ancestors)
        labelled[variant] = df
        report["variants"][variant] = {split: status_metrics(df[df["split"] == split]) for split in SPLITS}
        report["variants"][variant]["all"] = status_metrics(df)

    test = labelled["selected"][labelled["selected"]["split"] == "test"]
    report["test_calibration"] = calibration_report(test)
    report["test_precision_coverage"] = {v: precision_coverage_curve(labelled[v][labelled[v]["split"] == "test"])
                                         for v in ("rules", "learned") if v in labelled}
    report["test_breakdowns"] = breakdowns(test)
    report["test_errors"] = error_analysis(test, people)

    # Alternative operating points: thresholds chosen on calibration for lower precision targets.
    test_people = people[people["split"] == "test"]
    report["test_operating_points"] = {}
    for target, th in selected.metadata.get("operating_points", {}).items():
        model = Model.from_json(selected.to_json())
        model.thresholds = Thresholds(**th)
        pred = predict_people(test_people, model)
        m = status_metrics(label_predictions(pred, truth, ancestors))
        report["test_operating_points"][target] = {"thresholds": th, "direct": m["direct"], "ambiguous": m["ambiguous"],
                                                   "ancestor": m["ancestor"], "resolved": m["resolved"]}
    report["seconds"] = round(time.perf_counter() - started, 1)
    paths.reports_dir.mkdir(parents=True, exist_ok=True)
    (paths.reports_dir / "inference_eval.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    (paths.reports_dir / "inference_eval.md").write_text(render_markdown(report))
    return report


def _table(rows: list[list[Any]], header: list[str]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join("" if v is None else str(v).replace("|", "\\|") for v in r) + " |" for r in rows]
    return out


def render_markdown(r: dict[str, Any]) -> str:
    sel = r["variants"]["selected"]
    lines = ["# Reporting-line inference evaluation", "",
             f"Selected model: **{r['selected_variant']}**. Thresholds were chosen on calibration companies "
             f"(direct {r['thresholds']['direct']}, ambiguous {r['thresholds']['ambiguous']}, ancestor "
             f"{r['thresholds']['ancestor']}). Test companies are scored once, here.", "",
             "How to read the statuses. They make different claims and are scored separately:", "",
             "* **DIRECT** names the exact manager. It is correct only for the exact true manager; a skip-level "
             "answer is wrong.",
             "* **ANCESTOR** names someone somewhere above the employee. It is correct if that person is anywhere in "
             "the true chain. It is **not** a direct-manager claim, so its precision is not manager accuracy.",
             "* **AMBIGUOUS** names 2-3 candidates. It is correct if the true manager is among them.",
             "* **Any claim (mixed)** pools DIRECT, AMBIGUOUS and ANCESTOR (the `resolved` key in the JSON). It is "
             "not a direct-manager accuracy.", ""]
    if r.get("selection"):
        lines += [f"Selection rule: {r['selection']['rule']}. Learned minus rules on calibration: any-claim coverage "
                  f"{r['selection']['learned_minus_rules_resolved_coverage_pp']} pp, DIRECT coverage "
                  f"{r['selection']['learned_minus_rules_direct_coverage_pp']} pp.", ""]
    metrics = [
        ("employees", lambda m: m["employees"]),
        ("DIRECT coverage %", lambda m: m["direct"]["coverage_pct"]),
        ("DIRECT precision %", lambda m: m["direct"]["precision_pct"]),
        ("DIRECT recall (resolvable) %", lambda m: m["direct"]["recall_resolvable_pct"]),
        ("AMBIGUOUS coverage %", lambda m: m["ambiguous"]["coverage_pct"]),
        ("AMBIGUOUS containment %", lambda m: m["ambiguous"]["containment_precision_pct"]),
        ("ANCESTOR coverage %", lambda m: m["ancestor"]["coverage_pct"]),
        ("ANCESTOR precision %", lambda m: m["ancestor"]["precision_pct"]),
        ("any claim (mixed) coverage %", lambda m: m["resolved"]["coverage_pct"]),
        ("any claim (mixed) precision %", lambda m: m["resolved"]["precision_pct"]),
        ("forced top-1 accuracy %", lambda m: m["forced_top1_accuracy_pct"]),
    ]
    lines += ["## Selected model by split", ""]
    lines += _table([[n, *(fn(sel[s]) for s in (*SPLITS, "all"))] for n, fn in metrics],
                    ["metric", *SPLITS, "all"])
    lines += ["", "## Baseline comparison (test companies)", ""]
    variants = [v for v in VARIANTS if v in r["variants"]]
    lines += _table([[n, *(fn(r["variants"][v]["test"]) for v in variants)] for n, fn in metrics], ["metric", *variants])
    lines += ["", "`naive` always predicts DIRECT (nearest more senior manager in the same department); its precision "
              "equals what forcing every prediction achieves.", ""]
    lines += ["## Operating points (selected model, test)", "",
              "Thresholds chosen on calibration companies for each precision target, then applied once to test.", ""]
    lines += _table([[t, op["thresholds"]["direct"], op["direct"]["coverage_pct"], op["direct"]["precision_pct"],
                      op["ambiguous"]["coverage_pct"], op["ambiguous"]["containment_precision_pct"],
                      op["ancestor"]["coverage_pct"], op["ancestor"]["precision_pct"], op["resolved"]["coverage_pct"],
                      op["resolved"]["precision_pct"]] for t, op in r["test_operating_points"].items()],
                    ["target", "direct thr", "DIRECT cov %", "DIRECT prec %", "AMB cov %", "AMB contain %",
                     "ANC cov %", "ANC prec %", "any claim cov %", "any claim prec %"])
    lines += ["", "## DIRECT precision vs coverage (test, top candidate per employee)", ""]
    curves = r["test_precision_coverage"]
    lines += _table([[pt["threshold"], *(f"{c[i]['coverage_pct']} / {c[i]['precision_pct']}" for c in curves.values())]
                     for i, pt in enumerate(next(iter(curves.values())))],
                    ["p threshold", *(f"{v}: coverage / precision %" for v in curves)])
    cal = r["test_calibration"]
    lines += ["", "## Calibration (test)", ""]
    lines += _table([[k, v["n"], v["ece_pct"], v["brier"]] for k, v in cal.items()], ["quantity", "n", "ECE %", "Brier"])
    lines += ["", "Reliability of the top candidate's direct probability:", ""]
    lines += _table([[b["bin"], b["n"], b["mean_confidence"], b["observed_accuracy"]]
                     for b in cal["top_candidate_direct_probability"]["bins"]],
                    ["confidence bin", "n", "mean confidence", "observed accuracy"])
    lines += ["", "## Breakdowns (selected model, test)", ""]
    for key, rows in r["test_breakdowns"].items():
        lines += [f"### by {key.replace('_', ' ')}", ""]
        lines += _table([[row[key], row["employees"], row["direct_coverage_pct"], row["direct_precision_pct"],
                          row["ancestor_coverage_pct"], row["ancestor_precision_pct"], row["resolved_coverage_pct"],
                          row["resolved_precision_pct"], row["forced_top1_accuracy_pct"]] for row in rows],
                        [key, "employees", "DIRECT cov %", "DIRECT prec %", "ANC cov %", "ANC prec %",
                         "any claim cov %", "any claim prec %", "forced top-1 %"])
        lines.append("")
    err = r["test_errors"]
    lines += ["## Errors (selected model, test)", "", "DIRECT error kinds:", ""]
    lines += _table([[k, v] for k, v in err["direct_error_kinds"].items()], ["kind", "count"])
    lines += ["", "Highest-confidence DIRECT errors:", ""]
    lines += _table([[e["employee"], e["predicted"], e["true_manager"], e["confidence"], e["kind"]]
                     for e in err["highest_confidence_direct_errors"]],
                    ["employee title", "predicted manager", "true manager", "confidence", "kind"])
    lines += ["", "UNRESOLVED reasons:", ""]
    lines += _table([[k, v] for k, v in err["unresolved_reasons"].items()], ["reason", "count"])
    lines += ["", "## Structural invariants", ""]
    names = list(next(iter(r["invariants"].values())))
    lines += _table([[n, *("PASS" if r["invariants"][v][n] else "FAIL" for v in r["invariants"])] for n in names],
                    ["check", *r["invariants"]])
    return "\n".join(lines) + "\n"
