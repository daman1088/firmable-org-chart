"""Evaluate title normalisation against ground truth.

Joins derived/normalized_titles.parquet with the truth tables and reports, per
company split (train / calibration / test):

* seniority: coverage, exact accuracy, within-one accuracy, MAE
* management authority: accuracy, manager precision/recall, and the false-manager
  rate for technical-seniority ICs (Staff / Principal / Lead, level 4)
* department: coverage and accuracy, with and without a department field
* specialty and scope: precision when predicted, recall where the title shows it
* non-employee (board/advisor) detection
* breakdowns by industry and by observation-noise type, confidence calibration,
  unresolved reasons and the most frequent unresolved titles

Rules were developed on train companies; calibration and test are the held-out
numbers, and test additionally contains title styles never seen in train.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from orgchart.paths import DatasetPaths
from orgchart.schemas import SPLITS
from orgchart.storage.truth import connect_with_truth

# Truth department keys that the normaliser names differently.
TRUTH_TO_NORMALIZED_DEPT = {"mfg_engineering": "hardware_engineering", "store_operations": "retail_operations"}

# Noise that makes the true level unrecoverable from the title itself.
TITLE_UNRECOVERABLE_TAGS = ("stale_title", "inflated_title", "vague_title", "empty_title")


def _connect(root: str | Path):
    paths = DatasetPaths.of(root)
    con = connect_with_truth(root)
    mapping = " ".join(f"WHEN '{k}' THEN '{v}'" for k, v in TRUTH_TO_NORMALIZED_DEPT.items())
    con.execute(f"CREATE VIEW normalized AS SELECT * FROM read_parquet('{paths.normalized_titles}')")
    con.execute(f"""
        CREATE VIEW ev AS
        SELECT n.*, c.split, tc.archetype,
               t.level AS t_level, t.role_type AS t_role, t.is_people_manager AS t_manager,
               CASE t.department {mapping} ELSE t.department END AS t_department,
               t.specialty AS t_specialty, t.scope AS t_scope, t.employment_type AS t_employment,
               t.noise_tags,
               len(list_intersect(t.noise_tags, {list(TITLE_UNRECOVERABLE_TAGS)})) = 0 AS title_recoverable
        FROM normalized n
        JOIN truth_employees t USING (employee_id)
        JOIN companies c ON c.company_id = n.company_id
        JOIN truth_companies tc ON tc.company_id = n.company_id""")
    return con


def _pct(x: Any) -> float | None:
    return None if x is None else round(100.0 * float(x), 2)


def _split_metrics(con, where: str) -> dict[str, Any]:
    emp = f"FROM ev WHERE t_role <> 'board' AND {where}"
    q = lambda sql: con.execute(sql).fetchone()  # noqa: E731
    m: dict[str, Any] = {}
    n, cov, exact, within1, mae, exact_all = q(f"""
        SELECT count(*), avg((level IS NOT NULL)::INT),
               avg((level = t_level)::INT) FILTER (WHERE level IS NOT NULL),
               avg((abs(level - t_level) <= 1)::INT) FILTER (WHERE level IS NOT NULL),
               avg(abs(level - t_level)) FILTER (WHERE level IS NOT NULL),
               avg(coalesce(level = t_level, false)::INT) {emp}""")
    rec_exact, rec_within1 = q(f"""
        SELECT avg((level = t_level)::INT), avg((abs(level - t_level) <= 1)::INT)
        {emp} AND level IS NOT NULL AND title_recoverable""")
    ceiling, ceiling_dept = q(f"""
        WITH lv AS (SELECT title_clean, t_level, count(*) AS n {emp} AND title_clean IS NOT NULL GROUP BY 1, 2),
             lm AS (SELECT max(n) AS best, sum(n) AS total FROM lv GROUP BY title_clean),
             dv AS (SELECT title_clean, department_raw, t_department, count(*) AS n {emp} AND title_clean IS NOT NULL
                    GROUP BY 1, 2, 3),
             dm AS (SELECT max(n) AS best, sum(n) AS total FROM dv GROUP BY title_clean, department_raw)
        SELECT (SELECT sum(best) / sum(total) FROM lm), (SELECT sum(best) / sum(total) FROM dm)""")
    m["employees"] = n
    m["seniority"] = {
        "title_only_ceiling_pct": _pct(ceiling),
        "coverage_pct": _pct(cov), "exact_pct": _pct(exact), "within_one_pct": _pct(within1),
        "mae_levels": None if mae is None else round(mae, 3), "exact_incl_unresolved_pct": _pct(exact_all),
        "exact_title_recoverable_pct": _pct(rec_exact), "within_one_title_recoverable_pct": _pct(rec_within1),
    }
    acc, mgr_p, mgr_r, ic_p, cov_m = q(f"""
        SELECT avg((manages_people = t_manager)::INT) FILTER (WHERE manages_people IS NOT NULL),
               avg(t_manager::INT) FILTER (WHERE manages_people),
               avg(coalesce(manages_people, false)::INT) FILTER (WHERE t_manager),
               avg((NOT t_manager)::INT) FILTER (WHERE manages_people = false),
               avg((manages_people IS NOT NULL)::INT) {emp}""")
    staff_n, staff_false_mgr = q(f"""
        SELECT count(*), avg(coalesce(manages_people, false)::INT) {emp} AND t_role = 'ic' AND t_level = 4""")
    tech_n, tech_false_mgr = q(f"""
        SELECT count(*), avg(coalesce(manages_people, false)::INT) {emp} AND NOT t_manager
          AND regexp_matches(lower(coalesce(title_raw, '')), '\\b(staff|principal|lead) [a-z]*\\s*(engineer|developer|scientist|consultant|accountant|designer|analyst)')""")
    m["management"] = {
        "coverage_pct": _pct(cov_m), "accuracy_pct": _pct(acc), "manager_precision_pct": _pct(mgr_p),
        "manager_recall_pct": _pct(mgr_r), "ic_precision_pct": _pct(ic_p),
        "level4_ic_count": staff_n, "level4_ic_false_manager_pct": _pct(staff_false_mgr),
        "staff_principal_title_count": tech_n, "staff_principal_title_false_manager_pct": _pct(tech_false_mgr),
    }
    dcov, dacc, dacc_nofield, dacc_field = q(f"""
        SELECT avg((department IS NOT NULL)::INT),
               avg((department = t_department)::INT) FILTER (WHERE department IS NOT NULL),
               avg((department = t_department)::INT) FILTER (WHERE department IS NOT NULL AND department_raw IS NULL),
               avg((department = t_department)::INT) FILTER (WHERE department IS NOT NULL AND department_raw IS NOT NULL)
        {emp}""")
    m["department"] = {"coverage_pct": _pct(dcov), "accuracy_pct": _pct(dacc), "title_and_field_ceiling_pct": _pct(ceiling_dept),
                       "accuracy_without_field_pct": _pct(dacc_nofield), "accuracy_with_field_pct": _pct(dacc_field)}
    s_cov, s_prec, s_vis = q(f"""
        SELECT avg((specialty IS NOT NULL)::INT) FILTER (WHERE t_specialty IS NOT NULL),
               avg((lower(specialty) = lower(t_specialty))::INT) FILTER (WHERE specialty IS NOT NULL),
               avg(coalesce(lower(specialty) = lower(t_specialty), false)::INT)
                 FILTER (WHERE t_specialty IS NOT NULL AND contains(lower(coalesce(title_raw, '')), lower(t_specialty)))
        {emp}""")
    m["specialty"] = {"coverage_pct": _pct(s_cov), "precision_pct": _pct(s_prec),
                      "recall_where_in_title_pct": _pct(s_vis)}
    sc_prec, sc_vis, sc_n = q(f"""
        SELECT avg((lower(scope) = lower(t_scope))::INT) FILTER (WHERE scope IS NOT NULL),
               avg(coalesce(lower(scope) = lower(t_scope), false)::INT)
                 FILTER (WHERE t_scope IS NOT NULL AND contains(lower(coalesce(title_raw, '')), lower(t_scope))),
               count(*) FILTER (WHERE scope IS NOT NULL)
        {emp}""")
    m["scope"] = {"predicted": sc_n, "precision_pct": _pct(sc_prec), "recall_where_in_title_pct": _pct(sc_vis)}
    b_p, b_r = q(f"""
        SELECT avg((t_role = 'board')::INT) FILTER (WHERE is_non_employee),
               avg(is_non_employee::INT) FILTER (WHERE t_role = 'board') FROM ev WHERE {where}""")
    c_p, i_p, i_r = q(f"""
        SELECT avg((t_employment = 'contractor')::INT) FILTER (WHERE is_contractor),
               avg((t_level = 0)::INT) FILTER (WHERE is_intern),
               avg(is_intern::INT) FILTER (WHERE t_level = 0) {emp}""")
    m["flags"] = {"non_employee_precision_pct": _pct(b_p), "non_employee_recall_pct": _pct(b_r),
                  "contractor_precision_pct": _pct(c_p), "intern_precision_pct": _pct(i_p),
                  "intern_recall_pct": _pct(i_r)}
    return m


def _breakdowns(con, where: str) -> dict[str, Any]:
    emp = f"FROM ev WHERE t_role <> 'board' AND {where}"
    out: dict[str, Any] = {}
    out["by_industry"] = {
        row[0]: {"employees": row[1], "seniority_exact_pct": _pct(row[2]), "seniority_within_one_pct": _pct(row[3]),
                 "department_accuracy_pct": _pct(row[4]), "management_accuracy_pct": _pct(row[5])}
        for row in con.execute(f"""
            SELECT archetype, count(*),
                   avg((level = t_level)::INT) FILTER (WHERE level IS NOT NULL),
                   avg((abs(level - t_level) <= 1)::INT) FILTER (WHERE level IS NOT NULL),
                   avg((department = t_department)::INT) FILTER (WHERE department IS NOT NULL),
                   avg((manages_people = t_manager)::INT) FILTER (WHERE manages_people IS NOT NULL)
            {emp} GROUP BY 1 ORDER BY 1""").fetchall()
    }
    out["by_noise_tag"] = {
        row[0]: {"employees": row[1], "seniority_exact_pct": _pct(row[2]), "department_accuracy_pct": _pct(row[3]),
                 "level_resolved_pct": _pct(row[4])}
        for row in con.execute(f"""
            SELECT tag, count(*),
                   avg((level = t_level)::INT) FILTER (WHERE level IS NOT NULL),
                   avg((department = t_department)::INT) FILTER (WHERE department IS NOT NULL),
                   avg((level IS NOT NULL)::INT)
            FROM (SELECT *, unnest(CASE WHEN len(noise_tags) = 0 THEN ['(none)'] ELSE noise_tags END) AS tag {emp})
            GROUP BY 1 ORDER BY 2 DESC""").fetchall()
    }
    out["confidence_calibration"] = [
        {"bucket": row[0], "share_pct": _pct(row[1]), "fully_correct_pct": _pct(row[2]),
         "seniority_exact_pct": _pct(row[3])}
        for row in con.execute(f"""
            WITH b AS (SELECT *, CASE WHEN parse_confidence >= 0.85 THEN '0.85-1.00'
                                      WHEN parse_confidence >= 0.70 THEN '0.70-0.85'
                                      WHEN parse_confidence >= 0.50 THEN '0.50-0.70'
                                      WHEN parse_confidence > 0 THEN '0.00-0.50' ELSE 'unresolved (0)' END AS bucket
                       {emp})
            SELECT bucket, count(*) / sum(count(*)) OVER (),
                   avg((coalesce(level = t_level, false) AND coalesce(department = t_department, false)
                        AND coalesce(manages_people = t_manager, false))::INT),
                   avg(coalesce(level = t_level, false)::INT)
            FROM b GROUP BY 1 ORDER BY 1 DESC""").fetchall()
    ]
    out["reason_codes"] = dict(con.execute(f"""
        SELECT code, count(*) FROM (SELECT unnest(reason_codes) AS code {emp}) GROUP BY 1 ORDER BY 2 DESC""").fetchall())
    out["top_unresolved_titles"] = [
        {"title": row[0], "count": row[1], "reasons": row[2]}
        for row in con.execute(f"""
            SELECT title_raw, count(*), any_value(array_to_string(reason_codes, ','))
            {emp} AND (level IS NULL OR department IS NULL) GROUP BY 1 ORDER BY 2 DESC LIMIT 15""").fetchall()
    ]
    out["top_seniority_errors"] = [
        {"title_clean": row[0], "true_level": row[1], "predicted_level": row[2], "count": row[3]}
        for row in con.execute(f"""
            SELECT title_clean, t_level, level, count(*) {emp} AND level IS NOT NULL AND level <> t_level
              AND title_recoverable GROUP BY ALL ORDER BY 4 DESC LIMIT 15""").fetchall()
    ]
    out["top_department_errors"] = [
        {"title_clean": row[0], "department_raw": row[1], "true": row[2], "predicted": row[3], "count": row[4]}
        for row in con.execute(f"""
            SELECT title_clean, department_raw, t_department, department, count(*) {emp}
              AND department IS NOT NULL AND department <> t_department GROUP BY ALL ORDER BY 5 DESC LIMIT 15""").fetchall()
    ]
    return out


def evaluate_normalization(root: str | Path) -> dict[str, Any]:
    con = _connect(root)
    report: dict[str, Any] = {"splits": {}, "held_out_title_styles": {}}
    for split in (*SPLITS, "all"):
        where = "true" if split == "all" else f"split = '{split}'"
        report["splits"][split] = _split_metrics(con, where)
    # Test companies only: rows with title styles never present in train vs. the rest of test.
    for label, cond in (("test_with_holdout_styles", "len(list_filter(noise_tags, lambda x: x LIKE 'holdout:%')) > 0"),
                        ("test_without_holdout_styles", "len(list_filter(noise_tags, lambda x: x LIKE 'holdout:%')) = 0")):
        report["held_out_title_styles"][label] = _split_metrics(con, f"split = 'test' AND {cond}")
    report["test_breakdowns"] = _breakdowns(con, "split = 'test'")
    report["all_breakdowns"] = {k: v for k, v in _breakdowns(con, "true").items()
                                if k in ("reason_codes", "top_unresolved_titles")}
    paths = DatasetPaths.of(root)
    paths.reports_dir.mkdir(parents=True, exist_ok=True)
    (paths.reports_dir / "normalization_eval.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    (paths.reports_dir / "normalization_eval.md").write_text(render_markdown(report))
    return report


def _table(rows: list[list[Any]], header: list[str]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join("" if v is None else str(v) for v in r) + " |" for r in rows]
    return lines


def render_markdown(report: dict[str, Any]) -> str:
    s = report["splits"]
    order = ["train", "calibration", "test", "all"]
    lines = ["# Title normalisation evaluation", "",
             "Rules were developed on **train** companies only; calibration and test are held out.", ""]
    lines += ["## Headline metrics by split", ""]
    rows = []
    metrics = [
        ("employees", lambda m: m["employees"]),
        ("seniority coverage %", lambda m: m["seniority"]["coverage_pct"]),
        ("seniority exact %", lambda m: m["seniority"]["exact_pct"]),
        ("seniority within ±1 %", lambda m: m["seniority"]["within_one_pct"]),
        ("seniority exact % (title-recoverable)", lambda m: m["seniority"]["exact_title_recoverable_pct"]),
        ("seniority title-only ceiling % (oracle)", lambda m: m["seniority"]["title_only_ceiling_pct"]),
        ("seniority MAE (levels)", lambda m: m["seniority"]["mae_levels"]),
        ("management accuracy %", lambda m: m["management"]["accuracy_pct"]),
        ("manager precision %", lambda m: m["management"]["manager_precision_pct"]),
        ("manager recall %", lambda m: m["management"]["manager_recall_pct"]),
        ("Staff/Principal titles flagged manager %", lambda m: m["management"]["staff_principal_title_false_manager_pct"]),
        ("level-4 ICs flagged manager %", lambda m: m["management"]["level4_ic_false_manager_pct"]),
        ("department coverage %", lambda m: m["department"]["coverage_pct"]),
        ("department accuracy %", lambda m: m["department"]["accuracy_pct"]),
        ("department ceiling % (oracle on title + field)", lambda m: m["department"]["title_and_field_ceiling_pct"]),
        ("department accuracy, no dept field %", lambda m: m["department"]["accuracy_without_field_pct"]),
        ("specialty precision %", lambda m: m["specialty"]["precision_pct"]),
        ("specialty recall (shown in title) %", lambda m: m["specialty"]["recall_where_in_title_pct"]),
        ("scope precision %", lambda m: m["scope"]["precision_pct"]),
        ("scope recall (shown in title) %", lambda m: m["scope"]["recall_where_in_title_pct"]),
        ("board/advisor precision %", lambda m: m["flags"]["non_employee_precision_pct"]),
        ("board/advisor recall %", lambda m: m["flags"]["non_employee_recall_pct"]),
    ]
    for name, fn in metrics:
        rows.append([name] + [fn(s[k]) for k in order])
    lines += _table(rows, ["metric", *order])
    h = report["held_out_title_styles"]
    lines += ["", "## Held-out title styles (test companies)", ""]
    lines += _table([[name, fn(h["test_with_holdout_styles"]), fn(h["test_without_holdout_styles"])]
                     for name, fn in metrics[:12]], ["metric", "rows with held-out styles", "other test rows"])
    b = report["test_breakdowns"]
    lines += ["", "## Test split by industry", ""]
    lines += _table([[k, v["employees"], v["seniority_exact_pct"], v["seniority_within_one_pct"],
                      v["department_accuracy_pct"], v["management_accuracy_pct"]] for k, v in b["by_industry"].items()],
                    ["industry", "employees", "seniority exact %", "within ±1 %", "department %", "management %"])
    lines += ["", "## Test split by observation noise", ""]
    lines += _table([[k, v["employees"], v["level_resolved_pct"], v["seniority_exact_pct"], v["department_accuracy_pct"]]
                     for k, v in b["by_noise_tag"].items()],
                    ["noise tag", "employees", "level resolved %", "seniority exact %", "department %"])
    lines += ["", "## Confidence vs correctness (test)", ""]
    lines += _table([[r["bucket"], r["share_pct"], r["fully_correct_pct"], r["seniority_exact_pct"]]
                     for r in b["confidence_calibration"]],
                    ["parse_confidence", "share %", "level+dept+mgmt all correct %", "seniority exact %"])
    lines += ["", "## Unresolved (all splits)", ""]
    a = report["all_breakdowns"]
    lines += _table([[k, v] for k, v in a["reason_codes"].items()], ["reason code", "rows"])
    lines += [""]
    lines += _table([[r["title"], r["count"], r["reasons"]] for r in a["top_unresolved_titles"]],
                    ["unresolved title", "rows", "reasons"])
    lines += ["", "## Most frequent seniority errors (test, title-recoverable rows)", ""]
    lines += _table([[r["title_clean"], r["true_level"], r["predicted_level"], r["count"]] for r in b["top_seniority_errors"]],
                    ["title (clean)", "true", "predicted", "rows"])
    lines += ["", "## Most frequent department errors (test)", ""]
    lines += _table([[r["title_clean"], r["department_raw"], r["true"], r["predicted"], r["count"]]
                     for r in b["top_department_errors"]], ["title (clean)", "dept field", "true", "predicted", "rows"])
    return "\n".join(lines) + "\n"
