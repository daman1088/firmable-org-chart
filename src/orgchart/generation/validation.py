"""Dataset validation: hard structural checks and soft realism statistics.

Hard checks must all pass for a dataset to be usable; `orgchart validate` exits
non-zero otherwise. Realism statistics are compared against the configured
`realism_targets` and reported as warnings, not failures.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import duckdb

from orgchart.paths import DatasetPaths
from orgchart.schemas import FORBIDDEN_OBSERVED_FRAGMENTS, SPLITS
from orgchart.storage.parquet_io import sha256_file
from orgchart.storage.truth import connect_with_truth


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


def _scalar(con: duckdb.DuckDBPyConnection, sql: str) -> Any:
    return con.execute(sql).fetchone()[0]


def structural_checks(root: str | Path, manifest: dict) -> list[Check]:
    paths = DatasetPaths.of(root)
    con = connect_with_truth(root)
    cfg = manifest["config"]
    min_obs = cfg["dataset"]["min_observed_employees"]
    checks: list[Check] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append(Check(name, bool(passed), detail))

    n_companies = _scalar(con, "SELECT count(*) FROM companies")
    add("company_count", n_companies == cfg["dataset"]["n_companies"],
        f"{n_companies} companies (config {cfg['dataset']['n_companies']})")

    dup_obs = _scalar(con, "SELECT count(*) - count(DISTINCT employee_id) FROM employees")
    dup_truth = _scalar(con, "SELECT count(*) - count(DISTINCT employee_id) FROM truth_employees")
    add("unique_employee_ids", dup_obs == 0 and dup_truth == 0, f"duplicates observed={dup_obs} truth={dup_truth}")

    orphans = _scalar(con, """
        SELECT count(*) FROM employees o LEFT JOIN truth_employees t USING (employee_id)
        WHERE t.employee_id IS NULL OR NOT t.is_observed OR t.company_id <> o.company_id""")
    flagged = _scalar(con, "SELECT count(*) FROM truth_employees WHERE is_observed")
    n_obs = _scalar(con, "SELECT count(*) FROM employees")
    add("observed_matches_truth_flags", orphans == 0 and flagged == n_obs,
        f"observed rows={n_obs}, truth is_observed={flagged}, mismatches={orphans}")

    min_company = con.execute(f"""
        SELECT c.company_id, count(t.employee_id) AS n
        FROM companies c LEFT JOIN truth_employees t
          ON t.company_id = c.company_id AND t.is_observed AND t.role_type <> 'board'
        GROUP BY 1 ORDER BY n LIMIT 1""").fetchone()
    add("min_observed_employees", min_company[1] >= min_obs,
        f"smallest company {min_company[0]} has {min_company[1]} observed employees (required >= {min_obs})")

    bad_roots = _scalar(con, """
        SELECT count(*) FROM (
          SELECT company_id, count(*) FILTER (WHERE manager_id IS NULL AND role_type <> 'board') AS roots,
                 count(*) FILTER (WHERE manager_id IS NULL AND role_type = 'ceo' AND level = 12) AS ceos
          FROM truth_employees GROUP BY 1) WHERE roots <> 1 OR ceos <> 1""")
    add("single_ceo_root", bad_roots == 0, f"{bad_roots} companies without exactly one CEO root")

    dangling = _scalar(con, """
        SELECT count(*) FROM truth_employees e LEFT JOIN truth_employees m ON e.manager_id = m.employee_id
        WHERE e.manager_id IS NOT NULL AND (m.employee_id IS NULL OR m.company_id <> e.company_id)""")
    add("managers_exist_in_same_company", dangling == 0, f"{dangling} dangling manager references")

    reach = con.execute("""
        WITH RECURSIVE walk(employee_id, depth) AS (
          SELECT employee_id, 0 FROM truth_employees WHERE manager_id IS NULL AND role_type = 'ceo'
          UNION ALL
          SELECT e.employee_id, w.depth + 1 FROM truth_employees e JOIN walk w ON e.manager_id = w.employee_id
        )
        SELECT (SELECT count(*) FROM walk),
               (SELECT count(*) FROM truth_employees WHERE role_type <> 'board'),
               (SELECT count(*) FROM walk w JOIN truth_employees t USING (employee_id) WHERE t.depth <> w.depth)
        """).fetchone()
    add("acyclic_and_connected", reach[0] == reach[1] and reach[2] == 0,
        f"reachable from CEO={reach[0]} of {reach[1]} employees; depth mismatches={reach[2]}")

    inversions = _scalar(con, """
        SELECT count(*) FROM truth_employees e JOIN truth_employees m ON e.manager_id = m.employee_id
        WHERE m.level <= e.level""")
    add("strict_level_decrease", inversions == 0, f"{inversions} edges where manager level <= report level")

    bad_reports = _scalar(con, """
        SELECT count(*) FROM truth_employees e LEFT JOIN (
          SELECT manager_id, count(*) AS n FROM truth_employees WHERE manager_id IS NOT NULL GROUP BY 1
        ) c ON c.manager_id = e.employee_id
        WHERE e.n_direct_reports <> coalesce(c.n, 0) OR e.is_people_manager <> (coalesce(c.n, 0) > 0)""")
    add("direct_report_counts", bad_reports == 0, f"{bad_reports} rows with inconsistent report counts")

    bad_nearest = _scalar(con, """
        SELECT count(*) FROM truth_employees e LEFT JOIN truth_employees m ON e.manager_id = m.employee_id
        WHERE e.manager_id IS NOT NULL AND e.nearest_observed_manager_id IS DISTINCT FROM
              (CASE WHEN m.is_observed THEN m.employee_id ELSE m.nearest_observed_manager_id END)""")
    nearest_unobserved = _scalar(con, """
        SELECT count(*) FROM truth_employees e JOIN truth_employees n ON e.nearest_observed_manager_id = n.employee_id
        WHERE NOT n.is_observed""")
    add("nearest_observed_manager", bad_nearest == 0 and nearest_unobserved == 0,
        f"{bad_nearest} incorrect, {nearest_unobserved} pointing at unobserved people")

    bad_board = _scalar(con, """
        SELECT count(*) FROM truth_employees
        WHERE role_type = 'board' AND (manager_id IS NOT NULL OR n_direct_reports > 0 OR NOT is_observed)""")
    add("board_profiles_detached", bad_board == 0, f"{bad_board} board profiles attached to the tree")

    columns = [r[0] for r in con.execute("SELECT column_name FROM (DESCRIBE employees)").fetchall()]
    columns += [r[0] for r in con.execute("SELECT column_name FROM (DESCRIBE companies)").fetchall()]
    leaked = [c for c in columns if any(f in c.lower() for f in FORBIDDEN_OBSERVED_FRAGMENTS)]
    add("observed_schema_has_no_truth_columns", not leaked, f"suspicious columns: {leaked or 'none'}")

    split_counts = dict(con.execute("SELECT split, count(*) FROM companies GROUP BY 1").fetchall())
    ratios = cfg["splits"]
    worst = max(abs(split_counts.get(s, 0) / n_companies - ratios[s]) for s in SPLITS)
    add("splits_present_and_proportional",
        all(split_counts.get(s, 0) > 0 for s in SPLITS if ratios[s] > 0) and set(split_counts) <= set(SPLITS)
        and worst <= 0.05 + 1.0 / n_companies,
        f"counts={split_counts}, max ratio deviation={worst:.3f}")

    if n_companies >= 100:  # diversity is only meaningful at assessment scale, not for tiny samples
        missing_cells = con.execute("""
            WITH a AS (SELECT DISTINCT archetype FROM truth_companies), s AS (SELECT DISTINCT split FROM companies)
            SELECT a.archetype, s.split FROM a CROSS JOIN s
            EXCEPT SELECT t.archetype, c.split FROM truth_companies t JOIN companies c USING (company_id)""").fetchall()
        n_archetypes = _scalar(con, "SELECT count(DISTINCT archetype) FROM truth_companies")
        scope_kinds = {r[0] for r in con.execute("SELECT DISTINCT scope_kind FROM truth_companies").fetchall()}
        add("diversity_per_split",
            not missing_cells and n_archetypes == len(cfg["industries"]) and scope_kinds == {"none", "regional", "divisional"},
            f"{n_archetypes} archetypes, missing archetype x split cells={missing_cells or 'none'}, "
            f"scope kinds={sorted(scope_kinds)}")
    else:
        add("diversity_per_split", True, f"skipped for small dataset ({n_companies} companies < 100)")

    leaks = _scalar(con, """
        SELECT count(*) FROM truth_employees t JOIN companies c USING (company_id)
        WHERE c.split <> 'test' AND array_to_string(t.noise_tags, ',') LIKE '%holdout:%'""")
    add("holdout_noise_only_in_test", leaks == 0, f"{leaks} non-test rows with holdout noise")

    mismatched = [rel for rel, digest in manifest["files"].items() if sha256_file(paths.root / rel) != digest]
    add("manifest_hashes", not mismatched, f"mismatched files: {mismatched or 'none'}")
    return checks


def realism_stats(root: str | Path) -> dict[str, Any]:
    con = connect_with_truth(root)
    q = lambda sql: con.execute(sql).fetchall()  # noqa: E731
    stats: dict[str, Any] = {}

    stats["companies"] = _scalar(con, "SELECT count(*) FROM companies")
    stats["observed_profiles"] = _scalar(con, "SELECT count(*) FROM employees")
    stats["true_employees"] = _scalar(con, "SELECT count(*) FROM truth_employees WHERE role_type <> 'board'")
    stats["observed_employees_per_company"] = dict(zip(
        ["min", "p10", "median", "p90", "max", "mean"],
        con.execute("""
            SELECT min(n), quantile_cont(n, 0.1), median(n), quantile_cont(n, 0.9), max(n), avg(n)
            FROM (SELECT company_id, observed_employee_count AS n FROM truth_companies)""").fetchone(),
        strict=True))
    stats["companies_by_archetype"] = dict(q("SELECT archetype, count(*) FROM truth_companies GROUP BY 1 ORDER BY 2 DESC"))
    stats["companies_by_split"] = dict(q("SELECT split, count(*) FROM companies GROUP BY 1 ORDER BY 1"))
    stats["companies_by_scope_kind"] = dict(q("SELECT scope_kind, count(*) FROM truth_companies GROUP BY 1 ORDER BY 2 DESC"))
    stats["companies_by_country"] = dict(q("SELECT hq_country, count(*) FROM companies GROUP BY 1 ORDER BY 2 DESC"))
    stats["coverage_actual"] = dict(zip(
        ["mean", "min", "max"],
        con.execute("SELECT avg(coverage_actual), min(coverage_actual), max(coverage_actual) FROM truth_companies").fetchone(),
        strict=True))

    emp = "FROM truth_employees WHERE role_type <> 'board'"
    stats["manager_ratio"] = _scalar(con, f"SELECT avg(is_people_manager::INT) {emp}")
    stats["span_of_control"] = dict(zip(
        ["mean", "median", "p90", "max"],
        con.execute(f"""SELECT avg(n_direct_reports), median(n_direct_reports), quantile_cont(n_direct_reports, 0.9),
                               max(n_direct_reports) {emp} AND is_people_manager""").fetchone(),
        strict=True))
    stats["span_by_level"] = {
        int(lvl): round(float(avg), 2)
        for lvl, avg in q(f"SELECT level, avg(n_direct_reports) {emp} AND is_people_manager GROUP BY 1 ORDER BY 1 DESC")
    }
    stats["max_depth"] = dict(zip(
        ["min", "median", "max"],
        con.execute("SELECT min(max_depth), median(max_depth), max(max_depth) FROM truth_companies").fetchone(),
        strict=True))
    total = stats["true_employees"]
    stats["level_distribution_pct"] = {
        name: round(100 * n / total, 2) for name, n in q(f"SELECT level_name, count(*) {emp} GROUP BY level, 1 ORDER BY level")
    }
    stats["department_distribution_pct"] = {
        d: round(100 * n / total, 2) for d, n in q(f"SELECT department, count(*) {emp} GROUP BY 1 ORDER BY 2 DESC")
    }
    stats["missing_rate_by_band"] = dict(q(f"""
        SELECT CASE WHEN level >= 9 THEN 'exec (9-12)' WHEN level >= 4 THEN 'middle (4-8)' ELSE 'ic (0-3)' END AS band,
               round(1 - avg(is_observed::INT), 4)
        {emp} GROUP BY 1 ORDER BY 1"""))
    stats["observed_with_unobserved_manager_pct"] = _scalar(con, f"""
        SELECT round(100 * avg((e.manager_id IS NOT NULL AND NOT m.is_observed)::INT), 2)
        FROM truth_employees e LEFT JOIN truth_employees m ON e.manager_id = m.employee_id
        WHERE e.is_observed AND e.role_type NOT IN ('board', 'ceo')""")
    stats["observed_null_rates_pct"] = dict(zip(
        ["title", "department", "location", "start_date"],
        [round(100 * float(x), 2) for x in con.execute("""
            SELECT avg((title IS NULL)::INT), avg((department IS NULL)::INT), avg((location IS NULL)::INT),
                   avg((start_date IS NULL)::INT) FROM employees""").fetchone()],
        strict=True))
    distinct_raw = _scalar(con, "SELECT count(DISTINCT lower(title)) FROM employees")
    stats["distinct_raw_titles"] = distinct_raw
    stats["distinct_canonical_titles"] = _scalar(con, "SELECT count(DISTINCT canonical_title) FROM truth_employees")
    stats["distinct_title_ratio"] = distinct_raw / max(1, stats["observed_profiles"])
    stats["noise_tag_frequency_pct"] = {
        tag: round(100 * n / stats["observed_profiles"], 2)
        for tag, n in q("""SELECT tag, count(*) FROM (SELECT unnest(noise_tags) AS tag FROM truth_employees)
                           GROUP BY 1 ORDER BY 2 DESC""")
    }
    stats["top_raw_titles"] = q("SELECT title, count(*) FROM employees GROUP BY 1 ORDER BY 2 DESC LIMIT 15")
    stats["sample_noisy_titles"] = q("""
        SELECT t.canonical_title, o.title FROM employees o JOIN truth_employees t USING (employee_id)
        WHERE o.title IS DISTINCT FROM t.canonical_title ORDER BY hash(o.employee_id) LIMIT 20""")
    return stats


def realism_warnings(stats: dict[str, Any], targets: dict[str, list[float]]) -> list[Check]:
    observed = {
        "mean_span_of_control": stats["span_of_control"]["mean"],
        "manager_ratio": stats["manager_ratio"],
        "median_max_depth": stats["max_depth"]["median"],
        "mean_coverage": stats["coverage_actual"]["mean"],
        "distinct_title_ratio": stats["distinct_title_ratio"],
    }
    out = []
    for name, (lo, hi) in targets.items():
        value = float(observed[name])
        out.append(Check(name, lo <= value <= hi, f"{value:.3f} (target {lo}-{hi})"))
    return out


def _markdown(checks: list[Check], realism: list[Check], stats: dict[str, Any]) -> str:
    lines = ["# Dataset validation report", "", "## Structural checks (hard)", "", "| Check | Result | Detail |", "|---|---|---|"]
    lines += [f"| {c.name} | {'PASS' if c.passed else 'FAIL'} | {c.detail} |" for c in checks]
    lines += ["", "## Realism targets (soft)", "", "| Metric | Result | Value |", "|---|---|---|"]
    lines += [f"| {c.name} | {'OK' if c.passed else 'WARN'} | {c.detail} |" for c in realism]
    lines += ["", "## Statistics", ""]
    for key, value in stats.items():
        if key in ("top_raw_titles", "sample_noisy_titles"):
            continue
        lines.append(f"- **{key}**: {json.dumps(value, default=str)}")
    lines += ["", "## Most common observed titles", "", "| Title | Count |", "|---|---|"]
    lines += [f"| {t} | {n} |" for t, n in stats["top_raw_titles"]]
    lines += ["", "## Sample canonical -> observed titles", "", "| Canonical | Observed |", "|---|---|"]
    lines += [f"| {c} | {o} |" for c, o in stats["sample_noisy_titles"]]
    return "\n".join(lines) + "\n"


def validate_dataset(root: str | Path) -> dict[str, Any]:
    paths = DatasetPaths.of(root)
    manifest = json.loads(paths.manifest.read_text())
    checks = structural_checks(root, manifest)
    stats = realism_stats(root)
    realism = realism_warnings(stats, manifest["config"]["realism_targets"])
    report = {
        "passed": all(c.passed for c in checks),
        "checks": [asdict(c) for c in checks],
        "realism": [asdict(c) for c in realism],
        "stats": stats,
    }
    paths.reports_dir.mkdir(parents=True, exist_ok=True)
    (paths.reports_dir / "validation.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    (paths.reports_dir / "realism.md").write_text(_markdown(checks, realism, stats))
    return report
