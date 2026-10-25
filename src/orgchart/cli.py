"""`orgchart` command-line interface."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table
from rich.tree import Tree

from orgchart.config import load_config, parse_override

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Org chart inference platform.")
console = Console()

DEFAULT_CONFIG = Path("configs/generation.yaml")


@app.command()
def generate(
    out: Annotated[Path, typer.Option(help="Output dataset directory.")] = Path("data/full"),
    config: Annotated[Path, typer.Option(help="Generation config YAML.")] = DEFAULT_CONFIG,
    n_companies: Annotated[int | None, typer.Option(help="Override dataset.n_companies.")] = None,
    seed: Annotated[int | None, typer.Option(help="Override dataset.seed.")] = None,
    name: Annotated[str | None, typer.Option(help="Override dataset.name.")] = None,
    set_: Annotated[list[str] | None, typer.Option("--set", help="Override any config value: key.path=value.")] = None,
    workers: Annotated[int, typer.Option(help="Worker processes (output is identical for any value).")] = max(1, (os.cpu_count() or 2) - 1),
    force: Annotated[bool, typer.Option(help="Overwrite an existing dataset.")] = False,
    validate: Annotated[bool, typer.Option(help="Run validation after generating.")] = True,
) -> None:
    """Generate a synthetic dataset (observed + truth tables, manifest)."""
    from orgchart.generation.generator import generate_dataset

    overrides = dict(parse_override(s) for s in (set_ or []))
    if n_companies is not None:
        overrides["dataset.n_companies"] = n_companies
    if seed is not None:
        overrides["dataset.seed"] = seed
    if name is not None:
        overrides["dataset.name"] = name
    cfg = load_config(config, overrides)
    started = time.perf_counter()
    manifest = generate_dataset(cfg, out, workers=workers, force=force)
    elapsed = time.perf_counter() - started
    counts = manifest["counts"]
    console.print(
        f"[green]Generated[/] {counts['companies']} companies, {counts['observed_profiles']:,} observed profiles "
        f"({counts['true_employees']:,} true employees) in {elapsed:.1f}s -> {out}"
    )
    console.print(f"splits: {counts['splits']}  config fingerprint: {manifest['config_fingerprint']}")
    if validate:
        _run_validation(out)


def _run_validation(dataset: Path) -> None:
    from orgchart.generation.validation import validate_dataset

    report = validate_dataset(dataset)
    table = Table(title="Structural checks")
    for col in ("check", "result", "detail"):
        table.add_column(col)
    for c in report["checks"]:
        table.add_row(c["name"], "[green]PASS[/]" if c["passed"] else "[red]FAIL[/]", c["detail"])
    console.print(table)
    realism = Table(title="Realism targets")
    for col in ("metric", "result", "value"):
        realism.add_column(col)
    for c in report["realism"]:
        realism.add_row(c["name"], "[green]OK[/]" if c["passed"] else "[yellow]WARN[/]", c["detail"])
    console.print(realism)
    console.print(f"Reports written to {dataset / 'reports'}")
    if not report["passed"]:
        raise typer.Exit(code=1)


@app.command("validate")
def validate_cmd(dataset: Annotated[Path, typer.Argument(help="Dataset directory.")] = Path("data/full")) -> None:
    """Run structural checks and realism statistics on a dataset."""
    _run_validation(dataset)


@app.command()
def stats(dataset: Annotated[Path, typer.Argument(help="Dataset directory.")] = Path("data/full")) -> None:
    """Print realism statistics as JSON."""
    from orgchart.generation.validation import realism_stats

    console.print_json(json.dumps(realism_stats(dataset), default=str))


@app.command()
def sql(
    query: Annotated[str, typer.Argument(help="SQL over views: companies, employees (+ truth_* with --with-truth).")],
    dataset: Annotated[Path, typer.Option(help="Dataset directory.")] = Path("data/full"),
    with_truth: Annotated[bool, typer.Option(help="Also expose truth_companies / truth_employees.")] = False,
    limit: Annotated[int, typer.Option(help="Max rows to print.")] = 50,
) -> None:
    """Run ad-hoc DuckDB SQL against a dataset."""
    if with_truth:
        from orgchart.storage.truth import connect_with_truth as connect
    else:
        from orgchart.storage.observed import connect_observed as connect
    rel = connect(dataset).sql(query)
    table = Table()
    for col in rel.columns:
        table.add_column(col)
    for row in rel.limit(limit).fetchall():
        table.add_row(*("" if v is None else str(v) for v in row))
    console.print(table)


@app.command()
def normalize(
    dataset: Annotated[Path, typer.Argument(help="Dataset directory.")] = Path("data/full"),
    evaluate: Annotated[bool, typer.Option(help="Also evaluate against ground truth.")] = True,
) -> None:
    """Normalise every observed title (observed data only) -> derived/normalized_titles.parquet."""
    from orgchart.normalization.pipeline import normalize_dataset

    summary = normalize_dataset(dataset)
    console.print(
        f"[green]Normalised[/] {summary['rows']:,} profiles in {summary['seconds']}s: level resolved "
        f"{summary['level_resolved_pct']}%, department resolved {summary['department_resolved_pct']}%, "
        f"{summary['non_employee_profiles']} board/advisor profiles -> {summary['output']}"
    )
    if evaluate:
        evaluate_normalization_cmd(dataset)


@app.command("evaluate-normalization")
def evaluate_normalization_cmd(
    dataset: Annotated[Path, typer.Argument(help="Dataset directory.")] = Path("data/full"),
) -> None:
    """Score normalised titles against ground truth (per split) -> reports/normalization_eval.{json,md}."""
    from orgchart.evaluation.normalization_eval import evaluate_normalization

    report = evaluate_normalization(dataset)
    table = Table(title="Title normalisation (rules developed on train; calibration/test held out)")
    for col in ("metric", "train", "calibration", "test"):
        table.add_column(col)
    s = report["splits"]
    rows = [
        ("seniority exact %", lambda m: m["seniority"]["exact_pct"]),
        ("seniority within ±1 %", lambda m: m["seniority"]["within_one_pct"]),
        ("seniority coverage %", lambda m: m["seniority"]["coverage_pct"]),
        ("management accuracy %", lambda m: m["management"]["accuracy_pct"]),
        ("Staff/Principal flagged manager %", lambda m: m["management"]["staff_principal_title_false_manager_pct"]),
        ("department accuracy %", lambda m: m["department"]["accuracy_pct"]),
        ("department coverage %", lambda m: m["department"]["coverage_pct"]),
        ("specialty precision %", lambda m: m["specialty"]["precision_pct"]),
        ("scope precision %", lambda m: m["scope"]["precision_pct"]),
    ]
    for name, fn in rows:
        table.add_row(name, *(str(fn(s[k])) for k in ("train", "calibration", "test")))
    console.print(table)
    console.print(f"Full report: {dataset / 'reports' / 'normalization_eval.md'}")


@app.command("parse-title")
def parse_title_cmd(
    title: Annotated[str, typer.Argument(help="Raw job title.")],
    department: Annotated[str | None, typer.Option(help="Raw department field, if any.")] = None,
    industry: Annotated[str, typer.Option(help="Industry label, e.g. 'Financial Services'.")] = "Software Development",
) -> None:
    """Parse one title and print the result with its explanation."""
    import dataclasses

    from orgchart.normalization.parser import parse_title

    result = dataclasses.asdict(parse_title(title, department, industry))
    explanation = result.pop("explanation")
    result.pop("unresolved_qualifiers")
    console.print_json(json.dumps(result))
    for line in explanation:
        console.print(f"  • {line}")


@app.command()
def fit(
    dataset: Annotated[Path, typer.Argument(help="Dataset directory.")] = Path("data/full"),
    target_precision: Annotated[float, typer.Option(help="Precision target used to pick thresholds.")] = 0.95,
) -> None:
    """Fit learned weights (train), calibration + thresholds (calibration companies) -> models/*.json."""
    from orgchart.learning.fit import fit_models

    summary = fit_models(dataset, target_precision=target_precision)
    table = Table(title=f"Calibration companies at {target_precision:.0%} target precision")
    for col in ("variant", "DIRECT thr", "DIRECT n / prec", "AMBIGUOUS thr", "ANCESTOR thr", "ANCESTOR n / prec",
                "resolved coverage"):
        table.add_column(col)
    for variant, rep in summary["calibration"].items():
        table.add_row(variant, str(rep["direct"]["threshold"]), f"{rep['direct']['predictions']} / {rep['direct']['precision']}",
                      str(rep["ambiguous"]["threshold"]), str(rep["ancestor"]["threshold"]),
                      f"{rep['ancestor']['predictions']} / {rep['ancestor']['precision']}",
                      f"{100 * rep['resolved_coverage']:.1f}%")
    console.print(table)
    console.print(f"[green]Selected[/] {summary['selected']}: {summary['selection']['rule']} "
                  f"(resolved coverage gain {summary['selection']['learned_minus_rules_resolved_coverage_pp']} pp)")


@app.command()
def infer(dataset: Annotated[Path, typer.Argument(help="Dataset directory.")] = Path("data/full")) -> None:
    """Predict reporting lines for every observed profile (observed data + model files only)."""
    from orgchart.inference.pipeline import infer_dataset

    summary = infer_dataset(dataset)
    for variant, counts in summary["variants"].items():
        console.print(f"{variant:>8}: " + ", ".join(f"{k} {v:,}" for k, v in sorted(counts.items())))
    console.print(f"[green]Inferred[/] {summary['profiles']:,} profiles in {summary['companies']} companies "
                  f"in {summary['seconds']}s (selected variant: {summary['selected']})")


@app.command("check-predictions")
def check_predictions(
    dataset: Annotated[Path, typer.Argument(help="Dataset directory.")] = Path("data/full"),
    variant: Annotated[str, typer.Option(help="Prediction variant to check.")] = "selected",
) -> None:
    """Structural checks on predictions: one manager, no self-loops, no cycles, same company, seniority."""
    from orgchart.inference.validate import validate_dataset_predictions

    checks = validate_dataset_predictions(dataset, variant)
    table = Table(title=f"Prediction invariants ({variant})")
    for col in ("check", "result", "violations"):
        table.add_column(col)
    for c in checks:
        table.add_row(c["name"], "[green]PASS[/]" if c["passed"] else "[red]FAIL[/]", str(c["violations"]))
    console.print(table)
    if not all(c["passed"] for c in checks):
        raise typer.Exit(code=1)


@app.command("evaluate")
def evaluate_cmd(dataset: Annotated[Path, typer.Argument(help="Dataset directory.")] = Path("data/full")) -> None:
    """Score predictions against ground truth -> reports/inference_eval.{json,md}."""
    from orgchart.evaluation.inference_eval import evaluate_inference

    report = evaluate_inference(dataset)
    table = Table(title="Held-out TEST companies (thresholds fixed on calibration)")
    for col in ("variant", "DIRECT (exact manager) cov / prec %", "ANCESTOR (in chain above) cov / prec %",
                "AMBIGUOUS cov / contain %", "any claim (mixed) cov / prec %"):
        table.add_column(col)
    for variant, splits in report["variants"].items():
        t = splits["test"]
        table.add_row(variant, f"{t['direct']['coverage_pct']} / {t['direct']['precision_pct']}",
                      f"{t['ancestor']['coverage_pct']} / {t['ancestor']['precision_pct']}",
                      f"{t['ambiguous']['coverage_pct']} / {t['ambiguous']['containment_precision_pct']}",
                      f"{t['resolved']['coverage_pct']} / {t['resolved']['precision_pct']}")
    console.print(table)
    cal = report["test_calibration"]["top_candidate_direct_probability"]
    console.print(f"Calibration (test): top-candidate ECE {cal['ece_pct']}%, Brier {cal['brier']}")
    console.print("DIRECT names the exact manager. ANCESTOR names someone in the chain above the employee, who may "
                  "be the direct manager or higher, and does not claim which. 'Any claim' pools all statuses and is not manager accuracy.")
    console.print(f"Full report: {dataset / 'reports' / 'inference_eval.md'}")


@app.command()
def explain(
    employee_id: Annotated[str, typer.Argument(help="Employee ID, e.g. E0123456789ab.")],
    dataset: Annotated[Path, typer.Option(help="Dataset directory.")] = Path("data/full"),
) -> None:
    """Show one employee's prediction, ranked candidates and per-feature contributions (no ground truth)."""
    import pandas as pd

    from orgchart.inference.decide import top_contributions
    from orgchart.inference.scoring import Model
    from orgchart.paths import DatasetPaths

    paths = DatasetPaths.of(dataset)
    pred = pd.read_parquet(paths.predictions(), filters=[("employee_id", "==", employee_id)])
    if pred.empty:
        raise typer.BadParameter(f"unknown employee {employee_id}")
    row = pred.iloc[0]
    titles = pd.read_parquet(paths.normalized_titles, columns=["employee_id", "title_raw", "company_id"],
                             filters=[("company_id", "==", row["company_id"])]).set_index("employee_id")["title_raw"]
    console.print(f"[bold]{titles.get(employee_id)}[/] ({employee_id}, {row['company_id']}): "
                  f"[cyan]{row['status']}[/] confidence={row['confidence']}  reasons={list(row['reason_codes'])}")
    for line in row["explanation"]:
        console.print(f"  • {line}")
    model = Model.load(paths.model())
    cands = pd.read_parquet(paths.candidates, filters=[("employee_id", "==", employee_id)]).sort_values("rank")
    table = Table(title="Ranked candidates")
    for col in ("rank", "candidate", "p direct", "p ancestor", "top contributions (direct head)"):
        table.add_column(col)
    for (_, c), contrib in zip(cands.iterrows(), top_contributions(cands, model.direct.vector(), k=3), strict=True):
        table.add_row(str(c["rank"]), f"{titles.get(c['candidate_id'])} ({c['candidate_id']})", f"{c['p_direct']:.3f}",
                      f"{c['p_ancestor']:.3f}", "; ".join(contrib))
    console.print(table)


@app.command()
def pipeline(
    dataset: Annotated[Path, typer.Argument(help="Dataset directory.")] = Path("data/full"),
    generate_data: Annotated[bool, typer.Option("--generate/--no-generate", help="Regenerate the dataset first.")] = False,
    config: Annotated[Path, typer.Option(help="Generation config YAML (with --generate).")] = DEFAULT_CONFIG,
    workers: Annotated[int, typer.Option(help="Generation workers.")] = max(1, (os.cpu_count() or 2) - 1),
) -> None:
    """End to end: [generate] -> normalize -> fit -> infer -> check-predictions -> evaluate, with timings."""
    timings: dict[str, float] = {}

    def step(name: str, fn) -> None:
        started = time.perf_counter()
        console.rule(name)
        fn()
        timings[name] = round(time.perf_counter() - started, 1)

    if generate_data:
        step("generate", lambda: generate(out=dataset, config=config, n_companies=None, seed=None, name=None, set_=None,
                                          workers=workers, force=True, validate=True))
    step("normalize", lambda: normalize(dataset, evaluate=True))
    step("fit", lambda: fit(dataset))
    step("infer", lambda: infer(dataset))
    step("check-predictions", lambda: check_predictions(dataset, "selected"))
    step("evaluate", lambda: evaluate_cmd(dataset))
    timings["total"] = round(sum(timings.values()), 1)
    (dataset / "reports").mkdir(parents=True, exist_ok=True)
    (dataset / "reports" / "pipeline_runtime.json").write_text(json.dumps(timings, indent=2) + "\n")
    console.print("Runtime (s): " + ", ".join(f"{k} {v}" for k, v in timings.items()))


@app.command()
def verify(
    dataset: Annotated[Path, typer.Argument(help="Dataset directory.")] = Path("data/full"),
    min_companies: Annotated[int, typer.Option(help="Required number of companies.")] = 600,
) -> None:
    """Delivery checks on a built dataset: data validation, scale, prediction invariants, determinism."""
    import pandas as pd

    from orgchart.generation.validation import validate_dataset
    from orgchart.inference.data import load_people
    from orgchart.inference.pipeline import decision_view, predict_people
    from orgchart.inference.scoring import Model
    from orgchart.inference.validate import validate_predictions
    from orgchart.paths import DatasetPaths
    from orgchart.storage.truth import connect_with_truth

    paths = DatasetPaths.of(dataset)
    results: list[tuple[str, bool, str]] = []
    report = validate_dataset(dataset)
    failed = [c["name"] for c in report["checks"] if not c["passed"]]
    results.append(("dataset structural checks", not failed, f"{len(report['checks'])} checks; failed: {failed or 'none'}"))
    # Delivery check (not inference): count observed employees excluding board/advisor profiles via truth.
    n, min_obs = connect_with_truth(dataset).execute("""
        SELECT count(DISTINCT c.company_id), min(k) FROM companies c
        JOIN (SELECT e.company_id, count(*) k FROM employees e JOIN truth_employees t USING (employee_id)
              WHERE t.role_type <> 'board' GROUP BY 1) e USING (company_id)""").fetchone()
    results.append((f"at least {min_companies} companies", n >= min_companies, f"{n} companies"))
    results.append(("observed employees per company >= 100 (board excluded)", min_obs >= 100,
                    f"minimum {min_obs} employees"))
    people = load_people(dataset)
    for variant in ("selected", "naive"):
        checks = validate_predictions(pd.read_parquet(paths.predictions(variant)), people)
        bad = [c["name"] for c in checks if not c["passed"]]
        results.append((f"prediction invariants ({variant})", not bad, f"{len(checks)} checks; failed: {bad or 'none'}"))
    sample = sorted(people["company_id"].unique())[:: max(1, n // 12)]
    subset = people[people["company_id"].isin(sample)]
    again = predict_people(subset, Model.load(paths.model()))
    stored = pd.read_parquet(paths.predictions(), filters=[("company_id", "in", list(sample))])
    same = decision_view(again).equals(decision_view(stored))
    results.append(("inference is deterministic (re-run on sample companies)", same, f"{len(sample)} companies"))
    table = Table(title="Delivery verification")
    for col in ("check", "result", "detail"):
        table.add_column(col)
    for name, ok, detail in results:
        table.add_row(name, "[green]PASS[/]" if ok else "[red]FAIL[/]", detail)
    console.print(table)
    if not all(ok for _, ok, _ in results):
        raise typer.Exit(code=1)


@app.command("show-company")
def show_company(
    company_id: Annotated[str, typer.Argument(help="Company ID, e.g. C0001.")],
    dataset: Annotated[Path, typer.Option(help="Dataset directory.")] = Path("data/full"),
    max_depth: Annotated[int, typer.Option(help="Tree depth to print.")] = 3,
    max_children: Annotated[int, typer.Option(help="Children shown per node.")] = 8,
) -> None:
    """Print a company's TRUE org tree (ground truth; for dataset review, not inference)."""
    from orgchart.storage.truth import connect_with_truth

    con = connect_with_truth(dataset)
    rows = con.execute("""
        SELECT t.employee_id, t.manager_id, t.level, t.canonical_title, t.is_observed, o.title, t.department, t.n_direct_reports
        FROM truth_employees t LEFT JOIN employees o USING (employee_id)
        WHERE t.company_id = ? AND t.role_type <> 'board'""", [company_id]).fetchall()
    if not rows:
        raise typer.BadParameter(f"unknown company {company_id}")
    company = con.execute("SELECT company_name, industry, hq_city, split FROM companies WHERE company_id = ?",
                          [company_id]).fetchone()
    children: dict[str | None, list[tuple]] = {}
    for r in rows:
        children.setdefault(r[1], []).append(r)
    for kids in children.values():
        kids.sort(key=lambda r: (-r[2], -r[7], r[3]))

    def label(r: tuple) -> str:
        observed = f"[cyan]{r[5]!s}[/]" if r[4] else "[red]<not observed>[/]"
        return f"[bold]{r[3]}[/] (L{r[2]}, {r[6]}, {r[7]} reports) — observed: {observed}"

    root = children[None][0]
    tree = Tree(f"{company[0]} · {company[1]} · {company[2]} · split={company[3]}\n{label(root)}")

    def add(node: Tree, emp_id: str, depth: int) -> None:
        if depth >= max_depth:
            return
        kids = children.get(emp_id, [])
        for r in kids[:max_children]:
            add(node.add(label(r)), r[0], depth + 1)
        if len(kids) > max_children:
            node.add(f"[dim]... {len(kids) - max_children} more[/]")

    add(tree, root[0], 0)
    console.print(tree)


if __name__ == "__main__":
    app()
