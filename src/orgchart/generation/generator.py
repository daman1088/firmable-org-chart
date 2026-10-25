"""Dataset orchestration: plan companies, assign splits, generate in parallel, write tables + manifest."""

from __future__ import annotations

import datetime as dt
import json
import math
import platform
import re
import shutil
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import duckdb
import faker
import numpy as np
import pyarrow as pa

import orgchart
from orgchart.config import GenerationConfig
from orgchart.generation.catalog import (
    ARCHETYPES,
    COMPANY_NAME_MIDDLES,
    COMPANY_NAME_PREFIXES,
    COUNTRIES,
)
from orgchart.generation.company import CompanyPlan, CompanyRows, generate_company
from orgchart.generation.rand import pick, weighted_pick
from orgchart.paths import DatasetPaths
from orgchart.schemas import OBSERVED_COMPANIES, OBSERVED_EMPLOYEES, SPLITS, TRUTH_COMPANIES, TRUTH_EMPLOYEES
from orgchart.storage.parquet_io import sha256_file, write_table

SIZE_BANDS = (200, 500, 1000)


def _size_band(n: int) -> int:
    return sum(n > b for b in SIZE_BANDS)


def _company_name(rng: np.random.Generator, archetype: str, country: str, taken: set[str]) -> tuple[str, str, str]:
    arch = ARCHETYPES[archetype]
    ctry = COUNTRIES[country]
    for _ in range(1000):
        prefix = pick(rng, COMPANY_NAME_PREFIXES)
        middle = pick(rng, COMPANY_NAME_MIDDLES)
        root = f"{prefix}{middle.lower()}" if middle else prefix
        short = f"{root} {pick(rng, arch.name_words)}"
        if short not in taken:
            taken.add(short)
            suffix = pick(rng, ctry.legal_suffixes)
            full = f"{short} {suffix}".strip()
            domain = re.sub(r"[^a-z0-9]", "", short.lower()) + "." + ctry.tld
            return full, short, domain
    raise RuntimeError("could not generate a unique company name")


def assign_splits(strata: list[tuple], ratios: dict[str, float], rng: np.random.Generator) -> list[str]:
    """Stratified, deterministic split assignment.

    Companies are grouped by stratum, shuffled within the stratum, then dealt out
    so that every split's running count tracks its target ratio (largest-deficit
    first). Each stratum therefore gets near-proportional representation.
    """
    by_stratum: dict[tuple, list[int]] = defaultdict(list)
    for i, key in enumerate(strata):
        by_stratum[key].append(i)
    ordered: list[int] = []
    for key in sorted(by_stratum):
        members = by_stratum[key]
        rng.shuffle(members)
        ordered.extend(members)
    names = [s for s in SPLITS if ratios[s] > 0]
    assigned = {s: 0 for s in names}
    result = [""] * len(strata)
    for k, idx in enumerate(ordered, start=1):
        split = max(names, key=lambda s: (ratios[s] * k - assigned[s], ratios[s]))
        assigned[split] += 1
        result[idx] = split
    return result


def plan_companies(cfg: GenerationConfig) -> list[CompanyPlan]:
    rng = np.random.default_rng(cfg.dataset.seed)
    sz, cov = cfg.company_size, cfg.coverage
    taken: set[str] = set()
    drafts = []
    for i in range(cfg.dataset.n_companies):
        archetype = weighted_pick(rng, cfg.industries)
        country = weighted_pick(rng, cfg.countries)
        cities = COUNTRIES[country].cities[:3]
        hq_city = weighted_pick(rng, dict(zip([c.name for c in cities], (0.5, 0.3, 0.2)[: len(cities)], strict=True)))
        observed_target = int(round(sz.observed_median * math.exp(sz.sigma * rng.standard_normal())))
        observed_target = int(np.clip(observed_target, cfg.dataset.min_observed_employees, sz.max_observed))
        coverage = float(np.clip(rng.beta(cov.mean * cov.concentration, (1 - cov.mean) * cov.concentration), cov.min, cov.max))
        lo, hi = ARCHETYPES[archetype].founded_range
        founded = int(rng.integers(lo, hi + 1))
        name, short, domain = _company_name(rng, archetype, country, taken)
        seed = int(rng.integers(0, 2**63 - 1))
        drafts.append((i, archetype, country, hq_city, observed_target, coverage, founded, name, short, domain, seed))

    strata = [(d[1], _size_band(d[4])) for d in drafts]
    splits = assign_splits(strata, cfg.splits.model_dump(), rng)
    return [
        CompanyPlan(
            index=i, company_id=f"C{i:04d}", name=name, short_name=short, domain=domain, archetype=archetype,
            country=country, hq_city=hq_city, founded_year=founded, observed_target=target, coverage=coverage,
            split=split, seed=seed,
        )
        for (i, archetype, country, hq_city, target, coverage, founded, name, short, domain, seed), split
        in zip(drafts, splits, strict=True)
    ]


def _generate_all(plans: list[CompanyPlan], cfg: GenerationConfig, workers: int) -> list[CompanyRows]:
    fn = partial(generate_company, cfg=cfg)
    if workers <= 1:
        return [fn(p) for p in plans]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, plans, chunksize=max(1, len(plans) // (workers * 8))))


def _table(rows: list[dict], schema: pa.Schema, sort_keys: list[str]) -> pa.Table:
    table = pa.Table.from_pylist(rows, schema=schema)
    return table.sort_by([(k, "ascending") for k in sort_keys])


def _prepare_output(paths: DatasetPaths, force: bool) -> None:
    outputs = (paths.observed_dir, paths.truth_dir, paths.reports_dir, paths.derived_dir, paths.models_dir, paths.manifest)
    existing = [p for p in outputs if p.exists()]
    if existing and not force:
        raise FileExistsError(f"dataset already exists at {paths.root} (use --force to overwrite)")
    for p in existing:
        shutil.rmtree(p) if p.is_dir() else p.unlink()
    paths.root.mkdir(parents=True, exist_ok=True)


def generate_dataset(cfg: GenerationConfig, out_dir: str | Path, workers: int = 1, force: bool = False) -> dict:
    paths = DatasetPaths.of(out_dir)
    _prepare_output(paths, force)
    plans = plan_companies(cfg)
    results = _generate_all(plans, cfg, workers)

    obs_companies = _table([r.observed_company for r in results], OBSERVED_COMPANIES, ["company_id"])
    truth_companies = _table([r.truth_company for r in results], TRUTH_COMPANIES, ["company_id"])
    obs_employees = _table([e for r in results for e in r.observed_employees], OBSERVED_EMPLOYEES,
                           ["company_id", "employee_id"])
    truth_employees = _table([e for r in results for e in r.truth_employees], TRUTH_EMPLOYEES,
                             ["company_id", "employee_id"])

    write_table(obs_companies, paths.observed_companies, OBSERVED_COMPANIES)
    write_table(obs_employees, paths.observed_employees, OBSERVED_EMPLOYEES)
    write_table(truth_companies, paths.truth_companies, TRUTH_COMPANIES)
    write_table(truth_employees, paths.truth_employees, TRUTH_EMPLOYEES)

    split_counts = {s: sum(1 for p in plans if p.split == s) for s in SPLITS}
    manifest = {
        "dataset": cfg.dataset.name,
        "generator_version": orgchart.__version__,
        "created_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "config_fingerprint": cfg.fingerprint(),
        "config": cfg.to_jsonable(),
        "counts": {
            "companies": obs_companies.num_rows,
            "observed_profiles": obs_employees.num_rows,
            "true_employees": int(sum(r.truth_company["true_headcount"] for r in results)),
            "observed_employees": int(sum(r.truth_company["observed_employee_count"] for r in results)),
            "splits": split_counts,
        },
        "files": {str(p.relative_to(paths.root)): sha256_file(p) for p in paths.data_files()},
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pyarrow": pa.__version__,
            "duckdb": duckdb.__version__,
            "faker": faker.VERSION,
        },
    }
    paths.manifest.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    return manifest
