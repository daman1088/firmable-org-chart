"""Generate one company end-to-end and return its observed and truth rows."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from orgchart.config import GenerationConfig
from orgchart.generation import noise, people
from orgchart.generation.catalog import ARCHETYPES, COUNTRIES
from orgchart.generation.rand import chance
from orgchart.generation.structure import build_tree
from orgchart.generation.titles import assign_canonical_titles, canonical_title
from orgchart.ladder import LEVEL_NAMES


@dataclass(frozen=True)
class CompanyPlan:
    index: int
    company_id: str
    name: str
    short_name: str
    domain: str
    archetype: str
    country: str
    hq_city: str
    founded_year: int
    observed_target: int
    coverage: float
    split: str
    seed: int


@dataclass
class CompanyRows:
    observed_company: dict[str, Any]
    truth_company: dict[str, Any]
    observed_employees: list[dict[str, Any]] = field(default_factory=list)
    truth_employees: list[dict[str, Any]] = field(default_factory=list)


HEADCOUNT_BANDS = ((200, "51-200"), (500, "201-500"), (1000, "501-1,000"), (5000, "1,001-5,000"), (10**9, "5,001-10,000"))


def headcount_band(n: int) -> str:
    return next(label for upper, label in HEADCOUNT_BANDS if n <= upper)


def generate_company(plan: CompanyPlan, cfg: GenerationConfig) -> CompanyRows:
    rng = np.random.default_rng(plan.seed)
    arch = ARCHETYPES[plan.archetype]
    country = COUNTRIES[plan.country]
    hq = next(c for c in country.cities if c.name == plan.hq_city)

    true_size = max(cfg.dataset.min_observed_employees + 1, math.ceil(plan.observed_target / plan.coverage))
    offices = people.plan_offices(rng, plan.country, hq, true_size)
    regions, region_offices = people.regions_for(offices)
    tree = build_tree(rng, cfg, plan.archetype, true_size, regions)

    assign_canonical_titles(rng, tree, arch)

    places = people.assign_locations(rng, tree, offices, region_offices)
    observed = noise.simulate_coverage(rng, tree, plan.coverage, cfg.coverage, cfg.dataset.min_observed_employees)
    nearest = noise.nearest_observed_manager(tree, observed)
    depth = tree.depth_of()

    ids = people.make_employee_ids(rng, len(tree.nodes))
    names = people.make_names(rng, plan.country, len(tree.nodes))
    ceo_is_founder = "Founder" in tree.root.title
    holdout = plan.split == "test"
    ctx = noise.TitleContext(
        cfg=cfg.noise,
        holdout=holdout,
        small_company=true_size < cfg.noise.small_company_size,
        company_short=plan.short_name,
        render_at_level=lambda n, lvl: canonical_title(rng, n, tree.nodes, arch, level=lvl, allow_decorations=False),
    )

    rows = CompanyRows(observed_company={}, truth_company={})
    for node in tree.nodes:
        is_obs = node.nid in observed
        tags: list[str] = []
        if is_obs:
            raw_title, title_tags = noise.observe_title(rng, node, ctx)
            raw_dept, dept_tags = noise.observe_department(rng, node, cfg.noise, tree.departments)
            tags = title_tags + dept_tags
            location = None
            if not chance(rng, cfg.noise.missing_location):
                location = people.format_location(rng, places[node.nid], cfg.noise.greater_area_location, plan.country)
            started = None
            if not chance(rng, cfg.noise.missing_start_date):
                started = people.start_date(rng, node, cfg.dataset.snapshot_date, plan.founded_year, ceo_is_founder)
            rows.observed_employees.append(
                {
                    "employee_id": ids[node.nid],
                    "company_id": plan.company_id,
                    "full_name": names[node.nid],
                    "title": raw_title,
                    "department": raw_dept,
                    "location": location,
                    "start_date": started,
                }
            )
        nearest_mgr = nearest.get(node.nid)
        rows.truth_employees.append(
            {
                "employee_id": ids[node.nid],
                "company_id": plan.company_id,
                "manager_id": None if node.parent is None else ids[node.parent],
                "level": node.level,
                "level_name": None if node.level is None else LEVEL_NAMES[node.level],
                "department": node.dept,
                "specialty": node.specialty,
                "scope": node.scope,
                "role_type": node.role_type,
                "employment_type": node.employment_type,
                "canonical_title": node.title,
                "is_people_manager": bool(node.children),
                "n_direct_reports": len(node.children),
                "depth": depth.get(node.nid),
                "is_observed": is_obs,
                "nearest_observed_manager_id": None if nearest_mgr is None else ids[nearest_mgr],
                "noise_tags": tags,
            }
        )

    employees = tree.employees
    n_obs_employees = sum(1 for n in employees if n.nid in observed)
    rows.observed_company = {
        "company_id": plan.company_id,
        "company_name": plan.name,
        "domain": plan.domain,
        "industry": arch.industry_label,
        "hq_city": plan.hq_city,
        "hq_country": country.name,
        "headcount_band": headcount_band(len(employees)),
        "founded_year": plan.founded_year,
        "offices": [o.name for o in offices],
        "split": plan.split,
    }
    rows.truth_company = {
        "company_id": plan.company_id,
        "archetype": plan.archetype,
        "scope_kind": tree.scope_kind if tree.scopes else "none",
        "true_headcount": len(employees),
        "observed_employee_count": n_obs_employees,
        "observed_profile_count": len(observed),
        "coverage_target": plan.coverage,
        "coverage_actual": n_obs_employees / len(employees),
        "max_depth": max(depth.values()),
        "departments": tree.departments,
        "holdout_noise": holdout,
        "company_seed": plan.seed,
    }
    return rows
