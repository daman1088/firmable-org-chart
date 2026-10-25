"""Person-level attributes: offices and locations, names, start dates, opaque IDs."""

from __future__ import annotations

import datetime as dt
from functools import lru_cache

import numpy as np
from faker import Faker

from orgchart.generation.catalog import COUNTRIES, City
from orgchart.generation.rand import chance, pick, weighted_pick
from orgchart.generation.structure import CompanyTree, Node
from orgchart.ladder import Level

REMOTE = "REMOTE"


# ----------------------------------------------------------------------------- offices / regions
def plan_offices(rng: np.random.Generator, country: str, hq: City, true_size: int) -> list[City]:
    if true_size < 300:
        n_extra = int(rng.choice([0, 1, 1, 2]))
    elif true_size < 1000:
        n_extra = int(rng.integers(1, 4))
    else:
        n_extra = int(rng.integers(2, 6))
    offices = [hq]
    domestic = [c for c in COUNTRIES[country].cities if c != hq]
    foreign = [c for code, ctry in COUNTRIES.items() if code != country for c in ctry.cities[:2]]
    for _ in range(n_extra):
        pool = domestic if domestic and chance(rng, 0.6) else foreign
        candidates = [c for c in pool if c not in offices]
        if candidates:
            offices.append(pick(rng, candidates))
    return offices


def regions_for(offices: list[City]) -> tuple[tuple[str, ...], dict[str, City]]:
    """Regions a regional company organises by: global regions if multinational, else states."""
    by_global: dict[str, City] = {}
    for office in offices:
        by_global.setdefault(COUNTRIES[office.country].global_region, office)
    if len(by_global) >= 2:
        return tuple(by_global), by_global
    by_state: dict[str, City] = {}
    for office in offices:
        by_state.setdefault(office.state, office)
    if len(by_state) >= 2:
        return tuple(by_state), by_state
    return (), {}


def format_location(rng: np.random.Generator, place: City | str, greater_area_p: float, home_country: str) -> str:
    if place == REMOTE:
        country = COUNTRIES[home_country].name
        return pick(rng, ("Remote", f"{country} (Remote)", country))
    assert isinstance(place, City)
    country = COUNTRIES[place.country].name
    if place.name == country:  # city-states
        return country
    if chance(rng, greater_area_p):
        return f"Greater {place.name} Area"
    if chance(rng, 0.05):
        return country
    return f"{place.name}, {place.state}, {country}"


def assign_locations(rng: np.random.Generator, tree: CompanyTree, offices: list[City],
                     region_offices: dict[str, City]) -> dict[int, City | str]:
    hq = offices[0]
    office_weights = {i: (0.5 if i == 0 else 0.5 / max(1, len(offices) - 1)) for i in range(len(offices))}

    def random_office() -> City:
        return offices[weighted_pick(rng, office_weights)]

    places: dict[int, City | str] = {}
    for node in tree.bfs_order():
        parent_place = places.get(node.parent) if node.parent is not None else None
        if node.role_type == "ceo":
            place: City | str = hq
        elif node.role_type == "head":
            place = hq if chance(rng, 0.8) else random_office()
        elif node.role_type in ("assistant", "chief_of_staff"):
            place = parent_place if parent_place != REMOTE else hq
        elif node.role_type == "manager":
            parent = tree.nodes[node.parent]
            if node.scope and node.scope != parent.scope and node.scope in region_offices:
                place = region_offices[node.scope]
            else:
                place = parent_place if chance(rng, 0.8) and parent_place != REMOTE else random_office()
        else:  # ic
            r = rng.random()
            if r < 0.85 and parent_place != REMOTE:
                place = parent_place
            elif r < 0.95:
                place = random_office()
            else:
                place = REMOTE
        places[node.nid] = place
    for node in tree.nodes:
        if node.role_type == "board":
            places[node.nid] = hq
    return places


# ----------------------------------------------------------------------------- names
@lru_cache(maxsize=None)
def _faker(locale: str) -> Faker:
    return Faker(locale)


def make_names(rng: np.random.Generator, country: str, n: int) -> list[str]:
    locales = dict(COUNTRIES[country].name_locales)
    base_seed = int(rng.integers(0, 2**31 - 1))
    fakers = {}
    for i, locale in enumerate(sorted(locales)):
        fake = _faker(locale)
        fake.seed_instance(base_seed + i)
        fakers[locale] = fake
    names = []
    for _ in range(n):
        fake = fakers[weighted_pick(rng, locales)]
        names.append(f"{fake.first_name()} {fake.last_name()}")
    return names


# ----------------------------------------------------------------------------- start dates / ids
def _tenure_mean_years(node: Node, ceo_is_founder: bool, company_age: float) -> float:
    if node.role_type == "board":
        return 4.0
    if node.role_type == "ceo":
        return company_age if ceo_is_founder else 6.0
    level = node.level or 0
    if level == Level.INTERN:
        return 0.4
    if level <= Level.SENIOR:
        return 2.2
    if level <= Level.SENIOR_MANAGER:
        return 3.5
    if level <= Level.SVP:
        return 4.5
    return 5.0


def start_date(rng: np.random.Generator, node: Node, snapshot: dt.date, founded_year: int, ceo_is_founder: bool) -> dt.date:
    company_age = max(1.0, snapshot.year - founded_year + 0.5)
    mean = _tenure_mean_years(node, ceo_is_founder, company_age)
    if node.role_type == "ceo" and ceo_is_founder:
        tenure = company_age - float(rng.uniform(0, 0.5))
    else:
        tenure = float(rng.gamma(2.0, mean / 2.0))
    tenure = min(max(tenure, 1 / 12), company_age)
    day = snapshot - dt.timedelta(days=int(tenure * 365.25))
    return day.replace(day=1)


def make_employee_ids(rng: np.random.Generator, n: int) -> list[str]:
    """Opaque random IDs: no ordering information that could leak the hierarchy."""
    ids: list[str] = []
    seen: set[str] = set()
    while len(ids) < n:
        candidate = f"E{int(rng.integers(0, 2**48)):012x}"
        if candidate not in seen:
            seen.add(candidate)
            ids.append(candidate)
    return ids
