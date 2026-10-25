"""Parse public-profile location strings into (city, country, remote).

Handles the formats seen on profiles: "City, Region, Country", "Greater City
Area", a bare country, "Remote" and "Country (Remote)". A city-state such as
Singapore is both city and country.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

COUNTRIES = {
    "australia", "united states", "united kingdom", "india", "singapore", "canada", "new zealand", "ireland",
    "germany", "france", "netherlands", "japan", "hong kong", "philippines", "malaysia", "indonesia",
}
CITY_STATES = {"singapore", "hong kong"}
_GREATER = re.compile(r"^greater (.+?) area$")
_REMOTE_SUFFIX = re.compile(r"^(.*?)\s*\(remote\)$")


@dataclass(frozen=True)
class Location:
    city: str | None = None
    country: str | None = None
    remote: bool = False


@lru_cache(maxsize=4096)
def parse_location(raw: object) -> Location:
    if not isinstance(raw, str) or not raw.strip():
        return Location()
    text = re.sub(r"\s+", " ", raw.strip().lower())
    if text in ("remote", "anywhere", "worldwide"):
        return Location(remote=True)
    m = _REMOTE_SUFFIX.match(text)
    if m:
        inner = parse_location(m.group(1))
        return Location(city=inner.city, country=inner.country, remote=True)
    m = _GREATER.match(text)
    if m:
        return Location(city=m.group(1))
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) >= 2:
        country = parts[-1] if parts[-1] in COUNTRIES or len(parts) >= 3 else None
        return Location(city=parts[0], country=country)
    single = parts[0]
    if single in CITY_STATES:
        return Location(city=single, country=single)
    if single in COUNTRIES:
        return Location(country=single)
    return Location(city=single)
