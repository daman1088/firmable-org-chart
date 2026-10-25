"""Canonical seniority ladder shared by generation, inference and evaluation.

The ladder is vocabulary, not ground truth: inference maps parsed titles onto
these ranks, and evaluation compares those ranks with the true levels.
"""

from enum import IntEnum


class Level(IntEnum):
    INTERN = 0
    JUNIOR = 1
    INTERMEDIATE = 2
    SENIOR = 3
    LEAD = 4  # team lead (manager track) or staff/principal IC
    MANAGER = 5
    SENIOR_MANAGER = 6
    DIRECTOR = 7
    SENIOR_DIRECTOR = 8
    VP = 9
    SVP = 10
    C_LEVEL = 11
    CEO = 12


LEVEL_NAMES: dict[int, str] = {lvl.value: lvl.name.lower() for lvl in Level}

# Lowest level that can hold people-manager titles.
MIN_MANAGER_LEVEL = Level.LEAD
