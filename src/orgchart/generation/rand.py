"""Small helpers over numpy's Generator that return plain Python values."""

from collections.abc import Mapping, Sequence
from typing import TypeVar

import numpy as np

T = TypeVar("T")


def pick(rng: np.random.Generator, items: Sequence[T]) -> T:
    return items[int(rng.integers(len(items)))]


def weighted_pick(rng: np.random.Generator, weights: Mapping[T, float]) -> T:
    keys = list(weights)
    p = np.asarray([weights[k] for k in keys], dtype=float)
    return keys[int(rng.choice(len(keys), p=p / p.sum()))]


def chance(rng: np.random.Generator, p: float) -> bool:
    return bool(rng.random() < p)
