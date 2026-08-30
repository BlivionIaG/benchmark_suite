"""benchmark_suite/_matrix.py — matrix axis expansion (cartesian product over dotted paths)."""
from __future__ import annotations

import copy
from itertools import product
from typing import NamedTuple

from benchmark_suite.recipe import Recipe

AxisValue = str | int | bool


class MatrixAxis(NamedTuple):
    path: tuple[str, ...]
    values: list[AxisValue]


def _set_dotted(obj: object, path: tuple[str, ...], value: AxisValue) -> None:
    """Set a value at a dotted path (e.g. ``cell.attn`` or ``runtime.env.VLLM_X``)."""
    *parents, leaf = path
    for seg in parents:
        obj = getattr(obj, seg)
    if isinstance(obj, dict):
        obj[leaf] = value
    else:
        setattr(obj, leaf, value)


def expand_matrix(recipe: Recipe, axes: list[MatrixAxis]) -> list[Recipe]:
    """Return one deep-copied Recipe variant per cartesian combination of axis values."""
    if not axes:
        return [recipe]
    variants: list[Recipe] = []
    for combo in product(*(axis.values for axis in axes)):
        variant = copy.deepcopy(recipe)
        for axis, value in zip(axes, combo, strict=True):
            _set_dotted(variant, axis.path, value)
        variants.append(variant)
    return variants