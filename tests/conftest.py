"""Shared fixtures for benchmark_suite tests."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml


@pytest.fixture
def valid_recipe_dict() -> dict[str, Any]:
    """Dict form of a minimal valid recipe (meta only; everything else defaults)."""
    return {"meta": {"name": "test-recipe", "description": "minimal valid recipe"}}


@pytest.fixture
def tmp_recipe_path(tmp_path: Path, valid_recipe_dict: dict[str, Any]) -> Path:
    """Write the minimal valid recipe to a YAML file and return its path."""
    p = tmp_path / "recipe.yaml"
    p.write_text(yaml.safe_dump(valid_recipe_dict))
    return p