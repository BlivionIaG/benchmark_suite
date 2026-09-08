"""Shared fixtures for benchmark_suite tests."""
from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml

# Rich/typer colorize dash-separated option names (`--lmx-bin` becomes
# `-` + `-lmx` + `-bin` with SGR codes between). Help assertions must strip them.
_ANSI_SGR = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    """Remove ANSI SGR sequences from CLI output."""
    return _ANSI_SGR.sub("", text)


@pytest.fixture
def strip_ansi() -> Callable[[str], str]:
    return _strip_ansi


@pytest.fixture(autouse=True)
def _mock_kv_metrics_endpoint(request: pytest.FixtureRequest) -> None:
    """Sauce polls GET /metrics after waves/turns; default 404 so KV is omitted.

    Tests that need a real KV sample register a more specific route afterwards
    (respx matches last-registered first).
    """
    if "respx_mock" not in request.fixturenames:
        return
    router = request.getfixturevalue("respx_mock")
    router.get(url__regex=r"https?://[^/]+/metrics$").mock(
        return_value=httpx.Response(404, text="")
    )


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