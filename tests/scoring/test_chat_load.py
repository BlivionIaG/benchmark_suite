"""Tests for benchmark_suite.scoring.chat_load — concurrent diverse chat waves."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
import respx
from pydantic import ValidationError
from respx.models import Call

from benchmark_suite.recipe import ChatLoadScorer, Recipe
from benchmark_suite.scoring.base import ScoreStatus
from benchmark_suite.scoring.chat_load import ChatLoadScorerImpl

ENDPOINT = "http://127.0.0.1:8000"


def _ok_response() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": "ok " * 20}}],
            "usage": {"prompt_tokens": 80, "completion_tokens": 8},
        },
    )


def _recipe(**scoring: object) -> Recipe:
    return Recipe.model_validate(
        {
            "meta": {"name": "chat-load-test"},
            "backend": {"type": "external"},
            "endpoint": {"url": ENDPOINT, "model_name": "candidate"},
            "resources": {"max_model_len": 4096},
            "bench": {"scoring": [{"kind": "chat_load", **scoring}]},
        }
    )


def _bodies(route: respx.Route) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for call in cast(list[Call], list(route.calls)):
        data: object = json.loads(call.request.content.decode())
        assert isinstance(data, dict)
        out.append(cast(dict[str, Any], data))
    return out


def _first_message(body: dict[str, Any]) -> dict[str, Any]:
    messages = body["messages"]
    assert isinstance(messages, list)
    first: object = cast(list[object], messages)[0]
    assert isinstance(first, dict)
    return cast(dict[str, Any], first)


def test_chat_load_c16_posts_sixteen_distinct_systems(
    respx_mock: respx.MockRouter, tmp_path: Path
) -> None:
    route = respx_mock.post(f"{ENDPOINT}/v1/chat/completions").mock(
        return_value=_ok_response()
    )
    recipe = _recipe(
        stream=False,
        ladder=[16],
        suites=[{"name": "short", "input_tokens": 128, "output_tokens": 8}],
    )
    cfg = recipe.bench.scoring[0]
    assert isinstance(cfg, ChatLoadScorer)
    rec = ChatLoadScorerImpl(cfg).score(recipe, result_dir=tmp_path)
    assert rec.status == ScoreStatus.SUCCESS
    assert route.call_count == 16
    systems: list[str] = []
    for body in _bodies(route):
        assert body["model"] == "candidate"
        assert body["max_tokens"] == 8
        assert body["stream"] is False
        first = _first_message(body)
        assert first["role"] == "system"
        systems.append(str(first["content"]))
    assert len(set(systems)) == 16
    assert rec.metrics["successful"] == 16
    assert rec.metrics["concurrency"] == 16
    assert (tmp_path / "artifacts" / "chat_load_short_c16.json").is_file()


def test_chat_load_skips_suite_that_exceeds_max_model_len(
    respx_mock: respx.MockRouter, tmp_path: Path
) -> None:
    route = respx_mock.post(f"{ENDPOINT}/v1/chat/completions").mock(
        return_value=_ok_response()
    )
    recipe = _recipe(
        stream=False,
        ladder=[1],
        suites=[
            {"name": "short", "input_tokens": 128, "output_tokens": 8},
            {"name": "long", "input_tokens": 16384, "output_tokens": 1024},
        ],
    )
    cfg = recipe.bench.scoring[0]
    assert isinstance(cfg, ChatLoadScorer)
    rec = ChatLoadScorerImpl(cfg).score(recipe, result_dir=tmp_path)
    assert rec.status == ScoreStatus.SUCCESS
    assert route.call_count == 1
    skipped = rec.notes["skipped_suites"]
    assert isinstance(skipped, list)
    assert any("long" in str(item) for item in cast(list[object], skipped))
    suites = rec.notes["suites"]
    assert isinstance(suites, list)
    suite_names: list[str] = []
    for row in cast(list[object], suites):
        assert isinstance(row, dict)
        suite_names.append(str(cast(dict[str, Any], row)["name"]))
    assert suite_names == ["short"]


def test_chat_load_endpoint_failure_is_recorded(
    respx_mock: respx.MockRouter, tmp_path: Path
) -> None:
    respx_mock.post(f"{ENDPOINT}/v1/chat/completions").mock(
        return_value=httpx.Response(500, text="boom")
    )
    recipe = _recipe(
        stream=False,
        ladder=[1],
        suites=[{"name": "short", "input_tokens": 64, "output_tokens": 8}],
    )
    cfg = recipe.bench.scoring[0]
    assert isinstance(cfg, ChatLoadScorer)
    rec = ChatLoadScorerImpl(cfg).score(recipe, result_dir=tmp_path)
    assert rec.status == ScoreStatus.FAILURE
    assert rec.metrics.get("failed") == 1 or rec.error


def test_chat_load_unknown_ladder_rejected_by_schema() -> None:
    with pytest.raises(ValidationError):
        _recipe(ladder=[3])
