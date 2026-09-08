"""Tests for the sauce session phase — growing coding-agent turns."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import httpx
import respx
from respx.models import Call

from benchmark_suite.recipe import Recipe, SauceScorer
from benchmark_suite.scoring.base import ScoreStatus
from benchmark_suite.scoring.sauce import SauceScorerImpl
from benchmark_suite.workloads.session_catalog import plan_turn_input_targets

ENDPOINT = "http://127.0.0.1:8000"


def _ok_response() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": "patched module_1.py"}}],
            "usage": {"prompt_tokens": 200, "completion_tokens": 12},
        },
    )


def _recipe(**session: object) -> Recipe:
    return Recipe.model_validate(
        {
            "meta": {"name": "sauce-session-test"},
            "backend": {"type": "external"},
            "endpoint": {"url": ENDPOINT, "model_name": "candidate"},
            "resources": {"max_model_len": 8192},
            "bench": {
                "scoring": [
                    {
                        "kind": "sauce",
                        "chat": {"enabled": False},
                        "session": session,
                    }
                ]
            },
        }
    )


def _bodies(route: respx.Route) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for call in cast(list[Call], list(route.calls)):
        data: object = json.loads(call.request.content.decode())
        assert isinstance(data, dict)
        out.append(cast(dict[str, Any], data))
    return out


def test_session_messages_grow_and_keep_system_prompt(
    respx_mock: respx.MockRouter, tmp_path: Path
) -> None:
    route = respx_mock.post(f"{ENDPOINT}/v1/chat/completions").mock(
        return_value=_ok_response()
    )
    recipe = _recipe(
        stream=False,
        n_sessions=1,
        max_turns=3,
        output_tokens=16,
        output_reserve_tokens=256,
        max_context_tokens=200000,
    )
    cfg = recipe.bench.scoring[0]
    assert isinstance(cfg, SauceScorer)
    rec = SauceScorerImpl(cfg).score(recipe, result_dir=tmp_path)
    assert rec.kind == "sauce"
    assert rec.status == ScoreStatus.SUCCESS
    assert route.call_count == 3
    sizes: list[int] = []
    for body in _bodies(route):
        messages = body["messages"]
        assert isinstance(messages, list)
        typed = cast(list[object], messages)
        first = typed[0]
        assert isinstance(first, dict)
        first_d = cast(dict[str, Any], first)
        assert first_d["role"] == "system"
        assert "coding agent" in str(first_d["content"]).lower()
        total = 0
        for message in typed:
            assert isinstance(message, dict)
            total += len(str(cast(dict[str, Any], message)["content"]))
        sizes.append(total)
        assert body["max_tokens"] == 16
    assert sizes[0] < sizes[1] < sizes[2]
    assert rec.metrics["session_turns"] == 3
    assert rec.metrics["session_success_rate"] == 1.0
    assert rec.metrics["successful"] == 1
    assert (tmp_path / "artifacts" / "session_sessions.json").is_file()


def test_session_small_context_plans_fewer_turns_than_200k() -> None:
    small = plan_turn_input_targets(budget=6000)
    large = plan_turn_input_targets(budget=197952)
    assert len(small) < len(large)
    recipe_small = _recipe(
        stream=False,
        n_sessions=1,
        output_tokens=16,
        output_reserve_tokens=256,
        max_context_tokens=200000,
    )
    assert recipe_small.resources.max_model_len == 8192


def test_session_http_failure_marks_session_failed(
    respx_mock: respx.MockRouter, tmp_path: Path
) -> None:
    respx_mock.post(f"{ENDPOINT}/v1/chat/completions").mock(
        return_value=httpx.Response(503, text="nope")
    )
    recipe = _recipe(
        stream=False,
        n_sessions=1,
        max_turns=2,
        output_tokens=16,
        output_reserve_tokens=256,
    )
    cfg = recipe.bench.scoring[0]
    assert isinstance(cfg, SauceScorer)
    rec = SauceScorerImpl(cfg).score(recipe, result_dir=tmp_path)
    assert rec.status == ScoreStatus.FAILURE
    assert rec.metrics.get("failed") == 1 or rec.error
