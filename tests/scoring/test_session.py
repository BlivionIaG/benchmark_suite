"""Tests for benchmark_suite.scoring.session — growing coding-agent turns."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import respx

from benchmark_suite.recipe import Recipe, SessionScorer
from benchmark_suite.scoring.base import ScoreStatus
from benchmark_suite.scoring.session import SessionScorerImpl
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


def _recipe(**scoring: object) -> Recipe:
    return Recipe.model_validate(
        {
            "meta": {"name": "session-test"},
            "backend": {"type": "external"},
            "endpoint": {"url": ENDPOINT, "model_name": "candidate"},
            "resources": {"max_model_len": 8192},
            "bench": {"scoring": [{"kind": "session", **scoring}]},
        }
    )


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
    assert isinstance(cfg, SessionScorer)
    rec = SessionScorerImpl(cfg).score(recipe, result_dir=tmp_path)
    assert rec.status == ScoreStatus.SUCCESS
    assert route.call_count == 3
    sizes: list[int] = []
    for call in route.calls:
        body = json.loads(call.request.content.decode())
        assert body["messages"][0]["role"] == "system"
        assert "coding agent" in body["messages"][0]["content"].lower()
        sizes.append(sum(len(m["content"]) for m in body["messages"]))
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
    assert isinstance(cfg, SessionScorer)
    rec = SessionScorerImpl(cfg).score(recipe, result_dir=tmp_path)
    assert rec.status == ScoreStatus.FAILURE
    assert rec.metrics.get("failed") == 1 or rec.error
