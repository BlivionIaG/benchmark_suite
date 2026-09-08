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
from benchmark_suite.workloads.session_catalog import (
    SESSION_SPECS,
    plan_turn_input_targets,
    session_followups,
)

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


def test_session_keeps_assistant_reply_and_sends_followup(
    respx_mock: respx.MockRouter, tmp_path: Path
) -> None:
    route = respx_mock.post(f"{ENDPOINT}/v1/chat/completions").mock(
        return_value=_ok_response()
    )
    recipe = _recipe(
        stream=False,
        n_sessions=1,
        max_turns=2,
        output_tokens=16,
        output_reserve_tokens=256,
        temperature=0.15,
    )
    cfg = recipe.bench.scoring[0]
    assert isinstance(cfg, SauceScorer)
    rec = SauceScorerImpl(cfg).score(recipe, result_dir=tmp_path)
    assert rec.status == ScoreStatus.SUCCESS
    bodies = _bodies(route)
    assert len(bodies) == 2
    assert bodies[0]["temperature"] == 0.15
    second = bodies[1]["messages"]
    assert isinstance(second, list)
    typed = cast(list[object], second)
    roles = [
        str(cast(dict[str, Any], m)["role"])
        for m in typed
        if isinstance(m, dict)
    ]
    assert roles[0] == "system"
    assert "assistant" in roles
    assistant_text = [
        str(cast(dict[str, Any], m)["content"])
        for m in typed
        if isinstance(m, dict) and cast(dict[str, Any], m)["role"] == "assistant"
    ]
    assert any("patched module_1.py" in text for text in assistant_text)
    spec = SESSION_SPECS[0]
    last = typed[-1]
    assert isinstance(last, dict)
    assert spec.feature in str(cast(dict[str, Any], last)["content"])
    assert spec.feature in session_followups(spec)[0]


def test_session_n_sessions_two_runs_two_conversations(
    respx_mock: respx.MockRouter, tmp_path: Path
) -> None:
    respx_mock.post(f"{ENDPOINT}/v1/chat/completions").mock(
        return_value=_ok_response()
    )
    recipe = _recipe(
        stream=False,
        n_sessions=2,
        max_turns=1,
        output_tokens=16,
        output_reserve_tokens=256,
    )
    cfg = recipe.bench.scoring[0]
    assert isinstance(cfg, SauceScorer)
    rec = SauceScorerImpl(cfg).score(recipe, result_dir=tmp_path)
    assert rec.status == ScoreStatus.SUCCESS
    assert rec.metrics["successful"] == 2
    session_notes_obj: object = rec.notes["session"]
    assert isinstance(session_notes_obj, dict)
    session_notes = cast(dict[str, Any], session_notes_obj)
    sessions_obj: object = session_notes["sessions"]
    assert isinstance(sessions_obj, list)
    ids = [
        str(cast(dict[str, Any], row)["session_id"])
        for row in cast(list[object], sessions_obj)
        if isinstance(row, dict)
    ]
    assert ids == [SESSION_SPECS[0].session_id, SESSION_SPECS[1].session_id]


def test_session_budget_too_small_fails(tmp_path: Path) -> None:
    recipe = Recipe.model_validate(
        {
            "meta": {"name": "sauce-session-tiny"},
            "backend": {"type": "external"},
            "endpoint": {"url": ENDPOINT, "model_name": "candidate"},
            "resources": {"max_model_len": 200},
            "bench": {
                "scoring": [
                    {
                        "kind": "sauce",
                        "chat": {"enabled": False},
                        "session": {
                            "stream": False,
                            "n_sessions": 1,
                            "output_tokens": 1024,
                            "output_reserve_tokens": 2048,
                        },
                    }
                ]
            },
        }
    )
    cfg = recipe.bench.scoring[0]
    assert isinstance(cfg, SauceScorer)
    rec = SauceScorerImpl(cfg).score(recipe, result_dir=tmp_path)
    assert rec.status == ScoreStatus.FAILURE
    assert rec.error is not None
    assert "session:" in rec.error
    assert "too small" in rec.error


def test_session_small_max_model_len_plans_fewer_turns(
    respx_mock: respx.MockRouter, tmp_path: Path
) -> None:
    respx_mock.post(f"{ENDPOINT}/v1/chat/completions").mock(
        return_value=_ok_response()
    )

    def _score(max_model_len: int) -> list[object]:
        recipe = Recipe.model_validate(
            {
                "meta": {"name": f"sauce-session-len-{max_model_len}"},
                "backend": {"type": "external"},
                "endpoint": {"url": ENDPOINT, "model_name": "candidate"},
                "resources": {"max_model_len": max_model_len},
                "bench": {
                    "scoring": [
                        {
                            "kind": "sauce",
                            "chat": {"enabled": False},
                            "session": {
                                "stream": False,
                                "n_sessions": 1,
                                "max_context_tokens": 200000,
                                "output_tokens": 16,
                                "output_reserve_tokens": 256,
                            },
                        }
                    ]
                },
            }
        )
        cfg = recipe.bench.scoring[0]
        assert isinstance(cfg, SauceScorer)
        rec = SauceScorerImpl(cfg).score(recipe, result_dir=tmp_path / str(max_model_len))
        session_notes_obj: object = rec.notes["session"]
        assert isinstance(session_notes_obj, dict)
        session_notes = cast(dict[str, Any], session_notes_obj)
        targets_obj: object = session_notes["turn_targets"]
        assert isinstance(targets_obj, list)
        return cast(list[object], targets_obj)

    small = _score(4096)
    large = _score(200000)
    assert len(small) < len(large)


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


def test_session_records_per_turn_latency_and_timeseries(
    respx_mock: respx.MockRouter, tmp_path: Path
) -> None:
    respx_mock.post(f"{ENDPOINT}/v1/chat/completions").mock(return_value=_ok_response())
    respx_mock.get(f"{ENDPOINT}/metrics").mock(
        return_value=httpx.Response(200, text="vllm:gpu_cache_usage_perc 0.4\n")
    )
    recipe = _recipe(
        stream=False,
        n_sessions=1,
        max_turns=3,
        output_tokens=16,
        output_reserve_tokens=256,
    )
    cfg = recipe.bench.scoring[0]
    assert isinstance(cfg, SauceScorer)
    rec = SauceScorerImpl(cfg).score(recipe, result_dir=tmp_path)
    assert rec.status == ScoreStatus.SUCCESS
    assert rec.metrics["session_turns"] == 3
    assert rec.metrics["session_ttft_mean_ms"] > 0
    assert rec.metrics["session_prefill_tok_s"] > 0
    assert rec.metrics["session_kv_cache_perc"] == 40.0
    assert rec.metrics["kv_cache_perc"] == 40.0
    data = json.loads((tmp_path / "artifacts" / "session_sessions.json").read_text())
    turns = data["sessions"][0]["turns_detail"]
    assert isinstance(turns, list)
    assert len(turns) == 3
    first = cast(dict[str, Any], turns[0])
    assert first["turn"] == 0
    assert "ttft_ms" in first
    assert "prefill_tok_s" in first
    assert first["kv_cache_perc"] == 40.0
    ts = json.loads((tmp_path / "artifacts" / "session_timeseries.json").read_text())
    events = ts["events"]
    assert isinstance(events, list)
    assert len(events) == 3
    assert cast(dict[str, Any], events[0])["phase"] == "session"
    assert cast(dict[str, Any], events[1])["turn"] == 1
    sauce_ts = json.loads((tmp_path / "artifacts" / "sauce_timeseries.json").read_text())
    assert len(sauce_ts["events"]) == 3
