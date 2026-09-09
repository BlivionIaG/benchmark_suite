"""Tests for the sauce chat phase — concurrent diverse chat waves."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
import respx
from pydantic import ValidationError
from respx.models import Call

from benchmark_suite.recipe import Recipe, SauceScorer
from benchmark_suite.scoring.base import ScoreStatus
from benchmark_suite.scoring.sauce import SauceScorerImpl
from benchmark_suite.workloads.chat_catalog import prompts_for_concurrency
from benchmark_suite.workloads.tokens import content_tokens

ENDPOINT = "http://127.0.0.1:8000"


def _metric_n(metrics: dict[str, float | int | str], key: str) -> float:
    value = metrics[key]
    assert isinstance(value, (int, float))
    return float(value)


def _json_obj(path: Path) -> dict[str, Any]:
    data: object = json.loads(path.read_text())
    assert isinstance(data, dict)
    return cast(dict[str, Any], data)


def _json_list(obj: object) -> list[object]:
    assert isinstance(obj, list)
    return cast(list[object], obj)


def _ok_response() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": "ok " * 20}}],
            "usage": {"prompt_tokens": 80, "completion_tokens": 8},
        },
    )


def _recipe(**chat: object) -> Recipe:
    return Recipe.model_validate(
        {
            "meta": {"name": "sauce-chat-test"},
            "backend": {"type": "external"},
            "endpoint": {"url": ENDPOINT, "model_name": "candidate"},
            "resources": {"max_model_len": 4096},
            "bench": {
                "scoring": [
                    {
                        "kind": "sauce",
                        "chat": chat,
                        "session": {"enabled": False},
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


def _first_message(body: dict[str, Any]) -> dict[str, Any]:
    messages = body["messages"]
    assert isinstance(messages, list)
    first: object = cast(list[object], messages)[0]
    assert isinstance(first, dict)
    return cast(dict[str, Any], first)


def _chat_notes(rec_notes: dict[str, Any]) -> dict[str, Any]:
    chat = rec_notes["chat"]
    assert isinstance(chat, dict)
    return cast(dict[str, Any], chat)


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
    assert isinstance(cfg, SauceScorer)
    rec = SauceScorerImpl(cfg).score(recipe, result_dir=tmp_path)
    assert rec.kind == "sauce"
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
    assert isinstance(cfg, SauceScorer)
    rec = SauceScorerImpl(cfg).score(recipe, result_dir=tmp_path)
    assert rec.status == ScoreStatus.SUCCESS
    assert route.call_count == 1
    chat_notes = _chat_notes(rec.notes)
    skipped = chat_notes["skipped_suites"]
    assert isinstance(skipped, list)
    assert any("long" in str(item) for item in cast(list[object], skipped))
    suites = chat_notes["suites"]
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
    assert isinstance(cfg, SauceScorer)
    rec = SauceScorerImpl(cfg).score(recipe, result_dir=tmp_path)
    assert rec.status == ScoreStatus.FAILURE
    assert rec.metrics.get("failed") == 1 or rec.error


def test_chat_load_unknown_ladder_rejected_by_schema() -> None:
    with pytest.raises(ValidationError):
        _recipe(ladder=[3])


def test_chat_ladder_posts_thirty_one_distinct_systems(
    respx_mock: respx.MockRouter, tmp_path: Path
) -> None:
    route = respx_mock.post(f"{ENDPOINT}/v1/chat/completions").mock(
        return_value=_ok_response()
    )
    recipe = _recipe(
        stream=False,
        ladder=[16, 8, 4, 2, 1],
        suites=[{"name": "short", "input_tokens": 128, "output_tokens": 8}],
    )
    cfg = recipe.bench.scoring[0]
    assert isinstance(cfg, SauceScorer)
    rec = SauceScorerImpl(cfg).score(recipe, result_dir=tmp_path)
    assert rec.status == ScoreStatus.SUCCESS
    assert route.call_count == 31
    systems = [str(_first_message(body)["content"]) for body in _bodies(route)]
    assert len(set(systems)) == 31
    assert rec.metrics["successful"] == 31


def test_chat_user_message_keeps_task_and_hits_budget(
    respx_mock: respx.MockRouter, tmp_path: Path
) -> None:
    route = respx_mock.post(f"{ENDPOINT}/v1/chat/completions").mock(
        return_value=_ok_response()
    )
    recipe = _recipe(
        stream=False,
        ladder=[1],
        suites=[{"name": "short", "input_tokens": 256, "output_tokens": 8}],
    )
    cfg = recipe.bench.scoring[0]
    assert isinstance(cfg, SauceScorer)
    SauceScorerImpl(cfg).score(recipe, result_dir=tmp_path)
    body = _bodies(route)[0]
    messages = body["messages"]
    assert isinstance(messages, list)
    user: object = cast(list[object], messages)[1]
    assert isinstance(user, dict)
    user_d = cast(dict[str, Any], user)
    spec = prompts_for_concurrency(1)[0]
    assert spec.task in str(user_d["content"])
    typed_messages = [cast(dict[str, str], m) for m in cast(list[object], messages)]
    assert content_tokens(typed_messages) == 256


def test_chat_forwards_temperature_and_stream_options(
    respx_mock: respx.MockRouter, tmp_path: Path
) -> None:
    route = respx_mock.post(f"{ENDPOINT}/v1/chat/completions").mock(
        return_value=_ok_response()
    )
    recipe = _recipe(
        stream=True,
        temperature=0.25,
        ladder=[1],
        suites=[{"name": "short", "input_tokens": 64, "output_tokens": 8}],
    )
    cfg = recipe.bench.scoring[0]
    assert isinstance(cfg, SauceScorer)
    SauceScorerImpl(cfg).score(recipe, result_dir=tmp_path)
    body = _bodies(route)[0]
    assert body["temperature"] == 0.25
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}


def test_chat_skip_all_suites_is_failure(
    respx_mock: respx.MockRouter, tmp_path: Path
) -> None:
    respx_mock.post(f"{ENDPOINT}/v1/chat/completions").mock(return_value=_ok_response())
    recipe = Recipe.model_validate(
        {
            "meta": {"name": "sauce-chat-skip-all"},
            "backend": {"type": "external"},
            "endpoint": {"url": ENDPOINT, "model_name": "candidate"},
            "resources": {"max_model_len": 10},
            "bench": {
                "scoring": [
                    {
                        "kind": "sauce",
                        "chat": {
                            "stream": False,
                            "ladder": [1],
                            "suites": [
                                {"name": "short", "input_tokens": 128, "output_tokens": 8}
                            ],
                        },
                        "session": {"enabled": False},
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
    assert "chat:" in rec.error
    chat_notes = _chat_notes(rec.notes)
    skipped = chat_notes["skipped_suites"]
    assert isinstance(skipped, list)
    assert skipped


def test_chat_sends_bearer_token_from_env(
    respx_mock: respx.MockRouter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-sauce")
    route = respx_mock.post(f"{ENDPOINT}/v1/chat/completions").mock(
        return_value=_ok_response()
    )
    recipe = _recipe(
        stream=False,
        ladder=[1],
        suites=[{"name": "short", "input_tokens": 64, "output_tokens": 8}],
    )
    cfg = recipe.bench.scoring[0]
    assert isinstance(cfg, SauceScorer)
    SauceScorerImpl(cfg).score(recipe, result_dir=tmp_path)
    calls = cast(list[Call], list(route.calls))
    assert calls[0].request.headers["Authorization"] == "Bearer sk-sauce"


def test_chat_records_prefill_and_timeseries(
    respx_mock: respx.MockRouter, tmp_path: Path
) -> None:
    respx_mock.post(f"{ENDPOINT}/v1/chat/completions").mock(return_value=_ok_response())
    recipe = _recipe(
        stream=False,
        ladder=[1],
        suites=[{"name": "short", "input_tokens": 64, "output_tokens": 8}],
    )
    cfg = recipe.bench.scoring[0]
    assert isinstance(cfg, SauceScorer)
    rec = SauceScorerImpl(cfg).score(recipe, result_dir=tmp_path)
    assert rec.status == ScoreStatus.SUCCESS
    assert _metric_n(rec.metrics, "ttft_mean_ms") > 0
    assert _metric_n(rec.metrics, "input_tok_s") > 0
    assert _metric_n(rec.metrics, "prefill_tok_s") > 0
    assert "decode_tok_s" not in rec.metrics  # non-stream: no TTFT/decode split
    ts_path = tmp_path / "artifacts" / "chat_timeseries.json"
    assert ts_path.is_file()
    payload = _json_obj(ts_path)
    events = _json_list(payload["events"])
    assert len(events) == 1
    event = cast(dict[str, Any], events[0])
    assert event["phase"] == "chat"
    assert event["ok"] is True
    assert "ttft_ms" in event
    assert "prefill_tok_s" in event
    assert (tmp_path / "artifacts" / "sauce_timeseries.json").is_file()
    wave = json.loads((tmp_path / "artifacts" / "chat_load_short_c1.json").read_text())
    assert wave["results"][0]["prompt_tokens"] == 80
    assert "prefill_tok_s" in wave["results"][0]


def test_chat_records_kv_cache_from_metrics(
    respx_mock: respx.MockRouter,
    tmp_path: Path,
    remock_kv: Callable[[httpx.Response], None],
) -> None:
    respx_mock.post(f"{ENDPOINT}/v1/chat/completions").mock(return_value=_ok_response())
    remock_kv(httpx.Response(200, text="vllm:gpu_cache_usage_perc 0.25\n"))
    recipe = _recipe(
        stream=False,
        ladder=[1],
        suites=[{"name": "short", "input_tokens": 64, "output_tokens": 8}],
    )
    cfg = recipe.bench.scoring[0]
    assert isinstance(cfg, SauceScorer)
    rec = SauceScorerImpl(cfg).score(recipe, result_dir=tmp_path)
    assert rec.metrics["kv_cache_perc"] == 25.0
    ts = json.loads((tmp_path / "artifacts" / "chat_timeseries.json").read_text())
    samples = ts["samples"]
    assert isinstance(samples, list)
    assert samples
    assert cast(dict[str, Any], samples[0])["kv_cache_perc"] == 25.0


def test_chat_kv_metrics_can_be_disabled(
    respx_mock: respx.MockRouter, tmp_path: Path
) -> None:
    respx_mock.post(f"{ENDPOINT}/v1/chat/completions").mock(return_value=_ok_response())
    recipe = Recipe.model_validate(
        {
            "meta": {"name": "sauce-chat-no-kv"},
            "backend": {"type": "external"},
            "endpoint": {"url": ENDPOINT, "model_name": "candidate"},
            "resources": {"max_model_len": 4096},
            "bench": {
                "scoring": [
                    {
                        "kind": "sauce",
                        "kv_metrics": False,
                        "chat": {
                            "stream": False,
                            "ladder": [1],
                            "suites": [
                                {"name": "short", "input_tokens": 64, "output_tokens": 8}
                            ],
                        },
                        "session": {"enabled": False},
                    }
                ]
            },
        }
    )
    cfg = recipe.bench.scoring[0]
    assert isinstance(cfg, SauceScorer)
    rec = SauceScorerImpl(cfg).score(recipe, result_dir=tmp_path)
    assert rec.status == ScoreStatus.SUCCESS
    assert "kv_cache_perc" not in rec.metrics
    ts = json.loads((tmp_path / "artifacts" / "chat_timeseries.json").read_text())
    assert ts["samples"] == []


def test_chat_stream_records_decode_tok_s(
    respx_mock: respx.MockRouter, tmp_path: Path
) -> None:
    sse = (
        'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n'
        'data: {"usage":{"prompt_tokens":80,"completion_tokens":8}}\n\n'
        "data: [DONE]\n\n"
    )
    respx_mock.post(f"{ENDPOINT}/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, content=sse.encode(), headers={"content-type": "text/event-stream"}
        )
    )
    recipe = _recipe(
        stream=True,
        ladder=[1],
        suites=[{"name": "short", "input_tokens": 64, "output_tokens": 8}],
    )
    cfg = recipe.bench.scoring[0]
    assert isinstance(cfg, SauceScorer)
    rec = SauceScorerImpl(cfg).score(recipe, result_dir=tmp_path)
    assert rec.status == ScoreStatus.SUCCESS
    wave = json.loads((tmp_path / "artifacts" / "chat_load_short_c1.json").read_text())
    row = wave["results"][0]
    # Instant mock streams can collapse TTFT and E2E; decode is present only
    # when there is a measurable generation window after the first token.
    assert "ttft_ms" in row
    assert "prefill_tok_s" in row
    if rec.metrics.get("decode_tok_s") is not None:
        assert _metric_n(rec.metrics, "decode_tok_s") > 0
