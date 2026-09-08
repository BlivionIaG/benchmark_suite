"""Tests for the shared /v1/chat/completions client used by kind=sauce."""

from __future__ import annotations

import json
from typing import Any, cast

import httpx
import respx
from respx.models import Call

from benchmark_suite.scoring.chat_http import (
    auth_headers,
    chat_completions_url,
    post_chat_completion,
    run_concurrent_wave,
)

URL = "http://127.0.0.1:8000/v1/chat/completions"


def test_chat_completions_url_appends_v1_once() -> None:
    assert (
        chat_completions_url("http://127.0.0.1:8000")
        == "http://127.0.0.1:8000/v1/chat/completions"
    )
    assert (
        chat_completions_url("http://127.0.0.1:8000/")
        == "http://127.0.0.1:8000/v1/chat/completions"
    )
    assert (
        chat_completions_url("http://127.0.0.1:8000/v1")
        == "http://127.0.0.1:8000/v1/chat/completions"
    )


def test_auth_headers_only_when_key_present() -> None:
    assert auth_headers("") == {}
    assert auth_headers("sk-test") == {"Authorization": "Bearer sk-test"}


def test_post_chat_completion_success(respx_mock: respx.MockRouter) -> None:
    respx_mock.post(URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "hello"}}],
                "usage": {"prompt_tokens": 8, "completion_tokens": 2},
            },
        )
    )
    with httpx.Client() as client:
        rec = post_chat_completion(
            client,
            url=URL,
            model="candidate",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=8,
            temperature=0.2,
            stream=False,
            headers={},
            prompt_id="p1",
        )
    assert rec.ok is True
    assert rec.text == "hello"
    assert rec.prompt_tokens == 8
    assert rec.completion_tokens == 2
    assert rec.prompt_id == "p1"
    assert rec.cached_tokens is None
    assert rec.started_unix_ms > 0


def test_post_chat_completion_http_error_is_not_raised(
    respx_mock: respx.MockRouter,
) -> None:
    respx_mock.post(URL).mock(return_value=httpx.Response(500, text="boom"))
    with httpx.Client() as client:
        rec = post_chat_completion(
            client,
            url=URL,
            model="candidate",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=8,
            temperature=0.7,
            stream=False,
            headers={},
            prompt_id="p1",
        )
    assert rec.ok is False
    assert rec.error is not None
    assert rec.text == ""


def test_post_chat_completion_stream_reads_sse(respx_mock: respx.MockRouter) -> None:
    sse = (
        'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n'
        'data: {"usage":{"prompt_tokens":4,"completion_tokens":2}}\n\n'
        "data: [DONE]\n\n"
    )
    route = respx_mock.post(URL).mock(
        return_value=httpx.Response(
            200, content=sse.encode(), headers={"content-type": "text/event-stream"}
        )
    )
    with httpx.Client() as client:
        rec = post_chat_completion(
            client,
            url=URL,
            model="candidate",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=8,
            temperature=0.7,
            stream=True,
            headers={},
            prompt_id="p1",
        )
    assert rec.ok is True
    assert rec.text == "Hello"
    assert rec.completion_tokens == 2
    calls = cast(list[Call], list(route.calls))
    body: object = json.loads(calls[0].request.content.decode())
    assert isinstance(body, dict)
    typed = cast(dict[str, Any], body)
    assert typed["stream"] is True
    assert typed["stream_options"] == {"include_usage": True}


def test_run_concurrent_wave_preserves_item_order(respx_mock: respx.MockRouter) -> None:
    respx_mock.post(URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )
    )
    items = [
        ("a", [{"role": "user", "content": "a"}]),
        ("b", [{"role": "user", "content": "b"}]),
        ("c", [{"role": "user", "content": "c"}]),
    ]
    with httpx.Client() as client:
        results = run_concurrent_wave(
            client,
            url=URL,
            model="candidate",
            items=items,
            max_tokens=4,
            temperature=0.0,
            stream=False,
            headers={},
        )
    assert [r.prompt_id for r in results] == ["a", "b", "c"]
    assert all(r.ok for r in results)


def test_post_chat_completion_reads_cached_tokens(respx_mock: respx.MockRouter) -> None:
    respx_mock.post(URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "hello"}}],
                "usage": {
                    "prompt_tokens": 40,
                    "completion_tokens": 2,
                    "prompt_tokens_details": {"cached_tokens": 17},
                },
            },
        )
    )
    with httpx.Client() as client:
        rec = post_chat_completion(
            client,
            url=URL,
            model="candidate",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=8,
            temperature=0.2,
            stream=False,
            headers={},
            prompt_id="p1",
        )
    assert rec.ok is True
    assert rec.cached_tokens == 17
