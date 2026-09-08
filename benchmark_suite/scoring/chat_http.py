"""POST /v1/chat/completions with optional SSE streaming for TTFT."""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, cast

import httpx


@dataclass
class ChatCompletionResult:
    """One chat completion attempt against an OpenAI-compatible endpoint."""

    prompt_id: str
    ok: bool
    text: str
    error: str | None
    ttft_ms: float
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int


def chat_completions_url(endpoint_url: str) -> str:
    """``{endpoint}/v1/chat/completions`` with a single ``/v1`` suffix."""
    u = endpoint_url.rstrip("/")
    base = u if u.endswith("/v1") else u + "/v1"
    return base + "/chat/completions"


def auth_headers(api_key: str) -> dict[str, str]:
    """Bearer header when a key is present; empty dict otherwise."""
    if not api_key:
        return {}
    return {"Authorization": f"Bearer {api_key}"}


def _usage_ints(usage: object) -> tuple[int, int]:
    if not isinstance(usage, dict):
        return 0, 0
    data = cast(dict[str, Any], usage)
    prompt = data.get("prompt_tokens", 0)
    completion = data.get("completion_tokens", 0)
    p = int(prompt) if isinstance(prompt, (int, float)) else 0
    c = int(completion) if isinstance(completion, (int, float)) else 0
    return p, c


def _choice_dict(body: dict[str, Any]) -> dict[str, Any] | None:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first: object = cast(list[object], choices)[0]
    if not isinstance(first, dict):
        return None
    return cast(dict[str, Any], first)


def _str_field(obj: dict[str, Any], key: str) -> str:
    value = obj.get(key)
    return value if isinstance(value, str) else ""


def _message_text(body: dict[str, Any]) -> str:
    first = _choice_dict(body)
    if first is None:
        return ""
    message = first.get("message")
    if isinstance(message, dict):
        content = _str_field(cast(dict[str, Any], message), "content")
        if content:
            return content
    return _str_field(first, "text")


def _delta_content(event: dict[str, Any]) -> str:
    first = _choice_dict(event)
    if first is None:
        return ""
    delta = first.get("delta")
    if isinstance(delta, dict):
        content = _str_field(cast(dict[str, Any], delta), "content")
        if content:
            return content
    message = first.get("message")
    if isinstance(message, dict):
        return _str_field(cast(dict[str, Any], message), "content")
    return ""


def post_chat_completion(
    client: httpx.Client,
    *,
    url: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    stream: bool,
    headers: dict[str, str],
    prompt_id: str,
) -> ChatCompletionResult:
    """One ``chat/completions`` call; never raises (errors land on ``ok=False``)."""
    payload: dict[str, object] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }
    if stream:
        payload["stream_options"] = {"include_usage": True}

    t0 = time.perf_counter()
    try:
        if stream:
            return _post_stream(
                client,
                url=url,
                payload=payload,
                headers=headers,
                prompt_id=prompt_id,
                t0=t0,
            )
        resp = client.post(url, json=payload, headers=headers)
        t1 = time.perf_counter()
        resp.raise_for_status()
        body = cast(dict[str, Any], resp.json())
        text = _message_text(body)
        prompt_toks, completion_toks = _usage_ints(body.get("usage"))
        if completion_toks <= 0 and text:
            completion_toks = max(1, len(text) // 4)
        if prompt_toks <= 0:
            prompt_toks = max(1, sum(len(m["content"]) for m in messages) // 4)
        latency_ms = (t1 - t0) * 1000.0
        return ChatCompletionResult(
            prompt_id=prompt_id,
            ok=True,
            text=text,
            error=None,
            ttft_ms=latency_ms,
            latency_ms=latency_ms,
            prompt_tokens=prompt_toks,
            completion_tokens=completion_toks,
        )
    except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as exc:
        t1 = time.perf_counter()
        return ChatCompletionResult(
            prompt_id=prompt_id,
            ok=False,
            text="",
            error=f"{type(exc).__name__}: {exc}",
            ttft_ms=(t1 - t0) * 1000.0,
            latency_ms=(t1 - t0) * 1000.0,
            prompt_tokens=0,
            completion_tokens=0,
        )


def _post_stream(
    client: httpx.Client,
    *,
    url: str,
    payload: dict[str, object],
    headers: dict[str, str],
    prompt_id: str,
    t0: float,
) -> ChatCompletionResult:
    parts: list[str] = []
    usage_obj: object = None
    first_token_at: float | None = None
    with client.stream("POST", url, json=payload, headers=headers) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line or not line.startswith("data:"):
                continue
            payload_line = line[5:].strip()
            if payload_line == "[DONE]":
                break
            try:
                event = json.loads(payload_line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            event_dict = cast(dict[str, Any], event)
            if event_dict.get("usage") is not None:
                usage_obj = event_dict.get("usage")
            chunk = _delta_content(event_dict)
            if chunk:
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                parts.append(chunk)
    t1 = time.perf_counter()
    text = "".join(parts)
    prompt_toks, completion_toks = _usage_ints(usage_obj)
    if completion_toks <= 0 and text:
        completion_toks = max(1, len(text) // 4)
    latency_ms = (t1 - t0) * 1000.0
    ttft_ms = (
        (first_token_at - t0) * 1000.0 if first_token_at is not None else latency_ms
    )
    return ChatCompletionResult(
        prompt_id=prompt_id,
        ok=True,
        text=text,
        error=None,
        ttft_ms=ttft_ms,
        latency_ms=latency_ms,
        prompt_tokens=prompt_toks,
        completion_tokens=completion_toks,
    )


def run_concurrent_wave(
    client: httpx.Client,
    *,
    url: str,
    model: str,
    items: list[tuple[str, list[dict[str, str]]]],
    max_tokens: int,
    temperature: float,
    stream: bool,
    headers: dict[str, str],
) -> list[ChatCompletionResult]:
    """Fire one chat completion per item with ``max_workers=len(items)``."""
    if not items:
        return []
    results: dict[int, ChatCompletionResult] = {}

    def _one(idx: int) -> ChatCompletionResult:
        prompt_id, messages = items[idx]
        return post_chat_completion(
            client,
            url=url,
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
            headers=headers,
            prompt_id=prompt_id,
        )

    workers = max(1, len(items))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_one, i): i for i in range(len(items))}
        for fut in as_completed(futs):
            idx = futs[fut]
            results[idx] = fut.result()
    return [results[i] for i in range(len(items))]
