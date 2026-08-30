"""benchmark_suite/runner/endpoint.py — OpenAI endpoint capability probe."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, cast

import httpx

_TRANSIENT_STATUS = frozenset({502, 503, 504})
_MAX_BACKOFF_S = 30.0


@dataclass(frozen=True)
class EndpointCapabilities:
    """What this endpoint supports."""

    reachable: bool
    served_models: tuple[str, ...]
    requested_model_served: bool
    logprobs_supported: bool
    server_version: str = ""
    raw_health_json: dict[str, Any] | None = None


def _base_v1(url: str) -> str:
    u = url.rstrip("/")
    return u if u.endswith("/v1") else u + "/v1"


def _health_url(url: str) -> str:
    return url.rstrip("/") + "/health"


def _safe_json(resp: httpx.Response) -> dict[str, Any] | None:
    try:
        data: Any = resp.json()
    except ValueError:
        return None
    return cast(dict[str, Any], data) if isinstance(data, dict) else None


def _server_version(resp: httpx.Response) -> str:
    for header in ("x-vllm-version", "x-server-version", "server"):
        value = resp.headers.get(header)
        if value:
            return value
    return ""


def _request_with_retry(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    headers: dict[str, str],
    json_body: dict[str, object] | None = None,
    timeout_s: float,
    max_retries: int,
    retry_backoff_s: float,
) -> httpx.Response | None:
    """Perform one request, retrying transient failures with exponential backoff.

    Returns the final response, or ``None`` if every attempt raised a connection
    error. Never raises for transient failures.
    """
    backoff = retry_backoff_s
    resp: httpx.Response | None = None
    for attempt in range(max_retries + 1):
        try:
            resp = client.request(
                method, url, headers=headers, json=json_body, timeout=timeout_s
            )
        except (httpx.ConnectError, httpx.TimeoutException):
            resp = None
        if resp is not None and resp.status_code not in _TRANSIENT_STATUS:
            return resp
        if attempt < max_retries:
            time.sleep(backoff)
            backoff = min(backoff * 2, _MAX_BACKOFF_S)
    return resp


def probe_endpoint(
    url: str,
    *,
    api_key_env: str = "OPENAI_API_KEY",
    requested_model: str = "",
    timeout_s: float = 30.0,
    max_retries: int = 3,
    retry_backoff_s: float = 1.0,
) -> EndpointCapabilities:
    """Probe an OpenAI-compatible endpoint for health, models, and logprobs support."""
    api_key = os.environ.get(api_key_env, "")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    base_v1 = _base_v1(url)
    health_url = _health_url(url)

    with httpx.Client() as client:
        health_resp = _request_with_retry(
            client,
            "GET",
            health_url,
            headers=headers,
            timeout_s=timeout_s,
            max_retries=max_retries,
            retry_backoff_s=retry_backoff_s,
        )
        if health_resp is None or health_resp.status_code != 200:
            return EndpointCapabilities(
                reachable=False,
                served_models=(),
                requested_model_served=False,
                logprobs_supported=False,
            )

        raw_health_json = _safe_json(health_resp)

        models_resp = _request_with_retry(
            client,
            "GET",
            f"{base_v1}/models",
            headers=headers,
            timeout_s=timeout_s,
            max_retries=max_retries,
            retry_backoff_s=retry_backoff_s,
        )
        served_models: tuple[str, ...] = ()
        server_version = ""
        if models_resp is not None and models_resp.status_code == 200:
            data = _safe_json(models_resp)
            if data is not None:
                served_models = tuple(
                    str(item["id"]) for item in data.get("data", []) if "id" in item
                )
            server_version = _server_version(models_resp)

        requested_model_served = bool(requested_model) and requested_model in served_models

        model_for_probe = requested_model or (served_models[0] if served_models else "test")
        logprobs_resp = _request_with_retry(
            client,
            "POST",
            f"{base_v1}/completions",
            headers=headers,
            json_body={
                "model": model_for_probe,
                "prompt": "test",
                "max_tokens": 1,
                "logprobs": 5,
            },
            timeout_s=timeout_s,
            max_retries=max_retries,
            retry_backoff_s=retry_backoff_s,
        )
        logprobs_supported = False
        if logprobs_resp is not None and logprobs_resp.status_code == 200:
            data = _safe_json(logprobs_resp)
            if data is not None:
                choices = data.get("choices", [])
                logprobs_supported = bool(choices) and choices[0].get("logprobs") is not None

        return EndpointCapabilities(
            reachable=True,
            served_models=served_models,
            requested_model_served=requested_model_served,
            logprobs_supported=logprobs_supported,
            server_version=server_version,
            raw_health_json=raw_health_json,
        )