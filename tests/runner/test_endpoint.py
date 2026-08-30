"""Tests for benchmark_suite.runner.endpoint — capability probe via respx."""
from __future__ import annotations

from typing import cast

import httpx
import pytest
import respx
from respx.models import Call

from benchmark_suite.runner.endpoint import probe_endpoint

from .conftest import BASE_URL


def test_probe_health_ok(
    mock_endpoint_health: respx.MockRouter,
    mock_endpoint_models: respx.MockRouter,
    mock_endpoint_logprobs_ok: respx.MockRouter,
) -> None:
    caps = probe_endpoint(BASE_URL, requested_model="test-model")
    assert caps.reachable is True
    assert caps.served_models == ("test-model",)
    assert caps.requested_model_served is True
    assert caps.logprobs_supported is True
    assert caps.server_version == "0.20.1"
    assert caps.raw_health_json == {"status": "ok"}


def test_probe_unreachable(mock_endpoint_unreachable: respx.MockRouter) -> None:
    caps = probe_endpoint(BASE_URL, max_retries=1, retry_backoff_s=0.01)
    assert caps.reachable is False
    assert caps.served_models == ()
    assert caps.requested_model_served is False
    assert caps.logprobs_supported is False


def test_probe_models_match_requested(
    mock_endpoint_health: respx.MockRouter,
    mock_endpoint_models: respx.MockRouter,
    mock_endpoint_logprobs_ok: respx.MockRouter,
) -> None:
    caps = probe_endpoint(BASE_URL, requested_model="test-model")
    assert caps.requested_model_served is True


def test_probe_models_mismatch_requested(
    mock_endpoint_health: respx.MockRouter,
    mock_endpoint_models: respx.MockRouter,
    mock_endpoint_logprobs_ok: respx.MockRouter,
) -> None:
    caps = probe_endpoint(BASE_URL, requested_model="other-model")
    assert caps.requested_model_served is False


def test_probe_logprobs_supported(
    mock_endpoint_health: respx.MockRouter,
    mock_endpoint_models: respx.MockRouter,
    mock_endpoint_logprobs_ok: respx.MockRouter,
) -> None:
    caps = probe_endpoint(BASE_URL)
    assert caps.logprobs_supported is True


def test_probe_logprobs_unsupported(
    mock_endpoint_health: respx.MockRouter,
    mock_endpoint_models: respx.MockRouter,
    mock_endpoint_logprobs_unsupported: respx.MockRouter,
) -> None:
    caps = probe_endpoint(BASE_URL)
    assert caps.logprobs_supported is False


def test_probe_uses_api_key_from_env(
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
    mock_endpoint_health: respx.MockRouter,
    mock_endpoint_models: respx.MockRouter,
    mock_endpoint_logprobs_ok: respx.MockRouter,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    probe_endpoint(BASE_URL, requested_model="test-model")
    calls = cast(list[Call], respx_mock.calls)
    auth_headers = {dict(call.request.headers).get("authorization") for call in calls}
    assert "Bearer test-key" in auth_headers


def test_probe_retries_on_502(
    respx_mock: respx.MockRouter,
    mock_endpoint_models: respx.MockRouter,
    mock_endpoint_logprobs_ok: respx.MockRouter,
) -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(502)
        return httpx.Response(200, json={"status": "ok"})

    respx_mock.get(f"{BASE_URL}/health").mock(side_effect=handler)
    caps = probe_endpoint(BASE_URL, max_retries=3, retry_backoff_s=0.01)
    assert caps.reachable is True
    assert len(calls) == 2


def test_probe_max_retries_exhausted(respx_mock: respx.MockRouter) -> None:
    respx_mock.get(f"{BASE_URL}/health").mock(return_value=httpx.Response(502))
    caps = probe_endpoint(BASE_URL, max_retries=3, retry_backoff_s=0.01)
    assert caps.reachable is False
    assert caps.served_models == ()


def test_probe_backoff_exponential(
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    respx_mock.get(f"{BASE_URL}/health").mock(return_value=httpx.Response(502))
    sleeps: list[float] = []

    def record_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr("benchmark_suite.runner.endpoint.time.sleep", record_sleep)
    probe_endpoint(BASE_URL, max_retries=3, retry_backoff_s=1.0)
    assert sleeps == [1.0, 2.0, 4.0]