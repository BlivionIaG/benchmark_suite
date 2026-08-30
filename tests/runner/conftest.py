"""Fixtures for benchmark_suite.runner tests (respx-mocked OpenAI endpoint)."""
from __future__ import annotations

import httpx
import pytest
import respx

BASE_URL = "http://127.0.0.1:8000"


@pytest.fixture
def mock_endpoint_health(respx_mock: respx.MockRouter) -> respx.MockRouter:
    """GET /health -> 200."""
    respx_mock.get(f"{BASE_URL}/health").mock(
        return_value=httpx.Response(200, json={"status": "ok"})
    )
    return respx_mock


@pytest.fixture
def mock_endpoint_models(respx_mock: respx.MockRouter) -> respx.MockRouter:
    """GET /v1/models -> {"data": [{"id": "test-model"}]}."""
    respx_mock.get(f"{BASE_URL}/v1/models").mock(
        return_value=httpx.Response(
            200,
            json={"data": [{"id": "test-model"}]},
            headers={"x-vllm-version": "0.20.1"},
        )
    )
    return respx_mock


@pytest.fixture
def mock_endpoint_logprobs_ok(respx_mock: respx.MockRouter) -> respx.MockRouter:
    """POST /v1/completions -> choices[0].logprobs populated."""
    respx_mock.post(f"{BASE_URL}/v1/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "text": "test",
                        "logprobs": {"tokens": ["test"], "token_logprobs": [-0.1]},
                    }
                ]
            },
        )
    )
    return respx_mock


@pytest.fixture
def mock_endpoint_logprobs_unsupported(respx_mock: respx.MockRouter) -> respx.MockRouter:
    """POST /v1/completions -> choices[0].logprobs = null (OpenAI may omit the field)."""
    respx_mock.post(f"{BASE_URL}/v1/completions").mock(
        return_value=httpx.Response(
            200,
            json={"choices": [{"text": "test", "logprobs": None}]},
        )
    )
    return respx_mock


@pytest.fixture
def mock_endpoint_unreachable(respx_mock: respx.MockRouter) -> respx.MockRouter:
    """GET /health -> 502 (transient failure)."""
    respx_mock.get(f"{BASE_URL}/health").mock(return_value=httpx.Response(502))
    return respx_mock