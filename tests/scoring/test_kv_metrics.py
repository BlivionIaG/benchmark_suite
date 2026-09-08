"""Prometheus KV-cache scrape used by the sauce scorer."""

from __future__ import annotations

import httpx
import respx

from benchmark_suite.scoring.kv_metrics import (
    fetch_kv_cache_perc,
    metrics_url,
    parse_kv_cache_perc,
)


def test_metrics_url_strips_v1_suffix() -> None:
    assert metrics_url("http://127.0.0.1:8000") == "http://127.0.0.1:8000/metrics"
    assert metrics_url("http://127.0.0.1:8000/") == "http://127.0.0.1:8000/metrics"
    assert metrics_url("http://127.0.0.1:8000/v1") == "http://127.0.0.1:8000/metrics"
    assert metrics_url("http://127.0.0.1:8000/v1/") == "http://127.0.0.1:8000/metrics"
    assert (
        metrics_url("http://127.0.0.1:8000", path="engine-metrics")
        == "http://127.0.0.1:8000/engine-metrics"
    )


def test_parse_vllm_gpu_cache_usage_perc_as_fraction() -> None:
    text = (
        "# HELP vllm:gpu_cache_usage_perc GPU KV-cache usage. 1 means 100 percent.\n"
        "# TYPE vllm:gpu_cache_usage_perc gauge\n"
        'vllm:gpu_cache_usage_perc{engine="0"} 0.42\n'
    )
    assert parse_kv_cache_perc(text) == 42.0


def test_parse_vllm_already_percent() -> None:
    assert parse_kv_cache_perc("vllm:gpu_cache_usage_perc 18.5\n") == 18.5


def test_parse_llama_ratio() -> None:
    assert parse_kv_cache_perc("llama_kv_cache_usage_ratio 0.1\n") == 10.0


def test_parse_missing_series_is_none() -> None:
    assert parse_kv_cache_perc("# TYPE http_requests_total counter\nhttp_requests_total 3\n") is None
    assert parse_kv_cache_perc("") is None


def test_fetch_kv_cache_perc_ok(respx_mock: respx.MockRouter) -> None:
    respx_mock.get("http://127.0.0.1:8000/metrics").mock(
        return_value=httpx.Response(200, text="vllm:kv_cache_usage_perc 0.5\n")
    )
    with httpx.Client() as client:
        assert fetch_kv_cache_perc(client, "http://127.0.0.1:8000/metrics") == 50.0


def test_fetch_kv_cache_perc_404_is_none(respx_mock: respx.MockRouter) -> None:
    respx_mock.get("http://127.0.0.1:8000/metrics").mock(
        return_value=httpx.Response(404, text="no")
    )
    with httpx.Client() as client:
        assert fetch_kv_cache_perc(client, "http://127.0.0.1:8000/metrics") is None
