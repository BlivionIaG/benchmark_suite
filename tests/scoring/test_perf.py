"""Latency-rate helpers used by the sauce scorer (TTFT / TPOT / prefill / decode)."""

from __future__ import annotations

from benchmark_suite.scoring.perf import (
    cached_tokens_from_usage,
    compact,
    latency_rates,
)


def test_prefill_tok_s_is_prompt_tokens_over_ttft() -> None:
    rates = latency_rates(
        prompt_tokens=1000,
        completion_tokens=11,
        ttft_ms=500.0,
        latency_ms=1500.0,
    )
    assert rates.prefill_tok_s == 2000.0


def test_decode_tok_s_and_tpot_exclude_first_token() -> None:
    rates = latency_rates(
        prompt_tokens=1000,
        completion_tokens=11,
        ttft_ms=500.0,
        latency_ms=1500.0,
    )
    assert rates.decode_tok_s == 10.0
    assert rates.tpot_ms == 100.0


def test_non_stream_has_prefill_but_no_decode_split() -> None:
    rates = latency_rates(
        prompt_tokens=200,
        completion_tokens=8,
        ttft_ms=800.0,
        latency_ms=800.0,
    )
    assert rates.prefill_tok_s == 250.0
    assert rates.decode_tok_s is None
    assert rates.tpot_ms is None


def test_single_completion_token_has_no_tpot() -> None:
    rates = latency_rates(
        prompt_tokens=50,
        completion_tokens=1,
        ttft_ms=10.0,
        latency_ms=40.0,
    )
    assert rates.prefill_tok_s == 5000.0
    assert rates.decode_tok_s is None
    assert rates.tpot_ms is None


def test_zero_ttft_or_prompt_skips_prefill() -> None:
    assert (
        latency_rates(
            prompt_tokens=0, completion_tokens=4, ttft_ms=10.0, latency_ms=20.0
        ).prefill_tok_s
        is None
    )
    assert (
        latency_rates(
            prompt_tokens=10, completion_tokens=4, ttft_ms=0.0, latency_ms=20.0
        ).prefill_tok_s
        is None
    )


def test_cached_tokens_from_prompt_tokens_details() -> None:
    assert (
        cached_tokens_from_usage(
            {"prompt_tokens": 100, "prompt_tokens_details": {"cached_tokens": 80}}
        )
        == 80
    )


def test_cached_tokens_from_usage_top_level_fallback() -> None:
    assert cached_tokens_from_usage({"cached_tokens": 12}) == 12


def test_cached_tokens_missing_is_none() -> None:
    assert cached_tokens_from_usage({"prompt_tokens": 4}) is None
    assert cached_tokens_from_usage(None) is None
    assert cached_tokens_from_usage("nope") is None


def test_compact_drops_none_keeps_zero() -> None:
    assert compact({"a": 1, "b": None, "c": 0.0, "d": False}) == {
        "a": 1,
        "c": 0.0,
        "d": False,
    }
