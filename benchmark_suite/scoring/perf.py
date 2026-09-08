"""Per-request latency rates (TTFT, TPOT, prefill tok/s, decode tok/s).

Prefill tok/s = prompt_tokens / (ttft_ms / 1000).
Decode tok/s / TPOT exclude the first output token (counted in TTFT).
Non-streaming replies have ttft == latency, so decode/TPOT are omitted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast


@dataclass(frozen=True)
class LatencyRates:
    """Per-request prefill/decode split. None when the split is not measurable."""

    prefill_tok_s: float | None
    decode_tok_s: float | None
    tpot_ms: float | None


def latency_rates(
    *,
    prompt_tokens: int,
    completion_tokens: int,
    ttft_ms: float,
    latency_ms: float,
) -> LatencyRates:
    """Derive prefill tok/s, decode tok/s, and TPOT from one request's timings."""
    prefill: float | None = None
    if prompt_tokens > 0 and ttft_ms > 0:
        prefill = prompt_tokens / (ttft_ms / 1000.0)

    decode: float | None = None
    tpot: float | None = None
    gen_ms = latency_ms - ttft_ms
    if completion_tokens > 1 and gen_ms > 0:
        n_decode = completion_tokens - 1
        decode = n_decode / (gen_ms / 1000.0)
        tpot = gen_ms / n_decode
    return LatencyRates(prefill_tok_s=prefill, decode_tok_s=decode, tpot_ms=tpot)


def cached_tokens_from_usage(usage: object) -> int | None:
    """Best-effort cached-token count from OpenAI-compatible ``usage``.

    Prefers ``prompt_tokens_details.cached_tokens`` (OpenAI / vLLM / Azure),
    then top-level ``cached_tokens``. Missing → None (never 0).
    """
    if not isinstance(usage, dict):
        return None
    data = cast(dict[str, Any], usage)
    details = data.get("prompt_tokens_details")
    if isinstance(details, dict):
        cached = details.get("cached_tokens")
        if isinstance(cached, (int, float)):
            return int(cached)
    cached_top = data.get("cached_tokens")
    if isinstance(cached_top, (int, float)):
        return int(cached_top)
    return None


def compact(row: dict[str, Any]) -> dict[str, Any]:
    """Drop None values so artifacts never store a missing metric as 0."""
    return {key: value for key, value in row.items() if value is not None}
