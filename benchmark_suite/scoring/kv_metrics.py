"""Best-effort KV-cache usage from an OpenAI-compatible server's /metrics.

Never raises into the scorer: 404, missing series, or a down endpoint → None.
vLLM historically exports ``gpu_cache_usage_perc`` as a 0–1 fraction despite
the name; values in (0, 1] are treated as fractions, values > 1 as percents.
"""

from __future__ import annotations

import re

import httpx

_SAMPLE = re.compile(
    r"^(?P<name>[A-Za-z_:][A-Za-z0-9_:]*)"
    r"(?:\{[^}]*\})?"
    r"\s+(?P<value>[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*$"
)

# First match in this list wins when several series are present.
_KNOWN: tuple[tuple[str, bool], ...] = (
    ("vllm:gpu_cache_usage_perc", False),
    ("vllm:kv_cache_usage_perc", False),
    ("gpu_cache_usage_perc", False),
    ("kv_cache_usage_perc", False),
    ("llama_kv_cache_usage_ratio", True),
    ("llamacpp:kv_cache_usage_ratio", True),
    ("sglang:cache_usage", False),
)


def metrics_url(endpoint_url: str, path: str = "/metrics") -> str:
    """``{endpoint}/metrics`` with a trailing ``/v1`` stripped from the base."""
    base = endpoint_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3].rstrip("/")
    suffix = path if path.startswith("/") else f"/{path}"
    return base + suffix


def _as_percent(value: float, *, ratio: bool) -> float:
    if ratio or 0.0 <= value <= 1.0:
        return value * 100.0
    return value


def parse_kv_cache_perc(text: str) -> float | None:
    """Parse a Prometheus text exposition for a known KV-cache gauge."""
    found: dict[str, float] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _SAMPLE.match(line)
        if match is None:
            continue
        name = match.group("name")
        try:
            value = float(match.group("value"))
        except ValueError:
            continue
        found[name] = value
    for name, ratio in _KNOWN:
        if name in found:
            return _as_percent(found[name], ratio=ratio)
    return None


def fetch_kv_cache_perc(client: httpx.Client, url: str) -> float | None:
    """GET ``url`` and parse KV-cache %; never raises."""
    try:
        resp = client.get(url)
    except httpx.HTTPError:
        return None
    if resp.status_code != 200:
        return None
    return parse_kv_cache_perc(resp.text)
