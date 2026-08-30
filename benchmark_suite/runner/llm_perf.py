"""benchmark_suite/runner/llm_perf.py — llm-perf (iopsystems/llm-perf) wrapper.

llm-perf is a TOML-config-driven Rust binary (Apache-2.0). This module renders
the TOML config by hand (llm-perf must parse exactly what we write), invokes the
binary, and parses its JSON report. The CLI shape (verified against the v0.1.16
source) is positional::

    llm-perf bench <config.toml>
    llm-perf logprobs <config.toml>
    llm-perf kl-divergence <baseline.jsonl> <candidate.jsonl> --format json --output out.json
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmark_suite.recipe import Recipe


@dataclass(frozen=True)
class LLMPerfResult:
    """Parsed llm-perf JSON output (one row per request stream)."""

    raw: dict[str, Any]
    output_tok_s: float
    total_tok_s: float
    peak_output_tok_s: float
    ttft_mean_ms: float
    ttft_median_ms: float
    ttft_p99_ms: float
    tpot_mean_ms: float
    tpot_median_ms: float
    duration_s: float
    successful: int
    failed: int


def parse_llm_perf_json(raw: dict[str, Any]) -> LLMPerfResult:
    """Map the llm-perf JSON report to LLMPerfResult.

    llm-perf has no separate "peak output throughput" metric, so
    ``peak_output_tok_s`` is set to the steady-state output throughput.
    """
    throughput = raw["throughput"]
    latency = raw["latency"]
    summary = raw["summary"]
    duration = raw["duration"]
    duration_s = float(duration["secs"]) + float(duration["nanos"]) / 1e9
    output_tok_s = float(throughput["output_tokens_per_second"])
    return LLMPerfResult(
        raw=raw,
        output_tok_s=output_tok_s,
        total_tok_s=float(throughput["input_tokens_per_second"]) + output_tok_s,
        peak_output_tok_s=output_tok_s,
        ttft_mean_ms=float(latency["ttft_mean_ms"]),
        ttft_median_ms=float(latency["ttft_p50_ms"]),
        ttft_p99_ms=float(latency["ttft_p99_ms"]),
        tpot_mean_ms=float(latency["tpot_mean_ms"]),
        tpot_median_ms=float(latency["tpot_p50_ms"]),
        duration_s=duration_s,
        successful=int(summary["requests_successful"]),
        failed=int(summary["requests_failed"]),
    )


def _endpoint_lines(recipe: Recipe, *, base_url: str, max_tokens: int) -> list[str]:
    api_key = os.environ.get(recipe.endpoint.api_key_env, "")
    lines = [
        "[endpoint]",
        f'base_url = "{base_url}"',
        f'model = "{recipe.endpoint.model_name}"',
        f"timeout = {int(recipe.endpoint.timeout_s)}",
        f"max_retries = {recipe.endpoint.max_retries}",
    ]
    if api_key:
        lines.append(f'api_key = "{api_key}"')
    # max_tokens is required for synthetic mode to bound output length.
    lines.append(f"max_tokens = {max_tokens}")
    lines.append("")
    return lines


def _input_lines(recipe: Recipe, *, prompts_file: Path | None = None) -> list[str]:
    load = recipe.bench.load
    lines = ["[input]"]
    if prompts_file is not None:
        lines.append(f'file = "{prompts_file}"')
    elif load.dataset is not None:
        lines.append(f'file = "{load.dataset}"')
    else:
        lines.append('file = "synthetic"')
        lines.append(f"sample_size = {load.num_prompts}")
    lines.append("")
    if prompts_file is None and load.dataset is None:
        lines.append("[input.synthetic]")
        lines.append(f"prompt_tokens = {load.input_len}")
        lines.append("")
    return lines


def generate_toml(
    recipe: Recipe,
    *,
    concurrency: int,
    endpoint_base_url: str | None = None,
    output_file: Path | None = None,
    extra_flags: list[str] | None = None,
) -> str:
    """Render an llm-perf v0.1.x TOML config from a Recipe.

    ``extra_flags`` is reserved for future use: llm-perf is TOML-driven, so
    there are no per-run CLI flags to inject into the config.
    """
    del extra_flags  # reserved; llm-perf is TOML-driven
    load = recipe.bench.load
    base_url = endpoint_base_url or recipe.endpoint.base_url_v1

    lines: list[str] = []
    lines.extend(_endpoint_lines(recipe, base_url=base_url, max_tokens=load.output_len))

    lines.append("[load]")
    lines.append(f"concurrent_requests = {concurrency}")
    if load.qps is not None:
        lines.append(f"qps = {load.qps:g}")
    if load.duration_s is not None:
        lines.append(f"duration_seconds = {int(load.duration_s)}")
    else:
        lines.append(f"total_requests = {load.num_prompts}")
    if load.warmup_requests:
        lines.append(f"warmup_requests = {load.warmup_requests}")
    lines.append("")

    lines.extend(_input_lines(recipe))

    lines.append("[output]")
    lines.append('format = "json"')
    if output_file is not None:
        lines.append(f'file = "{output_file}"')
    lines.append("")

    lines.append("[log]")
    lines.append('level = "info"')
    lines.append("")

    return "\n".join(lines)


def _generate_logprobs_toml(
    recipe: Recipe,
    *,
    output_jsonl_path: Path,
    max_tokens: int,
    prompts_file: Path | None,
) -> str:
    lines: list[str] = []
    lines.extend(
        _endpoint_lines(
            recipe, base_url=recipe.endpoint.base_url_v1, max_tokens=max_tokens
        )
    )
    lines.extend(_input_lines(recipe, prompts_file=prompts_file))
    lines.append("[logprobs]")
    lines.append("enabled = true")
    lines.append("top_logprobs = 5")
    lines.append(f'output = "{output_jsonl_path}"')
    lines.append("")
    return "\n".join(lines)


def _resolve_binary(binary_path: str) -> str:
    resolved = shutil.which(binary_path)
    if resolved is None:
        raise FileNotFoundError(f"llm-perf binary not found: {binary_path}")
    return resolved


def _run_with_toml(
    binary: str,
    subcommand: str,
    toml_str: str,
    *,
    timeout_s: float,
    extra_env: dict[str, str] | None = None,
) -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as fh:
        fh.write(toml_str)
        toml_path = Path(fh.name)
    try:
        env = {**os.environ, **(extra_env or {})}
        subprocess.run(
            [binary, subcommand, str(toml_path)],
            check=True,
            env=env,
            timeout=timeout_s,
            capture_output=True,
            text=True,
        )
    finally:
        toml_path.unlink(missing_ok=True)


def run_llm_perf_bench(
    recipe: Recipe,
    *,
    concurrency: int,
    binary_path: str = "llm-perf",
    output_json_path: Path,
    extra_env: dict[str, str] | None = None,
    timeout_s: float = 1800.0,
) -> LLMPerfResult:
    """Synthesize TOML, write to temp, invoke ``llm-perf bench``, parse JSON.

    Raises FileNotFoundError if ``binary_path`` is not on PATH (callers
    distinguish "missing binary" from "run failed"). Raises
    subprocess.CalledProcessError if llm-perf exits non-zero.
    """
    binary = _resolve_binary(binary_path)
    toml_str = generate_toml(
        recipe, concurrency=concurrency, output_file=output_json_path
    )
    _run_with_toml(binary, "bench", toml_str, timeout_s=timeout_s, extra_env=extra_env)
    raw = json.loads(output_json_path.read_text())
    return parse_llm_perf_json(raw)


def run_llm_perf_logprobs(
    recipe: Recipe,
    *,
    output_jsonl_path: Path,
    max_tokens: int = 1,
    prompts: list[str] | None = None,
    binary_path: str = "llm-perf",
    timeout_s: float = 1800.0,
) -> Path:
    """Capture per-prompt logprobs to a JSONL file. Returns the path."""
    binary = _resolve_binary(binary_path)
    prompts_file: Path | None = None
    if prompts is not None:
        prompts_file = output_jsonl_path.with_suffix(".prompts.jsonl")
        prompts_file.write_text(
            "".join(json.dumps({"prompt": p}) + "\n" for p in prompts)
        )
    toml_str = _generate_logprobs_toml(
        recipe,
        output_jsonl_path=output_jsonl_path,
        max_tokens=max_tokens,
        prompts_file=prompts_file,
    )
    try:
        _run_with_toml(binary, "logprobs", toml_str, timeout_s=timeout_s)
    finally:
        if prompts_file is not None:
            prompts_file.unlink(missing_ok=True)
    return output_jsonl_path


def run_llm_perf_kl_divergence(
    baseline_jsonl: Path,
    candidate_jsonl: Path,
    *,
    output_json_path: Path,
    binary_path: str = "llm-perf",
    timeout_s: float = 600.0,
) -> dict[str, Any]:
    """Invoke ``llm-perf kl-divergence`` and parse the JSON report."""
    binary = _resolve_binary(binary_path)
    subprocess.run(
        [
            binary,
            "kl-divergence",
            str(baseline_jsonl),
            str(candidate_jsonl),
            "--format",
            "json",
            "--output",
            str(output_json_path),
        ],
        check=True,
        timeout=timeout_s,
        capture_output=True,
        text=True,
    )
    return json.loads(output_json_path.read_text())