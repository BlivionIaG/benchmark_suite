"""benchmark_suite/runner/vllm_bench.py — `vllm bench serve` fallback wrapper.

The 11 regex patterns below are ported VERBATIM from the parent project's
``benchmark_results/parse_matrix_results.py`` so that the legacy ``summary.csv``
columns keep working byte-for-byte. The command is ``vllm bench serve`` (the
online benchmark that emits the "Serving Benchmark Result" block the parent
parser targets), not the offline ``vllm bench throughput``.
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass

from benchmark_suite.recipe import Recipe


@dataclass(frozen=True)
class VLLMBenchResult:
    """Parsed ``vllm bench serve`` stdout.

    Column names match the parent's ``parse_matrix_results.py`` regex
    conventions byte-for-byte so existing reports keep working.
    """

    raw_stdout: str
    output_tok_s: float
    peak_output_tok_s: float
    total_tok_s: float
    ttft_mean_ms: float
    ttft_median_ms: float
    ttft_p99_ms: float
    tpot_mean_ms: float
    tpot_median_ms: float
    duration_s: float
    successful: int
    failed: int


# The 11 regex patterns ported VERBATIM from the parent's
# benchmark_results/parse_matrix_results.py. Field names match the parent's
# legacy `summary.csv` columns exactly.
VLLM_BENCH_REGEXES: dict[str, str] = {
    "output_tok_s": r"Output token throughput \(tok/s\):\s+([\d.]+)",
    "peak_output_tok_s": r"Peak output token throughput \(tok/s\):\s+([\d.]+)",
    "total_tok_s": r"Total token throughput \(tok/s\):\s+([\d.]+)",
    "ttft_mean_ms": r"Mean TTFT \(ms\):\s+([\d.]+)",
    "ttft_median_ms": r"Median TTFT \(ms\):\s+([\d.]+)",
    "ttft_p99_ms": r"P99 TTFT \(ms\):\s+([\d.]+)",
    "tpot_mean_ms": r"Mean TPOT \(ms\):\s+([\d.]+)",
    "tpot_median_ms": r"Median TPOT \(ms\):\s+([\d.]+)",
    "duration_s": r"Benchmark duration \(s\):\s+([\d.]+)",
    "successful": r"Successful requests:\s+(\d+)",
    "failed": r"Failed requests:\s+(\d+)",
}


def _search_field(stdout: str, field: str) -> str:
    match = re.search(VLLM_BENCH_REGEXES[field], stdout)
    if match is None:
        raise ValueError(f"missing field in vllm bench output: {field}")
    return match.group(1)


def parse_vllm_bench_stdout(stdout: str) -> VLLMBenchResult:
    """Apply the 11 regexes to stdout; raise ValueError if any field is missing."""
    return VLLMBenchResult(
        raw_stdout=stdout,
        output_tok_s=float(_search_field(stdout, "output_tok_s")),
        peak_output_tok_s=float(_search_field(stdout, "peak_output_tok_s")),
        total_tok_s=float(_search_field(stdout, "total_tok_s")),
        ttft_mean_ms=float(_search_field(stdout, "ttft_mean_ms")),
        ttft_median_ms=float(_search_field(stdout, "ttft_median_ms")),
        ttft_p99_ms=float(_search_field(stdout, "ttft_p99_ms")),
        tpot_mean_ms=float(_search_field(stdout, "tpot_mean_ms")),
        tpot_median_ms=float(_search_field(stdout, "tpot_median_ms")),
        duration_s=float(_search_field(stdout, "duration_s")),
        successful=int(_search_field(stdout, "successful")),
        failed=int(_search_field(stdout, "failed")),
    )


def build_vllm_bench_cmd(
    recipe: Recipe,
    *,
    endpoint_url: str,
    concurrency: int,
    extra_flags: list[str] | None = None,
) -> list[str]:
    """Build argv for ``vllm bench serve`` against a running endpoint.

    Maps ``recipe.bench.load`` to the online-benchmark flags. The user's
    ``backend.vllm.*`` knobs (compilation-config etc.) are server-side config
    and are intentionally NOT passed here.
    """
    load = recipe.bench.load
    cmd = [
        "vllm",
        "bench",
        "serve",
        "--base-url",
        endpoint_url,
        "--model",
        recipe.endpoint.model_name,
        "--dataset-name",
        "random",
        "--input-len",
        str(load.input_len),
        "--output-len",
        str(load.output_len),
        "--num-prompts",
        str(load.num_prompts),
        "--max-concurrency",
        str(concurrency),
    ]
    if extra_flags:
        cmd.extend(extra_flags)
    return cmd


def run_vllm_bench(
    recipe: Recipe,
    *,
    endpoint_url: str,
    concurrency: int,
    timeout_s: float = 1800.0,
) -> VLLMBenchResult:
    """Invoke ``vllm bench serve`` via subprocess, capture stdout, parse."""
    cmd = build_vllm_bench_cmd(recipe, endpoint_url=endpoint_url, concurrency=concurrency)
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout_s, check=True
    )
    return parse_vllm_bench_stdout(proc.stdout)