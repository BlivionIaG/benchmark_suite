"""Tests for benchmark_suite.runner.llm_perf — TOML gen + fake-binary invocation."""
from __future__ import annotations

import json
import subprocess
import tomllib
from pathlib import Path

import pytest

from benchmark_suite.recipe import Recipe
from benchmark_suite.runner.llm_perf import (
    generate_toml,
    run_llm_perf_bench,
    run_llm_perf_kl_divergence,
    run_llm_perf_logprobs,
)

# Canned llm-perf JSON report (values mirror the vllm_bench golden fixture).
CANNED_REPORT = {
    "summary": {"requests_successful": 32, "requests_failed": 0},
    "throughput": {
        "output_tokens_per_second": 76.57,
        "input_tokens_per_second": 68.64,
    },
    "latency": {
        "ttft_mean_ms": 128.45,
        "ttft_p50_ms": 96.32,
        "ttft_p99_ms": 312.18,
        "tpot_mean_ms": 13.06,
        "tpot_p50_ms": 12.91,
    },
    "duration": {"secs": 45, "nanos": 320000000},
}


def _recipe(**overrides: object) -> Recipe:
    data: dict[str, object] = {
        "meta": {"name": "x", "description": "y"},
        "endpoint": {"model_name": "test-model"},
    }
    data.update(overrides)
    return Recipe.model_validate(data)


def _fake_binary(tmp_path: Path, name: str, body: str, **subs: str) -> Path:
    for key, value in subs.items():
        body = body.replace(key, value)
    fake = tmp_path / name
    fake.write_text("#!/usr/bin/env bash\n" + body)
    fake.chmod(0o755)
    return fake


# Shell snippet: extract the `file = "..."` value from the [output] section of a TOML.
_OUTPUT_FILE_EXTRACT = (
    'out=$(sed -n \'/^\\[output\\]/,$p\' "$config" '
    '| sed -n \'s/^file = "\\(.*\\)"/\\1/p\' | head -1)\n'
)


def test_generate_toml_minimal_recipe() -> None:
    recipe = _recipe(bench={"load": {"concurrencies": [1]}})
    data = tomllib.loads(generate_toml(recipe, concurrency=1))
    assert data["endpoint"]["base_url"] == "http://127.0.0.1:8000/v1"
    assert data["endpoint"]["model"] == "test-model"
    assert data["load"]["concurrent_requests"] == 1
    assert data["input"]["file"] == "synthetic"
    assert data["output"]["format"] == "json"


def test_generate_toml_per_concurrency_emits_correct_concurrency() -> None:
    recipe = _recipe(bench={"load": {"concurrencies": [1, 4, 8]}})
    data = tomllib.loads(generate_toml(recipe, concurrency=4))
    assert data["load"]["concurrent_requests"] == 4


def test_generate_toml_uses_recipe_input_output_len() -> None:
    recipe = _recipe(bench={"load": {"input_len": 512, "output_len": 128}})
    data = tomllib.loads(generate_toml(recipe, concurrency=1))
    assert data["endpoint"]["max_tokens"] == 128
    assert data["input"]["synthetic"]["prompt_tokens"] == 512


def test_run_llm_perf_bench_calls_binary_with_config_path(tmp_path: Path) -> None:
    argv_file = tmp_path / "argv.txt"
    body = (
        'echo "$@" > __ARGV__\n'
        'config="${@: -1}"\n'
        'if [ -f "$config" ]; then echo "config-exists: yes" >> __ARGV__; '
        'else echo "config-exists: no" >> __ARGV__; fi\n'
        + _OUTPUT_FILE_EXTRACT
        + 'echo \'__JSON__\' > "$out"\n'
    )
    fake = _fake_binary(
        tmp_path,
        "fake-llm-perf.sh",
        body,
        __ARGV__=str(argv_file),
        __JSON__=json.dumps(CANNED_REPORT),
    )
    recipe = _recipe()
    out = tmp_path / "out.json"
    run_llm_perf_bench(
        recipe, concurrency=1, binary_path=str(fake), output_json_path=out
    )
    argv_text = argv_file.read_text()
    assert "bench" in argv_text
    assert "config-exists: yes" in argv_text
    config_path = argv_text.splitlines()[0].split()[-1]
    assert config_path.endswith(".toml")


def test_run_llm_perf_bench_writes_json_then_parses(tmp_path: Path) -> None:
    body = (
        'config="${@: -1}"\n'
        + _OUTPUT_FILE_EXTRACT
        + 'echo \'__JSON__\' > "$out"\n'
    )
    fake = _fake_binary(
        tmp_path,
        "fake-llm-perf.sh",
        body,
        __JSON__=json.dumps(CANNED_REPORT),
    )
    recipe = _recipe()
    out = tmp_path / "out.json"
    result = run_llm_perf_bench(
        recipe, concurrency=1, binary_path=str(fake), output_json_path=out
    )
    assert result.output_tok_s == 76.57
    assert result.total_tok_s == pytest.approx(145.21)
    assert result.ttft_mean_ms == 128.45
    assert result.ttft_median_ms == 96.32
    assert result.ttft_p99_ms == 312.18
    assert result.tpot_mean_ms == 13.06
    assert result.tpot_median_ms == 12.91
    assert result.duration_s == pytest.approx(45.32)
    assert result.successful == 32
    assert result.failed == 0


def test_run_llm_perf_bench_missing_binary_raises_filenotfound(tmp_path: Path) -> None:
    recipe = _recipe()
    with pytest.raises(FileNotFoundError):
        run_llm_perf_bench(
            recipe,
            concurrency=1,
            binary_path="/nonexistent/path",
            output_json_path=tmp_path / "out.json",
        )


def test_run_llm_perf_bench_nonzero_exit_raises(tmp_path: Path) -> None:
    fake = _fake_binary(tmp_path, "fake-llm-perf.sh", "exit 1\n")
    recipe = _recipe()
    with pytest.raises(subprocess.CalledProcessError):
        run_llm_perf_bench(
            recipe,
            concurrency=1,
            binary_path=str(fake),
            output_json_path=tmp_path / "out.json",
        )


def test_run_llm_perf_kl_divergence_invokes_correct_subcommand(tmp_path: Path) -> None:
    argv_file = tmp_path / "argv.txt"
    body = (
        'echo "$@" > __ARGV__\n'
        'out=""; prev=""\n'
        'for a in "$@"; do\n'
        '  if [ "$prev" = "--output" ]; then out="$a"; fi\n'
        '  prev="$a"\n'
        'done\n'
        'if [ -n "$out" ]; then echo \'{"kl_divergence": 0.123}\' > "$out"; fi\n'
    )
    fake = _fake_binary(tmp_path, "fake-llm-perf.sh", body, __ARGV__=str(argv_file))
    baseline = tmp_path / "baseline.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    baseline.write_text("")
    candidate.write_text("")
    out = tmp_path / "kld.json"
    result = run_llm_perf_kl_divergence(
        baseline, candidate, output_json_path=out, binary_path=str(fake)
    )
    argv = argv_file.read_text().strip()
    assert "kl-divergence" in argv
    assert str(baseline) in argv
    assert str(candidate) in argv
    assert result == {"kl_divergence": 0.123}


def test_run_llm_perf_logprobs_writes_jsonl(tmp_path: Path) -> None:
    body = (
        'config="${@: -1}"\n'
        'out=$(sed -n \'/^\\[logprobs\\]/,$p\' "$config" '
        '| sed -n \'s/^output = "\\(.*\\)"/\\1/p\' | head -1)\n'
        "printf '%s\\n' "
        "'{\"token\":\"a\",\"logprob\":-0.1}' "
        "'{\"token\":\"b\",\"logprob\":-0.2}' "
        "'{\"token\":\"c\",\"logprob\":-0.3}' > \"$out\"\n"
    )
    fake = _fake_binary(tmp_path, "fake-llm-perf.sh", body)
    recipe = _recipe()
    out = tmp_path / "logprobs.jsonl"
    result_path = run_llm_perf_logprobs(
        recipe, output_jsonl_path=out, binary_path=str(fake)
    )
    assert result_path == out
    assert out.exists()
    lines = [json.loads(ln) for ln in out.read_text().splitlines()]
    assert len(lines) == 3
    assert lines[0]["token"] == "a"