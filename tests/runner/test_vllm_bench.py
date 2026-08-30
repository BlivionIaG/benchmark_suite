"""Tests for benchmark_suite.runner.vllm_bench — golden stdout parse + argv build."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from benchmark_suite.recipe import Recipe
from benchmark_suite.runner.vllm_bench import (
    build_vllm_bench_cmd,
    parse_vllm_bench_stdout,
    run_vllm_bench,
)

FIXTURE_PATH = (
    Path(__file__).resolve().parent.parent / "fixtures" / "vllm_bench_stdout_sample.txt"
)


@pytest.fixture
def vllm_bench_stdout() -> str:
    return FIXTURE_PATH.read_text()


def _recipe(**overrides: object) -> Recipe:
    data: dict[str, object] = {
        "meta": {"name": "x", "description": "y"},
        "endpoint": {"model_name": "test-model"},
    }
    data.update(overrides)
    return Recipe.model_validate(data)


def test_parse_vllm_bench_stdout_full_fixture(vllm_bench_stdout: str) -> None:
    result = parse_vllm_bench_stdout(vllm_bench_stdout)
    assert result.output_tok_s == 76.57
    assert result.peak_output_tok_s == 84.32
    assert result.total_tok_s == 145.21
    assert result.ttft_mean_ms == 128.45
    assert result.ttft_median_ms == 96.32
    assert result.ttft_p99_ms == 312.18
    assert result.tpot_mean_ms == 13.06
    assert result.tpot_median_ms == 12.91
    assert result.duration_s == 45.32
    assert result.successful == 32
    assert result.failed == 0
    assert result.raw_stdout == vllm_bench_stdout


def test_parse_vllm_bench_stdout_missing_field_raises(vllm_bench_stdout: str) -> None:
    lines = [ln for ln in vllm_bench_stdout.splitlines() if "Mean TPOT" not in ln]
    stdout = "\n".join(lines) + "\n"
    with pytest.raises(ValueError, match="tpot_mean_ms"):
        parse_vllm_bench_stdout(stdout)


def test_parse_vllm_bench_stdout_partial_int_field(vllm_bench_stdout: str) -> None:
    result = parse_vllm_bench_stdout(vllm_bench_stdout)
    assert result.successful == 32
    assert type(result.successful) is int
    assert type(result.failed) is int
    assert type(result.output_tok_s) is float


def test_build_vllm_bench_cmd_minimal() -> None:
    recipe = _recipe()
    cmd = build_vllm_bench_cmd(
        recipe, endpoint_url="http://127.0.0.1:8000", concurrency=1
    )
    assert cmd == [
        "vllm",
        "bench",
        "serve",
        "--base-url",
        "http://127.0.0.1:8000",
        "--model",
        "test-model",
        "--dataset-name",
        "random",
        "--input-len",
        "256",
        "--output-len",
        "64",
        "--num-prompts",
        "32",
        "--max-concurrency",
        "1",
    ]


def test_build_vllm_bench_cmd_with_concurrency_and_num_prompts() -> None:
    recipe = _recipe(
        bench={"load": {"concurrencies": [8], "num_prompts": 64}},
    )
    cmd = build_vllm_bench_cmd(
        recipe, endpoint_url="http://127.0.0.1:8000", concurrency=8
    )
    assert cmd[cmd.index("--num-prompts") + 1] == "64"
    assert cmd[cmd.index("--max-concurrency") + 1] == "8"


def test_run_vllm_bench_invokes_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = tmp_path / "vllm"
    fake.write_text(f"#!/usr/bin/env bash\ncat '{FIXTURE_PATH}'\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")

    recipe = _recipe()
    result = run_vllm_bench(
        recipe, endpoint_url="http://127.0.0.1:8000", concurrency=8
    )
    assert result.output_tok_s == 76.57
    assert result.successful == 32
    assert result.ttft_mean_ms == 128.45