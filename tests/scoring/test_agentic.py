"""Tests for benchmark_suite.scoring.agentic — inspect-ai + terminal-bench."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import cast

import pytest

from benchmark_suite.recipe import AgenticScorer, Recipe
from benchmark_suite.scoring.agentic import (
    AgenticScorerImpl,
    build_inspect_argv,
    build_inspect_env,
    build_terminal_bench_argv,
    check_docker_installed,
    check_inspect_installed,
    parse_inspect_results,
)
from benchmark_suite.scoring.base import ScoreStatus

ENDPOINT_URL = "http://127.0.0.1:8000"
MODEL_NAME = "candidate-model"


def _recipe() -> Recipe:
    return Recipe.model_validate(
        {
            "meta": {"name": "agentic-recipe"},
            "backend": {"type": "external"},
            "endpoint": {"url": ENDPOINT_URL, "model_name": MODEL_NAME},
        }
    )


def _agentic_config(**overrides: object) -> AgenticScorer:
    base: dict[str, object] = {
        "kind": "agentic",
        "harness": "inspect",
        "tasks": ["gaia"],
    }
    base.update(overrides)
    return AgenticScorer.model_validate(base)


# --- build_inspect_argv ---


def test_build_inspect_argv_minimal() -> None:
    recipe = _recipe()
    argv = build_inspect_argv(
        recipe,
        endpoint_url=ENDPOINT_URL,
        tasks=["gaia"],
        model_name=MODEL_NAME,
    )
    assert argv[0] == "inspect"
    assert argv[1] == "eval"
    assert "gaia" in argv
    assert f"openai-api/{MODEL_NAME}" in argv
    assert "--sandbox" in argv
    assert "docker" in argv
    assert "--display" in argv
    assert "plain" in argv


def test_build_inspect_argv_with_limit() -> None:
    recipe = _recipe()
    argv = build_inspect_argv(
        recipe,
        endpoint_url=ENDPOINT_URL,
        tasks=["gaia"],
        model_name=MODEL_NAME,
        limit=5,
    )
    assert "--limit" in argv
    assert argv[argv.index("--limit") + 1] == "5"


def test_build_inspect_argv_with_sandbox() -> None:
    recipe = _recipe()
    argv = build_inspect_argv(
        recipe,
        endpoint_url=ENDPOINT_URL,
        tasks=["gaia"],
        model_name=MODEL_NAME,
        sandbox="local",
    )
    assert "--sandbox" in argv
    assert argv[argv.index("--sandbox") + 1] == "local"


# --- build_inspect_env ---


def test_build_inspect_env_sets_openai_base_url() -> None:
    recipe = _recipe()
    env = build_inspect_env(recipe, ENDPOINT_URL)
    assert env["OPENAI_BASE_URL"] == ENDPOINT_URL


def test_build_inspect_env_merges_recipe_env() -> None:
    recipe = _recipe()
    recipe.runtime.env["VLLM_X"] = "1"
    env = build_inspect_env(recipe, ENDPOINT_URL)
    assert env["VLLM_X"] == "1"
    assert env["OPENAI_BASE_URL"] == ENDPOINT_URL


# --- parse_inspect_results ---


def test_parse_inspect_results_walk(tmp_path: Path) -> None:
    run_dir = tmp_path / "2026-08-30T00-00-00+00-00"
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True)
    (logs_dir / "gaia.json").write_text(
        json.dumps(
            {
                "results": {
                    "scores": [
                        {"name": "gaia", "metrics": {"accuracy": {"value": 0.8}}}
                    ]
                }
            }
        )
    )
    parsed = parse_inspect_results(tmp_path)
    assert parsed == {"gaia": {"accuracy": 0.8}}


def test_parse_inspect_results_missing_dir_returns_empty(tmp_path: Path) -> None:
    assert parse_inspect_results(tmp_path / "does-not-exist") == {}


# --- check_inspect_installed ---


def test_check_inspect_installed_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    inspect = fake_bin / "inspect"
    inspect.write_text("#!/bin/sh\nprintf 'inspect-ai 0.3.260\\n'\n")
    inspect.chmod(0o755)
    monkeypatch.setenv("PATH", str(fake_bin))
    ok, version = check_inspect_installed()
    assert ok is True
    assert "0.3.260" in version


def test_check_inspect_installed_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PATH", str(tmp_path))
    ok, version = check_inspect_installed()
    assert ok is False
    assert version == ""


# --- check_docker_installed ---


def test_check_docker_installed_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    docker = fake_bin / "docker"
    docker.write_text("#!/bin/sh\nprintf 'Docker version 27.0.0\\n'\n")
    docker.chmod(0o755)
    monkeypatch.setenv("PATH", str(fake_bin))
    ok, version = check_docker_installed()
    assert ok is True
    assert "27.0.0" in version


def test_check_docker_installed_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PATH", str(tmp_path))
    ok, version = check_docker_installed()
    assert ok is False
    assert version == ""


# --- build_terminal_bench_argv ---


def test_build_terminal_bench_argv(tmp_path: Path) -> None:
    recipe = _recipe()
    output_path = tmp_path / "tb.json"
    argv = build_terminal_bench_argv(
        recipe,
        tasks=["gaia"],
        output_path=output_path,
    )
    assert argv[0] == "terminal-bench"
    assert "run" in argv
    assert "gaia" in argv
    assert "--output" in argv
    assert str(output_path) in argv


# --- scorer end-to-end ---


def test_agentic_scorer_runs_inspect_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe = _recipe()
    config = _agentic_config()

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        # Write canned inspect output into the output-dir passed via argv.
        argv = cast("list[str]", args[0])
        output_dir = Path(argv[argv.index("--output-dir") + 1])
        logs = output_dir / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        (logs / "gaia.json").write_text(
            json.dumps(
                {
                    "results": {
                        "scores": [
                            {
                                "name": "gaia",
                                "metrics": {"accuracy": {"value": 0.75}},
                            }
                        ]
                    }
                }
            )
        )
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr("benchmark_suite.scoring.agentic.subprocess.run", fake_run)
    monkeypatch.setattr(
        "benchmark_suite.scoring.agentic.check_inspect_installed",
        lambda: (True, "0.3.260"),
    )
    impl = AgenticScorerImpl(config)
    record = impl.score(recipe, result_dir=tmp_path, endpoint_url=ENDPOINT_URL)
    assert record.status == ScoreStatus.SUCCESS
    assert record.metrics["agentic_gaia_accuracy"] == 0.75


def test_agentic_scorer_handles_missing_inspect_gracefully(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe = _recipe()
    config = _agentic_config()
    monkeypatch.setenv("PATH", str(tmp_path))
    impl = AgenticScorerImpl(config)
    record = impl.score(recipe, result_dir=tmp_path, endpoint_url=ENDPOINT_URL)
    assert record.status == ScoreStatus.FAILURE
    assert record.error is not None
    assert "inspect" in record.error


def test_agentic_scorer_terminal_bench_requires_docker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe = _recipe()
    config = _agentic_config(harness="terminal-bench")
    monkeypatch.setenv("PATH", str(tmp_path))
    impl = AgenticScorerImpl(config)
    record = impl.score(recipe, result_dir=tmp_path, endpoint_url=ENDPOINT_URL)
    assert record.status == ScoreStatus.FAILURE
    assert record.error is not None
    assert "docker" in record.error.lower()