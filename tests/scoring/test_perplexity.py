"""Tests for benchmark_suite.scoring.perplexity — lm-eval subprocess scorer."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark_suite.recipe import PerplexityScorer, Recipe
from benchmark_suite.scoring.base import ScoreStatus
from benchmark_suite.scoring.perplexity import (
    PerplexityScorerImpl,
    build_lm_eval_argv,
    check_lm_eval_installed,
    parse_lm_eval_results,
)


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
    # Absolute shebang: PATH is monkeypatched to only tmp_path in these tests,
    # so `#!/usr/bin/env bash` would fail to resolve `bash`.
    fake.write_text("#!/bin/bash\n" + body)
    fake.chmod(0o755)
    return fake


# --- build_lm_eval_argv ---


def test_build_lm_eval_argv_minimal() -> None:
    recipe = _recipe()
    argv = build_lm_eval_argv(
        recipe,
        endpoint_url="http://127.0.0.1:8000",
        output_path=Path("/tmp/out"),
        tasks=["wikitext"],
    )
    assert argv == [
        "lm_eval",
        "--model",
        "local-completions",
        "--model_args",
        "base_url=http://127.0.0.1:8000/v1,model=test-model",
        "--tasks",
        "wikitext",
        "--num_fewshot",
        "0",
        "--output_path",
        "/tmp/out",
    ]


def test_build_lm_eval_argv_appends_v1_to_endpoint() -> None:
    recipe = _recipe()
    argv = build_lm_eval_argv(
        recipe,
        endpoint_url="http://127.0.0.1:8000",
        output_path=Path("/tmp/out"),
        tasks=["wikitext"],
    )
    model_args = argv[argv.index("--model_args") + 1]
    assert model_args.startswith("base_url=http://127.0.0.1:8000/v1,")


def test_build_lm_eval_argv_no_double_v1() -> None:
    recipe = _recipe()
    argv = build_lm_eval_argv(
        recipe,
        endpoint_url="http://127.0.0.1:8000/v1",
        output_path=Path("/tmp/out"),
        tasks=["wikitext"],
    )
    model_args = argv[argv.index("--model_args") + 1]
    assert model_args.startswith("base_url=http://127.0.0.1:8000/v1,")
    assert "/v1/v1" not in model_args


def test_build_lm_eval_argv_with_limit() -> None:
    recipe = _recipe()
    argv = build_lm_eval_argv(
        recipe,
        endpoint_url="http://127.0.0.1:8000",
        output_path=Path("/tmp/out"),
        tasks=["wikitext"],
        limit=100,
    )
    assert "--limit" in argv
    assert argv[argv.index("--limit") + 1] == "100"


def test_build_lm_eval_argv_with_extra_args() -> None:
    recipe = _recipe()
    argv = build_lm_eval_argv(
        recipe,
        endpoint_url="http://127.0.0.1:8000",
        output_path=Path("/tmp/out"),
        tasks=["wikitext"],
        extra_args=["--seed", "42"],
    )
    assert argv[-2:] == ["--seed", "42"]


# --- parse_lm_eval_results ---


def test_parse_lm_eval_results_wikitext(tmp_path: Path) -> None:
    out = tmp_path / "results.json"
    out.write_text(
        json.dumps({"results": {"wikitext": {"ppl": 12.34, "ppl_stderr": 0.05}}})
    )
    parsed = parse_lm_eval_results(tmp_path)
    assert parsed == {"wikitext": {"ppl": 12.34, "ppl_stderr": 0.05}}


def test_parse_lm_eval_results_missing_file_returns_empty(tmp_path: Path) -> None:
    assert parse_lm_eval_results(tmp_path) == {}


# --- check_lm_eval_installed ---


def test_check_lm_eval_installed_present(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_binary(tmp_path, "lm_eval", 'echo "lm-eval 0.4.12"\n')
    monkeypatch.setenv("PATH", str(tmp_path))
    ok, version = check_lm_eval_installed()
    assert ok is True
    assert "0.4.12" in version


def test_check_lm_eval_installed_absent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", str(tmp_path))
    ok, version = check_lm_eval_installed()
    assert ok is False
    assert version == ""


# --- PerplexityScorerImpl.score ---


def test_perplexity_scorer_runs_subprocess_and_parses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Fake lm_eval: handle --version, and on a real run write results.json to
    # the --output_path directory with a known ppl. Only shell builtins are
    # used (echo, redirection) so it works with PATH replaced by tmp_path.
    body = (
        'if [ "$1" = "--version" ]; then echo "lm-eval 0.4.12"; exit 0; fi\n'
        'out=""; prev=""\n'
        'for a in "$@"; do\n'
        '  if [ "$prev" = "--output_path" ]; then out="$a"; fi\n'
        '  prev="$a"\n'
        'done\n'
        'echo \'__JSON__\' > "$out/results.json"\n'
    )
    _fake_binary(
        tmp_path,
        "lm_eval",
        body,
        __JSON__=json.dumps({"results": {"wikitext": {"ppl": 12.34, "ppl_stderr": 0.05}}}),
    )
    monkeypatch.setenv("PATH", str(tmp_path))

    recipe = _recipe()
    config = PerplexityScorer(tasks=["wikitext"])
    result_dir = tmp_path / "results"
    # score() writes into result_dir/lm_eval; pre-create it since the fake
    # script uses only builtins (no mkdir).
    (result_dir / "lm_eval").mkdir(parents=True)

    record = PerplexityScorerImpl(config).score(recipe, result_dir=result_dir)

    assert record.status == ScoreStatus.SUCCESS
    assert record.metrics["perplexity_wikitext"] == 12.34


def test_perplexity_scorer_handles_missing_lm_eval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PATH", str(tmp_path))
    recipe = _recipe()
    config = PerplexityScorer(tasks=["wikitext"])
    result_dir = tmp_path / "results"
    result_dir.mkdir()

    record = PerplexityScorerImpl(config).score(recipe, result_dir=result_dir)

    assert record.status == ScoreStatus.FAILURE
    assert record.error is not None
    assert "lm_eval" in record.error
    assert "install" in record.error